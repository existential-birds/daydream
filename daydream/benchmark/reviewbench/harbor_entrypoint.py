"""Run the ordinary native reviewer against an admitted public Harbor snapshot.

Only this public lane admits origin-free frozen tasks. Supplied verifier assets
remain in Harbor's separate verifier environment; no private scorer is imported.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from daydream.benchmark.reviewbench.contract import Case
from daydream.benchmark.reviewbench.native import CaseExecution, PublicHarborTask, review_case
from daydream.config_file import DaydreamFileConfig
from daydream.redaction import redact_structured_text
from daydream.review_profile import resolve_harbor_profile
from daydream.run_config import RunConfig
from daydream.trajectory import RUN_DOCUMENT_NAME


class PublicReviewerError(ValueError):
    """Unscorable or unsafe public reviewer invocation."""


def load_public_task(profile_path: Path, pr_path: Path, patch_path: Path) -> tuple[PublicHarborTask, dict[str, Any]]:
    """Bind exported profile bytes, supplied PR metadata and frozen public patch."""
    profile = json.loads(profile_path.read_text())
    if not isinstance(profile, dict) or profile.get('schema') != 1:
        raise PublicReviewerError('invalid public reviewer profile schema')
    digest = profile.get('profile_sha256')
    policy = {key: value for key, value in profile.items() if key != 'profile_sha256'}
    calculated = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
                                .encode()).hexdigest()
    if digest != calculated:
        raise PublicReviewerError('public reviewer profile digest differs')
    case = Case.from_dict(profile['pr'])
    pr = json.loads(pr_path.read_text())
    if Case.from_dict(pr) != case or pr != profile['pr']:
        raise PublicReviewerError('public reviewer PR metadata differs')
    reviewer = profile.get('reviewer')
    if (not isinstance(reviewer, dict) or not isinstance(reviewer.get('model'), str) or not reviewer['model'].strip()
            or reviewer.get('reasoning_effort') not in {'none', 'minimal', 'low', 'medium', 'high', 'xhigh'}):
        raise PublicReviewerError('public reviewer model and effort must be explicit')
    title, body = pr.get('title', ''), pr.get('body', '')
    if (not isinstance(title, str) or not isinstance(body, str)
            or profile['pr'].get('title', '') != title or profile['pr'].get('body', '') != body):
        raise PublicReviewerError('public reviewer author context differs from frozen profile')
    description = title + '\n\n' + body if title or body else None
    return PublicHarborTask(case, profile['upstream_revision'], patch_path.read_text(),
                            profile['task_id'], profile['evaluation_id'], digest, description), reviewer


def _reviewer_environment(source: Mapping[str, str], codex_home: Path) -> dict[str, str]:
    """Give the native Codex transport only role-local subscription credentials."""
    result = {key: value for key, value in source.items()
              if key in {'PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'SSL_CERT_FILE', 'SSL_CERT_DIR'}}
    result['CODEX_HOME'] = str(codex_home)
    return result


def _write_new_bytes(destination: Path, body: bytes) -> None:
    """Install finalized bytes without replacing a prior attempt's artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.daydream-', dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(destination: Path, value: Any) -> None:
    body = redact_structured_text(json.dumps(value, indent=2, allow_nan=False)) + '\n'
    _write_new_bytes(destination, body.encode('utf-8'))


def _publish_native_evidence(staged: Path, evidence: Path, harbor_trajectory: Path) -> None:
    """Copy only joined native evidence onto Harbor's non-exchange log mount.

    The native artifact owner retains its private filesystem and live-exchange
    guarantees. Harbor receives exact finalized exports after every writer has
    joined, through a no-clobber install local to its own destination filesystem.
    """
    evidence.mkdir(parents=True, exist_ok=False)
    for source in sorted(staged.rglob('*')):
        if source.is_symlink():
            raise PublicReviewerError('native evidence contains an unexpected symlink')
        if source.is_file():
            _write_new_bytes(evidence / source.relative_to(staged), source.read_bytes())
    trajectory = staged / RUN_DOCUMENT_NAME
    if trajectory.is_file():
        _write_new_bytes(harbor_trajectory, trajectory.read_bytes())


async def main(environment: Mapping[str, str]) -> int:
    """Review once, retaining terminal evidence and refusing stale/partial output."""
    repository = Path(environment.get('DAYDREAM_RB_REPOSITORY', '/work/repo'))
    inputs = Path(environment.get('DAYDREAM_RB_INPUTS', '/work/pr'))
    output = Path(environment.get('DAYDREAM_RB_OUTPUT', '/work/out/findings.json'))
    evidence = Path(environment.get('DAYDREAM_RB_EVIDENCE', '/logs/agent/daydream'))
    canonical = Path(environment.get('DAYDREAM_RB_CODEX_HOME', '/run/reviewbench-codex'))
    harbor_trajectory = evidence.parent / RUN_DOCUMENT_NAME
    started = time.monotonic()
    try:
        if (output.exists() or evidence.exists()
                or harbor_trajectory.exists() or harbor_trajectory.is_symlink()):
            raise PublicReviewerError('prior reviewer artifacts exist; use a fresh trial')
        task, reviewer = load_public_task(inputs / 'daydream-profile.json', inputs / 'pr.json', inputs / 'diff.patch')
        # Import lazily: the ordinary standalone native adapter needs no lease.
        from daydream.benchmark.reviewbench.harbor_auth import subscription_lease

        execution = None
        cancelled = False
        with tempfile.TemporaryDirectory(prefix='daydream-reviewbench-runtime-') as private_directory:
            private = Path(private_directory).resolve()
            staged = private / 'native-evidence'
            try:
                with subscription_lease(canonical, private / 'codex', role='reviewer') as role_home:
                    # A fresh private home prevents user config, API providers and
                    # project settings from changing this frozen subscription lane.
                    (role_home / 'config.toml').write_text('model_provider = "openai"\n'
                        'cli_auth_credentials_store = "file"\n')
                    controls = _reviewer_environment(environment, role_home)
                    config = RunConfig(target=str(repository), backend='codex', model=reviewer['model'],
                        reasoning_effort=reviewer['reasoning_effort'], file_config=DaydreamFileConfig(),
                        review_profile=resolve_harbor_profile(candidate_env='DAYDREAM_RB_UNUSED', env={}),
                        output_mode='review', non_interactive=True, dataset_capture_disabled=True,
                        archive=False, run_eval=False)
                    # Resolve native controls from the dedicated role environment;
                    # there is no mutation of global authentication or provider state.
                    from daydream.backends import BackendExecutionInput
                    from daydream.git_ops import StaticGitHubAuth
                    from daydream.github_app import GitHubExecutionInput
                    from daydream.runner import RunnerExecutionInput

                    native = BackendExecutionInput.from_environment(controls, backend='codex')
                    runtime = ({'codex': {'subscription_access': 'chatgpt-oauth', 'provider': 'openai',
                                         'configuration': 'fresh-role-home'}},
                               RunnerExecutionInput(native, GitHubExecutionInput(auth=StaticGitHubAuth(controls)),
                                                    backends={'codex': native}))
                    execution = await review_case(task.case, repository, config, 'daydream-native-codex',
                        runtime=runtime, public_task=task, evidence_directory=staged)
            except asyncio.CancelledError:
                cancelled = True
                raise
            except Exception:
                if execution is not None:
                    execution.status = 'failed'
                    execution.candidate = None
                    execution.diagnostics.append('subscription credential lease did not finalize')
                raise
            finally:
                if execution is not None or staged.exists():
                    if execution is None:
                        execution = CaseExecution('cancelled' if cancelled else 'failed', diagnostics=[
                            'native reviewer cancelled before returning an execution result' if cancelled else
                            'native reviewer did not return an execution result'])
                    if not (staged / RUN_DOCUMENT_NAME).is_file():
                        execution.diagnostics.append('native trajectory unavailable; no ATIF export was manufactured')
                        if execution.status in {'complete', 'completed_empty'}:
                            execution.status = 'failed'
                            execution.candidate = None
                    _publish_native_evidence(staged, evidence, harbor_trajectory)
                    record = asdict(execution)
                    record.update(task_id=task.task_id, evaluation_id=task.evaluation_id,
                        profile_sha256=task.profile_sha256, elapsed_seconds=time.monotonic() - started,
                        subscription_access='chatgpt-oauth',
                        cost_interpretation='API-equivalent estimate when available')
                    _write_new_json(evidence / 'execution.json', record)
        if execution is None:
            raise PublicReviewerError('native review did not return an execution result')
        if execution.status not in {'complete', 'completed_empty'} or execution.candidate is None:
            raise PublicReviewerError('native review is unscorable: ' + execution.status)
        _write_new_json(output, execution.candidate)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'public ReviewBench reviewer failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main(dict(os.environ))))
