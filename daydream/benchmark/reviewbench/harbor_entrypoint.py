"""Run the ordinary native reviewer against an admitted public Harbor snapshot.

Only this public lane admits origin-free frozen tasks. Supplied verifier assets
remain in Harbor's separate verifier environment; no private scorer is imported.
"""
from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import anyio

from daydream.benchmark.reviewbench.contract import Case
from daydream.benchmark.reviewbench.native import CaseExecution, PublicHarborTask, review_case, reviewer_config
from daydream.json_utils import atomic_create_bytes
from daydream.redaction import redact_structured_text
from daydream.trajectory import RUN_DOCUMENT_NAME, flush_active_signal_recorders, run_directory


class PublicReviewerError(ValueError):
    """Unscorable or unsafe public reviewer invocation."""


def load_public_task(profile_path: Path, pr_path: Path, patch_path: Path) -> tuple[PublicHarborTask, dict[str, Any]]:
    """Bind exported profile bytes, supplied PR metadata and frozen public patch."""
    profile = json.loads(profile_path.read_text())
    if not isinstance(profile, dict) or profile.get('schema') != 3:
        raise PublicReviewerError('unsupported public reviewer profile; regenerate the export')
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
    if (not isinstance(reviewer, dict) or reviewer.get('backend') not in {'pi', 'codex'}
            or not isinstance(reviewer.get('model'), str) or not reviewer['model'].strip()
            or type(reviewer.get('fanout_concurrency')) is not int or reviewer['fanout_concurrency'] != 1):
        raise PublicReviewerError('public reviewer backend and model must be explicit')
    reviewer_config(reviewer, pr_path.parent)
    title, body = pr.get('title', ''), pr.get('body', '')
    if not isinstance(title, str) or not isinstance(body, str):
        raise PublicReviewerError('public reviewer author context differs from frozen profile')
    description = title + '\n\n' + body if title or body else None
    return PublicHarborTask(case, profile['upstream_revision'], patch_path.read_text(),
                            profile['task_id'], profile['evaluation_id'], digest, description), profile


def _reviewer_environment(
    source: Mapping[str, str], role_home: Path, reviewer: Mapping[str, Any], canonical: Path,
) -> dict[str, str]:
    """Give only the selected native transport its independent credential capability."""
    result = {key: value for key, value in source.items()
              if key in {'PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_ALL', 'SSL_CERT_FILE', 'SSL_CERT_DIR'}}
    if reviewer['backend'] == 'codex':
        result['CODEX_HOME'] = str(role_home)
        result['DAYDREAM_FANOUT_CONCURRENCY'] = str(reviewer['fanout_concurrency'])
    else:
        from daydream.benchmark.reviewbench.harbor_auth import REVIEWER_PI_BOOTSTRAP

        result['PI_CODING_AGENT_DIR'] = str(role_home)
        result['PI_PROVIDER'] = reviewer['provider']
        if reviewer.get('thinking') is not None:
            result['PI_THINKING'] = reviewer['thinking']
        result['DAYDREAM_PI_FANOUT_CONCURRENCY'] = str(reviewer['fanout_concurrency'])
        result['DAYDREAM_RB_PI_AUTH_PATH'] = str((canonical / 'auth.json').resolve())
        result['NODE_OPTIONS'] = '--import=' + json.dumps(REVIEWER_PI_BOOTSTRAP)
    return result


def _write_new_json(destination: Path, value: Any) -> None:
    body = redact_structured_text(json.dumps(value, indent=2, allow_nan=False)) + '\n'
    atomic_create_bytes(destination, body.encode('utf-8'))


@contextmanager
def _private_archive_directory(directory: Path) -> Iterator[None]:
    """Confine dump assembly's archive side effects to this single-run process.

    The ordinary dump pipeline also maintains an archive index. Its supported
    environment selector must point at disposable private storage, never the
    operator's archive. This selector is excluded from backend environments.
    """
    previous = os.environ.get('DAYDREAM_ARCHIVE_DIR')
    os.environ['DAYDREAM_ARCHIVE_DIR'] = str(directory)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('DAYDREAM_ARCHIVE_DIR', None)
        else:
            os.environ['DAYDREAM_ARCHIVE_DIR'] = previous


def _publish_native_evidence(staged: Path, evidence: Path, harbor_trajectory: Path) -> None:
    """Copy only joined native evidence onto Harbor's non-exchange log mount.

    The native artifact owner retains its private filesystem and live-exchange
    guarantees. Harbor receives exact finalized exports after every writer has
    joined, through a no-clobber install local to its own destination filesystem.
    """
    evidence.mkdir(parents=True, exist_ok=False)
    trajectory = staged / RUN_DOCUMENT_NAME
    bundle = evidence
    if trajectory.is_file():
        document = json.loads(trajectory.read_bytes())
        session_id = document.get('session_id') if isinstance(document, dict) else None
        if not isinstance(session_id, str) or str(uuid.UUID(session_id)) != session_id:
            raise PublicReviewerError('native evidence has an invalid session identity')
        # The sanctioned dump projects the immutable root and every sibling into
        # one run bundle. Use its owner layout so original logical sibling refs
        # resolve from the retained evidence root without rewriting ATIF bytes.
        bundle = run_directory(evidence, session_id)
    for source in sorted(staged.rglob('*')):
        if source.is_symlink():
            raise PublicReviewerError('native evidence contains an unexpected symlink')
        if source.is_file():
            atomic_create_bytes(bundle / source.relative_to(staged), source.read_bytes())
    if trajectory.is_file():
        atomic_create_bytes(evidence / RUN_DOCUMENT_NAME, trajectory.read_bytes())
        atomic_create_bytes(harbor_trajectory, trajectory.read_bytes())


async def main(environment: Mapping[str, str]) -> int:
    """Review once, retaining terminal evidence and refusing stale/partial output."""
    repository = Path(environment.get('DAYDREAM_RB_REPOSITORY', '/work/repo'))
    inputs = Path(environment.get('DAYDREAM_RB_INPUTS', '/work/pr'))
    output = Path(environment.get('DAYDREAM_RB_OUTPUT', '/work/out/findings.json'))
    evidence = Path(environment.get('DAYDREAM_RB_EVIDENCE', '/logs/agent/daydream'))
    harbor_trajectory = evidence.parent / RUN_DOCUMENT_NAME
    started = time.monotonic()
    try:
        if (output.exists() or evidence.exists()
                or harbor_trajectory.exists() or harbor_trajectory.is_symlink()):
            raise PublicReviewerError('prior reviewer artifacts exist; use a fresh trial')
        task, profile = load_public_task(inputs / 'daydream-profile.json', inputs / 'pr.json', inputs / 'diff.patch')
        reviewer = profile['reviewer']
        backend = reviewer['backend']
        from daydream.benchmark.reviewbench.diagnostics import InterruptedReviewDiagnostics
        from daydream.benchmark.reviewbench.harbor_process import warn_diagnostics_unavailable

        declared_outer = environment.get('DAYDREAM_RB_DECLARED_OUTER_TIMEOUT')
        metadata = {
            'candidate': profile['candidate'], 'profile_sha256': task.profile_sha256,
            'task_id': task.task_id, 'evaluation_id': task.evaluation_id,
            'upstream_revision': task.upstream_revision, 'case': asdict(task.case),
            'patch_sha256': hashlib.sha256(task.patch.encode()).hexdigest(),
            'reviewer': reviewer,
            'budget_hierarchy': {'harbor_outer_deadline_seconds': float(declared_outer) if declared_outer else None,
                'outer_deadline_availability': 'declared trial config'
                    if 'DAYDREAM_RB_DECLARED_OUTER_TIMEOUT' in environment else 'unavailable',
                'docker_exec_timeout_seconds': 3600,
                'native_allowance_availability': 'observed per dispatch in backend_lifecycle'},
            'invocation_id': environment.get('DAYDREAM_RB_INVOCATION'),
        }
        diagnostics = None
        checkpoint_error = None
        try:
            diagnostics = InterruptedReviewDiagnostics(evidence, metadata)
        except OSError as exc:
            if exc.errno not in {errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EMFILE, errno.ENFILE}:
                raise
            # Checkpoint capacity is optional. Path admission, competing bytes,
            # and the canonical native/finding publication remain authoritative.
            checkpoint_error = f'checkpoint initialization unavailable: {type(exc).__name__} errno={exc.errno}'
            warn_diagnostics_unavailable('checkpoint initialization', exc)
        canonical = Path(environment.get('DAYDREAM_RB_CODEX_HOME' if backend == 'codex'
            else 'DAYDREAM_RB_REVIEWER_HOME', '/run/reviewbench-codex' if backend == 'codex'
            else '/run/reviewbench-reviewer'))
        from daydream.benchmark.reviewbench.harbor_auth import reviewer_credentials

        execution = None
        cancelled = False
        with tempfile.TemporaryDirectory(prefix='daydream-reviewbench-runtime-') as private_directory:
            private = Path(private_directory).resolve()
            staged = private / 'native-evidence'
            try:
                with (_private_archive_directory(private / 'archive'),
                      reviewer_credentials(canonical, private / backend, backend=backend,
                                           provider=reviewer['provider']) as role_home):
                    if backend == 'codex':
                        (role_home / 'config.toml').write_text('model_provider = "openai"\n'
                            'cli_auth_credentials_store = "file"\n')
                    elif reviewer.get('pi_models') is not None:
                        (role_home / 'models.json').write_text(json.dumps(reviewer['pi_models']))
                    controls = _reviewer_environment(environment, role_home, reviewer, canonical)
                    config = reviewer_config(reviewer, repository)
                    from daydream.backends import BackendExecutionInput
                    from daydream.git_ops import StaticGitHubAuth
                    from daydream.github_app import GitHubExecutionInput
                    from daydream.runner import RunnerExecutionInput

                    native = BackendExecutionInput.from_environment(controls, backend=backend)
                    if backend == 'pi':
                        native = replace(native, pi_settings=reviewer['pi_settings'])
                    capability = RunnerExecutionInput(native, GitHubExecutionInput(auth=StaticGitHubAuth(controls)),
                        backends={backend: native},
                        on_trajectory_snapshot=diagnostics.retain if diagnostics is not None else None)
                    invocation = environment.get('DAYDREAM_RB_INVOCATION')
                    if invocation is not None:
                        from daydream.benchmark.reviewbench.harbor_process import invocation_directory

                        if (invocation_directory(invocation) / 'cancel').exists():
                            raise asyncio.CancelledError
                    checkpoint_task = asyncio.create_task(_checkpoint_loop())
                    try:
                        execution = await review_case(task.case, repository, config, 'daydream-native-' + backend,
                            execution_input=capability, candidate_sha256=profile['candidate']['artifact_sha256'],
                            public_task=task, evidence_directory=staged)
                        await anyio.lowlevel.checkpoint_if_cancelled()
                    finally:
                        checkpoint_task.cancel()
                        with anyio.CancelScope(shield=True):
                            await asyncio.gather(checkpoint_task, return_exceptions=True)
            except asyncio.CancelledError:
                cancelled = True
                if execution is not None:
                    execution.status = 'cancelled'
                    execution.candidate = None
                    execution.diagnostics.append('native reviewer cancelled after returning an execution result')
                raise
            except Exception:
                if execution is not None:
                    execution.status = 'failed'
                    execution.candidate = None
                    execution.diagnostics.append('reviewer credential invocation did not finalize')
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
                        reviewer_backend=backend, reviewer_provider=reviewer['provider'],
                        reviewer_authentication='chatgpt-oauth' if backend == 'codex' else 'native-pi',
                        cost_interpretation='API-equivalent estimate when available',
                        invocation_id=environment.get('DAYDREAM_RB_INVOCATION'),
                        checkpoint_error=diagnostics.last_error if diagnostics is not None else checkpoint_error,
                        interruption='reviewer_task_cancelled' if cancelled else None)
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


async def _checkpoint_loop() -> None:
    """Project coherent immutable snapshots; live native storage never leaves its owner."""
    while True:
        await asyncio.sleep(2)
        flush_active_signal_recorders()


async def _executable_main() -> int:
    """Translate SIGTERM once into cancellation with ordinary joined finalization."""
    loop = asyncio.get_running_loop()
    requested = False
    result = 130
    try:
        with anyio.CancelScope() as shutdown:
            def interrupt() -> None:
                nonlocal requested
                if not requested:
                    requested = True
                    # A raw Task.cancel() bypasses AnyIO's writer shields.
                    shutdown.cancel()
                    flush_active_signal_recorders()

            loop.add_signal_handler(signal.SIGTERM, interrupt)
            try:
                result = await main(dict(os.environ))
            finally:
                loop.remove_signal_handler(signal.SIGTERM)
    except asyncio.CancelledError:
        return 130
    return 130 if requested else result


if __name__ == '__main__':
    raise SystemExit(asyncio.run(_executable_main()))
