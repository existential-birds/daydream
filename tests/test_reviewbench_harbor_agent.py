"""Public reviewer boundary: real frozen Git, runner and role-private auth files."""
from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import json
import os
import shlex
import sys
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.artifacts import external
from daydream.backends import AgentEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.codex import CodexBackend
from daydream.benchmark.reviewbench.candidate import candidate_package_digest
from daydream.benchmark.reviewbench.contract import REVIEWBENCH_REVISION
from daydream.benchmark.reviewbench.harbor_agent import DaydreamPublicReviewAgent
from daydream.benchmark.reviewbench.harbor_entrypoint import main
from daydream.benchmark.reviewbench.native import freeze_reviewer
from daydream.trajectory import RUNS_DIRNAME, run_directory, run_document_path
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git
from tests.test_reviewbench_native import finding, frozen_case


def public_input(tmp_path: Path, *, additional_sources: int = 0) -> dict[str, str]:
    repo, case = frozen_case(tmp_path, additional_sources=additional_sources)
    git(repo, 'remote', 'remove', 'origin')
    git_ops.update_refs(repo, {'refs/reviewbench/base': case.base, 'refs/reviewbench/head': case.head})
    inputs = tmp_path / 'pr'
    inputs.mkdir()
    pr = {**asdict(case), 'title': 'Frozen supplied title', 'body': 'Frozen supplied author explanation'}
    (inputs / 'pr.json').write_text(json.dumps(pr))
    (inputs / 'diff.patch').write_text(git_ops.diff_paths(repo, case.base, case.head, ['.']))
    profile = {'schema': 3, 'upstream_revision': REVIEWBENCH_REVISION, 'pr': pr,
               'task_id': 7, 'evaluation_id': 99,
               'reviewer': freeze_reviewer(empty_review_config(repo, tmp_path / 'ignored', backend='codex',
                   model='gpt-6.1-sol', reasoning_effort='high')),
               'candidate': {'artifact_sha256': candidate_package_digest()}}
    profile['profile_sha256'] = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':'))
                                               .encode()).hexdigest()
    (inputs / 'daydream-profile.json').write_text(json.dumps(profile))
    canonical = tmp_path / 'subscription'
    canonical.mkdir()
    payload = base64.urlsafe_b64encode(json.dumps({'exp': 9000000000}).encode()).decode().rstrip('=')
    (canonical / 'auth.json').write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {
        'access_token': f'header.{payload}.signature', 'refresh_token': 'private-refresh-canary',
        'id_token': 'private-id-canary', 'account_id': 'test-account'}}))
    return {'DAYDREAM_RB_REPOSITORY': str(repo), 'DAYDREAM_RB_INPUTS': str(inputs),
            'DAYDREAM_RB_OUTPUT': str(tmp_path / 'out/findings.json'),
            'DAYDREAM_RB_EVIDENCE': str(tmp_path / 'logs/daydream'),
            'DAYDREAM_RB_CODEX_HOME': str(canonical), 'PATH': os.environ['PATH'], 'HOME': str(tmp_path),
            'OPENAI_API_KEY': 'forbidden-api-canary', 'GH_TOKEN': 'forbidden-github-canary'}


@pytest.mark.parametrize('stage', ['setup', 'run'])
async def test_public_agent_python_startup_cannot_import_reviewed_repository_packages(
    tmp_path: Path, stage: str,
) -> None:
    """Execute the real agent commands from hostile source using real Python.

    Direct entrypoint tests import the operator's package before selecting a
    repository, so they miss Python's cwd/PYTHONPATH precedence at process startup.
    Both a candidate-named package and a default extension in the reviewed source
    must stay unexecuted, including the initial setup import.
    """
    environment = public_input(tmp_path)
    repository = Path(environment['DAYDREAM_RB_REPOSITORY'])
    marker = tmp_path / 'reviewed-python-executed'
    for name in ('daydream', 'daydream_ext'):
        package = repository / name
        package.mkdir()
        (package / '__init__.py').write_text('from pathlib import Path\nPath(' + repr(str(marker))
            + ').write_text("reviewed package executed")\nraise RuntimeError("hostile source imported")\n')
    # Setup only checks CLI availability; this stand-in cannot run inference.
    binary = tmp_path / 'bin' / 'codex'
    binary.parent.mkdir()
    binary.write_text(f'#!{sys.executable}\nraise SystemExit("unexpected CLI execution")\n')
    binary.chmod(0o755)
    (Path(environment['DAYDREAM_RB_CODEX_HOME']) / 'auth.json').unlink()
    child_environment = {**os.environ, **environment, 'PYTHONPATH': str(repository),
                         'PATH': os.pathsep.join((str(binary.parent), os.environ['PATH']))}

    class Environment:
        async def exec(self, command: str, **kwargs: Any) -> Any:
            arguments = shlex.split(command)
            assert arguments[0] == 'python'
            # Harbor's image resolves `python` to its installed environment.
            # Execute those actual arguments with this test environment's Python.
            arguments[0] = sys.executable
            child = await asyncio.create_subprocess_exec(*arguments, cwd=repository, env=child_environment,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=30)
            return SimpleNamespace(return_code=child.returncode, stdout=stdout.decode(), stderr=stderr.decode())

    logs = tmp_path / 'agent'
    agent = DaydreamPublicReviewAgent(logs_dir=logs)
    if stage == 'setup':
        await agent.setup(Environment())
    else:
        with pytest.raises(RuntimeError, match='native reviewer failed'):
            await agent.run('supplied task', Environment(), SimpleNamespace())
        assert 'SubscriptionAuthError' in (logs / 'reviewer-entrypoint.log').read_text()
    assert not marker.exists()
    assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()


@pytest.mark.parametrize('outcome', ['complete', 'incomplete', 'conversion_limited', 'provider_failed'])
async def test_public_harbor_native_boundary_keeps_terminal_evidence_and_refuses_unscorable_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, outcome: str,
    dataset_store: Path, archive_dir: Path,
) -> None:
    """A successful subprocess alone must not publish partial or discarded findings.

    The existing native tests do not exercise public origin-free snapshots,
    reviewer credential locking, private trajectory retention or Harbor candidate output.
    Only the external generation transport is replaced; runner and Git stay real.
    """
    environment = public_input(tmp_path)
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    exchange = external._AtomicNameExchange.call

    def logs_do_not_support_exchange(
        binding: external._AtomicNameExchange, parent_fd: int, staged_name: str, target_name: str,
    ) -> external._NameExchangeResult:
        # Fault at the actual name-exchange boundary, like Docker Desktop's
        # mounted logs. Native private storage must still use real exchanges.
        if evidence.exists() and os.fstat(parent_fd).st_ino == evidence.stat().st_ino:
            return external._NameExchangeResult(-1, errno.ENOTSUP)
        return exchange(binding, parent_fd, staged_name, target_name)

    monkeypatch.setattr(external._AtomicNameExchange, 'call', logs_do_not_support_exchange)
    role_homes: list[Path] = []
    backends: list[EmptyReviewBackend] = []
    active = peak_active = 0
    archive_roots: set[Path] = set()
    protected_archive = tmp_path / 'operator-archive'
    protected_archive.mkdir()
    (protected_archive / 'keep').write_text('unrelated operator history')
    monkeypatch.setenv('DAYDREAM_ARCHIVE_DIR', str(protected_archive))

    class SerializedBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            try:
                await asyncio.sleep(0.01)
                stack = ('structure' if 'you are the structural reviewer' in prompt.lower() else
                         'python' if 'you are reviewing the python stack' in prompt.lower() else None)
                if stack is not None:
                    yield TextEvent('Retained raw reviewer turn: ' + stack)
                    yield ToolStartEvent(id='read-' + stack, name='Read', input={'file_path': 'api.py'})
                    yield ToolResultEvent(id='read-' + stack, output='Retained raw read result: ' + stack,
                                          is_error=False)
                async for event in super().execute(cwd, prompt, *args, **kwargs):
                    yield event
            finally:
                active -= 1

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        assert name == 'codex'
        assert kwargs['execution_input'].fanout_concurrency == 1
        child = kwargs['execution_input'].child_environment()
        assert not {'OPENAI_API_KEY', 'GH_TOKEN', 'OPENAI_BASE_URL', 'DAYDREAM_ARCHIVE_DIR'} & child.keys()
        archive_roots.add(Path(os.environ['DAYDREAM_ARCHIVE_DIR']))
        home = Path(child['CODEX_HOME'])
        role_homes.append(home)
        assert home != Path(environment['DAYDREAM_RB_CODEX_HOME'])
        auth = json.loads((home / 'auth.json').read_text())
        assert auth['auth_mode'] == 'chatgpt'
        auth['tokens']['refresh_token'] = 'private-rotated-canary'
        (home / 'auth.json').write_text(json.dumps(auth))
        # StubBackend's target is only a fixture root for non-review replies;
        # actual native review passes its independent checkout to execute().
        backend = SerializedBackend(Path(environment['DAYDREAM_RB_REPOSITORY']),
            forbid_merge=False, forbid_supervise=False,
            review_by_stack={'structure': [finding(line=500 if outcome == 'conversion_limited' else 2)]},
            incomplete_stack='python' if outcome == 'incomplete' else None,
            responder=(lambda prompt: [RuntimeError('subscription quota exhausted')] if outcome == 'provider_failed'
                and ('you are reviewing the ' in prompt.lower() or 'you are the structural reviewer' in prompt.lower())
                else None))
        # Preserve the real native constructor's public scheduling hint while
        # replacing only its external generation stream with this fixture.
        backend.fanout_concurrency = CodexBackend(
            model=kwargs['model'], reasoning_effort=kwargs['reasoning_effort'],
            execution_input=kwargs['execution_input']).fanout_concurrency
        backend.merge_echo_records = True
        backends.append(backend)
        return backend

    monkeypatch.setattr(runner, 'create_backend', create)
    result = await main(environment)
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    record = json.loads((evidence / 'execution.json').read_text())
    assert record['status'] == ('failed' if outcome == 'provider_failed' else outcome), record['diagnostics']
    assert (evidence / 'trajectory.json').is_file()
    assert (evidence.parent / 'trajectory.json').read_bytes() == (evidence / 'trajectory.json').read_bytes()
    assert record['terminal_result']['analyzed_revision']['head_sha'] == record['analyzed_comparison']['head_sha']
    assert record['task_id'] == 7 and record['evaluation_id'] == 99
    assert record['reviewer_authentication'] == 'chatgpt-oauth'
    assert active == 0 and peak_active == 1
    assert record['reviewer_backend'] == 'codex' and record['reviewer_provider'] == 'openai'
    assert record['configuration']['phases']['per_stack_review']['backend'] == 'codex'
    assert record['configuration']['phases']['per_stack_review']['model'] == 'gpt-6.1-sol'
    assert role_homes and all(not home.exists() for home in role_homes)
    assert json.loads((Path(environment['DAYDREAM_RB_CODEX_HOME']) / 'auth.json').read_text())[
        'tokens']['refresh_token'] == 'private-rotated-canary'
    assert any('Frozen supplied author explanation' in call['prompt'] for backend in backends for call in backend.calls)
    assert 'private-refresh-canary' not in (evidence / 'execution.json').read_text()
    assert 'forbidden-api-canary' not in (evidence / 'trajectory.json').read_text()
    trajectory = json.loads((evidence / 'trajectory.json').read_text())
    bundle = run_directory(evidence, trajectory['session_id'])
    assert run_document_path(bundle).read_bytes() == (evidence / 'trajectory.json').read_bytes()
    assert (bundle / 'manifest.json').is_file() and (bundle / 'diff.patch').is_file()
    siblings = [entry for entry in trajectory['extra']['subtrajectories'] if 'sibling_trajectory_ref' in entry]
    assert {entry['descriptor'] for entry in siblings} == {'deep-python', 'deep-structure'}
    for entry in siblings:
        sibling_path = evidence / entry['sibling_trajectory_ref']
        sibling = json.loads(sibling_path.read_text())
        assert sibling['session_id'] == trajectory['session_id']
        assert sibling['trajectory_id'] == entry['trajectory_id']
        stack = entry['descriptor'].removeprefix('deep-')
        assert 'Retained raw reviewer turn: ' + stack in sibling_path.read_text()
        assert 'Retained raw read result: ' + stack in sibling_path.read_text()
    assert archive_roots and all(root != protected_archive and not root.exists() for root in archive_roots)
    assert os.environ['DAYDREAM_ARCHIVE_DIR'] == str(protected_archive)
    assert [p.name for p in protected_archive.iterdir()] == ['keep']
    assert not list(archive_dir.iterdir()) and not dataset_store.exists()
    for path in evidence.rglob('*'):
        if path.is_file():
            assert b'private-refresh-canary' not in path.read_bytes()
            assert b'private-rotated-canary' not in path.read_bytes()
            assert b'forbidden-api-canary' not in path.read_bytes()
            assert path.name != 'auth.json'
    output = Path(environment['DAYDREAM_RB_OUTPUT'])
    assert output.exists() == (outcome == 'complete')
    assert result == (0 if outcome == 'complete' else 1)
    if output.exists():
        assert json.loads(output.read_text())['findings'][0]['file'] == 'api.py'
    assert not fake_gh.calls('POST')


@pytest.mark.parametrize('fault', ['schema', 'profile', 'patch', 'refs', 'origin', 'stale', 'stale_trajectory', 'auth'])
async def test_public_harbor_refuses_unbound_source_or_stale_artifacts_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, fault: str,
) -> None:
    """Public provenance must not weaken ordinary origin checks or replay artifacts."""
    environment = public_input(tmp_path)
    prior_archive = os.environ.get('DAYDREAM_ARCHIVE_DIR')
    inputs = Path(environment['DAYDREAM_RB_INPUTS'])
    if fault in {'schema', 'profile'}:
        profile_path = inputs / 'daydream-profile.json'
        profile = json.loads(profile_path.read_text())
        if fault == 'schema':
            profile['schema'] = 1
            policy = {key: value for key, value in profile.items() if key != 'profile_sha256'}
            profile['profile_sha256'] = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':'))
                                                      .encode()).hexdigest()
        else:
            profile['reviewer']['model'] = 'unfrozen-model'
        profile_path.write_text(json.dumps(profile))
    elif fault == 'patch':
        (inputs / 'diff.patch').write_text('unrelated supplied patch')
    elif fault == 'refs':
        git(Path(environment['DAYDREAM_RB_REPOSITORY']), 'update-ref', '-d', 'refs/reviewbench/base')
    elif fault == 'origin':
        git(Path(environment['DAYDREAM_RB_REPOSITORY']), 'remote', 'add', 'origin', 'https://github.com/other/repo')
    elif fault == 'stale_trajectory':
        trajectory = Path(environment['DAYDREAM_RB_EVIDENCE']).parent / 'trajectory.json'
        trajectory.parent.mkdir(parents=True)
        trajectory.write_text('previous-native-trajectory')
    elif fault == 'stale':
        output = Path(environment['DAYDREAM_RB_OUTPUT'])
        output.parent.mkdir()
        output.write_text('previous-evidence')
    else:
        (Path(environment['DAYDREAM_RB_CODEX_HOME']) / 'auth.json').unlink()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('invalid public task reached generation transport')

    monkeypatch.setattr(runner, 'create_backend', forbidden)
    assert await main(environment) == 1
    assert os.environ.get('DAYDREAM_ARCHIVE_DIR') == prior_archive
    output = Path(environment['DAYDREAM_RB_OUTPUT'])
    if fault == 'stale':
        assert output.read_text() == 'previous-evidence'
    else:
        assert not output.exists()
    if fault == 'stale_trajectory':
        assert trajectory.read_text() == 'previous-native-trajectory'


async def test_native_storage_failure_retains_diagnostics_without_manufacturing_atif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
    artifact_runtime_root: Path,
) -> None:
    """Storage preflight can fail before any trajectory or model exists.

    Existing provider-failure rows still have a native recorder. Faulting the
    actual private session-directory creation covers owner admission rejection
    without relying on an optional external trajectory's exchange preflight.
    It prevents a missing-file exception from replacing the primary evidence.
    """
    environment = public_input(tmp_path)
    mkdir = Path.mkdir
    rejected: list[Path] = []

    def read_only_private_storage(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.parent.name == RUNS_DIRNAME and path.is_relative_to(artifact_runtime_root):
            rejected.append(path)
            raise OSError(errno.EROFS, 'private artifact storage is read-only')
        mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'mkdir', read_only_private_storage)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('artifact owner preflight failure reached model generation')

    monkeypatch.setattr(runner, 'create_backend', forbidden)
    assert await main(environment) == 1
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    record = json.loads((evidence / 'execution.json').read_text())
    assert len(rejected) == 1
    assert record['status'] == 'failed' and record['terminal_result'] is None
    assert 'native reviewer raised OSError' in record['diagnostics']
    assert any('did not produce a finalized terminal result' in reason for reason in record['diagnostics'])
    assert any('native trajectory unavailable' in reason for reason in record['diagnostics'])
    assert not (evidence / 'trajectory.json').exists()
    assert not (evidence.parent / 'trajectory.json').exists()
    assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
    assert 'FileNotFoundError' not in capsys.readouterr().err


async def test_cancelled_public_reviewer_exports_joined_native_evidence_and_propagates_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, dataset_store: Path, archive_dir: Path,
) -> None:
    """Cancellation cannot delete private staging before its joined evidence is exported.

    Provider failures return a CaseExecution; cancellation unwinds review_case
    without returning it. Keep the real runner, recorder and event loop while
    cancelling a blocked external-generation stream, and prove cleanup has joined
    before the available native trajectory is copied to Harbor's logs.
    """
    environment = public_input(tmp_path)
    entered, joined = asyncio.Event(), asyncio.Event()
    role_homes: list[Path] = []
    monkeypatch.delenv('DAYDREAM_ARCHIVE_DIR', raising=False)
    prior_archive = os.environ.get('DAYDREAM_ARCHIVE_DIR')
    emitted = 'Synthetic generation stream before cancellation'
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    destinations = {evidence / 'trajectory.json', evidence.parent / 'trajectory.json'}
    published: list[Path] = []
    link = os.link

    def publish_after_join(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        target = Path(destination)
        if target in destinations:
            assert joined.is_set(), 'ATIF publication raced generation cleanup'
            published.append(target)
        link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, 'link', publish_after_join)

    class BlockedBackend(EmptyReviewBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if ('you are reviewing the ' not in prompt.lower()
                    and 'you are the structural reviewer' not in prompt.lower()):
                async for event in super().execute(cwd, prompt, *args, **kwargs):
                    yield event
                return
            yield TextEvent(emitted)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                joined.set()

    def create(*args: Any, **kwargs: Any) -> EmptyReviewBackend:
        home = Path(kwargs['execution_input'].child_environment()['CODEX_HOME'])
        role_homes.append(home)
        auth = json.loads((home / 'auth.json').read_text())
        auth['tokens']['refresh_token'] = 'private-cancelled-rotation'
        (home / 'auth.json').write_text(json.dumps(auth))
        return BlockedBackend(Path(environment['DAYDREAM_RB_REPOSITORY']))

    monkeypatch.setattr(runner, 'create_backend', create)
    running = asyncio.create_task(main(environment))
    await asyncio.wait_for(entered.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=5)
    assert joined.is_set()
    assert set(published) == destinations
    assert (evidence / 'trajectory.json').is_file()
    assert (evidence.parent / 'trajectory.json').read_bytes() == (evidence / 'trajectory.json').read_bytes()
    trajectory = json.loads((evidence / 'trajectory.json').read_text())
    siblings = [entry for entry in trajectory['extra']['subtrajectories'] if 'sibling_trajectory_ref' in entry]
    assert siblings, 'cancellation must retain the real fork, not only the intent root'
    documents = [json.loads((evidence / entry['sibling_trajectory_ref']).read_text()) for entry in siblings]
    assert all(document['session_id'] == trajectory['session_id'] for document in documents)
    assert any(step['source'] == 'agent' and step.get('message') == emitted
               and step.get('extra', {}).get('stop_reason') == 'cancelled'
               for document in documents for step in document['steps'])
    assert run_document_path(run_directory(evidence, trajectory['session_id'])).read_bytes() == (
        evidence / 'trajectory.json').read_bytes()
    assert os.environ.get('DAYDREAM_ARCHIVE_DIR') == prior_archive
    assert not list(archive_dir.iterdir()) and not dataset_store.exists()
    assert trajectory['extra']['partial'] is True
    assert any(event['event'] == 'phase_end' and event.get('status') == 'cancelled'
               for event in trajectory['extra']['phase_events'])
    record = json.loads((evidence / 'execution.json').read_text())
    assert record['status'] == 'cancelled' and record['terminal_result'] is None
    assert 'cancelled before returning' in record['diagnostics'][0]
    assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()

    assert role_homes and all(not home.exists() for home in role_homes)
    from daydream.benchmark.reviewbench.harbor_auth import reviewer_credentials

    canonical = Path(environment['DAYDREAM_RB_CODEX_HOME'])
    with reviewer_credentials(canonical, tmp_path / 'after-cancel') as home:
        assert json.loads((home / 'auth.json').read_text())['tokens']['refresh_token'] == 'private-cancelled-rotation'
