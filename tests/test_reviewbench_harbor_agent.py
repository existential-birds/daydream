"""Public reviewer boundary: real frozen Git, runner and role-private auth files."""
from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import shlex
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream import runner
from daydream.backends import AgentEvent, TextEvent
from daydream.benchmark.reviewbench.harbor_agent import DaydreamPublicReviewAgent
from daydream.benchmark.reviewbench.harbor_entrypoint import main
from daydream.trajectory import RUNS_DIRNAME, run_directory, run_document_path
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git
from tests.harness.reviewbench import public_input


@pytest.mark.parametrize('stage', ['setup', 'run'])
async def test_public_agent_python_startup_cannot_import_reviewed_repository_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    """Isolate reviewed imports while allowing the launcher and reviewer to have different versions."""
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
    monkeypatch.setattr(agent, 'version', lambda: '0.0.0-host')
    if stage == 'setup':
        await agent.setup(Environment())
    else:
        with pytest.raises(RuntimeError, match='native reviewer failed'):
            await agent.run('supplied task', Environment(), SimpleNamespace())
        assert 'SubscriptionAuthError' in (logs / 'reviewer-entrypoint.log').read_text()
    assert not marker.exists()
    assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()


@pytest.mark.parametrize('fault', [
    'schema', 'profile', 'patch', 'refs', 'origin', 'dirty', 'stale', 'stale_trajectory', 'auth',
])
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
    elif fault == 'dirty':
        (Path(environment['DAYDREAM_RB_REPOSITORY']) / 'untracked.py').write_text('unfrozen source')
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
    """A failed private artifact mkdir retains diagnostics without fabricating ATIF or model work."""
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
    """Cancellation joins native streams before exporting fork evidence and restoring auth isolation."""
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
