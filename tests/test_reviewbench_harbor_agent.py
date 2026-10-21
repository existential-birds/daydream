"""Public reviewer boundary: real frozen Git, runner and role-private auth files."""
from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import json
import os
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.artifacts import external
from daydream.backends import AgentEvent, TextEvent
from daydream.backends.codex import CodexBackend
from daydream.benchmark.reviewbench.contract import REVIEWBENCH_REVISION
from daydream.benchmark.reviewbench.harbor_entrypoint import main
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git
from tests.test_reviewbench_native import finding, frozen_case


def public_input(tmp_path: Path) -> dict[str, str]:
    repo, case = frozen_case(tmp_path)
    git(repo, 'remote', 'remove', 'origin')
    git_ops.update_refs(repo, {'refs/reviewbench/base': case.base, 'refs/reviewbench/head': case.head})
    inputs = tmp_path / 'pr'
    inputs.mkdir()
    pr = {**asdict(case), 'title': 'Frozen supplied title', 'body': 'Frozen supplied author explanation'}
    (inputs / 'pr.json').write_text(json.dumps(pr))
    (inputs / 'diff.patch').write_text(git_ops.diff_paths(repo, case.base, case.head, ['.']))
    profile = {'schema': 1, 'upstream_revision': REVIEWBENCH_REVISION, 'pr': pr,
               'task_id': 7, 'evaluation_id': 99,
               'reviewer': {'model': 'gpt-6.1-sol', 'reasoning_effort': 'high', 'fanout_concurrency': 1}}
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


@pytest.mark.parametrize('outcome', ['complete', 'incomplete', 'conversion_limited', 'provider_failed'])
async def test_public_harbor_native_boundary_keeps_terminal_evidence_and_refuses_unscorable_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, outcome: str,
) -> None:
    """A successful subprocess alone must not publish partial or discarded findings.

    The existing native tests do not exercise public origin-free snapshots,
    subscription leases, private trajectory retention or Harbor candidate output.
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

    class SerializedBackend(EmptyReviewBackend):
        async def execute(self, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            try:
                await asyncio.sleep(0.01)
                async for event in super().execute(*args, **kwargs):
                    yield event
            finally:
                active -= 1

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        assert name == 'codex'
        assert kwargs['execution_input'].fanout_concurrency == 1
        child = kwargs['execution_input'].child_environment()
        assert not {'OPENAI_API_KEY', 'GH_TOKEN', 'OPENAI_BASE_URL'} & child.keys()
        home = Path(child['CODEX_HOME'])
        role_homes.append(home)
        assert home != Path(environment['DAYDREAM_RB_CODEX_HOME'])
        assert json.loads((home / 'auth.json').read_text())['auth_mode'] == 'chatgpt'
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
    assert record['subscription_access'] == 'chatgpt-oauth'
    assert active == 0 and peak_active == 1
    assert record['configuration']['native']['codex']['fanout_concurrency'] == 1
    assert record['configuration']['native']['codex']['subscription_access'] == 'chatgpt-oauth'
    assert role_homes and all(not home.exists() for home in role_homes)
    assert any('Frozen supplied author explanation' in call['prompt'] for backend in backends for call in backend.calls)
    assert 'private-refresh-canary' not in (evidence / 'execution.json').read_text()
    assert 'forbidden-api-canary' not in (evidence / 'trajectory.json').read_text()
    output = Path(environment['DAYDREAM_RB_OUTPUT'])
    assert output.exists() == (outcome == 'complete')
    assert result == (0 if outcome == 'complete' else 1)
    if output.exists():
        assert json.loads(output.read_text())['findings'][0]['file'] == 'api.py'
    assert not fake_gh.calls('POST')


@pytest.mark.parametrize('fault', ['profile', 'patch', 'refs', 'origin', 'stale', 'stale_trajectory', 'auth'])
async def test_public_harbor_refuses_unbound_source_or_stale_artifacts_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, fault: str,
) -> None:
    """Public provenance must not weaken ordinary origin checks or replay artifacts."""
    environment = public_input(tmp_path)
    inputs = Path(environment['DAYDREAM_RB_INPUTS'])
    if fault == 'profile':
        profile_path = inputs / 'daydream-profile.json'
        profile = json.loads(profile_path.read_text())
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
    output = Path(environment['DAYDREAM_RB_OUTPUT'])
    if fault == 'stale':
        assert output.read_text() == 'previous-evidence'
    else:
        assert not output.exists()
    if fault == 'stale_trajectory':
        assert trajectory.read_text() == 'previous-native-trajectory'


async def test_native_storage_failure_retains_diagnostics_without_manufacturing_atif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    """Storage preflight can fail before any trajectory or model exists.

    Existing provider-failure rows still have a native recorder. Faulting the
    actual filesystem exchange operation covers the earlier owner rejection and
    prevents a misleading missing-file exception from replacing its evidence.
    """
    environment = public_input(tmp_path)
    monkeypatch.setattr(external._AtomicNameExchange, 'call',
                        lambda *_args: external._NameExchangeResult(-1, errno.ENOTSUP))

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('artifact owner preflight failure reached model generation')

    monkeypatch.setattr(runner, 'create_backend', forbidden)
    assert await main(environment) == 1
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    record = json.loads((evidence / 'execution.json').read_text())
    assert record['status'] == 'failed' and record['terminal_result'] is None
    assert any('did not produce a finalized terminal result' in reason for reason in record['diagnostics'])
    assert any('native trajectory unavailable' in reason for reason in record['diagnostics'])
    assert not (evidence / 'trajectory.json').exists()
    assert not (evidence.parent / 'trajectory.json').exists()
    assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
    assert 'FileNotFoundError' not in capsys.readouterr().err


async def test_cancelled_public_reviewer_exports_joined_native_evidence_and_propagates_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    """Cancellation cannot delete private staging before its joined evidence is exported.

    Provider failures return a CaseExecution; cancellation unwinds review_case
    without returning it. Keep the real runner, recorder and event loop while
    cancelling a blocked external-generation stream, and prove cleanup has joined
    before the available native trajectory is copied to Harbor's logs.
    """
    environment = public_input(tmp_path)
    entered, joined = asyncio.Event(), asyncio.Event()
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
        async def execute(self, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            yield TextEvent(emitted)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                joined.set()

    monkeypatch.setattr(runner, 'create_backend', lambda *args, **kwargs:
                        BlockedBackend(Path(environment['DAYDREAM_RB_REPOSITORY'])))
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
    assert any(step['source'] == 'agent' and step.get('message') == emitted
               and step.get('extra', {}).get('stop_reason') == 'cancelled' for step in trajectory['steps'])
    assert trajectory['extra']['partial'] is True
    assert any(event['event'] == 'phase_end' and event.get('status') == 'cancelled'
               for event in trajectory['extra']['phase_events'])
    record = json.loads((evidence / 'execution.json').read_text())
    assert record['status'] == 'cancelled' and record['terminal_result'] is None
    assert 'cancelled before returning' in record['diagnostics'][0]
    assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
