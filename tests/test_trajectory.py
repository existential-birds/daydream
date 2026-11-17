"""Recorder, invocation, and redactor tests use schema validity plus behavior predicates."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import pytest

import daydream.trajectory as trajectory_module
from daydream.backends import (
    AgentEvent,
    CostEvent,
    DiagnosticEvent,
    MetricsEvent,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.cli import _signal_handler
from daydream.trajectory import (
    DaydreamPhase,
    EvidenceRecorder,
    Invocation,
    RunWriteSnapshot,
    TrajectoryDocumentReference,
    TrajectoryDocumentSnapshot,
    flush_active_signal_recorders,
    get_current_recorder,
)
from daydream.trajectory.journal import read_journal
from daydream.trajectory.view import SNAPSHOT_EVENT
from daydream.ui import get_shutdown_panel, set_shutdown_panel
from tests.harness.trajectory import (
    make_recorder,
    observe_text_and_result,
    read_trajectory,
)


def _agent_steps(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """The trajectory's agent-sourced steps, in order."""
    return [s for s in traj["steps"] if s["source"] == "agent"]


async def test_diagnostic_is_json_safe_redacted_and_persisted_in_arrival_order(tmp_path: Path,) -> None:
    """Diagnostics cross one fail-closed recorder privacy/type boundary."""
    secret = "sk-diagnostic123456"
    unsupported = object()
    inv, steps = await _record_events(tmp_path,
        DiagnosticEvent(code=secret, message=f"API_KEY={secret}",
            metadata={"nested": {secret: f"TOKEN={secret}", "api_key": secret}, "unsupported": unsupported,
                "unsupported_key": {42: "retained safely"}, "non_finite": float("nan"),
            },
        ), DiagnosticEvent(code="second", message="safe", metadata={"count": 2}),
        MetricsEvent(message_id="m1", prompt_tokens=5, completion_tokens=3, cached_tokens=0, cost_usd=0.001),
        TurnEndEvent(message_id="m1"),
        ResultEvent(structured_output=None, continuation=None),
    )

    lifecycle = inv._generation_ledger.to_dict()
    assert lifecycle["drafts"] == [] and lifecycle["billing_owner"] == "unresolved"
    assert "generation_lifecycle" not in read_trajectory(inv.recorder.path)["extra"]["subtrajectories"][0]
    assert len(steps) == 1
    records = steps[0]["extra"]["backend_diagnostics"]
    assert [record["code"] for record in records] == ["[REDACTED_API_KEY]", "second"]
    encoded = json.dumps(records, allow_nan=False)
    assert secret not in encoded
    assert "[REDACTED" in encoded
    assert records[0]["metadata"]["unsupported"] == "[UNSUPPORTED_DIAGNOSTIC_VALUE]"
    assert records[0]["metadata"]["unsupported_key"] == {"[UNSUPPORTED_DIAGNOSTIC_KEY]": "retained safely"}
    assert records[0]["metadata"]["non_finite"] == "[UNSUPPORTED_DIAGNOSTIC_VALUE]"
    assert json.loads(encoded) == records

async def test_diagnostic_redaction_failure_persists_fixed_safe_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-neverpersist123456"

    def fail_redaction(value: Any, sensitive: bool = False) -> Any:
        raise RuntimeError("redaction unavailable")

    monkeypatch.setattr("daydream.trajectory.invocation.redact_value", fail_redaction)
    _, steps = await _record_events(tmp_path, DiagnosticEvent(code=secret, message=secret, metadata={"raw": secret}),
        ResultEvent(structured_output=None, continuation=None),
    )

    records = steps[0]["extra"]["backend_diagnostics"]
    assert records == [
        {"code": "diagnostic_redaction_failed", "message": "[DIAGNOSTIC_REDACTION_FAILED]", "metadata": {}}
    ]
    assert secret not in json.dumps(records)


async def _record_events(tmp_path: Path, *events: AgentEvent) -> tuple[Invocation, list[dict[str, Any]]]:
    """Observe *events* in one invocation; return it with its materialized steps."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            for event in events:
                inv.observe(event)
    return inv, inv.steps


INCOMPLETE_CONTENT = "[interrupted: call did not complete before invocation ended]"


async def test_dispatch_failure_is_caught_and_run_continues(
    recorder: EvidenceRecorder, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavior 6: Recorder boundary catches dispatch exceptions; run continues."""
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="before-failure"))

            # Force a single dispatch failure on the next observe().
            original_dispatch = inv._dispatch
            call_count = {"n": 0}

            def boom(event: Any) -> None:
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("simulated dispatch failure")
                original_dispatch(event)

            monkeypatch.setattr(inv, "_dispatch", boom)

            # This call's dispatch raises; observe() must catch it.
            inv.observe(TextEvent(text="will-fail"))
            observe_text_and_result(inv, "after-failure")

    traj = read_trajectory(recorder.path)
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    # "before-failure" + "after-failure" both make it; "will-fail" was dropped.
    assert "before-failure" in agent_steps[0]["message"]
    assert "after-failure" in agent_steps[0]["message"]
    assert "will-fail" not in agent_steps[0]["message"]


async def test_dispatch_body_failure_and_cancellation_own_terminal_status(tmp_path: Path) -> None:
    """An escaping body exception fails the dispatch; cancellation overrides an optimistic finish."""
    for ending in ('empty-error', 'optimistic-cancel'):
        recorder = make_recorder(tmp_path / ending)

        async def dispatch_body(scope: anyio.CancelScope | None, *, recorder: Any = recorder,
                                ending: str = ending) -> None:
            async with trajectory_module.dispatch_scope(recorder, phase=DaydreamPhase.FIX,
                                                         descriptors=('child',)) as dispatch:
                assert dispatch is not None
                if ending == 'optimistic-cancel':
                    dispatch.finish(trajectory_module.LifecycleStatus.SUCCEEDED)
                async with recorder.fork('child', dispatch=dispatch) as child:
                    if ending == 'optimistic-cancel':
                        async with child.invocation(phase=DaydreamPhase.FIX) as invocation:
                            observe_text_and_result(invocation, 'before cancellation')
                if scope is None:
                    raise RuntimeError('dispatch body failed')
                scope.cancel()
                await anyio.sleep_forever()

        async with recorder:
            if ending == 'empty-error':
                with pytest.raises(RuntimeError, match='dispatch body failed'):
                    await dispatch_body(None)
            else:
                with anyio.CancelScope() as scope:
                    await dispatch_body(scope)
        dispatches = [step for step in read_trajectory(recorder.path)['steps']
                      if step.get('llm_call_count') == 0 and 'dispatch_id' in (step.get('extra') or {})]
        assert len(dispatches) == 1, ending
        status = dispatches[0]['extra']
        if ending == 'empty-error':
            assert (status['dispatch_status'], status['reason_code']) == ('failed', 'uncaught_exception')
            assert status['attempted_count'] == 1 and status['completed_count'] == 0
        else:
            assert (status['dispatch_status'], status['reason_code']) == ('cancelled', 'cancelled')


async def test_seal_failure_is_optional_and_reads_back_as_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed seal warns once, keeps the run going and publishes the accepted prefix as partial."""
    warnings: list[str] = []
    snapshots: list[RunWriteSnapshot] = []
    monkeypatch.setattr('daydream.ui.print_warning', lambda _console, message: warnings.append(message))
    recorder = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError('denied')

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            observe_text_and_result(invocation, 'retained response')
        monkeypatch.setattr('os.fsync', fail)
    assert len(warnings) == 1 and 'Evidence write failed (PermissionError)' in warnings[0]
    snapshot, = snapshots
    assert snapshot.status == 'partial'
    view = snapshot.documents[0].validated_payload(recorder.session_id)
    assert view['extra']['partial'] is True and 'run_ended_at' not in view['extra']
    assert [step['message'] for step in _agent_steps(view)] == ['retained response']


UNICODE_TEXT = 'caf\u00e9 \U0001f600 \u2028\x00'
HOST_NAME = 'caf\udce9'  # A host value from an undecodable file name keeps a lone surrogate.


@pytest.mark.parametrize(('label', 'planned', 'identified'), [
    ('auth=Bearer [REDACTED][REDACTED_CREDENTIAL]', 'auth=Bearer [REDACTED][REDACTED_CREDENTIAL]', True),
    ('{"type":"api_key","key":"opaque-first-canary"}', None, True),
])
async def test_partial_graph_preserves_empty_ancestry_and_original_private_identities(
    tmp_path: Path, label: str, planned: str | None, identified: bool,
) -> None:
    private = 'opaque-first-canary' in label
    second_label = label.replace('opaque-first-canary', 'opaque-second-canary') if private else 'second-public'
    queued = '{"type":"oauth","access":"opaque-queued-canary"}'
    snapshots: list[RunWriteSnapshot] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    root.record_profile(schema_version=1, name=HOST_NAME, source_kind='file', digest='d' * 64)
    async with root:
        async with trajectory_module.dispatch_scope(root if identified else None, phase=DaydreamPhase.DEEP,
            descriptors=[label, second_label, queued]) as dispatch:
            async with root.fork(label, dispatch=dispatch) as parent:
                root.write_partial()
                assert len(snapshots[-1].documents) == 2  # Entered forks exist before their first invocation.
                async with parent.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                    root.write_partial()
                    first = snapshots[-1]
                    frozen = tuple(document.read_bytes() for document in first.documents)
                    assert len(first.documents) == 2
                    for document in first.documents:
                        payload = document.validated_payload(root.session_id)
                        assert payload['steps'] == []  # Evidence never invents a step for an empty snapshot.
                        assert payload['extra']['snapshot_at'] == first.cutoff_at
                        assert 'total_prompt_tokens' not in payload['final_metrics']
                    active = read_trajectory(first.documents[1])['extra']['subtrajectories'][0]
                    assert active['ended_at'] is None and active['step_ids'] == []
                    secret = '{"type":"oauth","access":"opaque-event-canary"}'
                    split = secret.index('opaque-event-canary') + len('opaque-event-canary')
                    invocation.observe(TextEvent(secret[:split]))
                    root.write_partial()
                    interrupted = snapshots[-1]
                    interrupted_bytes = tuple(document.read_bytes() for document in interrupted.documents)
                    assert b'opaque-event-canary' not in b''.join(interrupted_bytes)
                    invocation.observe(TextEvent(secret[split:]))
                    events: list[AgentEvent] = [ToolStartEvent('private', 'Inspect',
                        {'type': 'api_key', 'key': 'opaque-input-canary'}), ToolResultEvent('private', secret, False),
                        RequestEvent(prompt='Private prompt', config=PiRequestConfig(read_only=True),
                            model_name='sk-request-canary', provider_name='public-provider', reasoning_effort='medium'),
                        CostEvent(input_tokens=0, output_tokens=None, cached_tokens=0, cost_usd=1.5e-05),
                        TurnEndEvent()]
                    for event in events:
                        invocation.observe(event)
                    async with parent.fork('completed-leaf') as leaf:
                        async with leaf.invocation(phase=DaydreamPhase.REVIEW) as inner:
                            observe_text_and_result(inner, 'completed nested evidence')
                    completed_bytes = leaf.path.read_bytes()
                    # Once a fork exits, the run keeps only its written-file reference.
                    leaf_reference, = root.completed_child_documents()
                    assert isinstance(leaf_reference, TrajectoryDocumentReference)
                    assert (leaf_reference.private_path, leaf_reference.size) == (leaf.path, len(completed_bytes))
                    async with root.fork(second_label, dispatch=dispatch) as second:
                        async with second.invocation(phase=DaydreamPhase.REVIEW):
                            root.write_partial()
                            previous = snapshots[-1]
                            root.write_partial()
                            assert snapshots[-1].cutoff_at == previous.cutoff_at
                            invocation.observe_user_step(prompt=UNICODE_TEXT)  # Later context, same membership.
                            root.write_partial()
                            snapshot = snapshots[-1]
                            assert snapshot.cutoff_at != previous.cutoff_at
                            documents = {document.trajectory_id: document for document in snapshot.documents}
                            assert set(documents) == {root.trajectory_id, parent.trajectory_id, leaf.trajectory_id,
                                                      second.trajectory_id}
                            assert parent.trajectory_id != second.trajectory_id and parent.path != second.path
                            assert (parent.descriptor != label) == private
                            assert (second.descriptor != second_label) == private
                            if not identified and not private:
                                assert parent.trajectory_id == root.session_id + ':' + label
                                assert second.trajectory_id == root.session_id + ':' + second_label
                            for document in snapshot.documents:
                                raw = document.read_bytes()
                                payload = document.validated_payload(root.session_id)
                                assert raw.isascii() and '-canary' not in raw.decode()
                                if document.trajectory_id != leaf.trajectory_id:
                                    assert (payload['extra']['snapshot_at'], payload['extra']['partial']) == (
                                        snapshot.cutoff_at, True)
                                    assert document.path.read_bytes() == raw  # Nothing follows the snapshot yet.
                                for entry in payload.get('extra', {}).get('subtrajectories', []):
                                    if reference := entry.get('sibling_trajectory_ref'):
                                        target = documents[entry['trajectory_id']]
                                        logical = target.path.relative_to(tmp_path / '.daydream').as_posix()
                                        assert logical == reference
                            assert documents[leaf.trajectory_id] == leaf_reference
                            assert leaf_reference.read_bytes() == completed_bytes  # Standalone writes exact bytes.
                            partial = read_trajectory(documents[parent.trajectory_id])
                            assert partial['final_metrics']['total_prompt_tokens'] == 0
                            assert 'total_completion_tokens' not in partial['final_metrics']
                            assert partial['final_metrics']['total_cost_usd'] == 1.5e-05
                            assert partial['steps'][-1]['message'] == UNICODE_TEXT
                            assert tuple(document.read_bytes() for document in first.documents) == frozen
                            assert '-canary' not in str(parent.path) + str(second.path)
                            with pytest.raises(AttributeError):
                                setattr(snapshot, 'cutoff_at', 'changed')
                            if dispatch is not None:
                                extra = read_trajectory(documents[root.trajectory_id])['extra']
                                assert extra['profile_name'] == HOST_NAME
                                plan, = extra['active_dispatches']
                                assert plan['dispatch_id'] == dispatch.dispatch_id
                                assert plan['planned_descriptors'][:2] == [
                                    parent.descriptor if planned is None else planned, second.descriptor]
                                assert len(plan['queued_descriptors']) == 1 and '-canary' not in json.dumps(plan)
                                assert plan['attempted_count'] == 2 and plan['completed_count'] == 0
    assert tuple(document.read_bytes() for document in interrupted.documents) == interrupted_bytes
    final_parent = read_trajectory(parent.path)
    message = next(step['message'] for step in final_parent['steps'] if step['source'] == 'agent')
    assert json.loads(message)['type'] == 'oauth'  # Snapshot redaction must not mutate the live text buffer.
    usage = root.usage.usage()
    requests = [request for invocation in read_trajectory(parent.path)['extra']['subtrajectories']
                for request in invocation.get('requests', [])]
    assert usage['prompt_tokens'] == usage['cached_tokens'] == 0 and 'completion_tokens' not in usage
    assert len(requests) == 1 and requests[0]['backend'] == 'pi' and requests[0]['controls']['read_only']
    assert requests[0]['provider'] == 'public-provider' and requests[0]['reasoning_effort'] == 'medium'
    final_bytes = parent.path.read_text() + (root.path.read_text() if identified else '')
    assert '-canary' not in json.dumps(requests) + final_bytes


# Fork / Sibling / Continuation tests (Phase 3, SUBA-01..09)


async def test_fork_paths_confine_public_descriptors(tmp_path: Path) -> None:
    label = '   '
    root = make_recorder(tmp_path)
    async with root:
        with pytest.raises(ValueError, match='empty slug'):
            async with root.fork(label):
                pass


async def test_fork_write_failure_degrades_to_truthful_incomplete_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that cannot seal keeps no reference, restores its parent, and marks the run partial."""
    snapshots: list[RunWriteSnapshot] = []
    recorder = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    warnings_emitted: list[str] = []

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)
        with patch("daydream.ui.print_warning", lambda _console, message: warnings_emitted.append(message)):
            async with recorder.fork("fail-child") as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    inv.observe(MetricsEvent(message_id="lost", prompt_tokens=5, completion_tokens=1,
                                             cached_tokens=None, cost_usd=None))
                    observe_text_and_result(inv)
                journal, fsync = child._journal, os.fsync

                def denied(fd: int) -> None:
                    if fd == journal._fd:
                        raise OSError("disk full")
                    fsync(fd)

                monkeypatch.setattr(os, "fsync", denied)

        assert get_current_recorder() is recorder
        assert recorder.completed_child_documents() == ()  # A failed child keeps no reference.
        # Nor does its usage fold, in total or in its phase's share.
        assert (recorder.usage.prompt, recorder.usage.phases) == (None, {})
        flush_active_signal_recorders()
        assert [document.trajectory_id for document in snapshots[-1].documents] == [recorder.trajectory_id]

    assert any("Sibling evidence write failed" in m for m in warnings_emitted)
    final = snapshots[-1]
    assert final.status == "partial"
    root = final.documents[0].validated_payload(recorder.session_id)
    assert root["extra"]["partial"] is True and root["extra"]["snapshot_at"] == root["extra"]["run_ended_at"]
    seal = read_journal(recorder.path.read_bytes()).seal
    assert seal is not None and seal.body["failed_documents"] == [child.trajectory_id]
    assert journal.failure == "OSError" and not read_journal(child.path.read_bytes()).complete


# write_partial tests (CLI-03, D-07 SIGINT partial flush)


async def test_partial_writes_preserve_live_state_and_joined_steps(tmp_path: Path) -> None:
    snapshots: list[RunWriteSnapshot] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe_user_step(prompt='retained prompt')
            invocation.observe(ToolStartEvent(id='pending', name='Read', input={'file_path': 'a.py'}))
            root.write_partial()
            first = snapshots[-1].documents[0].validated_payload(root.session_id)
        root.write_partial()
        joined_snapshot = snapshots[-1]
        root.write_partial()
        assert snapshots[-1] == joined_snapshot  # No new evidence republishes the same snapshot.
        joined = joined_snapshot.documents[0].validated_payload(root.session_id)
    final = read_trajectory(root.path)
    assert len(joined['steps']) == len(final['steps'])
    for document in (first, joined, final):
        observation, = _agent_steps(document)[0]['observation']['results']
        assert observation == {'content': INCOMPLETE_CONTENT,
                               'extra': {'is_error': True, 'status': 'interrupted'}}


def _snapshot_count(recorder: EvidenceRecorder) -> int:
    """Signal snapshots appended to one document."""
    return sum(record.event_name == SNAPSHOT_EVENT for record in read_journal(recorder.path.read_bytes()).records)


async def _hold_fork(
    parent: EvidenceRecorder, descriptor: str, marker: str, entered: anyio.Event, release: anyio.Event,
    children: dict[str, EvidenceRecorder],
) -> None:
    """Hold one public fork invocation open across a signal-flush barrier."""
    async with parent.fork(descriptor) as child:
        children[descriptor] = child
        async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
            assert get_current_recorder() is child
            observe_text_and_result(inv, marker)
            entered.set()
            await release.wait()


async def test_signal_flush_prepare_failure_never_publishes_disconnected_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots: list[RunWriteSnapshot] = []
    warnings: list[str] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    monkeypatch.setattr('daydream.ui.print_warning', lambda _console, message: warnings.append(message))
    async with root:
        async with root.fork('active-child') as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                invocation.observe(TextEvent('valid previous evidence'))
                flush_active_signal_recorders()
                previous = snapshots[-1]
                saved = {document.trajectory_id: document.read_bytes() for document in previous.documents}

                def fail() -> Any:
                    raise RuntimeError('document preparation failed')

                invocation.observe(TextEvent(' later evidence'))
                with monkeypatch.context() as patcher:
                    patcher.setattr('daydream.trajectory.recorder.redact_value', lambda value: fail())
                    flush_active_signal_recorders()
                assert snapshots == [previous] and len(warnings) == 1
                flush_active_signal_recorders()
                current = snapshots[-1]
                assert current.cutoff_at != previous.cutoff_at
                assert {document.trajectory_id for document in current.documents} == {
                    root.trajectory_id, child.trajectory_id}
                for document in current.documents:
                    payload = document.validated_payload(root.session_id)
                    assert payload['extra']['snapshot_at'] == current.cutoff_at
                assert all(document.path.read_bytes() == document.read_bytes() for document in current.documents)
                assert {document.trajectory_id: document.read_bytes() for document in previous.documents} == saved


@pytest.mark.parametrize("exit_on", ["body", "final-write"])
async def test_signal_flush_excludes_exited_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  exit_on: str) -> None:
    """A child that exits, even through a BaseException from its own seal, takes no later snapshot."""
    recorder = make_recorder(tmp_path)

    def fail_final_write() -> None:
        raise SystemExit(23)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")
        with pytest.raises(SystemExit) as exc:
            async with recorder.fork(f"exited-{exit_on}") as child:
                async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                    observe_text_and_result(inv, "EXITED_CHILD_ONLY")
                if exit_on == "body":
                    raise SystemExit(17)
                monkeypatch.setattr(child, "_write_final", fail_final_write)
        assert exc.value.code == (17 if exit_on == "body" else 23)
        flush_active_signal_recorders()
        assert _snapshot_count(recorder) == 1
        assert _snapshot_count(child) == 0


def _finish_shutdown_panel() -> None:

    panel = get_shutdown_panel()
    if panel is not None:
        panel.finish()
        set_shutdown_panel(None)

async def test_signal_handler_flushes_siblings_despite_a_writer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One sibling's failed write marks only that document incomplete; the others take the snapshot."""
    signum = signal.SIGTERM
    snapshots: list[RunWriteSnapshot] = []
    recorder = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    # The second label keeps a lone surrogate from an undecodable file name; the flush still never raises.
    entered = {name: anyio.Event() for name in ('signal-a', 'signal-b-caf\udce9')}
    release = {name: anyio.Event() for name in entered}
    children: dict[str, EvidenceRecorder] = {}
    writes = os.write
    warnings: list[str] = []
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, 'ROOT_ONLY')
        async with anyio.create_task_group() as group:
            for name in entered:
                group.start_soon(_hold_fork, recorder, name, name, entered[name], release[name], children)
                await entered[name].wait()
            failed = children['signal-a']._journal

            def write(fd: int, data: Any) -> int:
                if fd == failed._fd:
                    raise PermissionError('denied sibling')
                return writes(fd, data)

            monkeypatch.setattr(os, 'write', write)
            monkeypatch.setattr('daydream.ui.print_warning', lambda _console, message: warnings.append(message))
            try:
                with pytest.raises(KeyboardInterrupt):
                    _signal_handler(signum, None)
                snapshot, = snapshots
                assert snapshot.status == 'partial' and len(snapshot.documents) == 3
                for document in snapshot.documents:
                    view = document.validated_payload(recorder.session_id)
                    taken = view['extra'].get('snapshot_at') == snapshot.cutoff_at
                    assert taken == (document.trajectory_id != children['signal-a'].trajectory_id)
                assert failed.failure == 'PermissionError' and len(warnings) == 1
            finally:
                _finish_shutdown_panel()
                for event in release.values():
                    event.set()


async def test_explicit_document_writer_failure_never_replaces_primary(tmp_path: Path) -> None:
    """A failing explicit-path document writer during finalization never replaces the body's primary error."""
    def writer(document: TrajectoryDocumentSnapshot, status: str) -> TrajectoryDocumentReference:
        raise OSError('secondary writer failure')

    recorder = make_recorder(tmp_path, document_writer=writer, explicit_path=True)
    primary = RuntimeError('authoritative body failure')
    propagated: RuntimeError | None = None
    try:
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                observe_text_and_result(invocation, 'retained evidence')
            raise primary
    except RuntimeError as caught:
        propagated = caught
    assert propagated is primary
    assert any('evidence finalization' in note for note in primary.__notes__)
