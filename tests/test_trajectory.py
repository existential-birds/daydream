"""Recorder, invocation, and redactor tests use schema validity plus behavior predicates."""

from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import pytest

import daydream.trajectory as trajectory_module
from daydream.atif import validate as atif_validate
from daydream.atif.models import Step
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
    RUN_DOCUMENT_NAME,
    DaydreamPhase,
    Invocation,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
    flush_active_signal_recorders,
    get_current_recorder,
    partial_document_path,
    snapshot_trajectories,
)
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
    records = (steps[0].extra or {})["backend_diagnostics"]
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

    records = (steps[0].extra or {})["backend_diagnostics"]
    assert records == [
        {"code": "diagnostic_redaction_failed", "message": "[DIAGNOSTIC_REDACTION_FAILED]", "metadata": {}}
    ]
    assert secret not in json.dumps(records)


async def _record_events(tmp_path: Path, *events: AgentEvent) -> tuple[Invocation, list[Step]]:
    """Observe *events* in one invocation; return it with its materialized steps."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            for event in events:
                inv.observe(event)
    return inv, inv.steps


INCOMPLETE_CONTENT = "[interrupted: call did not complete before invocation ended]"


async def test_dispatch_failure_is_caught_and_run_continues(
    recorder: TrajectoryRecorder, monkeypatch: pytest.MonkeyPatch,
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
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    # "before-failure" + "after-failure" both make it; "will-fail" was dropped.
    assert "before-failure" in agent_steps[0]["message"]
    assert "after-failure" in agent_steps[0]["message"]
    assert "will-fail" not in agent_steps[0]["message"]


@pytest.mark.parametrize('stage', ['final'])
async def test_filesystem_writer_failure_is_optional(
    recorder: TrajectoryRecorder, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr('daydream.ui.print_warning', lambda _console, message: warnings.append(message))

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError('denied')

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            observe_text_and_result(invocation, 'retained response')
        if stage == 'partial':
            with monkeypatch.context() as patcher:
                patcher.setattr(Path, 'write_text', fail)
                recorder.write_partial()
        else:
            monkeypatch.setattr('os.replace', fail)
    expected = 'Partial trajectory write failed' if stage == 'partial' else 'Trajectory write failed'
    assert len(warnings) == 1 and expected in warnings[0]


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
    async with root:
        async with trajectory_module.dispatch_scope(root if identified else None, phase=DaydreamPhase.DEEP,
            descriptors=[label, second_label, queued]) as dispatch:
            async with root.fork(label, dispatch=dispatch) as parent:
                root.write_partial()
                assert len(snapshots[-1].documents) == 2  # Entered forks exist before their first invocation.
                async with parent.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                    root.write_partial()
                    first = snapshots[-1]
                    frozen = tuple(document.json_bytes for document in first.documents)
                    assert len(first.documents) == 2
                    for document in first.documents:
                        payload = json.loads(document.json_bytes)
                        assert payload['steps'][0]['extra']['host_event'] == 'partial_snapshot'
                        assert payload['extra']['snapshot_at'] == first.cutoff_at
                        assert payload['final_metrics'].get('total_prompt_tokens') is None
                    active = json.loads(first.documents[1].json_bytes)['extra']['subtrajectories'][0]
                    assert active['ended_at'] is None and active['step_ids'] == []
                    secret = '{"type":"oauth","access":"opaque-event-canary"}'
                    split = secret.index('opaque-event-canary') + len('opaque-event-canary')
                    invocation.observe(TextEvent(secret[:split]))
                    root.write_partial()
                    interrupted = snapshots[-1]
                    interrupted_bytes = tuple(document.json_bytes for document in interrupted.documents)
                    assert b'opaque-event-canary' not in b''.join(interrupted_bytes)
                    invocation.observe(TextEvent(secret[split:]))
                    events: list[AgentEvent] = [ToolStartEvent('private', 'Inspect',
                        {'type': 'api_key', 'key': 'opaque-input-canary'}), ToolResultEvent('private', secret, False),
                        RequestEvent(prompt='Private prompt', config=PiRequestConfig(read_only=True),
                            model_name='sk-request-canary', provider_name='public-provider', reasoning_effort='medium'),
                        CostEvent(input_tokens=0, output_tokens=None, cached_tokens=0, cost_usd=None), TurnEndEvent()]
                    for event in events:
                        invocation.observe(event)
                    async with parent.fork('completed-leaf') as leaf:
                        async with leaf.invocation(phase=DaydreamPhase.REVIEW) as inner:
                            observe_text_and_result(inner, 'completed nested evidence')
                    completed_bytes = leaf.path.read_bytes()
                    async with root.fork(second_label, dispatch=dispatch) as second:
                        async with second.invocation(phase=DaydreamPhase.REVIEW):
                            root.write_partial()
                            previous = snapshots[-1]
                            root.write_partial()
                            assert snapshots[-1].cutoff_at == previous.cutoff_at
                            invocation.observe_user_step(prompt='later context with unchanged graph membership')
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
                                payload = json.loads(document.json_bytes)
                                assert atif_validate(payload, validate_images=False)
                                assert '-canary' not in document.json_bytes.decode()
                                if document.trajectory_id != leaf.trajectory_id:
                                    assert payload['extra']['snapshot_at'] == snapshot.cutoff_at
                                for entry in payload.get('extra', {}).get('subtrajectories', []):
                                    if reference := entry.get('sibling_trajectory_ref'):
                                        target = documents[entry['trajectory_id']]
                                        logical = target.path.relative_to(tmp_path / '.daydream').as_posix()
                                        assert logical.removesuffix('.partial') == reference
                            assert documents[leaf.trajectory_id].json_bytes == completed_bytes
                            partial = json.loads(documents[parent.trajectory_id].json_bytes)
                            assert partial['final_metrics']['total_prompt_tokens'] == 0
                            assert partial['final_metrics'].get('total_completion_tokens') is None
                            assert tuple(document.json_bytes for document in first.documents) == frozen
                            assert '-canary' not in str(parent.path) + str(second.path)
                            assert snapshot_trajectories(snapshot)['main']['_source_file'] == RUN_DOCUMENT_NAME
                            with pytest.raises(AttributeError):
                                setattr(snapshot, 'cutoff_at', 'changed')
                            if dispatch is not None:
                                extra = json.loads(documents[root.trajectory_id].json_bytes)['extra']
                                plan, = extra['active_dispatches']
                                assert plan['dispatch_id'] == dispatch.dispatch_id
                                assert plan['planned_descriptors'][:2] == [
                                    parent.descriptor if planned is None else planned, second.descriptor]
                                assert len(plan['queued_descriptors']) == 1 and '-canary' not in json.dumps(plan)
                                assert plan['attempted_count'] == 2 and plan['completed_count'] == 0
    assert tuple(document.json_bytes for document in interrupted.documents) == interrupted_bytes
    final_parent = json.loads(parent.path.read_bytes())
    message = next(step['message'] for step in final_parent['steps'] if step['source'] == 'agent')
    assert json.loads(message)['type'] == 'oauth'  # Snapshot redaction must not mutate the live text buffer.
    usage = root.review_usage()
    requests = [request for invocation in json.loads(parent.path.read_bytes())['extra']['subtrajectories']
                for request in invocation.get('requests', [])]
    assert usage['prompt_tokens'] == usage['cached_tokens'] == 0 and 'completion_tokens' not in usage
    assert len(requests) == 1 and requests[0]['backend'] == 'pi' and requests[0]['controls']['read_only']
    assert requests[0]['provider'] == 'public-provider' and requests[0]['reasoning_effort'] == 'medium'
    final_bytes = parent.path.read_text() + (root.path.read_text() if identified else '')
    assert '-canary' not in json.dumps(requests) + final_bytes


# Fork / Sibling / Continuation tests (Phase 3, SUBA-01..09)


@pytest.mark.parametrize('label', ['   '])
async def test_fork_paths_confine_public_descriptors(tmp_path: Path, label: str) -> None:
    root = make_recorder(tmp_path)
    async with root:
        if label in {'', '...', '   '}:
            with pytest.raises(ValueError, match='empty slug'):
                async with root.fork(label):
                    pass
        else:
            async with root.fork(label) as child:
                assert child.descriptor == label
                assert child.path.is_relative_to(tmp_path / '.daydream/runs/test/trajectories')
                async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                    observe_text_and_result(invocation, 'public activity')
            assert child.path.is_file()


async def test_fork_write_failure_degrades(tmp_path: Path) -> None:
    """If child _write() raises, parent ContextVar is restored, no crash."""
    recorder = make_recorder(tmp_path)
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)
        with patch("daydream.ui.print_warning", fake_print_warning):
            async with recorder.fork("fail-child") as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    observe_text_and_result(inv)
                original_path = child.path
                # A file in the parent path forces mkdir failure even with a writable root.
                blocker = tmp_path / "not-a-dir"
                blocker.write_text("x")
                child.path = blocker / "child.json"

        assert get_current_recorder() is recorder
        child.path = original_path

        flush_active_signal_recorders()
        assert recorder.path.with_suffix(".json.partial").exists()
        assert not child.path.with_suffix(".json.partial").exists()

    assert any("Sibling trajectory write failed" in m for m in warnings_emitted)
    assert recorder.path.exists()


# write_partial tests (CLI-03, D-07 SIGINT partial flush)


@pytest.mark.parametrize('state', ['tool', ])
async def test_partial_writes_preserve_live_state_and_joined_steps(tmp_path: Path, state: str) -> None:
    root = make_recorder(tmp_path)
    async with root:
        if state == 'empty':
            root.write_partial()
            assert not partial_document_path(root.path).exists()
            return
        if state == 'child':
            async with root.invocation(phase=DaydreamPhase.REVIEW) as parent_invocation:
                observe_text_and_result(parent_invocation, 'parent before child')
        async with trajectory_module.maybe_fork(root if state == 'child' else None, 'child') as child:
            recorder = child if child is not None else root
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                invocation.observe_user_step(prompt='retained prompt')
                if state == 'tool':
                    invocation.observe(ToolStartEvent(id='pending', name='Read', input={'file_path': 'a.py'}))
                elif state == 'diagnostic':
                    invocation.observe(DiagnosticEvent(code='parser_gap', message='bounded', metadata={'count': 1}))
                else:
                    observe_text_and_result(invocation, 'retained response')
                recorder.write_partial()
                first = partial_document_path(recorder.path).read_bytes()
                if state == 'child':
                    parent = read_trajectory(partial_document_path(root.path))
                    assert 'parent before child' in json.dumps(parent)
                    reference = next(row for row in parent['extra']['subtrajectories']
                                     if row.get('trajectory_id') == recorder.trajectory_id)
                    assert reference['sibling_trajectory_ref'] == recorder.path.relative_to(
                        tmp_path / '.daydream').as_posix()
            recorder.write_partial()
            joined_bytes = partial_document_path(recorder.path).read_bytes()
            recorder.write_partial()
            assert partial_document_path(recorder.path).read_bytes() == joined_bytes
            joined = json.loads(joined_bytes)
        final_path = recorder.path
    final = read_trajectory(final_path)
    assert len(joined['steps']) == len(final['steps'])
    for document in (json.loads(first), joined, final):
        assert atif_validate(document, validate_images=False)
        if state == 'tool':
            observation, = _agent_steps(document)[0]['observation']['results']
            assert observation == {'content': INCOMPLETE_CONTENT,
                                   'extra': {'is_error': True, 'status': 'interrupted'}}
        elif state == 'diagnostic':
            diagnostics = [record for step in _agent_steps(document)
                           for record in step.get('extra', {}).get('backend_diagnostics', [])]
            assert diagnostics == [{'code': 'parser_gap', 'message': 'bounded', 'metadata': {'count': 1}}]
        else:
            assert sum('retained response' in step.get('message', '') for step in _agent_steps(document)) == 1


def _partial_path(recorder: TrajectoryRecorder) -> Path:
    return recorder.path.with_suffix(recorder.path.suffix + ".partial")


async def _hold_fork(
    parent: TrajectoryRecorder, descriptor: str, marker: str, entered: anyio.Event, release: anyio.Event,
    children: dict[str, TrajectoryRecorder],
) -> None:
    """Hold one public fork invocation open across a signal-flush barrier."""
    async with parent.fork(descriptor) as child:
        children[descriptor] = child
        async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
            assert get_current_recorder() is child
            observe_text_and_result(inv, marker)
            entered.set()
            await release.wait()


@pytest.mark.parametrize("failed_recorder", ["root", ])
async def test_signal_flush_prepare_failure_never_publishes_disconnected_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_recorder: str,
) -> None:
    snapshots: list[RunWriteSnapshot] = []
    warnings: list[str] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    monkeypatch.setattr('daydream.ui.print_warning', lambda _console, message: warnings.append(message))
    async with root:
        if failed_recorder == 'child':
            async with root.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                observe_text_and_result(invocation, 'root evidence')
        async with root.fork('active-child') as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                invocation.observe(TextEvent('valid previous evidence'))
                flush_active_signal_recorders()
                previous = snapshots[-1]
                saved = {document.path: document.json_bytes for document in previous.documents}
                target = root if failed_recorder == 'root' else child

                def fail(**kwargs: Any) -> Any:
                    raise RuntimeError('document preparation failed')

                invocation.observe(TextEvent(' later evidence'))
                with monkeypatch.context() as patcher:
                    patcher.setattr(target, '_prepare_document', fail)
                    flush_active_signal_recorders()
                assert snapshots == [previous] and len(warnings) == 1
                assert all(path.read_bytes() == body for path, body in saved.items())
                flush_active_signal_recorders()
                current = snapshots[-1]
                assert current.cutoff_at != previous.cutoff_at
                assert {document.trajectory_id for document in current.documents} == {
                    root.trajectory_id, child.trajectory_id}
                for document in current.documents:
                    payload = json.loads(document.json_bytes)
                    assert payload['extra']['snapshot_at'] == current.cutoff_at
                    assert atif_validate(payload, validate_images=False)
                assert all(document.path.read_bytes() == document.json_bytes for document in current.documents)
                assert {document.path: document.json_bytes for document in previous.documents} == saved


@pytest.mark.parametrize("exit_kind", ["system-exit"])
async def test_signal_flush_excludes_exited_child(recorder: TrajectoryRecorder, exit_kind: str) -> None:
    """Every child exit shape unregisters before a later root-only flush."""

    async def child_body() -> TrajectoryRecorder:
        async with recorder.fork(f"exited-{exit_kind}") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                observe_text_and_result(inv, "EXITED_CHILD_ONLY")
                if exit_kind == "runtime":
                    raise RuntimeError("child body")
                if exit_kind == "system-exit":
                    raise SystemExit(17)
            return child

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")

        if exit_kind == "runtime":
            with pytest.raises(RuntimeError, match="child body"):
                await child_body()
        elif exit_kind == "system-exit":
            with pytest.raises(SystemExit) as exc:
                await child_body()
            assert exc.value.code == 17
        elif exit_kind == "cancel":
            with anyio.CancelScope() as scope:
                async with recorder.fork("exited-cancel") as active_child:
                    async with active_child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                        observe_text_and_result(inv, "EXITED_CHILD_ONLY")
                        scope.cancel()
                        await anyio.sleep_forever()
        else:
            await child_body()

        flush_active_signal_recorders()
        assert _partial_path(recorder).exists()
        child_path = recorder._sibling_path_for(f"exited-{exit_kind}")
        assert not child_path.with_suffix(child_path.suffix + ".partial").exists()

async def test_signal_flush_excludes_child_after_final_write_system_exit(
    recorder: TrajectoryRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BaseException from the child final write cannot leak registry membership."""

    child: TrajectoryRecorder
    child_path: Path

    def fail_final_write() -> None:
        raise SystemExit(23)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")
        with pytest.raises(SystemExit) as exc:
            async with recorder.fork("final-system-exit") as child:
                child_path = child.path
                async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                    observe_text_and_result(inv, "STALE_CHILD_ONLY")
                monkeypatch.setattr(child, "_write", fail_final_write)
        assert exc.value.code == 23
        child.path = child_path
        flush_active_signal_recorders()
        assert _partial_path(recorder).exists()
        assert not _partial_path(child).exists()

def _finish_shutdown_panel() -> None:

    panel = get_shutdown_panel()
    if panel is not None:
        panel.finish()
        set_shutdown_panel(None)

@pytest.mark.parametrize('signum', [signal.SIGTERM])
@pytest.mark.parametrize('denied', [True])
async def test_signal_handler_flushes_siblings_despite_a_writer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signum: signal.Signals, denied: bool,
) -> None:
    snapshots: list[RunWriteSnapshot] = []
    recorder = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    entered = {name: anyio.Event() for name in ('signal-a', 'signal-b')}
    release = {name: anyio.Event() for name in entered}
    children: dict[str, TrajectoryRecorder] = {}
    writes = Path.write_text
    warnings: list[str] = []
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, 'ROOT_ONLY')
        async with anyio.create_task_group() as group:
            for name in entered:
                group.start_soon(_hold_fork, recorder, name, name, entered[name], release[name], children)
                await entered[name].wait()
            failed_path = partial_document_path(children['signal-a'].path)

            def write(path: Path, *args: Any, **kwargs: Any) -> int:
                if denied and path == failed_path:
                    raise PermissionError('denied sibling')
                return writes(path, *args, **kwargs)

            monkeypatch.setattr(Path, 'write_text', write)
            monkeypatch.setattr('daydream.ui.print_warning', lambda _console, message: warnings.append(message))
            try:
                with pytest.raises(KeyboardInterrupt):
                    _signal_handler(signum, None)
                assert len(snapshots) == 1 and snapshots[0].status == 'partial'
                for child in (recorder, *children.values()):
                    path = partial_document_path(child.path)
                    assert path.exists() == (not denied or child is not children['signal-a'])
                    if path.exists():
                        assert atif_validate(read_trajectory(path), validate_images=False)
                assert len(warnings) == int(denied)
            finally:
                _finish_shutdown_panel()
                for event in release.values():
                    event.set()


@pytest.mark.parametrize('failure', ['final'])
async def test_document_writer_failure_preserves_capture_and_primary(tmp_path: Path, failure: str) -> None:
    snapshots: list[RunWriteSnapshot] = []

    def writer(document: TrajectoryDocumentSnapshot, status: str) -> None:
        if status == failure or failure == 'final':
            raise OSError('secondary writer failure')

    recorder = make_recorder(tmp_path, document_writer=writer, explicit_path=failure == 'final',
        on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    primary = RuntimeError('authoritative body failure')
    propagated: RuntimeError | None = None
    try:
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                observe_text_and_result(invocation, 'retained evidence')
            if failure == 'final':
                raise primary
            recorder.write_partial()
    except RuntimeError as caught:
        propagated = caught
    assert propagated is (primary if failure == 'final' else None)
    if failure == 'final':
        assert any('trajectory finalization' in note for note in primary.__notes__)
    if failure == 'partial':
        assert [snapshot.status for snapshot in snapshots] == ['partial', 'complete']
        assert json.loads(snapshots[0].documents[0].json_bytes)['extra']['partial'] is True
