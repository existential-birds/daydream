"""Recorder, invocation, and redactor tests use schema validity plus behavior predicates."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import pytest

import daydream.trajectory as trajectory_module
from daydream import git_ops
from daydream.atif import validate as atif_validate
from daydream.atif.models import Step
from daydream.backends import (
    AgentEvent,
    CostEvent,
    DiagnosticEvent,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.cli import _signal_handler
from daydream.eval.analyzer import analyze_costs
from daydream.json_utils import atomic_write_bytes
from daydream.phases.publish import (
    _do_commit,
)
from daydream.run_config import RunConfig
from daydream.trajectory import (
    RUN_DOCUMENT_NAME,
    DaydreamPhase,
    DaydreamRunFlow,
    Invocation,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
    flush_active_signal_recorders,
    get_current_recorder,
    host_phase_scope,
    partial_document_path,
    snapshot_trajectories,
)
from daydream.ui import get_shutdown_panel, set_shutdown_panel
from tests.harness.backend import ScriptedBackend
from tests.harness.trajectory import (
    make_recorder,
    observe_metrics_and_result,
    observe_text_and_result,
    read_trajectory,
)


def _agent_steps(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """The trajectory's agent-sourced steps, in order."""
    return [s for s in traj["steps"] if s["source"] == "agent"]


def only_dispatch(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Return the sole identified deterministic dispatch step."""
    dispatches = [step
        for step in trajectory["steps"]
        if step.get("llm_call_count") == 0 and isinstance(step.get("extra"), dict) and "dispatch_id" in step["extra"]
    ]
    assert len(dispatches) == 1
    dispatch = dispatches[0]
    assert isinstance(dispatch, dict)
    return dispatch


def dispatch_refs(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a dispatch's ordered ATIF sibling references."""
    observation = step.get("observation")
    assert isinstance(observation, dict)
    refs: list[dict[str, Any]] = []
    for result in observation["results"]:
        refs.extend(result.get("subagent_trajectory_ref", []))
    return refs


async def test_diagnostic_is_json_safe_redacted_and_persisted_in_arrival_order(tmp_path: Path,) -> None:
    """Diagnostics cross one fail-closed recorder privacy/type boundary."""
    secret = "sk-diagnostic123456"
    unsupported = object()
    _, steps = await _record_events(tmp_path,
        DiagnosticEvent(code=secret, message=f"API_KEY={secret}",
            metadata={"nested": {secret: f"TOKEN={secret}", "api_key": secret}, "unsupported": unsupported,
                "unsupported_key": {42: "retained safely"}, "non_finite": float("nan"),
            },
        ), DiagnosticEvent(code="second", message="safe", metadata={"count": 2}),
        ResultEvent(structured_output=None, continuation=None),
    )

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


@pytest.mark.parametrize('closed,result,aborted', [(False, None, False), (True, None, False),
    (True, False, False), (True, True, False), (False, None, True)])
async def test_tool_completion_and_abort_preserve_the_host_step(
    tmp_path: Path, closed: bool, result: bool | None, aborted: bool,
) -> None:
    """Late results amend their original step; only unfinished calls receive interruption markers."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.FIX) as invocation:
            invocation.observe(ToolStartEvent(id='tool', name='shell', input={'command': 'sleep 999'}))
            if closed:
                invocation.observe(TurnEndEvent(message_id='host'))
            if result is not None:
                invocation.observe(ToolResultEvent(id='tool', output='completed', is_error=result,
                                                  exit_code=int(result), status='completed'))
            if aborted:
                invocation.mark_aborted('budget_exceeded')
            invocation.observe(ResultEvent(structured_output=None, continuation=None))
    document = read_trajectory(recorder.path)
    assert atif_validate(document, validate_images=False)
    step, = _agent_steps(document)
    observation, = step['observation']['results']
    assert observation['content'] == (INCOMPLETE_CONTENT if result is None else 'completed')
    assert observation['extra'] == ({'is_error': True, 'status': 'interrupted'} if result is None else
                                   {'is_error': result, 'exit_code': int(result), 'status': 'completed'})
    assert observation.get('source_call_id') == (None if result is None else 'tool')
    if aborted:
        assert step['extra']['stop_reason'] == 'budget_exceeded'


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


@pytest.mark.parametrize('stage', ['partial', 'final'])
async def test_filesystem_writer_failure_is_optional(
    recorder: TrajectoryRecorder, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    """Actual partial/final filesystem failures warn without replacing a healthy invocation."""
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


async def test_dispatch_repeated_descriptor_scopes_keep_distinct_children(recorder: TrajectoryRecorder,) -> None:
    """Repeated semantic labels cannot overwrite or migrate across dispatches."""
    assert hasattr(trajectory_module, "dispatch_scope")
    child_paths: list[Path] = []
    async with recorder:
        for output in ("first", "second"):
            async with trajectory_module.dispatch_scope(recorder, phase=DaydreamPhase.FIX, descriptors=("repeat",),
            ) as dispatch:
                assert dispatch is not None
                async with trajectory_module.maybe_fork(recorder, "repeat", dispatch=dispatch,) as child:
                    child_paths.append(child.path)
                    async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                        observe_text_and_result(inv, output)

    root = read_trajectory(recorder.path)
    dispatches = [
        step for step in root["steps"] if step.get("llm_call_count") == 0 and "dispatch_id" in step.get("extra", {})
    ]
    assert len(dispatches) == 2
    assert len(set(child_paths)) == 2
    first_ref, second_ref = (dispatch_refs(step)[0] for step in dispatches)
    assert first_ref["trajectory_path"] != second_ref["trajectory_path"]
    assert first_ref["trajectory_id"] != second_ref["trajectory_id"]
    assert [step["extra"]["dispatch_id"] for step in dispatches] == [
        f"{recorder.session_id}:dispatch:1", f"{recorder.session_id}:dispatch:2",
    ]
    assert "first" in json.dumps(read_trajectory(child_paths[0]))
    assert "second" in json.dumps(read_trajectory(child_paths[1]))

@pytest.mark.parametrize('ending', ['optimistic-cancel', 'empty-cancel', 'empty-error'])
async def test_dispatch_body_failure_owns_terminal_status(recorder: TrajectoryRecorder, ending: str) -> None:
    """Cancellation/body failure outrank optimistic success and an empty child's secondary write failure."""
    async def dispatch_body() -> None:
        async with trajectory_module.dispatch_scope(recorder, phase=DaydreamPhase.FIX,
                                                     descriptors=('child',)) as dispatch:
            assert dispatch is not None
            if ending == 'optimistic-cancel':
                dispatch.finish(trajectory_module.LifecycleStatus.SUCCEEDED)
            async with recorder.fork('child', dispatch=dispatch) as child:
                if ending == 'optimistic-cancel':
                    async with child.invocation(phase=DaydreamPhase.FIX) as invocation:
                        observe_text_and_result(invocation, 'before cancellation')
            if ending == 'empty-error':
                raise RuntimeError('dispatch body failed')
            scope.cancel()
            await anyio.sleep_forever()

    async with recorder:
        if ending == 'empty-error':
            with pytest.raises(RuntimeError, match='dispatch body failed'):
                await dispatch_body()
        else:
            with anyio.CancelScope() as scope:
                await dispatch_body()
    status = only_dispatch(read_trajectory(recorder.path))['extra']
    assert status['dispatch_status'] == ('failed' if ending == 'empty-error' else 'cancelled')
    assert status['reason_code'] == ('uncaught_exception' if ending == 'empty-error' else 'cancelled')
    if ending != 'optimistic-cancel':
        assert status['attempted_count'] == 1 and status['completed_count'] == 0


_GRAPH_LABELS = [
    ('ordinary-python', 'ordinary-python'),
    ('{"scope":"python"}', '{"scope": "python"}'),
    ('{"scope":"python","scope":"rust"}', '{"scope": "rust"}'),
    ('auth=Bearer [REDACTED][REDACTED_CREDENTIAL]', 'auth=Bearer [REDACTED][REDACTED_CREDENTIAL]'),
    ('api_key=[REDACTED][REDACTED_CREDENTIAL]', 'api_key=[REDACTED][REDACTED_CREDENTIAL]'),
    ('TOKEN=[REDACTED_ENV_VAR]; public diagnostics', 'TOKEN=[REDACTED_ENV_VAR]; public diagnostics'),
    ('{"type":"api_key","key":"[REDACTED]"}', '{"type": "api_key", "key": "[REDACTED]"}'),
    ('{"type":"api_key","key":"opaque-first-canary"}', None),
]


@pytest.mark.parametrize(('label', 'planned', 'identified'), [
    *[(*case, True) for case in _GRAPH_LABELS],
    ('ordinary-python', 'ordinary-python', False),
    ('{"type":"api_key","key":"opaque-first-canary"}', None, False),
])
async def test_partial_graph_preserves_empty_ancestry_and_original_private_identities(
    tmp_path: Path, label: str, planned: str | None, identified: bool,
) -> None:
    """Live empty ancestry and private labels retain one immutable rooted graph before any finalizer."""
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


async def test_dispatch_step_noop_when_no_siblings(recorder: TrajectoryRecorder) -> None:
    """An empty declared fan-out adds no dispatch step."""
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)
        steps_before = len(recorder.steps)
        async with trajectory_module.dispatch_scope(recorder, phase=DaydreamPhase.FIX, descriptors=[]) as dispatch:
            assert dispatch is None
        assert len(recorder.steps) == steps_before


@pytest.mark.parametrize('label', ['Fix_Issue (3)', 'UPPER--CASE', '../etc/passwd', '', '...', '   '])
async def test_fork_paths_confine_public_descriptors(tmp_path: Path, label: str) -> None:
    """The public fork interface confines labels to owned paths and refuses unusable filenames."""
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


@pytest.mark.parametrize('state', ['empty', 'tool', 'diagnostic', 'completed', 'child'])
async def test_partial_writes_preserve_live_state_and_joined_steps(tmp_path: Path, state: str) -> None:
    """Partial files are stable projections; flushing cannot consume live events or duplicate joined steps."""
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


async def test_partial_dispatch_retains_planned_queued_scopes_before_fanout_joins(tmp_path: Path) -> None:
    """A first-scope checkpoint retains queued plans without declaring them attempted or complete."""
    snapshots: list[RunWriteSnapshot] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    limiter = anyio.CapacityLimiter(1)
    entered, queued, release = anyio.Event(), anyio.Event(), anyio.Event()
    async with root:
        async with trajectory_module.dispatch_scope(
            root, phase=DaydreamPhase.DEEP, descriptors=['first', 'queued'],
        ) as dispatch:
            assert dispatch is not None

            async def review(descriptor: str) -> None:
                if descriptor == 'queued':
                    queued.set()
                async with limiter:
                    async with root.fork(descriptor, dispatch=dispatch) as child:
                        async with child.invocation(phase=DaydreamPhase.DEEP) as invocation:
                            invocation.observe(TextEvent('scope activity: ' + descriptor))
                            if descriptor == 'first':
                                entered.set()
                            await release.wait()

            async with anyio.create_task_group() as group:
                group.start_soon(review, 'first')
                await entered.wait()
                group.start_soon(review, 'queued')
                await queued.wait()
                try:
                    root.write_partial()
                    snapshot = snapshots[-1]
                    parent = json.loads(snapshot.documents[0].json_bytes)
                    active = parent['extra']['active_dispatches']
                    assert len(active) == 1
                    plan = active[0]
                    assert plan['dispatch_id'] == dispatch.dispatch_id and plan['phase'] == 'deep'
                    assert plan['planned_descriptors'] == ['first', 'queued']
                    assert plan['queued_descriptors'] == ['queued']
                    assert plan['planned_count'] == 2 and plan['attempted_count'] == 1
                    assert plan['completed_count'] == 0 and plan['completed_fork_ids'] == []
                    assert plan['ended_at'] is None and 'dispatch_status' not in plan
                    assert plan['forks'] == [{'fork_id': snapshot.documents[1].trajectory_id,
                                              'descriptor': 'first', 'ordinal': 1}]
                    assert len(snapshot.documents) == 2
                    assert root.steps == []
                finally:
                    release.set()
    final = read_trajectory(root.path)
    assert 'active_dispatches' not in final['extra']
    dispatch_step = only_dispatch(final)
    assert dispatch_step['extra']['planned_count'] == dispatch_step['extra']['completed_count'] == 2


@pytest.mark.parametrize("failed_recorder", ["root", "child"])
async def test_signal_flush_prepare_failure_never_publishes_disconnected_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_recorder: str,
) -> None:
    """A preparation failure preserves a previously valid graph; retry publishes coherent later bytes."""
    snapshots: list[RunWriteSnapshot] = []
    warnings: list[str] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    monkeypatch.setattr('daydream.ui.print_warning', lambda _console, message: warnings.append(message))
    async with root:
        async with root.fork('active-child') as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                invocation.observe(TextEvent('valid previous evidence'))
                root.write_partial()
                previous = snapshots[-1]
                saved = {document.path: document.json_bytes for document in previous.documents}
                target = root if failed_recorder == 'root' else child

                def fail(**kwargs: Any) -> Any:
                    raise RuntimeError('document preparation failed')

                invocation.observe(TextEvent(' later evidence'))
                with monkeypatch.context() as patcher:
                    patcher.setattr(target, '_prepare_document', fail)
                    root.write_partial()
                assert snapshots == [previous] and len(warnings) == 1
                assert all(path.read_bytes() == body for path, body in saved.items())
                root.write_partial()
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


@pytest.mark.parametrize("exit_kind", ["normal", "runtime", "cancel", "system-exit"])
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

async def test_signal_flush_selects_latest_independent_root(tmp_path: Path) -> None:
    """A nested independent root is targeted until it exits, then outer resumes."""

    writes: list[tuple[str, str]] = []
    outer = make_recorder(tmp_path / "outer", on_write=lambda _rec, snapshot: writes.append(("outer", snapshot.status)),
    )
    inner = make_recorder(tmp_path / "inner", on_write=lambda _rec, snapshot: writes.append(("inner", snapshot.status)),
    )
    async with outer:
        async with outer.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "OUTER_ONLY")
        async with inner:
            async with inner.invocation(phase=DaydreamPhase.REVIEW) as inv:
                observe_text_and_result(inv, "INNER_ONLY")
            flush_active_signal_recorders()
            assert writes == [("inner", "partial")]
        writes.clear()
        flush_active_signal_recorders()
        assert writes == [("outer", "partial")]


def _finish_shutdown_panel() -> None:

    panel = get_shutdown_panel()
    if panel is not None:
        panel.finish()
        set_shutdown_panel(None)

@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize('denied', [False, True])
async def test_signal_handler_flushes_siblings_despite_a_writer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signum: signal.Signals, denied: bool,
) -> None:
    """The real handler captures once, writes healthy siblings and isolates an actual filesystem failure."""
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


async def test_recorder_marks_partial_on_exception_exit(recorder: TrajectoryRecorder) -> None:
    """When __aexit__ receives an exception, the trajectory is marked partial."""
    with pytest.raises(RuntimeError, match="boom"):
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe_user_step(prompt="hello")
                inv.observe(TextEvent(text="partial output"))
                raise RuntimeError("boom")

    assert recorder.path.exists()
    traj = read_trajectory(recorder.path)
    assert traj.get("extra", {}).get("partial") is True


async def test_nested_fork_billing_reaches_root_once(tmp_path: Path) -> None:
    """Root totals include each nested owner once; empty forks and diagnostic projections add no billing."""
    snapshots: list[RunWriteSnapshot] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            observe_metrics_and_result(invocation, 'root', message_id='root', prompt_tokens=100,
                                       completion_tokens=20, cached_tokens=10, cost_usd=1.0)
        async with root.fork('empty') as empty:
            pass
        async with root.fork('outer') as outer:
            async with outer.invocation(phase=DaydreamPhase.DEEP) as invocation:
                observe_metrics_and_result(invocation, 'outer', message_id='outer', prompt_tokens=20,
                                           completion_tokens=4, cached_tokens=2, cost_usd=0.2)
            async with outer.fork('inner') as inner:
                async with inner.invocation(phase=DaydreamPhase.DEEP) as invocation:
                    observe_metrics_and_result(invocation, 'inner', message_id='inner', prompt_tokens=20,
                                               completion_tokens=4, cached_tokens=2, cost_usd=0.3)
        root.write_partial()
    assert not empty.path.exists()
    final = read_trajectory(root.path)
    assert not any(row.get('descriptor') == 'empty' for row in final['extra']['subtrajectories'])
    assert final['final_metrics']['total_prompt_tokens'] == 140
    assert final['final_metrics']['total_cost_usd'] == pytest.approx(1.5)
    costs = analyze_costs(snapshot_trajectories(snapshots[-1]))
    assert costs['total_cost_usd'] == pytest.approx(1.5)
    assert costs['total_prompt_tokens_raw'] == 140 and costs['total_completion_tokens'] == 28
    assert len(costs['by_agent']) == 3
    assert sum(row['cost_usd'] for row in costs['by_agent']) == pytest.approx(1.5)


@pytest.mark.parametrize('backend', ['', 'codex'])
async def test_trajectory_backend_metadata_omits_unavailable_phases(tmp_path: Path, backend: str) -> None:
    """Only resolved nonempty phase identities enter persisted metadata, including Improve."""
    recorder = make_recorder(tmp_path, run_flow=DaydreamRunFlow.IMPROVE, backend_name=backend,
                             review_backend_name=backend, fix_backend_name='', test_backend_name='')
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            observe_text_and_result(invocation, 'public response')
    extra = read_trajectory(recorder.path)['extra']
    assert {key: extra[key] for key in ('backend', 'review_backend', 'fix_backend', 'test_backend')
            if key in extra} == ({'backend': backend, 'review_backend': backend} if backend else {})


async def test_host_phase_scope_noop_without_recorder() -> None:

    async with host_phase_scope(DaydreamPhase.COMMIT):
        pass  # must not raise when no recorder is active

@pytest.mark.parametrize(("stop_reason", "expected_status", "expected_reason"),
    [("completed", "succeeded", None), ("passed", "succeeded", None), ("no_ci", "succeeded", None),
        ("timed_out", "timed_out", "timed_out"), ("cancelled", "cancelled", "cancelled"),
        ("interrupted", "cancelled", "cancelled"), ("failed", "failed", "domain_failure"),
        ("missing", "failed", "domain_failure"), ("unavailable", "failed", "domain_failure"),
        ("superseded", "failed", "domain_failure"), ("pending", "failed", "domain_failure"),
    ],
)
async def test_remote_ci_host_phases_record_exact_terminal_reasons(
    recorder: TrajectoryRecorder, stop_reason: str, expected_status: str, expected_reason: str | None,
) -> None:
    """Every admitted remote-CI reason has one closed lifecycle projection."""
    recorder.document_writer = lambda document, _status: atomic_write_bytes(document.path, document.json_bytes)
    async with recorder:
        async with host_phase_scope(DaydreamPhase.REMOTE_CI) as phase:
            await anyio.sleep(0)
            phase.stop_reason = stop_reason

    remote_events = [event
        for event in read_trajectory(recorder.path)["extra"]["phase_events"]
        if event["phase"] == DaydreamPhase.REMOTE_CI.value
    ]
    assert [event["event"] for event in remote_events] == ["phase_start", "phase_end"]
    ends = [event for event in remote_events if event["event"] == "phase_end"]
    assert len(ends) == 1
    assert ends[0]["status"] == expected_status
    assert ends[0].get("reason_code") == expected_reason
    assert ends[0]["metadata"]["stop_reason"] == stop_reason
    assert all(event["metadata"]["duration_ms"] >= 0 for event in ends)


@pytest.mark.parametrize('state', ['unbound', 'empty', 'runtime', 'cancel', 'child', 'root-and-child'])
async def test_host_document_publication(tmp_path: Path, state: str) -> None:
    """The host writer owns ordered bytes and one rooted capture, including empty/aborted runs."""
    snapshots: list[RunWriteSnapshot] = []
    writes: list[TrajectoryDocumentSnapshot] = []
    order: list[str] = []
    root_path = tmp_path / 'private/runs/test/trajectory.json'

    def writer(document: TrajectoryDocumentSnapshot, status: str) -> None:
        assert status == 'complete'
        order.append(document.trajectory_id)
        writes.append(document)
        if state != 'root-and-child':
            document.path.parent.mkdir(parents=True, exist_ok=True)
            document.path.write_bytes(document.json_bytes)

    def capture(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot) -> None:
        assert get_current_recorder() is recorder
        order.append('callback')
        snapshots.append(snapshot)

    recorder = make_recorder(tmp_path, path=root_path, agent_model_name='', on_write=capture,
        artifact_run_dir=None if state == 'unbound' else root_path.parent,
        document_writer=None if state == 'unbound' else writer)
    primary = RuntimeError('primary body failure') if state == 'runtime' else anyio.get_cancelled_exc_class()()
    caught_error: BaseException | None = None
    try:
        async with recorder:
            async with trajectory_module.phase_scope(DaydreamPhase.MERGE):
                if state in {'runtime', 'cancel'}:
                    raise primary
            if state in {'child', 'root-and-child'}:
                async with recorder.fork('completed-child') as child:
                    async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                        observe_text_and_result(invocation, 'child evidence')
                if state == 'root-and-child':
                    async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                        observe_text_and_result(invocation, 'root evidence')
    except BaseException as caught:
        caught_error = caught
    assert caught_error is (primary if state in {'runtime', 'cancel'} else None)
    assert get_current_recorder() is None
    if state == 'unbound':
        assert not root_path.exists() and not snapshots and not writes
        return
    snapshot, = snapshots
    assert snapshot.status == 'complete' and snapshot.root_trajectory_id == recorder.trajectory_id
    expected_ids = ([child.trajectory_id] if state in {'child', 'root-and-child'} else []) + [recorder.trajectory_id]
    assert [document.trajectory_id for document in writes] == expected_ids
    assert order == [*expected_ids, 'callback']
    assert len(writes) == len(snapshot.documents) == len(expected_ids)
    assert [document.trajectory_id for document in snapshot.documents] == list(reversed(expected_ids))
    for document in snapshot.documents:
        payload = json.loads(document.json_bytes)
        assert atif_validate(payload, validate_images=False)
        if state == 'root-and-child':
            assert not document.path.exists()
        else:
            assert document.path.read_bytes() == document.json_bytes
    payload = json.loads(snapshot.documents[0].json_bytes)
    assert payload['extra']['run_ended_at'] == snapshot.cutoff_at
    if state != 'root-and-child':
        assert len(payload['steps']) == 1 and payload['steps'][0]['source'] == 'system'
        assert payload['steps'][0]['extra']['host_event'] == 'host_only_final_snapshot'
        assert payload['steps'][0]['timestamp'] == snapshot.cutoff_at
        expected_metrics: dict[str, Any] = {'total_steps': 1}
        if state == 'child':
            expected_metrics['extra'] = {'daydream_metric_scope': 'whole_run_including_forks',
                                         'total_steps_scope': 'local_trajectory'}
        assert payload['final_metrics'] == expected_metrics
        assert not payload['agent'].get('model_name')
    if state in {'runtime', 'cancel'}:
        assert payload['extra']['partial'] is True
    expected = 'failed' if state == 'runtime' else 'cancelled' if state == 'cancel' else 'succeeded'
    assert payload['extra']['phase_events'][-1]['status'] == expected
    if state in {'child', 'root-and-child'}:
        summaries = payload['extra']['subtrajectories']
        assert len(summaries) == (2 if state == 'root-and-child' else 1)
        reference, = (summary for summary in summaries if 'sibling_trajectory_ref' in summary)
        assert reference['trajectory_id'] == snapshot.documents[1].trajectory_id
        assert 'invocation_id' not in reference
        assert writes[0].path.parent == root_path.parent / 'trajectories'


@pytest.mark.parametrize('failure', ['partial', 'final'])
async def test_document_writer_failure_preserves_capture_and_primary(tmp_path: Path, failure: str) -> None:
    """A partial writer error retains frozen capture; final failure cannot replace an active body error."""
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


async def test_do_commit_records_commit_phase_event(git_repo: Path, make_work: Any,) -> None:
    """Real-path: _do_commit's host-native commit emits a distinct ``commit``
    phase event with duration_ms + stop_reason (issue #726 task 12)."""

    work = make_work(git_repo)
    (git_repo / "app.py").write_text("x = 0\n")
    _git_add_commit(git_repo)
    (git_repo / "app.py").write_text("x = 1\n")

    rec = make_recorder(git_repo,
        document_writer=lambda document, _status: atomic_write_bytes(document.path, document.json_bytes))
    async with rec:
        ok = await _do_commit(
            ScriptedBackend(), work, push=False, config=RunConfig(test_command="true"),
            retained_paths=frozenset({"app.py"}),
            retained_states=git_ops.snapshot_worktree_paths(git_repo, {"app.py"}),
            initial_index=git_ops.snapshot_index(git_repo),
        )
    assert ok.committed is True
    assert ok.push is None

    commit_ends = [event for event in read_trajectory(rec.path)["extra"]["phase_events"]
                   if event["phase"] == "commit" and event["event"] == "phase_end"]
    assert len(commit_ends) == 1
    assert commit_ends[0]["metadata"]["stop_reason"] == "completed"
    assert commit_ends[0]["metadata"]["duration_ms"] >= 0


def _git_add_commit(repo: Path) -> None:

    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "base"], cwd=repo, check=True,)

async def test_dispatch_registers_late_dynamic_fork_before_scope_exit(recorder: TrajectoryRecorder) -> None:
    async with recorder:
        async with trajectory_module.dispatch_scope(
            recorder, phase=DaydreamPhase.DEEP, descriptors=("deep-python", "deep-react"),
        ) as dispatch:
            assert dispatch is not None
            for descriptor in ("deep-python", "deep-react", "deep-structure"):
                async with trajectory_module.maybe_fork(recorder, descriptor, dispatch=dispatch) as child:
                    async with child.invocation(phase=DaydreamPhase.DEEP) as inv:
                        observe_text_and_result(inv, descriptor)
    step = only_dispatch(read_trajectory(recorder.path))
    assert step["extra"]["planned_count"] == 3
    assert step["extra"]["attempted_count"] == 3
    assert step["extra"]["completed_count"] == 3
    assert step["extra"]["dispatch_status"] == "succeeded"
    assert [result["content"] for result in step["observation"]["results"]] == [
        "Dispatched to deep-python", "Dispatched to deep-react", "Dispatched to deep-structure",
    ]
