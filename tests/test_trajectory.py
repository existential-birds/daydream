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
from daydream.atif import Step as AtifStep, validate as atif_validate
from daydream.atif.models import Step
from daydream.backends import (
    AgentEvent,
    DiagnosticEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.cli import _signal_handler
from daydream.eval.analyzer import analyze_costs
from daydream.phases.publish import (
    _do_commit,
)
from daydream.trajectory import (
    PARTIAL_SUFFIX,
    RUN_DOCUMENT_NAME,
    DaydreamPhase,
    DaydreamRunFlow,
    Invocation,
    Redactor,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
    flush_active_signal_recorders,
    get_current_recorder,
    host_phase_scope,
    now_iso,
    partial_document_path,
    redact_text,
    run_directory,
    run_document_path,
    sibling_document_path,
    snapshot_trajectories,
)
from daydream.trajectory.recorder import _safe_descriptor
from daydream.ui import get_shutdown_panel, set_shutdown_panel
from tests.harness.backend import ScriptedBackend
from tests.harness.trajectory import (
    make_recorder,
    observe_metrics_and_result,
    observe_text_and_result,
    read_trajectory,
    trajectory_payload,
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


def _single_observation_result(steps: list[Step], index: int) -> Any:
    """The single ObservationResult of *steps[index]*, asserting the shape."""
    step = steps[index]
    assert step.observation is not None
    assert len(step.observation.results) == 1
    return step.observation.results[0]


async def test_late_result_on_closed_step_carries_extra(tmp_path: Path) -> None:
    """The closed-step amendment path persists extra too, not just content."""
    inv, steps = await _record_events(
        tmp_path, ToolStartEvent(id="c4", name="shell", input={}), TurnEndEvent(message_id="m1"),
        ToolResultEvent(id="c4", output="late", is_error=True, exit_code=1, status="completed"),
        ResultEvent(structured_output=None, continuation=None),
    )
    result = _single_observation_result(steps, 0)
    assert result.extra == {"is_error": True, "exit_code": 1, "status": "completed"}


INCOMPLETE_CONTENT = "[interrupted: call did not complete before invocation ended]"

@pytest.mark.parametrize("host_closed", [False, True], ids=["open-step", "closed-step"])
async def test_finish_marks_in_flight_tool_on_its_host_step(tmp_path: Path, host_closed: bool) -> None:
    """Open and closed host steps both receive an incomplete-call marker."""
    events: list[AgentEvent] = [ToolStartEvent(
            id="d2" if host_closed else "d1", name="shell", input={} if host_closed else {"command": "sleep 999"},
        )
    ]
    if host_closed:
        events.append(TurnEndEvent(message_id="m1"))
    events.append(ResultEvent(structured_output=None, continuation=None))
    _, steps = await _record_events(tmp_path, *events)
    result = _single_observation_result(steps, 0)
    # The marker is distinct from a completed tool-call result.
    assert result.source_call_id is None
    assert result.content == INCOMPLETE_CONTENT
    assert result.extra == {"is_error": True, "status": "interrupted"}

async def test_late_result_before_finish_still_amends_normally(tmp_path: Path) -> None:
    """A result arriving before the final flush amends its step and suppresses the marker."""
    inv, steps = await _record_events(
        tmp_path, ToolStartEvent(id="d3", name="shell", input={}), TurnEndEvent(message_id="m1"),
        ToolResultEvent(id="d3", output="made", is_error=False, exit_code=0, status="completed"),
        ResultEvent(structured_output=None, continuation=None),
    )
    result = _single_observation_result(steps, 0)
    assert result.content == "made"
    assert result.extra == {"is_error": False, "exit_code": 0, "status": "completed"}


async def test_mark_aborted_stamps_stop_reason_on_closing_step(recorder: TrajectoryRecorder) -> None:
    """An aborted invocation closes cleanly with extra['stop_reason'] set."""
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
            inv.observe(ToolStartEvent(id="tool-1", name="Bash", input={"command": "ls"}))
            inv.mark_aborted("budget_exceeded")

    traj = read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = _agent_steps(traj)
    assert len(agent_steps) == 1
    assert agent_steps[0]["extra"]["stop_reason"] == "budget_exceeded"


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


async def test_write_failure_degrades_with_warning(recorder: TrajectoryRecorder, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavior C: PermissionError on write emits warning; run does not raise."""
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    monkeypatch.setattr("daydream.ui.print_warning", fake_print_warning)
    monkeypatch.setattr("os.replace", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),)

    # Exit MUST NOT raise even though _write fails inside __aexit__
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "hi")

    assert any("Trajectory write failed" in m for m in warnings_emitted)


async def test_context_var_set_inside_and_cleared_after(recorder: TrajectoryRecorder) -> None:
    """Behavior E: get_current_recorder is the recorder inside, None after."""
    assert get_current_recorder() is None
    async with recorder:
        assert get_current_recorder() is recorder
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "hi")
    assert get_current_recorder() is None



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

async def test_dispatch_cancellation_overrides_optimistic_terminal(recorder: TrajectoryRecorder,) -> None:
    """Cancellation propagates while closing the dispatch with a stable code."""
    assert hasattr(trajectory_module, "dispatch_scope")
    async with recorder:
        with anyio.CancelScope() as cancel_scope:
            async with trajectory_module.dispatch_scope(
                recorder, phase=DaydreamPhase.FIX, descriptors=("cancelled-child",),
            ) as dispatch:
                assert dispatch is not None
                dispatch.finish(trajectory_module.LifecycleStatus.SUCCEEDED)
                async with trajectory_module.maybe_fork(recorder, "cancelled-child", dispatch=dispatch,) as child:
                    async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                        observe_text_and_result(inv, "before cancellation")
                    cancel_scope.cancel()
                    await anyio.sleep_forever()

    step = only_dispatch(read_trajectory(recorder.path))
    assert step["extra"]["dispatch_status"] == "cancelled"
    assert step["extra"]["reason_code"] == "cancelled"

async def test_dispatch_cancellation_overrides_empty_child_write_failure(recorder: TrajectoryRecorder,) -> None:
    """Cancellation remains authoritative when an empty child cannot be written."""
    async with recorder:
        with anyio.CancelScope() as cancel_scope:
            async with trajectory_module.dispatch_scope(recorder, phase=DaydreamPhase.FIX, descriptors=("empty-child",),
            ) as dispatch:
                assert dispatch is not None
                async with trajectory_module.maybe_fork(recorder, "empty-child", dispatch=dispatch,):
                    pass
                cancel_scope.cancel()
                await anyio.sleep_forever()

    step = only_dispatch(read_trajectory(recorder.path))
    assert step["extra"]["dispatch_status"] == "cancelled"
    assert step["extra"]["reason_code"] == "cancelled"
    assert step["extra"]["attempted_count"] == 1
    assert step["extra"]["completed_count"] == 0

async def test_dispatch_exception_overrides_empty_child_write_failure(recorder: TrajectoryRecorder,) -> None:
    """An escaping exception retains its cause when an empty child also failed."""
    async with recorder:
        with pytest.raises(RuntimeError, match="dispatch body failed"):
            async with trajectory_module.dispatch_scope(recorder, phase=DaydreamPhase.FIX, descriptors=("empty-child",),
            ) as dispatch:
                assert dispatch is not None
                async with trajectory_module.maybe_fork(recorder, "empty-child", dispatch=dispatch,):
                    pass
                raise RuntimeError("dispatch body failed")

    step = only_dispatch(read_trajectory(recorder.path))
    assert step["extra"]["dispatch_status"] == "failed"
    assert step["extra"]["reason_code"] == "uncaught_exception"
    assert step["extra"]["attempted_count"] == 1
    assert step["extra"]["completed_count"] == 0


SESSION = "11111111-2222-3333-4444-555555555555"


def _replace_snapshot_root(snapshot: RunWriteSnapshot, root_path: Path) -> RunWriteSnapshot:
    documents = tuple(TrajectoryDocumentSnapshot(document.trajectory_id, root_path, document.json_bytes)
        if document.trajectory_id == snapshot.root_trajectory_id
        else document
        for document in snapshot.documents
    )
    return RunWriteSnapshot(
        status=snapshot.status, cutoff_at=snapshot.cutoff_at, root_trajectory_id=snapshot.root_trajectory_id,
        documents=documents,
    )

def test_producer_labels_and_partial_paths_come_from_the_layout_surface(tmp_path: Path) -> None:
    """`_source_file` names and the partial path are derived, not retyped."""
    run_dir = run_directory(tmp_path / ".daydream", SESSION)
    sibling = sibling_document_path(run_dir, "deep-python.json")
    snapshot = RunWriteSnapshot(status="complete", cutoff_at="2026-01-01T00:00:00Z", root_trajectory_id=SESSION,
        documents=(TrajectoryDocumentSnapshot(SESSION, run_document_path(run_dir), trajectory_payload(SESSION)),
            TrajectoryDocumentSnapshot("fork-1", sibling, trajectory_payload("fork-1")),
        ),
    )
    frozen = snapshot_trajectories(snapshot)
    assert frozen["main"]["_source_file"] == RUN_DOCUMENT_NAME
    assert [d["_source_file"] for d in frozen["forked"]] == ["deep-python.json"]

    partial = partial_document_path(run_document_path(run_dir))
    assert partial.name == f"{RUN_DOCUMENT_NAME}{PARTIAL_SUFFIX}"
    assert (snapshot_trajectories(_replace_snapshot_root(snapshot, partial))["main"]["_source_file"]
        == RUN_DOCUMENT_NAME
    )

def test_lifecycle_snapshot_value_types_are_frozen(tmp_path: Path) -> None:
    """Task 1 freezes the immutable payload types consumed by Task 5."""
    assert hasattr(trajectory_module, "TrajectoryDocumentSnapshot")
    document = trajectory_module.TrajectoryDocumentSnapshot(
        trajectory_id="root", path=tmp_path / "trajectory.json", json_bytes=b"{}",
    )
    snapshot = trajectory_module.RunWriteSnapshot(
        status="partial", cutoff_at="2026-01-01T00:00:00Z", root_trajectory_id="root", documents=(document,),
    )
    assert snapshot.documents == (document,)
    with pytest.raises(AttributeError):
        setattr(snapshot, "cutoff_at", "changed")


def test_redactor_is_passthrough() -> None:
    """Clean steps preserve semantic values; a fresh model copy is allowed."""

    step = AtifStep(step_id=1, timestamp=now_iso(), source="user", message="hello",)
    out = Redactor().redact_step(step)
    assert out.message == step.message
    assert out.reasoning_content == step.reasoning_content
    assert out.tool_calls == step.tool_calls
    assert out.observation == step.observation
    assert out.extra == step.extra

def test_empty_secret_assignment_does_not_consume_the_following_line() -> None:
    """An empty API_KEY= assignment must preserve the following newline and content."""
    block = (
        "CLERK_SECRET_KEY=\n"
        "CLOUDFLARE_ACCOUNT_ID=\n"
        "CLOUDFLARE_API_TOKEN=\n"
        "CLOUDFLARE_ACCOUNT_HASH=\n"
        "INTERNAL_SERVICE_SECRET="
    )

    assert redact_text(block) == block

def test_secret_value_on_the_same_line_is_still_redacted() -> None:
    assert redact_text("API_KEY= sk-live-abc123") == "API_KEY=[REDACTED_ENV_VAR]"
    assert redact_text("TOKEN=abc\nPLAIN_SETTING=1") == "TOKEN=[REDACTED_ENV_VAR]\nPLAIN_SETTING=1"

def test_public_text_redactor_is_fail_closed() -> None:
    secret = "OPENAI_API_KEY=sk-secret123456"

    assert redact_text(secret) == "OPENAI_API_KEY=[REDACTED_ENV_VAR]"

    class _FailingPattern:
        def sub(self, replacement: str, value: str) -> str:
            raise RuntimeError("synthetic redaction failure")

    with patch("daydream.redaction._REDACTION_RULES", ((_FailingPattern(), "[REDACTED]"),),):
        assert redact_text(secret) == "[REDACTION_FAILED]"


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


def test_safe_descriptor_slugification() -> None:
    """Various inputs correctly slugified by _safe_descriptor."""
    assert _safe_descriptor("fix-0") == "fix-0"
    assert _safe_descriptor("deep-python") == "deep-python"
    assert _safe_descriptor("explore-pattern-scanner") == "explore-pattern-scanner"
    assert _safe_descriptor("Fix_Issue (3)") == "fix-issue-3"
    assert _safe_descriptor("UPPER--CASE") == "upper-case"
    assert _safe_descriptor("-leading-trailing-") == "leading-trailing"
    assert _safe_descriptor("../etc/passwd") == "etc-passwd"

def test_safe_descriptor_rejects_degenerate_inputs() -> None:
    """Degenerate inputs that produce empty slugs raise ValueError (CR-01)."""
    with pytest.raises(ValueError, match="empty slug"):
        _safe_descriptor("")
    with pytest.raises(ValueError, match="empty slug"):
        _safe_descriptor("...")
    with pytest.raises(ValueError, match="empty slug"):
        _safe_descriptor("   ")


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


async def test_fork_child_no_steps_no_file(tmp_path: Path) -> None:
    """Pitfall 6: Child with 0 steps writes no sibling file; parent has no registration."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("empty-child"):
            pass
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)

    traj_dir = tmp_path / ".daydream" / "trajectories"
    assert not traj_dir.exists() or len(list(traj_dir.iterdir())) == 0
    root = read_trajectory(recorder.path)
    assert not any("sibling_trajectory_ref" in item for item in root["extra"]["subtrajectories"])


# write_partial tests (CLI-03, D-07 SIGINT partial flush)


async def test_write_partial_no_op_when_steps_empty(recorder: TrajectoryRecorder) -> None:
    """write_partial skips disk write when steps list is empty (matches _write)."""
    async with recorder:
        recorder.write_partial()
        partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
        assert not partial_path.exists()

async def test_write_partial_is_idempotent(recorder: TrajectoryRecorder) -> None:
    """Calling write_partial twice yields a single .partial file with latest contents."""
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "first")
        recorder.write_partial()
        partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
        first = partial_path.read_text(encoding="utf-8")
        recorder.write_partial()
        second = partial_path.read_text(encoding="utf-8")
        assert first == second

async def test_write_partial_failure_emits_warning_does_not_raise(
    recorder: TrajectoryRecorder, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disk-write failure during partial flush degrades with warning, never raises."""
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    monkeypatch.setattr("daydream.ui.print_warning", fake_print_warning)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "hi")

        monkeypatch.setattr(
            Path, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
        )
        recorder.write_partial()

    assert any("Partial trajectory write failed" in m for m in warnings_emitted)


async def test_write_partial_records_in_flight_tool_like_finish(recorder: TrajectoryRecorder) -> None:
    """A partial flush mid-invocation carries the same interrupted marker the
    final flush (finish()) emits, and the snapshot leaves live state untouched."""
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(ToolStartEvent(id="ip1", name="Read", input={"file_path": "a.py"}))
            recorder.write_partial()
            # Signal-safe snapshot: nothing consumed; finish() still marks the call.
            assert "ip1" in inv._in_flight_tools

    partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
    partial = json.loads(partial_path.read_text(encoding="utf-8"))
    final = read_trajectory(recorder.path)

    def marker_results(traj: dict[str, Any]) -> list[dict[str, Any]]:
        return [r
            for s in traj["steps"]
            if s["source"] == "agent"
            for r in (s.get("observation") or {}).get("results") or []
            if r.get("extra", {}).get("status") == "interrupted"
        ]

    expected = [{"content": INCOMPLETE_CONTENT, "extra": {"is_error": True, "status": "interrupted"}}]
    assert marker_results(partial) == expected
    assert marker_results(final) == expected

async def test_write_partial_preserves_in_flight_diagnostic_once(recorder: TrajectoryRecorder,) -> None:
    """Signal-safe snapshots include normalized diagnostics without consuming them."""
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(DiagnosticEvent(code="parser_gap", message="bounded", metadata={"count": 1}))
            recorder.write_partial()

    partial = json.loads(recorder.path.with_suffix(recorder.path.suffix + ".partial").read_text(encoding="utf-8"))
    final = read_trajectory(recorder.path)

    for trajectory in (partial, final):
        diagnostics = [diagnostic
            for step in trajectory["steps"]
            if step["source"] == "agent"
            for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])
        ]
        assert diagnostics == [{"code": "parser_gap", "message": "bounded", "metadata": {"count": 1}}]

async def test_write_partial_no_double_count_after_invocation_exit(tmp_path: Path,) -> None:
    """An exited invocation must not duplicate steps already transferred to the recorder."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="hi")
            observe_text_and_result(inv, "ok")

        # Now invocation has exited, steps flushed
        assert recorder._active_invocations == []
        flushed_count = len(recorder.steps)
        assert flushed_count >= 2

        recorder.write_partial()

    partial_path = tmp_path / ".daydream" / "trajectory.json.partial"
    data = json.loads(partial_path.read_text(encoding="utf-8"))
    assert len(data["steps"]) == flushed_count, f"Expected {flushed_count} steps post-exit, got {len(data['steps'])}"


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


async def test_signal_flush_with_child_evidence_freezes_schema_valid_empty_root(tmp_path: Path,) -> None:
    """An early fan-out signal retains root lifecycle evidence without an LLM call."""

    snapshots: list[RunWriteSnapshot] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    entered = anyio.Event()
    release = anyio.Event()
    children: dict[str, TrajectoryRecorder] = {}

    async with root:
        assert root.steps == []
        async with anyio.create_task_group() as tg:
            tg.start_soon(_hold_fork, root, "initial-exploration", "CHILD_ONLY", entered, release, children,)
            await entered.wait()

            flush_active_signal_recorders()
            assert len(snapshots) == 1
            snapshot = snapshots[0]
            assert [document.trajectory_id for document in snapshot.documents] == [
                root.trajectory_id, children["initial-exploration"].trajectory_id,
            ]
            root_payload = json.loads(snapshot.documents[0].json_bytes)
            assert atif_validate(root_payload, validate_images=False)
            assert root_payload["steps"] == [{"step_id": 1, "timestamp": snapshot.cutoff_at, "source": "system",
                    "message": "Daydream run snapshot",
                    "extra": {"daydream_run_flow": root.run_flow.value, "host_event": "partial_snapshot"},
                }
            ]
            assert root.steps == []
            assert trajectory_module.compute_timing_summary(snapshot) is not None
            release.set()

async def test_signal_flush_reuses_cutoff_until_any_document_state_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged run snapshot reuses bytes; child progress advances its cutoff."""

    ticks = iter(f"2026-09-06T00:00:{second:02d}.000000Z" for second in range(60))
    monkeypatch.setattr("daydream.timeutil.now_iso", lambda: next(ticks))
    snapshots: list[RunWriteSnapshot] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))

    async with root:
        async with trajectory_module.maybe_fork(root, "active-child") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                observe_text_and_result(inv, "first state")
                flush_active_signal_recorders()
                first = snapshots[-1]
                first_bytes = tuple(document.json_bytes for document in first.documents)

                flush_active_signal_recorders()
                second = snapshots[-1]
                assert second.cutoff_at == first.cutoff_at
                assert tuple(document.json_bytes for document in second.documents) == first_bytes

                inv.observe_user_step(prompt="later context")
                flush_active_signal_recorders()
                third = snapshots[-1]
                assert third.cutoff_at != first.cutoff_at
                third_bytes = tuple(document.json_bytes for document in third.documents)
                assert third_bytes != first_bytes
                assert {json.loads(document.json_bytes)["extra"]["snapshot_at"]
                    for document in third.documents
                } == {third.cutoff_at}
                assert tuple(document.json_bytes for document in first.documents) == first_bytes

async def test_signal_flush_root_prepare_failure_never_publishes_rootless_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root freeze failure writes no child-only snapshot and a retry recovers."""

    snapshots: list[RunWriteSnapshot] = []
    warnings: list[str] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot))
    original_prepare = root._prepare_document
    failed_once = False

    def fail_first_root_prepare(**kwargs: Any) -> Any:
        nonlocal failed_once
        if kwargs["status"] == "partial" and not failed_once:
            failed_once = True
            raise RuntimeError("root preparation failed")
        return original_prepare(**kwargs)

    monkeypatch.setattr(root, "_prepare_document", fail_first_root_prepare)
    monkeypatch.setattr("daydream.ui.print_warning", lambda _console, message: warnings.append(message),)

    async with root:
        async with root.fork("active-child") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe(TextEvent(text="child evidence"))

                flush_active_signal_recorders()
                assert len(warnings) == 1
                assert snapshots == []
                assert not _partial_path(root).exists()
                assert not _partial_path(child).exists()

                flush_active_signal_recorders()
                assert len(warnings) == 1
                assert len(snapshots) == 1
                snapshot = snapshots[0]
                assert [document.trajectory_id for document in snapshot.documents] == [
                    root.trajectory_id, child.trajectory_id,
                ]
                assert all(atif_validate(json.loads(document.json_bytes), validate_images=False)
                    for document in snapshot.documents
                )
                assert {json.loads(document.json_bytes)["extra"]["snapshot_at"]
                    for document in snapshot.documents
                } == {snapshot.cutoff_at}
                assert all(document.path.exists() for document in snapshot.documents)

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

@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
async def test_signal_handler_flushes_all_siblings_once(tmp_path: Path, signum: signal.Signals) -> None:
    """The real handler flushes root and both siblings without parent recursion."""

    root_statuses: list[str] = []
    root = make_recorder(tmp_path, on_write=lambda _rec, snapshot: root_statuses.append(snapshot.status))
    entered = {name: anyio.Event() for name in ("signal-a", "signal-b")}
    release = {name: anyio.Event() for name in entered}
    children: dict[str, TrajectoryRecorder] = {}
    markers = {"signal-a": "SIBLING_A_ONLY", "signal-b": "SIBLING_B_ONLY"}

    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")
        async with anyio.create_task_group() as tg:
            for name in ("signal-a", "signal-b"):
                tg.start_soon(_hold_fork, root, name, markers[name], entered[name], release[name], children,)
                await entered[name].wait()
            try:
                with pytest.raises(KeyboardInterrupt):
                    _signal_handler(signum, None)
                assert root_statuses == ["partial"]
                paths = [_partial_path(root), *(_partial_path(children[n]) for n in entered)]
                assert all(path.exists() for path in paths)
            finally:
                _finish_shutdown_panel()
                for event in release.values():
                    event.set()

@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
async def test_signal_handler_isolates_sibling_write_failure(
    recorder: TrajectoryRecorder, monkeypatch: pytest.MonkeyPatch, signum: signal.Signals,
) -> None:
    """One denied sibling partial cannot prevent healthy siblings or shutdown setup."""

    entered = {name: anyio.Event() for name in ("signal-a", "signal-b")}
    release = {name: anyio.Event() for name in entered}
    children: dict[str, TrajectoryRecorder] = {}
    warnings: list[str] = []
    real_write_text = Path.write_text

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "ROOT_ONLY")
        async with anyio.create_task_group() as tg:
            for name, marker in (("signal-a", "SIBLING_A_ONLY"), ("signal-b", "SIBLING_B_ONLY"),):
                tg.start_soon(_hold_fork, recorder, name, marker, entered[name], release[name], children,)
                await entered[name].wait()

            denied_path = _partial_path(children["signal-a"])

            def selective_write(path: Path, *args: Any, **kwargs: Any) -> int:
                if path == denied_path:
                    raise PermissionError("denied sibling A")
                return real_write_text(path, *args, **kwargs)

            monkeypatch.setattr(Path, "write_text", selective_write)
            monkeypatch.setattr("daydream.ui.print_warning", lambda _console, message: warnings.append(message),
            )
            try:
                with pytest.raises(KeyboardInterrupt):
                    _signal_handler(signum, None)
                assert not denied_path.exists()
                for candidate in (recorder, children["signal-b"]):
                    path = _partial_path(candidate)
                    assert path.exists()
                    assert atif_validate(read_trajectory(path), validate_images=False)
                matching_warnings = [
                    warning for warning in warnings if "Partial trajectory write failed: PermissionError" in warning
                ]
                assert len(matching_warnings) == 1
            finally:
                _finish_shutdown_panel()
                for event in release.values():
                    event.set()

async def test_forked_child_write_partial_captures_in_flight_steps(recorder: TrajectoryRecorder,) -> None:
    """A direct child write keeps the established child-to-parent cascade."""

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "parent-before-child")
        async with recorder.fork("child-branch") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe_user_step(prompt="forked-prompt")
                observe_text_and_result(inv, "forked-response")

                child.write_partial()

    partial_path = child.path.with_suffix(child.path.suffix + ".partial")
    parent_partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
    assert partial_path.exists(), "Child partial trajectory should be written"
    assert parent_partial_path.exists(), "Direct child partial should cascade to parent"

    data = json.loads(partial_path.read_text(encoding="utf-8"))
    parent_data = json.loads(parent_partial_path.read_text(encoding="utf-8"))
    assert data.get("extra", {}).get("partial") is True
    assert parent_data.get("extra", {}).get("partial") is True
    assert "parent-before-child" in json.dumps(parent_data, sort_keys=True)
    assert len(data["steps"]) >= 2, f"Child partial missing in-flight steps: {data['steps']!r}"

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


async def test_empty_fork_folds_nothing_into_parent(recorder: TrajectoryRecorder) -> None:
    """A fork whose write produced no steps contributes no totals."""
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_metrics_and_result(inv, "parent-text", message_id="m-1", prompt_tokens=100,
                completion_tokens=20, cached_tokens=10, cost_usd=1.0,
            )
        async with recorder.fork("empty-child"):
            pass

    parent = read_trajectory(recorder.path)["final_metrics"]
    assert parent["total_prompt_tokens"] == 100
    assert parent["total_cost_usd"] == pytest.approx(1.0)

async def test_nested_fork_totals_reach_the_root(recorder: TrajectoryRecorder) -> None:
    """Fold is transitive: a fork of a fork reaches the root's final_metrics."""
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_metrics_and_result(
                inv, "root", message_id="m-1", prompt_tokens=10, completion_tokens=1, cached_tokens=0, cost_usd=0.1,
            )
        async with recorder.fork("outer") as outer:
            async with outer.invocation(phase=DaydreamPhase.DEEP) as oinv:
                observe_metrics_and_result(oinv, "outer", message_id="m-2", prompt_tokens=20,
                    completion_tokens=2, cached_tokens=0, cost_usd=0.2,
                )
            async with outer.fork("inner") as inner:
                async with inner.invocation(phase=DaydreamPhase.DEEP) as iinv:
                    observe_metrics_and_result(iinv, "inner", message_id="m-3", prompt_tokens=30,
                        completion_tokens=3, cached_tokens=0, cost_usd=0.3,
                    )

    root = read_trajectory(recorder.path)["final_metrics"]
    assert root["total_prompt_tokens"] == 60
    assert root["total_cost_usd"] == pytest.approx(0.6)

async def test_analyze_costs_total_comes_from_root_only(tmp_path: Path) -> None:
    """Root final_metrics is fork-inclusive, so analyze_costs must not re-sum forks."""

    session = "sess-fold-0001"
    daydream_dir = tmp_path / ".daydream"
    snapshots: list[trajectory_module.RunWriteSnapshot] = []
    recorder = make_recorder(tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot),
        path=daydream_dir / "runs" / session / "trajectory.json", session_id=session,
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_metrics_and_result(inv, "parent", message_id="m-1", prompt_tokens=100,
                completion_tokens=20, cached_tokens=10, cost_usd=1.0,
            )
        async with recorder.fork("deep-python") as child:
            async with child.invocation(phase=DaydreamPhase.DEEP) as cinv:
                observe_metrics_and_result(cinv, "child", message_id="m-2", prompt_tokens=40,
                    completion_tokens=8, cached_tokens=4, cost_usd=0.5,
                )

    costs = analyze_costs(trajectory_module.snapshot_trajectories(snapshots[-1]))
    assert costs["total_cost_usd"] == pytest.approx(1.5)  # not 2.0 (root 1.5 + fork 0.5)
    assert costs["total_prompt_tokens_raw"] == 140  # not 180
    assert costs["total_completion_tokens"] == 28

    # Subtract folded child totals from the root row to avoid double-counting.
    by_agent = {a["agent"]: a for a in costs["by_agent"]}
    assert len(by_agent) == 2
    assert any(a["cost_usd"] == pytest.approx(0.5) for a in costs["by_agent"])
    assert sum(a["cost_usd"] for a in costs["by_agent"]) == pytest.approx(costs["total_cost_usd"])


async def test_build_trajectory_omits_backend_when_unset(recorder: TrajectoryRecorder) -> None:
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "hello")
    extra = read_trajectory(recorder.path)["extra"]
    assert "backend" not in extra
    assert "review_backend" not in extra
    assert "fix_backend" not in extra
    assert "test_backend" not in extra

async def test_build_trajectory_omits_empty_per_phase_backend_keys(tmp_path: Path,) -> None:
    """Per-phase backend keys are omitted when their name is empty (improve flow)."""
    recorder = make_recorder(
        tmp_path, run_flow=DaydreamRunFlow.IMPROVE, backend_name="codex", review_backend_name="codex",
        fix_backend_name="", test_backend_name="",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "hello")
    extra = read_trajectory(recorder.path)["extra"]
    assert extra["backend"] == "codex"
    assert extra["review_backend"] == "codex"
    assert "fix_backend" not in extra
    assert "test_backend" not in extra


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

    async with recorder:
        async with host_phase_scope(DaydreamPhase.REMOTE_CI) as phase:
            await anyio.sleep(0)
            phase.stop_reason = stop_reason

    remote_events = [event
        for event in [x.to_dict() for x in recorder._phase_events]
        if event["phase"] == DaydreamPhase.REMOTE_CI.value
    ]
    assert [event["event"] for event in remote_events] == ["phase_start", "phase_end"]
    ends = [event for event in remote_events if event["event"] == "phase_end"]
    assert len(ends) == 1
    assert ends[0]["status"] == expected_status
    assert ends[0].get("reason_code") == expected_reason
    assert ends[0]["metadata"]["stop_reason"] == stop_reason
    assert all(event["metadata"]["duration_ms"] >= 0 for event in ends)


class _RecordingSink:
    """A host document sink that records writes and immutable captures in order."""

    def __init__(self, *, persist: bool = True, fail_on: str | None = None) -> None:
        self._persist, self._fail_on = persist, fail_on
        self.writes: list[tuple[Any, str]] = []
        self.snapshots: list[Any] = []
        self.recorders: list[Any] = []
        self.contexts: list[TrajectoryRecorder | None] = []
        self.order: list[str] = []

    def writer(self, document: Any, status: str) -> None:
        if self._fail_on in (status, "any"):
            raise OSError(f"injected {status} output failure")
        self.order.append(f"writer:{document.trajectory_id}")
        if self._persist:
            document.path.parent.mkdir(parents=True, exist_ok=True)
            document.path.write_bytes(document.json_bytes)
        self.writes.append((document, status))

    def capture(self, recorder: TrajectoryRecorder, snapshot: Any) -> None:
        self.order.append("callback")
        self.contexts.append(get_current_recorder())
        self.recorders.append(recorder)
        self.snapshots.append(snapshot)

    @property
    def written_ids(self) -> list[str]:
        return [document.trajectory_id for document, _ in self.writes]

    @property
    def all_complete(self) -> bool:
        return all(status == "complete" for _, status in self.writes)


def _sink_recorder(
    sink: _RecordingSink, tmp_path: Path, *, session_id: str, artifact_run_dir: Path | None = None, **kwargs: Any,
) -> TrajectoryRecorder:
    """A recorder whose documents flow through *sink* rather than its own path."""
    kwargs.setdefault("run_flow", DaydreamRunFlow.CUSTOM)
    kwargs.setdefault("target_dir", tmp_path)
    kwargs.setdefault("agent_model_name", "")
    kwargs.setdefault("path", (artifact_run_dir or tmp_path / "private") / "trajectory.json")
    return TrajectoryRecorder(artifact_run_dir=artifact_run_dir, document_writer=sink.writer,
        session_id=session_id, on_write=sink.capture, **kwargs,
    )


def _host_only_step(recorder: TrajectoryRecorder, cutoff_at: str) -> dict[str, Any]:
    """The one synthetic step a bound host-only root publishes."""
    return {"step_id": 1, "timestamp": cutoff_at, "source": "system", "message": "Daydream host-only run snapshot",
        "extra": {"daydream_run_flow": recorder.run_flow.value, "host_event": "host_only_final_snapshot"},
    }

async def test_bound_empty_root_writes_host_only_final_snapshot(tmp_path: Path,) -> None:
    """A P10-bound host-only root reaches its writer and immutable capture."""
    private_run = tmp_path / "private" / "runs" / "host-only"
    sink = _RecordingSink()
    recorder = _sink_recorder(sink, tmp_path, session_id="host-only", artifact_run_dir=private_run)
    async with recorder:
        async with trajectory_module.phase_scope(DaydreamPhase.MERGE):
            pass
        assert recorder.steps == []

    assert sink.order == ["writer:host-only", "callback"]
    assert len(sink.writes) == len(sink.snapshots) == 1
    document, status = sink.writes[0]
    snapshot = sink.snapshots[0]
    assert status == snapshot.status == "complete"
    assert snapshot.documents == (document,) and snapshot.documents[0] is document
    assert document.path.read_bytes() == document.json_bytes

    payload = json.loads(document.json_bytes)
    assert atif_validate(payload, validate_images=False)
    assert document.trajectory_id == recorder.trajectory_id == "host-only"
    assert snapshot.root_trajectory_id == "host-only"
    assert payload["trajectory_id"] == payload["session_id"] == "host-only"
    assert payload["extra"]["run_ended_at"] == snapshot.cutoff_at
    assert recorder._run_ended_at == snapshot.cutoff_at
    assert payload["steps"] == [_host_only_step(recorder, snapshot.cutoff_at)]
    phase_events = payload["extra"]["phase_events"]
    assert [event["event"] for event in phase_events] == ["phase_start", "phase_end"]
    assert [event["phase"] for event in phase_events] == ["merge", "merge"]
    assert phase_events[-1]["status"] == "succeeded"
    assert payload["final_metrics"] == {"total_steps": 1}
    assert not payload["agent"].get("model_name")
    assert recorder._invocation_counter == 0
    assert recorder._step_id_counter == 0
    assert recorder.steps == []

async def test_standalone_empty_root_still_writes_nothing(tmp_path: Path) -> None:
    """An on_write callback alone does not opt an empty root into output."""
    callbacks: list[Any] = []
    recorder = make_recorder(tmp_path, on_write=lambda _recorder, snapshot: callbacks.append(snapshot),)

    async with recorder:
        async with trajectory_module.phase_scope(DaydreamPhase.MERGE):
            pass

    assert recorder.steps == []
    assert not recorder.path.exists()
    assert callbacks == []

@pytest.mark.parametrize("exception_kind", ["runtime", "cancel"])
async def test_bound_empty_root_abort_writes_partial_host_only_snapshot(tmp_path: Path, exception_kind: str,) -> None:
    """Escaping failure stays primary while the complete snapshot admits partial truth."""
    sink = _RecordingSink()
    session_id = f"host-only-{exception_kind}"
    primary: BaseException = (RuntimeError("authoritative body failure")
        if exception_kind == "runtime"
        else anyio.get_cancelled_exc_class()()
    )
    recorder = _sink_recorder(sink, tmp_path, session_id=session_id)
    caught: BaseException | None = None
    try:
        async with recorder:
            async with trajectory_module.phase_scope(DaydreamPhase.MERGE):
                raise primary
    except BaseException as exc:
        caught = exc

    assert caught is primary
    assert sink.order == [f"writer:{session_id}", "callback"]
    assert len(sink.writes) == len(sink.snapshots) == 1
    document, status = sink.writes[0]
    snapshot = sink.snapshots[0]
    assert status == snapshot.status == "complete"
    assert snapshot.documents == (document,)
    payload = json.loads(document.json_bytes)
    assert atif_validate(payload, validate_images=False)
    assert payload["extra"]["partial"] is True
    assert payload["extra"]["run_ended_at"] == snapshot.cutoff_at
    assert payload["steps"] == [_host_only_step(recorder, snapshot.cutoff_at)]
    expected_phase_status = "failed" if exception_kind == "runtime" else "cancelled"
    assert payload["extra"]["phase_events"][-1]["status"] == expected_phase_status
    assert recorder.steps == []
    assert recorder._invocation_counter == 0

async def test_bound_empty_root_with_completed_child_retains_both_documents(tmp_path: Path,) -> None:
    """A completed child is retained once behind the synthetic root document."""
    sink = _RecordingSink()
    recorder = _sink_recorder(
        sink, tmp_path, session_id="parent", artifact_run_dir=tmp_path / "private" / "runs" / "parent",
    )
    async with recorder:
        async with recorder.fork("completed-child") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                observe_text_and_result(invocation, "child evidence")
        assert recorder.steps == []

    assert sink.order == [f"writer:{child.trajectory_id}", f"writer:{recorder.trajectory_id}", "callback"]
    assert sink.written_ids == [child.trajectory_id, recorder.trajectory_id]
    assert sink.all_complete
    assert len(sink.snapshots) == 1
    snapshot = sink.snapshots[0]
    assert snapshot.status == "complete"
    assert [document.trajectory_id for document in snapshot.documents] == [recorder.trajectory_id, child.trajectory_id,
    ]
    assert snapshot.documents[0] is sink.writes[1][0]
    assert snapshot.documents[1] is sink.writes[0][0]
    assert sum(document.trajectory_id == child.trajectory_id for document in snapshot.documents) == 1

    root_payload = json.loads(snapshot.documents[0].json_bytes)
    assert atif_validate(root_payload, validate_images=False)
    assert root_payload["steps"] == [_host_only_step(recorder, snapshot.cutoff_at)]
    summaries = root_payload["extra"]["subtrajectories"]
    assert len(summaries) == 1
    assert summaries[0]["trajectory_id"] == child.trajectory_id
    assert "invocation_id" not in summaries[0]
    assert recorder._invocation_counter == 0
    assert recorder._step_id_counter == 0
    assert recorder.steps == []

async def test_artifact_document_writer_precedes_root_capture_and_is_inherited_by_fork(tmp_path: Path,) -> None:
    """The host sink owns root/child bytes while P07 keeps one pure root callback."""
    private_run = tmp_path / "private" / "runs" / "test"
    sink = _RecordingSink(persist=False)
    recorder = _sink_recorder(
        sink, tmp_path, session_id="test", run_flow=DaydreamRunFlow.NORMAL, artifact_run_dir=private_run,
        agent_model_name="test",
    )
    async with recorder:
        async with recorder.fork("child") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                observe_text_and_result(invocation, "child")
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            observe_text_and_result(invocation, "root")

    # Both documents reach the sink before the single root capture fires.
    assert sink.order == ["writer:test:child", "writer:test", "callback"]
    assert sink.all_complete
    assert sink.writes[0][0].path.parent == private_run / "trajectories"
    assert sink.recorders == sink.contexts == [recorder]
    assert get_current_recorder() is None
    snapshot = sink.snapshots[0]
    assert snapshot.status == "complete"
    assert [document.trajectory_id for document in snapshot.documents] == ["test", "test:child"]
    assert not recorder.path.exists()


async def test_artifact_partial_writer_failure_still_delivers_immutable_capture(tmp_path: Path,) -> None:
    """A failed live partial write cannot erase the already prepared P07 bytes."""
    sink = _RecordingSink(fail_on="partial")
    recorder = _sink_recorder(
        sink, tmp_path, session_id="test", run_flow=DaydreamRunFlow.NORMAL, artifact_run_dir=tmp_path / "private",
        agent_model_name="test",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            observe_text_and_result(invocation, "partial")
        recorder.write_partial()
        recorder.document_writer = None

    assert len(sink.snapshots) == 2
    partial = sink.snapshots[0]
    assert partial.status == "partial"
    assert json.loads(partial.documents[0].json_bytes)["extra"]["partial"] is True

async def test_artifact_final_writer_failure_preserves_existing_primary_exception(tmp_path: Path,) -> None:
    """A secondary explicit-output failure cannot replace the active body error."""
    primary = RuntimeError("authoritative body failure")
    recorder = _sink_recorder(
        _RecordingSink(fail_on="any"), tmp_path, session_id="test", run_flow=DaydreamRunFlow.NORMAL,
        agent_model_name="test", path=tmp_path / "explicit.json", explicit_path=True,
    )

    with pytest.raises(RuntimeError) as raised:
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                observe_text_and_result(invocation, "body")
            raise primary

    assert raised.value is primary
    assert any("trajectory finalization" in note for note in primary.__notes__)


async def test_do_commit_records_commit_phase_event(git_repo: Path, make_work: Any,) -> None:
    """Real-path: _do_commit's host-native commit emits a distinct ``commit``
    phase event with duration_ms + stop_reason (issue #726 task 12)."""

    work = make_work(git_repo)
    (git_repo / "app.py").write_text("x = 0\n")
    _git_add_commit(git_repo)
    (git_repo / "app.py").write_text("x = 1\n")

    rec = make_recorder(git_repo)
    async with rec:
        ok = await _do_commit(
            ScriptedBackend(), work, push=False,
            retained_paths=frozenset({"app.py"}),
            retained_states=git_ops.snapshot_worktree_paths(git_repo, {"app.py"}),
            initial_index=git_ops.snapshot_index(git_repo),
        )
    assert ok.committed is True
    assert ok.push is None

    commit_ends = [x.to_dict() for x in rec._phase_events if x.phase.value == "commit" and x.event == "phase_end"]
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
