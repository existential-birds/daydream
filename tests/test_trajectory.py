"""Tests for daydream/trajectory.py — TrajectoryRecorder + Invocation + Redactor.

Per D-18, tests follow schema-validity + behavior-predicate patterns. Full-tree
snapshot equality is banned (Pitfall 11). Most assertions go through
``daydream.atif.validate()`` plus one or two specific behavioral predicates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.atif import validate as atif_validate
from daydream.backends import (
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    Invocation,
    Redactor,
    TrajectoryRecorder,
    _safe_descriptor,
    get_current_recorder,
    now_iso,
)


def _make_recorder(tmp_path: Path, *, agent_model_name: str = "opus") -> TrajectoryRecorder:
    """Construct a TrajectoryRecorder rooted in tmp_path (test helper)."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
        session_id="test",
    )


def _read_trajectory(path: Path) -> dict[str, Any]:
    """Load the produced trajectory JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


# Behavior 1: TextEvent + ResultEvent → exactly one agent Step with that text


async def test_text_event_then_result_produces_one_agent_step(tmp_path: Path) -> None:
    """Behavior 1: One agent Step from a single TextEvent + ResultEvent."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="Hello world"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == "Hello world"


# Behavior 2: Two consecutive TextEvent chunks coalesce into one step (D-03)


async def test_text_event_chunks_coalesce_into_one_step(tmp_path: Path) -> None:
    """Behavior 2: Two TextEvents concatenate into one Step.message (D-03)."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="Hello "))
            inv.observe(TextEvent(text="world"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == "Hello world"


# Behavior 2b: ResultEvent flushes accumulated text; new text starts a new step
# (TEST-02 gap fill: explicit flush-on-result-boundary)


async def test_result_event_flushes_text_and_starts_new_step(tmp_path: Path) -> None:
    """TEST-02: ResultEvent terminates the current step; subsequent text starts a new one."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="first chunk"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
            inv.observe(TextEvent(text="second chunk"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 2
    assert agent_steps[0]["message"] == "first chunk"
    assert agent_steps[1]["message"] == "second chunk"


# Behavior 3: ToolStart + ToolResult → tool_call & observation in SAME step
# (CORE-06, Pitfall 3)


async def test_tool_call_and_result_land_on_same_step(tmp_path: Path) -> None:
    """Behavior 3: ToolStartEvent + ToolResultEvent both land on same Step."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="Running a tool"))
            inv.observe(ToolStartEvent(id="tool-1", name="Bash", input={"command": "ls"}))
            inv.observe(ToolResultEvent(id="tool-1", output="file1\nfile2", is_error=False))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    step = agent_steps[0]
    assert step["tool_calls"][0]["tool_call_id"] == "tool-1"
    assert step["observation"]["results"][0]["source_call_id"] == "tool-1"


# Behavior: mark_aborted stamps extra["stop_reason"] on the closing step
# and the trajectory stays schema-valid (Task 2).


async def test_mark_aborted_stamps_stop_reason_on_closing_step(tmp_path: Path) -> None:
    """An aborted invocation closes cleanly with extra['stop_reason'] set."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
            inv.observe(ToolStartEvent(id="tool-1", name="Bash", input={"command": "ls"}))
            inv.mark_aborted("budget_exceeded")

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["extra"]["stop_reason"] == "budget_exceeded"


# Behavior 4: User step has source="user" and NO agent-only fields
# (Pitfall 4)


async def test_user_step_omits_agent_only_fields(tmp_path: Path) -> None:
    """Behavior 4: observe_user_step produces source='user' with NO agent fields."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="What is the answer?")
            inv.observe(TextEvent(text="42"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    user_steps = [s for s in traj["steps"] if s["source"] == "user"]
    assert len(user_steps) == 1
    user = user_steps[0]
    assert user["message"] == "What is the answer?"
    # Agent-only fields must be absent (not empty / not None — absent from JSON)
    for forbidden in ("model_name", "tool_calls", "metrics", "reasoning_content"):
        assert forbidden not in user, (
            f"User step must not carry agent-only field '{forbidden}'; got {user.keys()}"
        )


# Behavior 5: MetricsEvent attaches Metrics; cached_tokens is a SUBSET (D-15)


async def test_metrics_event_cached_tokens_is_subset_not_added(tmp_path: Path) -> None:
    """Behavior 5: MetricsEvent.cached_tokens is a SUBSET of prompt_tokens (D-15)."""
    recorder = _make_recorder(tmp_path)
    metrics_event = MetricsEvent(
        message_id="msg-1",
        prompt_tokens=500,
        completion_tokens=80,
        cached_tokens=100,
        cost_usd=0.01,
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="thinking..."))
            inv.observe(metrics_event)
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    metrics = agent_steps[0]["metrics"]
    # Critical: prompt_tokens stays 500 (NOT 600). cached_tokens is reported alongside.
    assert metrics["prompt_tokens"] == 500
    assert metrics["cached_tokens"] == 100
    assert metrics["completion_tokens"] == 80


# Behavior 6: dispatch exception is caught at the recorder boundary; run continues
# (Architecture Q7)


async def test_dispatch_failure_is_caught_and_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavior 6: Recorder boundary catches dispatch exceptions; run continues."""
    recorder = _make_recorder(tmp_path)

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
            inv.observe(TextEvent(text="after-failure"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    # "before-failure" + "after-failure" both make it; "will-fail" was dropped.
    assert "before-failure" in agent_steps[0]["message"]
    assert "after-failure" in agent_steps[0]["message"]
    assert "will-fail" not in agent_steps[0]["message"]


# Recorder-level Behavior A: clean exit writes a schema-valid JSON file


async def test_recorder_writes_schema_valid_trajectory_on_clean_exit(tmp_path: Path) -> None:
    """Behavior A: clean __aexit__ writes a JSON file passing daydream.atif.validate."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="hello")
            inv.observe(TextEvent(text="world"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert recorder.path.exists()
    assert atif_validate(recorder.path, validate_images=False) is True


# Recorder-level Behavior B: sequential step_id across two invocations (Pitfall 1)


async def test_step_ids_sequential_across_two_invocations(tmp_path: Path) -> None:
    """Behavior B: step_id is sequential 1..N across two invocations (Pitfall 1)."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv1:
            inv1.observe(TextEvent(text="review-output"))
            inv1.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv2:
            inv2.observe(TextEvent(text="fix-output"))
            inv2.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    step_ids = [s["step_id"] for s in traj["steps"]]
    assert step_ids == list(range(1, len(step_ids) + 1))
    assert len(step_ids) == 2


# Recorder-level Behavior C: write failure on __aexit__ degrades with warning
# (D-11)


async def test_write_failure_degrades_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavior C: PermissionError on write emits warning; run does not raise."""
    recorder = _make_recorder(tmp_path)
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    monkeypatch.setattr("daydream.trajectory.print_warning", fake_print_warning)
    monkeypatch.setattr(
        "os.replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    # Exit MUST NOT raise even though _write fails inside __aexit__
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert any("Trajectory write failed" in m for m in warnings_emitted)


# Recorder-level Behavior D: FinalMetrics totals are sum of per-step Metrics


async def test_final_metrics_totals_match_per_step_sum(tmp_path: Path) -> None:
    """Behavior D: FinalMetrics totals equal the sum of MetricsEvent values."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="step-one-text"))
            inv.observe(MetricsEvent(
                message_id="m-1", prompt_tokens=100, completion_tokens=20,
                cached_tokens=10, cost_usd=0.001,
            ))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv2:
            inv2.observe(TextEvent(text="step-two-text"))
            inv2.observe(MetricsEvent(
                message_id="m-2", prompt_tokens=200, completion_tokens=40,
                cached_tokens=20, cost_usd=0.002,
            ))
            inv2.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    fm = traj["final_metrics"]
    assert fm["total_prompt_tokens"] == 300
    assert fm["total_completion_tokens"] == 60
    assert fm["total_cached_tokens"] == 30
    assert abs(fm["total_cost_usd"] - 0.003) < 1e-9
    assert fm["total_steps"] == len(traj["steps"])


# Recorder-level Behavior E: ContextVar is set inside, cleared after


async def test_context_var_set_inside_and_cleared_after(tmp_path: Path) -> None:
    """Behavior E: get_current_recorder is the recorder inside, None after."""
    assert get_current_recorder() is None
    recorder = _make_recorder(tmp_path)
    async with recorder:
        assert get_current_recorder() is recorder
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    assert get_current_recorder() is None


# Recorder-level Behavior F (CORE-08): Trajectory.agent identity baked in


async def test_trajectory_agent_identity_is_daydream(tmp_path: Path) -> None:
    """Behavior F (CORE-08): agent.name='daydream', version non-empty, model_name passed-in."""
    recorder = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    assert traj["agent"]["name"] == "daydream"
    assert isinstance(traj["agent"]["version"], str) and traj["agent"]["version"]
    assert traj["agent"]["model_name"] == "opus"


# Bonus: schema_version + session_id present and well-formed


async def test_schema_version_and_session_id_present(tmp_path: Path) -> None:
    """schema_version pinned to ATIF-v1.6 (D-09); session_id is non-empty UUID."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = _read_trajectory(recorder.path)
    assert traj["schema_version"] == "ATIF-v1.6"
    assert isinstance(traj["session_id"], str)
    assert len(traj["session_id"]) > 0


# Sanity: now_iso, Redactor, Invocation public surface


def test_now_iso_ends_with_z() -> None:
    """now_iso returns ISO 8601 string ending in 'Z' (Pitfall 2)."""
    ts = now_iso()
    assert ts.endswith("Z")


def test_redactor_is_passthrough() -> None:
    """Redactor.redact_step preserves semantic equality on clean inputs.

    Phase 4: Redactor returns a fresh model_copy whenever ANY scannable text
    field is present. Identity (`out is step`) is NO LONGER guaranteed; the
    contract is field-by-field semantic equality on inputs containing no
    secret patterns.
    """
    from daydream.atif import Step as AtifStep

    step = AtifStep(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message="hello",
    )
    out = Redactor().redact_step(step)
    assert out.message == step.message
    assert out.reasoning_content == step.reasoning_content
    assert out.tool_calls == step.tool_calls
    assert out.observation == step.observation
    assert out.extra == step.extra


def test_redactor_scrubs_api_key_in_message() -> None:
    """Phase 4 RED: Redactor.redact_step replaces sk-* tokens in Step.message."""
    from daydream.atif import Step as AtifStep

    step = AtifStep(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message="token=sk-test-12345abcdef done",
    )
    out = Redactor().redact_step(step)
    assert isinstance(out.message, str)
    assert "sk-test-12345abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


def test_invocation_has_no_parent_field() -> None:
    """D-08: Invocation does not carry parent; parent linkage is on TrajectoryRecorder."""
    fields = {f.name for f in Invocation.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    assert "parent" not in fields, (
        f"Invocation must not carry a parent field in Phase 2 (D-08); got {fields}"
    )


# No-recorder no-op (CORE-09)


def test_no_recorder_no_op_get_current_returns_none() -> None:
    """CORE-09: outside any recorder context, get_current_recorder is None."""
    assert get_current_recorder() is None


# compute_wall_clock_seconds: derived from step timestamps, no --eval needed


def _append_step_at(recorder: TrajectoryRecorder, ts: str) -> None:
    """Append a minimal agent step with an explicit ISO-8601 timestamp."""
    from daydream.atif import Step as AtifStep

    recorder.steps.append(
        AtifStep(
            step_id=recorder._next_step_id(),
            timestamp=ts,
            source="agent",
            message="x",
        )
    )


def test_compute_wall_clock_seconds_spans_first_to_last(tmp_path: Path) -> None:
    """Span is max(timestamp) - min(timestamp), in seconds, regardless of insertion order."""
    recorder = _make_recorder(tmp_path)
    _append_step_at(recorder, "2026-05-31T10:00:00.000000Z")
    _append_step_at(recorder, "2026-05-31T10:00:07.500000Z")
    _append_step_at(recorder, "2026-05-31T10:00:03.000000Z")

    assert recorder.compute_wall_clock_seconds() == 7.5


def test_compute_wall_clock_seconds_single_step_is_none(tmp_path: Path) -> None:
    """Fewer than two timestamped steps means no measurable span."""
    recorder = _make_recorder(tmp_path)
    _append_step_at(recorder, "2026-05-31T10:00:00.000000Z")

    assert recorder.compute_wall_clock_seconds() is None


def test_compute_wall_clock_seconds_no_steps_is_none(tmp_path: Path) -> None:
    """An empty recorder yields None rather than raising."""
    recorder = _make_recorder(tmp_path)

    assert recorder.compute_wall_clock_seconds() is None


# Fork / Sibling / Continuation tests (Phase 3, SUBA-01..09)


def _observe_text_and_result(inv: Any, text: str = "output") -> None:
    """Helper: observe a TextEvent + ResultEvent to produce a minimal agent step."""
    inv.observe(TextEvent(text=text))
    inv.observe(ResultEvent(structured_output=None, continuation=None))


# SUBA-07: ContextVar isolation inside fork


async def test_fork_contextvar_isolation(tmp_path: Path) -> None:
    """SUBA-07: Inside fork scope get_current_recorder() returns child; outside returns parent."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        assert get_current_recorder() is recorder
        async with recorder.fork("fix-0") as child:
            assert get_current_recorder() is child
            assert get_current_recorder() is not recorder
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                _observe_text_and_result(inv)
        assert get_current_recorder() is recorder


# SUBA-06: Sibling inherits session_id


async def test_sibling_inherits_session_id(tmp_path: Path) -> None:
    """SUBA-06: Child trajectory file has same session_id as parent."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                _observe_text_and_result(inv)
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)

    parent_traj = _read_trajectory(recorder.path)
    sibling_path = child.path
    sibling_traj = _read_trajectory(sibling_path)
    assert parent_traj["session_id"] == sibling_traj["session_id"]
    assert parent_traj["session_id"] == recorder.session_id


# SUBA-06: Sibling file path format


async def test_sibling_file_path_format(tmp_path: Path) -> None:
    """SUBA-06: Sibling path is <target>/.daydream/runs/<session_id>/trajectories/<descriptor>.json."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("deep-python") as child:
            async with child.invocation(phase=DaydreamPhase.DEEP) as inv:
                _observe_text_and_result(inv)

    expected = (
        tmp_path
        / ".daydream"
        / "runs"
        / recorder.session_id
        / "trajectories"
        / "deep-python.json"
    )
    assert child.path == expected
    assert expected.exists()


# SUBA-08: Step ID isolation across siblings


async def test_step_id_isolation_across_siblings(tmp_path: Path) -> None:
    """SUBA-08: Parent and child step_ids both start from 1 independently."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv, "parent-step")
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                _observe_text_and_result(inv, "child-step")
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)

    parent_traj = _read_trajectory(recorder.path)
    child_traj = _read_trajectory(child.path)

    parent_ids = [s["step_id"] for s in parent_traj["steps"]]
    child_ids = [s["step_id"] for s in child_traj["steps"]]

    assert parent_ids == list(range(1, len(parent_ids) + 1))
    assert child_ids == list(range(1, len(child_ids) + 1))
    assert child_ids[0] == 1


# SUBA-09: Parent FinalMetrics exclude children


async def test_parent_metrics_exclude_children(tmp_path: Path) -> None:
    """SUBA-09: Parent FinalMetrics totals do NOT include child step metrics."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="parent-text"))
            inv.observe(MetricsEvent(
                message_id="m-parent", prompt_tokens=100, completion_tokens=10,
                cached_tokens=5, cost_usd=0.001,
            ))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                inv.observe(TextEvent(text="child-text"))
                inv.observe(MetricsEvent(
                    message_id="m-child", prompt_tokens=200, completion_tokens=20,
                    cached_tokens=10, cost_usd=0.002,
                ))
                inv.observe(ResultEvent(structured_output=None, continuation=None))
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)

    parent_traj = _read_trajectory(recorder.path)
    child_traj = _read_trajectory(child.path)

    assert parent_traj["final_metrics"]["total_prompt_tokens"] == 100
    assert child_traj["final_metrics"]["total_prompt_tokens"] == 200


# SUBA-02: Dispatch step has subagent_trajectory_ref


async def test_dispatch_step_has_subagent_trajectory_ref(tmp_path: Path) -> None:
    """SUBA-02: Dispatch step carries subagent_trajectory_ref entries."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                _observe_text_and_result(inv)
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)

    parent_traj = _read_trajectory(recorder.path)
    assert atif_validate(parent_traj, validate_images=False) is True

    dispatch_steps = [
        s for s in parent_traj["steps"]
        if s["source"] == "agent" and "Dispatching" in s.get("message", "")
    ]
    assert len(dispatch_steps) == 1
    dispatch = dispatch_steps[0]
    ref = dispatch["observation"]["results"][0]["subagent_trajectory_ref"][0]
    assert ref["session_id"] == recorder.session_id


# Dispatch step uses relative path (starts with "trajectories/")


async def test_dispatch_step_uses_relative_path(tmp_path: Path) -> None:
    """Dispatch step subagent_trajectory_ref.trajectory_path is relative to .daydream."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                _observe_text_and_result(inv)
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)

    parent_traj = _read_trajectory(recorder.path)
    dispatch_steps = [
        s for s in parent_traj["steps"]
        if s["source"] == "agent" and "Dispatching" in s.get("message", "")
    ]
    ref = dispatch_steps[0]["observation"]["results"][0]["subagent_trajectory_ref"][0]
    assert ref["trajectory_path"].startswith("runs/")
    assert ref["trajectory_path"].endswith(".json")


# Dispatch step no-op when no siblings


async def test_dispatch_step_noop_when_no_siblings(tmp_path: Path) -> None:
    """create_dispatch_step with empty _registered_siblings adds no steps."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)
        steps_before = len(recorder.steps)
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
        assert len(recorder.steps) == steps_before


# _safe_descriptor slugification


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


# SUBA-01: Sequential phases produce single file


async def test_sequential_phases_single_file(tmp_path: Path) -> None:
    """SUBA-01: Three sequential invocations produce one file with continuous step_ids."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        for phase in (DaydreamPhase.REVIEW, DaydreamPhase.PARSE, DaydreamPhase.FIX):
            async with recorder.invocation(phase=phase) as inv:
                _observe_text_and_result(inv, f"{phase.value}-output")

    assert recorder.path.exists()
    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True

    step_ids = [s["step_id"] for s in traj["steps"]]
    assert step_ids == list(range(1, len(step_ids) + 1))

    traj_dir = tmp_path / ".daydream" / "trajectories"
    assert not traj_dir.exists() or len(list(traj_dir.iterdir())) == 0


# SUBA-05: Continuation appends no sibling


async def test_continuation_appends_no_sibling(tmp_path: Path) -> None:
    """SUBA-05: Two invocations simulating continuation produce one file."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
            _observe_text_and_result(inv, "first")
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
            _observe_text_and_result(inv, "second")

    assert recorder.path.exists()
    traj = _read_trajectory(recorder.path)
    step_ids = [s["step_id"] for s in traj["steps"]]
    assert step_ids == [1, 2]

    traj_dir = tmp_path / ".daydream" / "trajectories"
    assert not traj_dir.exists() or len(list(traj_dir.iterdir())) == 0


# Fork write failure degrades gracefully


async def test_fork_write_failure_degrades(tmp_path: Path) -> None:
    """If child _write() raises, parent ContextVar is restored, no crash."""
    recorder = _make_recorder(tmp_path)
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    async with recorder:
        with patch("daydream.trajectory.print_warning", fake_print_warning):
            async with recorder.fork("fail-child") as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    _observe_text_and_result(inv)
                # Sabotage the write path to be an unwritable directory
                child.path = Path("/nonexistent-dir-xyz/child.json")

        assert get_current_recorder() is recorder
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)

    assert any("Sibling trajectory write failed" in m for m in warnings_emitted)
    assert recorder.path.exists()


# Pitfall 6: Fork child with no steps produces no file


async def test_fork_child_no_steps_no_file(tmp_path: Path) -> None:
    """Pitfall 6: Child with 0 steps writes no sibling file; parent has no registration."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("empty-child"):
            pass
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)

    traj_dir = tmp_path / ".daydream" / "trajectories"
    assert not traj_dir.exists() or len(list(traj_dir.iterdir())) == 0
    assert len(recorder._registered_siblings) == 0


# Multiple forks all registered


async def test_multiple_forks_all_registered(tmp_path: Path) -> None:
    """Three sequential forks all register with parent; dispatch step has 3 refs."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        for i in range(3):
            async with recorder.fork(f"fix-{i}") as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    _observe_text_and_result(inv, f"child-{i}")
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)

    parent_traj = _read_trajectory(recorder.path)
    assert atif_validate(parent_traj, validate_images=False) is True

    dispatch_steps = [
        s for s in parent_traj["steps"]
        if s["source"] == "agent" and "Dispatching" in s.get("message", "")
    ]
    assert len(dispatch_steps) == 1
    results = dispatch_steps[0]["observation"]["results"]
    assert len(results) == 3
    for r in results:
        assert len(r["subagent_trajectory_ref"]) == 1


# Fork validator accepts both parent and child


async def test_fork_validator_accepts_both(tmp_path: Path) -> None:
    """Both parent and child trajectories pass atif_validate."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                _observe_text_and_result(inv)
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv)

    parent_traj = _read_trajectory(recorder.path)
    child_traj = _read_trajectory(child.path)

    assert atif_validate(parent_traj, validate_images=False) is True
    assert atif_validate(child_traj, validate_images=False) is True


# write_partial tests (CLI-03, D-07 SIGINT partial flush)


async def test_write_partial_writes_partial_file_with_partial_flag(tmp_path: Path) -> None:
    """CLI-03: write_partial writes <path>.partial with extra.partial=true."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="in-flight"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        recorder.write_partial()

        partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
        assert partial_path.exists()
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        assert partial.get("extra", {}).get("partial") is True
        assert atif_validate(partial, validate_images=False) is True


def test_write_partial_no_op_when_steps_empty(tmp_path: Path) -> None:
    """write_partial skips disk write when steps list is empty (matches _write)."""
    recorder = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    )
    recorder.write_partial()
    partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
    assert not partial_path.exists()


async def test_write_partial_is_idempotent(tmp_path: Path) -> None:
    """Calling write_partial twice yields a single .partial file with latest contents."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="first"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        recorder.write_partial()
        partial_path = recorder.path.with_suffix(recorder.path.suffix + ".partial")
        first = partial_path.read_text(encoding="utf-8")
        recorder.write_partial()
        second = partial_path.read_text(encoding="utf-8")
        assert first == second


async def test_write_partial_failure_emits_warning_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disk-write failure during partial flush degrades with warning, never raises."""
    recorder = _make_recorder(tmp_path)
    warnings_emitted: list[str] = []

    def fake_print_warning(_console: Any, message: str) -> None:
        warnings_emitted.append(message)

    monkeypatch.setattr("daydream.trajectory.print_warning", fake_print_warning)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

        monkeypatch.setattr(
            Path,
            "write_text",
            lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
        )
        recorder.write_partial()

    assert any("Partial trajectory write failed" in m for m in warnings_emitted)


# Regression: WR-01 — write_partial must capture Invocation in-flight steps


async def test_write_partial_captures_in_flight_invocation_steps(tmp_path: Path) -> None:
    """WR-01 regression: SIGINT mid-run_agent() must include in-flight Invocation steps.

    Pre-fix bug: write_partial read recorder.steps directly, but Invocation
    accumulates steps in its own list and only flushes to recorder.steps on
    __aexit__. A partial flush mid-invocation lost every step from the
    in-flight invocation.
    """
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            # Steps observed inside the invocation but BEFORE __aexit__
            inv.observe_user_step(prompt="hello")
            inv.observe(TextEvent(text="response"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

            assert recorder.steps == [], "recorder.steps should be empty mid-invocation"
            assert len(inv.steps) >= 2, "Invocation should have accumulated steps"

            recorder.write_partial()

    partial_path = tmp_path / ".daydream" / "trajectory.json.partial"
    assert partial_path.exists(), "Partial trajectory file should be written"

    data = json.loads(partial_path.read_text(encoding="utf-8"))
    assert data.get("extra", {}).get("partial") is True
    assert len(data["steps"]) >= 2, (
        f"Partial trajectory missing in-flight steps: {data['steps']!r}"
    )
    messages = [s.get("message") for s in data["steps"] if isinstance(s, dict)]
    assert any("hello" in (m or "") for m in messages), (
        f"User step prompt not in partial: {messages!r}"
    )


async def test_write_partial_no_double_count_after_invocation_exit(tmp_path: Path) -> None:
    """After Invocation.__aexit__, write_partial should NOT double-count steps.

    Steps moved from invocation.steps to recorder.steps; write_partial reads
    recorder.steps + active_invocations. After the invocation exits,
    active_invocations is empty so we read only recorder.steps.
    """
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="hi")
            inv.observe(TextEvent(text="ok"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

        # Now invocation has exited, steps flushed
        assert recorder._active_invocations == []
        flushed_count = len(recorder.steps)
        assert flushed_count >= 2

        recorder.write_partial()

    partial_path = tmp_path / ".daydream" / "trajectory.json.partial"
    data = json.loads(partial_path.read_text(encoding="utf-8"))
    assert len(data["steps"]) == flushed_count, (
        f"Expected {flushed_count} steps post-exit, got {len(data['steps'])}"
    )


# Regression: WR-02 — get_signal_recorder reads module-level stack, not ContextVar


async def test_get_signal_recorder_returns_active_recorder(tmp_path: Path) -> None:
    """WR-02 regression: signal-handler-safe accessor returns the active recorder.

    Signal handlers fire in the main thread outside the asyncio task context,
    so ``ContextVar.get()`` is non-deterministic. ``get_signal_recorder`` reads
    a module-level stack populated synchronously in __aenter__, which is
    reliable regardless of where the signal interleaves.
    """
    from daydream.trajectory import get_signal_recorder

    assert get_signal_recorder() is None, "Stack should be empty before __aenter__"

    recorder = _make_recorder(tmp_path)
    async with recorder:
        assert get_signal_recorder() is recorder, "Signal recorder should be set inside async with"

    assert get_signal_recorder() is None, "Stack should be empty after __aexit__"


async def test_get_signal_recorder_returns_innermost_for_nested_recorders(
    tmp_path: Path,
) -> None:
    """Nested recorders push onto the stack; signal-handler reads the top (innermost)."""
    from daydream.trajectory import get_signal_recorder

    outer = _make_recorder(tmp_path / "outer")
    inner = _make_recorder(tmp_path / "inner")

    async with outer:
        assert get_signal_recorder() is outer
        async with inner:
            assert get_signal_recorder() is inner
        assert get_signal_recorder() is outer

    assert get_signal_recorder() is None


# Regression: forked child recorders must be visible to signal handler


async def test_forked_child_visible_to_signal_handler(tmp_path: Path) -> None:
    """Forked child recorder must appear on _ACTIVE_RECORDERS so SIGINT can flush it.

    Pre-fix bug: _ForkCM.__aenter__ set the ContextVar but never appended
    the child to _ACTIVE_RECORDERS, so get_signal_recorder() could not see
    it and write_partial() on the child was never called during SIGINT.
    """
    from daydream.trajectory import get_signal_recorder

    parent = _make_recorder(tmp_path)
    async with parent:
        assert get_signal_recorder() is parent
        async with parent.fork("child-branch") as child:
            assert get_signal_recorder() is child, (
                "Forked child should be top of signal-handler stack"
            )
        assert get_signal_recorder() is parent, (
            "Parent should be restored after child fork exits"
        )


async def test_forked_child_write_partial_captures_in_flight_steps(tmp_path: Path) -> None:
    """SIGINT mid-fork must flush child's in-flight steps via write_partial."""
    from daydream.trajectory import get_signal_recorder

    parent = _make_recorder(tmp_path)
    async with parent:
        async with parent.fork("child-branch") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe_user_step(prompt="forked-prompt")
                inv.observe(TextEvent(text="forked-response"))
                inv.observe(ResultEvent(structured_output=None, continuation=None))

                sig_recorder = get_signal_recorder()
                assert sig_recorder is child
                sig_recorder.write_partial()

    partial_path = child.path.with_suffix(child.path.suffix + ".partial")
    assert partial_path.exists(), "Child partial trajectory should be written"

    data = json.loads(partial_path.read_text(encoding="utf-8"))
    assert data.get("extra", {}).get("partial") is True
    assert len(data["steps"]) >= 2, (
        f"Child partial missing in-flight steps: {data['steps']!r}"
    )


async def test_recorder_marks_partial_on_exception_exit(tmp_path: Path) -> None:
    """When __aexit__ receives an exception, the trajectory is marked partial."""
    recorder = _make_recorder(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe_user_step(prompt="hello")
                inv.observe(TextEvent(text="partial output"))
                raise RuntimeError("boom")

    assert recorder.path.exists()
    traj = _read_trajectory(recorder.path)
    assert traj.get("extra", {}).get("partial") is True


async def test_forked_child_marks_partial_on_exception_exit(tmp_path: Path) -> None:
    """A sibling that dies mid-flight is marked partial on exception exit.

    Regression for the fork path: _ForkCM.__aexit__ must mirror the top-level
    recorder and set child._aborted when an exception escapes the fork scope,
    so the sibling trajectory's extra.partial reflects that it was aborted —
    not silently written as if it completed cleanly.
    """
    recorder = _make_recorder(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        async with recorder:
            async with recorder.fork("fix-0") as child:
                async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                    inv.observe_user_step(prompt="hello")
                    inv.observe(TextEvent(text="partial sibling output"))
                    raise RuntimeError("boom")

    assert child.path.exists(), "Sibling trajectory should be written on exception exit"
    sibling_traj = _read_trajectory(child.path)
    assert sibling_traj.get("extra", {}).get("partial") is True, (
        "Forked child trajectory must be marked partial when an exception escapes the fork"
    )


async def test_forked_child_does_not_mark_partial_on_clean_exit(tmp_path: Path) -> None:
    """A sibling that exits cleanly is NOT marked partial."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                inv.observe_user_step(prompt="hello")
                inv.observe(TextEvent(text="sibling output"))
                inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert child.path.exists()
    sibling_traj = _read_trajectory(child.path)
    assert "partial" not in sibling_traj.get("extra", {})


async def test_recorder_does_not_mark_partial_on_clean_exit(tmp_path: Path) -> None:
    """Clean exit does NOT mark the trajectory as partial."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="hello")
            inv.observe(TextEvent(text="world"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert recorder.path.exists()
    traj = _read_trajectory(recorder.path)
    assert "partial" not in traj.get("extra", {})
