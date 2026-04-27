"""Tests for daydream/trajectory.py — TrajectoryRecorder + Invocation + Redactor.

Per D-18, tests follow schema-validity + behavior-predicate patterns. Full-tree
snapshot equality is banned (Pitfall 11). Most assertions go through
``daydream.atif.validate()`` plus one or two specific behavioral predicates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daydream.atif import validate as atif_validate
from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    Invocation,
    Redactor,
    TrajectoryRecorder,
    _reset_recorder_for_tests,
    get_current_recorder,
    now_iso,
)


# Stub MetricsEvent for tests until Plan 02-02 lands the real one in
# daydream/backends/__init__.py. The recorder's _dispatch matches by class
# name to stay compatible with both forms.
@dataclass
class _StubMetricsEvent:
    """Stand-in for the future daydream.backends.MetricsEvent dataclass.

    Attributes match EVNT-02 verbatim (D-15 names: prompt_tokens /
    completion_tokens / cached_tokens / cost_usd, not input_tokens /
    output_tokens). Class name is `MetricsEvent` so type(event).__name__
    matches the production class once Plan 02-02 lands.
    """

    message_id: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    cost_usd: float | None


# Use the production class name so duck-type dispatch works regardless of
# whether daydream.backends.MetricsEvent exists yet.
_StubMetricsEvent.__name__ = "MetricsEvent"


@pytest.fixture(autouse=True)
def _reset_recorder() -> Any:
    """Reset _RECORDER_VAR before and after every test (mirrors D-17)."""
    _reset_recorder_for_tests()
    yield
    _reset_recorder_for_tests()


def _make_recorder(tmp_path: Path, *, agent_model_name: str = "opus") -> TrajectoryRecorder:
    """Construct a TrajectoryRecorder rooted in tmp_path (test helper)."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
    )


def _read_trajectory(path: Path) -> dict[str, Any]:
    """Load the produced trajectory JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Behavior 1: TextEvent + ResultEvent → exactly one agent Step with that text
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Behavior 2: Two consecutive TextEvent chunks coalesce into one step (D-03)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Behavior 3: ToolStart + ToolResult → tool_call & observation in SAME step
# (CORE-06, Pitfall 3)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Behavior 4: User step has source="user" and NO agent-only fields
# (Pitfall 4)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Behavior 5: MetricsEvent attaches Metrics; cached_tokens is a SUBSET (D-15)
# ---------------------------------------------------------------------------


async def test_metrics_event_cached_tokens_is_subset_not_added(tmp_path: Path) -> None:
    """Behavior 5: MetricsEvent.cached_tokens is a SUBSET of prompt_tokens (D-15)."""
    recorder = _make_recorder(tmp_path)
    metrics_event = _StubMetricsEvent(
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


# ---------------------------------------------------------------------------
# Behavior 6: dispatch exception is caught at the recorder boundary; run continues
# (Architecture Q7)
# ---------------------------------------------------------------------------


async def test_dispatch_failure_is_caught_and_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavior 6: Recorder boundary catches dispatch exceptions; run continues."""
    recorder = _make_recorder(tmp_path)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            # First observation succeeds
            inv.observe(TextEvent(text="before-failure"))

            # Force a single dispatch failure by monkeypatching _dispatch.
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
            # And the next event must dispatch normally.
            inv.observe(TextEvent(text="after-failure"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    # The run did not crash. The trajectory was written.
    traj = _read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    # "before-failure" + "after-failure" both make it; "will-fail" was dropped.
    assert "before-failure" in agent_steps[0]["message"]
    assert "after-failure" in agent_steps[0]["message"]
    assert "will-fail" not in agent_steps[0]["message"]


# ---------------------------------------------------------------------------
# Recorder-level Behavior A: clean exit writes a schema-valid JSON file
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Recorder-level Behavior B: sequential step_id across two invocations (Pitfall 1)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Recorder-level Behavior C: write failure on __aexit__ degrades with warning
# (D-11)
# ---------------------------------------------------------------------------


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
        Path,
        "write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    # Exit MUST NOT raise even though _write fails inside __aexit__
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    assert any("Trajectory write failed" in m for m in warnings_emitted)


# ---------------------------------------------------------------------------
# Recorder-level Behavior D: FinalMetrics totals are sum of per-step Metrics
# ---------------------------------------------------------------------------


async def test_final_metrics_totals_match_per_step_sum(tmp_path: Path) -> None:
    """Behavior D: FinalMetrics totals equal the sum of MetricsEvent values."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="step-one-text"))
            inv.observe(_StubMetricsEvent(
                message_id="m-1", prompt_tokens=100, completion_tokens=20,
                cached_tokens=10, cost_usd=0.001,
            ))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv2:
            inv2.observe(TextEvent(text="step-two-text"))
            inv2.observe(_StubMetricsEvent(
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


# ---------------------------------------------------------------------------
# Recorder-level Behavior E: ContextVar is set inside, cleared after
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Recorder-level Behavior F (CORE-08): Trajectory.agent identity baked in
# ---------------------------------------------------------------------------


async def test_trajectory_agent_identity_is_daydream(tmp_path: Path) -> None:
    """Behavior F (CORE-08): agent.name='daydream', version non-empty, model_name passed-in."""
    recorder = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
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


# ---------------------------------------------------------------------------
# Bonus: schema_version + session_id present and well-formed
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sanity: now_iso, Redactor, Invocation public surface
# ---------------------------------------------------------------------------


def test_now_iso_ends_with_z() -> None:
    """now_iso returns ISO 8601 string ending in 'Z' (Pitfall 2)."""
    ts = now_iso()
    assert ts.endswith("Z")


def test_redactor_is_passthrough() -> None:
    """Redactor.redact_step returns the input unchanged (D-12 no-op)."""
    from daydream.atif import Step as AtifStep

    step = AtifStep(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message="hello",
    )
    out = Redactor().redact_step(step)
    assert out is step


def test_invocation_has_no_parent_field() -> None:
    """D-08: Invocation does not carry parent; parent linkage is on TrajectoryRecorder."""
    fields = {f.name for f in Invocation.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    assert "parent" not in fields, (
        f"Invocation must not carry a parent field in Phase 2 (D-08); got {fields}"
    )


# ---------------------------------------------------------------------------
# No-recorder no-op (CORE-09)
# ---------------------------------------------------------------------------


def test_no_recorder_no_op_get_current_returns_none() -> None:
    """CORE-09: outside any recorder context, get_current_recorder is None."""
    assert get_current_recorder() is None
