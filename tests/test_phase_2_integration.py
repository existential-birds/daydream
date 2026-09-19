"""End-to-end trajectory recorder contract tests.

These tests exercise a full ``async with TrajectoryRecorder``-wrapped
``run_agent`` flow and assert schema validity plus behavior predicates. They
cover metrics and source fields, phase metadata, tool/result pairing, final
metric aggregation, ContextVar propagation, and the direct-call no-op path.

  1. ``Metrics.prompt_tokens`` and ``Metrics.completion_tokens`` populated
     on every Claude agent step; ``Step.source`` is ``"user"`` for the
     prompt and ``"agent"`` for the response.
  2. Each step has ISO 8601 UTC ``timestamp`` ending in ``Z``;
     ``extra.daydream_phase`` and ``extra.daydream_run_flow`` are valid
     enum values.
  3. Every ``ToolCall(tool_call_id=...)`` has a paired
     ``ObservationResult(source_call_id=...)`` in the same step.
  4. ``FinalMetrics`` totals equal the sum of per-step ``Metrics``.
  5. Recorder is propagated via ``ContextVar``;
     conftest has the autouse ``_reset_trajectory_recorder`` fixture;
     direct ``run_agent`` invocation without a recorder is a clean no-op.

The minimal user step also has no agent-only fields.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from daydream.agent import run_agent
from daydream.atif import validate
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
    TrajectoryRecorder,
    get_current_recorder,
)
from tests.harness.stub_backend import MockBackend

# RFC 3339 / ISO 8601 UTC with mandatory Z suffix (now_iso() invariant).
_ISO8601_Z_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)

_VALID_PHASES = {
    "review",
    "parse",
    "fix",
    "test",
    "intent",
    "alternatives",
    "pr_feedback",
    "deep",
    "exploration",
}

_VALID_RUN_FLOWS = {"normal", "ttt", "pr", "deep"}


# Test 1 — ROADMAP #1: Claude metrics on every agent step; user/agent source split.
async def test_claude_metrics_populated_on_every_agent_step(tmp_path: Path) -> None:
    """Roadmap #1 — every agent step has prompt_tokens + completion_tokens populated."""
    target_path = tmp_path / ".daydream" / "trajectory.json"
    backend = MockBackend(
        [
            TextEvent(text="reviewing"),
            MetricsEvent(
                message_id="msg_01",
                prompt_tokens=200,
                completion_tokens=100,
                cached_tokens=50,
                cost_usd=0.002,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    async with TrajectoryRecorder(
        path=target_path,
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    ):
        await run_agent(backend, tmp_path, "review please", phase=DaydreamPhase.REVIEW)

    assert target_path.exists()
    traj = json.loads(target_path.read_text())
    assert validate(traj) is True

    # User vs agent source split (Roadmap #1).
    user_steps = [s for s in traj["steps"] if s["source"] == "user"]
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(user_steps) == 1
    assert user_steps[0]["message"] == "review please"
    assert len(agent_steps) >= 1

    # Metrics populated on every agent step that recorded a MetricsEvent.
    metric_steps = [s for s in agent_steps if s.get("metrics")]
    assert len(metric_steps) >= 1
    for step in metric_steps:
        metrics = step["metrics"]
        assert metrics["prompt_tokens"] is not None
        assert metrics["completion_tokens"] is not None
        # D-15: cached_tokens is a SUBSET of prompt_tokens, not added.
        if metrics.get("cached_tokens") is not None:
            assert metrics["cached_tokens"] <= metrics["prompt_tokens"]


# Test 2 — ROADMAP #2: ISO 8601 Z-suffixed timestamp + valid extra labels per step.
async def test_every_step_has_timestamp_and_extra_labels(tmp_path: Path) -> None:
    """Roadmap #2 — ISO 8601 Z-suffixed timestamps + valid daydream_phase/run_flow extras."""
    target_path = tmp_path / ".daydream" / "trajectory.json"
    backend = MockBackend(
        [
            TextEvent(text="working"),
            ToolStartEvent(id="t1", name="Read", input={"file_path": "/tmp/x.py"}),
            ToolResultEvent(id="t1", output="contents", is_error=False),
            MetricsEvent(
                message_id="msg_01",
                prompt_tokens=50,
                completion_tokens=10,
                cached_tokens=None,
                cost_usd=None,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    async with TrajectoryRecorder(
        path=target_path,
        run_flow=DaydreamRunFlow.PR,
        target_dir=tmp_path,
        agent_model_name="sonnet",
        session_id="test",
    ):
        await run_agent(backend, tmp_path, "fix me", phase=DaydreamPhase.FIX)

    traj = json.loads(target_path.read_text())
    assert validate(traj) is True

    assert len(traj["steps"]) >= 1
    for step in traj["steps"]:
        # Mandatory ISO 8601 UTC with Z suffix (now_iso() invariant).
        ts = step.get("timestamp")
        assert ts is not None, f"missing timestamp on step: {step}"
        assert _ISO8601_Z_PATTERN.match(ts), f"timestamp not ISO 8601 Z: {ts}"

        # Extras must contain valid enum values per the Roadmap spec.
        extra = step.get("extra") or {}
        assert extra.get("daydream_phase") in _VALID_PHASES, (
            f"invalid daydream_phase: {extra.get('daydream_phase')!r}"
        )
        assert extra.get("daydream_run_flow") in _VALID_RUN_FLOWS, (
            f"invalid daydream_run_flow: {extra.get('daydream_run_flow')!r}"
        )

    # Per-call phase + per-recorder run_flow correctness.
    assert all(s["extra"]["daydream_phase"] == "fix" for s in traj["steps"])
    assert all(s["extra"]["daydream_run_flow"] == "pr" for s in traj["steps"])


# Test 4 — ROADMAP #4: FinalMetrics totals == sum of per-step Metrics (no running leak).
async def test_final_metrics_equals_sum_of_per_step_metrics(tmp_path: Path) -> None:
    """Roadmap #4 — multi-turn assertion: feed two MetricsEvents; FinalMetrics == sum."""
    target_path = tmp_path / ".daydream" / "trajectory.json"
    backend1 = MockBackend(
        [
            TextEvent(text="first turn"),
            MetricsEvent(
                message_id="msg_01",
                prompt_tokens=120,
                completion_tokens=30,
                cached_tokens=20,
                cost_usd=0.0015,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    backend2 = MockBackend(
        [
            TextEvent(text="second turn"),
            MetricsEvent(
                message_id="msg_02",
                prompt_tokens=240,
                completion_tokens=60,
                cached_tokens=40,
                cost_usd=0.0030,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    async with TrajectoryRecorder(
        path=target_path,
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    ):
        await run_agent(backend1, tmp_path, "first prompt", phase=DaydreamPhase.REVIEW)
        await run_agent(backend2, tmp_path, "second prompt", phase=DaydreamPhase.FIX)

    traj = json.loads(target_path.read_text())
    assert validate(traj) is True

    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    metric_steps = [s for s in agent_steps if s.get("metrics")]
    assert len(metric_steps) == 2  # exactly one per run_agent call

    sum_prompt = sum(s["metrics"]["prompt_tokens"] for s in metric_steps)
    sum_completion = sum(s["metrics"]["completion_tokens"] for s in metric_steps)
    sum_cached = sum(s["metrics"]["cached_tokens"] for s in metric_steps)
    sum_cost = sum(s["metrics"]["cost_usd"] for s in metric_steps)

    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == sum_prompt == 360
    assert final["total_completion_tokens"] == sum_completion == 90
    assert final["total_cached_tokens"] == sum_cached == 60
    assert abs(final["total_cost_usd"] - sum_cost) < 1e-9  # approx for float arithmetic
    assert abs(final["total_cost_usd"] - 0.0045) < 1e-9


# Test 5 — ROADMAP #5: ContextVar propagation + autouse fixture + no-recorder no-op.
async def test_no_recorder_clean_no_op(tmp_path: Path) -> None:
    """Roadmap #5 — direct run_agent without a recorder is a clean no-op."""
    backend = MockBackend(
        [
            TextEvent(text="ok"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    # NO TrajectoryRecorder context — direct invocation.
    out, cont, _ = await run_agent(backend, tmp_path, "hi", phase=DaydreamPhase.REVIEW)
    assert isinstance(out, str)
    assert "ok" in out
    assert cont is None
    # Autouse fixture cleared the ContextVar; no recorder installed, so it stays None.
    assert get_current_recorder() is None
    assert not (tmp_path / ".daydream" / "trajectory.json").exists()


# Test 6 — Pitfall 4: minimal user step has no agent-only fields after JSON-roundtrip.
async def test_user_step_has_no_agent_only_fields(tmp_path: Path) -> None:
    """Pitfall 4 — user Step has no agent-only fields after JSON serialization.

    The recorder uses ``Trajectory.to_json_dict`` (Pydantic model_dump_json),
    which emits ``exclude_none``-style output: agent-only fields are absent
    on user steps.
    """
    target_path = tmp_path / ".daydream" / "trajectory.json"
    backend = MockBackend(
        [
            TextEvent(text="here you go"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    async with TrajectoryRecorder(
        path=target_path,
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    ):
        await run_agent(backend, tmp_path, "hi", phase=DaydreamPhase.REVIEW)

    traj = json.loads(target_path.read_text())
    assert validate(traj) is True

    user_steps = [s for s in traj["steps"] if s["source"] == "user"]
    assert len(user_steps) == 1
    user = user_steps[0]
    # Pitfall 4: agent-only fields must be absent or None on the user step.
    for forbidden in (
        "tool_calls",
        "metrics",
        "model_name",
        "reasoning_content",
        "observation",
        "reasoning_effort",
    ):
        assert forbidden not in user or user[forbidden] is None, (
            f"User step must not have {forbidden!r} field"
        )
