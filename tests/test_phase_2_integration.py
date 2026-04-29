"""Phase 2 end-to-end integration test (ROADMAP Success Criteria 1-5).

These tests exercise a full ``async with TrajectoryRecorder``-wrapped
``run_agent`` flow and assert the produced trajectory satisfies all five
Phase 2 ROADMAP success criteria. Test pattern: schema-validity +
behavior-predicate per D-18; NO full-tree snapshot equality (Pitfall 11).

Roadmap success criteria mapping
================================

  1. ``Metrics.prompt_tokens`` and ``Metrics.completion_tokens`` populated
     on every Claude agent step; ``Step.source`` is ``"user"`` for the
     prompt and ``"agent"`` for the response.
  2. Each step has ISO 8601 UTC ``timestamp`` ending in ``Z``;
     ``extra.daydream_phase`` and ``extra.daydream_run_flow`` are valid
     enum values.
  3. Every ``ToolCall(tool_call_id=...)`` has a paired
     ``ObservationResult(source_call_id=...)`` in the same step.
  4. ``FinalMetrics`` totals equal the sum of per-step ``Metrics``.
  5. Recorder is propagated via ``ContextVar`` (NOT ``AgentState``);
     conftest has the autouse ``_reset_trajectory_recorder`` fixture;
     direct ``run_agent`` invocation without a recorder is a clean no-op.

Plus Pitfall 4: minimal user step has no agent-only fields.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.agent import run_agent
from daydream.atif import validate
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
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
    "plan",
    "pr_feedback",
    "deep",
    "exploration",
}

_VALID_RUN_FLOWS = {"normal", "ttt", "pr", "deep"}


@dataclass
class MockBackend:
    """Reusable mock backend that replays a canned event list.

    Mirrors the Plan 05 ``tests/test_agent_recorder_integration.py``
    pattern (structural Backend protocol; no inheritance). Included
    here for self-containment so the integration test does not depend
    on internal helpers from another test module.
    """

    events: list[AgentEvent]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        events = self.events

        async def _gen() -> AsyncIterator[AgentEvent]:
            for event in events:
                yield event

        return _gen()

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


# -----------------------------------------------------------------------------
# Test 1 — ROADMAP Success Criterion 1
# Claude-style metrics populated on every agent step; user/agent source split.
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Test 2 — ROADMAP Success Criterion 2
# Every step has ISO 8601 timestamp ending in Z + valid extra labels.
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Test 3 — ROADMAP Success Criterion 3
# ToolCall is paired with ObservationResult in the same step (CORE-06).
# -----------------------------------------------------------------------------


async def test_tool_call_paired_with_observation_in_same_step(tmp_path: Path) -> None:
    """Roadmap #3 — ToolCall.tool_call_id == ObservationResult.source_call_id, same step."""
    target_path = tmp_path / ".daydream" / "trajectory.json"
    backend = MockBackend(
        [
            TextEvent(text="running tests"),
            ToolStartEvent(id="t1", name="Bash", input={"command": "pytest -x"}),
            ToolResultEvent(id="t1", output="all green", is_error=False),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    async with TrajectoryRecorder(
        path=target_path,
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
    ):
        await run_agent(backend, tmp_path, "test it", phase=DaydreamPhase.TEST)

    traj = json.loads(target_path.read_text())
    # The vendored validator's validate_tool_call_references check is the
    # primary guard — a dangling ToolCall or unmatched ObservationResult
    # makes this False.
    assert validate(traj) is True

    found_pair = False
    for step in traj["steps"]:
        tool_calls = step.get("tool_calls") or []
        observation = step.get("observation") or {}
        results = observation.get("results") or []
        if not tool_calls or not results:
            continue
        for tc in tool_calls:
            for r in results:
                if tc["tool_call_id"] == r["source_call_id"]:
                    found_pair = True
                    # CORE-06: pair lives in the SAME step (this loop body
                    # iterates inside one step; that is the assertion).
    assert found_pair


# -----------------------------------------------------------------------------
# Test 4 — ROADMAP Success Criterion 4
# FinalMetrics totals equal the sum of per-step Metrics (no running-totals leak).
# -----------------------------------------------------------------------------


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
    # No running-totals leak: totals MUST equal sum of per-step Metrics.
    assert final["total_prompt_tokens"] == sum_prompt == 360
    assert final["total_completion_tokens"] == sum_completion == 90
    assert final["total_cached_tokens"] == sum_cached == 60
    # cost_usd uses approx for floating-point arithmetic.
    assert abs(final["total_cost_usd"] - sum_cost) < 1e-9
    assert abs(final["total_cost_usd"] - 0.0045) < 1e-9


# -----------------------------------------------------------------------------
# Test 5 — ROADMAP Success Criterion 5
# ContextVar propagation + autouse fixture + no-recorder no-op.
# -----------------------------------------------------------------------------


async def test_no_recorder_clean_no_op(tmp_path: Path) -> None:
    """Roadmap #5 — direct run_agent without a recorder is a clean no-op."""
    backend = MockBackend(
        [
            TextEvent(text="ok"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    # NO TrajectoryRecorder context — direct invocation.
    out, cont = await run_agent(backend, tmp_path, "hi", phase=DaydreamPhase.REVIEW)
    assert isinstance(out, str)
    assert "ok" in out
    assert cont is None
    # The autouse fixture cleared the ContextVar at test entry; no recorder
    # was ever installed, so it must remain None.
    assert get_current_recorder() is None
    # And no trajectory file was written when no recorder was active.
    assert not (tmp_path / ".daydream" / "trajectory.json").exists()


def test_autouse_fixture_present() -> None:
    """Roadmap #5 — verify the suite-wide autouse fixture exists with EXACT D-17 name."""
    import tests.conftest as conftest

    assert hasattr(conftest, "_reset_trajectory_recorder"), (
        "D-17 mandates the EXACT name `_reset_trajectory_recorder` in tests/conftest.py"
    )


# -----------------------------------------------------------------------------
# Test 6 — Pitfall 4
# Minimal user step has no agent-only fields after JSON-roundtrip.
# -----------------------------------------------------------------------------


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
        # Either absent (preferred) or explicitly None — both honor Pitfall 4.
        assert forbidden not in user or user[forbidden] is None, (
            f"User step must not have {forbidden!r} field"
        )
