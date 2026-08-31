"""Integration tests for ``run_agent`` and ``TrajectoryRecorder``.

Per D-18, tests follow schema-validity + behavior-predicate patterns. No
full-tree snapshot equality (Pitfall 11). Each test that produces a
trajectory asserts ``daydream.atif.validate(traj) is True`` plus one or
two specific behavioral predicates.

The tests also enforce that ``phase`` remains a required keyword-only argument
at the public boundary.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.atif import validate as atif_validate
from daydream.backends import (
    AgentEvent,
    Backend,
    ContinuationToken,
    CostEvent,
    MaxTurnsError,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
)
from tests.harness.trajectory import make_recorder


@dataclass
class MockBackend:
    """Minimal Backend implementation that replays a canned event list.

    Mirrors the Backend protocol surface (execute / cancel) without inheriting;
    tests substitute this in place of ClaudeBackend / CodexBackend so the event
    deterministic.
    """

    model = "mock-model"
    fanout_concurrency = 4
    events: list[AgentEvent]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        events = self.events

        async def _gen() -> AsyncGenerator[AgentEvent, None]:
            for event in events:
                yield event

        return _gen()

    async def cancel(self) -> None:
        return None


async def _run_with_recorder(
    backend: Backend,
    tmp_path: Path,
    *,
    phase: DaydreamPhase = DaydreamPhase.REVIEW,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL,
    prompt: str = "hello",
) -> tuple[dict[str, Any] | None, tuple[Any, Any, Any]]:
    """Drive run_agent inside a TrajectoryRecorder. Return (trajectory_dict, return_value)."""
    recorder = make_recorder(tmp_path, run_flow=run_flow)
    target_path = recorder.path
    async with recorder:
        result = await run_agent(backend, tmp_path, prompt, phase=phase)
    if target_path.exists():
        return json.loads(target_path.read_text()), result
    return None, result


async def test_user_prompt_becomes_user_step(tmp_path: Path) -> None:
    """MAP-01 + Pitfall 4 — Beagle prompt becomes Step(source='user'); no agent-only fields."""
    backend = MockBackend(
        [
            TextEvent(text="hello back"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    traj, _ = await _run_with_recorder(backend, tmp_path, prompt="hi")
    assert traj is not None
    assert atif_validate(traj) is True
    user_steps = [s for s in traj["steps"] if s["source"] == "user"]
    assert len(user_steps) == 1
    assert user_steps[0]["message"] == "hi"
    # Pitfall 4: agent-only fields must be absent on user step
    assert "tool_calls" not in user_steps[0] or user_steps[0]["tool_calls"] is None
    assert "metrics" not in user_steps[0] or user_steps[0]["metrics"] is None
    assert "model_name" not in user_steps[0] or user_steps[0]["model_name"] is None
    assert "reasoning_content" not in user_steps[0] or user_steps[0]["reasoning_content"] is None


async def test_text_event_creates_agent_step(tmp_path: Path) -> None:
    """MAP-02 — TextEvent becomes Step(source='agent', message=text)."""
    backend = MockBackend(
        [
            TextEvent(text="hello back"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    traj, _ = await _run_with_recorder(backend, tmp_path, prompt="hi")
    assert traj is not None
    assert atif_validate(traj) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == "hello back"


async def test_tool_call_paired_with_observation_in_same_step(tmp_path: Path) -> None:
    """CORE-06 / MAP-04 / MAP-05 / Pitfall 3 — same-step pairing."""
    backend = MockBackend(
        [
            TextEvent(text="running pytest"),
            ToolStartEvent(id="t1", name="Bash", input={"command": "pytest"}),
            ToolResultEvent(id="t1", output="OK", is_error=False),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    traj, _ = await _run_with_recorder(backend, tmp_path)
    assert traj is not None
    assert atif_validate(traj) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    step = agent_steps[0]
    assert step["tool_calls"] is not None
    assert step["tool_calls"][0]["tool_call_id"] == "t1"
    assert step["observation"] is not None
    assert step["observation"]["results"][0]["source_call_id"] == "t1"


async def test_metrics_event_lands_on_agent_step(tmp_path: Path) -> None:
    """MAP-06 + D-15 — cached_tokens is subset of prompt_tokens, not added."""
    backend = MockBackend(
        [
            TextEvent(text="ok"),
            MetricsEvent(
                message_id="msg_01",
                prompt_tokens=100,
                completion_tokens=50,
                cached_tokens=10,
                cost_usd=0.001,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    traj, _ = await _run_with_recorder(backend, tmp_path)
    assert traj is not None
    assert atif_validate(traj) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    metrics = agent_steps[0]["metrics"]
    assert metrics is not None
    assert metrics["prompt_tokens"] == 100  # NOT 110 — D-15 (cached is subset)
    assert metrics["cached_tokens"] == 10
    assert metrics["completion_tokens"] == 50
    assert metrics["cost_usd"] == 0.001


async def test_final_metrics_equal_sum_of_per_step_metrics(tmp_path: Path) -> None:
    """MAP-07 / Roadmap success criterion 4 — FinalMetrics totals match per-step sum."""
    recorder = make_recorder(tmp_path)
    target_path = recorder.path
    backend1 = MockBackend(
        [
            TextEvent(text="first"),
            MetricsEvent(
                message_id="msg_01",
                prompt_tokens=100,
                completion_tokens=20,
                cached_tokens=5,
                cost_usd=0.001,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    backend2 = MockBackend(
        [
            TextEvent(text="second"),
            MetricsEvent(
                message_id="msg_02",
                prompt_tokens=200,
                completion_tokens=40,
                cached_tokens=15,
                cost_usd=0.002,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    async with recorder:
        await run_agent(backend1, tmp_path, "first prompt", phase=DaydreamPhase.REVIEW)
        await run_agent(backend2, tmp_path, "second prompt", phase=DaydreamPhase.FIX)

    assert target_path.exists()
    traj = json.loads(target_path.read_text())
    assert atif_validate(traj) is True

    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    sum_prompt = sum(s["metrics"]["prompt_tokens"] for s in agent_steps if s.get("metrics"))
    sum_completion = sum(s["metrics"]["completion_tokens"] for s in agent_steps if s.get("metrics"))
    sum_cached = sum(s["metrics"]["cached_tokens"] for s in agent_steps if s.get("metrics"))
    sum_cost = sum(s["metrics"]["cost_usd"] for s in agent_steps if s.get("metrics"))

    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == sum_prompt == 300
    assert final["total_completion_tokens"] == sum_completion == 60
    assert final["total_cached_tokens"] == sum_cached == 20
    assert final["total_cost_usd"] == pytest.approx(sum_cost) == pytest.approx(0.003)


async def test_no_recorder_is_clean_no_op(tmp_path: Path) -> None:
    """CORE-09 — run_agent without active recorder runs cleanly."""
    backend = MockBackend(
        [
            TextEvent(text="ok"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    # NO TrajectoryRecorder context — recorder is None.
    out, cont, _ = await run_agent(backend, tmp_path, "hi", phase=DaydreamPhase.REVIEW)
    assert isinstance(out, str)
    assert "ok" in out
    assert cont is None
    # No trajectory.json should be written when no recorder is active.
    assert not (tmp_path / ".daydream" / "trajectory.json").exists()


async def test_extra_phase_and_run_flow_labels(tmp_path: Path) -> None:
    """MAP-08 + MAP-09 — every Step has both extra labels."""
    backend = MockBackend(
        [
            TextEvent(text="ok"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    traj, _ = await _run_with_recorder(
        backend,
        tmp_path,
        phase=DaydreamPhase.REVIEW,
        run_flow=DaydreamRunFlow.NORMAL,
    )
    assert traj is not None
    assert atif_validate(traj) is True
    for step in traj["steps"]:
        assert step["extra"]["daydream_phase"] == "review"
        assert step["extra"]["daydream_run_flow"] == "normal"


async def test_extra_labels_reflect_per_call_phase_and_run_flow(tmp_path: Path) -> None:
    """MAP-08 + MAP-09 — phase varies per run_agent call; run_flow per recorder."""
    recorder = make_recorder(tmp_path, run_flow=DaydreamRunFlow.PR)
    target_path = recorder.path
    backend1 = MockBackend(
        [
            TextEvent(text="reviewing"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    backend2 = MockBackend(
        [
            TextEvent(text="fixing"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    async with recorder:
        await run_agent(backend1, tmp_path, "review please", phase=DaydreamPhase.REVIEW)
        await run_agent(backend2, tmp_path, "fix please", phase=DaydreamPhase.FIX)

    assert target_path.exists()
    traj = json.loads(target_path.read_text())
    assert atif_validate(traj) is True

    review_steps = [s for s in traj["steps"] if s["extra"]["daydream_phase"] == "review"]
    fix_steps = [s for s in traj["steps"] if s["extra"]["daydream_phase"] == "fix"]
    assert len(review_steps) >= 1
    assert len(fix_steps) >= 1
    for step in traj["steps"]:
        # Run flow is recorder-level; same value on every step regardless of phase.
        assert step["extra"]["daydream_run_flow"] == "pr"


def test_run_agent_requires_phase_keyword() -> None:
    """The public ``phase`` argument is keyword-only."""
    sig = inspect.signature(run_agent)
    assert "phase" in sig.parameters
    assert sig.parameters["phase"].kind == inspect.Parameter.KEYWORD_ONLY


async def test_calling_run_agent_without_phase_raises_typeerror(tmp_path: Path) -> None:
    """Omitting the required ``phase`` argument raises ``TypeError``."""
    backend = MockBackend(
        [
            TextEvent(text="ok"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    with pytest.raises(TypeError) as excinfo:
        await run_agent(backend, tmp_path, "hi")  # type: ignore[call-arg]
    assert "phase" in str(excinfo.value).lower()


async def test_calling_run_agent_with_positional_phase_raises_typeerror(
    tmp_path: Path,
) -> None:
    """The required ``phase`` argument remains keyword-only."""
    backend = MockBackend([])
    with pytest.raises(TypeError):
        await run_agent(backend, tmp_path, "hi", DaydreamPhase.REVIEW)  # type: ignore[call-arg]


async def test_thinking_event_routes_to_agent_step(tmp_path: Path) -> None:
    """MAP-03 — ThinkingEvent populates Step.reasoning_content."""
    backend = MockBackend(
        [
            ThinkingEvent(text="let me think..."),
            TextEvent(text="answer"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    traj, _ = await _run_with_recorder(backend, tmp_path)
    assert traj is not None
    assert atif_validate(traj) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["reasoning_content"] == "let me think..."
    assert agent_steps[0]["message"] == "answer"


@dataclass
class MaxTurnsBackend:
    """Backend whose event stream raises MaxTurnsError mid-turn.

    Replays ``pre_events`` (an in-flight assistant turn), then raises
    ``MaxTurnsError(subtype="error_max_turns")`` — mirroring a Claude
    ``ResultMessage(is_error=True, subtype="error_max_turns")`` after the
    agent has already produced output. Exercises the realistic shape: the
    failure lands on a Step that already carries content, not an empty one.
    """

    model = "mock-model"
    fanout_concurrency = 4
    pre_events: list[AgentEvent]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        pre_events = self.pre_events

        async def _gen() -> AsyncGenerator[AgentEvent, None]:
            for event in pre_events:
                yield event
            raise MaxTurnsError("Claude agent run failed: error_max_turns", subtype="error_max_turns")

        return _gen()

    async def cancel(self) -> None:
        return None


async def test_max_turns_error_is_recorded_in_trajectory(tmp_path: Path) -> None:
    """Regression: a max-turns failure must NOT be invisible in the archive.

    Drives run_agent (the single-agent production entrypoint) with a backend
    that raises MaxTurnsError mid-stream. Asserts BOTH:
      (a) the typed MaxTurnsError propagates out of run_agent, and
      (b) the trajectory WRITTEN to disk carries an error marker + the
          ``error_max_turns`` subtype.
    Removing the __aexit__ recording step makes (b) fail.
    """
    recorder = make_recorder(tmp_path)
    target_path = recorder.path
    backend = MaxTurnsBackend(
        pre_events=[
            TextEvent(text="applying fix"),
            ToolStartEvent(id="t1", name="Edit", input={"path": "a.py"}),
            ToolResultEvent(id="t1", output="ok", is_error=False),
        ]
    )

    # (a) typed exception propagates through the production entrypoint.
    with pytest.raises(MaxTurnsError) as excinfo:
        async with recorder:
            await run_agent(backend, tmp_path, "fix this", phase=DaydreamPhase.FIX)
    assert excinfo.value.subtype == "error_max_turns"

    # (b) the emitted trajectory carries the error marker + subtype.
    assert target_path.exists()
    traj = json.loads(target_path.read_text())
    assert atif_validate(traj) is True
    errored = [s for s in traj["steps"] if s.get("extra", {}).get("error_subtype")]
    assert len(errored) == 1
    assert errored[0]["extra"]["error"] is True
    assert errored[0]["extra"]["error_subtype"] == "error_max_turns"
    # The marker lands on the in-flight Step that already held the turn's
    # output — not a synthetic empty step.
    assert errored[0]["message"] == "applying fix"
    assert errored[0]["tool_calls"][0]["tool_call_id"] == "t1"


async def test_cost_event_does_not_break_recording(tmp_path: Path) -> None:
    """CostEvent contributes its usage to the agent step and final metrics."""
    backend = MockBackend(
        [
            TextEvent(text="ok"),
            CostEvent(cost_usd=0.005, input_tokens=50, output_tokens=10, cached_tokens=None),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    traj, _ = await _run_with_recorder(backend, tmp_path)
    assert traj is not None
    assert atif_validate(traj) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    metrics = agent_steps[0]["metrics"]
    assert metrics is not None
    assert metrics["prompt_tokens"] == 50
    assert metrics["completion_tokens"] == 10
    assert metrics["cost_usd"] == pytest.approx(0.005)

    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == 50
    assert final["total_completion_tokens"] == 10
    assert final["total_cost_usd"] == pytest.approx(0.005)
