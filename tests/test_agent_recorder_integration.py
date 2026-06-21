"""Integration tests: agent.run_agent + TrajectoryRecorder (MAP-01..07).

Per D-18, tests follow schema-validity + behavior-predicate patterns. No
full-tree snapshot equality (Pitfall 11). Each test that produces a
trajectory asserts ``daydream.atif.validate(traj) is True`` plus one or
two specific behavioral predicates.

Sequencing note (Plan 02-05 / 02-06): Plan 05 introduces a required
keyword-only ``phase`` argument to ``run_agent``; Plan 06 updates every
call site to pass it. Until Plan 06 lands, the full 343-test suite is
INTENTIONALLY red (the transitional sentinel raises a clear TypeError
on un-updated call sites). These integration tests are self-contained —
they pass ``phase=DaydreamPhase.X`` directly and validate in isolation.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
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
    TrajectoryRecorder,
    _reset_recorder_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_recorder() -> Any:
    """Reset _RECORDER_VAR before and after every test (mirrors D-17).

    The autouse conftest fixture is added in Plan 07; this file-local
    fixture mirrors the pattern from tests/test_trajectory.py for now.
    """
    _reset_recorder_for_tests()
    yield
    _reset_recorder_for_tests()


@dataclass
class MockBackend:
    """Minimal Backend implementation that replays a canned event list.

    Mirrors the three-method Backend protocol surface (execute / cancel /
    format_skill_invocation) without inheriting; tests substitute this
    in place of ClaudeBackend / CodexBackend so the event sequence is
    deterministic.
    """

    model = "mock-model"
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


def _make_recorder(
    tmp_path: Path,
    *,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL,
    agent_model_name: str = "opus",
) -> TrajectoryRecorder:
    """Construct a TrajectoryRecorder rooted in tmp_path (test helper)."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=run_flow,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
        session_id="test",
    )


async def _run_with_recorder(
    backend: Backend,
    tmp_path: Path,
    *,
    phase: DaydreamPhase = DaydreamPhase.REVIEW,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL,
    prompt: str = "hello",
) -> tuple[dict[str, Any] | None, tuple[Any, Any, Any]]:
    """Drive run_agent inside a TrajectoryRecorder. Return (trajectory_dict, return_value)."""
    recorder = _make_recorder(tmp_path, run_flow=run_flow)
    target_path = recorder.path
    async with recorder:
        result = await run_agent(backend, tmp_path, prompt, phase=phase)
    if target_path.exists():
        return json.loads(target_path.read_text()), result
    return None, result


async def test_user_prompt_becomes_user_step(tmp_path: Path) -> None:
    """MAP-01 + Pitfall 4 — Beagle prompt becomes Step(source='user'); no agent-only fields."""
    backend = MockBackend([
        TextEvent(text="hello back"),
        ResultEvent(structured_output=None, continuation=None),
    ])
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
    backend = MockBackend([
        TextEvent(text="hello back"),
        ResultEvent(structured_output=None, continuation=None),
    ])
    traj, _ = await _run_with_recorder(backend, tmp_path, prompt="hi")
    assert traj is not None
    assert atif_validate(traj) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == "hello back"


async def test_tool_call_paired_with_observation_in_same_step(tmp_path: Path) -> None:
    """CORE-06 / MAP-04 / MAP-05 / Pitfall 3 — same-step pairing."""
    backend = MockBackend([
        TextEvent(text="running pytest"),
        ToolStartEvent(id="t1", name="Bash", input={"command": "pytest"}),
        ToolResultEvent(id="t1", output="OK", is_error=False),
        ResultEvent(structured_output=None, continuation=None),
    ])
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
    backend = MockBackend([
        TextEvent(text="ok"),
        MetricsEvent(
            message_id="msg_01",
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=10,
            cost_usd=0.001,
        ),
        ResultEvent(structured_output=None, continuation=None),
    ])
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
    recorder = _make_recorder(tmp_path)
    target_path = recorder.path
    backend1 = MockBackend([
        TextEvent(text="first"),
        MetricsEvent(
            message_id="msg_01",
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=5,
            cost_usd=0.001,
        ),
        ResultEvent(structured_output=None, continuation=None),
    ])
    backend2 = MockBackend([
        TextEvent(text="second"),
        MetricsEvent(
            message_id="msg_02",
            prompt_tokens=200,
            completion_tokens=40,
            cached_tokens=15,
            cost_usd=0.002,
        ),
        ResultEvent(structured_output=None, continuation=None),
    ])
    async with recorder:
        await run_agent(backend1, tmp_path, "first prompt", phase=DaydreamPhase.REVIEW)
        await run_agent(backend2, tmp_path, "second prompt", phase=DaydreamPhase.FIX)

    assert target_path.exists()
    traj = json.loads(target_path.read_text())
    assert atif_validate(traj) is True

    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    sum_prompt = sum(s["metrics"]["prompt_tokens"] for s in agent_steps if s.get("metrics"))
    sum_completion = sum(
        s["metrics"]["completion_tokens"] for s in agent_steps if s.get("metrics")
    )
    sum_cached = sum(s["metrics"]["cached_tokens"] for s in agent_steps if s.get("metrics"))
    sum_cost = sum(s["metrics"]["cost_usd"] for s in agent_steps if s.get("metrics"))

    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == sum_prompt == 300
    assert final["total_completion_tokens"] == sum_completion == 60
    assert final["total_cached_tokens"] == sum_cached == 20
    assert final["total_cost_usd"] == pytest.approx(sum_cost) == pytest.approx(0.003)


async def test_no_recorder_is_clean_no_op(tmp_path: Path) -> None:
    """CORE-09 — run_agent without active recorder runs cleanly."""
    backend = MockBackend([
        TextEvent(text="ok"),
        ResultEvent(structured_output=None, continuation=None),
    ])
    # NO TrajectoryRecorder context — recorder is None.
    out, cont, _ = await run_agent(backend, tmp_path, "hi", phase=DaydreamPhase.REVIEW)
    assert isinstance(out, str)
    assert "ok" in out
    assert cont is None
    # No trajectory.json should be written when no recorder is active.
    assert not (tmp_path / ".daydream" / "trajectory.json").exists()


async def test_extra_phase_and_run_flow_labels(tmp_path: Path) -> None:
    """MAP-08 + MAP-09 — every Step has both extra labels."""
    backend = MockBackend([
        TextEvent(text="ok"),
        ResultEvent(structured_output=None, continuation=None),
    ])
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
    recorder = _make_recorder(tmp_path, run_flow=DaydreamRunFlow.PR)
    target_path = recorder.path
    backend1 = MockBackend([
        TextEvent(text="reviewing"),
        ResultEvent(structured_output=None, continuation=None),
    ])
    backend2 = MockBackend([
        TextEvent(text="fixing"),
        ResultEvent(structured_output=None, continuation=None),
    ])
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
    """Signature change (D-05) — phase is keyword-only.

    Plan 05 introduces the keyword-only ``phase`` argument. Plan 07
    re-tightens to a strict required-no-default; through Plans 05–06 a
    transitional sentinel default keeps the suite recoverable. This test
    enforces the keyword-only kind, which is the part that survives the
    transition.
    """
    sig = inspect.signature(run_agent)
    assert "phase" in sig.parameters
    assert sig.parameters["phase"].kind == inspect.Parameter.KEYWORD_ONLY


async def test_calling_run_agent_without_phase_raises_typeerror(tmp_path: Path) -> None:
    """D-05 transitional — missing phase raises TypeError with a clear message."""
    backend = MockBackend([
        TextEvent(text="ok"),
        ResultEvent(structured_output=None, continuation=None),
    ])
    with pytest.raises(TypeError) as excinfo:
        # Intentionally call without the required `phase` kwarg.
        await run_agent(backend, tmp_path, "hi")  # type: ignore[call-arg]
    # Error message should reference the phase argument.
    assert "phase" in str(excinfo.value).lower()


async def test_thinking_event_routes_to_agent_step(tmp_path: Path) -> None:
    """MAP-03 — ThinkingEvent populates Step.reasoning_content."""
    backend = MockBackend([
        ThinkingEvent(text="let me think..."),
        TextEvent(text="answer"),
        ResultEvent(structured_output=None, continuation=None),
    ])
    traj, _ = await _run_with_recorder(backend, tmp_path)
    assert traj is not None
    assert atif_validate(traj) is True
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["reasoning_content"] == "let me think..."
    assert agent_steps[0]["message"] == "answer"


async def test_cost_event_does_not_break_recording(tmp_path: Path) -> None:
    """CostEvent is observed but does not produce a per-step Metrics (D-14)."""
    backend = MockBackend([
        TextEvent(text="ok"),
        CostEvent(cost_usd=0.005, input_tokens=50, output_tokens=10, cached_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ])
    traj, _ = await _run_with_recorder(backend, tmp_path)
    assert traj is not None
    assert atif_validate(traj) is True
    # Phase 2 prefers MetricsEvent for per-step Metrics; CostEvent path only
    # contributes to FinalMetrics in later phases (D-14). For now we just
    # assert the trajectory remains schema-valid with CostEvent in the stream.
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
