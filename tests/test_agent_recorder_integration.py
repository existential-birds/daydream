"""Agent/recorder integration: validate persisted trajectories and observable behavior.

The public agent boundary requires an explicit keyword-only phase.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.atif import validate as atif_validate
from daydream.backends import (
    AgentEvent,
    Backend,
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
from tests.harness.backend import ScriptedBackend
from tests.harness.trajectory import make_recorder


async def _run_with_recorder(
    backend: Backend, tmp_path: Path, *, phase: DaydreamPhase = DaydreamPhase.REVIEW,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL, prompt: str = "hello",
) -> tuple[dict[str, Any] | None, tuple[Any, Any, Any]]:
    """Drive run_agent inside a TrajectoryRecorder. Return (trajectory_dict, return_value)."""
    recorder = make_recorder(tmp_path, run_flow=run_flow)
    target_path = recorder.path
    async with recorder:
        result = await run_agent(backend, tmp_path, prompt, phase=phase)
    if target_path.exists():
        return json.loads(target_path.read_text()), result
    return None, result

def _scripted(events: list[AgentEvent]) -> ScriptedBackend:
    """ScriptedBackend whose turn ends with an empty structured result."""
    return ScriptedBackend(events=[*events, ResultEvent(structured_output=None, continuation=None)], model="mock-model")

def _single_agent_step(traj: dict[str, Any] | None) -> dict[str, Any]:
    """Validate *traj* and return its sole agent step. Every recorder test that inspects one agent step asserts the
    same two invariants first: the document is schema-valid and exactly one step has ``source == "agent"``."""
    assert traj is not None
    assert atif_validate(traj) is True
    steps: list[dict[str, Any]] = traj["steps"]
    agent_steps = [s for s in steps if s["source"] == "agent"]
    assert len(agent_steps) == 1
    return agent_steps[0]

async def test_user_prompt_becomes_user_step(tmp_path: Path) -> None:
    """MAP-01 + Pitfall 4 — Beagle prompt becomes Step(source='user'); no agent-only fields."""
    backend = _scripted([ TextEvent(text="hello back"), ])
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
    assert _single_agent_step(traj)["message"] == "hello back"
    for step in traj["steps"]:
        assert step["extra"]["daydream_phase"] == "review"
        assert step["extra"]["daydream_run_flow"] == "normal"

async def test_tool_call_paired_with_observation_in_same_step(tmp_path: Path) -> None:
    backend = _scripted([
        TextEvent(text="running pytest"), ToolStartEvent(id="t1", name="Bash", input={"command": "pytest"}),
        ToolResultEvent(id="t1", output="OK", is_error=False),
    ])
    traj, _ = await _run_with_recorder(backend, tmp_path)
    step = _single_agent_step(traj)
    assert step["tool_calls"] is not None
    assert step["tool_calls"][0]["tool_call_id"] == "t1"
    assert step["observation"] is not None
    assert step["observation"]["results"][0]["source_call_id"] == "t1"

async def test_metrics_event_lands_on_agent_step(tmp_path: Path) -> None:
    """MAP-06 + D-15 — cached_tokens is subset of prompt_tokens, not added."""
    backend = _scripted([
        TextEvent(text="ok"),
        MetricsEvent(message_id="msg_01", prompt_tokens=100, completion_tokens=50, cached_tokens=10, cost_usd=0.001),
    ])
    traj, _ = await _run_with_recorder(backend, tmp_path)
    metrics = _single_agent_step(traj)["metrics"]
    assert metrics is not None
    assert metrics["prompt_tokens"] == 100  # NOT 110 — D-15 (cached is subset)
    assert metrics["cached_tokens"] == 10
    assert metrics["completion_tokens"] == 50
    assert metrics["cost_usd"] == 0.001

async def test_final_metrics_equal_sum_of_per_step_metrics(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    target_path = recorder.path
    backend1 = _scripted([
        TextEvent(text="first"),
        MetricsEvent(message_id="msg_01", prompt_tokens=100, completion_tokens=20, cached_tokens=5, cost_usd=0.001),
    ])
    backend2 = _scripted([
        TextEvent(text="second"),
        MetricsEvent(message_id="msg_02", prompt_tokens=200, completion_tokens=40, cached_tokens=15, cost_usd=0.002),
    ])
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
    backend = _scripted([ TextEvent(text="ok"), ])
    # NO TrajectoryRecorder context — recorder is None.
    out, cont, _ = await run_agent(backend, tmp_path, "hi", phase=DaydreamPhase.REVIEW)
    assert isinstance(out, str)
    assert "ok" in out
    assert cont is None
    # No trajectory.json should be written when no recorder is active.
    assert not (tmp_path / ".daydream" / "trajectory.json").exists()


async def test_extra_labels_reflect_per_call_phase_and_run_flow(tmp_path: Path) -> None:
    """MAP-08 + MAP-09 — phase varies per run_agent call; run_flow per recorder."""
    recorder = make_recorder(tmp_path, run_flow=DaydreamRunFlow.PR)
    target_path = recorder.path
    backend1 = _scripted([ TextEvent(text="reviewing"), ])
    backend2 = _scripted([ TextEvent(text="fixing"), ])
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
    sig = inspect.signature(run_agent)
    assert "phase" in sig.parameters
    assert sig.parameters["phase"].kind == inspect.Parameter.KEYWORD_ONLY

async def test_calling_run_agent_without_phase_raises_typeerror(tmp_path: Path) -> None:
    backend = _scripted([ TextEvent(text="ok"), ])
    with pytest.raises(TypeError) as excinfo:
        await run_agent(backend, tmp_path, "hi")  # type: ignore[call-arg]
    assert "phase" in str(excinfo.value).lower()

async def test_calling_run_agent_with_positional_phase_raises_typeerror(tmp_path: Path) -> None:
    backend = ScriptedBackend(events=[], model="mock-model")
    with pytest.raises(TypeError):
        await run_agent(backend, tmp_path, "hi", DaydreamPhase.REVIEW)  # type: ignore[call-arg]

async def test_thinking_event_routes_to_agent_step(tmp_path: Path) -> None:
    backend = _scripted([ ThinkingEvent(text="let me think..."), TextEvent(text="answer"), ])
    traj, _ = await _run_with_recorder(backend, tmp_path)
    step = _single_agent_step(traj)
    assert step["reasoning_content"] == "let me think..."
    assert step["message"] == "answer"

def _max_turns_backend(pre_events: list[AgentEvent]) -> ScriptedBackend:
    """Replay partial output then MaxTurnsError, matching a mid-turn SDK limit failure."""
    return ScriptedBackend(
        events=[*pre_events, MaxTurnsError("Claude agent run failed: error_max_turns", subtype="error_max_turns")],
        model="mock-model",
    )

async def test_max_turns_error_is_recorded_in_trajectory(tmp_path: Path) -> None:
    """Propagate MaxTurnsError and persist its error marker/subtype on the in-flight step."""
    recorder = make_recorder(tmp_path)
    target_path = recorder.path
    backend = _max_turns_backend([
        TextEvent(text="applying fix"), ToolStartEvent(id="t1", name="Edit", input={"path": "a.py"}),
        ToolResultEvent(id="t1", output="ok", is_error=False),
    ])
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
    backend = _scripted([
        TextEvent(text="ok"), CostEvent(cost_usd=0.005, input_tokens=50, output_tokens=10, cached_tokens=None),
    ])
    traj, _ = await _run_with_recorder(backend, tmp_path)
    metrics = _single_agent_step(traj)["metrics"]
    assert metrics is not None
    assert metrics["prompt_tokens"] == 50
    assert metrics["completion_tokens"] == 10
    assert metrics["cost_usd"] == pytest.approx(0.005)
    assert traj is not None
    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == 50
    assert final["total_completion_tokens"] == 10
    assert final["total_cost_usd"] == pytest.approx(0.005)
