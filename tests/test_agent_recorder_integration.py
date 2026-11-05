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
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.claude import MaxTurnsError
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


async def test_unencodable_dispatch_measurement_does_not_change_agent_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Encoding failure keeps the backend outcome and never prints failed input or exception text."""
    class UnencodablePrompt(str):
        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            raise UnicodeError("PRIVATE_MEASUREMENT_FAILURE")

    backend = _scripted([TextEvent(text="done")])
    trajectory, result = await _run_with_recorder(
        backend, tmp_path, prompt=UnencodablePrompt("API_KEY=private-unencodable-credential"), phase=DaydreamPhase.FIX,
    )
    assert result == ("done", None, None)
    assert trajectory is not None
    assert "host_dispatch" not in trajectory["extra"]["subtrajectories"][0]
    assert _single_agent_step(trajectory)["message"] == "done"
    output = capsys.readouterr().out
    assert "Trajectory dispatch measurement failed: UnicodeError" in output
    assert "PRIVATE_MEASUREMENT_FAILURE" not in output
    assert "private-unencodable-credential" not in output + json.dumps(trajectory)


async def test_calling_run_agent_without_phase_raises_typeerror(tmp_path: Path) -> None:
    backend = _scripted([ TextEvent(text="ok"), ])
    with pytest.raises(TypeError) as excinfo:
        await run_agent(backend, tmp_path, "hi")  # type: ignore[call-arg]
    assert "phase" in str(excinfo.value).lower()
    assert inspect.signature(run_agent).parameters["phase"].kind is inspect.Parameter.KEYWORD_ONLY


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
