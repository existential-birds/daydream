"""Tests for run_agent wall-clock + tool-call budgets (Task 3).

Both budgets live inside run_agent: the loop is wrapped in
``anyio.move_on_after(wall_budget_s)`` and a ToolStartEvent counter trips the
tool-call ceiling. On abort, ``backend.cancel()`` is awaited (errors swallowed),
the recorder Invocation is marked aborted via ``inv.mark_aborted(reason)``, and
the partial output is returned — no exception reaches the caller.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
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
    _reset_recorder_for_tests()
    yield
    _reset_recorder_for_tests()


@dataclass
class _BurstBackend:
    """Backend that yields many ToolStartEvents and never a ResultEvent.

    Optionally sleeps between events so a wall-clock budget can trip.
    """

    model = "mock-model"
    count: int = 200
    sleep_s: float = 0.0
    cancel_calls: int = 0

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
        count = self.count
        sleep_s = self.sleep_s

        async def _gen() -> AsyncIterator[AgentEvent]:
            for i in range(count):
                if sleep_s:
                    await anyio.sleep(sleep_s)
                yield ToolStartEvent(id=f"tool-{i}", name="Bash", input={"command": "ls"})

        return _gen()

    async def cancel(self) -> None:
        self.cancel_calls += 1

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


def _make_recorder(tmp_path: Path) -> TrajectoryRecorder:
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="test",
    )


def _agent_step_with_stop_reason(traj: dict[str, Any]) -> dict[str, Any]:
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    for step in agent_steps:
        if step.get("extra", {}).get("stop_reason"):
            return step
    raise AssertionError(f"no agent step carried extra['stop_reason']: {agent_steps}")


async def test_run_agent_tool_call_ceiling(tmp_path: Path) -> None:
    """A 200-event burst with tool_call_budget=5 returns under budget, marked aborted."""
    backend = _BurstBackend(count=200, sleep_s=0.0)
    recorder = _make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            result, _ = await run_agent(
                backend,
                tmp_path,
                "go",
                phase=DaydreamPhase.FIX,
                tool_call_budget=5,
                wall_budget_s=None,
            )

    assert isinstance(result, str)
    assert backend.cancel_calls == 1
    traj = json.loads(recorder.path.read_text(encoding="utf-8"))
    step = _agent_step_with_stop_reason(traj)
    assert step["extra"]["stop_reason"] == "tool_call_budget_exceeded"


async def test_run_agent_wall_budget(tmp_path: Path) -> None:
    """A slow stream with wall_budget_s=0.2 returns, step marked wall_budget_exceeded."""
    backend = _BurstBackend(count=200, sleep_s=0.05)
    recorder = _make_recorder(tmp_path)

    with anyio.fail_after(5):
        async with recorder:
            result, _ = await run_agent(
                backend,
                tmp_path,
                "go",
                phase=DaydreamPhase.FIX,
                wall_budget_s=0.2,
                tool_call_budget=None,
            )

    assert isinstance(result, str)
    assert backend.cancel_calls == 1
    traj = json.loads(recorder.path.read_text(encoding="utf-8"))
    step = _agent_step_with_stop_reason(traj)
    assert step["extra"]["stop_reason"] == "wall_budget_exceeded"
