"""Authoritative session usage must reconcile onto steps so their sum equals
final metrics. run_agent must forward turn boundaries to preserve granularity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.trajectory import (
    make_recorder,
    read_trajectory,
    step_token_sum,
)


async def _run_write_agent(tmp_path: Path, *, content: str) -> dict[str, Any]:
    """One real run_agent call with a 16 KB Write argument to exercise the token floor."""
    events: list[AgentEvent] = [TextEvent(text="turn 0"),
        ToolStartEvent(id="w1", name="Write", input={"file_path": "/tmp/f.txt", "content": content}),
        ToolResultEvent(id="w1", output="ok", is_error=False),
        MetricsEvent(message_id="m0", prompt_tokens=100, completion_tokens=10, cached_tokens=None, cost_usd=None),
        TurnEndEvent(message_id="m0"),
        CostEvent(cost_usd=0.5, input_tokens=600, output_tokens=66_737, cached_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]
    recorder = make_recorder(tmp_path)
    async with recorder:
        await run_agent(
            ScriptedBackend(events=events, model="mock-model"), tmp_path, "prompt", phase=DaydreamPhase.REVIEW
        )
    return read_trajectory(recorder.path)


def _tool_argument_floor(traj: dict[str, Any]) -> Any:
    """floor = SUM(len(json.dumps(tool_call['arguments'])) / 4) over recorded calls."""
    total = 0
    for s in traj["steps"]:
        for tc in s.get("tool_calls") or []:
            total += len(json.dumps(tc["arguments"]))
    return total // 4

@pytest.mark.asyncio
async def test_tool_argument_invariant_holds_real_path(tmp_path: Path) -> None:
    traj = await _run_write_agent(tmp_path, content="y" * 16_000)
    assert traj["final_metrics"]["total_completion_tokens"] >= _tool_argument_floor(traj)


@pytest.mark.asyncio
async def test_metrics_no_agent_step_folds_to_minted_step(tmp_path: Path) -> None:
    """With only a user/context step, mint an agent step for metrics so the
    step sum still equals the final total."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe_user_step(prompt="go")  # non-agent user step only
            # No TextEvent/TurnEnd: self.steps non-empty but no agent step.
            inv.observe(MetricsEvent(message_id="m0", prompt_tokens=100,
                                     completion_tokens=25, cached_tokens=None,
                                     cost_usd=0.01))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = read_trajectory(recorder.path)
    final = traj["final_metrics"]
    step_sum = step_token_sum(traj, "completion_tokens")
    assert final.get("total_completion_tokens") == 25
    assert step_sum == 25
