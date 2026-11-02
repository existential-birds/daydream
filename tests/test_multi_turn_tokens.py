"""Recorded usage across sequential calls and multi-turn event streams.

Authoritative per-call totals must survive collapsed per-message usage;
repeated totals must not double-count tokens or cost."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.trajectory import make_recorder, read_trajectory, step_token_sum


async def _drive_one(tmp_path: Path, backend: Any) -> dict[str, Any]:
    """Drive a single run_agent() call through a real recorder."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        await run_agent(backend, tmp_path, "prompt", phase=DaydreamPhase.REVIEW)
    return read_trajectory(recorder.path)


def _metrics_only_backend(*, turns: int, in_tok: int, out_tok: int) -> ScriptedBackend:
    """One interim update omits optional cache usage; later totals retain the earlier count."""
    turn: list[AgentEvent | BaseException] = []
    for i in range(turns):
        turn += [TextEvent(text=f"turn {i + 1}"),
            MetricsEvent(
                message_id=f"m-{i}", prompt_tokens=in_tok, completion_tokens=out_tok,
                cached_tokens=(2, None, 4)[i], cost_usd=0.01,
            ),
        ]
    turn.append(ResultEvent(structured_output=None, continuation=None))
    return ScriptedBackend(events=turn, model="mock-model")

async def test_step_metrics_accumulate_across_turns(tmp_path: Path) -> None:
    traj = await _drive_one(tmp_path, _metrics_only_backend(turns=3, in_tok=100, out_tok=10))

    agent_metrics = [s["metrics"] for s in traj["steps"] if s.get("metrics")]
    assert agent_metrics[-1]["prompt_tokens"] == 300  # Σ turns, not 100
    assert agent_metrics[-1]["completion_tokens"] == 30
    assert agent_metrics[-1]["cached_tokens"] == 6
    assert agent_metrics[-1]["cost_usd"] == pytest.approx(0.03)
    assert traj["final_metrics"]["total_prompt_tokens"] == step_token_sum(traj, "prompt_tokens") == 300
