"""Recorded usage across sequential calls and multi-turn event streams.

Authoritative per-call totals must survive collapsed per-message usage;
repeated totals must not double-count tokens or cost."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.atif import validate as atif_validate
from daydream.backends import (
    AgentEvent,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    TurnEndEvent,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.trajectory import make_recorder, read_trajectory, step_token_sum

# -- Per-phase session totals (one run_agent call per phase) -----------------
# Claude-shaped stream: near-constant single-digit completion_tokens per
# message, with the authoritative whole-call session total on the CostEvent.
_PHASE_SESSION_TOTALS: list[int] = [66_737, 60_000, 55_000]
PHASES = [DaydreamPhase.REVIEW, DaydreamPhase.FIX, DaydreamPhase.TEST]


def _make_backend(turn_idx: int) -> ScriptedBackend:
    """Claude-shaped usage: single-digit per-message completion followed by
    an authoritative per-call CostEvent, in real SDK emission order."""
    return ScriptedBackend(events=[TextEvent(text=f"turn {turn_idx + 1} output"),
            MetricsEvent(message_id=f"msg_{turn_idx:02d}", prompt_tokens=[100, 150, 200][turn_idx],
                completion_tokens=12,   # near-constant single digit (SDK bug shape)
                cached_tokens=None, cost_usd=None,
            ), TurnEndEvent(message_id=f"msg_{turn_idx:02d}"), CostEvent(cost_usd=0.5, input_tokens=600,
                      output_tokens=_PHASE_SESSION_TOTALS[turn_idx], cached_tokens=None),
            ResultEvent(structured_output=None, continuation=None),
        ], model="mock-model",
    )


async def _run_three_turns(tmp_path: Path) -> dict[str, Any]:
    """Drive 3 sequential run_agent() calls, return the trajectory dict."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        for i in range(3):
            backend = _make_backend(i)
            await run_agent(backend, tmp_path, f"prompt {i + 1}", phase=PHASES[i])
    return read_trajectory(recorder.path)

async def test_per_call_token_values_not_cumulative(tmp_path: Path) -> None:
    traj = await _run_three_turns(tmp_path)
    assert atif_validate(traj) is True

    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]

    # Per-call values -- NOT cumulative. Read off the per-turn steps only
    # (completion == the near-constant single digit 12 distinguishes them from
    # the residual CostEvent steps; do not index agent_steps[i], the residual
    # steps shift indices).
    turn_steps = [s for s in agent_steps
        if s.get("metrics")
        and s["metrics"].get("prompt_tokens")
        and s["metrics"].get("completion_tokens") == 12
    ]
    assert [s["metrics"]["prompt_tokens"] for s in turn_steps] == [100, 150, 200]
    assert [step["extra"]["daydream_phase"] for step in agent_steps] == [
        "review", "review", "fix", "fix", "test", "test",
    ]
    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == step_token_sum(traj, "prompt_tokens") == 1800
    assert final["total_completion_tokens"] == step_token_sum(traj, "completion_tokens") == 181_737
    assert final["total_cost_usd"] == pytest.approx(1.5)


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
