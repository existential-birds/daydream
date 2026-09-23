"""TEST-06: Empirical multi-turn fixture verifying session total reconciliation.

Drives 3 sequential run_agent() calls through a ScriptedBackend with known token
values. Asserts the recorded final metrics reflect the authoritative per-call
session totals (reconciled via per-dimension take-max delta, issue #747), NOT
the collapsed per-message single digits — the under-count is caught, not
blessed. This is a gate test -- it passes or fails. No conditional
delta-subtraction logic.
"""

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
    """Claude-shaped mock: completion is a near-constant single digit per
    message (SDK bug shape); the authoritative whole-call session total rides
    the per-call CostEvent (mirrors the real claude-agent-sdk emission order:
    MetricsEvent per message, then CostEvent)."""
    return ScriptedBackend(
        events=[
            TextEvent(text=f"turn {turn_idx + 1} output"),
            MetricsEvent(
                message_id=f"msg_{turn_idx:02d}",
                prompt_tokens=[100, 150, 200][turn_idx],
                completion_tokens=12,   # near-constant single digit (SDK bug shape)
                cached_tokens=None,
                cost_usd=None,
            ),
            TurnEndEvent(message_id=f"msg_{turn_idx:02d}"),
            CostEvent(cost_usd=0.5, input_tokens=600,
                      output_tokens=_PHASE_SESSION_TOTALS[turn_idx], cached_tokens=None),
            ResultEvent(structured_output=None, continuation=None),
        ],
        model="mock-model",
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
    """SDK #112 gate: per-turn step prompt_tokens matches the per-call value,
    not cumulative (100, 250, 450)."""
    traj = await _run_three_turns(tmp_path)
    assert atif_validate(traj) is True

    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]

    # Per-call values -- NOT cumulative. Read off the per-turn steps only
    # (completion == the near-constant single digit 12 distinguishes them from
    # the residual CostEvent steps; do not index agent_steps[i], the residual
    # steps shift indices).
    turn_steps = [
        s for s in agent_steps
        if s.get("metrics")
        and s["metrics"].get("prompt_tokens")
        and s["metrics"].get("completion_tokens") == 12
    ]
    assert [s["metrics"]["prompt_tokens"] for s in turn_steps] == [100, 150, 200]


async def test_final_metrics_sum_matches_per_step_totals(tmp_path: Path) -> None:
    """FinalMetrics totals are the sum of per-step values across all 3 phases,
    matching the reconciled session totals (not the collapsed single digits)."""
    traj = await _run_three_turns(tmp_path)
    assert atif_validate(traj) is True

    final = traj["final_metrics"]
    assert final["total_completion_tokens"] == sum(_PHASE_SESSION_TOTALS)  # 181_737
    step_sum = step_token_sum(traj, "completion_tokens")
    assert final["total_completion_tokens"] == step_sum
    # Prompt: per-call authoritative CostEvent total (600) per phase.
    assert final["total_prompt_tokens"] == 600 * 3  # 1800
    # Cost: one CostEvent per phase.
    assert final["total_cost_usd"] == pytest.approx(0.5 * 3)  # 1.5


async def test_each_step_carries_correct_phase_label(tmp_path: Path) -> None:
    """Each agent step's extra.daydream_phase matches the phase passed to run_agent()."""
    traj = await _run_three_turns(tmp_path)
    assert atif_validate(traj) is True

    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    # 2 metric-bearing steps per phase (per-turn step + residual CostEvent step).
    phases = [s["extra"]["daydream_phase"] for s in agent_steps]
    assert phases == ["review", "review", "fix", "fix", "test", "test"]


# -- CostEvent must not re-count what MetricsEvents already reported ---------


def _codex_shaped_backend(*, turns: int, in_tok: int, out_tok: int, cost: float) -> ScriptedBackend:
    """Codex-shaped mock: per turn a MetricsEvent AND a CostEvent restating it.

    ``cost`` is the whole-invocation cost, split evenly across the per-turn
    CostEvents; the MetricsEvents carry no cost (codex's synth cost is the
    same value on both events, so this pins the tokens-only re-count).
    """
    turn: list[AgentEvent | BaseException] = []
    for i in range(turns):
        turn += [
            TextEvent(text=f"turn {i + 1}"),
            MetricsEvent(
                message_id="",
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                cached_tokens=None,
                cost_usd=None,
            ),
            CostEvent(
                cost_usd=cost / turns,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cached_tokens=None,
            ),
        ]
    turn.append(ResultEvent(structured_output=None, continuation=None))
    return ScriptedBackend(events=turn, model="mock-model")


def _pi_shaped_backend(*, turns: int, in_tok: int, out_tok: int, cost_per_turn: float) -> ScriptedBackend:
    """Pi-shaped mock: per-turn MetricsEvents WITH cost + a final CostEvent
    re-emitting the summed totals."""
    turn: list[AgentEvent | BaseException] = []
    for i in range(turns):
        turn += [
            TextEvent(text=f"turn {i + 1}"),
            MetricsEvent(
                message_id="",
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                cached_tokens=None,
                cost_usd=cost_per_turn,
            ),
        ]
    turn += [
        CostEvent(
            cost_usd=cost_per_turn * turns,
            input_tokens=in_tok * turns,
            output_tokens=out_tok * turns,
            cached_tokens=None,
        ),
        ResultEvent(structured_output=None, continuation=None),
    ]
    return ScriptedBackend(events=turn, model="mock-model")


async def _drive_one(tmp_path: Path, backend: Any) -> dict[str, Any]:
    """Drive a single run_agent() call through a real recorder."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        await run_agent(backend, tmp_path, "prompt", phase=DaydreamPhase.REVIEW)
    return read_trajectory(recorder.path)


async def test_cost_event_does_not_double_count(tmp_path: Path) -> None:
    """Codex shape: per-turn CostEvents restate the MetricsEvent tokens."""
    traj = await _drive_one(
        tmp_path, _codex_shaped_backend(turns=2, in_tok=100, out_tok=10, cost=0.5)
    )
    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == 200  # not 400
    assert final["total_completion_tokens"] == 20  # not 40
    # MetricsEvents carried no cost, so the CostEvents' cost counts exactly once.
    assert final["total_cost_usd"] == pytest.approx(0.5)


async def test_pi_shape_final_cost_event_does_not_double_count(tmp_path: Path) -> None:
    """Pi shape: per-turn MetricsEvents carry cost; the final CostEvent restates
    the summed totals and must contribute nothing."""
    traj = await _drive_one(
        tmp_path, _pi_shaped_backend(turns=3, in_tok=100, out_tok=10, cost_per_turn=0.25)
    )
    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == 300  # not 600
    assert final["total_completion_tokens"] == 30  # not 60
    assert final["total_cost_usd"] == pytest.approx(0.75)  # not 1.5


async def test_cost_event_only_backend_still_accumulates(tmp_path: Path) -> None:
    """A backend that emits no MetricsEvent at all keeps full CostEvent accumulation."""
    backend = ScriptedBackend(
        events=[
            TextEvent(text="only turn"),
            CostEvent(cost_usd=0.4, input_tokens=70, output_tokens=7, cached_tokens=3),
            ResultEvent(structured_output=None, continuation=None),
        ],
        model="mock-model",
    )

    traj = await _drive_one(tmp_path, backend)
    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == 70
    assert final["total_completion_tokens"] == 7
    assert final["total_cached_tokens"] == 3
    assert final["total_cost_usd"] == pytest.approx(0.4)


def _metrics_only_backend(*, turns: int, in_tok: int, out_tok: int) -> ScriptedBackend:
    """Emits N turns of MetricsEvents with no TurnEndEvent, so they all land on
    a single Step (``run_agent``'s normal loop does not forward TurnEndEvent)."""
    turn: list[AgentEvent | BaseException] = []
    for i in range(turns):
        turn += [
            TextEvent(text=f"turn {i + 1}"),
            MetricsEvent(
                message_id=f"m-{i}",
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                cached_tokens=2,
                cost_usd=0.01,
            ),
        ]
    turn.append(ResultEvent(structured_output=None, continuation=None))
    return ScriptedBackend(events=turn, model="mock-model")


async def test_step_metrics_accumulate_across_turns(tmp_path: Path) -> None:
    """A Step spanning 3 turns carries their sum, not the last turn's snapshot."""
    traj = await _drive_one(tmp_path, _metrics_only_backend(turns=3, in_tok=100, out_tok=10))

    agent_metrics = [s["metrics"] for s in traj["steps"] if s.get("metrics")]
    assert agent_metrics[-1]["prompt_tokens"] == 300  # Σ turns, not 100
    assert agent_metrics[-1]["completion_tokens"] == 30
    assert agent_metrics[-1]["cached_tokens"] == 6
    assert agent_metrics[-1]["cost_usd"] == pytest.approx(0.03)


async def test_step_metrics_sum_equals_final_metrics(tmp_path: Path) -> None:
    """The ``final == Σ steps`` invariant holds once steps accumulate."""
    traj = await _drive_one(tmp_path, _metrics_only_backend(turns=3, in_tok=100, out_tok=10))

    step_sum = step_token_sum(traj, "prompt_tokens")
    assert traj["final_metrics"]["total_prompt_tokens"] == step_sum == 300


async def test_turn_end_event_still_splits_steps(tmp_path: Path) -> None:
    """A TurnEndEvent still closes the Step, so per-turn metrics stay separate.

    Driven at the ``Invocation.observe`` seam, not through ``run_agent``:
    run_agent's normal loop deliberately does not forward TurnEndEvent, so the
    splitting behavior is only observable here.
    """
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            for i in range(2):
                inv.observe(TextEvent(text=f"turn {i + 1}"))
                inv.observe(MetricsEvent(
                    message_id=f"m-{i}", prompt_tokens=100, completion_tokens=10,
                    cached_tokens=None, cost_usd=None,
                ))
                inv.observe(TurnEndEvent(message_id=f"m-{i}"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    traj = read_trajectory(recorder.path)
    agent_metrics = [s["metrics"] for s in traj["steps"] if s.get("metrics")]
    assert [m["prompt_tokens"] for m in agent_metrics] == [100, 100]
    assert traj["final_metrics"]["total_prompt_tokens"] == 200
