"""Correct completion-token accounting & real turn granularity (issue #747).

Fix A: the recorder reconciles each CostEvent's totals against the
invocation's per-message sum (per-dimension take-max delta), so the recorded
final metrics reflect the authoritative session value instead of the
collapsed per-message single digits, and fresh CostEvent steps carry only the
residual so ``Σ steps == final`` holds.

Fix B: run_agent's normal loop forwards TurnEndEvent so each turn's already-
emitted MetricsEvent lands on its own Step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.trajectory import make_recorder, read_trajectory


@pytest.mark.asyncio
async def test_claude_shape_final_reflects_session_total(tmp_path):
    """Claude-shaped stream: per-message single-digit completion + authoritative
    CostEvent session total -> final.total_completion_tokens == session total,
    not the collapsed sum of per-message digits."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            for i, c in enumerate((12, 9, 11, 8, 10)):
                inv.observe(TextEvent(text=f"turn {i}"))
                inv.observe(MetricsEvent(message_id=f"m{i}", prompt_tokens=100,
                                         completion_tokens=c, cached_tokens=None,
                                         cost_usd=None))
                inv.observe(TurnEndEvent(message_id=f"m{i}"))
            inv.observe(CostEvent(cost_usd=0.5, input_tokens=600,
                                  output_tokens=66_737, cached_tokens=None))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = read_trajectory(recorder.path)
    assert traj["final_metrics"]["total_completion_tokens"] == 66_737


@pytest.mark.asyncio
async def test_claude_shape_step_sum_equals_final(tmp_path):
    """The whole-run session total is distributed so SUM(steps) == final, keeping
    the final == Σ steps invariant after per-turn steps + reconciliation."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            for i, c in enumerate((12, 9, 11, 8, 10)):
                inv.observe(TextEvent(text=f"turn {i}"))
                inv.observe(MetricsEvent(message_id=f"m{i}", prompt_tokens=100,
                                         completion_tokens=c, cached_tokens=None,
                                         cost_usd=None))
                inv.observe(TurnEndEvent(message_id=f"m{i}"))
            inv.observe(CostEvent(cost_usd=0.5, input_tokens=600,
                                  output_tokens=66_737, cached_tokens=None))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = read_trajectory(recorder.path)
    final = traj["final_metrics"]
    step_sum = sum(s["metrics"]["completion_tokens"] for s in traj["steps"]
                   if s.get("metrics") and s["metrics"].get("completion_tokens"))
    assert step_sum == 66_737
    assert final["total_completion_tokens"] == step_sum


@pytest.mark.asyncio
async def test_pi_shape_no_step_level_double_count(tmp_path):
    """Pi-shaped stream (per-turn MetricsEvent w/ cost + restated final CostEvent)
    must not double-count at the step level once per-turn steps are enabled."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            for i in range(3):
                inv.observe(TextEvent(text=f"turn {i}"))
                inv.observe(MetricsEvent(message_id="", prompt_tokens=100,
                                         completion_tokens=10, cached_tokens=None,
                                         cost_usd=0.25))
                inv.observe(TurnEndEvent(message_id=""))
            inv.observe(CostEvent(cost_usd=0.75, input_tokens=300,
                                  output_tokens=30, cached_tokens=None))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = read_trajectory(recorder.path)
    final = traj["final_metrics"]
    step_sum = sum(s["metrics"]["completion_tokens"] for s in traj["steps"]
                   if s.get("metrics") and s["metrics"].get("completion_tokens"))
    step_cost = sum(s["metrics"].get("cost_usd") or 0 for s in traj["steps"]
                    if s.get("metrics"))
    assert final["total_completion_tokens"] == 30
    assert step_sum == 30               # not 60
    assert final["total_cost_usd"] == pytest.approx(0.75)
    assert step_cost == pytest.approx(0.75)  # not 1.5
