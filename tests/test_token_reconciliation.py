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
