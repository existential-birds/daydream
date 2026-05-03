"""TEST-06: Empirical multi-turn fixture verifying per-call token semantics.

Drives 3 sequential run_agent() calls through a MockBackend with known token
values. Asserts each agent step's Metrics.prompt_tokens matches the per-call
value (not cumulative). This is a gate test -- it passes or fails. No
conditional delta-subtraction logic.

Per Phase 2 D-14, we trust per-call semantics for claude-agent-sdk==0.1.52.
If this test fails, the token extraction in backends/claude.py needs a
last_seen_cumulative subtract step.
"""

from __future__ import annotations

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
    ContinuationToken,
    MetricsEvent,
    ResultEvent,
    TextEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    TrajectoryRecorder,
)

# -- Token value constants for the 3-turn sequence --------------------------
TURN_1 = {"prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 5, "cost_usd": 0.001}
TURN_2 = {"prompt_tokens": 150, "completion_tokens": 30, "cached_tokens": 10, "cost_usd": 0.002}
TURN_3 = {"prompt_tokens": 200, "completion_tokens": 40, "cached_tokens": 15, "cost_usd": 0.003}
TURNS = [TURN_1, TURN_2, TURN_3]
PHASES = [DaydreamPhase.REVIEW, DaydreamPhase.FIX, DaydreamPhase.TEST]


@dataclass
class MockBackend:
    """Minimal Backend replaying a canned event list."""

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


def _make_recorder(tmp_path: Path, *, agent_model_name: str = "opus") -> TrajectoryRecorder:
    """Construct a TrajectoryRecorder rooted in tmp_path."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
    )


def _read_trajectory(path: Path) -> dict[str, Any]:
    """Load the produced trajectory JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _make_backend(turn_idx: int) -> MockBackend:
    """Build a MockBackend for the given turn index (0, 1, 2)."""
    t = TURNS[turn_idx]
    return MockBackend([
        TextEvent(text=f"turn {turn_idx + 1} output"),
        MetricsEvent(
            message_id=f"msg_{turn_idx + 1:02d}",
            prompt_tokens=t["prompt_tokens"],
            completion_tokens=t["completion_tokens"],
            cached_tokens=t["cached_tokens"],
            cost_usd=t["cost_usd"],
        ),
        ResultEvent(structured_output=None, continuation=None),
    ])


async def _run_three_turns(tmp_path: Path) -> dict[str, Any]:
    """Drive 3 sequential run_agent() calls, return the trajectory dict."""
    recorder = _make_recorder(tmp_path)
    async with recorder:
        for i in range(3):
            backend = _make_backend(i)
            await run_agent(backend, tmp_path, f"prompt {i + 1}", phase=PHASES[i])
    return _read_trajectory(recorder.path)


async def test_per_call_token_values_not_cumulative(tmp_path: Path) -> None:
    """SDK #112 gate: per-step prompt_tokens matches per-call value, not cumulative."""
    traj = await _run_three_turns(tmp_path)
    assert atif_validate(traj) is True

    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 3

    # Per-call values -- NOT cumulative (100, 250, 450)
    assert agent_steps[0]["metrics"]["prompt_tokens"] == 100
    assert agent_steps[1]["metrics"]["prompt_tokens"] == 150
    assert agent_steps[2]["metrics"]["prompt_tokens"] == 200

    assert agent_steps[0]["metrics"]["completion_tokens"] == 20
    assert agent_steps[1]["metrics"]["completion_tokens"] == 30
    assert agent_steps[2]["metrics"]["completion_tokens"] == 40

    assert agent_steps[0]["metrics"]["cached_tokens"] == 5
    assert agent_steps[1]["metrics"]["cached_tokens"] == 10
    assert agent_steps[2]["metrics"]["cached_tokens"] == 15


async def test_final_metrics_sum_matches_per_step_totals(tmp_path: Path) -> None:
    """FinalMetrics totals are the sum of per-step values across all 3 turns."""
    traj = await _run_three_turns(tmp_path)
    assert atif_validate(traj) is True

    final = traj["final_metrics"]
    assert final["total_prompt_tokens"] == 100 + 150 + 200  # 450
    assert final["total_completion_tokens"] == 20 + 30 + 40  # 90
    assert final["total_cached_tokens"] == 5 + 10 + 15  # 30
    assert final["total_cost_usd"] == pytest.approx(0.001 + 0.002 + 0.003)  # 0.006


async def test_each_step_carries_correct_phase_label(tmp_path: Path) -> None:
    """Each agent step's extra.daydream_phase matches the phase passed to run_agent()."""
    traj = await _run_three_turns(tmp_path)
    assert atif_validate(traj) is True

    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 3

    assert agent_steps[0]["extra"]["daydream_phase"] == "review"
    assert agent_steps[1]["extra"]["daydream_phase"] == "fix"
    assert agent_steps[2]["extra"]["daydream_phase"] == "test"
