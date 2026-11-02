"""Authoritative session usage must reconcile onto steps so their sum equals
final metrics. run_agent must forward turn boundaries to preserve granularity."""

from __future__ import annotations

import json
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

async def test_authoritative_usage_reconciles_onto_steps(tmp_path: Path) -> None:
    """Session totals reconcile per call, never cumulatively, and step sums equal final metrics."""
    traj = await _run_write_agent(tmp_path / "write", content="y" * 16_000)
    assert traj["final_metrics"]["total_completion_tokens"] >= _tool_argument_floor(traj)
    # With only a user/context step, mint an agent step for metrics.
    recorder = make_recorder(tmp_path / "minted")
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
    traj = await _run_three_turns(tmp_path / "phases")
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
