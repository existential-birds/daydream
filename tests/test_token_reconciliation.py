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
from dataclasses import dataclass
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
from daydream.trajectory import DaydreamPhase, Invocation
from tests.harness.trajectory import make_recorder, read_trajectory, step_token_sum


@dataclass
class _MockBackend:
    """Minimal Backend replaying a canned event list (mirrors the MockBackend
    in tests/test_multi_turn_tokens.py)."""

    model = "mock-model"
    fanout_concurrency = 4
    events: list[AgentEvent]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> Any:
        events = self.events

        async def _gen() -> Any:
            for event in events:
                yield event

        return _gen()

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


async def _run_write_agent(tmp_path: Path, *, content: str) -> dict[str, Any]:
    """Real-path run_agent with a Write-heavy tool call: a 16 KB content payload
    in the tool arguments (like _run_tool_agent, one turn)."""
    events: list[AgentEvent] = [
        TextEvent(text="turn 0"),
        ToolStartEvent(id="w1", name="Write",
                       input={"file_path": "/tmp/f.txt", "content": content}),
        ToolResultEvent(id="w1", output="ok", is_error=False),
        MetricsEvent(message_id="m0", prompt_tokens=100, completion_tokens=10,
                     cached_tokens=None, cost_usd=None),
        TurnEndEvent(message_id="m0"),
        CostEvent(cost_usd=0.5, input_tokens=600,
                  output_tokens=66_737, cached_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]
    recorder = make_recorder(tmp_path)
    async with recorder:
        await run_agent(
            _MockBackend(events=events), tmp_path, "prompt", phase=DaydreamPhase.REVIEW
        )
    return read_trajectory(recorder.path)


async def _run_tool_agent(
    tmp_path: Path, *, turns: int, tools_per_turn: int
) -> dict[str, Any]:
    """Real-path run_agent: per turn a TextEvent, tools_per_turn tool pairs, a
    MetricsEvent (single-digit completion), and a TurnEndEvent; then a CostEvent
    carrying the authoritative session total and a ResultEvent."""
    events: list[AgentEvent] = []
    for turn in range(turns):
        events.append(TextEvent(text=f"turn {turn}"))
        for j in range(tools_per_turn):
            events.append(ToolStartEvent(
                id=f"t{turn}-{j}", name="Bash", input={"command": "x"}
            ))
            events.append(ToolResultEvent(id=f"t{turn}-{j}", output="ok", is_error=False))
        events.append(MetricsEvent(
            message_id=f"m{turn}", prompt_tokens=100, completion_tokens=10,
            cached_tokens=None, cost_usd=None,
        ))
        events.append(TurnEndEvent(message_id=f"m{turn}"))
    events.append(CostEvent(cost_usd=0.5, input_tokens=600,
                            output_tokens=66_737, cached_tokens=None))
    events.append(ResultEvent(structured_output=None, continuation=None))
    recorder = make_recorder(tmp_path)
    async with recorder:
        await run_agent(
            _MockBackend(events=events), tmp_path, "prompt", phase=DaydreamPhase.REVIEW
        )
    return read_trajectory(recorder.path)



def _observe_claude_shape(inv: Invocation) -> None:
    """Shared Claude-shaped observe() body: 5 per-message single-digit
    MetricsEvents (one per turn) + the authoritative session-total CostEvent
    + ResultEvent. Both claude-shape tests replay this so a future token
    dimension is added in exactly one place."""
    for i, c in enumerate((12, 9, 11, 8, 10)):
        inv.observe(TextEvent(text=f"turn {i}"))
        inv.observe(MetricsEvent(message_id=f"m{i}", prompt_tokens=100,
                                 completion_tokens=c, cached_tokens=None,
                                 cost_usd=None))
        inv.observe(TurnEndEvent(message_id=f"m{i}"))
    inv.observe(CostEvent(cost_usd=0.5, input_tokens=600,
                          output_tokens=66_737, cached_tokens=None))
    inv.observe(ResultEvent(structured_output=None, continuation=None))


@pytest.mark.asyncio
async def test_claude_shape_final_reflects_session_total(tmp_path):
    """Claude-shaped stream: per-message single-digit completion + authoritative
    CostEvent session total -> final.total_completion_tokens == session total,
    not the collapsed sum of per-message digits."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_claude_shape(inv)
    traj = read_trajectory(recorder.path)
    assert traj["final_metrics"]["total_completion_tokens"] == 66_737


@pytest.mark.asyncio
async def test_claude_shape_step_sum_equals_final(tmp_path):
    """The whole-run session total is distributed so SUM(steps) == final, keeping
    the final == Σ steps invariant after per-turn steps + reconciliation."""
    recorder = make_recorder(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_claude_shape(inv)
    traj = read_trajectory(recorder.path)
    final = traj["final_metrics"]
    step_sum = step_token_sum(traj, "completion_tokens")
    assert step_sum == 66_737
    assert final["total_completion_tokens"] == step_sum


@pytest.mark.asyncio
async def test_multi_turn_turns_each_own_step(tmp_path):
    """run_agent forwards TurnEndEvent (fix B): a 24-tool-call, 3-turn agent
    records >2 steps and one metrics-bearing step per turn (not one collapsed)."""
    traj = await _run_tool_agent(tmp_path, turns=3, tools_per_turn=8)
    assert traj["final_metrics"]["total_steps"] > 2
    # One metrics-bearing step per turn: the per-message single-digit (10)
    # steps are the per-turn MetricsEvents; the CostEvent residual step
    # (completion 66707) is reconciliation, not a turn, so it is excluded.
    turn_steps = [s for s in traj["steps"]
                  if s["source"] == "agent" and s.get("metrics")
                  and s["metrics"].get("completion_tokens") == 10]
    assert len(turn_steps) == 3   # exactly one per turn, not collapsed


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
    step_sum = step_token_sum(traj, "completion_tokens")
    step_cost = sum(s["metrics"].get("cost_usd") or 0 for s in traj["steps"]
                    if s.get("metrics"))
    assert final["total_completion_tokens"] == 30
    assert step_sum == 30               # not 60
    assert final["total_cost_usd"] == pytest.approx(0.75)
    assert step_cost == pytest.approx(0.75)  # not 1.5


@pytest.mark.asyncio
async def test_16kb_write_reports_real_completion(tmp_path):
    """A Write-heavy agent (16 KB content in tool arguments) reports a real
    total_completion_tokens magnitude, not double digits (the pre-fix bug)."""
    traj = await _run_write_agent(tmp_path, content="x" * 16_000)
    completion = traj["final_metrics"]["total_completion_tokens"]
    assert completion >= 1_000   # right order of magnitude; pre-fix it was ~50


def _tool_argument_floor(traj):
    """floor = SUM(len(json.dumps(tool_call['arguments'])) / 4) over recorded calls."""
    total = 0
    for s in traj["steps"]:
        for tc in s.get("tool_calls") or []:
            total += len(json.dumps(tc["arguments"]))
    return total // 4


@pytest.mark.asyncio
async def test_tool_argument_invariant_holds_real_path(tmp_path):
    """total_completion_tokens >= floor on a real-path Write-heavy run (passes post-fix)."""
    traj = await _run_write_agent(tmp_path, content="y" * 16_000)
    assert traj["final_metrics"]["total_completion_tokens"] >= _tool_argument_floor(traj)


def test_tool_argument_invariant_fails_pre_fix_bundle():
    """The gate is non-trivial: a hand-built pre-fix bundle (collapsed total) FAILS it."""
    traj = _pre_fix_bundle()   # total_completion_tokens=50, one Write call with 16KB args
    completion = traj["final_metrics"]["total_completion_tokens"]
    assert completion < _tool_argument_floor(traj)


def _pre_fix_bundle() -> dict[str, Any]:
    """Minimal ATIF-shaped dict with a collapsed pre-fix completion total."""
    return {
        "final_metrics": {"total_completion_tokens": 50},
        "steps": [{
            "source": "agent",
            "tool_calls": [{
                "tool_call_id": "w1",
                "function_name": "Write",
                "arguments": {"file_path": "/tmp/f", "content": "z" * 16_000},
            }],
        }],
    }
