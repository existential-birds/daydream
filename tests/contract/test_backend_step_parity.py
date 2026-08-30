"""Parallel-implementation gate: ClaudeBackend and CodexBackend MUST produce
identical observable Step shapes for the same canonical agent script."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable

import pytest

from daydream.atif import Step
from daydream.atif import validate as atif_validate
from daydream.backends import AgentEvent, ToolResultEvent
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder

CANONICAL = Path(__file__).parent / "fixtures" / "canonical_script.json"

BackendLoader = Callable[..., AsyncIterator[AgentEvent]]


async def _run_backend_against_canonical(
    backend_loader: BackendLoader,
    tmp_path: Path,
    *,
    read_only: bool = False,
) -> list[Step]:
    script = json.loads(CANONICAL.read_text())
    recorder = TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="test-model",
        session_id="00000000-0000-0000-0000-0000000000ff",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            async for event in backend_loader(script, read_only=read_only):
                inv.observe(event)
    return [s for s in recorder.steps if s.source == "agent"]


def _compare_steps(left: list[Step], right: list[Step]) -> None:
    assert len(left) == len(right), f"step count: left={len(left)} right={len(right)}"
    for li, ri in zip(left, right, strict=True):
        assert li.message == ri.message
        assert li.reasoning_content == ri.reasoning_content
        assert (li.tool_calls or []) == (ri.tool_calls or [])
        l_obs = {
            r.source_call_id: r.content
            for r in (li.observation.results if li.observation else [])
        }
        r_obs = {
            r.source_call_id: r.content
            for r in (ri.observation.results if ri.observation else [])
        }
        assert l_obs == r_obs


@pytest.mark.asyncio
async def test_claude_and_codex_produce_identical_steps(tmp_path: Path) -> None:
    """The canonical script must produce identical selected Step fields."""
    from tests.contract._loaders import claude_loader, codex_loader

    claude_steps = await _run_backend_against_canonical(claude_loader, tmp_path / "claude")
    codex_steps = await _run_backend_against_canonical(codex_loader, tmp_path / "codex")
    _compare_steps(claude_steps, codex_steps)


@pytest.mark.asyncio
async def test_claude_and_codex_produce_identical_steps_read_only(tmp_path: Path) -> None:
    """Both backends must produce identical selected Step fields in read-only mode."""
    from tests.contract._loaders import claude_loader, codex_loader

    claude_steps = await _run_backend_against_canonical(
        claude_loader, tmp_path / "claude", read_only=True
    )
    codex_steps = await _run_backend_against_canonical(
        codex_loader, tmp_path / "codex", read_only=True
    )
    _compare_steps(claude_steps, codex_steps)


@pytest.mark.asyncio
async def test_pi_produces_expected_steps_from_canonical_fixture(tmp_path: Path) -> None:
    """Pi's normalized events and ATIF Steps match the canonical projection."""
    from tests.contract._loaders import pi_loader

    script: dict[str, Any] = json.loads(CANONICAL.read_text())
    tool_results_by_id = {result["id"]: result for result in script["tool_results"]}
    expected_steps: list[dict[str, Any]] = []
    for turn in script["turns"]:
        expected_steps.append(
            {
                "message": turn["text"],
                "reasoning_content": turn.get("thinking") or None,
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "arguments": tool_call.get("input") or {},
                    }
                    for tool_call in turn.get("tool_calls", [])
                ],
                "observations": [
                    {
                        "id": tool_call["id"],
                        "content": tool_results_by_id[tool_call["id"]].get("output", ""),
                    }
                    for tool_call in turn.get("tool_calls", [])
                    if tool_call["id"] in tool_results_by_id
                ],
            }
        )

    assert expected_steps, "canonical fixture must define expected agent turns"
    assert all(step["message"] for step in expected_steps)
    assert any(step["tool_calls"] for step in expected_steps)
    expected_tool_results = [
        (result["id"], result.get("output", ""), bool(result.get("is_error", False)))
        for result in script["tool_results"]
    ]
    assert expected_tool_results, "canonical fixture must define tool results"
    assert any(is_error for _, _, is_error in expected_tool_results)

    pi_events: list[AgentEvent] = []

    async def recording_pi_loader(
        loader_script: dict[str, Any], *, read_only: bool = False
    ) -> AsyncIterator[AgentEvent]:
        async for event in pi_loader(loader_script, read_only=read_only):
            pi_events.append(event)
            yield event

    pi_root = tmp_path / "pi"
    pi_steps = await _run_backend_against_canonical(recording_pi_loader, pi_root)

    actual_tool_results = [
        (event.id, event.output, event.is_error)
        for event in pi_events
        if isinstance(event, ToolResultEvent)
    ]
    assert actual_tool_results == expected_tool_results

    assert len(pi_steps) == len(expected_steps)
    for turn_number, (step, expected) in enumerate(zip(pi_steps, expected_steps, strict=True), 1):
        assert step.step_id == turn_number
        assert step.source == "agent"
        assert step.message == expected["message"]
        assert step.reasoning_content == expected["reasoning_content"]
        assert [
            {
                "id": tool_call.tool_call_id,
                "name": tool_call.function_name,
                "arguments": tool_call.arguments,
            }
            for tool_call in (step.tool_calls or [])
        ] == expected["tool_calls"]

        actual_observations = [
            {"id": result.source_call_id, "content": result.content}
            for result in (step.observation.results if step.observation else [])
        ]
        assert actual_observations == expected["observations"]
        assert (step.observation is not None) == bool(expected["observations"])

    trajectory_path = pi_root / "trajectory.json"
    assert trajectory_path.is_file(), "Pi recorder must write trajectory.json"
    assert atif_validate(trajectory_path, validate_images=False)


@pytest.mark.asyncio
async def test_pi_produces_identical_steps_read_only(tmp_path: Path) -> None:
    """Pi read_only=True must still match Claude's Step shape — the read_only
    tool restriction changes the CLI args, not the AgentEvent stream."""
    from tests.contract._loaders import claude_loader, pi_loader

    claude_steps = await _run_backend_against_canonical(
        claude_loader, tmp_path / "claude", read_only=True
    )
    pi_steps = await _run_backend_against_canonical(pi_loader, tmp_path / "pi", read_only=True)
    _compare_steps(claude_steps, pi_steps)
