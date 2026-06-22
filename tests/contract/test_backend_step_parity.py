"""Parallel-implementation gate: ClaudeBackend and CodexBackend MUST produce
identical observable Step shapes for the same canonical agent script."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Callable

import pytest

from daydream.atif import Step
from daydream.backends import AgentEvent
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder

CANONICAL = Path(__file__).parent / "fixtures" / "canonical_script.json"

BackendLoader = Callable[..., AsyncIterator[AgentEvent]]


async def _run_backend_against_canonical(
    backend_loader: BackendLoader, tmp_path: Path, *, read_only: bool = False
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
    """The canonical script must produce byte-identical message / reasoning /
    tool_calls / observation results across both backends."""
    from tests.contract._loaders import claude_loader, codex_loader

    claude_steps = await _run_backend_against_canonical(claude_loader, tmp_path / "claude")
    codex_steps = await _run_backend_against_canonical(codex_loader, tmp_path / "codex")
    _compare_steps(claude_steps, codex_steps)


@pytest.mark.asyncio
async def test_claude_and_codex_produce_identical_steps_read_only(tmp_path: Path) -> None:
    """Both backends must produce byte-identical Step shapes when read_only=True.

    Ensures the read_only kwarg added to Backend.execute() is forwarded
    consistently by both backends and does not alter the observable AgentEvent
    stream in a backend-specific way.
    """
    from tests.contract._loaders import claude_loader, codex_loader

    claude_steps = await _run_backend_against_canonical(
        claude_loader, tmp_path / "claude", read_only=True
    )
    codex_steps = await _run_backend_against_canonical(
        codex_loader, tmp_path / "codex", read_only=True
    )
    _compare_steps(claude_steps, codex_steps)


@pytest.mark.asyncio
async def test_pi_produces_identical_steps_to_claude(tmp_path: Path) -> None:
    """Pi must produce the same Step shape as Claude against the canonical
    script — the proof of ATIF trajectory parity (plan §8.2)."""
    from tests.contract._loaders import claude_loader, pi_loader

    claude_steps = await _run_backend_against_canonical(claude_loader, tmp_path / "claude")
    pi_steps = await _run_backend_against_canonical(pi_loader, tmp_path / "pi")
    _compare_steps(claude_steps, pi_steps)


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
