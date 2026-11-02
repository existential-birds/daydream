"""Tests for the flow engine (daydream/flows/engine.py)."""

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from daydream.extensions import (
    BreakLoop,
    FlowStep,
    Registry,
    Stop,
    UnresolvedExtensionError,
)
from daydream.flows.engine import FlowContext, run_flow
from daydream.run_config import RunConfig
from daydream.workspace import WorkContext


def _trace(tag: str) -> Callable[[FlowContext], Awaitable[Stop | BreakLoop | None]]:
    async def run(ctx: FlowContext) -> Stop | BreakLoop | None:
        ctx.data.setdefault("trace", []).append(tag)
        return None

    return run


def _ctx(reg: Registry, *, repo: Path = Path("."), config: RunConfig | None = None, run_id: str = "test-run",
) -> FlowContext:
    work = WorkContext(
        repo=repo, source=repo, base_branch="main", base_sha="0" * 40, head_branch="feature", head_sha="1" * 40,
        is_ephemeral=False, run_id=run_id,
    )
    return FlowContext(config=config if config is not None else RunConfig(), work=work, registry=reg)

async def test_missing_step_fails_before_any_step_runs() -> None:
    reg = Registry()
    reg.register_phase(FlowStep("a", run=_trace("a")))
    reg.set_flow("t", ["a", "ghost"])
    ctx = _ctx(reg)
    with pytest.raises(UnresolvedExtensionError, match=r"flow 't'.*'ghost'"):
        await run_flow(reg, "t", ctx)
    assert "trace" not in ctx.data
