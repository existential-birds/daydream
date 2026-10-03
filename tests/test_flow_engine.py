"""Tests for the flow engine (daydream/flows/engine.py)."""

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from daydream.backends import BackendExecutionInput
from daydream.backends.pi import PiBackend
from daydream.extensions import (
    BreakLoop,
    FlowStep,
    LoopGroup,
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


def _stop(code: int) -> Callable[[FlowContext], Awaitable[Stop | BreakLoop | None]]:
    async def run(ctx: FlowContext) -> Stop | BreakLoop | None:
        return Stop(code)

    return run


def _break_after(tag: str, times: int) -> Callable[[FlowContext], Awaitable[Stop | BreakLoop | None]]:
    calls = 0

    async def run(ctx: FlowContext) -> Stop | BreakLoop | None:
        nonlocal calls
        ctx.data.setdefault("trace", []).append(tag)
        calls += 1
        if calls >= times:
            return BreakLoop()
        return None

    return run


def _ctx(reg: Registry, *, repo: Path = Path("."), config: RunConfig | None = None, run_id: str = "test-run",
) -> FlowContext:
    work = WorkContext(
        repo=repo, source=repo, base_branch="main", base_sha="0" * 40, head_branch="feature", head_sha="1" * 40,
        is_ephemeral=False, run_id=run_id,
    )
    return FlowContext(config=config if config is not None else RunConfig(), work=work, registry=reg)

def test_backend_for_resolves_pi_model_from_workspace(tmp_path: Path) -> None:
    settings = tmp_path / ".pi" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"defaultModel": "gpt-5.6-luna"}')
    ctx = _ctx(Registry(), repo=tmp_path, config=RunConfig(backend="pi"))

    assert ctx.backend_for("review").model == "gpt-5.6-luna"

def test_captured_backend_execution_uses_context_workspace_and_cache(tmp_path: Path) -> None:
    settings = tmp_path / ".pi" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"defaultModel": "context-model"}')
    execution = BackendExecutionInput.from_environment({"HOME": str(tmp_path)}, backend="pi")
    ctx = _ctx(Registry(), repo=tmp_path, run_id="execution")
    ctx.config.backend = "pi"
    ctx.backend_execution = execution
    review = ctx.backend_for("review")
    assert review.model == "context-model"
    assert ctx.backend_for("review") is review
    assert len(ctx._backend_cache) == 1
    assert isinstance(review, PiBackend)
    assert review._execution_input is execution
    assert "backend_execution" not in repr(ctx)


async def test_order_gating_stop_and_loop() -> None:
    reg = Registry()
    reg.register_phase(FlowStep("a", run=_trace("a")))
    reg.register_phase(FlowStep("off", run=_trace("x"), enabled=lambda ctx: False))
    reg.register_phase(FlowStep("b", run=_break_after("b", times=2)))
    reg.register_phase(FlowStep("end", run=_stop(7)))
    reg.set_flow("t", ["a", "off", LoopGroup("it", ("b",), max_iterations=lambda ctx: 5), "end"])
    ctx = _ctx(reg)  # minimal FlowContext factory in this test file
    assert await run_flow(reg, "t", ctx) == 7
    assert ctx.data["trace"] == ["a", "b", "b"]

async def test_missing_step_fails_before_any_step_runs() -> None:
    reg = Registry()
    reg.register_phase(FlowStep("a", run=_trace("a")))
    reg.set_flow("t", ["a", "ghost"])
    ctx = _ctx(reg)
    with pytest.raises(UnresolvedExtensionError, match=r"flow 't'.*'ghost'"):
        await run_flow(reg, "t", ctx)
    assert "trace" not in ctx.data
