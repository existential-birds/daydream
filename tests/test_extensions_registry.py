"""Tests for the extension Registry (daydream/extensions)."""

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from daydream.extensions import (
    ExtensionError,
    FlowStep,
    LoopGroup,
    Registry,
    ToolDecision,
    ToolSupervisor,
    UnresolvedExtensionError,
)
from daydream.trajectory import DaydreamPhase


async def _noop(ctx: Any) -> None:
    return None


async def _other(ctx: Any) -> None:
    return None


def test_registry_has_no_skill_methods() -> None:
    """M7: the Registry has no skill-slot inventory or lookup."""
    reg = Registry()
    for method in ("override_skill", "skill", "skill_if_registered", "skill_slots", "stack_keys"):
        assert not hasattr(reg, method)


def test_flow_mutation_and_unresolved_lookup() -> None:
    reg = Registry()
    reg.register_phase(FlowStep(name="a", run=_noop))
    reg.register_phase(FlowStep(name="b", run=_noop))
    reg.set_flow("deep", ["a"])
    reg.insert_after("deep", anchor="a", step="b")
    reg.remove("deep", "a")
    assert reg.flow("deep") == ["b"]
    with pytest.raises(UnresolvedExtensionError, match="phase 'ghost'"):
        reg.phase("ghost")


def test_replace_requires_flag_and_prompt_roundtrip() -> None:
    reg = Registry()
    reg.register_phase(FlowStep(name="a", run=_noop))
    with pytest.raises(ExtensionError, match="'a' is already registered"):
        reg.register_phase(FlowStep(name="a", run=_noop))
    reg.register_phase(FlowStep(name="a", run=_other), replace=True)
    reg.override_prompt("review", lambda **kw: "X")
    assert reg.prompt("review")() == "X"


def test_introspection_lists_names_in_registration_order() -> None:
    """`daydream ext validate` enumerates namespaces through these accessors."""
    reg = Registry()
    reg.register_phase(FlowStep(name="b", run=_noop))
    reg.register_phase(FlowStep(name="a", run=_noop))
    reg.set_flow("deep", ["b", "a"])
    reg.set_flow("custom", ["ghost"])  # unresolved names are allowed until pre-flight
    reg.override_prompt("review", lambda **kw: "X")
    assert reg.phase_names() == ("b", "a")
    assert reg.flow_names() == ("deep", "custom")
    assert reg.prompt_names() == ("review",)


def test_remove_loop_internal_step_raises_descriptive_error() -> None:
    """remove() names the containing LoopGroup when the step is loop-internal."""
    reg = Registry()
    loop = LoopGroup(name="fix-loop", steps=("inner_step",), max_iterations=lambda ctx: 3)
    reg.set_flow("deep", [loop])
    with pytest.raises(UnresolvedExtensionError, match="inside loop group 'fix-loop'"):
        reg.remove("deep", "inner_step")


def test_tool_supervisor_registration_is_exclusive() -> None:
    reg = Registry()

    def supervisor(name: str, tool_input: dict[str, Any], *, phase: DaydreamPhase) -> ToolDecision:
        return ToolDecision(veto=name == "Write", reason="protected path")

    reg.register_tool_supervisor(cast(ToolSupervisor, supervisor))
    assert reg.tool_supervisor_if_registered() is supervisor
    assert supervisor("Write", {}, phase=DaydreamPhase.FIX).veto is True
    with pytest.raises(ExtensionError, match="tool supervisor.*already registered"):
        reg.register_tool_supervisor(supervisor)
    with pytest.raises(ValueError, match="veto.*reason"):
        ToolDecision(veto=True, reason="")


def test_tool_supervisor_registration_rejects_async_function() -> None:
    reg = Registry()

    async def supervisor(name: Any, tool_input: Any, *, phase: Any) -> Any:
        return ToolDecision(veto=False)

    with pytest.raises(ExtensionError, match="tool supervisor.*synchronous"):
        reg.register_tool_supervisor(cast(ToolSupervisor, supervisor))
    assert reg.tool_supervisor_if_registered() is None


def test_tool_supervisor_registration_rejects_async_callable_object() -> None:
    class AsyncSupervisor:
        async def __call__(self, name: Any, tool_input: Any, *, phase: Any) -> Any:
            return ToolDecision(veto=False)

    reg = Registry()
    with pytest.raises(ExtensionError, match="tool supervisor.*synchronous"):
        reg.register_tool_supervisor(cast(ToolSupervisor, AsyncSupervisor()))
    assert reg.tool_supervisor_if_registered() is None


def test_comment_contract_types_are_frozen_and_public() -> None:
    import daydream.extensions as ext

    cf = ext.CommentFinding(
        path="a.py",
        line=3,
        title="T",
        body="B",
        is_cross_stack=False,
        severity="high",
        confidence="HIGH",
        fingerprint="a" * 64,
    )
    with pytest.raises(FrozenInstanceError):
        cf.path = "x"  # type: ignore[misc]
    ctx = ext.FindingRenderContext(placement="summary")
    sc = ext.SummaryContext(
        findings=(ext.SummaryFinding(finding=cf, body_block="X"),),
        agent_prompt="P",
        review_info="I",
    )
    assert ctx.placement == "summary" and sc.findings[0].body_block == "X"
    # Issue #1113: ``diagrams`` is appended with a default, so this pre-existing
    # construction keeps working and an unaware fork sees None.
    assert sc.diagrams is None
    with_diagrams = ext.SummaryContext(
        findings=(), agent_prompt="P", review_info="I", diagrams="<details>D</details>"
    )
    assert with_diagrams.diagrams == "<details>D</details>"
    with pytest.raises(FrozenInstanceError):
        with_diagrams.diagrams = "x"  # type: ignore[misc]
    for name in ("CommentFinding", "FindingRenderContext", "SummaryFinding", "SummaryContext"):
        assert name in ext.__all__


def test_renderer_slot_override_and_lookup() -> None:
    reg = Registry()
    assert reg.renderer_if_registered("finding") is None
    with pytest.raises(UnresolvedExtensionError):
        reg.renderer("finding")
    def fn(finding: Any, ctx: Any) -> str:
        return "X"
    reg.override_renderer("finding", fn)
    assert reg.renderer("finding") is fn
    assert reg.renderer_if_registered("finding") is fn
    assert reg.renderer_names() == ("finding",)
