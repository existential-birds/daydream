"""Tests for the extension Registry (daydream/extensions)."""

from typing import Any

import pytest

from daydream.extensions import (
    ExtensionError,
    FlowStep,
    LoopGroup,
    Registry,
    ToolDecision,
    UnresolvedExtensionError,
)


async def _noop(ctx: Any) -> None:
    return None


async def _other(ctx: Any) -> None:
    return None


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


def test_replace_requires_flag_and_skill_prompt_roundtrip() -> None:
    reg = Registry()
    reg.register_phase(FlowStep(name="a", run=_noop))
    with pytest.raises(ExtensionError, match="'a' is already registered"):
        reg.register_phase(FlowStep(name="a", run=_noop))
    reg.register_phase(FlowStep(name="a", run=_other), replace=True)
    reg.override_skill("stack:python", "ro:review-python")
    assert reg.skill("stack:python") == "ro:review-python"
    reg.override_prompt("review", lambda **kw: "X")
    assert reg.prompt("review")() == "X"


def test_introspection_lists_names_in_registration_order() -> None:
    """`daydream ext validate` enumerates namespaces through these accessors."""
    reg = Registry()
    reg.register_phase(FlowStep(name="b", run=_noop))
    reg.register_phase(FlowStep(name="a", run=_noop))
    reg.set_flow("deep", ["b", "a"])
    reg.set_flow("custom", ["ghost"])  # unresolved names are allowed until pre-flight
    reg.override_skill("stack:python", "ro:review-python")
    reg.override_prompt("review", lambda **kw: "X")
    assert reg.phase_names() == ("b", "a")
    assert reg.flow_names() == ("deep", "custom")
    assert reg.skill_slots() == {"stack:python": "ro:review-python"}
    assert reg.prompt_names() == ("review",)


def test_remove_loop_internal_step_raises_descriptive_error() -> None:
    """remove() names the containing LoopGroup when the step is loop-internal."""
    reg = Registry()
    loop = LoopGroup(name="fix-loop", steps=("inner_step",), max_iterations=lambda ctx: 3)
    reg.set_flow("deep", [loop])
    with pytest.raises(UnresolvedExtensionError, match="inside loop group 'fix-loop'"):
        reg.remove("deep", "inner_step")


def test_stack_keys_returns_stack_slot_keys_only() -> None:
    reg = Registry()
    assert reg.stack_keys() == set()
    reg.override_skill("stack:python", "ro:review-python")
    reg.override_skill("stack:proto", "ro:review-proto")
    reg.override_skill("structural", "ro:review-structure")
    reg.override_skill("pr-feedback-fetch", "ro:fetch-pr-feedback")
    assert reg.stack_keys() == {"python", "proto"}


def test_tool_supervisor_registration_is_exclusive() -> None:
    reg = Registry()

    def supervisor(name, tool_input, *, phase):
        return ToolDecision(veto=name == "Write", reason="protected path")

    reg.register_tool_supervisor(supervisor)
    assert reg.tool_supervisor_if_registered() is supervisor
    assert supervisor("Write", {}, phase="fix").veto is True
    with pytest.raises(ExtensionError, match="tool supervisor.*already registered"):
        reg.register_tool_supervisor(supervisor)
    with pytest.raises(ValueError, match="veto.*reason"):
        ToolDecision(veto=True, reason="")


def test_tool_supervisor_registration_rejects_async_function() -> None:
    reg = Registry()

    async def supervisor(name, tool_input, *, phase):
        return ToolDecision(veto=False)

    with pytest.raises(ExtensionError, match="tool supervisor.*synchronous"):
        reg.register_tool_supervisor(supervisor)
    assert reg.tool_supervisor_if_registered() is None


def test_tool_supervisor_registration_rejects_async_callable_object() -> None:
    class AsyncSupervisor:
        async def __call__(self, name, tool_input, *, phase):
            return ToolDecision(veto=False)

    reg = Registry()
    with pytest.raises(ExtensionError, match="tool supervisor.*synchronous"):
        reg.register_tool_supervisor(AsyncSupervisor())
    assert reg.tool_supervisor_if_registered() is None
