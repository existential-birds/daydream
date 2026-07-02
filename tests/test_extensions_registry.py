"""Tests for the extension Registry (daydream/extensions)."""

from typing import Any

import pytest

from daydream.extensions import (
    ExtensionError,
    FlowStep,
    Registry,
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


def test_stack_keys_returns_stack_slot_keys_only() -> None:
    reg = Registry()
    assert reg.stack_keys() == set()
    reg.override_skill("stack:python", "ro:review-python")
    reg.override_skill("stack:proto", "ro:review-proto")
    reg.override_skill("structural", "ro:review-structure")
    reg.override_skill("pr-feedback-fetch", "ro:fetch-pr-feedback")
    assert reg.stack_keys() == {"python", "proto"}
