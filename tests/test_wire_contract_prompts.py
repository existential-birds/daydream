"""Focused coverage for deep-review wire-contract policy delivery."""

from pathlib import Path
from typing import TypedDict

from daydream.deep.prompts import (
    build_generic_fallback_prompt,
    build_per_stack_prompt,
    build_structural_prompt,
)
from daydream.prompts.wire_contract import (
    WIRE_CONTRACT_GENERIC_INSTRUCTION,
    WIRE_CONTRACT_RUST_INSTRUCTION,
)


class _PromptPaths(TypedDict):
    diff_path: Path
    intent_path: Path
    alternatives_path: Path
    output_path: Path
    cwd: Path


def _paths(tmp_path: Path) -> _PromptPaths:
    return {
        "diff_path": tmp_path / ".daydream" / "diff.patch",
        "intent_path": tmp_path / ".daydream" / "deep" / "intent.md",
        "alternatives_path": tmp_path / ".daydream" / "deep" / "alternatives.json",
        "output_path": tmp_path / ".daydream" / "deep" / "stack-review.md",
        "cwd": tmp_path,
    }


def _default_strategy(stage: str) -> str:
    from daydream import review_profile as _rp

    return _rp.build_default_profile().strategies[stage].content


def test_rust_wire_contract_requires_an_input_deserialization_contract() -> None:
    assert "#[serde(default)]" in WIRE_CONTRACT_RUST_INSTRUCTION
    assert "struct-level #[serde(default)]" in WIRE_CONTRACT_RUST_INSTRUCTION
    assert "with Default values" in WIRE_CONTRACT_RUST_INSTRUCTION
    assert "optional/custom deserialization contract" in WIRE_CONTRACT_RUST_INSTRUCTION
    assert "only affects output behavior" in WIRE_CONTRACT_RUST_INSTRUCTION
    assert "(or #[serde(skip_serializing_if" not in WIRE_CONTRACT_RUST_INSTRUCTION


def test_generic_wire_contract_distinguishes_parsers_from_component_safe_builders() -> None:
    assert "whole-URL parsers" in WIRE_CONTRACT_GENERIC_INSTRUCTION
    assert "must not receive interpolated components" in WIRE_CONTRACT_GENERIC_INSTRUCTION
    assert "component-aware URL builder" in WIRE_CONTRACT_GENERIC_INSTRUCTION


def test_wire_contract_checklists_are_delivered_only_to_their_intended_prompts(
    tmp_path: Path,
) -> None:
    p = _paths(tmp_path)
    rust = build_per_stack_prompt(
        strategy=_default_strategy("discovery.per_stack"),
        stack_name="rust",
        files=["src/main.rs"],
        **p,
    )
    python = build_per_stack_prompt(
        strategy=_default_strategy("discovery.per_stack"),
        stack_name="python",
        files=["api.py"],
        **p,
    )
    generic = build_generic_fallback_prompt(
        strategy=_default_strategy("discovery.generic_fallback"),
        files=["config.yaml"],
        **p)
    structural = build_structural_prompt(
        strategy=_default_strategy("discovery.structural"),
        files=["api.py"],
        **p,
    )

    assert WIRE_CONTRACT_RUST_INSTRUCTION in rust
    assert WIRE_CONTRACT_GENERIC_INSTRUCTION not in rust
    assert WIRE_CONTRACT_RUST_INSTRUCTION not in python
    assert WIRE_CONTRACT_GENERIC_INSTRUCTION not in python
    assert WIRE_CONTRACT_GENERIC_INSTRUCTION in generic
    assert WIRE_CONTRACT_RUST_INSTRUCTION not in generic
    assert WIRE_CONTRACT_RUST_INSTRUCTION not in structural
    assert WIRE_CONTRACT_GENERIC_INSTRUCTION not in structural
