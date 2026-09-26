"""Focused coverage for deep-review wire-contract policy delivery."""

from pathlib import Path

from daydream.deep.prompts import (
    build_generic_fallback_prompt,
    build_per_stack_prompt,
    build_structural_prompt,
)
from daydream.prompts.wire_contract import (
    WIRE_CONTRACT_GENERIC_INSTRUCTION,
    WIRE_CONTRACT_RUST_INSTRUCTION,
)
from tests.harness.review_profile import default_strategy as _default_strategy, prompt_paths


def test_wire_contract_checklists_are_delivered_only_to_their_intended_prompts(
    tmp_path: Path,
) -> None:
    p = prompt_paths(tmp_path)
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
