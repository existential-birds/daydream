"""Tests for the daydream.training.corpus skill-decode helper."""

from __future__ import annotations

from daydream.training.corpus import _stack_for_skill


def test_stack_for_skill_resolves_short_name() -> None:
    """Manifests store short skill names (e.g. 'python'); the stack
    derivation must round-trip them."""
    assert _stack_for_skill("python") == "python"
    assert _stack_for_skill("react") == "react"
    assert _stack_for_skill("beagle-python:review-python") == "python"
    assert _stack_for_skill(None) is None
    assert _stack_for_skill("unknown-stack") is None
