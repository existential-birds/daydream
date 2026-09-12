"""Tests for the projection share-cap stage (successor of corpus _stratify).

The legacy ``daydream.training.corpus._stratify`` helper was removed with the
legacy loader (issue #1093); stack-share capping now lives in
``daydream.training.corpus_projection.projector._apply_share_caps``. These
tests drive that stage through the behaviors the old module covered:
dominant-stack capping, uniform corpus, degenerate small corpus, and
deterministic output ordering.
"""

from __future__ import annotations

from daydream.training.corpus_projection.projector import _apply_share_caps


def _records() -> list[dict[str, object]]:
    """8 react + 2 python, in a fixed order with deterministic session ids."""
    react: list[dict[str, object]] = [
        {"record_id": f"r{i:02d}", "session_id": f"r{i:02d}", "stack": "react"} for i in range(1, 9)
    ]
    python_side: list[dict[str, object]] = [
        {"record_id": "p01", "session_id": "p01", "stack": "python"},
        {"record_id": "p02", "session_id": "p02", "stack": "python"},
    ]
    return react + python_side


def test_share_caps_limit_dominant_stack() -> None:
    """8 react + 2 python at stack share 0.6 caps react to 6 kept records."""
    records = _records()
    kept, exclusions = _apply_share_caps(records, max_stack_share=0.6, max_repo_share=None, max_profile_share=None)
    assert len([r for r in kept if r["stack"] == "react"]) <= 6
    assert kept and exclusions


def test_uniform_corpus_keeps_everything() -> None:
    """A corpus with no over-share group keeps every record (nothing excluded)."""
    react: list[dict[str, object]] = [
        {"record_id": "r01", "session_id": "r01", "stack": "react"},
        {"record_id": "r02", "session_id": "r02", "stack": "react"},
        {"record_id": "r03", "session_id": "r03", "stack": "react"},
        {"record_id": "r04", "session_id": "r04", "stack": "react"},
    ]
    python_side: list[dict[str, object]] = [
        {"record_id": f"p{i:02d}", "session_id": f"p{i:02d}", "stack": "python"} for i in range(1, 7)
    ]
    records = python_side + react
    kept, exclusions = _apply_share_caps(records, max_stack_share=0.6, max_repo_share=None, max_profile_share=None)
    assert len(kept) == 10
    assert exclusions == {}


def test_degenerate_small_corpus_fails_closed_when_cap_empties_population() -> None:
    """A cap that would empty the population fails closed (ValueError)."""
    import pytest

    records: list[dict[str, object]] = [
        {"record_id": f"x{i}", "session_id": f"x{i}", "stack": "python"} for i in range(4)
    ]
    with pytest.raises(ValueError):
        _apply_share_caps(records, max_stack_share=0.5, max_repo_share=None, max_profile_share=None)


def test_output_order_is_deterministic() -> None:
    """Same input always produces the same kept order (deterministic cap)."""
    kept_a, _ = _apply_share_caps(_records(), max_stack_share=0.6, max_repo_share=None, max_profile_share=None)
    kept_b, _ = _apply_share_caps(_records(), max_stack_share=0.6, max_repo_share=None, max_profile_share=None)
    assert [r["record_id"] for r in kept_a] == [r["record_id"] for r in kept_b]
