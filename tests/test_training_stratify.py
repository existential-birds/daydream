"""Tests for daydream.training.export ``_stratify`` (Wave 5).

Stratification caps any one stack's share of the corpus at
``max_stack_share`` while preserving within-stack input order and emitting
output sorted by ``session_id``. These tests use inline dict literals — no
fixture archive needed — to drive the helper through the four behaviors
called out in plan §6: dominant-stack capping, uniform corpus, degenerate
small corpus (the ``max(1, ...)`` guard), and global session_id ordering.
"""

from __future__ import annotations

from daydream.training.export import _stratify


def test_stratify_caps_dominant_stack() -> None:
    """8 react + 2 python at share=0.6 caps react at floor(10*0.6)=6."""
    records = [
        {"session_id": f"r0{i}", "stack": "react"} for i in range(1, 9)
    ] + [
        {"session_id": "p01", "stack": "python"},
        {"session_id": "p02", "stack": "python"},
    ]

    out = _stratify(records, max_stack_share=0.6)

    assert len(out) == 8
    react_records = [r for r in out if r["stack"] == "react"]
    assert len(react_records) == 6
    react_ids = {r["session_id"] for r in react_records}
    assert react_ids == {"r01", "r02", "r03", "r04", "r05", "r06"}
    assert [r["session_id"] for r in out] == [
        "p01",
        "p02",
        "r01",
        "r02",
        "r03",
        "r04",
        "r05",
        "r06",
    ]


def test_stratify_handles_uniform_corpus() -> None:
    """A single-stack corpus at share=1.0 keeps every record."""
    records = [
        {"session_id": f"p0{i}", "stack": "python"} for i in range(1, 5)
    ]

    out = _stratify(records, max_stack_share=1.0)

    assert len(out) == 4


def test_stratify_degenerate_small_corpus() -> None:
    """floor(2 * 0.6) = 1 → max(1, 1) keeps one per stack → 2 total."""
    records = [
        {"session_id": "p01", "stack": "python"},
        {"session_id": "r01", "stack": "react"},
    ]

    out = _stratify(records, max_stack_share=0.6)

    assert len(out) == 2
    assert {r["session_id"] for r in out} == {"p01", "r01"}


def test_stratify_output_sorted_by_session_id() -> None:
    """Output is sorted by session_id regardless of input ordering."""
    records = [
        {"session_id": "z01", "stack": "a"},
        {"session_id": "a01", "stack": "b"},
        {"session_id": "m01", "stack": "c"},
    ]

    out = _stratify(records, max_stack_share=1.0)

    assert [r["session_id"] for r in out] == ["a01", "m01", "z01"]


def test_stratify_empty_input_returns_empty() -> None:
    """Empty input short-circuits to an empty list."""
    assert _stratify([], 0.6) == []


def test_stratify_input_not_mutated() -> None:
    """The input list and its dict elements are not mutated."""
    records = [
        {"session_id": "r01", "stack": "react"},
        {"session_id": "r02", "stack": "react"},
        {"session_id": "p01", "stack": "python"},
    ]
    original_len = len(records)
    original_ids = [r["session_id"] for r in records]
    original_identities = tuple(id(r) for r in records)

    _stratify(records, max_stack_share=0.6)

    assert len(records) == original_len
    assert [r["session_id"] for r in records] == original_ids
    assert tuple(id(r) for r in records) == original_identities
