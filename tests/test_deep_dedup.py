"""Dedup pre-filter tests (D-27).

Covers ``daydream.deep.dedup.build_dedup_candidates`` which emits
``CandidatePair`` entries where a per-stack record and a TTT alternative
finding share at least one file AND have normalized-title bigram Jaccard
similarity >= 0.5.
"""

import pytest

from daydream.deep.dedup import (
    build_dedup_candidates,
    build_record_dedup_candidates,
    descriptions_match,
)


def test_file_overlap_without_title_similarity_produces_no_pair() -> None:
    """D-27: file overlap alone is insufficient — both gates must hold.

    Matching file paths with disjoint titles must NOT surface a candidate
    pair; D-27 requires BOTH file overlap AND title similarity >= 0.5.
    """
    per_stack = [
        {"id": "1", "file": "api.py", "line": 1, "description": "Missing return type"},
    ]
    ttt_alts = [{"files": ["api.py"], "title": "Frontend styling drift"}]
    pairs = build_dedup_candidates(per_stack, ttt_alts)
    assert pairs == []


def test_candidate_pairs_disjoint() -> None:
    """D-27: no file overlap AND no title overlap -> no pair."""
    records = [{"id": "1", "file": "api.py", "line": 1, "description": "SQL injection"}]
    alt_issues = [{"title": "Frontend styling drift", "files": ["App.tsx"]}]
    pairs = build_dedup_candidates(records, alt_issues=alt_issues)
    assert pairs == []


def test_jaccard_similarity_threshold_met() -> None:
    """D-27: records with high-similarity titles + shared file -> pair."""
    records = [
        {
            "id": "r1",
            "file": "api.py",
            "line": 10,
            "description": "Missing input validation on login endpoint",
        }
    ]
    alt_issues = [
        {
            "title": "Input validation missing on login endpoint",
            "files": ["api.py"],
        }
    ]
    pairs = build_dedup_candidates(records, alt_issues)
    assert len(pairs) == 1
    assert pairs[0].similarity >= 0.5
    assert pairs[0].record_id == "r1"
    assert "api.py" in pairs[0].alt_files


# --- Record ↔ Record dedup tests -------------------------------------------


def test_record_dedup_identical_descriptions() -> None:
    """Near-identical descriptions across different files produce a pair."""
    desc = "CLI audit entry points share duplicated logic"
    records = [
        {"id": "1", "file": "cli/audit.ts", "line": 133, "description": desc},
        {"id": "2", "file": "cli/audit-storybook.ts", "line": 260, "description": desc},
    ]
    pairs = build_record_dedup_candidates(records, sources=["typescript", "typescript"])
    assert len(pairs) == 1
    assert pairs[0].record_a_id == "1"
    assert pairs[0].record_b_id == "2"
    assert pairs[0].record_a_source == "typescript"
    assert pairs[0].record_b_source == "typescript"
    assert pairs[0].similarity >= 0.5


def test_record_dedup_no_pair_for_different_descriptions() -> None:
    """Records with unrelated descriptions should not be paired."""
    records = [
        {"id": "1", "file": "api.py", "line": 10, "description": "SQL injection in login query"},
        {"id": "2", "file": "ui.tsx", "line": 50, "description": "Missing alt text on images"},
    ]
    pairs = build_record_dedup_candidates(records, sources=["python", "react"])
    assert pairs == []


def test_record_dedup_same_file_similar_description() -> None:
    """Duplicate findings on the same file are also caught."""
    records = [
        {"id": "1", "file": "api.py", "line": 10, "description": "Report files overwritten on each viewport"},
        {"id": "2", "file": "api.py", "line": 80, "description": "Report files overwritten on each viewport iteration"},
    ]
    pairs = build_record_dedup_candidates(records, sources=["python", "python"])
    assert len(pairs) == 1
    assert pairs[0].record_a_source == "python"
    assert pairs[0].record_b_source == "python"


def test_record_dedup_empty_records() -> None:
    """Empty input produces no pairs."""
    assert build_record_dedup_candidates([], sources=[]) == []


def test_record_dedup_single_record() -> None:
    """A single record cannot form a pair."""
    records = [{"id": "1", "file": "api.py", "line": 1, "description": "Some issue"}]
    assert build_record_dedup_candidates(records, sources=["python"]) == []



def test_record_dedup_mismatched_sources_raises() -> None:
    """Mismatched sources/records lengths raise ValueError."""
    records = [
        {"id": "1", "file": "api.py", "line": 1, "description": "Issue one"},
        {"id": "2", "file": "api.py", "line": 2, "description": "Issue two"},
    ]
    with pytest.raises(ValueError, match="sources must contain exactly one entry per record"):
        build_record_dedup_candidates(records, sources=["python"])


def test_record_dedup_cross_stack_source_disambiguation() -> None:
    """Records with the same ID from different stacks get distinct source fields."""
    desc = "Missing input validation on user endpoint"
    records = [
        {"id": "1", "file": "api.py", "line": 10, "description": desc},
        {"id": "1", "file": "routes.py", "line": 42, "description": desc},
    ]
    pairs = build_record_dedup_candidates(records, sources=["python", "react"])
    assert len(pairs) == 1
    assert pairs[0].record_a_id == "1"
    assert pairs[0].record_b_id == "1"
    assert pairs[0].record_a_source == "python"
    assert pairs[0].record_b_source == "react"
    assert pairs[0].similarity >= 0.5


# ---------------------------------------------------------------------------
# descriptions_match: the scalar form of the pre-filter's similarity gate
# ---------------------------------------------------------------------------


def test_descriptions_match_agrees_with_the_pairwise_builder() -> None:
    """The predicate and ``build_record_dedup_candidates`` share one threshold.

    Issue #1103's host-side structural fold calls the predicate directly, so a
    drift between the two would let the host collapse findings the pre-filter
    considers distinct (or the reverse).
    """
    a = "Missing input validation on the user endpoint"
    b = "Missing input validation on user endpoints"
    records = [
        {"id": "1", "file": "api.py", "line": 1, "description": a},
        {"id": "2", "file": "api.py", "line": 2, "description": b},
    ]
    pairs = build_record_dedup_candidates(records, sources=["python", "structure"])
    assert bool(pairs) is descriptions_match(a, b) is True


def test_descriptions_match_rejects_unrelated_descriptions() -> None:
    assert not descriptions_match(
        "Missing input validation on the user endpoint",
        "The 1000-line file budget is exceeded",
    )


def test_descriptions_match_is_symmetric() -> None:
    a = "Wrong cache URL in the staging block"
    b = "The staging block has the wrong cache URL"
    assert descriptions_match(a, b) == descriptions_match(b, a)


def test_descriptions_match_never_collapses_contentless_descriptions() -> None:
    """Two descriptions that normalize to nothing are not "the same finding"."""
    assert not descriptions_match("", "")
    assert not descriptions_match("the a an", "of to for")
    assert not descriptions_match("", "Missing input validation")


# ---------------------------------------------------------------------------
# threshold= knob on build_record_dedup_candidates (issue #1106): an eval axis
# needs the full similarity distribution, because a threshold-only count can
# never fire when the threshold itself is mis-set.
# ---------------------------------------------------------------------------

# The issue's worked example. These two descriptions are the same defect
# ("config reading is duplicated") in different words, yet score ~0.1538 --
# far under the 0.5 pre-filter bar.
_ISSUE_1106_A = "New helper duplicates the existing loader"
_ISSUE_1106_B = "Config reading is implemented twice in this module"


def _issue_1106_records() -> list[dict[str, object]]:
    return [
        {"id": "1", "file": "loader.py", "line": 10, "description": _ISSUE_1106_A},
        {"id": "2", "file": "config.py", "line": 20, "description": _ISSUE_1106_B},
    ]


def test_record_dedup_threshold_zero_yields_the_full_similarity_distribution() -> None:
    """``threshold=0.0`` emits every comparable pair with its similarity attached."""
    pairs = build_record_dedup_candidates(
        _issue_1106_records(), sources=["python", "python"], threshold=0.0
    )
    assert len(pairs) == 1
    assert pairs[0].record_a_id == "1"
    assert pairs[0].record_b_id == "2"
    assert 0.0 < pairs[0].similarity < 0.5


def test_record_dedup_default_threshold_still_rejects_the_sub_bar_pair() -> None:
    """The default call is unchanged: the same pair is below the 0.5 bar."""
    assert build_record_dedup_candidates(_issue_1106_records(), sources=["python", "python"]) == []
