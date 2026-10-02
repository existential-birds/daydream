"""Dedup candidates require file overlap and normalized-title bigram Jaccard similarity of at least 0.5."""

import pytest

from daydream.deep.dedup import (
    build_dedup_candidates,
    build_record_dedup_candidates,
    descriptions_match,
)
from daydream.deep.records import mint_record_uid, stamp_record_uids


def test_file_overlap_without_title_similarity_produces_no_pair() -> None:
    per_stack = [{"id": "1", "file": "api.py", "line": 1, "description": "Missing return type"},]
    ttt_alts = [{"files": ["api.py"], "title": "Frontend styling drift"}]
    pairs = build_dedup_candidates(per_stack, ttt_alts)
    assert pairs == []

def test_candidate_pairs_disjoint() -> None:
    records = [{"id": "1", "file": "api.py", "line": 1, "description": "SQL injection"}]
    alt_issues = [{"title": "Frontend styling drift", "files": ["App.tsx"]}]
    pairs = build_dedup_candidates(records, alt_issues=alt_issues)
    assert pairs == []

def test_jaccard_similarity_threshold_met() -> None:
    records = [{"id": "r1", "file": "api.py", "line": 10, "description": "Missing input validation on login endpoint",}]
    alt_issues = [{"title": "Input validation missing on login endpoint", "files": ["api.py"],}]
    pairs = build_dedup_candidates(records, alt_issues)
    assert len(pairs) == 1
    assert pairs[0].similarity >= 0.5
    assert pairs[0].record_id == "r1"
    assert "api.py" in pairs[0].alt_files

# --- Record ↔ Record dedup tests -------------------------------------------

def test_record_dedup_identical_descriptions() -> None:
    """Near-identical descriptions across different files produce a pair."""
    desc = "CLI audit entry points share duplicated logic"
    records = [{"id": "1", "file": "cli/audit.ts", "line": 133, "description": desc},
        {"id": "2", "file": "cli/audit-storybook.ts", "line": 260, "description": desc},
    ]
    # Stamp fixture records exactly as production does.
    stamp_record_uids(records, "typescript")
    pairs = build_record_dedup_candidates(records, sources=["typescript", "typescript"])
    assert len(pairs) == 1
    assert pairs[0].record_a_id == "1"
    assert pairs[0].record_b_id == "2"
    assert pairs[0].record_a_uid == "typescript:1"
    assert pairs[0].record_b_uid == "typescript:2"
    assert pairs[0].record_a_source == "typescript"
    assert pairs[0].record_b_source == "typescript"
    assert pairs[0].similarity >= 0.5

def test_record_dedup_no_pair_for_different_descriptions() -> None:
    records = [{"id": "1", "file": "api.py", "line": 10, "description": "SQL injection in login query"},
        {"id": "2", "file": "ui.tsx", "line": 50, "description": "Missing alt text on images"},
    ]
    pairs = build_record_dedup_candidates(records, sources=["python", "react"])
    assert pairs == []

def test_record_dedup_same_file_similar_description() -> None:
    records = [{"id": "1", "file": "api.py", "line": 10, "description": "Report files overwritten on each viewport"},
        {"id": "2", "file": "api.py", "line": 80, "description": "Report files overwritten on each viewport iteration"},
    ]
    pairs = build_record_dedup_candidates(records, sources=["python", "python"])
    assert len(pairs) == 1
    assert pairs[0].record_a_source == "python"
    assert pairs[0].record_b_source == "python"

def test_record_dedup_empty_records() -> None:
    assert build_record_dedup_candidates([], sources=[]) == []

def test_record_dedup_single_record() -> None:
    records = [{"id": "1", "file": "api.py", "line": 1, "description": "Some issue"}]
    assert build_record_dedup_candidates(records, sources=["python"]) == []

def test_record_dedup_mismatched_sources_raises() -> None:
    records = [{"id": "1", "file": "api.py", "line": 1, "description": "Issue one"},
        {"id": "2", "file": "api.py", "line": 2, "description": "Issue two"},
    ]
    with pytest.raises(ValueError, match="sources must contain exactly one entry per record"):
        build_record_dedup_candidates(records, sources=["python"])

def test_record_dedup_cross_stack_source_disambiguation() -> None:
    desc = "Missing input validation on user endpoint"
    records = [{"id": "1", "file": "api.py", "line": 10, "description": desc},
        {"id": "1", "file": "routes.py", "line": 42, "description": desc},
    ]
    pairs = build_record_dedup_candidates(records, sources=["python", "react"])
    assert len(pairs) == 1
    assert pairs[0].record_a_id == "1"
    assert pairs[0].record_b_id == "1"
    assert pairs[0].record_a_source == "python"
    assert pairs[0].record_b_source == "react"
    assert pairs[0].similarity >= 0.5

# Host uids distinguish per-stack id collisions in persisted dedup-candidates.json.

_SHARED_DESC = "Missing input validation on the user endpoint"

def test_record_dedup_carries_both_record_uids() -> None:
    records = [
        {"id": "1", "file": "api.py", "line": 10, "description": _SHARED_DESC, "uid": mint_record_uid("python", 1),},
        {"id": "1", "file": "routes.ts", "line": 42, "description": _SHARED_DESC,
            "uid": mint_record_uid("typescript", 7),
        },
    ]
    pairs = build_record_dedup_candidates(records, sources=["python", "typescript"])
    assert len(pairs) == 1
    # The ids collide (both "1") -- the uids are what tell the two sides apart.
    assert (pairs[0].record_a_id, pairs[0].record_b_id) == ("1", "1")
    assert (pairs[0].record_a_uid, pairs[0].record_b_uid) == ("python:1", "typescript:7")

def test_record_dedup_emits_empty_uid_for_a_record_that_has_none() -> None:
    """Unstamped post-merge items still participate in shipped-duplication evals with an empty uid."""
    records = [{"id": "1", "file": "api.py", "line": 10, "description": _SHARED_DESC},
        {"id": "2", "file": "api.py", "line": 42, "description": _SHARED_DESC, "uid": mint_record_uid("python", 2),},
        {"id": "3", "file": "api.py", "line": 90, "description": _SHARED_DESC},
    ]
    pairs = build_record_dedup_candidates(records, sources=["merged", "python", "merged"])
    assert len(pairs) == 3
    by_ids = {(p.record_a_id, p.record_b_id): (p.record_a_uid, p.record_b_uid) for p in pairs}
    assert by_ids[("1", "2")] == ("", "python:2")   # missing on the a side only
    assert by_ids[("2", "3")] == ("python:2", "")   # missing on the b side only
    assert by_ids[("1", "3")] == ("", "")           # missing on both sides

def test_record_dedup_uid_totally_orders_pairs_that_tie_on_both_ids() -> None:
    """Host uids totally order pairs whose reviewer-assigned ids collide across stacks."""
    python_records = [{"id": "1", "file": "api.py", "line": 10, "description": _SHARED_DESC},
        {"id": "2", "file": "api.py", "line": 20, "description": _SHARED_DESC},
    ]
    react_records = [{"id": "1", "file": "App.tsx", "line": 30, "description": _SHARED_DESC},
        {"id": "2", "file": "App.tsx", "line": 40, "description": _SHARED_DESC},
    ]
    stamp_record_uids(python_records, "python")
    stamp_record_uids(react_records, "react")
    records = [*python_records, *react_records]
    sources = ["python", "python", "react", "react"]
    pairs = build_record_dedup_candidates(records, sources=sources)

    assert [(p.record_a_id, p.record_b_id, p.record_a_uid, p.record_b_uid) for p in pairs] == [
        ("1", "1", "python:1", "react:1"), ("1", "2", "python:1", "python:2"), ("1", "2", "python:1", "react:2"),
        ("1", "2", "react:1", "react:2"), ("2", "1", "python:2", "react:1"), ("2", "2", "python:2", "react:2"),
    ]
    # Reproducible: the same input yields byte-identical ordering every call.
    assert build_record_dedup_candidates(records, sources=sources) == pairs

# descriptions_match: the scalar form of the pre-filter's similarity gate

def test_descriptions_match_agrees_with_the_pairwise_builder() -> None:
    """Structural folding and pairwise dedup must share the same similarity threshold."""
    a = "Missing input validation on the user endpoint"
    b = "Missing input validation on user endpoints"
    records = [{"id": "1", "file": "api.py", "line": 1, "description": a},
        {"id": "2", "file": "api.py", "line": 2, "description": b},
    ]
    pairs = build_record_dedup_candidates(records, sources=["python", "structure"])
    assert bool(pairs) is descriptions_match(a, b) is True

def test_descriptions_match_rejects_unrelated_descriptions() -> None:
    assert not descriptions_match(
        "Missing input validation on the user endpoint", "The 1000-line file budget is exceeded",
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

# Eval needs the full distribution to detect a misconfigured similarity threshold.

# Equivalent defects phrased differently score about 0.1538, below the 0.5 gate.
_ISSUE_1106_A = "New helper duplicates the existing loader"
_ISSUE_1106_B = "Config reading is implemented twice in this module"

def _issue_1106_records() -> list[dict[str, object]]:
    return [{"id": "1", "file": "loader.py", "line": 10, "description": _ISSUE_1106_A},
        {"id": "2", "file": "config.py", "line": 20, "description": _ISSUE_1106_B},
    ]

def test_record_dedup_threshold_zero_yields_the_full_similarity_distribution() -> None:
    pairs = build_record_dedup_candidates(_issue_1106_records(), sources=["python", "python"], threshold=0.0)
    assert len(pairs) == 1
    assert pairs[0].record_a_id == "1"
    assert pairs[0].record_b_id == "2"
    assert 0.0 < pairs[0].similarity < 0.5

def test_record_dedup_default_threshold_still_rejects_the_sub_bar_pair() -> None:
    assert build_record_dedup_candidates(_issue_1106_records(), sources=["python", "python"]) == []
