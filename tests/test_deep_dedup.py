"""Dedup candidates require file overlap and normalized-title bigram Jaccard similarity of at least 0.5."""

import pytest

from daydream.deep.dedup import (
    build_dedup_candidates,
    build_record_dedup_candidates,
    descriptions_match,
)
from daydream.deep.records import mint_record_uid, stamp_record_uids


def test_jaccard_similarity_threshold_met() -> None:
    records = [{"id": "r1", "file": "api.py", "line": 10, "description": "Missing input validation on login endpoint",}]
    alt_issues = [{"title": "Input validation missing on login endpoint", "files": ["api.py"],}]
    pairs = build_dedup_candidates(records, alt_issues)
    assert len(pairs) == 1
    assert pairs[0].similarity >= 0.5
    assert pairs[0].record_id == "r1"
    assert "api.py" in pairs[0].alt_files

# --- Record ↔ Record dedup tests -------------------------------------------

def test_record_dedup_mismatched_sources_raises() -> None:
    records = [{"id": "1", "file": "api.py", "line": 1, "description": "Issue one"},
        {"id": "2", "file": "api.py", "line": 2, "description": "Issue two"},
    ]
    with pytest.raises(ValueError, match="sources must contain exactly one entry per record"):
        build_record_dedup_candidates(records, sources=["python"])

# Host uids distinguish per-stack id collisions in persisted dedup-candidates.json.

_SHARED_DESC = "Missing input validation on the user endpoint"

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
