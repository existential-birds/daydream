"""Dedup pre-filter tests (D-27).

Every test is xfail(strict=True) until Wave 3 plan 05-07 implements the
``daydream.deep.dedup`` module with ``build_dedup_candidates(...)``.
"""

import pytest


@pytest.mark.xfail(reason="Wave 3 plan 05-07 not yet implemented", strict=True)
def test_candidate_pairs_file_overlap_only() -> None:
    """D-27: file-overlap heuristic emits a candidate pair.

    When a per-stack record and a TTT alternative share a file path, the
    dedup pre-filter must surface them as a candidate pair for the merge
    agent to make the final same-concern judgment.
    """
    from daydream.deep.dedup import build_dedup_candidates

    per_stack = [{"id": "1", "file": "api.py", "title": "Missing return type"}]
    ttt_alts = [{"files": ["api.py"], "title": "Type hint coverage"}]
    pairs = build_dedup_candidates(per_stack, ttt_alts)
    assert len(pairs) >= 1
    assert any(p.record_file == "api.py" and "api.py" in p.alt_files for p in pairs)


@pytest.mark.xfail(reason="Wave 3 plan 05-07 not yet implemented", strict=True)
def test_candidate_pairs_file_and_title() -> None:
    """D-27: file overlap + similar title -> pair."""
    from daydream.deep.dedup import build_dedup_candidates

    records = [
        {"id": "1", "file": "api.py", "line": 42, "description": "Missing input validation on login"},
    ]
    alt_issues = [
        {"title": "Input validation missing on login endpoint", "files": ["api.py"]},
    ]
    pairs = build_dedup_candidates(records, alt_issues=alt_issues)
    assert len(pairs) >= 1


@pytest.mark.xfail(reason="Wave 3 plan 05-07 not yet implemented", strict=True)
def test_candidate_pairs_disjoint() -> None:
    """D-27: no file overlap AND no title overlap -> no pair."""
    from daydream.deep.dedup import build_dedup_candidates

    records = [{"id": "1", "file": "api.py", "line": 1, "description": "SQL injection"}]
    alt_issues = [{"title": "Frontend styling drift", "files": ["App.tsx"]}]
    pairs = build_dedup_candidates(records, alt_issues=alt_issues)
    assert pairs == []
