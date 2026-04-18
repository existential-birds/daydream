"""Dedup pre-filter tests (D-27).

Covers ``daydream.deep.dedup.build_dedup_candidates`` which emits
``CandidatePair`` entries where a per-stack record and a TTT alternative
finding share at least one file AND have normalized-title bigram Jaccard
similarity >= 0.5.
"""

from daydream.deep.dedup import build_dedup_candidates


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


def test_candidate_pairs_file_and_title() -> None:
    """D-27: file overlap + similar title -> pair."""
    records = [
        {"id": "1", "file": "api.py", "line": 42, "description": "Missing input validation on login"},
    ]
    alt_issues = [
        {"title": "Input validation missing on login endpoint", "files": ["api.py"]},
    ]
    pairs = build_dedup_candidates(records, alt_issues=alt_issues)
    assert len(pairs) >= 1


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


def test_file_overlap_alone_insufficient() -> None:
    """D-27: file overlap without title similarity -> no pair."""
    records = [{"id": "r1", "file": "api.py", "line": 1, "description": "SQL injection"}]
    alt_issues = [{"title": "Logging verbosity too high", "files": ["api.py"]}]
    pairs = build_dedup_candidates(records, alt_issues)
    assert pairs == []
