"""Dedup pre-filter tests (D-27).

Covers ``daydream.deep.dedup.build_dedup_candidates`` which emits
``CandidatePair`` entries where a per-stack record and a TTT alternative
finding share at least one file AND have normalized-title bigram Jaccard
similarity >= 0.5.
"""

from daydream.deep.dedup import build_dedup_candidates, build_record_dedup_candidates


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


# --- Record ↔ Record dedup tests -------------------------------------------


def test_record_dedup_identical_descriptions() -> None:
    """Near-identical descriptions across different files produce a pair."""
    desc = "CLI audit entry points share duplicated logic"
    records = [
        {"id": "1", "file": "cli/audit.ts", "line": 133, "description": desc},
        {"id": "2", "file": "cli/audit-storybook.ts", "line": 260, "description": desc},
    ]
    pairs = build_record_dedup_candidates(records)
    assert len(pairs) == 1
    assert pairs[0].record_a_id == "1"
    assert pairs[0].record_b_id == "2"
    assert pairs[0].similarity >= 0.5


def test_record_dedup_no_pair_for_different_descriptions() -> None:
    """Records with unrelated descriptions should not be paired."""
    records = [
        {"id": "1", "file": "api.py", "line": 10, "description": "SQL injection in login query"},
        {"id": "2", "file": "ui.tsx", "line": 50, "description": "Missing alt text on images"},
    ]
    pairs = build_record_dedup_candidates(records)
    assert pairs == []


def test_record_dedup_same_file_similar_description() -> None:
    """Duplicate findings on the same file are also caught."""
    records = [
        {"id": "1", "file": "api.py", "line": 10, "description": "Report files overwritten on each viewport"},
        {"id": "2", "file": "api.py", "line": 80, "description": "Report files overwritten on each viewport iteration"},
    ]
    pairs = build_record_dedup_candidates(records)
    assert len(pairs) == 1


def test_record_dedup_empty_records() -> None:
    """Empty input produces no pairs."""
    assert build_record_dedup_candidates([]) == []


def test_record_dedup_single_record() -> None:
    """A single record cannot form a pair."""
    records = [{"id": "1", "file": "api.py", "line": 1, "description": "Some issue"}]
    assert build_record_dedup_candidates(records) == []
