"""Dedup candidates require file overlap and normalized-title bigram Jaccard similarity of at least 0.5."""

import pytest

from daydream.deep.dedup import (
    build_dedup_candidates,
    build_record_dedup_candidates,
    descriptions_match,
)


def test_dedup_candidates_require_similarity_and_aligned_sources() -> None:
    records = [{"id": "r1", "file": "api.py", "line": 10, "description": "Missing input validation on login endpoint",}]
    alt_issues = [{"title": "Input validation missing on login endpoint", "files": ["api.py"],}]
    pairs = build_dedup_candidates(records, alt_issues)
    assert len(pairs) == 1
    assert pairs[0].similarity >= 0.5
    assert pairs[0].record_id == "r1"
    assert "api.py" in pairs[0].alt_files
    pair = [{"id": "1", "file": "api.py", "line": 1, "description": "Issue one"},
            {"id": "2", "file": "api.py", "line": 2, "description": "Issue two"}]
    with pytest.raises(ValueError, match="sources must contain exactly one entry per record"):
        build_record_dedup_candidates(pair, sources=["python"])
    # Two descriptions that normalize to nothing are not "the same finding".
    assert not descriptions_match("", "")
    assert not descriptions_match("the a an", "of to for")
    assert not descriptions_match("", "Missing input validation")
