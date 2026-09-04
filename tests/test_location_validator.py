"""Tests for the pre-report finding-location validator (issue #745)."""

from __future__ import annotations

from daydream.deep.location_validator import validate_finding, validate_records
from daydream.hunk_index import parse_hunks

DIFF = (
    "diff --git a/orchestrator.py b/orchestrator.py\n--- a/orchestrator.py\n+++ b/orchestrator.py\n"
    "@@ -2270,3 +2284,5 @@\n x\n+x1\n+x2\n"
)
INDEX = parse_hunks(DIFF)


def test_validate_finding_all_five_fields() -> None:
    check = validate_finding(INDEX, "orchestrator.py", 2285)
    assert check.file_exists is True
    assert check.line_exists is True
    assert check.in_hunk is True
    assert check.nearest_hunk == (2284, 2288)
    assert check.distance == 0


def test_validate_finding_item6_coordinates_demote_beyond_tolerance() -> None:
    # a80b9373 item-6: line 2272, nearest hunk 2284..2288, distance 12 > tolerance 3
    check = validate_finding(INDEX, "orchestrator.py", 2272, tolerance=3)
    assert check.file_exists is True
    assert check.in_hunk is False
    assert check.nearest_hunk == (2284, 2288)
    assert check.distance == 12
    assert check.line_exists is True  # 2272 <= max new_end (2288) in fixture


def test_validate_finding_snaps_within_tolerance() -> None:
    # line 2281 is 3 below hunk start 2284 -> within tolerance, snap candidate
    check = validate_finding(INDEX, "orchestrator.py", 2281, tolerance=3)
    assert check.in_hunk is False
    assert check.distance == 3
    assert check.nearest_hunk == (2284, 2288)


def test_validate_finding_missing_file_is_all_false() -> None:
    check = validate_finding(INDEX, "nope.py", 10)
    assert check.file_exists is False
    assert check.line_exists is False
    assert check.in_hunk is False
    assert check.nearest_hunk is None
    assert check.distance is None


def test_validate_records_snaps_and_demotes() -> None:
    records: list[dict[str, object]] = [
        {
            "id": 1,
            "file": "orchestrator.py",
            "line": 2281,
            "evidence": "orchestrator.py:2281",  # within tol -> snap line + evidence
        },
        {  # beyond tol -> demote severity/confidence in place + location_note
            "id": 2,
            "file": "orchestrator.py",
            "line": 2272,
            "severity": "high",
            "confidence": "HIGH",
        },
        {"id": 3, "file": "orchestrator.py", "line": 2285},   # in-hunk -> untouched
        {"id": 4},                                            # no file -> untouched
        {"id": 5, "file": "orchestrator.py", "line": "x"},    # non-int line -> untouched
    ]
    out = validate_records(INDEX, records, tolerance=3)
    assert out[0]["line"] == 2284
    assert out[0]["evidence"] == "orchestrator.py:2284"
    assert out[1]["severity"] == "low" and out[1]["confidence"] == "LOW"
    assert "location_note" in out[1]
    assert out[2]["line"] == 2285 and "location_note" not in out[2]
    assert out[3] == {"id": 4}
    assert out[4]["line"] == "x"


# ---------------------------------------------------------------------------
# location_cited_line: the non-destructive record of the reviewer's citation
# (issue #1106). The snap overwrites ``line``, so without this field the line
# the reviewer actually cited is unrecoverable and location accuracy is
# unmeasurable post-hoc.
# ---------------------------------------------------------------------------


def test_snapped_record_preserves_the_cited_line_alongside_the_snapped_line() -> None:
    """A snap is non-destructive: both the snapped and the cited line survive."""
    records: list[dict[str, object]] = [
        {"id": 1, "file": "orchestrator.py", "line": 2281, "evidence": "orchestrator.py:2281"},
    ]
    out = validate_records(INDEX, records, tolerance=3)
    assert out[0]["line"] == 2284  # snapped to the nearest hunk boundary
    assert out[0]["location_cited_line"] == 2281  # what the reviewer cited
    assert out[0]["evidence"] == "orchestrator.py:2284"


def test_demoted_record_preserves_the_cited_line_next_to_the_demotion_marks() -> None:
    """Beyond tolerance: the cited line is machine-readable, not just prose."""
    records: list[dict[str, object]] = [
        {
            "id": 2,
            "file": "orchestrator.py",
            "line": 2272,
            "severity": "high",
            "confidence": "HIGH",
        },
    ]
    out = validate_records(INDEX, records, tolerance=3)
    assert out[0]["location_cited_line"] == 2272
    assert out[0]["line"] == 2272  # demotion does not move the line
    assert out[0]["location_distrust"] is True
    assert out[0]["severity_before_demotion"] == "high"
    assert out[0]["severity"] == "low"
    assert out[0]["confidence"] == "LOW"


def test_in_hunk_record_gets_no_cited_line_key() -> None:
    """The field means "relocated or distrusted" -- an untouched record lacks it."""
    records: list[dict[str, object]] = [{"id": 3, "file": "orchestrator.py", "line": 2285}]
    out = validate_records(INDEX, records, tolerance=3)
    assert "location_cited_line" not in out[0]
    assert out[0] == {"id": 3, "file": "orchestrator.py", "line": 2285}


def test_structural_whole_file_record_gets_no_cited_line_key() -> None:
    """The ``lens="structural"`` / ``line: 0`` carve-out stays fully untouched."""
    records: list[dict[str, object]] = [
        {"id": 4, "file": "orchestrator.py", "line": 0, "lens": "structural", "severity": "high"},
    ]
    out = validate_records(INDEX, records, tolerance=3)
    assert "location_cited_line" not in out[0]
    assert out[0] == {
        "id": 4,
        "file": "orchestrator.py",
        "line": 0,
        "lens": "structural",
        "severity": "high",
    }
