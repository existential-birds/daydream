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
