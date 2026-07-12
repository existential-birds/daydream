"""Tests for runtime findings and tool supervision."""

from __future__ import annotations

from daydream.supervision import revise_finding_fields


def test_revise_finding_fields_updates_whitelist_only() -> None:
    item = {
        "id": 7,
        "severity": "high",
        "confidence": "medium",
        "description": "original",
        "rationale": "because",
        "evidence": ["line 1"],
        "file": "safe.py",
        "line": 12,
    }
    item_id = id(item)

    revise_finding_fields(
        item,
        {"severity": "low", "file": "hacked.py", "line": 999, "reason": "x"},
    )

    assert id(item) == item_id
    assert item["severity"] == "low"
    assert item["file"] == "safe.py"
    assert item["line"] == 12
    assert item["id"] == 7
