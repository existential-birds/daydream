"""Tests for deterministic merged-items → review_comments mapping."""

from daydream.benchmark.mapping import merged_items_to_review_comments

TS = "2026-06-03T00:00:00Z"


def _item(file, line, **kw):
    base = {
        "id": 1,
        "description": "Race on cache write",
        "file": file,
        "line": line,
        "confidence": "HIGH",
        "rationale": "Two goroutines...",
        "lens": "structural",
        "severity": "high",
    }
    base.update(kw)
    return base


def test_maps_fields_and_folds_description_into_body():
    [c] = merged_items_to_review_comments({"items": [_item("a/b.go", 48)]}, created_at=TS)
    assert c["path"] == "a/b.go" and c["line"] == 48 and c["created_at"] == TS
    assert "Race on cache write" in c["body"] and "Two goroutines" in c["body"]
    assert set(c) == {"path", "line", "body", "created_at", "confidence", "severity"}


def test_structured_confidence_severity_preserved():
    doc = {"items": [
        _item("a/b.go", 48, id=1, severity="high", confidence="HIGH"),
        _item("f.py", None, id=2, severity="", confidence="", description="y", rationale="y"),
        _item("c.rs", 10, id=3, severity="CRITICAL", confidence="low"),
        _item("d.ts", 5, id=4, lens="cross-stack", severity="medium", confidence="MEDIUM"),
    ]}
    out = merged_items_to_review_comments(doc, created_at=TS)
    by_path = {c["path"]: c for c in out}

    a = by_path["a/b.go"]
    assert a["severity"] == "high" and a["confidence"] == "HIGH"
    assert "**Severity:** high" in a["body"] and "**Confidence:** HIGH" in a["body"]

    b = by_path["f.py"]
    assert b["severity"] is None and b["confidence"] is None and b["line"] is None

    c = by_path["c.rs"]
    assert c["severity"] == "critical" and c["confidence"] == "LOW"

    d = by_path["d.ts"]
    assert d["severity"] == "medium" and d["confidence"] == "MEDIUM"
    assert "**Severity:** medium" in d["body"] and "**Confidence:** MEDIUM" in d["body"]


def test_empty_items_yields_no_comments():
    assert merged_items_to_review_comments({"items": []}, created_at=TS) == []


def test_empty_file_is_skipped_and_null_line_preserved():
    doc = {"items": [_item("", 1, id=1), _item("f.py", None, id=2, description="y", rationale="y")]}
    out = merged_items_to_review_comments(doc, created_at=TS)
    assert len(out) == 1 and out[0]["path"] == "f.py" and out[0]["line"] is None


def test_rationale_equal_to_description_not_duplicated():
    doc = {"items": [_item("f.py", 1, description="same", rationale="same")]}
    [c] = merged_items_to_review_comments(doc, created_at=TS)
    assert c["body"].count("same") == 1


def test_mapping_skips_items_that_would_produce_empty_body():
    doc = {"items": [
        {"file": "a.py", "line": 1, "description": "", "rationale": "", "severity": "", "confidence": ""},
        {"file": "b.py", "line": 2, "description": "real finding", "severity": "high", "confidence": "HIGH"},
    ]}
    out = merged_items_to_review_comments(doc, created_at="T")
    assert [c["path"] for c in out] == ["b.py"]          # empty-body item dropped
    assert all(c["body"].strip() for c in out)           # no harvested comment is ever empty


def test_min_confidence_high_excludes_medium_and_unrated():
    doc = {"items": [
        _item("high.py", 1, id=1, confidence="HIGH"),
        _item("med.py", 2, id=2, confidence="MEDIUM"),
        _item("none.py", 3, id=3, confidence=""),  # normalises to None
    ]}
    out = merged_items_to_review_comments(doc, created_at=TS, min_confidence="HIGH")
    assert [c["path"] for c in out] == ["high.py"]


def test_min_severity_high_excludes_medium_and_unrated():
    doc = {"items": [
        _item("high.py", 1, id=1, severity="high"),
        _item("med.py", 2, id=2, severity="medium"),
        _item("none.py", 3, id=3, severity=""),  # normalises to None
    ]}
    out = merged_items_to_review_comments(doc, created_at=TS, min_severity="high")
    assert [c["path"] for c in out] == ["high.py"]


def test_combined_thresholds_item_must_clear_both():
    # HIGH confidence clears --min-confidence, but medium severity fails --min-severity.
    doc = {"items": [_item("f.py", 1, confidence="HIGH", severity="medium")]}
    out = merged_items_to_review_comments(doc, created_at=TS, min_confidence="HIGH", min_severity="high")
    assert out == []
