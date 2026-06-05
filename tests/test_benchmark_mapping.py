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
    assert set(c) == {"path", "line", "body", "created_at"}


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
