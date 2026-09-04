import jsonschema
import pytest

from daydream.phases import MERGED_ITEMS_SCHEMA, normalize_items


def test_schema_accepts_related_files() -> None:
    item = {"id": 1, "description": "d", "file": "a.py", "line": 4,
            "confidence": "HIGH", "rationale": "r", "evidence": "a.py:4",
            "lens": "cross-stack", "severity": "high",
            "related_files": ["b.py", "svc/handler.py"],
            "source_uids": ["python:1", "react:2"]}
    jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)  # must pass


def test_schema_rejects_non_string_related_files() -> None:
    item = {"id": 1, "description": "d", "file": "a.py", "line": 4,
            "confidence": "HIGH", "rationale": "r", "evidence": "a.py:4",
            "lens": "per-stack", "severity": "medium",
            "related_files": [42], "source_uids": []}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)


def test_schema_requires_lens_and_severity() -> None:
    item = {"id": 1, "description": "d", "file": "a.py", "line": 4,
            "confidence": "HIGH", "rationale": "r", "evidence": "a.py:4",
            "lens": "structural", "severity": "high", "related_files": None,
            "source_uids": ["structure:1"]}
    jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)  # passes
    bad = {k: v for k, v in item.items() if k != "lens"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"items": [bad]}, MERGED_ITEMS_SCHEMA)


def test_normalize_assigns_unique_ids_across_lenses() -> None:
    raw = [{"id": 1, "lens": "per-stack", "file": "a.py", "line": 1, "description": "x",
            "confidence": "HIGH", "rationale": "r", "severity": "low"},
           {"id": 1, "lens": "structural", "file": "b.py", "line": 1, "description": "y",
            "confidence": "HIGH", "rationale": "r", "severity": "high"}]
    out = normalize_items(raw)
    assert len({i["id"] for i in out}) == 2   # collision resolved, not preserved


def test_verdict_join_matches_after_collision_resolution() -> None:
    from daydream.deep.orchestrator import _attach_verdicts

    items = normalize_items([
        {"id": 1, "lens": "structural", "file": "b.py", "line": 1, "description": "y",
         "confidence": "HIGH", "rationale": "r", "severity": "high"},
        {"id": 1, "lens": "per-stack", "file": "a.py", "line": 1, "description": "x",
         "confidence": "HIGH", "rationale": "r", "severity": "low"}])
    payload = {"verdicts": [{"issue_id": items[1]["id"], "verdict": "contradicts",
                             "evidence": "e", "unverified_assumptions": []}]}
    joined = _attach_verdicts(items, payload)
    assert joined[0].get("verifier_verdict") is None       # structural NOT mismatched
    assert joined[1]["verifier_verdict"] == "contradicts"  # right item got the verdict


def test_schema_accepts_wonder_lens() -> None:
    item = {"id": 1, "description": "d", "file": "a.py", "line": 4,
            "confidence": "MEDIUM", "rationale": "r", "evidence": "a.py:4",
            "lens": "wonder", "severity": "medium", "related_files": None,
            "source_uids": None}
    jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)  # must pass


def test_schema_requires_source_uids() -> None:
    """Provenance is not optional in the contract (issue #1111).

    Strict mode rejects optional properties, so the merge agent must always emit
    the key; ``null`` or ``[]`` is how it says "cannot attribute this item",
    rather than saying it by omission. Omission would be indistinguishable from
    a model that simply ignored the field.
    """
    item = {"id": 1, "description": "d", "file": "a.py", "line": 4,
            "confidence": "HIGH", "rationale": "r", "evidence": "a.py:4",
            "lens": "cross-stack", "severity": "high", "related_files": None,
            "source_uids": ["python:1"]}
    jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)  # must pass
    bad = {k: v for k, v in item.items() if k != "source_uids"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"items": [bad]}, MERGED_ITEMS_SCHEMA)


def test_schema_rejects_non_string_source_uids() -> None:
    """A uid is a string handle; a numeric entry is a malformed citation.

    The host validates uid *values* against the run's real record pool, but the
    schema still pins the shape so a structurally wrong payload is caught at the
    boundary rather than silently filtered later.
    """
    item = {"id": 1, "description": "d", "file": "a.py", "line": 4,
            "confidence": "HIGH", "rationale": "r", "evidence": "a.py:4",
            "lens": "per-stack", "severity": "medium", "related_files": None,
            "source_uids": [7]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)
