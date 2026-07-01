import jsonschema
import pytest

from daydream.phases import MERGED_ITEMS_SCHEMA, normalize_items


def test_schema_requires_lens_and_severity():
    item = {"id": 1, "description": "d", "file": "a.py", "line": 4,
            "confidence": "HIGH", "rationale": "r", "evidence": "a.py:4",
            "lens": "structural", "severity": "high"}
    jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)  # passes
    bad = {k: v for k, v in item.items() if k != "lens"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"items": [bad]}, MERGED_ITEMS_SCHEMA)


def test_normalize_assigns_unique_ids_across_lenses():
    raw = [{"id": 1, "lens": "per-stack", "file": "a.py", "line": 1, "description": "x",
            "confidence": "HIGH", "rationale": "r", "severity": "low"},
           {"id": 1, "lens": "structural", "file": "b.py", "line": 1, "description": "y",
            "confidence": "HIGH", "rationale": "r", "severity": "high"}]
    out = normalize_items(raw)
    assert len({i["id"] for i in out}) == 2   # collision resolved, not preserved


def test_verdict_join_matches_after_collision_resolution():
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
