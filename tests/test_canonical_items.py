import jsonschema
import pytest

from daydream.deep.fix_steps import _attach_verdicts
from daydream.phases import MERGED_ITEMS_SCHEMA, normalize_items


def _raw_item(**overrides: object) -> dict[str, object]:
    """Build one merge-agent item, defaults filled, any field overridable."""
    item: dict[str, object] = {"id": 1, "lens": "per-stack", "file": "a.py", "line": 1,
                               "description": "d", "confidence": "HIGH", "rationale": "r",
                               "evidence": "a.py:1", "severity": "medium",
                               "source_uids": ["python:1"]}
    item.update(overrides)
    return item

def test_schema_accepts_related_files() -> None:
    item = _raw_item(line=4, evidence="a.py:4", lens="cross-stack", severity="high",
                     related_files=["b.py", "svc/handler.py"],
                     source_uids=["python:1", "react:2"])
    jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)  # must pass

def test_schema_rejects_non_string_related_files() -> None:
    item = _raw_item(line=4, evidence="a.py:4", related_files=[42], source_uids=[])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)

def test_schema_requires_lens_and_severity() -> None:
    item = _raw_item(line=4, evidence="a.py:4", lens="structural", severity="high",
                     related_files=None, source_uids=["structure:1"])
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
    items = normalize_items([{"id": 1, "lens": "structural", "file": "b.py", "line": 1, "description": "y",
         "confidence": "HIGH", "rationale": "r", "severity": "high"},
        {"id": 1, "lens": "per-stack", "file": "a.py", "line": 1, "description": "x",
         "confidence": "HIGH", "rationale": "r", "severity": "low"}])
    payload = {"verdicts": [{"issue_id": items[1]["id"], "verdict": "contradicts",
                             "evidence": "e", "unverified_assumptions": []}]}
    joined = _attach_verdicts(items, payload)
    assert joined[0].get("verifier_verdict") is None       # structural NOT mismatched
    assert joined[1]["verifier_verdict"] == "contradicts"  # right item got the verdict

def test_schema_accepts_wonder_lens() -> None:
    item = _raw_item(line=4, evidence="a.py:4", confidence="MEDIUM", lens="wonder",
                     related_files=None, source_uids=None)
    jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)  # must pass

def test_schema_requires_source_uids() -> None:
    """Strict schema requires source_uids; null/[] expresses unattributed items."""
    item = _raw_item(line=4, evidence="a.py:4", lens="cross-stack", severity="high",
                     related_files=None, source_uids=["python:1"])
    jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)  # must pass
    bad = {k: v for k, v in item.items() if k != "source_uids"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"items": [bad]}, MERGED_ITEMS_SCHEMA)

def test_schema_rejects_non_string_source_uids() -> None:
    """Schema rejects numeric citations before host validation checks actual UID membership."""
    item = _raw_item(line=4, evidence="a.py:4", related_files=None, source_uids=[7])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"items": [item]}, MERGED_ITEMS_SCHEMA)

def test_normalize_mints_a_durable_handle_beside_the_renumbered_id() -> None:
    """Every item receives a durable item_uid beside its reassigned display ordinal,
    including records arriving with duplicate or missing IDs.
    """
    raw = [_raw_item(id=1, description="x"),
           _raw_item(id=1, description="y", lens="structural"),
           {k: v for k, v in _raw_item(description="z").items() if k != "id"}]

    out = normalize_items(raw)

    assert [item["id"] for item in out] == [1, 2, 3]
    uids = [item["item_uid"] for item in out]
    assert all(isinstance(uid, str) and uid for uid in uids), uids
    assert len(set(uids)) == len(uids), uids
    # The inputs are never mutated: the durable handle lands on the fresh dicts.
    assert all("item_uid" not in item for item in raw), raw

def test_existing_item_uid_is_preserved_while_id_is_reassigned() -> None:
    """Changing the dense display ordinal preserves the existing durable item_uid."""
    raw = [_raw_item(id=42, description="already identified", item_uid="item:9"),
           _raw_item(id=42, description="newcomer")]

    out = normalize_items(raw)

    assert [item["id"] for item in out] == [1, 2]
    assert out[0]["item_uid"] == "item:9", "a minted handle was reassigned"
    assert out[1]["item_uid"] not in ("", None) and out[1]["item_uid"] != "item:9", out

def test_preserved_item_uid_out_of_position_does_not_collide_with_a_minted_one() -> None:
    """Minting skips existing UIDs even when their ordinals differ from display positions.

    Preserving item:2 in position 1 must not mint item:2 again for position 2.
    """
    raw = [_raw_item(id=1, description="preserved out of position", item_uid="item:2"),
           _raw_item(id=2, description="unstamped, lands on position 2"),
           _raw_item(id=3, description="unstamped, lands on position 3")]

    out = normalize_items(raw)

    uids = [item["item_uid"] for item in out]
    assert len(set(uids)) == len(uids), f"two items were minted the same identity: {uids}"
    assert out[0]["item_uid"] == "item:2", uids
    assert [item["id"] for item in out] == [1, 2, 3]
