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


def _raw_item(**overrides: object) -> dict[str, object]:
    """Build one merge-agent item, defaults filled, any field overridable."""
    item: dict[str, object] = {"id": 1, "lens": "per-stack", "file": "a.py", "line": 1,
                               "description": "d", "confidence": "HIGH", "rationale": "r",
                               "evidence": "a.py:1", "severity": "medium",
                               "source_uids": ["python:1"]}
    item.update(overrides)
    return item


def test_normalize_mints_a_durable_handle_beside_the_renumbered_id() -> None:
    """Renumbering ``id`` is not enough: the item also needs a stable handle (#1111).

    ``id`` is the dense, human-facing ordinal and is reassigned on every call, so
    nothing can hold onto it across a merge write. ``item_uid`` is what a holder
    keys on instead, so it has to be minted for every item -- including the ones
    that arrived with colliding ids or no id at all, which is the normal shape
    since per-stack numbering restarts at 1 in every stack.
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
    """The divergence that makes two fields necessary (#1111).

    A dense display ordinal *must* shift when the set changes; a durable handle
    *must not*. Here they are forced apart in one call: the item's ``id`` is
    renumbered from 42 to 1 while the ``item_uid`` it already carries survives
    untouched. If one field served both roles, this assertion could not hold.
    """
    raw = [_raw_item(id=42, description="already identified", item_uid="item:9"),
           _raw_item(id=42, description="newcomer")]

    out = normalize_items(raw)

    assert [item["id"] for item in out] == [1, 2]
    assert out[0]["item_uid"] == "item:9", "a minted handle was reassigned"
    assert out[1]["item_uid"] not in ("", None) and out[1]["item_uid"] != "item:9", out


def test_preserved_item_uid_out_of_position_does_not_collide_with_a_minted_one() -> None:
    """REGRESSION: minting per display position could emit a duplicate (#1111).

    The first cut of this minted inline, per position -- ``item_uid(item) or
    mint_item_uid(new_id)``. Feed it an item preserving ``item:2`` from anywhere
    other than position 2 and the unstamped item that lands *on* position 2 was
    minted the very same ``item:2``: two shipped findings sharing one identity,
    from the helper whose entire purpose is to make that impossible. That is the
    defect class #1111 exists to remove -- a non-unique value used as a unique
    handle -- reintroduced by the fix for it.

    On this input the old code produced ``["item:2", "item:2", "item:3"]``.
    Minting now skips ordinals an already-stamped item in the list holds, so the
    preserved value is honoured and the newcomers route around it.
    """
    raw = [_raw_item(id=1, description="preserved out of position", item_uid="item:2"),
           _raw_item(id=2, description="unstamped, lands on position 2"),
           _raw_item(id=3, description="unstamped, lands on position 3")]

    out = normalize_items(raw)

    uids = [item["item_uid"] for item in out]
    assert len(set(uids)) == len(uids), f"two items were minted the same identity: {uids}"
    assert out[0]["item_uid"] == "item:2", uids
    assert [item["id"] for item in out] == [1, 2, 3]
