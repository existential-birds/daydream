
from daydream.phases import normalize_items


def _raw_item(**overrides: object) -> dict[str, object]:
    """Build one merge-agent item, defaults filled, any field overridable."""
    item: dict[str, object] = {"id": 1, "lens": "per-stack", "file": "a.py", "line": 1,
                               "description": "d", "confidence": "HIGH", "rationale": "r",
                               "evidence": "a.py:1", "severity": "medium",
                               "source_uids": ["python:1"]}
    item.update(overrides)
    return item

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
