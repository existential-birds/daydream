
from daydream.phases import normalize_items


def _raw_item(**overrides: object) -> dict[str, object]:
    """Build one merge-agent item, defaults filled, any field overridable."""
    item: dict[str, object] = {"id": 1, "lens": "per-stack", "file": "a.py", "line": 1,
                               "description": "d", "confidence": "HIGH", "rationale": "r",
                               "evidence": "a.py:1", "severity": "medium",
                               "source_uids": ["python:1"]}
    item.update(overrides)
    return item


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
