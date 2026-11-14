"""Pin referential identities used by dedup, arbitration, and record rewriting.

Real-path pipeline coverage lives in tests/deep_orchestrator/."""

from __future__ import annotations

from typing import Any

from daydream.deep.records import stamp_item_uids, union_source_uids


def test_union_drops_empty_and_non_string_entries() -> None:
    assert union_source_uids(["python:1", "", None, 3], []) == ["python:1"]
    # A minted item identity never repeats one that an item already carries out of position.
    items: list[dict[str, Any]] = [{"id": 1}, {"id": 2, "item_uid": "item:1"}, {"id": 3}]
    stamp_item_uids(items)
    assert [item["item_uid"] for item in items] == ["item:2", "item:1", "item:3"]
