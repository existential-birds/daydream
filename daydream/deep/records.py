"""Host-assigned identities and provenance for review records and merged items.

Pre-merge records use deterministic stack:ordinal UIDs, stamped after schema
validation. Merged items have separate durable identities and source UID lists;
display IDs may change during normalization. Content fingerprints identify
similar defects, whereas these UIDs identify particular records/items.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

RECORD_UID_KEY = "uid"
RECORD_SOURCE_UIDS_KEY = "source_uids"
# Separate from record identity: structural items can carry both keys.
ITEM_UID_KEY = "item_uid"
_UID_SEPARATOR = ":"


def record_uid(record: dict[str, Any]) -> str:
    """Return the pre-merge UID, or empty for absent/non-string values."""
    value = record.get(RECORD_UID_KEY)
    return value if isinstance(value, str) else ""


def stamp_item_uids(items: list[dict[str, Any]]) -> None:
    """Stamp missing durable item identities, preserving existing IDs and skipping collisions.

    Unlike display IDs reassigned by normalize_items, item UIDs survive renumbering.
    """
    _stamp_unique(items, key=ITEM_UID_KEY, getter=item_uid, mint=mint_item_uid)


def _stamp_unique(
    entries: list[dict[str, Any]],
    *,
    key: str,
    getter: Callable[[dict[str, Any]], str],
    mint: Callable[[int], str],
) -> None:
    """Fill missing identities in place, preserving existing values and skipping collisions."""
    taken = {uid for uid in (getter(entry) for entry in entries) if uid}
    ordinal = 0
    for entry in entries:
        if getter(entry):
            continue
        ordinal += 1
        while (candidate := mint(ordinal)) in taken:
            ordinal += 1
        entry[key] = candidate
        taken.add(candidate)


def mint_item_uid(ordinal: int) -> str:
    """Return the durable identity for the *ordinal*-th merged item, e.g. ``item:3``."""
    return f"item{_UID_SEPARATOR}{ordinal}"


def item_uid(item: dict[str, Any]) -> str:
    """Return the item's durable identity, or empty for absent/non-string values.

    Source UIDs cannot identify an item: several items may derive from one record.
    """
    value = item.get(ITEM_UID_KEY)
    return value if isinstance(value, str) else ""


def item_source_uids(item: dict[str, Any]) -> list[str]:
    """Return nonempty source_uids in first-seen order, falling back to the record UID.

    Host-appended items may bypass merging and retain only their pre-merge UID.
    """
    raw = item.get(RECORD_SOURCE_UIDS_KEY)
    if isinstance(raw, list):
        attributed = union_source_uids(uid for uid in raw if isinstance(uid, str) and uid)
        if attributed:
            return attributed
    own = record_uid(item)
    return [own] if own else []


def union_source_uids(*uid_groups: Any) -> list[str]:
    """Combine UID iterables in first-seen order, ignoring empty/non-string values."""
    seen: dict[str, None] = {}
    for group in uid_groups:
        for uid in group:
            if isinstance(uid, str) and uid:
                seen.setdefault(uid, None)
    return list(seen)


def record_issues(records: Any) -> list[Any] | None:
    """Read the current issues envelope; malformed payloads return None."""
    issues = records.get('issues') if isinstance(records, dict) else None
    return issues if isinstance(issues, list) else None


def record_issues_or_empty(records: Any) -> list[Any]:
    """Read current records, contributing no evidence for a malformed envelope."""
    return record_issues(records) or []


