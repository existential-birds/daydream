"""Host-assigned identities and provenance for review records and merged items.

Pre-merge records use deterministic stack:ordinal UIDs, stamped after schema
validation. Merged items have separate durable identities and source UID lists;
display IDs may change during normalization. Content fingerprints identify
similar defects, whereas these UIDs identify particular records/items.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

RECORD_UID_KEY = "uid"
RECORD_SOURCE_UIDS_KEY = "source_uids"
# Separate from record identity: structural items can carry both keys.
ITEM_UID_KEY = "item_uid"
_UID_SEPARATOR = ":"

_RECORDS_FILENAME_PREFIX = "stack-"
_RECORDS_FILENAME_SUFFIX = "-records.json"


def mint_record_uid(stack_name: str, ordinal: int) -> str:
    """Return the ``uid`` for the *ordinal*-th record of *stack_name*, e.g. ``python:1``."""
    return f"{stack_name}{_UID_SEPARATOR}{ordinal}"


def stack_name_from_records_source(source: str) -> str:
    """Normalize stack-<name>-records.json or return a bare/unrecognized source unchanged."""
    if source.startswith(_RECORDS_FILENAME_PREFIX) and source.endswith(_RECORDS_FILENAME_SUFFIX):
        return source[len(_RECORDS_FILENAME_PREFIX) : -len(_RECORDS_FILENAME_SUFFIX)]
    return source


def stack_name_from_uid(uid: str) -> str:
    """Return the stack before the final separator, or empty when no separator exists."""
    stack_name, separator, _ordinal = uid.rpartition(_UID_SEPARATOR)
    return stack_name if separator else ""


def record_uid(record: dict[str, Any]) -> str:
    """Return the pre-merge UID, or empty for absent/non-string values."""
    value = record.get(RECORD_UID_KEY)
    return value if isinstance(value, str) else ""


def stamp_record_uids(records: list[dict[str, Any]], stack_name: str) -> None:
    """Stamp missing UIDs in place, accepting a bare stack name or records filename.

    Existing UIDs survive; minted ordinals skip all occupied values.
    """
    normalized = stack_name_from_records_source(stack_name)
    _stamp_unique(
        records,
        key=RECORD_UID_KEY,
        getter=record_uid,
        mint=lambda ordinal: mint_record_uid(normalized, ordinal),
    )


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


def duplicate_record_uids(records: list[dict[str, Any]]) -> list[str]:
    """Return sorted colliding record UIDs; missing identities are checked separately."""
    counts = Counter(uid for uid in (record_uid(record) for record in records) if uid)
    return sorted(uid for uid, count in counts.items() if count > 1)


def record_issues(records: Any) -> list[Any] | None:
    """Normalize current {issues: [...]} and legacy bare-list record payloads.

    None means malformed; callers retain their own failure policy.
    """
    if isinstance(records, dict):
        issues = records.get("issues")
        return issues if isinstance(issues, list) else None
    return records if isinstance(records, list) else None


def record_issues_or_empty(records: Any) -> list[Any]:
    """Normalize a loaded per-stack records file to a bare issues list.

    A non-list load yields ``[]`` for a dict, otherwise the raw load.
    """
    issues = record_issues(records)
    if issues is None:
        return [] if isinstance(records, dict) else records
    return issues


def partition_record_sources(
    adjudicated: list[dict[str, Any]],
    adjudicated_sources: list[str],
    structural_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str]]:
    """Partition aligned records/sources by structural UID, preserving order and objects.

    Mismatched list lengths raise; UIDs outside structural_ids stay in language.
    """
    all_records: list[dict[str, Any]] = []
    record_sources: list[str] = []
    structural_records: list[dict[str, Any]] = []
    structural_sources: list[str] = []
    for rec, src in zip(adjudicated, adjudicated_sources, strict=True):
        if record_uid(rec) in structural_ids:
            structural_records.append(rec)
            structural_sources.append(src)
        else:
            all_records.append(rec)
            record_sources.append(src)
    return all_records, record_sources, structural_records, structural_sources
