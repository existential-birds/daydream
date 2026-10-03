"""Host-assigned identities and provenance for review records and merged items.

Pre-merge records use deterministic stack:ordinal UIDs, stamped after schema
validation. Merged items have separate durable identities and source UID lists;
display IDs may change during normalization. Content fingerprints identify
similar defects, whereas these UIDs identify particular records/items.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    """Read the current issues envelope; malformed payloads return None."""
    issues = records.get('issues') if isinstance(records, dict) else None
    return issues if isinstance(issues, list) else None


def record_issues_or_empty(records: Any) -> list[Any]:
    """Read current records, contributing no evidence for a malformed envelope."""
    return record_issues(records) or []


@dataclass
class RecordPool:
    """Current scoped envelopes, ordered records and their authoritative file paths."""

    scopes: dict[str, dict[str, Any]]
    paths: dict[str, Path]

    @property
    def language(self) -> list[dict[str, Any]]:
        return [record for scope, envelope in self.scopes.items() if scope != "structure"
                for record in envelope["issues"]]

    @property
    def structural(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self.scopes.get("structure", {}).get("issues", []))

    @property
    def records(self) -> list[dict[str, Any]]:
        return self.language + self.structural

    @property
    def language_paths(self) -> list[Path]:
        return [path for scope, path in self.paths.items() if scope != "structure"]

    @property
    def structural_path(self) -> Path | None:
        return self.paths.get("structure")

    def replace(self, records: Iterable[dict[str, Any]]) -> None:
        by_scope: dict[str, list[dict[str, Any]]] = {scope: [] for scope in self.scopes}
        for record in records:
            scope = stack_name_from_uid(record_uid(record))
            if scope not in by_scope:
                raise ValueError(f"Record UID does not identify a loaded review scope: {record_uid(record)!r}")
            by_scope[scope].append(record)
        for scope, selected in by_scope.items():
            self.scopes[scope]["issues"] = selected

    def save(self) -> None:
        from daydream.json_utils import atomic_write_json

        for scope, envelope in self.scopes.items():
            atomic_write_json(self.paths[scope], envelope)

    def reload(self, analyzed_revision: dict[str, Any]) -> RecordPool:
        import json

        from daydream.phases.review import valid_record_artifact

        scopes = {}
        for scope, path in self.paths.items():
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if not valid_record_artifact(envelope, scope_id=scope, analyzed_revision=analyzed_revision):
                raise ValueError(f"Invalid restored records for scope {scope}")
            scopes[scope] = envelope
        restored = RecordPool(scopes, self.paths)
        if duplicate_record_uids(restored.records):
            raise ValueError("Restored records contain duplicate UIDs")
        return restored
