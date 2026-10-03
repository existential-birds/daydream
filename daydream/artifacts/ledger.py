"""Durable destination ledger serialization and identity validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

from daydream.artifacts import filesystem
from daydream.artifacts.models import (
    _PUBLIC_LABELS,
    _SCHEMA_VERSION,
    ArtifactVisibilityError,
    DestinationDelivery,
    OutputLabel,
    _DestinationRecord,
)


def _write_baseline_manifest(transaction: Path, index: int, record: _DestinationRecord) -> None:
    """Persist one record's baseline manifest, which never changes after capture."""
    filesystem._atomic_json(
        transaction / f"destination-{index:04d}-baseline-manifest.json", filesystem._manifest_payload(record.baseline)
    )


def _write_destination_records(
    transaction: Path,
    records: Sequence[_DestinationRecord],
    *,
    include_published: bool,
    include_baseline: bool = False,
) -> None:
    """Rewrite the destination ledger.

    Baseline manifests are immutable once captured, so they are only written
    when this transaction has not seen them yet (``include_baseline``).
    """
    registry: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if include_baseline:
            _write_baseline_manifest(transaction, index, record)
        if include_published:
            filesystem._atomic_json(
                transaction / f"destination-{index:04d}-published-manifest.json",
                filesystem._manifest_payload(record.published),
            )
        registry.append(
            {
                "index": index,
                **{
                    field.name: getattr(record, field.name)
                    for field in fields(_DestinationRecord)
                    if field.name not in {"baseline", "published"}
                },
            }
        )
    filesystem._atomic_json(
        transaction / "destinations.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "includes_published": include_published,
            "destinations": registry,
        },
    )


_DESTINATION_KEYS = frozenset(
    "index record_id requested base relative label delivery expected_kind baseline_state "
    "missing_parents expected_dev expected_ino prepared_sha256 installed_sha256".split()
)


def _load_destination_records(transaction: Path, *, include_published: bool) -> tuple[_DestinationRecord, ...]:
    raw_destinations = filesystem._load_ledger(
        transaction / "destinations.json",
        items_key="destinations",
        message="artifact destination registry is malformed",
        identity=(("includes_published", include_published),),
    )
    result: list[_DestinationRecord] = []
    for expected_index, raw in enumerate(raw_destinations):
        if not isinstance(raw, dict) or set(raw) != _DESTINATION_KEYS:
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        index = raw["index"]
        record_id = raw["record_id"]
        requested = raw["requested"]
        base = raw["base"]
        relative = raw["relative"]
        label_value = raw["label"]
        delivery_value = raw["delivery"]
        expected_kind = raw["expected_kind"]
        baseline_state = raw["baseline_state"]
        missing_raw = raw["missing_parents"]
        expected_dev = raw["expected_dev"]
        expected_ino = raw["expected_ino"]
        prepared_sha256 = raw["prepared_sha256"]
        installed_sha256 = raw["installed_sha256"]
        if (
            type(index) is not int
            or index != expected_index
            or not isinstance(record_id, str)
            or record_id != f"destination-{index:04d}"
            or not isinstance(requested, str)
            or not isinstance(base, str)
            or not isinstance(relative, str)
            or not isinstance(missing_raw, list)
            or not all(isinstance(value, str) for value in missing_raw)
            or expected_kind not in ("file", "directory")
            or baseline_state not in ("absent", "file", "directory")
            or not all(value is None or type(value) is int for value in (expected_dev, expected_ino))
            or not all(value is None or filesystem._is_sha256(value) for value in (prepared_sha256, installed_sha256))
        ):
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        filesystem._validate_relative_name(relative)
        requested_path = Path(requested)
        base_path = Path(base)
        if (
            not requested_path.is_absolute()
            or filesystem._absolute_lexical(requested_path) != requested_path
            or not base_path.is_absolute()
            or filesystem._absolute_lexical(base_path) != base_path
            or base_path / relative != requested_path
        ):
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        try:
            label = OutputLabel(label_value)
            delivery = DestinationDelivery(delivery_value)
        except (TypeError, ValueError) as exc:
            raise ArtifactVisibilityError("artifact destination registry is malformed") from exc
        if label in _PUBLIC_LABELS:
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        baseline = filesystem._parse_manifest(transaction / f"destination-{index:04d}-baseline-manifest.json")
        published = (
            filesystem._parse_manifest(transaction / f"destination-{index:04d}-published-manifest.json")
            if include_published
            else ()
        )
        if not filesystem._entries_belong_to(relative, baseline) or not filesystem._entries_belong_to(
            relative, published
        ):
            raise ArtifactVisibilityError("artifact destination manifest identity is malformed")
        if baseline_state == "absent" and baseline:
            raise ArtifactVisibilityError("artifact destination baseline identity is malformed")
        root_entry = filesystem._entry_at(baseline, relative)
        if baseline_state != "absent" and (root_entry is None or root_entry.kind != baseline_state):
            raise ArtifactVisibilityError("artifact destination baseline identity is malformed")
        missing_parents = tuple(cast(list[str], missing_raw))
        for missing in missing_parents:
            filesystem._validate_relative_name(missing)
            if Path(missing) not in Path(relative).parents:
                raise ArtifactVisibilityError("artifact destination parent identity is malformed")
        if any(filesystem._overlaps(requested_path, Path(existing.requested)) for existing in result):
            raise ArtifactVisibilityError("artifact destination registry contains overlapping paths")
        # Shape and field invariants have all been checked above. Keep the
        # persisted field names when constructing the record instead of
        # maintaining a second positional serialization order.
        values = {key: value for key, value in raw.items() if key != "index"}
        values.update(label=label, delivery=delivery, baseline=baseline,
                      missing_parents=missing_parents, published=published)
        result.append(_DestinationRecord(**cast(dict[str, Any], values)))
    return tuple(result)


def _persist_live_destination_record(transaction: Path, updated: _DestinationRecord) -> None:
    registry = filesystem._load_json(transaction / "destinations.json")
    includes_published = registry.get("includes_published")
    if type(includes_published) is not bool:
        raise ArtifactVisibilityError("artifact destination registry is malformed")
    records = list(_load_destination_records(transaction, include_published=includes_published))
    for index, record in enumerate(records):
        if record.record_id == updated.record_id:
            records[index] = updated
            _write_destination_records(transaction, records, include_published=includes_published)
            return
    raise ArtifactVisibilityError("live external destination ledger identity is missing")
