"""Durable destination ledger serialization and identity validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from daydream.artifacts import filesystem
from daydream.artifacts.models import (
    _PUBLIC_LABELS,
    _SCHEMA_VERSION,
    ArtifactVisibilityError,
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


_DESTINATION_ADMISSION = TypeAdapter(_DestinationRecord)

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
        if (
            type(index) is not int
            or index != expected_index
            or not isinstance(raw["missing_parents"], list)
            or not all(isinstance(raw[name], str) for name in ("label", "delivery"))
            or not all(raw[name] is None or type(raw[name]) is int for name in ("expected_dev", "expected_ino"))
        ):
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        try:
            record = _DESTINATION_ADMISSION.validate_python(
                {
                    **{key: value for key, value in raw.items() if key != "index"},
                    "baseline": (),
                }
            )
        except ValidationError:
            raise ArtifactVisibilityError("artifact destination registry is malformed") from None
        if (
            record.record_id != f"destination-{index:04d}"
            or not all(
                value is None or filesystem._is_sha256(value)
                for value in (record.prepared_sha256, record.installed_sha256)
            )
        ):
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        relative = record.relative
        filesystem._validate_relative_name(relative)
        requested_path = Path(record.requested)
        base_path = Path(record.base)
        if (
            not requested_path.is_absolute()
            or filesystem._absolute_lexical(requested_path) != requested_path
            or not base_path.is_absolute()
            or filesystem._absolute_lexical(base_path) != base_path
            or base_path / relative != requested_path
        ):
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        if record.label in _PUBLIC_LABELS:
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
        if record.baseline_state == "absent" and baseline:
            raise ArtifactVisibilityError("artifact destination baseline identity is malformed")
        root_entry = filesystem._entry_at(baseline, relative)
        if record.baseline_state != "absent" and (root_entry is None or root_entry.kind != record.baseline_state):
            raise ArtifactVisibilityError("artifact destination baseline identity is malformed")
        for missing in record.missing_parents:
            filesystem._validate_relative_name(missing)
            if Path(missing) not in Path(relative).parents:
                raise ArtifactVisibilityError("artifact destination parent identity is malformed")
        if any(filesystem._overlaps(requested_path, Path(existing.requested)) for existing in result):
            raise ArtifactVisibilityError("artifact destination registry contains overlapping paths")
        result.append(replace(record, baseline=baseline, published=published))
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
