"""Verify a disposable annotation export against canonical frozen evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore
from daydream.json_utils import atomic_write_bytes, canonical_json
from daydream.training.adjudication.materialize import _ANNOTATIONS_FILENAME, _MANIFEST_FILENAME, annotation_records
from daydream.training.record_evidence import validate_output_path


class AnnotationDriftError(ValueError):
    def __init__(self, message: str, requeued_record_ids: list[str]) -> None:
        super().__init__(message)
        self.requeued_record_ids = requeued_record_ids


def _evidence_after_as_of(record: Mapping[str, Any], as_of: str | None) -> bool:
    if not as_of:
        return False
    pin = datetime.fromisoformat(as_of)
    return any(
        isinstance(e, Mapping) and e.get("created_at") and datetime.fromisoformat(str(e["created_at"])) > pin
        for e in record.get("evidence") or []
    )


def read_jsonl(path: Path, *, missing: str, invalid: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(missing)
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        raise ValueError(invalid) from None


def run_canonical_harvest(store_dir: Path, snapshot_id: str, materialize_dir: Path) -> dict[str, Any]:
    """Fail before export writes on changed evidence or an incomplete population.

    Judgments already live in LocalRecordStore; materialization is a disposable
    projection, never a second authoritative annotation store.
    """
    validate_output_path(store_dir, materialize_dir)
    manifest_path = materialize_dir / _MANIFEST_FILENAME
    pin = json.loads(manifest_path.read_text())
    if pin.get("schema_version") != "daydream.annotation-export.v2":
        raise ValueError("invalid annotation export schema")
    old = read_jsonl(
        materialize_dir / _ANNOTATIONS_FILENAME,
        missing="annotation export missing",
        invalid="invalid annotation export",
    )
    fresh = annotation_records(LocalRecordStore(store_dir).read_snapshot(snapshot_id))
    previous = {r["record_id"]: r for r in old}
    if len(previous) != len(old):
        raise ValueError("duplicate finding identities in annotation export")
    current = {r["record_id"]: r for r in fresh}
    drifted = sorted(
        set(previous).symmetric_difference(current)
        | {
            identity
            for identity in previous.keys() & current.keys()
            if previous[identity]["evidence_digest"] != current[identity]["evidence_digest"]
        }
    )
    if drifted:
        raise AnnotationDriftError("annotation evidence changed; rematerialize and adjudicate", drifted)
    atomic_write_bytes(
        materialize_dir / _ANNOTATIONS_FILENAME,
        "".join(canonical_json(r) + "\n" for r in fresh).encode(),
        mode=0o600,
        fsync=True,
        dir_fsync=True,
    )
    return {
        "snapshot_id": snapshot_id,
        "record_count": len(fresh),
        "evidence_after_as_of": [r["record_id"] for r in fresh if _evidence_after_as_of(r, r.get("as_of"))],
    }
