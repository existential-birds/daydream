"""Deterministic disposable annotation exports from a frozen record snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore, SnapshotRecords
from daydream.json_utils import atomic_write_bytes, canonical_json
from daydream.training.adjudication.snapshot import build_canonical_record
from daydream.training.labeler_signals import resolution_from_dict
from daydream.training.record_evidence import sessions_from_snapshot, validate_output_path

_MANIFEST_FILENAME = "preview-manifest.json"
_ANNOTATIONS_FILENAME = "annotations.jsonl"


def annotation_records(records: SnapshotRecords) -> list[dict[str, Any]]:
    result = []
    for session in sessions_from_snapshot(records):
        for resolution in session["resolutions"]:
            item = build_canonical_record(
                {**session, "resolutions": [resolution]},
                resolution_from_dict(resolution),
                evidence_observed_at=records.snapshot["observed_before"],
                as_of=records.snapshot["valid_before"],
            )
            item["item_uid"] = resolution["item_uid"]
            result.append(item)
    return sorted(result, key=lambda r: r["record_id"])


def run_materialize(store_dir: Path, snapshot_id: str, out_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    validate_output_path(store_dir, out_dir)
    records = LocalRecordStore(store_dir).read_snapshot(snapshot_id)
    annotations = annotation_records(records)
    summary = {"snapshot_id": snapshot_id, "record_count": len(annotations)}
    if not dry_run:
        atomic_write_bytes(
            out_dir / _ANNOTATIONS_FILENAME,
            "".join(canonical_json(r) + "\n" for r in annotations).encode(),
            mode=0o600,
            fsync=True,
            dir_fsync=True,
        )
        atomic_write_bytes(
            out_dir / _MANIFEST_FILENAME,
            (canonical_json({"schema_version": "daydream.annotation-export.v2", **summary}) + "\n").encode(),
            mode=0o600,
            fsync=True,
            dir_fsync=True,
        )
    return summary
