"""Disposable evidence-digest ledgers over immutable record snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore
from daydream.json_utils import atomic_write_bytes, canonical_json
from daydream.training.adjudication.observations import prior_adjudications
from daydream.training.adjudication.queue import build_queue
from daydream.training.record_evidence import finding_observations, sessions_from_snapshot, validate_output_path


def preview_ledger_digest(ledger: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json({k: ledger[k] for k in ("snapshot_id", "items")}).encode()).hexdigest()


def run_preview(store_dir: Path, snapshot_id: str, ledger_path: Path) -> dict[str, Any]:
    validate_output_path(store_dir, ledger_path)
    records = LocalRecordStore(store_dir).read_snapshot(snapshot_id)
    items = build_queue(
        sessions_from_snapshot(records, overlay_judgments=False),
        prior_observations=prior_adjudications([o for o in finding_observations(records) if o["role"] != "automatic"]),
    )
    ledger = {
        "snapshot_id": snapshot_id,
        "items": [
            {k: item[k] for k in ("disposition", "evidence_digest", "fingerprint", "record_id", "status")}
            for item in items
        ],
    }
    ledger["ledger_digest"] = preview_ledger_digest(ledger)
    old = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {"items": []}
    prior = {i["record_id"]: i["evidence_digest"] for i in old["items"]}
    drifted = [
        i["record_id"] for i in items if i["record_id"] in prior and prior[i["record_id"]] != i["evidence_digest"]
    ]
    atomic_write_bytes(ledger_path, (canonical_json(ledger) + "\n").encode(), mode=0o600)
    return {
        "snapshot_id": snapshot_id,
        "ledger_digest": ledger["ledger_digest"],
        "item_count": len(items),
        "drifted_record_ids": drifted,
    }
