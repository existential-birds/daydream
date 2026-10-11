"""Minimal raw records and public snapshot reads shared by dataset tests."""
from typing import Any

from daydream.dataset import LocalRecordStore, SnapshotRecords, semantic_evidence_digest


def run_record(run_id: str = "run-1", **overrides: Any) -> dict[str, Any]:
    return {
        "schema_version": "daydream.run.v1", "run_id": run_id,
        "captured_at": "2026-10-04T10:00:00Z", "outcome": "success",
        "findings": {"status": "available", "value": {"claims": [],
            "items": [{"item_uid": "item:1", "source_uids": [], "fingerprint": "a" * 64}],
            "derivation": {}, "terminal_coverage": None}}, **overrides,
    }


def observation(observation_id: str = "judgment-1", **overrides: Any) -> dict[str, Any]:
    return {
        "schema_version": "daydream.observation.v3", "observation_id": observation_id,
        "run_id": "run-1", "item_uid": "item:1", "valid_at": "2026-10-04T11:00:00Z",
        "observed_at": "2026-10-04T12:00:00Z", "source": "manual", "author": "alice",
        "role": "rater", "policy_version": "policy-1", "rubric_version": "rubric-1",
        "evidence_digest": semantic_evidence_digest({"reply": "confirmed"}),
        "semantic_evidence": {"reply": "confirmed"},
        "payload": {"type": "finding-judgment", "disposition": "accepted", "rationale": "reproduced"}, **overrides,
    }


def read_records(store: LocalRecordStore, **cutoffs: Any) -> SnapshotRecords:
    return store.read_snapshot(store.select_snapshot(**{"observed_before": "2100-01-01T00:00:00Z", **cutoffs}))
