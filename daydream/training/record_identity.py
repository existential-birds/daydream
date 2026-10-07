"""Versioned record identity binds run, trace segment, and host finding identity."""

import hashlib
import json


def record_finding_id(run_id: str, trajectory_id: str, segment_id: str, item_uid: str) -> str:
    """Versioned record identity preserves host findings independently of fingerprints."""
    for name, value in (
        ("run_id", run_id),
        ("trajectory_id", trajectory_id),
        ("segment_id", segment_id),
        ("item_uid", item_uid),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"record_finding_id: missing required component {name!r}")
    payload = json.dumps(["record-snapshot-v1", run_id, trajectory_id, segment_id, item_uid], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
