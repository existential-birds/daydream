"""Validate and serialize projector-format adjudication rows.

Recompute record identity, require all export fields and a nonempty evidence
digest. Fail with the offending key/record ID; never silently skip rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from daydream.json_utils import atomic_write_bytes, canonical_json as _canonical, umask_derived_mode
from daydream.training.record_identity import record_finding_id

__all__ = ["EXPORT_KEYS", "validate_export_rows", "write_export_rows"]

EXPORT_KEYS = (
    "record_id",
    "item_uid",
    "evidence_digest",
    "fingerprint",
    "disposition",
    "evidence",
    "exclusion_reason",
    "profile",
    "stack",
    "session_id",
    "trajectory_id",
    "segment_id",
    "tier",
    "posterior_eligible",
    "rubric_version",
)


def validate_export_rows(rows: list[dict[str, Any]]) -> None:
    """Validate the export shape; raise ``ValueError`` naming key + record_id.

    - Every key in ``EXPORT_KEYS`` must be present.
    - ``record_id`` must be recomputable from the four identity components
      ``(session_id, trajectory_id, segment_id, fingerprint)``.
    - ``evidence_digest`` must be a non-empty string.
    """
    for row in rows:
        shown_id = str(row.get("record_id", "<missing record_id>"))
        for key in EXPORT_KEYS:
            if key not in row:
                raise ValueError(f"export row for record_id {shown_id!r} is missing required key {key!r}")
        recomputed = record_finding_id(
            str(row["session_id"]), str(row["trajectory_id"]), str(row["segment_id"]), str(row["item_uid"])
        )
        if recomputed != str(row["record_id"]):
            raise ValueError(
                f"export row record_id {row['record_id']!r} does not match the identity "
                f"recomputed from (session_id={row['session_id']!r}, "
                f"trajectory_id={row['trajectory_id']!r}, segment_id={row['segment_id']!r}, "
                f"fingerprint={row['fingerprint']!r}): expected {recomputed!r}"
            )
        if not str(row["evidence_digest"]):
            raise ValueError(f"export row for record_id {shown_id!r} has an empty 'evidence_digest'")


def write_export_rows(rows: list[dict[str, Any]], out_path: Path) -> str:
    """Serialize rows canonically and write them to ``out_path``.

    Returns the SHA-256 of the written bytes.
    """
    payload = "".join(_canonical(row) + "\n" for row in rows).encode("utf-8")
    atomic_write_bytes(out_path, payload, fsync=False, dir_fsync=False, mode=umask_derived_mode())
    return hashlib.sha256(payload).hexdigest()
