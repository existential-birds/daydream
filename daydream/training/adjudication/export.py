"""Projector-format export + dry-run validation for adjudication results (issue #984, Task 12).

One serializer, two entry points: the canonical harvest
(:func:`daydream.training.adjudication.harvest.run_harvest`) and the
``corpus adjudicate export`` CLI verb both write through
:func:`write_export_rows`, so the on-disk shape — the
``corpus_v2.projector.project_findings`` adjudication entry shape plus
``record_id``/``evidence_digest`` — is produced by exactly one code path.

``validate_export_rows`` is the dry-run gate: every row's required keys are
checked, ``record_id`` is recomputed from its four identity components, and a
non-empty ``evidence_digest`` is required. A violation raises ``ValueError``
naming the offending key and ``record_id`` — never a silent skip, which could
overstate gold coverage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from daydream.training.corpus_v2.identity import record_id as compute_record_id

__all__ = ["EXPORT_KEYS", "validate_export_rows", "write_export_rows"]

EXPORT_KEYS = (
    "record_id",
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

# Keys the projector consumer needs beyond the adjudication entry shape.
_MINIMAL_CONSUMER_KEYS = (
    "record_id",
    "fingerprint",
    "disposition",
    "evidence",
    "evidence_digest",
    "exclusion_reason",
)


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
                raise ValueError(
                    f"export row for record_id {shown_id!r} is missing required key {key!r}"
                )
        recomputed = compute_record_id(
            str(row["session_id"]),
            str(row["trajectory_id"]),
            str(row["segment_id"]),
            str(row["fingerprint"]),
        )
        if recomputed != str(row["record_id"]):
            raise ValueError(
                f"export row record_id {row['record_id']!r} does not match the identity "
                f"recomputed from (session_id={row['session_id']!r}, "
                f"trajectory_id={row['trajectory_id']!r}, segment_id={row['segment_id']!r}, "
                f"fingerprint={row['fingerprint']!r}): expected {recomputed!r}"
            )
        if not str(row["evidence_digest"]):
            raise ValueError(
                f"export row for record_id {shown_id!r} has an empty 'evidence_digest'"
            )


def write_export_rows(rows: list[dict[str, Any]], out_path: Path) -> str:
    """Serialize rows canonically and write them atomically to ``out_path``.

    Temp-file + ``os.replace`` so an interrupted writer leaves no partial
    export. Returns the SHA-256 of the written bytes.
    """
    import hashlib

    payload = "".join(_canonical(row) + "\n" for row in rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, out_path)
    return hashlib.sha256(out_path.read_bytes()).hexdigest()
