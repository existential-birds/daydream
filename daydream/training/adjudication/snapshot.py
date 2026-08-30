"""Shared serializer for the per-finding annotation snapshot pipeline (issue #1055).

One canonical per-finding record shape + evidence digest, consumed by both
preview materialization (``materialize.py``) and canonical harvest
(``canonical.py``). Pure module: no I/O, no wall-clock reads — determinism is
by construction (C4).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from daydream.training.corpus_v2.identity import record_id
from daydream.training.corpus_v2.provenance import extract_provenance
from daydream.training.labeler_versions import (
    ADJUDICATION_LABELER_VERSION,
    ANNOTATION_SNAPSHOT_SCHEMA_VERSION,
    HUMAN_LABELER_VERSION,
    REPLY_CLASSIFIER_VERSION,
    reply_evidence_digest,
)

__all__ = [
    "ANNOTATION_SNAPSHOT_SCHEMA_VERSION",
    "build_canonical_record",
    "record_evidence_digest",
    "snapshot_id",
]

# The K2 preview-pin components: the snapshot id is a content-addressed digest
# over exactly these fields, so idempotence and drift detection are by
# construction — any pin change produces a new snapshot id (AC 8).
_PIN_FIELDS = (
    "curation_id",
    "sanitized_hub_commit",
    "source_hub_commit",
    "archive_index_digest",
    "evidence_observed_at",
    "as_of",
    "labeler_version",
    "rubric_version",
    "classifier_version",
)


def record_evidence_digest(per_finding_evidence_lists: Sequence[Sequence[dict[str, Any]]]) -> str:
    """Digest over the session's flattened per-finding reply evidence.

    Delegates to ``labeler_versions.reply_evidence_digest`` — the shared
    implementation, never a re-implementation (K4/K5; spike-verified
    byte-identical to ``training/harvest.py:_reply_evidence_digest``).
    """
    evidence = [entry for per_finding in per_finding_evidence_lists for entry in per_finding]
    return reply_evidence_digest(evidence)


def build_canonical_record(
    session: Mapping[str, Any],
    resolution: Any,
    *,
    evidence_observed_at: str,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Build the canonical per-finding annotation record for one resolution.

    ``record_id`` is always recomputed via ``corpus_v2.identity.record_id`` —
    never trusted from any stored copy. Missing required fields fail closed
    with ``ValueError`` naming the offending field (no fallback coercion).
    """
    fingerprint = resolution.fingerprint
    evidence_digest = resolution.evidence_digest
    if not isinstance(evidence_digest, str) or not evidence_digest:
        raise ValueError(
            f"build_canonical_record: resolution for fingerprint {fingerprint!r} is missing "
            "required field 'evidence_digest'"
        )

    session_id = session["session_id"]
    trajectory_id = session["trajectory_id"]
    segment_id = session["segment_id"]

    # Provenance comes from the session's resolution row joined by fingerprint.
    rows = [row for row in session.get("resolutions") or [] if row.get("fingerprint") == fingerprint]
    if len(rows) != 1:
        raise ValueError(
            f"build_canonical_record: session {session_id!r} has {len(rows)} resolution rows "
            f"for fingerprint {fingerprint!r}; expected exactly 1"
        )
    provenance = extract_provenance(rows[0])

    record: dict[str, Any] = {
        "record_id": record_id(session_id, trajectory_id, segment_id, fingerprint),
        "fingerprint": fingerprint,
        "disposition": resolution.disposition,
        "evidence": list(resolution.evidence),
        "evidence_digest": evidence_digest,
        "session_id": session_id,
        "trajectory_id": trajectory_id,
        "segment_id": segment_id,
        "profile": provenance["profile"],
        "stack": provenance["stack"],
        "rubric_version": ADJUDICATION_LABELER_VERSION,
        "classifier_version": REPLY_CLASSIFIER_VERSION,
        "labeler_version": HUMAN_LABELER_VERSION,
        "schema_version": f"annotation-snapshot/{ANNOTATION_SNAPSHOT_SCHEMA_VERSION}",
        "evidence_observed_at": evidence_observed_at,
    }
    if as_of is not None:
        record["as_of"] = as_of
    return record


def snapshot_id(pin: Mapping[str, str]) -> str:
    """Content-addressed snapshot id: sha256 over canonical JSON of the pin.

    The digest covers exactly the ``_PIN_FIELDS`` components with sorted keys
    and compact separators, so the caller's key insertion order cannot affect
    the id. Any missing or empty component raises ``ValueError`` naming it.
    """
    components = {}
    for field in _PIN_FIELDS:
        value = pin.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"snapshot_id: pin is missing required component {field!r}")
        components[field] = value
    canonical = json.dumps(
        components, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
