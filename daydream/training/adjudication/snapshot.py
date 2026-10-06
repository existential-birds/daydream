"""Pure canonical finding serializer and evidence digests shared by preview and harvest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from daydream.training._immutable_json import thaw_json
from daydream.training.corpus_projection.provenance import extract_provenance
from daydream.training.labeler_versions import (
    ADJUDICATION_LABELER_VERSION,
    ANNOTATION_SNAPSHOT_SCHEMA_VERSION,
    HUMAN_LABELER_VERSION,
    REPLY_CLASSIFIER_VERSION,
    reply_evidence_digest,
)
from daydream.training.record_identity import record_finding_id

__all__ = [
    "ANNOTATION_SNAPSHOT_SCHEMA_VERSION",
    "build_canonical_record",
    "record_evidence_digest",
]


def record_evidence_digest(
    per_finding_evidence_lists: Sequence[Sequence[Mapping[str, Any]]],
) -> str | None:
    """Use the shared reply-evidence digest over flattened session evidence.

    No replies yields None, retaining the distinct digest-less dedup identity.
    """
    evidence = [thaw_json(entry) for per_finding in per_finding_evidence_lists for entry in per_finding]
    return reply_evidence_digest(evidence) if evidence else None


def build_canonical_record(
    session: Mapping[str, Any],
    resolution: Any,
    *,
    evidence_observed_at: str,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Build the canonical per-finding annotation record for one resolution.

    ``record_id`` is always recomputed via ``record_identity.record_finding_id`` —
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
    rows = list(session.get("resolutions") or [])
    if len(rows) != 1:
        rows = [row for row in rows if row.get("fingerprint") == fingerprint]
    if len(rows) != 1:
        raise ValueError(
            f"build_canonical_record: session {session_id!r} has {len(rows)} resolution rows "
            f"for fingerprint {fingerprint!r}; expected exactly 1"
        )
    provenance = extract_provenance(rows[0])

    record: dict[str, Any] = {
        "record_id": record_finding_id(session_id, trajectory_id, segment_id, rows[0]["item_uid"]),
        "fingerprint": fingerprint,
        "disposition": resolution.disposition,
        "evidence": [thaw_json(entry) for entry in resolution.evidence],
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
        # Self-contained session view: consumers that rebuild the queue or the
        # projection from canonical records (``project_findings``/``build_queue``)
        # consume the session shape (``resolutions`` list), and the materialized
        # per-finding record must be directly consumable without a second shape.
        "resolutions": [dict(rows[0])],
    }
    if rows[0].get("item_uid"):
        record["item_uid"] = rows[0]["item_uid"]
    if as_of is not None:
        record["as_of"] = as_of
    return record
