"""Version constants for the reply-labeling pipeline and reply evidence digest.

Independent version axes (M13): rubric schema, labeler policy, reply classifier,
adjudication labeler, human labeler, annotation snapshot schema, and evidence
digest format each evolve separately from ``reward.REWARD_VERSION``.
This module imports nothing from the rest of the training package, so the
archive import path can depend on it without a cycle into the training
adjudication package.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

RUBRIC_SCHEMA_VERSION = "980-rubric-r2"
LABELER_POLICY_VERSION = "980-policy-r1"
REPLY_CLASSIFIER_VERSION = "980-classifier-r1"
ADJUDICATION_LABELER_VERSION = "984-adjudicate-r1"
HUMAN_LABELER_VERSION = "1055-human-r1"
ANNOTATION_SNAPSHOT_SCHEMA_VERSION = "1055-snapshot-r1"

# Data-driven allowlist of known labeler-version strings (KD3): a value set
# that can be extended without touching import logic. Any imported observation
# whose version axes fall outside this allowlist (or stamped ``STALE_LEGACY``)
# still imports as evidence but is never gold-eligible (M6): unknown provenance
# must never be decisive.
KNOWN_LABELER_VERSIONS: frozenset[str] = frozenset(
    {
        RUBRIC_SCHEMA_VERSION,
        LABELER_POLICY_VERSION,
        REPLY_CLASSIFIER_VERSION,
        ADJUDICATION_LABELER_VERSION,
        HUMAN_LABELER_VERSION,
    }
)

# Legacy-schema rows (missing version columns) surface this sentinel string
# (Assumption 4) and are never gold-eligible.
STALE_LEGACY = "legacy"


def reply_evidence_digest(replies: list[dict[str, Any]]) -> str:
    """Stable sha256 hexdigest over the canonical reply-evidence JSON.

    Replies are sorted by ``reply_id`` (falling back to ``id`` for raw
    review-comment dicts), so the order normalization actually runs on the
    evidence entries this pipeline passes in — keyed ``reply_id``/``body_sha256``
    — and then serialized with sorted keys. Missing keys contribute ``""``
    (via ``sort_keys``-safe defaults) rather than raising; semantic fallbacks
    are the caller's contract, never applied here.
    """
    normalized = [
        {**reply, "body": reply.get("body", "")}
        for reply in sorted(replies, key=lambda r: str(r.get("reply_id", r.get("id", ""))))
    ]
    canonical = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
