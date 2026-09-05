"""Data-driven allowlist of known labeler-version strings.

Seeded from the version axes in ``daydream/training/labeler_versions.py``
(KD3): a value set that can be extended without touching import logic. This
module imports nothing beyond the constants module — no cycle into the
training adjudication package.

Any imported observation whose version axes fall outside this allowlist (or
stamped ``"legacy"``) still imports as evidence but is never gold-eligible
(M6): unknown provenance must never be decisive.
"""

from __future__ import annotations

from daydream.training.labeler_versions import (
    ADJUDICATION_LABELER_VERSION,
    HUMAN_LABELER_VERSION,
    LABELER_POLICY_VERSION,
    REPLY_CLASSIFIER_VERSION,
    RUBRIC_SCHEMA_VERSION,
)

__all__ = ["KNOWN_LABELER_VERSIONS", "STALE_LEGACY"]

# Legacy-schema rows (missing version columns) surface this sentinel string
# (Assumption 4) and are never gold-eligible.
STALE_LEGACY = "legacy"

KNOWN_LABELER_VERSIONS: frozenset[str] = frozenset(
    {
        RUBRIC_SCHEMA_VERSION,
        LABELER_POLICY_VERSION,
        REPLY_CLASSIFIER_VERSION,
        ADJUDICATION_LABELER_VERSION,
        HUMAN_LABELER_VERSION,
    }
)
