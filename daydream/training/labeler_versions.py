"""Reply-labeling and evidence-digest version axes evolve independently of reward.REWARD_VERSION. This
module has no training-package imports, allowing archive consumers to use its stamps without an
adjudication import cycle.
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


def reply_evidence_digest(replies: list[dict[str, Any]]) -> str:
    """Hash canonical reply-evidence JSON with SHA-256. Sort by stringified reply_id, falling back to
    id then an empty string when absent; default missing body to an empty string. Serialize with
    sorted keys and default=str. Semantic fallbacks remain the caller's responsibility.
    """
    normalized = [
        {**reply, "body": reply.get("body", "")}
        for reply in sorted(replies, key=lambda r: str(r.get("reply_id", r.get("id", ""))))
    ]
    canonical = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
