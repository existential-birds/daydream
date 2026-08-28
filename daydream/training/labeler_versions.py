"""Version constants for the reply-labeling pipeline and reply evidence digest.

Independent version axes (M13): rubric schema, labeler policy, reply classifier,
and evidence digest format each evolve separately from ``reward.REWARD_VERSION``.
This module imports nothing from the rest of the training package.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

RUBRIC_SCHEMA_VERSION = "980-rubric-r2"
LABELER_POLICY_VERSION = "980-policy-r1"
REPLY_CLASSIFIER_VERSION = "980-classifier-r1"
REPLY_EVIDENCE_DIGEST_FORMAT = "sha256/1"


def reply_evidence_digest(replies: list[dict[str, Any]]) -> str:
    """Stable sha256 hexdigest over the canonical reply-evidence JSON.

    Replies are sorted by ``id`` and serialized with sorted keys. Missing keys
    contribute ``""`` (via ``sort_keys``-safe defaults) rather than raising;
    semantic fallbacks are the caller's contract, never applied here.
    """
    normalized = [
        {**reply, "body": reply.get("body", "")}
        for reply in sorted(replies, key=lambda r: r.get("id", ""))
    ]
    canonical = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
