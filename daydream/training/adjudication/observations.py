"""Append-only JSONL judgments retain rationale, labeler, role, timestamps, rubric version, and
evidence digest. Canonically identical observations are no-ops, allowing interrupted labeling to
resume without rewriting history.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Mapping, Sequence

# Recognize versioned model/classifier identities so they cannot act as human adjudicators.
_MODEL_LABELER_RE = re.compile(
    r"(?:^|[-_])(?:claude|gpt|llm|model|classifier|anthropic|openai|codex|gemini)(?:$|[-_0-9])",
    re.IGNORECASE,
)

_DISPOSITIONS = frozenset({"accepted", "rejected", "ambiguous", "unknown"})
_ROLES = frozenset({"rater", "adjudicator", "model-suggested"})
_REQUIRED_FIELDS = (
    "record_id",
    "disposition",
    "evidence_digest",
    "evidence_digest_scheme",
    "labeler",
    "role",
    "rationale",
    "valid_at",
    "observed_at",
    "rubric_version",
)


def validate_observation(obs: Mapping[str, Any]) -> None:
    """Validate a judgment before planning or appending observation state."""
    for field in _REQUIRED_FIELDS:
        if field not in obs:
            raise ValueError(f"observation missing required field: {field}")
        if not isinstance(obs[field], str) or not obs[field]:
            raise ValueError(f"observation field must be a non-empty string: {field}")
    if obs.get("evidence") is None:
        # Reject missing evidence here because effective_adjudication requires it.
        raise ValueError("observation missing required field: evidence")
    record_id = obs["record_id"]
    if len(record_id) != 64 or any(c not in "0123456789abcdefABCDEF" for c in record_id):
        raise ValueError(f"record_id must be a 64-hex digest, got: {record_id!r}")
    if obs["disposition"] not in _DISPOSITIONS:
        raise ValueError(f"invalid disposition: {obs['disposition']}")
    if obs["role"] not in _ROLES:
        raise ValueError(f"invalid role: {obs['role']}")
    if obs["role"] == "adjudicator" and _MODEL_LABELER_RE.search(obs["labeler"]):
        raise ValueError(
            f"labeler {obs['labeler']!r} matches a model/LLM labeler pattern and cannot "
            "hold the adjudicator role: unreviewed model output is never an adjudicator"
        )
    for field in ("valid_at", "observed_at"):
        try:
            datetime.fromisoformat(obs[field])
        except ValueError as e:
            raise ValueError(f"observation field is not ISO-8601: {field}={obs[field]!r}") from e


def append_observation(store: Any, obs: Mapping[str, Any], *, run_id: str, item_uid: str) -> bool:
    """Append a validated judgment to the canonical record store."""
    from daydream.json_utils import canonical_json
    from daydream.training.labeler_versions import HUMAN_LABELER_VERSION

    validate_observation(obs)
    value = {
        "schema_version": "daydream.observation.v1",
        "run_id": run_id,
        "item_uid": item_uid,
        "valid_at": obs["valid_at"],
        "observed_at": obs["observed_at"],
        "source": "adjudication",
        "author": obs["labeler"],
        "role": obs["role"],
        "policy_version": HUMAN_LABELER_VERSION,
        "rubric_version": obs["rubric_version"],
        "evidence_digest": obs["evidence_digest"],
        "evidence_digest_scheme": obs["evidence_digest_scheme"],
        "semantic_evidence": obs["evidence"],
        "review_required": obs["role"] == "model-suggested",
        "payload": {
            "type": "finding-judgment",
            "disposition": obs["disposition"],
            "rationale": obs["rationale"],
            "record_id": obs["record_id"],
        },
    }
    identity = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return bool(store.append_observation({"observation_id": identity, **value}).committed)


def prior_adjudications(observations: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Resolve complete per-finding history for queue visibility and reopening."""
    from daydream.training.adjudication.precedence import effective_adjudication

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for observation in observations:
        grouped.setdefault(str(observation["record_id"]), []).append(observation)
    return {record_id: effective_adjudication(rows) for record_id, rows in grouped.items()}


def group_observations_by_record(
    observations: Sequence[Mapping[str, Any]],
    known_record_ids: set[str],
    context: str,
) -> dict[str, list[Mapping[str, Any]]]:
    """Group observations by ``record_id``, fail-closed on an unknown id."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for obs in observations:
        record_id = str(obs["record_id"])
        if record_id not in known_record_ids:
            raise ValueError(
                f"{context}: observation references record_id {record_id!r} "
                f"which is not in the adjudication queue "
                f"(observation evidence digest {obs.get('evidence_digest')!r})"
            )
        grouped.setdefault(record_id, []).append(obs)
    return grouped
