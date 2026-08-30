"""Append-only observation store for human adjudication judgments.

Each observation is one JSONL line with full provenance (rationale, labeler,
role, timestamps, rubric version, evidence digest). Lines are never rewritten
or deleted; appending an observation whose canonical JSON already exists is a
no-op, so interrupted labeling sessions can safely re-run.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

# Labeler names that identify a model/LLM classifier rather than a human
# (mirrors the versioned classifier identity convention, e.g.
# ``REPLY_CLASSIFIER_VERSION`` consumers stamping ``claude-classifier``).
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
    "labeler",
    "role",
    "rationale",
    "valid_at",
    "observed_at",
    "rubric_version",
)


def _validate(obs: Mapping[str, Any]) -> None:
    for field in _REQUIRED_FIELDS:
        if field not in obs:
            raise ValueError(f"observation missing required field: {field}")
        if not isinstance(obs[field], str) or not obs[field]:
            raise ValueError(f"observation field must be a non-empty string: {field}")
    if obs.get("evidence") is None:
        # The resolver (precedence.effective_adjudication) hard-requires
        # evidence, so the store must reject evidence-less rows up front
        # instead of accepting rows the resolver later crashes on.
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


def _canonical(obs: Mapping[str, Any]) -> str:
    return json.dumps(obs, sort_keys=True)


def append_observation(path: Path, obs: Mapping[str, Any]) -> None:
    """Validate then append one observation line; idempotent for identical lines.

    Validation happens before any bytes are written, so a failed append leaves
    the file byte-identical. Never rewrites or deletes existing lines.
    """
    _validate(obs)
    if obs["role"] == "model-suggested":
        # Model-suggested labels are always review-required; the writer forces
        # the flag so callers cannot omit or clear it.
        obs = {**obs, "review_required": True}
    line = _canonical(obs)
    if path.exists():
        for existing in path.read_text(encoding="utf-8").splitlines():
            if existing.strip() == line:
                return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def load_observations(path: Path) -> list[dict[str, Any]]:
    """Return observations in append order; a missing file is an empty store."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
