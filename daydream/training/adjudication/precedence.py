"""Adjudication precedence and gold eligibility for immutable per-finding observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from daydream.json_utils import canonical_json
from daydream.training.dispositions import DECISIVE_DISPOSITIONS as DECISIVE_DISPOSITIONS

HUMAN_ROLES = frozenset({"rater", "adjudicator"})


def _is_human(obs: Mapping[str, Any]) -> bool:
    return obs.get("role") in HUMAN_ROLES


def observation_recency(observation: Mapping[str, Any]) -> tuple[datetime, str]:
    """Order source generations by observation time and immutable identity."""
    return datetime.fromisoformat(str(_required(observation, "observed_at"))), str(observation["observation_id"])


def _sorted_by_recency(observations: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        observations,
        key=lambda o: (
            datetime.fromisoformat(str(_required(o, "observed_at"))),
            str(o.get("labeler", o.get("author", ""))),
            str(o.get("observation_id", "")),
            canonical_json(o),
        ),
    )


def retained_reply_content(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Retain the latest envelope for every source reply in one evidence generation.

    Callers restrict observations to the selected semantic digest first. A judgment
    without captured text does not erase content acquired in an earlier observation.
    Capture-only enrichment replaces the retained representation deterministically.
    """
    captures: dict[tuple[str, str | None], Any] = {}
    correction = None
    for observation in _sorted_by_recency(observations):
        singular = observation.get("correction")
        if singular is not None:
            correction = singular
        for capture in [*(observation.get("reply_captures") or []), *([singular] if singular else [])]:
            captures[(capture["source_reply_id"], capture.get("body_sha256"))] = capture
    return {
        "reply_captures": [captures[key] for key in sorted(captures, key=lambda key: (key[0], key[1] or ""))],
        "correction": correction,
    }


def _required(obs: Mapping[str, Any], field: str) -> Any:
    value = obs.get(field)
    if value is None:
        msg = f"observation missing required field {field!r} for record_id {obs.get('record_id')!r}"
        raise ValueError(msg)
    return value


def has_rater_conflict(observations: Sequence[Mapping[str, Any]]) -> bool:
    """Detect differing human dispositions for the same record_id/evidence_digest. Only a decisive
    adjudicator disposition clears that conflict.
    """
    rater_dispositions: dict[tuple[str, str], set[str]] = {}
    adjudicator_dispositions: dict[tuple[str, str], str | None] = {}
    for obs in observations:
        if not _is_human(obs):
            continue
        key = (str(_required(obs, "record_id")), str(_required(obs, "evidence_digest")))
        if obs.get("role") == "adjudicator":
            disposition = str(_required(obs, "disposition"))
            adjudicator_dispositions[key] = disposition if disposition in DECISIVE_DISPOSITIONS else None
        else:
            rater_dispositions.setdefault(key, set()).add(str(_required(obs, "disposition")))
    for key, dispositions in rater_dispositions.items():
        if len(dispositions) < 2:
            continue
        if key in adjudicator_dispositions and adjudicator_dispositions[key] is not None:
            continue
        return True
    return False


def reopen_on_digest_change(observation: Mapping[str, Any], current_digest: str) -> bool:
    """Reopen when the pinned and current digest strings differ exactly, without normalization or
    fallback. Judgments cannot transfer silently to changed evidence.
    """
    return str(observation["evidence_digest"]) != str(current_digest)


def effective_adjudication(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Resolve current automatic evidence by adjudicator, else human rater, else automatic entry.

    Source edits reopen judgments; capture-only changes retain matching human labels.
    Recency is observed_at, labeler, then observation identity. Empty input raises ValueError.

    Return the effective judgment and provenance plus conflict, review_required, and gold_eligible.
    Gold requires a decisive disposition, nonempty evidence, no unresolved rater conflict, and no
    review requirement.
    """
    if not observations:
        msg = "effective_adjudication called with empty observation list for record_id"
        raise ValueError(msg)

    record_id = _required(observations[0], "record_id")
    ordered = _sorted_by_recency(observations)
    automatic = [o for o in ordered if o.get("role") == "automatic"]
    if automatic:
        current_digest = _required(max(automatic, key=observation_recency), "evidence_digest")
        ordered = [o for o in ordered if _required(o, "evidence_digest") == current_digest]

    adjudicators = [o for o in ordered if o.get("role") == "adjudicator"]
    human_raters = [o for o in ordered if o.get("role") == "rater"]
    automatic = [o for o in ordered if not _is_human(o)]

    if adjudicators:
        effective = adjudicators[-1]
    elif human_raters:
        effective = human_raters[-1]
    elif automatic:
        effective = automatic[-1]
    else:  # pragma: no cover - role sets above are exhaustive over HUMAN_ROLES
        msg = f"no resolvable observation for record_id {record_id!r}"
        raise ValueError(msg)

    disposition = str(_required(effective, "disposition"))
    evidence_digest = str(_required(effective, "evidence_digest"))
    evidence = _required(effective, "evidence")
    conflict = has_rater_conflict(ordered)

    if conflict and adjudicators and adjudicators[-1].get("disposition") in DECISIVE_DISPOSITIONS:
        conflict = False

    # A fresh explicit adjudication can resolve a persisted judgment's
    # review requirement without deleting that immutable historical row.
    review_required = bool(effective.get("review_required", False))
    if not adjudicators:
        review_required = review_required or any(bool(o.get("review_required", False)) for o in human_raters)
    gold_eligible = disposition in DECISIVE_DISPOSITIONS and bool(evidence) and not conflict and not review_required

    return {
        **retained_reply_content([o for o in ordered if o["evidence_digest"] == evidence_digest]),
        "disposition": disposition,
        "labeler": str(_required(effective, "labeler")),
        "evidence": evidence,
        "evidence_digest": evidence_digest,
        "role": str(_required(effective, "role")),
        "conflict": conflict,
        "review_required": review_required,
        "gold_eligible": gold_eligible,
    }
