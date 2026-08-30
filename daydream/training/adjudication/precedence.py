"""Three-tier adjudication precedence: explicit adjudicator > latest human rater > automatic.

Finding-level twin of ``classify_tier``'s fail-closed gate: a finding is gold-eligible only
when the effective disposition is decisive, evidence is non-empty, and no unresolved rater
conflict stands. Conflicting human raters keep the finding non-gold until an explicit
adjudicator resolution with a decisive disposition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DECISIVE_DISPOSITIONS = frozenset({"accepted", "rejected"})

_HUMAN_ROLES = frozenset({"rater", "adjudicator"})

# Secondary sort key so identical timestamps still resolve deterministically
# regardless of input order.
_SORT_TIEBREAK_KEYS = ("observed_at", "labeler")


def _is_human(obs: Mapping[str, Any]) -> bool:
    return obs.get("role") in _HUMAN_ROLES


def _sorted_by_recency(observations: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        observations,
        key=lambda o: tuple(str(o.get(k, "")) for k in _SORT_TIEBREAK_KEYS),
    )


def _required(obs: Mapping[str, Any], field: str) -> Any:
    value = obs.get(field)
    if value is None:
        msg = f"observation missing required field {field!r} for record_id {obs.get('record_id')!r}"
        raise ValueError(msg)
    return value


def has_rater_conflict(observations: Sequence[Mapping[str, Any]]) -> bool:
    """True iff >=2 human rater observations for the same record_id + evidence_digest differ in disposition.

    An adjudicator observation clears the conflict only when its own disposition is
    decisive; a non-decisive adjudicator entry leaves the conflict standing.
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


def effective_adjudication(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Resolve one record_id's observation list to its effective adjudication.

    Returns ``{"disposition", "labeler", "evidence", "evidence_digest", "role",
    "conflict", "gold_eligible"}``. Precedence: any ``role="adjudicator"`` observation
    (latest if several), else the latest human rater by ``observed_at``, else the
    automatic entry. An empty observation list is a caller bug and raises
    ``ValueError`` naming ``record_id``.
    """
    if not observations:
        msg = "effective_adjudication called with empty observation list for record_id"
        raise ValueError(msg)

    record_id = _required(observations[0], "record_id")
    ordered = _sorted_by_recency(observations)

    adjudicators = [o for o in ordered if o.get("role") == "adjudicator"]
    human_raters = [o for o in ordered if o.get("role") == "rater"]
    automatic = [o for o in ordered if not _is_human(o)]

    if adjudicators:
        effective = adjudicators[-1]
    elif human_raters:
        effective = human_raters[-1]
    elif automatic:
        effective = automatic[-1]
    else:  # pragma: no cover - role sets above are exhaustive over _HUMAN_ROLES
        msg = f"no resolvable observation for record_id {record_id!r}"
        raise ValueError(msg)

    disposition = str(_required(effective, "disposition"))
    evidence_digest = str(_required(effective, "evidence_digest"))
    evidence = _required(effective, "evidence")
    conflict = has_rater_conflict(observations)

    # Adjudicator presence clears the conflict only when its disposition is decisive.
    if conflict and adjudicators and adjudicators[-1].get("disposition") in DECISIVE_DISPOSITIONS:
        conflict = False

    review_required = bool(effective.get("review_required", False))
    gold_eligible = (
        disposition in DECISIVE_DISPOSITIONS
        and bool(evidence)
        and not conflict
        and not review_required
    )

    return {
        "disposition": disposition,
        "labeler": str(_required(effective, "labeler")),
        "evidence": evidence,
        "evidence_digest": evidence_digest,
        "role": str(_required(effective, "role")),
        "conflict": conflict,
        "gold_eligible": gold_eligible,
    }
