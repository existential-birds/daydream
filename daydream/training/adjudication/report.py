"""Coverage / class-balance / inter-rater report over adjudication queue items (AC 5).

Aggregate twin of ``classify_tier``'s disjoint-set discipline: silver/task-only
records never enter the outcome-bearing 80% denominator — they are counted
separately. Every grouping key is sorted so the report is deterministic
regardless of input order, and a malformed item raises ``ValueError`` naming
the offending field rather than being silently skipped (a silent skip could
overcount the 80% gate).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["build_report"]

_DECISIVE_DISPOSITIONS = frozenset({"accepted", "rejected"})
_ADMISSION_GATE_VERSION = 1


def _required(item: Mapping[str, object], field: str) -> object:
    value = item.get(field)
    if value is None:
        raise ValueError(f"adjudication item missing required field {field!r}: {item!r}")
    return value


def _human_dispositions(observations: Sequence[Mapping[str, object]]) -> list[str]:
    # Anything that is not an automatic/model signal is a human rater; roles may
    # be suffixed ("rater2") to distinguish individual raters.
    return [
        str(_required(obs, "disposition"))
        for obs in observations
        if str(obs.get("role", "automatic")) not in {"automatic", "model-suggested"}
    ]


def build_report(items: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Build the coverage report.

    - ``outcome_coverage``: ``{"adjudicated": n, "total": m}`` over records whose
      disposition is decisive (``accepted``/``rejected``); silver/task-only
      records are excluded from both numerator and denominator.
    - ``silver_task_only_count``: non-decisive records, reported separately.
    - ``class_balance``: accepted vs rejected counts.
    - ``unresolved``: records with no effective human decision.
    - ``inter_rater``: agreement over records with >=2 human rater observations.
    - ``strata``: counts keyed by ``(stack, profile)``.
    - ``evidence_after_as_of``: sorted record ids flagged by the canonical
      harvest as having evidence observed after the ``as_of`` pin — recorded
      and flagged, never gold-eligible (C5/M9).
    - ``admission_gate``: the #94 admission gate over **outcome-bearing**
      records only. ``outcome_coverage.adjudicated`` counts gold-tier,
      ``posterior_eligible``, ``pr_review`` records and excludes
      ``evidence_after_as_of`` records; task-only items are reported but never
      feed the 80% denominator.
    """
    adjudicated = 0
    decisive_total = 0
    silver_task_only = 0
    accepted = 0
    rejected = 0
    unresolved = 0
    inter_rater_items = 0
    inter_rater_agreeing = 0
    strata: dict[tuple[str, str], int] = {}
    evidence_after_as_of: list[str] = []

    for item in items:
        record_id = str(_required(item, "record_id"))
        disposition = str(_required(item, "disposition"))
        stack = str(item.get("stack", ""))
        profile = str(item.get("profile", ""))
        strata[(stack, profile)] = strata.get((stack, profile), 0) + 1

        raw_obs = item.get("observations")
        observations: Sequence[Mapping[str, object]] = list(raw_obs) if isinstance(raw_obs, Sequence) else []
        human = _human_dispositions(observations)
        has_human_decision = bool(human)

        # Recorded-and-flagged as_of edge policy: the record keeps its evidence
        # but is never gold-eligible, so it is excluded from the outcome-bearing
        # numerator regardless of disposition.
        if bool(item.get("evidence_after_as_of", False)):
            evidence_after_as_of.append(record_id)

        tier = str(item.get("tier", ""))
        posterior_eligible = bool(item.get("posterior_eligible", False))
        decisive = disposition in _DECISIVE_DISPOSITIONS
        # Outcome-bearing records (C5/M9): gold-tier, posterior-eligible
        # pr_review records only — task-only and silver never count, and
        # neither do evidence-after-as_of records. Only these can ever be
        # adjudicated, so the 80% denominator, the numerator, unresolved and
        # class balance share one scope and a record that can never count
        # cannot permanently block the gate.
        outcome_bearing = (
            decisive
            and tier == "gold"
            and posterior_eligible
            and profile == "pr_review"
            and not bool(item.get("evidence_after_as_of", False))
        )
        if decisive:
            if outcome_bearing:
                decisive_total += 1
                adjudicated += 1
                if not has_human_decision:
                    unresolved += 1
                if disposition == "accepted":
                    accepted += 1
                else:
                    rejected += 1
        else:
            silver_task_only += 1

        # Inter-rater accounting covers disputed multi-rater records: ``items``
        # counts multi-rater records whose human dispositions disagree;
        # ``agreeing`` counts those resolved by a decisive explicit adjudicator.
        if len(human) >= 2 and len(set(human)) > 1:
            inter_rater_items += 1
            if any(
                str(obs.get("role")) == "adjudicator" and str(obs.get("disposition")) in {"accepted", "rejected"}
                for obs in observations
            ):
                inter_rater_agreeing += 1

    gate_passes = decisive_total > 0 and adjudicated * 5 >= decisive_total * 4
    return {
        "outcome_coverage": {"adjudicated": adjudicated, "total": decisive_total},
        "silver_task_only_count": silver_task_only,
        "class_balance": {"accepted": accepted, "rejected": rejected},
        "unresolved": unresolved,
        "inter_rater": {"items": inter_rater_items, "agreeing": inter_rater_agreeing},
        "strata": {k: strata[k] for k in sorted(strata)},
        "evidence_after_as_of": sorted(evidence_after_as_of),
        "admission_gate": {
            "outcome_bearing_total": adjudicated,
            "total": decisive_total,
            "passes_80pct": gate_passes,
            "class_balance_ok": accepted > 0 and rejected > 0,
            "gate_version": _ADMISSION_GATE_VERSION,
        },
    }
