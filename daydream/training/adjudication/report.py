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

    for item in items:
        str(_required(item, "record_id"))  # fail-closed validation; id not needed for aggregation
        disposition = str(_required(item, "disposition"))
        stack = str(item.get("stack", ""))
        profile = str(item.get("profile", ""))
        strata[(stack, profile)] = strata.get((stack, profile), 0) + 1

        raw_obs = item.get("observations")
        observations: Sequence[Mapping[str, object]] = list(raw_obs) if isinstance(raw_obs, Sequence) else []
        human = _human_dispositions(observations)
        has_human_decision = bool(human)

        if disposition in {"accepted", "rejected"}:
            decisive_total += 1
            if has_human_decision:
                adjudicated += 1
            else:
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

    return {
        "outcome_coverage": {"adjudicated": adjudicated, "total": decisive_total},
        "silver_task_only_count": silver_task_only,
        "class_balance": {"accepted": accepted, "rejected": rejected},
        "unresolved": unresolved,
        "inter_rater": {"items": inter_rater_items, "agreeing": inter_rater_agreeing},
        "strata": {k: strata[k] for k in sorted(strata)},
    }
