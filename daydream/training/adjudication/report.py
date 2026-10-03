"""Deterministic coverage, class-balance, and inter-rater reports. Missing required fields raise
ValueError rather than silently shrinking the reported population.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from daydream.training.adjudication.observations import group_observations_by_record
from daydream.training.adjudication.precedence import effective_adjudication
from daydream.training.adjudication.snapshot import evidence_after_as_of
from daydream.training.corpus_projection.tiers import classify_tier
from daydream.training.dispositions import DECISIVE_DISPOSITIONS

__all__ = ["adjudicated_items", "build_report"]

_ADMISSION_GATE_VERSION = 1


def _required(item: Mapping[str, object], field: str) -> object:
    value = item.get(field)
    if value is None:
        raise ValueError(f"adjudication item missing required field {field!r}: {item!r}")
    return value


def _human_dispositions(observations: Sequence[Mapping[str, object]]) -> list[str]:
    # All roles except automatic/model-suggested count as human, including suffixed rater roles.
    return [
        str(_required(obs, "disposition"))
        for obs in observations
        if str(obs.get("role", "automatic")) not in {"automatic", "model-suggested"}
    ]


def adjudicated_items(
    items: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Apply the same observation, precedence, and tier rules to CLI and final coverage reports.

    Stamp temporal eligibility before tier classification. Only decisive human
    judgments matching the fresh evidence digest count toward gold; automatic
    or stale-evidence decisions remain task-only. Unknown observation record_ids
    raise ValueError rather than disappearing from the admission denominator.
    """
    queue_ids = {str(item["record_id"]) for item in items}
    grouped = group_observations_by_record(observations, queue_ids, "report")

    enriched: list[dict[str, Any]] = []
    for item in items:
        enriched_item = dict(item)
        record_obs = grouped.get(str(item["record_id"]), [])
        enriched_item["observations"] = record_obs
        gold_eligible = False
        if record_obs:
            resolved = effective_adjudication(record_obs)
            # A human judgment made against different evidence is never
            # silently reused (the queue's digest-drift rule); gold
            # eligibility therefore holds only when the effective
            # observation's evidence_digest equals the item's fresh digest —
            # mirroring the disposition override two lines below and the
            # canonical merge's digest-match guard (canonical.py).
            fresh_match = resolved["evidence_digest"] == str(item["evidence_digest"])
            if fresh_match:
                gold_eligible = resolved["gold_eligible"]
            if (
                fresh_match
                and resolved["role"] in ("rater", "adjudicator")
                and resolved["disposition"] in DECISIVE_DISPOSITIONS
            ):
                enriched_item["disposition"] = resolved["disposition"]
        # Stamp the temporal axis FIRST so classify_tier sees it, matching the
        # canonical serializer and the corpus projection (tiers.py C5/M9): an
        # evidence-after-as_of record must classify "silver" here exactly as it
        # does on the canonical record — never gold/posterior_eligible.
        enriched_item["evidence_after_as_of"] = evidence_after_as_of(enriched_item, as_of)
        # The gold gate has one implementation (classify_tier); gold-eligibility
        # comes from the human-observation resolution (conflict/review-required
        # decisive judgments stay out of the gold tier). A classifier failure
        # fail-closes naming the record, never a silent skip.
        try:
            tier = classify_tier(enriched_item)
        except Exception as exc:
            raise ValueError(
                f"cannot classify tier for record_id {str(item['record_id'])!r}: {exc}"
            ) from exc
        if tier == "gold" and not gold_eligible:
            tier = "task-only"
        enriched_item["tier"] = tier
        enriched_item["posterior_eligible"] = tier == "gold" and str(
            enriched_item["profile"]
        ) == "pr_review"
        enriched.append(enriched_item)
    return enriched


def build_report(items: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Report coverage over decisive gold, posterior-eligible pr_review records, excluding
    evidence_after_as_of. This shared scope keeps ineligible records from blocking the outcome gate;
    nondecisive records are counted separately.

    Include accepted/rejected balance, unresolved human decisions, counts by stack/profile, and
    sorted after-as_of IDs. Inter-rater items count disagreements; agreeing counts decisive
    adjudicator resolutions. The admission gate reports this outcome-bearing population and requires
    both classes for class_balance_ok.
    """
    adjudicated = 0
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

        after_as_of = bool(item.get("evidence_after_as_of", False))
        if after_as_of:
            evidence_after_as_of.append(record_id)

        tier = str(item.get("tier", ""))
        posterior_eligible = bool(item.get("posterior_eligible", False))
        decisive = disposition in DECISIVE_DISPOSITIONS
        outcome_bearing = (
            decisive
            and tier == "gold"
            and posterior_eligible
            and profile == "pr_review"
            and not after_as_of
        )
        if decisive:
            if outcome_bearing:
                adjudicated += 1
                if not has_human_decision:
                    unresolved += 1
                if disposition == "accepted":
                    accepted += 1
                else:
                    rejected += 1
        else:
            silver_task_only += 1

        # Inter-rater agreement means a decisive adjudicator resolved disagreeing human
        # observations.
        if len(human) >= 2 and len(set(human)) > 1:
            inter_rater_items += 1
            if any(
                str(obs.get("role")) == "adjudicator" and str(obs.get("disposition")) in DECISIVE_DISPOSITIONS
                for obs in observations
            ):
                inter_rater_agreeing += 1

    gate_passes = adjudicated > 0
    return {
        "outcome_coverage": {"adjudicated": adjudicated, "total": adjudicated},
        "silver_task_only_count": silver_task_only,
        "class_balance": {"accepted": accepted, "rejected": rejected},
        "unresolved": unresolved,
        "inter_rater": {"items": inter_rater_items, "agreeing": inter_rater_agreeing},
        "strata": {k: strata[k] for k in sorted(strata)},
        "evidence_after_as_of": sorted(evidence_after_as_of),
        "admission_gate": {
            "outcome_bearing_total": adjudicated,
            "total": adjudicated,
            "passes_80pct": gate_passes,
            "class_balance_ok": accepted > 0 and rejected > 0,
            "gate_version": _ADMISSION_GATE_VERSION,
        },
    }
