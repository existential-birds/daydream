"""Per-run cost accounting (M19).

Reporting only — never a budget gate (C3). ``record_stage_costs`` accumulates
per-stage USD and token counts from the backend ``AgentEvent`` stream of a
stage run and writes a per-run JSON record. ``summarize_costs`` aggregates
those records into ``$/review`` and ``$/finding-that-mattered`` metrics.

The denominator for ``usd_per_finding_that_mattered`` is the count of findings
that survived to an accepted/contested label — findings a maintainer actually
acted on. A zero denominator means the metric was not measured, so it is
reported as ``None`` — never ``0.0`` or ``inf``.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.backends import AgentEvent, CostEvent, MetricsEvent


@dataclass(frozen=True)
class StageRunEvents:
    """One stage's slice of a run's event stream.

    Attributes:
        stage: Stage name (``stage0`` … ``stage3``).
        events: The ``AgentEvent`` sequence yielded for that stage.
    """

    stage: str
    events: Sequence[AgentEvent]


def _accumulate(events: Iterable[AgentEvent]) -> dict[str, float]:
    """Sum USD and tokens over one stage's events.

    Tokens and USD come from ``CostEvent`` when present; per-turn
    ``MetricsEvent`` usage and USD are added only when no ``CostEvent``
    reported them for the stage, so the same usage/cost is never counted
    twice.

    Backends emit per-turn ``MetricsEvent`` first and the ``CostEvent``
    carrying the same aggregated usage last (Claude per call, Codex/Pi per
    turn), so a sequence-ordered single pass would count that usage twice.
    Pre-scan which fields any ``CostEvent`` in the stage reported, then
    treat those fields as CostEvent-owned for the whole stage; ``MetricsEvent``
    supplies only fields no ``CostEvent`` reported.
    """
    usd = 0.0
    tokens = 0
    cost_usd_seen = False
    cost_tokens_seen = False
    events = tuple(events)
    for event in events:
        if isinstance(event, CostEvent):
            if event.cost_usd is not None:
                cost_usd_seen = True
            if event.input_tokens is not None or event.output_tokens is not None:
                cost_tokens_seen = True
    for event in events:
        if isinstance(event, CostEvent):
            if event.cost_usd is not None:
                usd += event.cost_usd
            if event.input_tokens is not None or event.output_tokens is not None:
                tokens += (event.input_tokens or 0) + (event.output_tokens or 0)
        elif isinstance(event, MetricsEvent):
            if event.cost_usd is not None and not cost_usd_seen:
                usd += event.cost_usd
            if not cost_tokens_seen:
                tokens += event.prompt_tokens + event.completion_tokens
    return {"usd": usd, "tokens": tokens}


def record_stage_costs(
    stage_run_events: Iterable[StageRunEvents],
    *,
    out: Path,
) -> Path:
    """Accumulate per-stage costs from one run's events and write ``out``.

    Mirrors the summary-dict shape of :func:`daydream.training.corpus._summary`
    with monetary/token accounting under per-stage keys. Reporting only —
    this function never gates a run or raises a threshold.

    Returns the path written.
    """
    stages: dict[str, dict[str, float]] = {}
    for stage_events in stage_run_events:
        stages[stage_events.stage] = _accumulate(stage_events.events)
    record: dict[str, Any] = {
        "stages": stages,
        "total_usd": sum(s["usd"] for s in stages.values()),
        "total_tokens": sum(s["tokens"] for s in stages.values()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    return out


def summarize_costs(costs: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Aggregate per-run cost records into per-review and per-finding metrics.

    Each record is one run's written cost record, optionally carrying
    ``reviews`` (number of reviews in the run) and
    ``findings_that_mattered`` (findings that survived to an
    accepted/contested label). Missing or zero denominators yield ``None``
    for the corresponding metric — a missing measurement, not a zero.
    """
    total_usd = 0.0
    reviews = 0
    findings = 0
    for record in costs:
        stages = record.get("stages", {})
        total_usd += sum(float(s.get("usd", 0.0)) for s in stages.values())
        reviews += int(record.get("reviews") or 0)
        findings += int(record.get("findings_that_mattered") or 0)
    return {
        "usd_per_review": total_usd / reviews if reviews > 0 else None,
        "usd_per_finding_that_mattered": (
            total_usd / findings if findings > 0 else None
        ),
    }
