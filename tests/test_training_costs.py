"""Tests for per-run cost accounting (M19).

Costs are reporting-only — never a budget gate (C3). ``record_stage_costs``
accumulates per-stage USD and token counts from backend run events;
``summarize_costs`` derives $/review and $/finding-that-mattered, returning
``None`` for any metric whose denominator is zero (a missing measurement is
never reported as 0.0 or inf).
"""

import json
import pathlib
from dataclasses import dataclass
from typing import Any, cast

from daydream.backends import CostEvent, MetricsEvent
from daydream.training.costs import record_stage_costs, summarize_costs


@dataclass
class _StageEvents:
    stage: str
    events: list[object]
def test_per_run_accounting_written(tmp_path: pathlib.Path) -> None:
    stage_run_events = [
        _StageEvents(
            stage="stage0",
            events=[
                CostEvent(cost_usd=0.12, input_tokens=100, output_tokens=50),
                CostEvent(cost_usd=0.08, input_tokens=80, output_tokens=40),
            ],
        ),
        _StageEvents(
            stage="stage3",
            events=[
                MetricsEvent(
                    message_id="m1",
                    prompt_tokens=200,
                    completion_tokens=100,
                    cached_tokens=None,
                    cost_usd=0.05,
                ),
            ],
        ),
    ]
    p = record_stage_costs(cast(Any, stage_run_events), out=tmp_path / "costs.json")
    data = json.loads(p.read_text())
    assert data["stages"]["stage0"]["usd"] >= 0.0
    assert data["stages"]["stage3"]["tokens"] > 0
    assert data["stages"]["stage0"]["usd"] == 0.2
    assert data["stages"]["stage0"]["tokens"] == 270
def test_reports_dollar_per_review_and_per_finding(tmp_path: object) -> None:
    costs_with_labels = [
        {
            "stages": {
                "stage0": {"usd": 0.5, "tokens": 1000},
                "stage3": {"usd": 1.5, "tokens": 2000},
            },
            "reviews": 2,
            "findings_that_mattered": 4,
        },
        {
            "stages": {"stage0": {"usd": 1.0, "tokens": 500}},
            "reviews": 1,
            "findings_that_mattered": 1,
        },
    ]
    s = summarize_costs(costs_with_labels)
    assert s["usd_per_review"] is not None and s["usd_per_review"] > 0
    # denominator: findings a maintainer accepted
    assert s["usd_per_finding_that_mattered"] is not None
    assert s["usd_per_finding_that_mattered"] > 0
    assert s["usd_per_review"] == (0.5 + 1.5 + 1.0) / 3
    assert s["usd_per_finding_that_mattered"] == 3.0 / 5
def test_zero_denominator_is_none_not_zero(tmp_path: object) -> None:
    costs_with_labels = [
        {
            "stages": {"stage0": {"usd": 1.0, "tokens": 10}},
            "reviews": 0,
            "findings_that_mattered": 0,
        },
    ]
    s = summarize_costs(costs_with_labels)
    assert s["usd_per_review"] is None
    assert s["usd_per_finding_that_mattered"] is None
