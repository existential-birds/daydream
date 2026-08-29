#!/usr/bin/env -S uv run --script
"""Self-contained corpus micro-metric aggregation for the Harbor verifier image.

Stdlib only, never imports daydream (or the bundled verifier_core): the
aggregation body is inlined at build time so a compiled task's ``metric.py``
needs nothing but the stdlib. Reads one JSONL line per task — a reward dict or
``null`` (unscored infrastructure failure) — and writes the pooled micro metrics to the ``-o`` path.
Invocation matches Harbor 0.22's ``uv run metric.py -i <rewards.jsonl> -o
<metric.json>``; a missing input raises ``FileNotFoundError`` (uncaught ->
nonzero exit, no output written — fail-closed, never partial).
"""
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


class VerifierError(Exception):
    """Raised on any invalid input (mirrors verifier_core's error type)."""


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerifierError(f"expected integer, got {value!r}")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerifierError(f"expected number, got {value!r}")
    return float(value)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 1.0
    return 2 * precision * recall / (precision + recall)


def _axis_aggregates(
    loc_tiers: dict[str, int],
    loc_credit_sum: float,
    location_pairs: int,
    sev_exact: int,
    sev_within_1: int,
    sev_distance_sum: float,
    sev_credit_sum: float,
    severity_pairs: int,
) -> dict[str, float | int]:
    """Pool the reported location/severity axes over scored pairs.

    Rates are counts over the axis pair denominator (0.0 with zero pairs).
    ``severity_mean_distance``/``severity_credit`` pool as the mean of each
    task's reported per-pair mean over the tasks where the axis is present
    (row-level summaries do not carry per-pair sums, so per-task means are the
    finest granularity available; equal-pair tasks pool to the exact per-pair
    mean).
    """
    n_loc = location_pairs
    n_sev = severity_pairs
    return {
        "location_exact": loc_tiers["exact"],
        "location_near": loc_tiers["near"],
        "location_file": loc_tiers["file"],
        "location_miss": loc_tiers["miss"],
        "location_exact_rate": loc_tiers["exact"] / n_loc if n_loc else 0.0,
        "location_near_rate": loc_tiers["near"] / n_loc if n_loc else 0.0,
        "location_file_rate": loc_tiers["file"] / n_loc if n_loc else 0.0,
        "location_miss_rate": loc_tiers["miss"] / n_loc if n_loc else 0.0,
        "location_credit": loc_credit_sum / n_loc if n_loc else 0.0,
        "location_pairs_scored": n_loc,
        "total_location_exact": loc_tiers["exact"],
        "total_location_near": loc_tiers["near"],
        "total_location_file": loc_tiers["file"],
        "total_location_miss": loc_tiers["miss"],
        "severity_exact": sev_exact,
        "severity_within_1": sev_within_1,
        "severity_exact_rate": sev_exact / n_sev if n_sev else 0.0,
        "severity_within_1_rate": sev_within_1 / n_sev if n_sev else 0.0,
        "severity_mean_distance": sev_distance_sum / n_sev if n_sev else 0.0,
        "severity_credit": sev_credit_sum / n_sev if n_sev else 0.0,
        "severity_pairs_scored": n_sev,
        "total_severity_exact": sev_exact,
        "total_severity_within_1": sev_within_1,
    }


# __AGGREGATION_BODY_BEGIN__
def aggregate_metrics(rows: list[dict[str, object] | None]) -> dict[str, float | int]:
    """Placeholder — replaced at build time by ``verifier_core.aggregate_metrics``.

    ``build.render_metric()`` splices ``inspect.getsource(verifier_core.
    aggregate_metrics)`` over the region between the two marker comments, so the
    compiled metric and the in-repo corpus pool can never drift.
    """
    raise NotImplementedError(
        "aggregation body is rendered at build time from verifier_core.aggregate_metrics"
    )
# __AGGREGATION_BODY_END__


def aggregate_rewards_file(path: str) -> dict[str, float | int]:
    """Read a JSONL rewards file (reward dict or null per line) and aggregate.

    A malformed/non-dict line is an unscored infrastructure ``None`` row, never
    an aborting exception. An empty file aggregates to ``task_count == 0`` /
    score 1.0.
    """
    rows: list[dict[str, object] | None] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            rows.append(None)
            continue
        rows.append(obj if isinstance(obj, dict) else None)
    return aggregate_metrics(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Harbor verifier rewards JSONL into pooled micro metrics.")
    parser.add_argument("-i", "--input", required=True, help="JSONL rewards input (one reward dict or null per task)")
    parser.add_argument("-o", "--output", required=True, help="JSON metric output path (written atomically)")
    args = parser.parse_args(argv)
    result = aggregate_rewards_file(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
