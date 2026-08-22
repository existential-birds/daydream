#!/usr/bin/env -S uv run --script
"""Self-contained corpus micro-metric aggregation for the Harbor verifier image.

Stdlib only, never imports daydream (or the bundled verifier_core): the
aggregation body is inlined so a compiled task's ``metric.py`` needs nothing but
the stdlib. Reads one JSONL line per task — a reward dict or ``null`` (failed
task) — and emits the pooled micro metrics.
"""
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import json
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


def aggregate_metrics(rows: list[dict[str, object] | None]) -> dict[str, float | int]:
    """Aggregate per-task reward JSONL rows into pooled corpus micro metrics.

    Inline copy of ``verifier_core.aggregate_metrics`` so the compiled metric is
    self-contained. A ``None`` row or a row with ``verifier_error == 1`` is a
    failed task: reward 0 to the mean, zero counts, ``failed_task_count`` += 1.
    Zero denominators evaluate to 1.0 throughout.
    """
    failed = 0
    clean_correct = 0
    clean_total = 0
    rewards: list[float] = []
    total_tp = total_fp = total_fn = 0

    for row in rows:
        if row is None or row.get("verifier_error") == 1:
            failed += 1
            rewards.append(0.0)
            continue
        total_tp += _as_int(row["tp"])
        total_fp += _as_int(row["fp"])
        total_fn += _as_int(row["fn"])
        rewards.append(_as_float(row["reward"]))
        if row.get("clean_task") == 1:
            clean_total += 1
            if _as_int(row["fp"]) == 0:
                clean_correct += 1

    task_count = len(rows)
    mean_task_score = sum(rewards) / task_count if task_count else 1.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    if total_tp == 0 and (total_tp + total_fp > 0 or total_tp + total_fn > 0):
        micro_f1 = 0.0
    else:
        micro_f1 = _f1(micro_precision, micro_recall)
    clean_accuracy = clean_correct / clean_total if clean_total else 1.0

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mean_task_score": mean_task_score,
        "clean_accuracy": clean_accuracy,
        "task_count": task_count,
        "clean_task_count": clean_total,
        "failed_task_count": failed,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def aggregate_rewards_file(path: str) -> dict[str, float | int]:
    """Read a JSONL rewards file (reward dict or null per line) and aggregate.

    A malformed/non-dict line is a failed-task ``None`` row, never an aborting
    exception. An empty file aggregates to ``task_count == 0`` / score 1.0.
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


def main() -> int:
    results = aggregate_rewards_file("/logs/verifier/reward.json")
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
