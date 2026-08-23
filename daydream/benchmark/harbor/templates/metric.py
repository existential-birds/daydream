#!/usr/bin/env -S uv run --script
"""Self-contained corpus micro-metric aggregation for the Harbor verifier image.

Stdlib only, never imports daydream (or the bundled verifier_core): the
aggregation body is inlined at build time so a compiled task's ``metric.py``
needs nothing but the stdlib. Reads one JSONL line per task — a reward dict or
``null`` (unscored infrastructure failure) — and writes the pooled micro metrics to the ``-o`` path.
Invocation matches Harbor 0.21's ``uv run metric.py -i <rewards.jsonl> -o
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
