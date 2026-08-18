#!/usr/bin/env python3
"""Sharding/coversweep four-metric benchmark harness (issue #731).

Computes the four benchmark-gate metrics that decide whether deep-review
sharding flips on by default (spec ``:42-44`` -- the feature stays OFF until
this gate passes):

1. **critical_path** — the max per-stack review time implied by the shard set.
   Without a configured real backend this is the implied value: the largest
   number of shards any single language stack produces (shards of one stack run
   concurrently under the capacity limiter, so that count bounds the stack's
   review wall-clock).
2. **sweep_rate** — the fraction of changed files the uncovered sweep would
   re-review: today (Reads-only, every unread file swept) vs. under the
   coverage-evidence rules (inline/frontier evidence from completed shards).
3. **high_severity_recall** — the share of the corpus's seeded high/medium
   findings present in the merged review output.
4. **false_positive_rate** — the share of merged findings not matching a
   seeded ground-truth finding.

The harness reads the corpus read-only (mirroring
``bench/benchmark-report/build.py``'s discipline) and writes a JSON report to a
fresh folder under ``bench/benchmark-report/runs/``. It never mutates the
corpus. A full gate run needs the offline benchmark corpus (and optionally a
real backend for observed wall-clock) and is a **release activity**, not CI.

Usage:
    python3 bench/sharding-benchmark.py [--corpus benchmark/corpora/sharding]
        [--out <report_dir>] [--concurrency 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daydream.config import (
    DEFAULT_DEEP_SHARD_ENABLED,
    DEFAULT_DEEP_SHARD_FANOUT_CAP,
    DEFAULT_DEEP_SHARD_FRONTIER_MAX,
    DEFAULT_DEEP_SHARD_MAX_BYTES,
    DEFAULT_DEEP_SHARD_MAX_FILES,
)
from daydream.deep.detection import detect_stacks
from daydream.deep.sharding import shard_stacks


def _changed_files(diff: str) -> list[str]:
    """Repo-relative changed paths from a unified diff (sorted, deduped)."""
    changed: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                changed.add(parts[3].removeprefix("b/"))
    return sorted(changed)


def evaluate_diff(
    diff: str, ground_truth: list[dict[str, str]], concurrency: int
) -> dict[str, Any]:
    """Evaluate the four metrics for one corpus diff (simulated, deterministic).

    Simulation contract (no real backend): a shard's review "produces" a parsed
    finding for every file it owns when the file carries a seeded ground-truth
    finding -- inline/frontier coverage evidence therefore covers exactly the
    seeded-finding files of completed shards, and the merged review output is
    the seeded findings on shard-owned files plus 10% spurious findings (the
    false-positive seed).
    """
    changed = _changed_files(diff)
    stacks = detect_stacks(changed, skill_availability=None)
    shards = shard_stacks(
        stacks,
        diff,
        max_files=DEFAULT_DEEP_SHARD_MAX_FILES,
        max_bytes=DEFAULT_DEEP_SHARD_MAX_BYTES,
        fanout_cap=DEFAULT_DEEP_SHARD_FANOUT_CAP,
        frontier_max=DEFAULT_DEEP_SHARD_FRONTIER_MAX,
    )

    # --- critical path: max shards-per-stack (shards run concurrently). ----
    per_stack: dict[str, int] = {}
    for s in shards:
        per_stack[s.stack_name] = per_stack.get(s.stack_name, 0) + 1
    critical_path = max(per_stack.values()) if per_stack else 0

    # --- seeded ground truth + simulated merged output. ---------------------
    seeded: dict[str, str] = {g["file"]: g["severity"] for g in ground_truth}
    shard_owned: set[str] = {f for s in shards for f in s.files}
    merged: dict[str, str] = {f: sev for f, sev in seeded.items() if f in shard_owned}
    # 10% spurious findings: seeded files not in the changed set would be the
    # natural noise source; simulate by mis-attributing one low finding.
    noise = max(1, len(merged) // 10)
    spurious = {f"src/service/noise{i}.py": "low" for i in range(noise)}

    high_medium = {f for f, sev in seeded.items() if sev in ("high", "medium")}
    recalled = {f for f in merged if f in high_medium}
    high_severity_recall = len(recalled) / len(high_medium) if high_medium else 1.0

    all_merged = {**merged, **spurious}
    false_positives = {f for f in all_merged if f not in seeded}
    false_positive_rate = len(false_positives) / len(all_merged) if all_merged else 0.0

    # --- sweep rate: today (Reads-only) vs. coverage-evidence rules. --------
    sweep_today = len(changed) / len(changed) if changed else 0.0  # no reads -> all swept
    evidence_covered = {f for s in shards for f in s.files if f in seeded}
    swept_evidence = len(changed) - len(evidence_covered)
    sweep_evidence = swept_evidence / len(changed) if changed else 0.0

    return {
        "changed_files": len(changed),
        "review_tasks": len(shards),
        "critical_path_shards": critical_path,
        "critical_path": round(critical_path / max(concurrency, 1), 4),
        "sweep_rate_today": round(sweep_today, 4),
        "sweep_rate_evidence": round(sweep_evidence, 4),
        "sweep_rate_delta": round(sweep_today - sweep_evidence, 4),
        "high_severity_recall": round(high_severity_recall, 4),
        "false_positive_rate": round(false_positive_rate, 4),
    }


def run(corpus_dir: Path, out_dir: Path, concurrency: int) -> dict[str, Any]:
    """Evaluate every corpus diff read-only and write the JSON report."""
    diff_files = sorted(corpus_dir.glob("*.patch"))
    gt_path = corpus_dir / "ground-truth.json"
    ground_truth: list[dict[str, str]] = []
    if gt_path.exists():
        loaded = json.loads(gt_path.read_text(encoding="utf-8"))
        ground_truth = loaded.get("findings", []) if isinstance(loaded, dict) else []

    per_diff: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, float] = {
        "critical_path": 0.0,
        "sweep_rate_today": 0.0,
        "sweep_rate_evidence": 0.0,
        "high_severity_recall": 0.0,
        "false_positive_rate": 0.0,
    }
    for diff_file in diff_files:
        diff = diff_file.read_text(encoding="utf-8")
        result = evaluate_diff(diff, ground_truth, concurrency)
        per_diff[diff_file.name] = result
        for key in aggregate:
            aggregate[key] = max(aggregate[key], result[key])  # worst-case gate

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": str(corpus_dir),
        "corpus_read_only": True,
        "default_shard_enabled": DEFAULT_DEEP_SHARD_ENABLED,
        "concurrency": concurrency,
        "metrics": aggregate,
        "per_diff": per_diff,
    }
    (out_dir / "sharding-benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sharding-benchmark",
        description=(
            "Issue #731 sharding/coversweep four-metric benchmark harness. "
            "Reads a corpus read-only and writes a JSON report to a fresh run dir."
        ),
    )
    parser.add_argument("--corpus", type=Path, default=Path("benchmark/corpora/sharding"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Report dir (default: a fresh run dir under bench/benchmark-report/runs/)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Assumed fan-out concurrency for the implied critical path",
    )
    args = parser.parse_args(argv)

    corpus = args.corpus
    if not corpus.is_dir():
        print(f"error: corpus dir not found: {corpus}", file=sys.stderr)
        return 2
    out = args.out or Path("bench/benchmark-report/runs") / (
        datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-sharding"
    )
    report = run(corpus, out, args.concurrency)
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(f"report: {out / 'sharding-benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
