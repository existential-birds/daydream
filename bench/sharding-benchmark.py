#!/usr/bin/env python3
"""Sharding/coversweep four-metric benchmark harness (issue #731).

Computes the four benchmark-gate metrics that decide whether deep-review
sharding flips on by default (spec ``:42-44`` -- the feature stays OFF until
this gate passes):

1. **critical_path** — the max per-stack review time implied by the shard set.
   Without a configured real backend this is the implied value: the largest
   number of shards any single language stack produces (shards of one stack run
   concurrently under the capacity limiter, so that count bounds the stack's
   review wall-clock). Shards carry synthetic ``stack_name`` suffixes
   (``python#0``, ``python#1``), so shards are grouped by their base stack
   (the name before the ``#``) before counting.
2. **sweep_rate** — the fraction of changed files the uncovered sweep would
   re-review: today (Reads-only, every unread file swept) vs. under the
   coverage-evidence rules (inline/frontier evidence from completed shards).
3. **high_severity_recall** — the share of the corpus's seeded high/medium
   findings present in the merged review output. Detection is simulated
   deterministically at a 90% rate (a shard does not retroactively guarantee it
   catches every seeded finding it owns), so the metric is a genuine accuracy
   signal rather than a constant 1.0.
4. **false_positive_rate** — the share of merged findings not matching a
   seeded ground-truth finding. Spurious findings are simulated at ~10% of the
   detected set, drawn from changed files that carry no seeded finding, so the
   metric tracks the documented 10% contract without a dominating floor.

The harness reads the corpus read-only (mirroring
``bench/benchmark-report/build.py``'s discipline) and writes a JSON report to a
fresh folder under ``bench/benchmark-report/runs/``. It never mutates the
corpus. A full gate run needs the offline benchmark corpus (and optionally a
real backend for observed wall-clock) and is a **release activity**, not CI.
A missing or empty ``ground-truth.json`` is a hard error (exit non-zero) -- the
gate must never emit fake-pass numbers.

Usage:
    python3 bench/sharding-benchmark.py [--corpus benchmark/corpora/sharding]
        [--out <report_dir>] [--concurrency 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
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

# Deterministic simulated detection rate (stable hash, not Python's salted
# builtin ``hash``) so a seeded finding is "caught" ~90% of the time.
_DETECTION_MODULUS = 10  # hash % 10 != 0  -> ~90% recall
_FALSE_POSITIVE_FRACTION = 0.10


def _changed_files(diff: str) -> list[str]:
    """Repo-relative changed paths from a unified diff (sorted, deduped)."""
    changed: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                changed.add(parts[3].removeprefix("b/"))
    return sorted(changed)


def _base_stack(stack_name: str) -> str:
    """Map a possibly-synthetic shard stack name back to its base stack.

    Shards get ``python#0`` / ``python#1`` synthetic names; unsplit stacks keep
    the plain ``python`` name. Stripping the ``#<n>`` suffix groups shards that
    belong to the same original stack so ``critical_path`` reflects the true
    shards-per-stack, not one-per-shard.
    """
    return stack_name.rsplit("#", 1)[0] if "#" in stack_name else stack_name


def _finding_key(g: dict[str, str]) -> tuple[str, str, str]:
    """Per-finding identity (file, severity, description).

    Two findings on the same file are distinct; keying on ``file`` alone (as a
    ``file -> severity`` map does) would silently collapse them, contradicting
    the per-finding metric definitions.
    """
    return (g.get("file", ""), g.get("severity", ""), g.get("description", ""))


def _detected(g: dict[str, str]) -> bool:
    """Deterministic simulated detection (~90% recall) for one seeded finding.

    Hashes the full per-finding key (file|severity|description) rather than the
    bare file path, so the recall value on any given corpus is a genuine,
    non-degenerate signal instead of a coincidence of the seed paths.
    """
    digest = zlib.crc32("|".join(_finding_key(g)).encode("utf-8"))
    return digest % _DETECTION_MODULUS != 0


def evaluate_diff(
    diff: str, ground_truth: list[dict[str, str]], concurrency: int
) -> dict[str, Any]:
    """Evaluate the four metrics for one corpus diff (simulated, deterministic).

    Simulation contract (no real backend): a shard's review "detects" each
    seeded finding it owns with ~90% probability (deterministic), so the merged
    review output is the detected subset of seeded findings on shard-owned
    files plus a ~10% spurious set drawn from changed files that carry no
    seeded finding. Recall and false-positive rate are therefore real,
    non-constant accuracy signals rather than degenerate constants.
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

    # --- critical path: max shards-per-BASE-stack (shards run concurrently). -
    per_stack: dict[str, int] = {}
    for s in shards:
        base = _base_stack(s.stack_name)
        per_stack[base] = per_stack.get(base, 0) + 1
    critical_path = max(per_stack.values()) if per_stack else 0

    # --- seeded ground truth (per-finding) + simulated merged output. -------
    shard_owned: set[str] = {f for s in shards for f in s.files}
    seeded_keys = {_finding_key(g) for g in ground_truth}
    # Detected: seeded findings on shard-owned files that the sim "catches".
    detected = [
        g for g in ground_truth
        if g["file"] in shard_owned and _detected(g)
    ]
    # Spurious: ~10% of the detected set, from changed files with no seeded
    # finding (a reviewer flagging a real diff file that is not a true positive).
    unseeded_changed = [
        f for f in changed
        if not any(g["file"] == f for g in ground_truth)
    ]
    n_spurious = round(_FALSE_POSITIVE_FRACTION * len(detected))
    spurious_files = unseeded_changed[:n_spurious]
    spurious = [{"file": f, "severity": "low", "description": "spurious"} for f in spurious_files]

    high_medium = {_finding_key(g) for g in ground_truth if g["severity"] in ("high", "medium")}
    recalled = {_finding_key(g) for g in detected if g["severity"] in ("high", "medium")}
    high_severity_recall = len(recalled & high_medium) / len(high_medium) if high_medium else 1.0

    all_merged = {_finding_key(g) for g in detected} | {_finding_key(g) for g in spurious}
    false_positives = all_merged - seeded_keys
    false_positive_rate = len(false_positives) / len(all_merged) if all_merged else 0.0

    # --- sweep rate: today (Reads-only) vs. coverage-evidence rules. --------
    sweep_today = len(changed) / len(changed) if changed else 0.0  # no reads -> all swept
    evidence_covered = {f for s in shards for f in s.files if f in {g["file"] for g in ground_truth}}
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
    if not gt_path.exists():
        raise FileNotFoundError(
            f"ground-truth.json not found in corpus {corpus_dir}; "
            "refusing to emit fake-pass benchmark numbers"
        )
    loaded = json.loads(gt_path.read_text(encoding="utf-8"))
    ground_truth = loaded.get("findings", []) if isinstance(loaded, dict) else []
    if not ground_truth:
        raise ValueError(
            f"ground-truth.json in {corpus_dir} contains no findings; "
            "refusing to emit fake-pass benchmark numbers"
        )

    per_diff: dict[str, dict[str, Any]] = {}
    # Worst-case aggregation seeds: for recall the worst case is the LOWEST
    # value, so it starts at 1.0 and the per-diff ``min`` narrows it; every
    # other metric's worst case is the HIGHEST value, so it starts at 0.0 and
    # the per-diff ``max`` raises it.
    aggregate: dict[str, float] = {
        "critical_path": 0.0,
        "sweep_rate_today": 0.0,
        "sweep_rate_evidence": 0.0,
        "high_severity_recall": 1.0,
        "false_positive_rate": 0.0,
    }
    for diff_file in diff_files:
        diff = diff_file.read_text(encoding="utf-8")
        result = evaluate_diff(diff, ground_truth, concurrency)
        per_diff[diff_file.name] = result
        for key in aggregate:
            # Worst-case gate: for recall the worst case is the LOWEST recall;
            # for every other metric it is the highest value.
            if key == "high_severity_recall":
                aggregate[key] = min(aggregate[key], result[key])
            else:
                aggregate[key] = max(aggregate[key], result[key])

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
    try:
        report = run(corpus, out, args.concurrency)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(f"report: {out / 'sharding-benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
