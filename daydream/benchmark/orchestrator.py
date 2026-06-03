"""Benchmark orchestrator — acquire → review → map → inject (+ optional score).

Drives the full benchmark sweep over the pinned evaluable PRs. For each
selected PR it acquires a checkout, runs a non-interactive daydream review,
maps the canonical findings to benchmark review comments, and injects a
synthetic ``daydream`` review into the corpus.

Path convention: the benchmark corpus is read from and written to
``config.benchmark_repo / "results" / "benchmark_data.json"`` (the layout the
withmartian ``code-review-benchmark`` step modules expect). The corpus is saved
after every PR so an interrupted sweep is resumable.

A single PR's failure is logged and recorded but does not abort the sweep; the
returned exit code is non-zero if any selected PR failed. When ``config.score``
is set the judge environment is verified up front (before any expensive review)
and, after the sweep, the step2/2.5/3 scoring pipeline runs and its per-PR and
aggregate precision/recall are printed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from daydream.agent import console
from daydream.benchmark.acquire import acquire_checkout
from daydream.benchmark.benchmark_data import (
    DAYDREAM_TOOL,
    inject_daydream_review,
    load_benchmark_data,
    save_benchmark_data,
)
from daydream.benchmark.daydream_run import run_daydream_review
from daydream.benchmark.mapping import merged_items_to_review_comments
from daydream.benchmark.prs import load_evaluable_prs
from daydream.benchmark.score import preflight_judge_env, run_scoring
from daydream.ui import print_dim, print_error, print_info, print_success, print_warning

if TYPE_CHECKING:
    from daydream.benchmark.config import BenchConfig
    from daydream.benchmark.prs import EvaluablePR

#: Deterministic timestamp stamped onto every mapped comment so the mapping
#: (and therefore the injected corpus) stays idempotent across reruns.
_CREATED_AT = "2026-06-03T00:00:00Z"


def _benchmark_data_path(config: BenchConfig) -> Path:
    """Resolve the corpus path under the benchmark repo's ``results`` dir."""
    return config.benchmark_repo / "results" / "benchmark_data.json"


def _select_prs(config: BenchConfig) -> list[EvaluablePR]:
    """Filter the pinned PRs by ``config.only`` (substring) and ``config.limit``."""
    prs = list(load_evaluable_prs())
    if config.only:
        needle = config.only
        prs = [pr for pr in prs if needle in pr.source_repo or needle in pr.golden_url]
    if config.limit is not None:
        prs = prs[: config.limit]
    return prs


def _trajectory_path(config: BenchConfig, pr: EvaluablePR) -> Path:
    """Build the per-PR trajectory file path: ``<repo>-<pr_number>.json``."""
    repo = pr.source_repo.replace("/", "_")
    return config.trajectory_dir / f"{repo}-{pr.pr_number}.json"


def _has_daydream_review(entry: dict) -> bool:
    """Return whether a ``tool:"daydream"`` review already exists in *entry*."""
    return any(review.get("tool") == DAYDREAM_TOOL for review in entry.get("reviews", []))


def _process_pr(config: BenchConfig, pr: EvaluablePR, data: dict) -> bool:
    """Acquire, review, map, and inject a single PR into *data* (mutated).

    Returns:
        ``True`` if the corpus was modified, ``False`` if the existing daydream
        review was left in place (idempotent no-op; see
        :func:`inject_daydream_review`).
    """
    checkout = acquire_checkout(
        pr.clone_url,
        pr.pr_number,
        pr.base_sha,
        pr.head_sha,
        cache_dir=config.cache_dir,
    )
    config.trajectory_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_daydream_review(
        checkout,
        base_sha=pr.base_sha,
        trajectory_path=_trajectory_path(config, pr),
    )
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    comments = merged_items_to_review_comments(doc, created_at=_CREATED_AT)
    return inject_daydream_review(data, pr.golden_url, comments, force=config.force)


def run_bench(config: BenchConfig) -> int:
    """Run the benchmark sweep over the selected PRs.

    Args:
        config: Immutable run configuration (selection, force, scoring, paths).

    Returns:
        ``0`` when every selected PR was injected (or skipped) and scoring (if
        requested) succeeded; non-zero if any selected PR failed or scoring
        failed.
    """
    if config.score:
        preflight_judge_env()

    data_path = _benchmark_data_path(config)
    data = load_benchmark_data(data_path)
    prs = _select_prs(config)

    failed = 0
    for pr in prs:
        entry = data.get(pr.golden_url)
        if entry is None:
            print_warning(console, f"{pr.golden_url} not in benchmark corpus; skipping")
            failed += 1
            continue

        try:
            modified = _process_pr(config, pr, data)
        except Exception as exc:  # noqa: BLE001 - isolate per-PR failure so the sweep continues
            failed += 1
            print_error(console, "Benchmark PR failed", f"{pr.golden_url}: {type(exc).__name__}: {exc}")
            continue

        if modified:
            save_benchmark_data(data_path, data)
            print_info(console, f"Injected daydream review for {pr.golden_url}")
        else:
            print_dim(console, f"Skipping {pr.golden_url} (daydream review already present)")

    if failed:
        print_warning(console, f"{failed} of {len(prs)} selected PR(s) failed")

    score_failed = False
    if config.score:
        try:
            scores = run_scoring(config.benchmark_repo, config.model)
        except Exception as exc:  # noqa: BLE001 - report scoring failure without raising past the CLI
            score_failed = True
            print_error(console, "Scoring failed", f"{type(exc).__name__}: {exc}")
        else:
            for golden_url, leaf in scores.per_pr.items():
                print_info(
                    console,
                    f"{golden_url}: tp={leaf.get('tp', 0)} fp={leaf.get('fp', 0)} fn={leaf.get('fn', 0)}",
                )
            print_success(
                console,
                f"daydream aggregate over {scores.scored_pr_count} PR(s): "
                f"precision={scores.precision:.3f} recall={scores.recall:.3f}",
            )

    return 0 if failed == 0 and not score_failed else 1
