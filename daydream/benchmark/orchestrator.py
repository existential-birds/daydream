"""Benchmark orchestrator — acquire → review → map → inject (+ optional score).

Drives the full benchmark sweep over the pinned evaluable PRs. For each
selected PR it acquires a checkout, runs a non-interactive daydream review,
maps the canonical findings to benchmark review comments, and injects a
synthetic ``daydream`` review into the corpus.

Path convention: the benchmark corpus is read from and written to
``config.benchmark_repo / "results" / "benchmark_data.json"`` (the layout the
benchmark scoring artifacts expect). The corpus is saved after every PR so an
interrupted sweep is resumable.

A single PR's failure is logged and recorded but does not abort the sweep; the
returned exit code is non-zero if any selected PR failed. When ``config.score``
is set the judge environment is verified up front (before any expensive review)
and, after the sweep, the step2/2.5/3 scoring pipeline runs and its per-PR and
aggregate precision/recall are printed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.agent import console
from daydream.benchmark.acquire import acquire_checkout
from daydream.benchmark.benchmark_data import (
    has_daydream_review,
    inject_daydream_review,
    load_benchmark_data,
    save_benchmark_data,
)
from daydream.benchmark.cli import _format_elapsed
from daydream.benchmark.daydream_run import run_daydream_review
from daydream.benchmark.mapping import merged_items_to_review_comments
from daydream.benchmark.prs import load_evaluable_prs
from daydream.benchmark.score import preflight_judge_env, resolve_judge_model, run_scoring
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


def _injected_comment_count(data: dict[str, Any], golden_url: str, tool: str) -> int:
    """Count the review comments injected for *golden_url* under *tool*."""
    entry = data.get(golden_url, {})
    for review in entry.get("reviews", []):
        if review.get("tool") == tool:
            return len(review.get("review_comments", []))
    return 0


def _process_pr(config: BenchConfig, pr: EvaluablePR, data: dict[str, Any]) -> bool:
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
    on_line = (lambda line: console.out(line, end="", highlight=False)) if config.verbose else None
    artifact = run_daydream_review(
        checkout,
        base_sha=pr.base_sha,
        trajectory_path=_trajectory_path(config, pr),
        backend=config.reviewer_backend,
        model=config.reviewer_model,
        provider=config.reviewer_provider,
        on_line=on_line,
    )
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    comments = merged_items_to_review_comments(doc, created_at=_CREATED_AT)
    return inject_daydream_review(data, pr.golden_url, comments, force=config.force, tool=config.tool_label)


def run_bench(config: BenchConfig) -> int:
    """Run the benchmark sweep over the selected PRs.

    Args:
        config: Immutable run configuration (selection, force, scoring, paths).

    Returns:
        ``0`` when every selected PR was injected (or skipped) and scoring (if
        requested) succeeded; non-zero if any selected PR failed or scoring
        failed.
    """
    judge_model = ""
    if config.score:
        preflight_judge_env(judge_route=config.judge_route)
        judge_model = resolve_judge_model(config.model)

    data_path = _benchmark_data_path(config)
    data = load_benchmark_data(data_path)
    prs = _select_prs(config)

    failed = 0
    total = len(prs)
    for i, pr in enumerate(prs, start=1):
        entry = data.get(pr.golden_url)
        if entry is None:
            print_warning(console, f"{pr.golden_url} not in benchmark corpus; skipping")
            failed += 1
            continue

        if not config.force and has_daydream_review(entry, tool=config.tool_label):
            print_dim(console, f"Skipping {pr.golden_url} (daydream review already present)")
            continue

        print_info(console, f"▶ [{i}/{total}] Reviewing {pr.golden_url} · reviewer {config.tool_label}…")
        elapsed = 0.0
        try:
            started = time.monotonic()
            if config.verbose:
                # Streaming and a spinner can't share one console; verbose forwards
                # the child output live, so the spinner is gated off here.
                modified = _process_pr(config, pr, data)
            else:
                with console.status(f"Reviewing {pr.golden_url}…"):
                    modified = _process_pr(config, pr, data)
            elapsed = time.monotonic() - started
        except Exception as exc:  # noqa: BLE001 - isolate per-PR failure so the sweep continues
            failed += 1
            print_error(console, "Benchmark PR failed", f"{pr.golden_url}: {type(exc).__name__}: {exc}")
            continue

        if modified:
            save_benchmark_data(data_path, data)
            print_info(console, f"Injected daydream review for {pr.golden_url}")

        count = _injected_comment_count(data, pr.golden_url, config.tool_label)
        noun = "finding" if count == 1 else "findings"
        print_success(console, f"Reviewed {pr.golden_url} in {_format_elapsed(elapsed)} · {count} {noun}")

    if failed:
        print_warning(console, f"{failed} of {len(prs)} selected PR(s) failed")

    score_failed = False
    if config.score:
        try:
            scores = run_scoring(
                config.benchmark_repo,
                judge_model,
                pr_count=len(prs),
                tool=config.tool_label,
                judge_route=config.judge_route,
            )
        except Exception as exc:  # noqa: BLE001 - report scoring failure without raising past the CLI
            score_failed = True
            print_error(console, "Scoring failed", f"{type(exc).__name__}: {exc}")
            injected = len(prs) - failed
            print_info(
                console,
                f"{injected} of {len(prs)} PR(s) injected successfully; "
                "corpus is saved and can be re-scored separately (re-run with --score)",
            )
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
