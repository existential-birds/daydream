"""Unified JSON bench report — per-PR cost/latency/tokens plus score leaves.

One report is emitted for every ``daydream bench`` run (single-shot and
multi-trial) at ``<corpus root>/.daydream-bench/report-<tool_label>.json``,
with the same shape for both corpus kinds (withmartian and harvested).

:func:`build_report` is pure: every value it needs is passed in, so the shape
can be tested without a corpus, a git checkout, or a clock.

Cost accounting: a trajectory that reports a non-zero ``total_cost_usd`` wins
(``cost_source == "measured"``). Otherwise the cost is synthesized from the
token counters via :mod:`daydream.pricing`, honoring the fact that the ``pi``
backend reports **disjoint** counters while the others report cached input as a
subset of the total (see :func:`synthesize_cost`). Price cards for models the
built-in table does not know (GLM via z.ai, for instance) enter through the
existing ``$DAYDREAM_PRICES_FILE`` override, not a table in this module.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream import pricing

if TYPE_CHECKING:
    from daydream.benchmark.score import DaydreamScores

#: Bump when the report's key set or semantics change incompatibly.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PRRun:
    """One PR's measured review outcome, as accumulated by the sweep.

    Attributes:
        golden_url: The corpus key for the PR.
        injected_comments: Number of review comments injected under the run's
            tool label.
        elapsed_s: Wall-clock seconds the review took (0.0 when skipped).
        trajectory_path: Where the reviewer's ATIF trajectory was written; may
            not exist when the review was skipped or failed.
        score_leaf: The PR's ``daydream`` scoring leaf, or None when scoring
            was off or the PR was not scored.
    """

    golden_url: str
    injected_comments: int
    elapsed_s: float
    trajectory_path: Path
    score_leaf: dict[str, Any] | None = field(default=None)


def _read_final_metrics(trajectory_path: Path) -> dict[str, Any]:
    """Read a trajectory's ``final_metrics`` block, tolerating missing/partial files.

    Returns:
        The ``final_metrics`` mapping, or ``{}`` when the file is absent,
        unreadable, not JSON, not an object, or carries no metrics block.
    """
    try:
        raw = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    metrics = raw.get("final_metrics")
    return metrics if isinstance(metrics, dict) else {}


def synthesize_cost(
    *,
    backend: str | None,
    model: str | None,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Synthesize a USD cost from token counters, or None for unknown models.

    The two branches are not interchangeable. The ``pi`` backend maps
    ``usage.input`` → prompt and ``usage.cacheRead`` → cached as **disjoint**
    counters (``daydream/backends/pi.py``), so both are billed in full. Claude
    and Codex report cached input as a *subset* of the total prompt count, so
    the cached portion must be subtracted before pricing fresh input —
    :func:`daydream.pricing.compute_cost_from_totals` does exactly that.
    Collapsing the branches under-bills pi runs, where cache reads routinely
    dwarf fresh input.
    """
    if model is None:
        return None
    # load_user_prices() is what reads $DAYDREAM_PRICES_FILE; resolve_prices()
    # alone would only ever see the built-in table.
    prices = pricing.resolve_prices(pricing.load_user_prices())
    if backend == "pi":
        return pricing.compute_cost(
            model, prompt_tokens, cached_tokens, completion_tokens, prices=prices
        )
    return pricing.compute_cost_from_totals(
        model,
        total_input_tokens=prompt_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=completion_tokens,
        prices=prices,
    )


def _int_metric(metrics: dict[str, Any], key: str) -> int:
    """Coerce a trajectory token counter to a non-negative int (None → 0)."""
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)


def _pr_entry(
    run: PRRun, *, reviewer_backend: str | None, reviewer_model: str | None
) -> dict[str, Any]:
    """Build one ``prs[]`` entry: latency, tokens, cost, and the score leaf."""
    metrics = _read_final_metrics(run.trajectory_path)
    prompt = _int_metric(metrics, "total_prompt_tokens")
    completion = _int_metric(metrics, "total_completion_tokens")
    cached = _int_metric(metrics, "total_cached_tokens")

    measured = metrics.get("total_cost_usd")
    cost: float | None
    if isinstance(measured, (int, float)) and not isinstance(measured, bool) and measured:
        cost, cost_source = float(measured), "measured"
    else:
        cost = synthesize_cost(
            backend=reviewer_backend,
            model=reviewer_model,
            prompt_tokens=prompt,
            cached_tokens=cached,
            completion_tokens=completion,
        )
        cost_source = "synthesized" if cost is not None else "unknown"

    leaf = run.score_leaf or {}
    return {
        "golden_url": run.golden_url,
        "injected_comments": run.injected_comments,
        "elapsed_s": round(run.elapsed_s, 3),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "cost_usd": cost,
        "cost_source": cost_source,
        "tp": leaf.get("tp"),
        "fp": leaf.get("fp"),
        "fn": leaf.get("fn"),
        "precision": leaf.get("precision"),
        "recall": leaf.get("recall"),
    }


def build_report(
    *,
    corpus: str,
    corpus_root: Path,
    tool_label: str,
    reviewer_backend: str | None,
    reviewer_model: str | None,
    reviewer_provider: str | None,
    judge_route: str,
    judge_model: str | None,
    git_sha: str,
    timestamp: str,
    pr_runs: Sequence[PRRun],
    aggregate: dict[str, Any] | None = None,
    distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the run report. Pure — no I/O beyond reading trajectory files.

    Args:
        corpus: The corpus kind (``"withmartian"`` / ``"harvested"``).
        aggregate: The run-level score summary, or None when scoring was off
            (or, for a multi-trial run, when ``distribution`` supersedes it).
        distribution: The multi-trial score distribution, or None.

    Returns:
        A JSON-ready dict whose key set does not depend on the corpus kind.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus": corpus,
        "corpus_root": str(corpus_root),
        "tool_label": tool_label,
        "reviewer_backend": reviewer_backend,
        "reviewer_model": reviewer_model,
        "reviewer_provider": reviewer_provider,
        "judge_route": judge_route,
        "judge_model": judge_model or None,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "prs": [
            _pr_entry(run, reviewer_backend=reviewer_backend, reviewer_model=reviewer_model)
            for run in pr_runs
        ],
        "aggregate": aggregate,
        "distribution": distribution,
    }


def write_report(path: Path, report: dict[str, Any]) -> Path:
    """Write *report* as indented JSON to *path*, creating parent dirs.

    Returns:
        The path written, so callers can print it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def aggregate_from_scores(scores: DaydreamScores) -> dict[str, Any]:
    """Project a :class:`~daydream.benchmark.score.DaydreamScores` to the report's aggregate block."""
    return {
        "scored_pr_count": scores.scored_pr_count,
        "tp": scores.total_tp,
        "fp": scores.total_fp,
        "fn": scores.total_fn,
        "precision": scores.precision,
        "recall": scores.recall,
        "f1": scores.f1,
    }
