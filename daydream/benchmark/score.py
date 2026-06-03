"""Judge-step scoring helpers for the benchmark harness.

Exports:
    preflight_judge_env: Verify the judge credential is present in the environment.
    model_results_dir: Resolve the per-model results directory for a benchmark repo.
    run_scoring: Run step2/2.5/3 and parse the resulting daydream precision/recall.
    parse_daydream_scores: Extract per-PR and aggregate daydream scores from evaluations.
    DaydreamScores: Aggregated daydream scoring result.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JUDGE_API_KEY_ENV = "MARTIAN_API_KEY"

#: Benchmark step modules, run in order. Each reads `results/benchmark_data.json`
#: by cwd only, so all three are invoked with ``cwd=benchmark_repo``.
_STEP2_MODULE = "code_review_benchmark.step2_extract_comments"
_STEP2_5_MODULE = "code_review_benchmark.step2_5_dedup_candidates"
_STEP3_MODULE = "code_review_benchmark.step3_judge_comments"

#: The tool under evaluation; injected into `benchmark_data.json` upstream.
_TOOL = "daydream"

#: Tail length (chars) of captured stderr included in failure messages.
_STDERR_TAIL = 4000


def preflight_judge_env() -> None:
    """Verify the judge API credential is present in the process environment.

    Reads `os.environ` only (no `.env` or secret-file parsing). The credential
    value is never logged.

    Raises:
        EnvironmentError: If `MARTIAN_API_KEY` is unset or empty.
    """
    if not os.environ.get(JUDGE_API_KEY_ENV):
        raise EnvironmentError(
            f"{JUDGE_API_KEY_ENV} is not set; export it (e.g. an OpenRouter sk-or-… key) "
            "before running the judge step."
        )


def model_results_dir(benchmark_repo: Path, model: str) -> Path:
    """Resolve the per-model results directory inside the benchmark repo.

    Mirrors the benchmark's `sanitize_model_name`, which only replaces `/` with
    `_` (dots and other characters are preserved).

    Args:
        benchmark_repo: Path to the external benchmark checkout.
        model: Judge model id (e.g. `anthropic/claude-opus-4.5`).

    Returns:
        The `results/<sanitized-model>` directory path.
    """
    return benchmark_repo / "results" / model.replace("/", "_")


@dataclass
class DaydreamScores:
    """Aggregated daydream scoring result parsed from `evaluations.json`.

    Attributes:
        per_pr: Mapping of golden PR URL to that PR's `daydream` leaf object only
            (other tools' leaves are excluded).
        scored_pr_count: Number of PRs that produced a `daydream` leaf.
        total_tp: Summed true positives across all scored PRs.
        total_fp: Summed false positives across all scored PRs.
        total_fn: Summed false negatives across all scored PRs.
        precision: Aggregate ΣTP / (ΣTP + ΣFP); 0.0 when the denominator is 0.
        recall: Aggregate ΣTP / (ΣTP + ΣFN); 0.0 when the denominator is 0.
    """

    per_pr: dict[str, dict[str, Any]] = field(default_factory=dict)
    scored_pr_count: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    precision: float = 0.0
    recall: float = 0.0


def parse_daydream_scores(evals: dict[str, dict[str, Any]]) -> DaydreamScores:
    """Extract per-PR and aggregate daydream scores from a parsed evaluations dict.

    Only the `daydream` leaf of each PR entry is retained; other tools' leaves are
    dropped. Aggregate precision/recall are computed from summed TP/FP/FN (micro
    averaging), with a zero denominator yielding 0.0.

    Args:
        evals: The parsed `evaluations.json` object — golden PR URL → tool → leaf.

    Returns:
        A `DaydreamScores` capturing per-PR daydream leaves and the aggregate.
    """
    per_pr: dict[str, dict[str, Any]] = {}
    total_tp = total_fp = total_fn = 0
    for golden_url, tools in evals.items():
        leaf = tools.get(_TOOL)
        if leaf is None:
            continue
        per_pr[golden_url] = leaf
        total_tp += int(leaf.get("tp", 0))
        total_fp += int(leaf.get("fp", 0))
        total_fn += int(leaf.get("fn", 0))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    return DaydreamScores(
        per_pr=per_pr,
        scored_pr_count=len(per_pr),
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        precision=precision,
        recall=recall,
    )


def _run_step(module: str, extra_args: list[str], *, cwd: Path) -> None:
    """Run one benchmark step module via `uv run python -m`, inheriting the env.

    The step modules read `results/benchmark_data.json` by cwd only, so `cwd` must
    be the benchmark checkout. `os.environ` is inherited so the steps see the
    `MARTIAN_*` credentials.

    Args:
        module: Dotted module path (e.g. `code_review_benchmark.step3_judge_comments`).
        extra_args: Module-specific CLI arguments appended after `--tool daydream`.
        cwd: The benchmark repo directory.

    Raises:
        RuntimeError: If the step exits non-zero; the message carries the module
            name and a tail of its stderr.
    """
    cmd = ["uv", "run", "python", "-m", module, "--tool", _TOOL, *extra_args]
    result = subprocess.run(  # noqa: S603 - args are not user-controlled; module names are fixed literals
        cmd,
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-_STDERR_TAIL:]
        raise RuntimeError(f"{module} failed (exit {result.returncode}):\n{stderr_tail}")


def run_scoring(benchmark_repo: Path, model: str) -> DaydreamScores:
    """Run step2/2.5/3 against the benchmark repo and parse daydream scores.

    Verifies the judge credential, then runs the three benchmark step modules in
    order (extract → dedup → judge), each with `--tool daydream` and `cwd` set to
    the benchmark checkout. step3 additionally receives `--dedup-groups` pointing
    at step2.5's output. Finally loads `evaluations.json` and parses it.

    Args:
        benchmark_repo: Path to the external benchmark checkout.
        model: Judge model id (e.g. `anthropic/claude-opus-4.5`).

    Returns:
        The parsed `DaydreamScores`.

    Raises:
        EnvironmentError: If the judge credential is unset (via `preflight_judge_env`).
        RuntimeError: If any step exits non-zero.
        FileNotFoundError: If `evaluations.json` is absent after a successful step3.
    """
    preflight_judge_env()

    results_dir = model_results_dir(benchmark_repo, model)
    dedup_groups = results_dir / "dedup_groups.json"

    _run_step(_STEP2_MODULE, [], cwd=benchmark_repo)
    _run_step(_STEP2_5_MODULE, [], cwd=benchmark_repo)
    _run_step(_STEP3_MODULE, ["--dedup-groups", str(dedup_groups)], cwd=benchmark_repo)

    evaluations_file = results_dir / "evaluations.json"
    if not evaluations_file.exists():
        raise FileNotFoundError(
            f"{evaluations_file} not found after step3; the judge step produced no evaluations."
        )
    evals = json.loads(evaluations_file.read_text())
    return parse_daydream_scores(evals)
