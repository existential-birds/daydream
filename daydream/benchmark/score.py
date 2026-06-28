"""Judge-step scoring helpers for the benchmark harness.

Exports:
    preflight_judge_env: Verify the judge credential is present in the environment.
    resolve_judge_model: Resolve the judge model from --model or the MARTIAN_MODEL env.
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
JUDGE_MODEL_ENV = "MARTIAN_MODEL"

#: Benchmark step modules, run in order. Each reads `results/benchmark_data.json`
#: by cwd only, so all three are invoked with ``cwd=benchmark_repo``.
_STEP2_MODULE = "code_review_benchmark.step2_extract_comments"
_STEP2_5_MODULE = "code_review_benchmark.step2_5_dedup_candidates"
_STEP3_MODULE = "code_review_benchmark.step3_judge_comments"

#: The tool under evaluation; injected into `benchmark_data.json` upstream.
_TOOL = "daydream"

#: Tail length (chars) of captured stderr included in failure messages.
_STDERR_TAIL = 4000

#: Fraction of attempted judge comparisons that may error before the whole
#: scoring run is treated as invalid. A bad ``MARTIAN_API_KEY`` makes every
#: judge call return HTTP 401; the upstream step3 records each as an error and
#: still exits 0, so without this guard a wholesale-failed judge is reported as
#: a clean ``precision=0.000 recall=0.000`` — indistinguishable from a genuinely
#: poor review. Above this ratio the scores are noise, not a real zero.
_JUDGE_ERROR_RATIO_THRESHOLD = 0.5

#: Maximum wall-clock seconds to wait for each benchmark step subprocess.
#: Step 3 calls an external judge API and can be slow; 30 minutes is generous.
_STEP_TIMEOUT = 1800


class JudgeEnvError(Exception):
    """Raised when the judge API credential is absent from the environment."""


class BenchmarkStepError(Exception):
    """Raised when a benchmark step module exits non-zero."""


class BenchmarkArtifactError(Exception):
    """Raised when an expected benchmark output file is absent after a successful step."""


class JudgeFailedError(Exception):
    """Raised when the judge errored on too many comparisons to trust the scores.

    The withmartian step3 records each failed judge LLM call (e.g. an HTTP 401
    from a bad ``MARTIAN_API_KEY``) as an error and still exits 0, emitting an
    `evaluations.json` whose tp/fp/fn collapse to a clean-looking zero. This is
    raised when the error ratio crosses `_JUDGE_ERROR_RATIO_THRESHOLD`, so a
    wholesale-failed judge surfaces loudly instead of as a genuine zero.
    """


def preflight_judge_env() -> None:
    """Verify the judge API credential is present in the process environment.

    Reads `os.environ` only (no `.env` or secret-file parsing). The credential
    value is never logged.

    Raises:
        JudgeEnvError: If `MARTIAN_API_KEY` is unset or empty.
    """
    if not os.environ.get(JUDGE_API_KEY_ENV):
        raise JudgeEnvError(
            f"{JUDGE_API_KEY_ENV} is not set; export it (e.g. an OpenRouter sk-or-… key) "
            "before running the judge step."
        )


def resolve_judge_model(model: str | None) -> str:
    """Resolve the judge model from an explicit ``--model`` or the environment.

    The judge model is the single source that drives where the harness writes
    its results (`get_model_dir` in every step derives `results/<model>` from
    `MARTIAN_MODEL`). To keep our reader and the harness writer in lockstep,
    `run_scoring` exports the resolved value back into `MARTIAN_MODEL` for the
    step subprocesses and derives the results dir from the same value — so the
    two cannot diverge. There is deliberately **no** hardcoded default: scoring
    with an unnamed judge would silently grade against whatever model the
    harness defaults to, in a directory we did not name.

    Args:
        model: Explicit judge model id from ``--model`` (or ``None``).

    Returns:
        The resolved judge model id (``model`` if given, else ``MARTIAN_MODEL``).

    Raises:
        JudgeEnvError: If ``model`` is ``None``/empty and ``MARTIAN_MODEL`` is unset.
    """
    resolved = model or os.environ.get(JUDGE_MODEL_ENV)
    if not resolved:
        raise JudgeEnvError(
            f"Judge model unspecified: pass --model or export {JUDGE_MODEL_ENV} "
            "(the withmartian judge model). Refusing to assume a default — the judge "
            "would grade against an unnamed model and write results to a dir we cannot locate."
        )
    return resolved


def model_results_dir(benchmark_repo: Path, model: str) -> Path:
    """Resolve the per-model results directory inside the benchmark repo.

    Mirrors the benchmark's `sanitize_model_name` exactly (`strip()` then
    replace `/` with `_`); ``model`` must be the resolved judge model that
    `run_scoring` also exports as `MARTIAN_MODEL`, so this path matches where
    the harness actually wrote.

    Args:
        benchmark_repo: Path to the external benchmark checkout.
        model: Resolved judge model id (see `resolve_judge_model`).

    Returns:
        The `results/<sanitized-model>` directory path.
    """
    return benchmark_repo / "results" / model.strip().replace("/", "_")


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
        total_errors: Summed judge-comparison errors across all scored PRs.
        total_comparisons: Summed attempted judge comparisons
            (Σ candidates × golden) across all scored PRs.
        precision: Aggregate ΣTP / (ΣTP + ΣFP); 0.0 when the denominator is 0.
        recall: Aggregate ΣTP / (ΣTP + ΣFN); 0.0 when the denominator is 0.
    """

    per_pr: dict[str, dict[str, Any]] = field(default_factory=dict)
    scored_pr_count: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_errors: int = 0
    total_comparisons: int = 0
    precision: float = 0.0
    recall: float = 0.0


def parse_daydream_scores(evals: dict[str, dict[str, Any]], *, tool: str = _TOOL) -> DaydreamScores:
    """Extract per-PR and aggregate scores for ``tool`` from a parsed evaluations dict.

    Only the ``tool`` leaf of each PR entry is retained; other tools' leaves are
    dropped. Aggregate precision/recall are computed from summed TP/FP/FN (micro
    averaging), with a zero denominator yielding 0.0.

    Args:
        evals: The parsed `evaluations.json` object — golden PR URL → tool → leaf.
        tool: Results label to extract (defaults to ``_TOOL``).

    Returns:
        A `DaydreamScores` capturing per-PR tool leaves and the aggregate.

    Raises:
        JudgeFailedError: If the judge errored on at least
            `_JUDGE_ERROR_RATIO_THRESHOLD` of all attempted comparisons — the
            scores are then noise, not a real zero (see `JudgeFailedError`).
    """
    per_pr: dict[str, dict[str, Any]] = {}
    total_tp = total_fp = total_fn = 0
    total_errors = total_comparisons = 0
    for golden_url, tools in evals.items():
        leaf = tools.get(tool)
        if leaf is None:
            continue
        per_pr[golden_url] = leaf
        total_tp += int(leaf.get("tp", 0))
        total_fp += int(leaf.get("fp", 0))
        total_fn += int(leaf.get("fn", 0))
        total_errors += int(leaf.get("errors_count", 0))
        # Each leaf compares every candidate against every golden comment, so the
        # attempted comparison count is the product (matches step3's task grid).
        total_comparisons += int(leaf.get("total_candidates", 0)) * int(leaf.get("total_golden", 0))

    if total_comparisons and total_errors / total_comparisons >= _JUDGE_ERROR_RATIO_THRESHOLD:
        ratio = total_errors / total_comparisons
        raise JudgeFailedError(
            f"Judge errored on {total_errors}/{total_comparisons} comparisons "
            f"({ratio:.0%}) for tool {tool!r}; the precision/recall are invalid, not a real "
            f"zero. Most likely the judge credential ({JUDGE_API_KEY_ENV}) was rejected "
            f"(HTTP 401). Fix the credential and re-run --score against the saved corpus."
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    return DaydreamScores(
        per_pr=per_pr,
        scored_pr_count=len(per_pr),
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_errors=total_errors,
        total_comparisons=total_comparisons,
        precision=precision,
        recall=recall,
    )


def _run_step(module: str, extra_args: list[str], *, cwd: Path, tool: str = _TOOL, judge_model: str) -> None:
    """Run one benchmark step module via `uv run python -m`.

    The step modules read `results/benchmark_data.json` by cwd only, so `cwd` must
    be the benchmark checkout. `os.environ` is inherited so the steps see the
    `MARTIAN_*` credentials; `MARTIAN_MODEL` is overridden to ``judge_model`` so
    every step's `get_model_dir` resolves to the same `results/<model>` directory
    that `run_scoring` reads from.

    Args:
        module: Dotted module path (e.g. `code_review_benchmark.step3_judge_comments`).
        extra_args: Module-specific CLI arguments appended after `--tool <tool>`.
        cwd: The benchmark repo directory.
        tool: Results label passed as `--tool` (defaults to ``_TOOL``).
        judge_model: Resolved judge model exported as `MARTIAN_MODEL` for the step.

    Raises:
        BenchmarkStepError: If the step exits non-zero; the message carries the module
            name and a tail of its stderr.
    """
    env = os.environ.copy()
    env[JUDGE_MODEL_ENV] = judge_model
    cmd = ["uv", "run", "python", "-m", module, "--tool", tool, *extra_args]  # noqa: S607 - uv is a trusted command
    try:
        result = subprocess.run(  # noqa: S603 - args are not user-controlled; module names are fixed literals
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=_STEP_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkStepError(
            f"{module} timed out after {_STEP_TIMEOUT}s"
        ) from exc
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-_STDERR_TAIL:]
        raise BenchmarkStepError(f"{module} failed (exit {result.returncode}):\n{stderr_tail}")


def run_scoring(
    benchmark_repo: Path, judge_model: str, *, pr_count: int | None = None, tool: str = _TOOL
) -> DaydreamScores:
    """Run step2/2.5/3 against the benchmark repo and parse the tool's scores.

    Runs the three benchmark step modules in order (extract → dedup → judge),
    each with `--tool <tool>`, `cwd` set to the benchmark checkout, and
    `MARTIAN_MODEL` exported as ``judge_model``. step3 additionally receives
    `--dedup-groups` pointing at step2.5's output. Finally loads
    `evaluations.json` and parses it.

    ``judge_model`` is the single source of truth: it is exported into every
    step's environment (so the harness writes to `results/<judge_model>`) **and**
    used to derive the results dir we read from — eliminating any drift between a
    bench-supplied model string and the model the harness actually ran. Resolve
    it via `resolve_judge_model` before calling. The judge credential is verified
    by the caller (``run_bench``) before any expensive reviews run.

    Args:
        benchmark_repo: Path to the external benchmark checkout.
        judge_model: Resolved judge model id (see `resolve_judge_model`).
        pr_count: When provided, passed as ``--limit`` to step3 so the judge
            only evaluates that many PRs.  Use this to bound judge cost when
            ``--limit`` was passed to the harness.
        tool: Results label under evaluation (defaults to ``_TOOL``).

    Returns:
        The parsed `DaydreamScores`.

    Raises:
        BenchmarkStepError: If any step exits non-zero.
        BenchmarkArtifactError: If `evaluations.json` is absent after a successful step3.
    """
    # Resolve to an absolute path: each step runs with ``cwd`` set to the
    # benchmark checkout, so any path handed to a step as an argument (notably
    # step3's ``--dedup-groups``) is re-interpreted against that cwd. A
    # benchmark-repo-relative path (e.g. ``../code-review-benchmark/offline``)
    # would double up and miss; an absolute path is cwd-independent.
    benchmark_repo = benchmark_repo.resolve()
    results_dir = model_results_dir(benchmark_repo, judge_model)
    dedup_groups = results_dir / "dedup_groups.json"

    _run_step(_STEP2_MODULE, [], cwd=benchmark_repo, tool=tool, judge_model=judge_model)
    _run_step(_STEP2_5_MODULE, [], cwd=benchmark_repo, tool=tool, judge_model=judge_model)
    step3_extra: list[str] = ["--dedup-groups", str(dedup_groups)]
    if pr_count is not None:
        step3_extra += ["--limit", str(pr_count)]
    _run_step(_STEP3_MODULE, step3_extra, cwd=benchmark_repo, tool=tool, judge_model=judge_model)

    evaluations_file = results_dir / "evaluations.json"
    if not evaluations_file.exists():
        raise BenchmarkArtifactError(
            f"{evaluations_file} not found after step3; the judge step produced no evaluations."
        )
    evals = json.loads(evaluations_file.read_text())
    return parse_daydream_scores(evals, tool=tool)
