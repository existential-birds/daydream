"""Judge-step scoring helpers for the benchmark harness.

Exports:
    preflight_judge_env: Verify the judge credential is present in the environment.
    model_results_dir: Resolve the per-model results directory for a benchmark repo.
"""

from __future__ import annotations

import os
from pathlib import Path

JUDGE_API_KEY_ENV = "MARTIAN_API_KEY"


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
