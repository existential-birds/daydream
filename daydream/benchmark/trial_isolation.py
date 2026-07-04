"""Isolated per-trial corpus dirs for repeated-trial benchmarking.

Each repeated trial materializes its own standard corpus dir — its own
``results/benchmark_data.json`` plus a trial-suffixed tool label — so every
trial stays independently re-scorable by the unmodified withmartian steps.
The canonical ``results/benchmark_data.json`` (the external leaderboard
contract) is never touched: trial data lives entirely under
``<benchmark_repo>/.daydream-bench/trials/<tool_label>/``.

Exports:
    trials_root: Root dir holding all trials for one reviewer label.
    trial_corpus_dir: Per-trial dir path.
    init_trial_corpus: Seed a trial dir with a copy of the canonical corpus.
    trial_tool_label: Suffix a base label with the trial index (e.g. ``daydream-t00``).
"""

from __future__ import annotations

import shutil
from pathlib import Path


def trials_root(benchmark_repo: Path, tool_label: str) -> Path:
    """Resolve the root dir holding every trial for one reviewer label."""
    return benchmark_repo / ".daydream-bench" / "trials" / tool_label


def trial_corpus_dir(benchmark_repo: Path, tool_label: str, trial_index: int) -> Path:
    """Resolve the isolated corpus dir for a single trial.

    Args:
        benchmark_repo: Path to the external benchmark checkout.
        tool_label: Base reviewer results label (e.g. ``daydream``).
        trial_index: Zero-based trial number.

    Returns:
        ``<benchmark_repo>/.daydream-bench/trials/<tool_label>/trial-<NN>``.
    """
    return trials_root(benchmark_repo, tool_label) / f"trial-{trial_index:02d}"


def init_trial_corpus(canonical_data_path: Path, trial_dir: Path) -> Path:
    """Seed a trial dir with a fresh copy of the canonical corpus.

    Creates ``<trial_dir>/results/`` and copies the canonical
    ``benchmark_data.json`` into it as the trial's independent working corpus.
    The trial's reviews are injected there, keeping the canonical corpus pristine.

    Args:
        canonical_data_path: Path to the canonical ``results/benchmark_data.json``.
        trial_dir: The per-trial dir (see :func:`trial_corpus_dir`).

    Returns:
        The path to the trial's ``results/benchmark_data.json``.
    """
    trial_results = trial_dir / "results"
    trial_results.mkdir(parents=True, exist_ok=True)
    dest = trial_results / "benchmark_data.json"
    shutil.copy2(canonical_data_path, dest)
    return dest


def trial_tool_label(base_label: str, trial_index: int) -> str:
    """Suffix a base results label with a zero-padded trial index.

    The suffix makes each trial write a distinct ``evaluations.json`` under the
    unmodified withmartian steps, so trials never overwrite one another.

    Args:
        base_label: The base reviewer label (e.g. ``daydream``).
        trial_index: Zero-based trial number.

    Returns:
        ``<base_label>-t<NN>`` (e.g. ``daydream-t00``).
    """
    return f"{base_label}-t{trial_index:02d}"
