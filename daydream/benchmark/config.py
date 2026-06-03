"""Configuration for the benchmark harness.

Exports:
    BenchConfig: Frozen dataclass carrying all `daydream bench` settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchConfig:
    """Immutable configuration for a benchmark run.

    Attributes:
        benchmark_repo: Path to the external `code-review-benchmark` checkout.
        cache_dir: Directory for per-PR blobless clones and fetched heads.
        model: Judge model id (also names the per-model results dir).
        force: Re-run PRs even if a `tool:"daydream"` review already exists.
        score: Whether to drive the step2/2.5/3 scoring pipeline.
        only: Optional PR selector (single PR key) to restrict the run.
        limit: Optional cap on the number of PRs processed.
        trajectory_dir: Directory for per-PR ATIF trajectory files.
    """

    benchmark_repo: Path
    cache_dir: Path
    force: bool
    score: bool
    only: str | None
    limit: int | None
    trajectory_dir: Path
    model: str = "anthropic/claude-opus-4.5"
