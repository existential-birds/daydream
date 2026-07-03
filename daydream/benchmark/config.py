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
        judge_route: Benchmark scoring route; "martian" preserves the existing
            OpenAI-compatible Martian path.
        model: Judge model id; None defers to the selected judge route's
            environment fallback. Whatever resolves drives both the judge and
            the per-model results dir.
        reviewer_backend: Optional daydream review backend (claude/codex/pi); None uses the default.
        reviewer_model: Optional model id for the reviewer backend; None uses the backend default.
        reviewer_provider: Optional provider for the reviewer backend (forwarded via env); None uses the default.
        tool_label: Results label for the reviewer tool; must be distinct per reviewer backend.
        force: Re-run PRs even if a `tool:"daydream"` review already exists.
        score: Whether to drive the step2/2.5/3 scoring pipeline.
        only: Optional PR selector (single PR key) to restrict the run.
        limit: Optional cap on the number of PRs processed.
        trajectory_dir: Directory for per-PR ATIF trajectory files.
        verbose: Stream the review subprocess output live (vs. the quiet spinner).
    """

    benchmark_repo: Path
    cache_dir: Path
    force: bool
    score: bool
    only: str | None
    limit: int | None
    trajectory_dir: Path
    judge_route: str = "martian"
    model: str | None = None
    reviewer_backend: str | None = None
    reviewer_model: str | None = None
    reviewer_provider: str | None = None
    tool_label: str = "daydream"
    verbose: bool = False
