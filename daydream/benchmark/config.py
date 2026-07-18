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
        benchmark_repo: Path to the external `code-review-benchmark` checkout,
            or None when the run is driven by a harvested corpus instead.
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
        min_confidence: When set, drop findings below this confidence
            (low/medium/high) before benchmark submission; None submits all.
        min_severity: When set, drop findings below this severity
            (low/medium/high) before benchmark submission; None submits all.
        trials: Number of full end-to-end trials to run per reviewer config
            (default 1 = single-shot, back-compat). N>1 isolates each trial in
            its own corpus dir and enables distribution reporting.
        harvest_dir: Root of a harvested bot-review corpus (see
            `daydream bench harvest`), or None for a withmartian run. Exactly
            one of `benchmark_repo` / `harvest_dir` is set; the CLI enforces it.
    """

    benchmark_repo: Path | None
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
    min_confidence: str | None = None
    min_severity: str | None = None
    trials: int = 1
    harvest_dir: Path | None = None

    @property
    def corpus_root(self) -> Path:
        """Resolve the corpus root: the harvest dir or the benchmark repo.

        Returns:
            Whichever of ``harvest_dir`` / ``benchmark_repo`` is set. Both being
            set is unreachable — the CLI rejects it before a config is built.

        Raises:
            ValueError: If neither is set.
        """
        if self.harvest_dir is not None:
            return self.harvest_dir
        if self.benchmark_repo is not None:
            return self.benchmark_repo
        raise ValueError("BenchConfig needs benchmark_repo or harvest_dir")
