"""Run archive manifest builder.

Assembles a ``manifest.json`` from the recorder, run config, git context,
and optional evaluation results. The manifest is the single source of
truth for what's in an archive bundle.

Exports:
    MANIFEST_SCHEMA_VERSION: Current schema version string.
    Manifest: Dataclass representing the manifest.
    build_manifest: Construct a Manifest from run context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from daydream.archive.git_context import GitContext

if TYPE_CHECKING:
    from pathlib import Path

    from daydream.runner import RunConfig
    from daydream.trajectory import TrajectoryRecorder

MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass
class Manifest:
    """Archive bundle manifest written to ``manifest.json``.

    Attributes:
        schema_version: Manifest schema version for forward compatibility.
        session_id: UUID4 session identifier from the trajectory recorder.
        archived_at: ISO 8601 timestamp of when the archive was created.
        status: Run status — ``complete``, ``partial``, or ``failed``.
        run_flow: Run flow type (normal, ttt, pr, deep).
        skill: Review skill used (python, react, etc.).
        model: Model name (opus, sonnet, haiku).
        backend: Backend used (claude, codex).
        review_only: Whether the run was review-only.
        deep: Whether deep review mode was used.
        loop: Whether loop mode was enabled.
        remote_url: Git remote origin URL.
        repo_slug: ``owner/repo`` extracted from remote URL.
        branch: Git branch name at run time.
        base_branch: Default branch (main/master).
        head_sha: Git HEAD commit SHA.
        pr_number: GitHub PR number if applicable.
        pr_repo: GitHub repo slug for PR.
        total_cost_usd: Total cost from trajectory final metrics.
        total_prompt_tokens: Non-cached prompt tokens.
        total_completion_tokens: Completion tokens.
        total_cached_tokens: Cached tokens.
        wall_clock_seconds: Wall-clock duration (from eval, if available).
        total_findings: Number of findings (from eval, if available).
        grounding_rate: Grounding rate (from eval, if available).
        coverage_ratio: File coverage ratio (from eval, if available).
        cost_per_finding_usd: Cost per finding (from eval, if available).
        outcome_labels: JSON-encoded list of outcome labels.
        labeled_at: ISO 8601 timestamp of last label update.
        archive_path: Absolute path to the archive directory.
    """

    schema_version: str = MANIFEST_SCHEMA_VERSION
    session_id: str = ""
    archived_at: str = ""
    status: str = "complete"

    # Run config
    run_flow: str = ""
    skill: str | None = None
    model: str | None = None
    backend: str = "claude"
    review_only: bool = False
    deep: bool = False
    loop: bool = False

    # Git context
    remote_url: str | None = None
    repo_slug: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    head_sha: str | None = None

    # PR context
    pr_number: int | None = None
    pr_repo: str | None = None

    # Metrics (from trajectory _final_totals)
    total_cost_usd: float | None = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None

    # Evaluation metrics (populated only with --eval)
    wall_clock_seconds: float | None = None
    total_findings: int | None = None
    grounding_rate: float | None = None
    coverage_ratio: float | None = None
    cost_per_finding_usd: float | None = None

    # Outcome labels (populated via `daydream label`)
    outcome_labels: str = field(default="[]")
    labeled_at: str | None = None

    # Archive location
    archive_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "archived_at": self.archived_at,
            "status": self.status,
            "run": {
                "flow": self.run_flow,
                "skill": self.skill,
                "model": self.model,
                "backend": self.backend,
                "review_only": self.review_only,
                "deep": self.deep,
                "loop": self.loop,
            },
            "git": {
                "remote_url": self.remote_url,
                "repo_slug": self.repo_slug,
                "branch": self.branch,
                "base_branch": self.base_branch,
                "head_sha": self.head_sha,
            },
            "pr": {
                "number": self.pr_number,
                "repo": self.pr_repo,
            },
            "metrics": {
                "total_cost_usd": self.total_cost_usd,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_cached_tokens": self.total_cached_tokens,
                "wall_clock_seconds": self.wall_clock_seconds,
                "total_findings": self.total_findings,
                "grounding_rate": self.grounding_rate,
                "coverage_ratio": self.coverage_ratio,
                "cost_per_finding_usd": self.cost_per_finding_usd,
            },
            "outcome": {
                "labels": json.loads(self.outcome_labels),
                "labeled_at": self.labeled_at,
            },
            "archive_path": self.archive_path,
        }


def build_manifest(
    *,
    recorder: TrajectoryRecorder,
    config: RunConfig,
    git_ctx: GitContext,
    status: str,
    archive_path: Path,
    evaluation: dict[str, Any] | None = None,
) -> Manifest:
    """Construct a Manifest from run context.

    Args:
        recorder: The TrajectoryRecorder that produced the trajectory.
        config: The RunConfig for this run.
        git_ctx: Captured git metadata.
        status: Run status (``complete``, ``partial``, ``failed``).
        archive_path: Absolute path to the archive directory for this run.
        evaluation: Optional ``analyze_session()`` result dict.

    Returns:
        A fully populated Manifest.
    """
    totals = recorder._final_totals  # noqa: SLF001 - intentional access to recorder internals

    m = Manifest(
        session_id=recorder.session_id,
        archived_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        run_flow=recorder.run_flow.value,
        skill=config.skill,
        model=config.model,
        backend=config.backend,
        review_only=config.review_only,
        deep=config.deep,
        loop=config.loop,
        remote_url=git_ctx.remote_url,
        repo_slug=git_ctx.repo_slug,
        branch=git_ctx.branch,
        base_branch=git_ctx.base_branch,
        head_sha=git_ctx.head_sha,
        pr_number=recorder.pr_number,
        pr_repo=recorder.pr_repo,
        total_cost_usd=totals["cost"] if totals.get("any_cost_seen") else None,
        total_prompt_tokens=totals["prompt"] or None,
        total_completion_tokens=totals["completion"] or None,
        total_cached_tokens=totals["cached"] or None,
        archive_path=str(archive_path),
    )

    if evaluation:
        timing = evaluation.get("timing", {})
        m.wall_clock_seconds = timing.get("total_wall_clock_seconds")

        findings = evaluation.get("findings", {})
        m.total_findings = findings.get("total")

        grounding = evaluation.get("grounding", {})
        m.grounding_rate = grounding.get("grounding_rate")

        coverage = evaluation.get("coverage", {})
        m.coverage_ratio = coverage.get("coverage_ratio")

        derived = evaluation.get("derived", {})
        m.cost_per_finding_usd = derived.get("cost_per_finding_usd")

    return m
