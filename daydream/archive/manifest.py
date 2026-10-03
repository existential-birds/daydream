"""Build archive manifests from immutable run snapshots.

``git`` and ``code_context`` describe the reviewed repository; ``daydream``
describes the executable. ``status`` aliases archive finalization, while
``pipeline_status`` records the independent workflow outcome.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from daydream.archive.git_context import GitContext
from daydream.run_snapshot import (
    ArchiveRecorderProvenance as ArchiveRecorderProvenance,
    ArchiveRunSnapshot,
)
from daydream.trajectory import DaydreamRunFlow, compute_timing_summary, snapshot_trajectories

if TYPE_CHECKING:
    from pathlib import Path

    from daydream.archive.provenance import ExecutableProvenance
    from daydream.trajectory import RunWriteSnapshot

MANIFEST_SCHEMA_VERSION = "1.0"


def archive_recorder_provenance_from_snapshot(
    *,
    write_snapshot: RunWriteSnapshot,
    run_flow: DaydreamRunFlow,
) -> ArchiveRecorderProvenance:
    """Validate and retain archive identity from the exact frozen root bytes."""
    session_id = write_snapshot.root_trajectory_id
    if not session_id or session_id in (".", "..") or set(session_id) & set("/\\\0"):
        raise ValueError("snapshot session_id is malformed")
    roots = [
        document for document in write_snapshot.documents if document.trajectory_id == session_id
    ]
    if len(roots) != 1:
        raise ValueError("frozen root trajectory is missing")
    try:
        payload = roots[0].validated_payload(session_id)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen root trajectory is malformed") from exc
    except ValueError as exc:
        raise ValueError(f"frozen root trajectory {exc}") from exc
    extra = payload.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("frozen root trajectory extra is malformed")
    pr_number = extra.get("pr_number")
    if "pr_number" in extra and (type(pr_number) is not int or pr_number <= 0):
        raise ValueError("frozen root pr_number is malformed")
    pr_repo = extra.get("pr_repo")
    if "pr_repo" in extra and (type(pr_repo) is not str or not pr_repo):
        raise ValueError("frozen root pr_repo is malformed")
    return ArchiveRecorderProvenance(
        session_id=session_id,
        run_flow=run_flow,
        pr_number=pr_number,
        pr_repo=pr_repo,
    )


def _omit_falsy(**fields: Any) -> dict[str, Any]:
    """Omit optional fields with falsy values, preserving the manifest wire format."""
    return {k: v for k, v in fields.items() if v}


@dataclass
class Manifest:
    """Archive metadata with separate executable, repository, and workflow provenance."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    # Missing recommended.patch means no recommendation when supported; only legacy
    # manifests fall back to diff.patch.
    recommended_patch_supported: bool = True
    # Capture provenance: post_test is the exact post-heal committed tree; pre_test is
    # the fix-phase fallback. Legacy manifests use None.
    recommended_patch_capture: str | None = None
    session_id: str = ""
    archived_at: str = ""
    status: str = "complete"
    # Archive finalization, byte-identical to the legacy status alias.
    archive_status: str = "complete"
    # Independent workflow outcome: succeeded/failed/partial/cancelled/unknown.
    pipeline_status: str = "unknown"
    # Phase states carry ran/status; legacy manifests omit them.
    phase_states: dict[str, Any] | None = None
    # Frozen agent_budget_stop summary; omitted when no retry ladder stopped.
    retry_summary: dict[str, Any] | None = None
    # Executable identity remains separate from reviewed-repository provenance.
    daydream: Any | None = None

    # Resolved profile identity is required for new runs and omitted on legacy
    # manifests.
    profile_schema_version: int | None = None
    profile_name: str | None = None
    profile_source_kind: str | None = None
    profile_digest: str | None = None

    # Run config
    run_flow: str = ""
    skill: str | None = None
    model: str | None = None
    backend: str = "claude"
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
    per_stack_review_backend: str | None = None
    per_stack_review_model: str | None = None
    review_only: bool = False
    deep: bool = False
    fix_failures: dict[str, str] | None = None
    fix_leftover_untracked: list[str] | None = None
    fix_quality_gate: dict[str, Any] | None = None

    # Git context
    source_path: str | None = None
    remote_url: str | None = None
    repo_slug: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    head_sha: str | None = None
    base_sha: str | None = None
    changed_files: list[str] = field(default_factory=list)

    # PR context
    pr_number: int | None = None
    pr_repo: str | None = None

    # Metrics (from trajectory _final_totals)
    total_cost_usd: float | None = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None

    # Timing comes from frozen events; evaluation supplies the remaining metrics unless
    # disabled.
    wall_clock_seconds: float | None = None
    phase_timings: dict[str, Any] | None = None
    timing_coverage: dict[str, Any] | None = None
    total_findings: int | None = None
    cost_per_finding_usd: float | None = None
    erosion: float | None = None
    verbosity: float | None = None
    location_in_hunk_rate: float | None = None
    shipped_duplicate_pairs: int | None = None

    # Outcome labels (populated via `daydream harvest`)
    outcome_labels: str = field(default="[]")
    labeled_at: str | None = None
    composite_reward: float | None = None

    # Archive location
    archive_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Project flat fields into the portable manifest's named sections."""
        def project(*names: str, omit_falsy: bool = False) -> dict[str, Any]:
            values = {name: getattr(self, name) for name in names}
            return _omit_falsy(**values) if omit_falsy else values

        return {
            **project("schema_version", "recommended_patch_supported"),
            **project("recommended_patch_capture", omit_falsy=True),
            **project("session_id", "archived_at", "status", "archive_status", "pipeline_status"),
            **project(
                "profile_schema_version", "profile_name", "profile_source_kind", "profile_digest",
                omit_falsy=True,
            ),
            **_omit_falsy(daydream=self.daydream.to_dict() if self.daydream is not None else None),
            **project("phase_states", "retry_summary", omit_falsy=True),
            "run": {
                "flow": self.run_flow,
                **project("skill", "model", "backend"),
                **project(
                    "review_backend", "fix_backend", "test_backend", "per_stack_review_backend",
                    "per_stack_review_model", omit_falsy=True,
                ),
                **project("review_only", "deep"),
            },
            **project("fix_failures", "fix_leftover_untracked", "fix_quality_gate"),
            "git": project("source_path", "remote_url", "repo_slug", "branch", "base_branch", "head_sha"),
            "code_context": {
                **project("head_sha", "base_branch", "branch", "base_sha"),
                "changed_files": list(self.changed_files),
            },
            "pr": {"number": self.pr_number, "repo": self.pr_repo},
            "metrics": {
                **project(
                    "total_cost_usd", "total_prompt_tokens", "total_completion_tokens", "total_cached_tokens",
                    "wall_clock_seconds", "phase_timings",
                ),
                **project("timing_coverage", omit_falsy=True),
                **project(
                    "total_findings", "cost_per_finding_usd", "erosion", "verbosity",
                    "location_in_hunk_rate", "shipped_duplicate_pairs",
                ),
            },
            "outcome": {
                "labels": json.loads(self.outcome_labels),
                **project("labeled_at", "composite_reward"),
            },
            "archive_path": self.archive_path,
        }


def build_manifest_from_snapshot(
    *,
    run: ArchiveRunSnapshot,
    git_ctx: GitContext,
    status: str,
    archive_path: Path,
    evaluation: Mapping[str, Any] | None = None,
    source_path: str | None = None,
    fix_failures: Mapping[str, str] | None = None,
    fix_leftover_untracked: Sequence[str] | None = None,
    fix_quality_gate: Mapping[str, Any] | None = None,
    recommended_capture: str | None = None,
    pipeline_status: str = "unknown",
    phase_states: Mapping[str, Any] | None = None,
    retry_summary: Mapping[str, Any] | None = None,
    provenance: ExecutableProvenance | None = None,
) -> Manifest:
    """Construct a manifest from captured identity and frozen trajectory bytes."""
    recorder_provenance = run.recorder_provenance
    write_snapshot = run.trajectories
    identity = run.identity
    if recorder_provenance.session_id != write_snapshot.root_trajectory_id:
        raise ValueError("archive provenance does not match frozen snapshot")
    frozen = snapshot_trajectories(write_snapshot)
    frozen_root = frozen.get("main")
    raw_final_metrics = frozen_root.get("final_metrics") if isinstance(frozen_root, dict) else None
    final_metrics: dict[str, Any] = raw_final_metrics if isinstance(raw_final_metrics, dict) else {}
    raw_cost = final_metrics.get("total_cost_usd")
    timing_summary = compute_timing_summary(write_snapshot)
    runs_fix = identity.phases.fix
    profile = identity.profile

    m = Manifest(
        session_id=recorder_provenance.session_id,
        archived_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        run_flow=recorder_provenance.run_flow.value,
        skill=identity.skill,
        model=identity.model,
        backend=identity.backend,
        review_backend=identity.review_backend,
        fix_backend=identity.fix_backend,
        test_backend=identity.test_backend,
        per_stack_review_backend=identity.per_stack_review_backend,
        per_stack_review_model=identity.per_stack_review_model,
        archive_status=status,
        pipeline_status=pipeline_status,
        phase_states=dict(phase_states) if phase_states is not None else None,
        retry_summary=dict(retry_summary) if retry_summary is not None else None,
        daydream=provenance,
        review_only=identity.review_only,
        deep=identity.deep,
        fix_failures=(dict(fix_failures) or None) if runs_fix and fix_failures is not None else None,
        fix_leftover_untracked=(
            (list(fix_leftover_untracked) or None) if runs_fix and fix_leftover_untracked is not None else None
        ),
        fix_quality_gate=(
            (dict(fix_quality_gate) or None) if runs_fix and fix_quality_gate is not None else None
        ),
        recommended_patch_capture=(
            recommended_capture
            if recommended_capture
            else ("pre_test" if runs_fix and recorder_provenance.run_flow is not DaydreamRunFlow.PR else None)
        ),
        profile_schema_version=profile.schema_version if profile is not None else None,
        profile_name=profile.name if profile is not None else None,
        profile_source_kind=profile.source_kind if profile is not None else None,
        profile_digest=profile.digest if profile is not None else None,
        source_path=source_path,
        remote_url=git_ctx.remote_url,
        repo_slug=git_ctx.repo_slug,
        branch=git_ctx.branch,
        base_branch=git_ctx.base_branch,
        head_sha=git_ctx.head_sha,
        base_sha=git_ctx.base_sha,
        changed_files=list(git_ctx.changed_files),
        pr_number=recorder_provenance.pr_number,
        pr_repo=recorder_provenance.pr_repo,
        total_cost_usd=(raw_cost or 0.0) if raw_cost is not None else None,
        total_prompt_tokens=final_metrics.get("total_prompt_tokens") or None,
        total_completion_tokens=final_metrics.get("total_completion_tokens") or None,
        total_cached_tokens=final_metrics.get("total_cached_tokens") or None,
        archive_path=str(archive_path),
    )

    # Prefer immutable lifecycle timing; evaluation fills only an unavailable span.
    if timing_summary is not None:
        m.wall_clock_seconds = timing_summary.wall_clock_seconds
        m.phase_timings = timing_summary.phase_timings
        m.timing_coverage = {
            "attributed_wall_clock_seconds": timing_summary.attributed_wall_clock_seconds,
            "unattributed_wall_clock_seconds": timing_summary.unattributed_wall_clock_seconds,
            "coverage_ratio": timing_summary.coverage_ratio,
            "agent_completeness": timing_summary.agent_completeness,
            "diagnostics": timing_summary.diagnostics,
        }

    if evaluation:
        timing = evaluation.get("timing", {})
        eval_wall_clock = timing.get("total_wall_clock_seconds")
        if eval_wall_clock is not None and timing_summary is None:
            m.wall_clock_seconds = eval_wall_clock

        findings = evaluation.get("findings", {})
        m.total_findings = findings.get("total")
        # Uncomputed duplicate counts remain None, including legacy archives.
        shipped_duplication = findings.get("shipped_duplication", {})
        m.shipped_duplicate_pairs = shipped_duplication.get("near_duplicate_pairs")

        # No scorable location means undefined, preserving None for reward
        # renormalization.
        location = evaluation.get("location", {})
        m.location_in_hunk_rate = location.get("in_hunk_rate")

        quality = evaluation.get("quality", {})
        m.erosion = quality.get("erosion")
        m.verbosity = quality.get("verbosity")

        derived = evaluation.get("derived", {})
        m.cost_per_finding_usd = derived.get("cost_per_finding_usd")

    return m
