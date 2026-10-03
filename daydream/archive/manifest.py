"""Build archive manifests from immutable run snapshots.

``git`` and ``code_context`` describe the reviewed repository; ``daydream``
describes the executable. ``status`` aliases archive finalization, while
``pipeline_status`` records the independent workflow outcome.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
    roots = [document for document in write_snapshot.documents if document.trajectory_id == session_id]
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
) -> dict[str, Any]:
    """Project captured identity and frozen trajectory bytes directly into archive wire metadata."""
    recorder = run.recorder_provenance
    snapshot = run.trajectories
    identity = run.identity
    if recorder.session_id != snapshot.root_trajectory_id:
        raise ValueError("archive provenance does not match frozen snapshot")
    frozen_root = snapshot_trajectories(snapshot).get("main")
    raw_metrics = frozen_root.get("final_metrics") if isinstance(frozen_root, dict) else None
    final_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    raw_cost = final_metrics.get("total_cost_usd")
    timing_summary = compute_timing_summary(snapshot)
    metrics = {
        "total_cost_usd": (raw_cost or 0.0) if raw_cost is not None else None,
        **{
            name: final_metrics.get(name) or None
            for name in (
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_cached_tokens",
            )
        },
        "wall_clock_seconds": None,
        "phase_timings": None,
    }
    # Immutable lifecycle timing wins even at zero; evaluation fills only an unavailable span.
    if timing_summary is not None:
        metrics.update(
            wall_clock_seconds=timing_summary.wall_clock_seconds,
            phase_timings=timing_summary.phase_timings,
            timing_coverage={
                "attributed_wall_clock_seconds": timing_summary.attributed_wall_clock_seconds,
                "unattributed_wall_clock_seconds": timing_summary.unattributed_wall_clock_seconds,
                "coverage_ratio": timing_summary.coverage_ratio,
                "agent_completeness": timing_summary.agent_completeness,
                "diagnostics": timing_summary.diagnostics,
            },
        )
    timing = evaluation.get("timing", {}) if evaluation else {}
    if timing.get("total_wall_clock_seconds") is not None and timing_summary is None:
        metrics["wall_clock_seconds"] = timing["total_wall_clock_seconds"]
    findings = evaluation.get("findings", {}) if evaluation else {}
    quality = evaluation.get("quality", {}) if evaluation else {}
    metrics.update(
        total_findings=findings.get("total"),
        cost_per_finding_usd=(evaluation.get("derived", {}) if evaluation else {}).get("cost_per_finding_usd"),
        erosion=quality.get("erosion"),
        verbosity=quality.get("verbosity"),
        location_in_hunk_rate=(evaluation.get("location", {}) if evaluation else {}).get("in_hunk_rate"),
        shipped_duplicate_pairs=findings.get("shipped_duplication", {}).get("near_duplicate_pairs"),
    )
    runs_fix = identity.phases.fix
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        # Missing recommended.patch means no recommendation for new captures; legacy
        # readers still honor absence of this marker and use their diff.patch fallback.
        "recommended_patch_supported": True,
        **_omit_falsy(
            recommended_patch_capture=(
                recommended_capture
                or ("pre_test" if runs_fix and recorder.run_flow is not DaydreamRunFlow.PR else None)
            )
        ),
        "session_id": recorder.session_id,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "archive_status": status,
        "pipeline_status": pipeline_status,
        **_omit_falsy(
            **{
                f"profile_{name}": getattr(identity.profile, name)
                for name in ("schema_version", "name", "source_kind", "digest")
            }
            if identity.profile is not None
            else {}
        ),
        **_omit_falsy(daydream=provenance.to_dict() if provenance is not None else None),
        **_omit_falsy(
            phase_states=dict(phase_states) if phase_states is not None else None,
            retry_summary=dict(retry_summary) if retry_summary is not None else None,
        ),
        "run": {
            "flow": recorder.run_flow.value,
            "skill": identity.skill,
            "model": identity.model,
            "backend": identity.backend,
            **_omit_falsy(
                **{
                    name: getattr(identity, name)
                    for name in (
                        "review_backend",
                        "fix_backend",
                        "test_backend",
                        "per_stack_review_backend",
                        "per_stack_review_model",
                    )
                }
            ),
            "review_only": identity.review_only,
            "deep": identity.deep,
        },
        "fix_failures": (dict(fix_failures) or None) if runs_fix and fix_failures is not None else None,
        "fix_leftover_untracked": (list(fix_leftover_untracked) or None)
        if runs_fix and fix_leftover_untracked is not None
        else None,
        "fix_quality_gate": (dict(fix_quality_gate) or None) if runs_fix and fix_quality_gate is not None else None,
        "git": {
            "source_path": source_path,
            **{
                name: getattr(git_ctx, name)
                for name in ("remote_url", "repo_slug", "branch", "base_branch", "head_sha")
            },
        },
        "code_context": {
            "head_sha": git_ctx.head_sha,
            "base_branch": git_ctx.base_branch,
            "branch": git_ctx.branch,
            "base_sha": git_ctx.base_sha,
            "changed_files": list(git_ctx.changed_files),
        },
        "pr": {"number": recorder.pr_number, "repo": recorder.pr_repo},
        "metrics": metrics,
        "outcome": {"labels": [], "labeled_at": None, "composite_reward": None},
        "archive_path": str(archive_path),
    }
