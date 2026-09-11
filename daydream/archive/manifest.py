"""Run archive manifest builder.

Assembles a ``manifest.json`` from an immutable public run snapshot, git
context, and optional evaluation results. The manifest is the single source
of truth for what's in an archive bundle.

Provenance namespaces: ``git.*`` and ``code_context.*`` record provenance
of the repository under review (target ``base_sha``/``head_sha``); the
``daydream.*`` block records the immutable Daydream executable that
produced the run. The two must never be conflated.

Status split: ``status`` is a backward-compat alias of ``archive_status``
(archive finalization — was the run cleanly archived?); ``pipeline_status``
is the pipeline-outcome signal (succeeded/failed/partial/cancelled) — a run
that merged-failed and never tested is cleanly archived but its pipeline
failed.

Exports:
    MANIFEST_SCHEMA_VERSION: Current schema version string.
    Manifest: Dataclass representing the manifest.
    build_manifest_from_snapshot: Construct a Manifest from one frozen run
        snapshot and its run context.
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
)
from daydream.run_snapshot import ArchiveRunSnapshot
from daydream.trajectory import DaydreamRunFlow

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
    """Validate and retain archive identity from the exact frozen root bytes.

    The session id is a path segment in the archive, so an empty, dot, or
    separator-bearing value is refused rather than resolved.
    """
    session_id = write_snapshot.root_trajectory_id
    if not session_id or session_id in (".", "..") or set(session_id) & set("/\\\0"):
        raise ValueError("snapshot session_id is malformed")
    roots = [
        document for document in write_snapshot.documents if document.trajectory_id == session_id
    ]
    if len(roots) != 1:
        raise ValueError("frozen root trajectory is missing")
    try:
        payload = json.loads(roots[0].json_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen root trajectory is malformed") from exc
    if not isinstance(payload, dict) or (
        payload.get("session_id") != session_id or payload.get("trajectory_id") != session_id
    ):
        raise ValueError("frozen root trajectory identity is malformed")
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
    """Return only the fields whose values are truthy.

    Collapses the repeated ``**({k: v} if v else {})`` conditional-splat guard
    used for optional manifest fields: ``None`` (and any other falsy value)
    drops the key entirely, mirroring the old inline pattern exactly.
    """
    return {k: v for k, v in fields.items() if v}


@dataclass
class Manifest:
    """Archive bundle manifest written to ``manifest.json``.

    Attributes:
        schema_version: Manifest schema version for forward compatibility.
        recommended_patch_supported: Provenance flag read by training signals.
            ``True`` for every manifest written by recommended.patch-aware
            daydream. When ``True``, a missing ``recommended.patch`` means the
            run made no recommendation (review-only / all-declined / wash), not
            a legacy archive — so ``_read_recommended_patch`` returns ``""``
            instead of falling back to ``diff.patch``. Legacy manifests omit the
            key.
        session_id: UUID4 session identifier from the trajectory recorder.
        archived_at: ISO 8601 timestamp of when the archive was created.
        status: Run status — ``complete`` alias of ``archive_status``, kept
            byte-identical for backward compatibility (archive finalization),
            never conflated with ``pipeline_status``.
        run_flow: Run flow type (normal, ttt, pr, deep).
        skill: Review skill used (python, react, etc.).
        model: Model name (opus, sonnet, haiku).
        backend: Phase-agnostic general default backend (claude, codex) resolved
            from config, never a per-phase override.
        review_backend: Review-specific backend override marker, ``None`` when
            review ran on the general default. NOT the effective review
            backend: a CLI ``--backend`` masks a file-config review override,
            so review may have run on ``backend`` even when this is set.
        fix_backend: Effective backend for the fix phase (override or general
            default), or ``None`` for flows whose executed step pipeline has no
            fix phase (improve, custom flows without fix, and TTT whose fix/test
            steps are gated off at runtime; #648).
        test_backend: Effective backend for the test phase (override or general
            default), or ``None`` for flows whose executed step pipeline has no
            test phase (improve, custom flows without test, and TTT/PR which
            never run the test step; #648).
        per_stack_review_backend: Per-stack review tier backend for runs that
            execute per-stack reviews (issue #646), resolved from the
            ``per_stack_review`` phase key — the key that actually drives
            per-stack execution — kept distinct from ``review_backend`` so
            "who reviewed" is never a misstatement. Every deep-flow mode
            executes per-stack reviews — loop, shallow (single collapsed
            stack), review, and comment; only improve/custom flows (which
            never invoke the deep orchestrator) have no per-stack fan-out and
            leave this ``None`` (omitted from ``to_dict()``).
        per_stack_review_model: Per-stack review tier model for runs that
            execute per-stack reviews (issue #646), resolved from the
            ``per_stack_review`` phase key. The model is the load-bearing part
            of the identity: per-stack defaults to Sonnet vs the ``review``
            tier's Opus, which a pure backend name cannot distinguish. ``None``
            (and omitted from ``to_dict()``) only for runs that never execute
            per-stack reviews (improve / custom flows). For a Pi run
            with no explicit override the backend default (``DEFAULT_PI_MODEL``)
            is recorded: Pi's default intentionally lives outside
            ``PHASE_DEFAULT_MODELS``, so ``_resolved_model`` alone would leave
            the load-bearing model NULL.
        review_only: Whether the run was review-only.
        deep: Whether deep review mode was used.
        source_path: Absolute path to the source repository at archive time.
        remote_url: Git remote origin URL.
        repo_slug: ``owner/repo`` extracted from remote URL.
        branch: Git branch name at run time.
        base_branch: Default branch (main/master).
        head_sha: Git HEAD commit SHA.
        base_sha: Merge-base SHA between ``base_branch`` and HEAD at archive
            time. ``None`` when no merge-base could be resolved.
        changed_files: Repo-relative paths changed between ``base_sha`` and
            ``head_sha``. Empty list when ``base_sha`` is ``None``.
        pr_number: GitHub PR number if applicable.
        pr_repo: GitHub repo slug for PR.
        total_cost_usd: Total cost from trajectory final metrics.
        total_prompt_tokens: Non-cached prompt tokens.
        total_completion_tokens: Completion tokens.
        total_cached_tokens: Cached tokens.
        wall_clock_seconds: Wall-clock duration derived from step timestamps
            on every run; refined by eval's fork-inclusive value when available.
        phase_timings: Per-phase wall-clock breakdown derived from explicit
            ``phase_start``/``phase_end`` events (issue #203). ``None`` when no
            phase events were emitted (pre-#203 runs or runs that skip phase
            wrapping). Each entry: ``{"wall_clock_seconds": float, "occurrences": int}``.
        fix_failures: Map of file-group -> failure reason for fix groups that
            were dropped (``phase_fix_parallel`` raised). ``None`` when every
            fix applied. When populated, ``status`` is forced to ``partial``
            because the working tree holds reverted/unapplied edits and must not
            be presented as a clean ``complete`` run.
        fix_leftover_untracked: Sorted list of untracked paths that appeared
            during a failed fix pass and survived tree-protection. Because
            parallel groups share one working tree these cannot be attributed to
            a specific group, so they are recorded (never deleted) to make the
            partial run fully auditable. ``None`` when none were left behind.
        fix_quality_gate: The fix-phase anti-degradation quality-gate verdict
            (issue #315): ``{"enabled": bool, "rounds": [...]}`` written to
            ``deep/fix-quality-gate.json`` by the orchestrator, covering
            per-file before/after erosion + verbosity deltas over the files the
            fix phase edited. ``None`` when the gate artifact is absent or
            malformed.
        total_findings: Number of findings (from eval, if available).
        grounding_rate: Grounding rate (from eval, if available).
        coverage_ratio: File coverage ratio (from eval, if available).
        cost_per_finding_usd: Cost per finding (from eval, if available).
        erosion: Structural erosion ratio of the post-fix workspace (from eval,
            if available).
        verbosity: Line-flagging verbosity ratio of the post-fix workspace
            (from eval, if available).
        location_in_hunk_rate: Share of scored shipped findings whose
            originally cited line -- ``location_cited_line`` when the
            validator snapped/demoted it, else ``line``; never the validator's
            post-snap position, see ``eval.analyzer._cited_line`` -- landed
            inside a diff hunk, i.e. the location validator's headline
            accuracy axis (from eval, if available). ``None`` when the
            run scored no locatable findings — undefined, never 0.0, so the
            reward pipeline renormalizes over present axes instead of reading
            an imputed perfect/zero score.
        shipped_duplicate_pairs: Number of near-duplicate finding pairs
            (similarity >= 0.5) that survived dedup into the shipped set (from
            eval, if available) — the escaped-duplication axis. ``None`` when
            the eval pass did not compute it.
        outcome_labels: JSON-encoded list of outcome labels.
        labeled_at: ISO 8601 timestamp of last label update.
        composite_reward: Cached composite reward scalar mirrored from the
            latest ``label_observations`` annotation; ``None`` until a
            ``harvest`` pass scores the run.
        archive_path: Absolute path to the archive directory.
    """

    schema_version: str = MANIFEST_SCHEMA_VERSION
    # Provenance flag consumed by labeler_signals._read_recommended_patch:
    # when True, a missing recommended.patch means "no recommendation"
    # (review-only / all-declined / wash), NOT a legacy archive, so the
    # diff.patch fallback must not fire. Defaults True for every new manifest;
    # legacy manifests simply omit the key.
    recommended_patch_supported: bool = True
    # Provenance flag recording which capture produced the archived
    # recommended.patch (issue #743): ``"post_test"`` = the post-heal
    # re-capture (the exact tree committed), ``"pre_test"`` = the fix-phase
    # fallback capture. ``None`` on legacy manifests (which predate the field).
    recommended_patch_capture: str | None = None
    session_id: str = ""
    archived_at: str = ""
    status: str = "complete"
    # archive_status is byte-identical to the legacy ``status`` alias (spec Key
    # Decision 1): archive finalization, distinct from pipeline_status. Kept as a
    # separate key so consumers distinguishing "cleanly archived" from "pipeline
    # succeeded" do not repurpose the legacy field.
    archive_status: str = "complete"
    # pipeline_status is the pipeline-outcome signal: succeeded / failed /
    # partial / cancelled / unknown. Distinct from archive_status: a run that
    # merged-failed and never tested is cleanly archived but its pipeline
    # failed.
    pipeline_status: str = "unknown"
    # Per-phase terminal states (``merge``/``fix``/``test``), each
    # ``{"ran": bool, "status": str}`` where status is one of
    # succeeded/failed/partial/absent/unknown. ``None``/omitted for legacy
    # manifests.
    phase_states: dict[str, Any] | None = None
    # Executable provenance: the immutable Daydream executable that produced
    # this run (vendor ``ExecutableProvenance``). Never merged into the
    # target-repo ``git.*`` / ``code_context.*`` blocks.
    daydream: Any | None = None

    # Review-profile provenance (issue #885, R12): the resolved profile this
    # run executed under (schema version, name, source kind, canonical digest).
    # ``None`` on legacy manifests (which predate the fields) and omitted
    # entirely from ``to_dict()``; required on new runs.
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

    # wall_clock_seconds and phase_timings are derived from step/phase events
    # on every run; the remaining metrics below are populated by the eval pass,
    # which runs by default (skipped only with --no-eval).
    wall_clock_seconds: float | None = None
    phase_timings: dict[str, Any] | None = None
    timing_coverage: dict[str, Any] | None = None
    total_findings: int | None = None
    grounding_rate: float | None = None
    coverage_ratio: float | None = None
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
        """Return a JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "recommended_patch_supported": self.recommended_patch_supported,
            **_omit_falsy(recommended_patch_capture=self.recommended_patch_capture),
            "session_id": self.session_id,
            "archived_at": self.archived_at,
            "status": self.status,
            "archive_status": self.archive_status,
            "pipeline_status": self.pipeline_status,
            **_omit_falsy(
                profile_schema_version=self.profile_schema_version,
                profile_name=self.profile_name,
                profile_source_kind=self.profile_source_kind,
                profile_digest=self.profile_digest,
            ),
            **_omit_falsy(
                daydream=self.daydream.to_dict() if self.daydream is not None else None,
                phase_states=self.phase_states,
            ),
            "run": {
                "flow": self.run_flow,
                "skill": self.skill,
                "model": self.model,
                "backend": self.backend,
                **_omit_falsy(
                    review_backend=self.review_backend,
                    fix_backend=self.fix_backend,
                    test_backend=self.test_backend,
                    per_stack_review_backend=self.per_stack_review_backend,
                    per_stack_review_model=self.per_stack_review_model,
                ),
                "review_only": self.review_only,
                "deep": self.deep,
            },
            "fix_failures": self.fix_failures,
            "fix_leftover_untracked": self.fix_leftover_untracked,
            "fix_quality_gate": self.fix_quality_gate,
            "git": {
                "source_path": self.source_path,
                "remote_url": self.remote_url,
                "repo_slug": self.repo_slug,
                "branch": self.branch,
                "base_branch": self.base_branch,
                "head_sha": self.head_sha,
            },
            "code_context": {
                "head_sha": self.head_sha,
                "base_branch": self.base_branch,
                "branch": self.branch,
                "base_sha": self.base_sha,
                "changed_files": list(self.changed_files),
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
                "phase_timings": self.phase_timings,
                **_omit_falsy(timing_coverage=self.timing_coverage),
                "total_findings": self.total_findings,
                "grounding_rate": self.grounding_rate,
                "coverage_ratio": self.coverage_ratio,
                "cost_per_finding_usd": self.cost_per_finding_usd,
                "erosion": self.erosion,
                "verbosity": self.verbosity,
                "location_in_hunk_rate": self.location_in_hunk_rate,
                "shipped_duplicate_pairs": self.shipped_duplicate_pairs,
            },
            "outcome": {
                "labels": json.loads(self.outcome_labels),
                "labeled_at": self.labeled_at,
                "composite_reward": self.composite_reward,
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
    provenance: ExecutableProvenance | None = None,
) -> Manifest:
    """Construct a manifest from captured identity and frozen trajectory bytes."""
    recorder_provenance = run.recorder_provenance
    write_snapshot = run.trajectories
    identity = run.identity
    if recorder_provenance.session_id != write_snapshot.root_trajectory_id:
        raise ValueError("archive provenance does not match frozen snapshot")
    from daydream.trajectory import compute_timing_summary, snapshot_trajectories

    frozen = snapshot_trajectories(write_snapshot)
    frozen_root = frozen.get("main")
    raw_final_metrics = frozen_root.get("final_metrics") if isinstance(frozen_root, dict) else None
    final_metrics: dict[str, Any] = raw_final_metrics if isinstance(raw_final_metrics, dict) else {}
    totals: dict[str, Any] = {
        "prompt": final_metrics.get("total_prompt_tokens") or 0,
        "completion": final_metrics.get("total_completion_tokens") or 0,
        "cached": final_metrics.get("total_cached_tokens") or 0,
        "cost": final_metrics.get("total_cost_usd") or 0.0,
        "any_cost_seen": final_metrics.get("total_cost_usd") is not None,
    }
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
        total_cost_usd=totals["cost"] if totals.get("any_cost_seen") else None,
        total_prompt_tokens=totals["prompt"] or None,
        total_completion_tokens=totals["completion"] or None,
        total_cached_tokens=totals["cached"] or None,
        archive_path=str(archive_path),
    )

    # Lifecycle timing comes from the immutable write snapshot. A snapshot the
    # reducer cannot span (no lifecycle stamps, or a cutoff it cannot match)
    # leaves the span unset for evaluation to fill below.
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
        # Escaped-duplication axis: near-duplicate pairs that survived dedup
        # into the shipped set. Absent on every run archived before the axis
        # existed, so the chained .get() must yield None (not 0).
        shipped_duplication = findings.get("shipped_duplication", {})
        m.shipped_duplicate_pairs = shipped_duplication.get("near_duplicate_pairs")

        # Location-validator accuracy axis. ``in_hunk_rate`` is deliberately
        # None when no finding was scorable; that None is preserved as
        # "undefined" rather than coerced, same as every other eval metric.
        location = evaluation.get("location", {})
        m.location_in_hunk_rate = location.get("in_hunk_rate")

        grounding = evaluation.get("grounding", {})
        m.grounding_rate = grounding.get("grounding_rate")

        coverage = evaluation.get("coverage", {})
        m.coverage_ratio = coverage.get("coverage_ratio")

        quality = evaluation.get("quality", {})
        m.erosion = quality.get("erosion")
        m.verbosity = quality.get("verbosity")

        derived = evaluation.get("derived", {})
        m.cost_per_finding_usd = derived.get("cost_per_finding_usd")

    return m
