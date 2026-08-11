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

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from daydream.archive.git_context import GitContext
from daydream.config_file import DaydreamFileConfig

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
        recommended_patch_supported: Provenance flag read by training signals.
            ``True`` for every manifest written by recommended.patch-aware
            daydream. When ``True``, a missing ``recommended.patch`` means the
            run made no recommendation (review-only / all-declined / wash), not
            a legacy archive — so ``_read_recommended_patch`` returns ``""``
            instead of falling back to ``diff.patch``. Legacy manifests omit the
            key.
        session_id: UUID4 session identifier from the trajectory recorder.
        archived_at: ISO 8601 timestamp of when the archive was created.
        status: Run status — ``complete``, ``partial``, or ``failed``.
        run_flow: Run flow type (normal, ttt, pr, deep).
        skill: Review skill used (python, react, etc.).
        model: Model name (opus, sonnet, haiku).
        backend: Backend used (claude, codex).
        review_backend: Per-phase backend override for review, if set.
        fix_backend: Per-phase backend override for fix, if set.
        test_backend: Per-phase backend override for test, if set.
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
        outcome_labels: JSON-encoded list of outcome labels.
        labeled_at: ISO 8601 timestamp of last label update.
        composite_reward: Cached composite reward scalar mirrored from the
            latest ``label_observations`` annotation; ``None`` until a
            ``harvest`` pass scores the run.
        archive_path: Absolute path to the archive directory.
        provenance: Additive exact-provenance block recording ``backend``,
            ``model``, ``provider``, ``config``, ``skill``, and ``runtime``
            (Python/uv versions) EXACTLY as resolved — never raw config
            defaults. ``None`` only when the manifest was built by a consumer
            that never populated it.
    """

    schema_version: str = MANIFEST_SCHEMA_VERSION
    # Provenance flag consumed by labeler_signals._read_recommended_patch:
    # when True, a missing recommended.patch means "no recommendation"
    # (review-only / all-declined / wash), NOT a legacy archive, so the
    # diff.patch fallback must not fire. Defaults True for every new manifest;
    # legacy manifests simply omit the key.
    recommended_patch_supported: bool = True
    session_id: str = ""
    archived_at: str = ""
    status: str = "complete"

    # Run config
    run_flow: str = ""
    skill: str | None = None
    model: str | None = None
    backend: str = "claude"
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
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
    total_findings: int | None = None
    grounding_rate: float | None = None
    coverage_ratio: float | None = None
    cost_per_finding_usd: float | None = None
    erosion: float | None = None
    verbosity: float | None = None

    # Outcome labels (populated via `daydream harvest`)
    outcome_labels: str = field(default="[]")
    labeled_at: str | None = None
    composite_reward: float | None = None

    # Archive location
    archive_path: str = ""

    # Additive exact-provenance block (Plan 008 Step 2 leaf). Populated by
    # build_manifest; a pre-provenance Manifest carries None.
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "recommended_patch_supported": self.recommended_patch_supported,
            "session_id": self.session_id,
            "archived_at": self.archived_at,
            "status": self.status,
            "run": {
                "flow": self.run_flow,
                "skill": self.skill,
                "model": self.model,
                "backend": self.backend,
                **({"review_backend": self.review_backend} if self.review_backend else {}),
                **({"fix_backend": self.fix_backend} if self.fix_backend else {}),
                **({"test_backend": self.test_backend} if self.test_backend else {}),
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
                "total_findings": self.total_findings,
                "grounding_rate": self.grounding_rate,
                "coverage_ratio": self.coverage_ratio,
                "cost_per_finding_usd": self.cost_per_finding_usd,
                "erosion": self.erosion,
                "verbosity": self.verbosity,
            },
            "outcome": {
                "labels": json.loads(self.outcome_labels),
                "labeled_at": self.labeled_at,
                "composite_reward": self.composite_reward,
            },
            "provenance": self.provenance,
            "archive_path": self.archive_path,
        }


def _uv_version() -> str | None:
    """Resolve the ``uv`` CLI version locally, or None when unavailable.

    Local subprocess only (no network). ``uv --version`` is the canonical
    runtime identity of the environment that produced the run.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - args are a module-local constant
            ["uv", "--version"],  # noqa: S607 - uv is a trusted command
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (proc.stdout.strip() or proc.stderr.strip()) or None


def _config_provenance(config: "RunConfig") -> dict[str, str] | None:
    """Record a content digest of the effective file-config overrides.

    Captures exactly what the file config resolved (global model/backend/
    reasoning effort plus every phase table) — not raw parser defaults. An
    absent or empty file config records ``None``.
    """
    file_config = config.file_config if config.file_config is not None else DaydreamFileConfig()
    effective: dict[str, Any] = {}
    for key, value in (
        ("model", file_config.model),
        ("backend", file_config.backend),
        ("reasoning_effort", file_config.reasoning_effort),
        ("phases", file_config.phases),
    ):
        if value:
            effective[key] = value
    if not effective:
        return None
    payload = json.dumps(effective, sort_keys=True, default=str)
    return {"digest": hashlib.sha256(payload.encode()).hexdigest()}


def _resolved_provider(backend_name: str) -> str | None:
    """Resolve the provider exactly as the Pi backend would at execute time.

    ``claude``/``codex`` use their native endpoints and have no provider axis;
    only ``pi`` resolves one: ``PI_PROVIDER`` wins, and when it is unset the
    provider falls back to Pi's default. That holds both for an explicitly
    resolved model (a CLI/file-config override, so pi is told to pass
    ``--provider``) and for the default configuration where neither daydream
    nor Pi's settings resolve a model — ``PiBackend.execute`` then runs
    ``DEFAULT_PI_MODEL`` against the default provider. The manifest cannot
    observe Pi's own settings resolution, so the provider is independent of
    the daydream-resolved model.
    """
    if backend_name != "pi":
        return None
    provider = os.environ.get("PI_PROVIDER")
    if provider is not None:
        return provider
    from daydream.backends.pi import _PI_DEFAULT_PROVIDER

    return _PI_DEFAULT_PROVIDER


def _build_provenance(config: "RunConfig") -> dict[str, Any]:
    """Resolve the exact backend/model/provider/config/skill/runtime stack.

    Every value is the *resolved* one (through the full precedence chain),
    never a raw config default. Guarded against partial/mock config objects
    so manifest consumers that stand in a minimal config still get a
    provenance block with the resolvable fields.
    """
    from daydream.runner import _resolved_backend_name, _resolved_model

    try:
        backend = _resolved_backend_name(config, "review")
    except (AttributeError, TypeError):
        backend = None
    try:
        model = _resolved_model(config, "review")
    except (AttributeError, TypeError):
        model = None
    try:
        config_provenance = _config_provenance(config)
    except (AttributeError, TypeError):
        config_provenance = None
    return {
        "backend": backend,
        "model": model,
        "provider": _resolved_provider(backend) if backend is not None else None,
        "config": config_provenance,
        "skill": getattr(config, "skill", None),
        "runtime": {
            "python": platform.python_version(),
            "uv": _uv_version(),
        },
    }


def build_manifest(
    *,
    recorder: TrajectoryRecorder,
    config: RunConfig,
    git_ctx: GitContext,
    status: str,
    archive_path: Path,
    evaluation: dict[str, Any] | None = None,
    source_path: str | None = None,
    fix_failures: dict[str, str] | None = None,
    fix_leftover_untracked: list[str] | None = None,
    fix_quality_gate: dict[str, Any] | None = None,
) -> Manifest:
    """Construct a Manifest from run context.

    Args:
        recorder: The TrajectoryRecorder that produced the trajectory.
        config: The RunConfig for this run.
        git_ctx: Captured git metadata.
        status: Run status (``complete``, ``partial``, ``failed``).
        archive_path: Absolute path to the archive directory for this run.
        evaluation: Optional ``analyze_session()`` result dict.
        source_path: Absolute path to the source repository at archive time.
        fix_failures: Map of dropped fix file-group -> reason, or ``None`` when
            every fix applied. Recorded verbatim on the manifest.
        fix_leftover_untracked: Sorted list of untracked paths left behind by a
            failed fix pass, or ``None``. Recorded verbatim on the manifest.
        fix_quality_gate: The fix-phase anti-degradation quality-gate verdict
            (issue #315), or ``None`` when the artifact is absent. Recorded
            verbatim on the manifest.

    Returns:
        A fully populated Manifest.
    """
    totals = recorder._final_totals  # noqa: SLF001 - intentional access to recorder internals

    # Deferred import breaks the module-level cycle: archive.manifest → runner → (lazy) archive.
    from daydream.runner import _resolved_backend_name  # noqa: PLC0415 - deferred import avoids cycle

    # Resolve the effective backend through the full precedence chain so the manifest
    # records what was actually used (raw config.backend is None when set via file-config);
    # "review" is the representative phase the orchestrator also prints as the default.
    backend_used = _resolved_backend_name(config, "review")

    m = Manifest(
        session_id=recorder.session_id,
        archived_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        run_flow=recorder.run_flow.value,
        skill=config.skill,
        model=None,
        backend=backend_used,
        review_backend=backend_used,
        fix_backend=_resolved_backend_name(config, "fix"),
        test_backend=_resolved_backend_name(config, "test"),
        review_only=config.output_mode == "review",
        deep=not config.shallow,
        fix_failures=fix_failures or None,
        fix_leftover_untracked=fix_leftover_untracked or None,
        fix_quality_gate=fix_quality_gate or None,
        source_path=source_path,
        remote_url=git_ctx.remote_url,
        repo_slug=git_ctx.repo_slug,
        branch=git_ctx.branch,
        base_branch=git_ctx.base_branch,
        head_sha=git_ctx.head_sha,
        base_sha=git_ctx.base_sha,
        changed_files=list(git_ctx.changed_files),
        pr_number=recorder.pr_number,
        pr_repo=recorder.pr_repo,
        total_cost_usd=totals["cost"] if totals.get("any_cost_seen") else None,
        total_prompt_tokens=totals["prompt"] or None,
        total_completion_tokens=totals["completion"] or None,
        total_cached_tokens=totals["cached"] or None,
        archive_path=str(archive_path),
    )

    # Derivable from step timestamps, so populated for every run; the eval pass's
    # fork-inclusive value (eval.analyzer.analyze_timing) takes precedence below.
    m.wall_clock_seconds = recorder.compute_wall_clock_seconds()
    # Per-phase breakdown from explicit phase_start/phase_end events (#203).
    m.phase_timings = recorder.compute_phase_timings()

    # Additive exact-provenance block (Plan 008 Step 2 leaf).
    m.provenance = _build_provenance(config)

    if evaluation:
        timing = evaluation.get("timing", {})
        eval_wall_clock = timing.get("total_wall_clock_seconds")
        if eval_wall_clock is not None:
            m.wall_clock_seconds = eval_wall_clock

        findings = evaluation.get("findings", {})
        m.total_findings = findings.get("total")

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
