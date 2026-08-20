"""Run archive manifest builder.

Assembles a ``manifest.json`` from the recorder, run config, git context,
and optional evaluation results. The manifest is the single source of
truth for what's in an archive bundle.

Provenance namespaces: ``git.*`` and ``code_context.*`` record provenance
of the repository under review (target ``base_sha``/``head_sha``); the
``daydream.*`` block records the immutable Daydream executable that
produced the run. The two must never be conflated.

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
from daydream.config import DEFAULT_PI_MODEL
from daydream.extensions import UnresolvedExtensionError, get_registry
from daydream.trajectory import DaydreamRunFlow

if TYPE_CHECKING:
    from pathlib import Path

    from daydream.runner import RunConfig
    from daydream.trajectory import TrajectoryRecorder

MANIFEST_SCHEMA_VERSION = "1.0"


def _runtime_flow_name(flow: DaydreamRunFlow, flow_name: str | None) -> str | None:
    """Return the registered flow name ``run_flow`` resolved for this label.

    The deep family (NORMAL/DEEP/TTT/PR — four mode labels of the single
    registered ``deep`` flow, #330) always resolves to ``"deep"`` regardless
    of ``config.flow_name``; ``IMPROVE`` resolves ``"improve"``; ``CUSTOM`` is
    the literal ``--flow`` name. Builtins are seeded before the session's
    registry loads, so a fork registering a built-in name is resolved exactly
    as it runs (issue #648).
    """
    if flow is DaydreamRunFlow.IMPROVE:
        return "improve"
    if flow is DaydreamRunFlow.CUSTOM:
        return flow_name
    return "deep"


def _flow_phase_steps(flow_name: str | None) -> set[str]:
    """Return the set of phase steps in the registered flow's pipeline.

    Introspects the per-run registry (builtins are seeded before extension
    load) — the same source ``run_flow`` resolves — so a fork flow composing
    the built-in ``fix``/``test`` phases is detected exactly as it runs them.
    An unknown/absent flow yields an empty set: we record no backend rather
    than invent one for phases that never ran (#648).
    """
    if not flow_name:
        return set()
    try:
        entries = get_registry().flow(flow_name)
    except UnresolvedExtensionError:
        return set()
    step_names: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            step_names.add(entry)
        else:
            step_names.update(entry.steps)
    return step_names


def _flow_fix_test_steps(flow: DaydreamRunFlow, flow_name: str | None) -> tuple[bool, bool]:
    """Return ``(runs_fix, runs_test)`` for the pipeline ``run_flow`` executes.

    Issue #648 gates the manifest's fix/test backend labels on the step
    pipeline ``run_flow`` actually executes, resolved from the per-run registry
    for every label — not just ``CUSTOM`` — because ``Registry.set_flow`` has no
    built-in-name guard: a fork overriding ``deep`` or ``improve`` runs its own
    pipeline while the label would suggest the built-in. Two labels are fixed
    by runtime mode, not the registry:

    - Review/comment (``TTT``) stops after ``post-review`` (``_review_only_mode``)
      and never runs the fix cycle, so it records neither backend.
    - Feedback (``PR``) runs its own ``fix-items`` phase (``_step_fix_items``,
      ``backend_for("fix")``) but never the ``test`` step, so it records a fix
      backend only — matching the trajectory's ``_recorder_backend_names``.

    ``NORMAL``/``DEEP``/``IMPROVE``/``CUSTOM`` are classified by the registered
    pipeline (``NORMAL``/``DEEP`` → the ``deep`` flow; ``IMPROVE`` → ``improve``;
    ``CUSTOM`` → the literal ``--flow`` name).
    """
    if flow is DaydreamRunFlow.TTT:
        return False, False
    if flow is DaydreamRunFlow.PR:
        # Feedback runs the fix phase (fix-items) but never the test phase.
        return True, False
    steps = _flow_phase_steps(_runtime_flow_name(flow, flow_name))
    return ("fix" in steps, "test" in steps)


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
            stack), review, and comment; only feedback mode (the review spine
            is skipped entirely) and improve/custom flows (which never invoke
            the deep orchestrator) have no per-stack fan-out and leave this
            ``None`` (omitted from ``to_dict()``).
        per_stack_review_model: Per-stack review tier model for runs that
            execute per-stack reviews (issue #646), resolved from the
            ``per_stack_review`` phase key. The model is the load-bearing part
            of the identity: per-stack defaults to Sonnet vs the ``review``
            tier's Opus, which a pure backend name cannot distinguish. ``None``
            (and omitted from ``to_dict()``) only for runs that never execute
            per-stack reviews (feedback / improve / custom flows). For a Pi run
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "recommended_patch_supported": self.recommended_patch_supported,
            "session_id": self.session_id,
            "archived_at": self.archived_at,
            "status": self.status,
            "archive_status": self.archive_status,
            "pipeline_status": self.pipeline_status,
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
    source_path: str | None = None,
    cwd: str | None = None,
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
        cwd: The repository directory daydream operated on (``work.repo``), used
            to mirror PiBackend's cwd-configured default model resolution.
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
    from daydream.runner import (  # noqa: PLC0415 - deferred import avoids cycle
        _DEEP_FLOW_ALIASES,
        _default_backend_name,
        _resolved_backend_name,
        _resolved_model,
        _resolved_review_backend_name,
    )

    # ``backend`` records the phase-agnostic general default (config.backend →
    # file-config global → "claude"), never a per-phase override.
    # ``review_backend`` is an override marker, NOT an effective value: it is
    # stamped only when a review-specific override exists, and it can differ
    # from the backend review actually ran on (a CLI ``--backend`` masks a
    # file-config review override). Sibling fields ``fix_backend``/
    # ``test_backend`` are effective per-phase values; ``review_backend`` is
    # not.

    # Per-stack reviewers (issue #646) execute on the "per_stack_review" phase key —
    # NOT the "review" tier — so archives record that tier's resolved backend and
    # model from its own key. The per-stack-reviews step runs in every deep-flow
    # mode (loop, shallow, review, comment); shallow is NOT excluded — a collapsed
    # single stack is still reviewed through phase_per_stack_reviews. Only feedback
    # mode (config.bot set — the review spine is skipped entirely) and improve/custom
    # flows (which never invoke the deep orchestrator) have no per-stack fan-out, so
    # those runs leave both fields None (and to_dict() omits them). The deep-flow
    # alias set is runner._DEEP_FLOW_ALIASES — the same list _dispatch_selected_flow
    # routes — so the gate cannot drift from the actual flow routing.
    # start_at defaults to "review" on the real RunConfig; getattr keeps the
    # gate robust to lighter config fakes that omit the field.
    _start_at = getattr(config, "start_at", "review")
    per_stack_reviews_ran = (
        config.bot is None
        and (config.flow_name is None or config.flow_name in _DEEP_FLOW_ALIASES)
        and _start_at not in ("merge", "fix")
    )
    per_stack_review_backend: str | None = None
    per_stack_review_model: str | None = None
    if per_stack_reviews_ran:
        per_stack_review_backend = _resolved_backend_name(config, "per_stack_review")
        per_stack_review_model = _resolved_model(config, "per_stack_review")
        if per_stack_review_model is None and per_stack_review_backend == "pi":
            # Pi's default is a backend fallback (resolved by PiBackend from cwd)
            # that intentionally never appears in PHASE_DEFAULT_MODELS, so
            # _resolved_model returns None here even though the per-stack reviewers
            # ran on a concrete model. Mirror PiBackend's own precedence — a
            # cwd-configured default first, then DEFAULT_PI_MODEL — so the archived
            # identity matches what actually ran (#646 finding 1).
            from pathlib import Path

            from daydream.backends.pi import _configured_pi_model
            per_stack_review_model = (
                _configured_pi_model(Path(cwd)) if cwd else None
            ) or DEFAULT_PI_MODEL

    # Gate fix/test backend labels on whether this flow's step pipeline actually
    # includes those phases (issue #648): improve never reaches the fix/test
    # STEPS, TTT (review/comment) gates them off at runtime (_fix_cycle_enabled
    # is loop/shallow only), PR (feedback) runs its own fix-items phase but
    # never test, and custom flows are classified from their registered pipeline
    # (every ``--flow`` run is stamped CUSTOM regardless of step composition), so
    # a fork composing the built-in fix/test steps records backends like the deep
    # family. Registry-resolved for every label so fork overrides of built-in
    # ``deep``/``improve`` are classified by the pipeline actually executed.
    runs_fix, runs_test = _flow_fix_test_steps(recorder.run_flow, config.flow_name)

    m = Manifest(
        session_id=recorder.session_id,
        archived_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        run_flow=recorder.run_flow.value,
        skill=config.skill,
        model=None,
        backend=_default_backend_name(config),
        review_backend=_resolved_review_backend_name(config),
        fix_backend=_resolved_backend_name(config, "fix") if runs_fix else None,
        test_backend=_resolved_backend_name(config, "test") if runs_test else None,
        per_stack_review_backend=per_stack_review_backend,
        per_stack_review_model=per_stack_review_model,
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
