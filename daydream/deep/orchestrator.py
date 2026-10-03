"""Compose deep and diagram flows, their preambles, and public entry points."""

from __future__ import annotations

import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markup import escape as escape_markup

from daydream import git_ops
from daydream.agent import console
from daydream.artifact_visibility import ArtifactVisibilityError, artifact_dir_for, artifact_session_active
from daydream.config import (
    DEFAULT_DEEP_SHARD_ENABLED,
    DEFAULT_DEEP_SHARD_FANOUT_CAP,
    DEFAULT_DEEP_SHARD_FRONTIER_MAX,
    DEFAULT_DEEP_SHARD_MAX_BYTES,
    DEFAULT_DEEP_SHARD_MAX_FILES,
    REVIEW_OUTPUT_FILE,
    STRUCTURE_STACK_NAME,
)
from daydream.deep import review_steps
from daydream.deep.adjudication_steps import _step_arbiter
from daydream.deep.artifacts import (
    DeepArtifact,
    check_deep_artifacts,
    deep_dir,
    diff_key,
    per_stack_records_path,
)
from daydream.deep.dependency import build_import_graph
from daydream.deep.detection import GENERIC_STACK, StackAssignment, detect_stacks
from daydream.deep.diagram_steps import _diagram_mode_for, _resolved_diagram_mode, _step_diagram, _step_post_diagram
from daydream.deep.diff import _diff_changed_files, bound_deep_diff
from daydream.deep.fix_steps import (
    _perform_cleanup,
    _step_commit,
    _step_fix,
    _step_fix_gate,
    _step_fix_verify,
    _step_test,
    _step_verify,
)
from daydream.deep.latency import diff_signals, route_for, summarize_risk
from daydream.deep.merge_steps import (
    _step_cross_stack_merge,
    _step_findings_out,
    _step_load_items,
    _step_post_review,
    _step_single_stack_merge,
    _step_supervise,
    _supervisor_mode,
)
from daydream.deep.remote_ci_steps import _step_remote_ci
from daydream.deep.render import _PIPELINE_STAGE_NAMES
from daydream.deep.reuse_store import build_reuse_cache
from daydream.deep.review_steps import (
    _step_exploration,
    _step_intent,
    _step_per_stack_parse,
    _step_wonder_and_per_stack,
)
from daydream.deep.routing_record import write_routing_record
from daydream.deep.settings import (
    _resolve_config_value,
    _resolve_non_negative_int,
    fold_default_alternatives,
    fresh_ttt,
)
from daydream.deep.sharding import shard_stacks
from daydream.deep.state import DeepState
from daydream.extensions import get_registry
from daydream.extensions.api import FlowStep
from daydream.flows.engine import BackendFactory, FlowContext, run_flow
from daydream.github_app import GitHubExecutionInput
from daydream.phases import PushReceipt
from daydream.review_budget import review_deadline_scope, review_scale_for_diff
from daydream.review_profile import Pipeline, build_default_profile, resolve_pipeline
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.test_execution import (
    RecipeConfinementError,
    persist_test_recipe,
    resolve_test_recipe,
)
from daydream.trajectory import DaydreamRunFlow
from daydream.ui import print_error, print_info, print_preflight_notice, print_warning
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.pr_review import PRInfo
    from daydream.run_artifacts import _RunArtifacts
    from daydream.run_config import RunConfig


def total_agent_count(stack_count: int) -> int:
    """Estimate two TTT calls, review/parse per stack, merge and conditional arbiter.

    Always budget the arbiter in preflight; user-gated fix agents are excluded.
    """
    return 2 + stack_count + stack_count + 1 + 1


# Tiny diffs collapse language fan-out and bypass merge/arbiter.
DEFAULT_SHALLOW_FANOUT_THRESHOLD = 2


def _single_stack_agent_count(stack_count: int) -> int:
    """Estimate two TTT calls plus review/parse per surviving stack, without merge or arbiter."""
    return 2 + stack_count + stack_count


def _config_pipeline(config: RunConfig) -> Pipeline:
    """Resolve the profile pipeline before a FlowContext exists."""
    return resolve_pipeline(config.review_profile)


def _shallow_fanout_threshold(config: RunConfig) -> int:
    """Resolve the tiny-diff short-circuit threshold (issue #172, AC7).

    ``0`` disables the short-circuit.
    """
    return _resolve_config_value(config, "shallow_fanout_threshold", DEFAULT_SHALLOW_FANOUT_THRESHOLD)


def _supervise_enabled(ctx: FlowContext) -> bool:
    """Run supervision on fresh flows, not on a fix-only resume."""
    return _supervisor_mode(ctx.config) in {"rules", "llm"} and ctx.config.start_at != "fix"


def _deep_shard_enabled(config: RunConfig, *, diff: str = "") -> bool:
    """Resolve CLI-over-file sharding; absent overrides enable the default for large diffs."""
    explicit = config.deep_shard_enabled
    if explicit is None and config.file_config is not None:
        explicit = config.file_config.deep_shard_enabled
    if explicit is not None:
        return explicit
    return DEFAULT_DEEP_SHARD_ENABLED or review_scale_for_diff(diff) >= 4


def _deep_shard_max_files(config: RunConfig) -> int:
    """Resolve the per-shard max file-count bound (issue #731)."""
    return _resolve_non_negative_int(config, "deep_shard_max_files", DEFAULT_DEEP_SHARD_MAX_FILES)



def _collapse_stacks_for_tiny_diff(
    stacks: list[StackAssignment],
    changed_files: list[str],
    *,
    threshold: int,
) -> tuple[list[StackAssignment], bool]:
    """Collapse language fan-out when 0 < changed-file count <= threshold.

    Keep structure separate. One real language absorbs generic/docs files; multiple
    real languages collapse to generic. A sole non-structural assignment is unchanged.
    Return the stacks and whether the gate is active; threshold 0 disables it.
    """
    if not (0 < len(changed_files) <= threshold):
        return stacks, False

    non_structural = [s for s in stacks if s.stack_name != STRUCTURE_STACK_NAME]
    structural = [s for s in stacks if s.stack_name == STRUCTURE_STACK_NAME]

    # Absorb generic files into a sole language; multiple languages collapse to generic.
    if len(non_structural) >= 2:
        combined_files = sorted({f for s in non_structural for f in s.files})
        real_language = [s for s in non_structural if s.stack_name != GENERIC_STACK]
        stack_name = real_language[0].stack_name if len(real_language) == 1 else GENERIC_STACK
        combined = StackAssignment(
            stack_name=stack_name,
            files=combined_files,
            is_docs_only=False,
        )
        return [*structural, combined], True

    # 0 or 1 non-structural stacks: nothing to collapse (lever 1 is a no-op), but
    # the gate is still active so the caller applies lever 2 (skip merge+arbiter).
    return stacks, True


def _stack_preflight_line(stack: StackAssignment) -> str:
    """Format one detected-stack line for the pre-flight notice (skill-free, M2)."""
    docs_suffix = " (docs-only)" if stack.is_docs_only else ""
    return f"{stack.stack_name}: {len(stack.files)} file(s){docs_suffix}"


def _preflight_stage_names(stacks: list[StackAssignment], *, folded_alternatives: bool = False) -> list[str]:
    """Return user-facing stages, including the structural review when active."""
    stages = list(_PIPELINE_STAGE_NAMES)
    if folded_alternatives:
        stages[1] = "design alternatives (included in structural review)"
    if any(stack.stack_name == STRUCTURE_STACK_NAME for stack in stacks):
        stages.insert(3, "structural review (parallel with per-stack reviews)")
    return stages




def _has_non_daydream_worktree_changes(status: str) -> bool:
    """Whether porcelain output names a path outside Daydream-owned artifacts."""
    for line in status.splitlines():
        paths = line[3:].split(" -> ")
        if not all(
            path == ".daydream"
            or path.startswith(".daydream/")
            or path == REVIEW_OUTPUT_FILE
            for path in paths
        ):
            return True
    return False


def _print_findings_target_mismatch(analyzed_head: str, pr_head: str | None) -> None:
    """Report a findings export target that is not the analyzed checkout.

    Shared by every mode that writes a findings artifact, so review and diagram
    runs reject a moved head with the same sentence: both commits, and what the
    operator can do next.
    """
    if pr_head is None:
        detail = "no pull request matches the analyzed checkout"
    else:
        detail = f"analyzed {analyzed_head[:12]}, PR now points at {pr_head[:12]}"
    print_error(console, "Findings Artifact",
                f"Target PR does not match the analyzed checkout ({detail}); "
                "re-run against the new head, or check out and review the superseded commit deliberately.")


def _capture_findings_target(
    target_dir: str,
    config: RunConfig,
    github_execution: GitHubExecutionInput,
    captured_head: str,
    captured_base: str,
    *,
    capture_base_tip: bool,
) -> tuple[PRInfo, str | None] | None:
    """Resolve and verify the PR that owns a commit-bound findings export.

    One trust gate for every mode that writes a findings artifact: resolve the
    PR, reject a head that moved, and rebind the base to the captured merge
    base. Returns ``(captured_pr, pr_base_sha)``, or ``None`` when the resolved
    PR does not match the captured head (already reported). Raises ``GitError``
    when the lookup itself fails, so the caller can name that failure instead of
    reporting it as a diff/base-branch problem.

    ``capture_base_tip`` is off for diagram runs: they carry no coverage record,
    so the base tip has no consumer.
    """
    from daydream.pr_review import capture_pr_base_tip, find_open_pr, find_pr_by_number

    captured_pr = (find_pr_by_number(target_dir, config.pr_number, auth=github_execution.auth)
                   if config.pr_number is not None else find_open_pr(target_dir, auth=github_execution.auth))
    if captured_pr is None or captured_pr.head_sha != captured_head:
        _print_findings_target_mismatch(captured_head, None if captured_pr is None else captured_pr.head_sha)
        return None
    pr_base_sha = capture_pr_base_tip(target_dir, captured_pr, auth=github_execution.auth) if capture_base_tip else None
    return replace(captured_pr, base_sha=captured_base), pr_base_sha


def _remote_ci_enabled(ctx: FlowContext) -> bool:
    """Run remote verification only after this flow recorded a successful push."""
    deep_state = DeepState(ctx.data)
    return isinstance(deep_state.push_receipt, PushReceipt)


def _fresh_ttt(ctx: FlowContext) -> bool:
    # Verbatim resume gate: --start-at per-stack/merge/fix skips the TTT phases.
    return fresh_ttt(ctx.config)


def _before_fix_resume(ctx: FlowContext) -> bool:
    # Verbatim resume gate: --start-at fix skips everything up to the fix gate.
    # For post-review, this also keeps the non-idempotent GitHub write off the
    # resume path (duplicate inline reviews on reruns).
    return ctx.config.start_at != "fix"


def _multi_stack_merge_enabled(ctx: FlowContext) -> bool:
    deep_state = DeepState(ctx.data)
    return ctx.config.start_at != "fix" and not deep_state.single_stack_mode


def _single_stack_merge_enabled(ctx: FlowContext) -> bool:
    deep_state = DeepState(ctx.data)
    return ctx.config.start_at != "fix" and deep_state.single_stack_mode


def _diagram_enabled(ctx: FlowContext) -> bool:
    """Run diagram eligibility checks unless mode is off or this is a fix-only resume.

    No eligible diagram means zero agent calls, but the step still records why.
    Diagram-only flows always pass this gate.
    """
    return _before_fix_resume(ctx) and _resolved_diagram_mode(ctx) != "off"


def _findings_out_enabled(ctx: FlowContext) -> bool:
    # Two-phase findings artifact (Phase A): emit the artifact and STOP —
    # never post to the PR and never apply fixes. Phase B posts later.
    return ctx.config.findings_out is not None


def _resolve_mode(config: RunConfig) -> str:
    """Map a RunConfig onto the single deep-flow mode key (#330).

    ``review`` / ``comment`` replace the review flow (stop after post-review);
    ``shallow`` replaces the shallow flow (single-stack deep); ``diagram``
    (issue #1113) reuses the spine's preamble but runs the two-step ``diagram``
    flow. ``loop`` is the unchanged default.
    """
    if config.flow_name in ("review", "shallow"):
        return config.flow_name
    # Issue #1113: checked before ``shallow`` so ``--diagram-only`` is never
    # reinterpreted as a shallow review by an unrelated flag combination.
    if config.output_mode in ("diagram", "review", "comment"):
        return config.output_mode
    if config.shallow:
        return "shallow"
    return "loop"


def _mode_of(ctx: FlowContext) -> str:
    """The active mode, set by ``run_deep``'s dispatch preamble."""
    deep_state = DeepState(ctx.data)
    return str(deep_state.mode)


def _fix_cycle_enabled(ctx: FlowContext) -> bool:
    """The apply-fix gate + verify/fix/test/commit run in loop + shallow modes."""
    return _mode_of(ctx) in ("loop", "shallow")


def _cleanup_applies(ctx: FlowContext) -> bool:
    """Return whether this mode writes ``.review-output.md``."""
    return _mode_of(ctx) in ("loop", "shallow", "review", "comment")


def _cleanup_should_run(ctx: FlowContext, exit_code: int) -> bool:
    """Whether terminal cleanup runs after a deep flow — success-path only.

    A zero exit is a successful completion (an early ``Stop(0)``, e.g. a
    declined fix gate, still honors ``--cleanup``); any non-zero (failure)
    exit skips cleanup so evidence survives (#335). ``--findings-out`` runs
    stop with exit 0 but must keep the rendered report the run was asked to
    produce, so they are excluded (``findings_out is None``).
    """
    return exit_code == 0 and _cleanup_applies(ctx) and ctx.config.findings_out is None


def _flow_kind_for_mode(mode: str) -> DaydreamRunFlow:
    """Recorder run-flow label per mode (preserves the pre-collapse mapping).

    ``diagram`` gets its own label rather than borrowing ``TTT`` (issue #1113):
    ``archive._flow_runs_merge`` returns True for TTT, so a diagram run would
    otherwise inherit a previous deep review's ``merged-items.json`` as its own
    pipeline state -- and diagram runs deliberately leave those artifacts on
    disk.
    """
    if mode == "shallow":
        return DaydreamRunFlow.NORMAL
    if mode == "diagram":
        return DaydreamRunFlow.DIAGRAM
    if mode in ("review", "comment"):
        return DaydreamRunFlow.TTT
    return DaydreamRunFlow.DEEP


def _flow_name_for_mode(mode: str) -> str:
    """The registered flow name a mode runs (issue #1113).

    Every PR-process mode runs the ``deep`` flow; only ``diagram`` has its own.
    ``loop`` / ``comment`` / ``review`` / ``shallow`` are modes, not registered
    flow names, so the raw mode string must never be passed to ``run_flow``.
    """
    return "diagram" if mode == "diagram" else "deep"


# Cleanup follows every successful flow exit, including an early declined gate.
# Failures retain evidence.
STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="exploration", run=_step_exploration),
    FlowStep(name="intent", run=_step_intent, enabled=_fresh_ttt),
    FlowStep(
        name="per-stack-reviews",
        run=_step_wonder_and_per_stack,
        config_phase="per_stack_review",
    ),
    FlowStep(name="per-stack-parse", run=_step_per_stack_parse, config_phase="parse", enabled=_before_fix_resume),
    FlowStep(name="arbiter", run=_step_arbiter, enabled=_multi_stack_merge_enabled),
    FlowStep(
        name="cross-stack-merge", run=_step_cross_stack_merge, config_phase="merge", enabled=_multi_stack_merge_enabled
    ),
    FlowStep(name="single-stack-merge", run=_step_single_stack_merge, enabled=_single_stack_merge_enabled),
    FlowStep(name="load-items", run=_step_load_items),
    FlowStep(name="supervise", run=_step_supervise, config_phase="supervise", enabled=_supervise_enabled),
    FlowStep(name="diagram", run=_step_diagram, config_phase="diagram", enabled=_diagram_enabled),
    FlowStep(name="findings-out", run=_step_findings_out, enabled=_findings_out_enabled),
    FlowStep(name="post-review", run=_step_post_review, enabled=_before_fix_resume),
    # Fix cycle: loop + shallow modes only (review/comment stop after post-review).
    FlowStep(name="fix-gate", run=_step_fix_gate, enabled=_fix_cycle_enabled),
    FlowStep(name="verify", run=_step_verify, enabled=_fix_cycle_enabled),
    FlowStep(name="fix", run=_step_fix, enabled=_fix_cycle_enabled),
    FlowStep(name="fix-verify", run=_step_fix_verify, enabled=_fix_cycle_enabled),
    FlowStep(name="test", run=_step_test, enabled=_fix_cycle_enabled),
    # config_phase "fix" mirrors the old body's use of the fix backend for the commit.
    FlowStep(name="commit", run=_step_commit, config_phase="fix", enabled=_fix_cycle_enabled),
    FlowStep(name="remote-ci", run=_step_remote_ci, enabled=_remote_ci_enabled),
)


# Keep diagram-only publication outside STEPS: builtins derives the ordinary deep
# flow from that tuple, where post-diagram would cause an unintended GitHub write.
DIAGRAM_STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="post-diagram", run=_step_post_diagram),
)


@bind_resolved_run_context
async def run_deep(
    config: RunConfig,
    work: WorkContext,
    *,
    run_artifacts: _RunArtifacts | None = None,
    run_context: RunContext | None = None,
    github_execution: GitHubExecutionInput | None = None,
    backend_factory: BackendFactory | None = None,
    allow_standalone: bool = False,
) -> int:
    """Prepare and execute the registered deep flow for review, comment, shallow or loop.

    Resolve diff, stacks, routing and preflight before dispatch. start_at supports
    TTT, per-stack, merge and fix resumes. Runner-managed calls supply run_artifacts;
    standalone calls require explicit allow_standalone=True and no active session.
    Return the pipeline exit code.
    """
    run_context = resolve_run_context(run_context)
    if run_artifacts is None and not allow_standalone:
        raise ArtifactVisibilityError("standalone deep flow requires allow_standalone=True")
    if run_artifacts is None and artifact_session_active():
        raise ArtifactVisibilityError("standalone deep flow cannot run inside an active artifact session")
    execution = github_execution or GitHubExecutionInput()
    return await _run_review_spine(
        config,
        work,
        _resolve_mode(config),
        run_artifacts=run_artifacts,
        run_context=run_context,
        github_execution=execution,
        backend_factory=backend_factory,
        allow_standalone=allow_standalone,
    )


def _collapse_stacks_for_shallow(
    stacks: list[StackAssignment],
    changed_files: list[str],
    config: RunConfig,
) -> tuple[list[StackAssignment], bool]:
    """Combine non-structural files into one assignment, retaining structure separately.

    An explicit stack wins; otherwise preserve a sole real language (absorbing
    generic/docs files), or use generic for multiple/no real languages.
    Return the assignments and True for single-stack mode.
    """
    structural = [s for s in stacks if s.stack_name == STRUCTURE_STACK_NAME]
    combined_files = sorted({f for s in stacks for f in s.files}) or changed_files

    real_language = [s for s in stacks if s.stack_name not in (STRUCTURE_STACK_NAME, GENERIC_STACK)]

    if config.stack is not None:
        stack_name = config.stack
    elif len(real_language) == 1:
        # Scope preservation: a sole real-language stack survives unchanged,
        # absorbing any generic/docs files.
        stack_name = real_language[0].stack_name
    else:
        # Multiple real-language stacks (one agent cannot cover two per-language
        # scopes) or no real language at all: the combined assignment uses the
        # native generic-fallback scope.
        stack_name = GENERIC_STACK
    combined = StackAssignment(
        stack_name=stack_name,
        files=combined_files,
        is_docs_only=False,
    )
    return [*structural, combined], True


def _prepare_review_stacks(
    config: RunConfig,
    changed_files: list[str],
    diff: str,
    target_dir: Path,
    mode: str,
) -> tuple[list[StackAssignment], bool, dict[str, set[str]]]:
    """Resolve review scopes, optional shards, and the diagram import graph."""
    # Stack detection (from diff file list). Built-in detection is
    # registry-independent (M1); fork stack rules still resolve via the
    # registry inside detect_stacks.
    stacks = detect_stacks(changed_files)
    # Apply the resolved structural gate before collapsing or sharding assignments.
    if not _config_pipeline(config).structural_enabled:
        stacks = [s for s in stacks if s.stack_name != STRUCTURE_STACK_NAME]
    # Recompute tiny-diff collapse on resumes so they follow the same merge/arbiter bypass.
    stacks, single_stack_mode = _collapse_stacks_for_tiny_diff(
        stacks, changed_files, threshold=_shallow_fanout_threshold(config)
    )
    # Issue #330 — ``--shallow`` forces the single-stack assignment regardless
    # of diff size, so no arbiter / cross-stack merge runs.
    if mode == "shallow":
        stacks, single_stack_mode = _collapse_stacks_for_shallow(stacks, changed_files, config)

    # Shard after collapse and before publishing scopes; use full persisted diff bytes.
    # Single-stack and sharding-off runs retain their assignments.
    import_graph: dict[str, set[str]] = {}
    sharding_enabled = _deep_shard_enabled(config, diff=diff)
    shard_this_run = sharding_enabled and not single_stack_mode
    # Build one import graph for sharding and diagram eligibility. The guard also
    # catches unsafe tree-sitter versions that can escape the fail-open builder.
    diagram_needs_graph = (
        config.start_at != "fix" and _diagram_mode_for(config, mode) != "off"
    )
    if shard_this_run or diagram_needs_graph:
        try:
            import_graph = build_import_graph(changed_files, target_dir)
        except Exception:
            import_graph = {}
    if shard_this_run:
        stacks = shard_stacks(
            stacks,
            diff,
            max_files=_deep_shard_max_files(config),
            max_bytes=_resolve_non_negative_int(config, "deep_shard_max_bytes", DEFAULT_DEEP_SHARD_MAX_BYTES),
            fanout_cap=_resolve_non_negative_int(config, "deep_shard_fanout_cap", DEFAULT_DEEP_SHARD_FANOUT_CAP),
            frontier_max=_resolve_non_negative_int(
                config, "deep_shard_frontier_max", DEFAULT_DEEP_SHARD_FRONTIER_MAX
            ),
            graph=import_graph,
        )

    return stacks, single_stack_mode, import_graph


@bind_resolved_run_context
async def _run_review_spine(
    config: RunConfig,
    work: WorkContext,
    mode: str,
    *,
    run_artifacts: _RunArtifacts | None,
    run_context: RunContext | None = None,
    github_execution: GitHubExecutionInput,
    backend_factory: BackendFactory | None = None,
    allow_standalone: bool = False,
) -> int:
    """Review-spine preamble for the deep pipeline (the former ``run_deep`` body)."""
    artifact_session = None if run_artifacts is None else run_artifacts.session
    run_context = resolve_run_context(run_context)
    # Late imports to avoid circular dependency with runner.
    from daydream.git_ops import GitError, GitTimeoutError
    from daydream.hunk_index import write_hunk_index
    from daydream.phases.inputs import (
        _git_branch,
        _git_log,
    )
    from daydream.run_artifacts import _open_recorder, _resolve_review_profile
    from daydream.run_config import _default_backend_name, _resolved_latency_profile

    target_dir = work.repo

    # Findings export is commit-bound, in every mode that writes an artifact:
    # capture the target once, before any work happens, and diff explicit SHA
    # endpoints. Diagram mode captures the same identity but keeps its own
    # two-dot diff selection below.
    from daydream.review_result import AnalyzedRevision, PlannedScope, ReviewCoverage

    captured_pr = None
    pr_base_sha = None
    dirty_snapshot = False
    diff: str | None
    try:
        captured_head = git_ops.head_sha(target_dir)
        captured_base = git_ops.resolve_diff_merge_base(target_dir, work.base_branch, captured_head)
    except GitTimeoutError as exc:
        print_error(console, "Git Timeout", f"git timed out under load: {exc}")
        return 1
    except GitError:
        diff = None
    else:
        if config.findings_out is not None:
            try:
                captured = _capture_findings_target(
                    target_dir, config, github_execution, captured_head, captured_base,
                    capture_base_tip=mode != "diagram",
                )
            except GitTimeoutError as exc:
                print_error(console, "Git Timeout", f"git timed out under load: {exc}")
                return 1
            except GitError as exc:
                print_error(console, "Git Error", f"Unable to resolve pull request for findings export: {exc}")
                return 1
            if captured is None:
                return 1
            captured_pr, pr_base_sha = captured
            # Commit-bound, in every artifact mode: a dirty analyzed checkout is
            # rejected below, because the diagram path still reads the diff (and
            # its grounding) from working-tree content.
            dirty_snapshot = _has_non_daydream_worktree_changes(git_ops.status_porcelain(target_dir))
        try:
            if config.findings_out is not None and mode != "diagram":
                paths = [".", *(f":(exclude){p.rstrip('/')}" for p in config.ignore_paths or [])]
                diff = git_ops.diff_paths(target_dir, captured_base, captured_head, paths)
            else:
                diff = git_ops.diff(work.repo, work.base_branch, exclude=config.ignore_paths)
        except GitTimeoutError as exc:
            print_error(console, "Git Timeout", f"git timed out under load: {exc}")
            return 1
        except GitError:
            diff = None
    log = _git_log(target_dir)
    branch = work.head_branch or _git_branch(target_dir)

    if diff is None:
        print_error(console, "Git Error", "Unable to determine base branch for diff")
        return 1
    no_diff = not diff.strip()
    if no_diff and mode == "diagram":
        subject = "diagram" if mode == "diagram" else "review"
        print_warning(console, f"No diff found -- nothing to {subject}")
        return 0

    daydream_dir = artifact_dir_for(
        target_dir,
        session=artifact_session,
        allow_standalone=allow_standalone,
    )
    daydream_dir.mkdir(exist_ok=True)
    diff_path = daydream_dir / "diff.patch"
    diff_path.write_text(diff)
    # Persist the hunk index immediately after the diff bytes, so the run-time
    # authority (changed file/line ranges) is available to every later step
    # (reviews, arbiter, merge) and never predates the patch.
    write_hunk_index(daydream_dir, diff)
    # Diff is immutable from here on; compute the tiering verdict once and reuse
    # it at both the exploration step's gate and the alternatives step's gate.
    # The latency route (issue #732) is resolved from the same immutable diff
    # just before ``run_flow``, once the in-memory diff is bounded.
    tier = review_steps.select_tier(review_steps.count_changed_files(diff))
    dd = deep_dir(
        target_dir,
        session=artifact_session,
        allow_standalone=allow_standalone,
    )
    current_diff_sha = diff_key(diff)
    # Diagram-only runs preserve prior deep-review artifacts and write no diff-key,
    # because they do not produce the review artifacts that key attests.
    if mode != "diagram" and config.start_at not in ("per-stack", "merge", "fix"):
        # Fresh run only: a resume must NOT rewrite the key it is checked
        # against, or the staleness gate would self-heal and pass every time.
        shutil.rmtree(dd, ignore_errors=True)
        dd.mkdir(parents=True, exist_ok=True)
        DeepArtifact.DIFF_KEY.at(dd).write_text(current_diff_sha, encoding="utf-8")

    async with _open_recorder(
        config=config, target_dir=target_dir, work=work, flow_kind=_flow_kind_for_mode(mode),
        run_artifacts=run_artifacts,
        allow_standalone=allow_standalone,
    ) as recorder:
        # Composition-root re-entry: resolution already happened in
        # ``_run_loop_deep`` before this recorder existed; this no-op resolve
        # records the profile onto the active recorder (R12).
        _resolve_review_profile(config)
        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Branch: {branch}")
        print_info(console, f"Default backend: {_default_backend_name(config)}")
        # Bot logins look like ``my-app[bot]``; escape so Rich doesn't eat the brackets.
        print_info(console, f"GitHub identity: {escape_markup(config.identity)}")
        console.print()

        changed_files = _diff_changed_files(diff)
        stacks, single_stack_mode, import_graph = _prepare_review_stacks(
            config, changed_files, diff, target_dir, mode
        )

        required_phases = ["no_diff"] if no_diff else ["intent", "alternatives", "merge"]
        if not no_diff and _supervisor_mode(config) in {"rules", "llm"}:
            required_phases.append("supervision")
        coverage = None if mode == "diagram" else ReviewCoverage(
            recorder.session_id,
            AnalyzedRevision(captured_head, captured_base, current_diff_sha,
                             pr_base_sha),
            [PlannedScope(stack.stack_name, stack.stack_name.split("#", 1)[0],
                          files=tuple(sorted(stack.files)),
                          shard=int(stack.stack_name.split("#", 1)[1]) if "#" in stack.stack_name else None)
             for stack in stacks] if not no_diff else [],
            required_phases,
        )

        # Resume gate (D-34, D-36, D-37) + diff-freshness gate.
        if config.start_at in ("per-stack", "merge", "fix"):
            try:
                check_deep_artifacts(
                    config.start_at, dd, current_diff_sha=current_diff_sha,
                    record_paths=[per_stack_records_path(dd, stack.stack_name) for stack in stacks],
                )
                if coverage is not None:
                    from daydream.deep.artifacts import restore_review_coverage
                    coverage = restore_review_coverage(dd, coverage)
                if _has_non_daydream_worktree_changes(git_ops.status_porcelain(target_dir)):
                    raise FileNotFoundError(
                        f"Cannot resume at stage '{config.start_at}' -- the worktree has changed "
                        "since the review artifacts were generated.\n\n"
                        "Resuming would review stale findings against changed code.\n"
                        "Re-run without --start-at to regenerate them."
                    )
            except (FileNotFoundError, ValueError) as exc:
                print_error(console, "Unusable Deep Artifacts", str(exc))
                return 1
        # Pre-flight notice (D-30). Agent count reflects the tiny-diff collapse
        # when single_stack_mode is active (issue #172): merge+arbiter are
        # skipped, so the estimate uses ``_single_stack_agent_count``.
        stack_lines = [
            _stack_preflight_line(stack)
            for stack in stacks
            if stack.stack_name != STRUCTURE_STACK_NAME
        ]
        notice_agent_count = (
            _single_stack_agent_count(len(stacks))
            if single_stack_mode
            else total_agent_count(len(stacks))
        )
        default_profile = build_default_profile()
        profile = config.review_profile.profile if config.review_profile is not None else default_profile
        alternatives_strategy = profile.strategies.get("alternatives", default_profile.strategies["alternatives"])
        registry = get_registry()
        folded_alternatives = fold_default_alternatives(
            stacks, alternatives_strategy.content, structural_prompt_builder=registry.prompt("structural"),
        )
        if folded_alternatives:
            notice_agent_count -= 1
        # Issue #1113: the notice hardcodes "Deep-review pipeline pre-flight",
        # the five deep pipeline stages and a 2+2N+2 agent estimate. A two-step
        # diagram flow executes none of that, so printing it would be a lie
        # about what the run is doing.
        if mode != "diagram":
            print_preflight_notice(
                console,
                stages=_preflight_stage_names(stacks, folded_alternatives=folded_alternatives),
                stack_lines=stack_lines,
                agent_count=notice_agent_count,
                exploration_available=review_steps.EXPLORATION_AVAILABLE,
            )

        # The context shares backends and bounded inline diff text. The full patch is
        # already persisted and remains the source for exploration, keys, and archival.
        bounded_diff, bound_info = bound_deep_diff(diff)
        if bound_info.truncated:
            dropped = (
                f"; dropped blocks: {', '.join(bound_info.dropped_paths)}"
                if bound_info.dropped_paths
                else ""
            )
            oversize = (
                f"; oversized block kept whole: {', '.join(bound_info.oversize_paths)}"
                if bound_info.oversize_paths
                else ""
            )
            print_warning(
                console,
                f"Deep diff truncated: {bound_info.original_bytes} -> "
                f"{bound_info.retained_bytes} bytes "
                f"({bound_info.retained_blocks}/{bound_info.total_blocks} blocks retained"
                f"{dropped}{oversize})",
            )
        # Publish one risk-escalated route before any phase resolves effort. Later stages
        # append their decisions through the same routing-record writer.
        latency_resolution = _resolved_latency_profile(config)
        latency_signals = diff_signals(
            diff=bounded_diff, changed_files=len(changed_files), stack_count=len(stacks)
        )
        latency_summary = summarize_risk(latency_signals)
        latency_route = route_for(latency_resolution.profile, latency_summary)
        config.latency_route = latency_route
        write_routing_record(
            dd,
            {
                "profile": {
                    "selected": latency_resolution.profile,
                    "requested": latency_resolution.requested,
                    "source": latency_resolution.source,
                    "fail_safe": latency_resolution.fail_safe,
                    "reason": latency_resolution.reason,
                },
                "risk": {
                    "signals": asdict(latency_signals),
                    "floors": list(latency_summary.floors),
                    "size_score": latency_summary.size_score,
                    "breadth_score": latency_summary.breadth_score,
                },
            },
        )
        ctx = FlowContext(
            config=config,
            work=work,
            registry=registry,
            review_profile=config.review_profile,
            private_workspace_owner=None if run_artifacts is None else run_artifacts.owner,
            artifacts=None if run_artifacts is None else run_artifacts.session,
            run_context=run_context,
            github_execution=github_execution,
            _backend_factory=backend_factory,
            data={
                "mode": mode,
                "review_coverage": coverage,
                "analyzed_pr": captured_pr,
                "snapshot_diff": diff,
                "diff": bounded_diff,
                # Carry reviewed origins for footprint enforcement; resumes can recover them
                # from the diff. Canonical finding paths have independent authorization.
                "changed_files": set(changed_files),
                "diff_path": diff_path,
                "diff_truncated": bound_info.truncated,
                "diff_truncation": bound_info,
                "tier": tier,
                # Issue #732: the resolved route plus the summary it was picked
                # from, so a later step (wonder, arbiter) can state its decision
                # and the risk floors that forced it without recomputing.
                "latency_route": latency_route,
                "risk_summary": latency_summary,
                "dd": dd,
                "stacks": stacks,
                # Issue #1113: the changed-file import graph, published so the
                # diagram step's cross-module rule can read it. ``{}`` simply
                # denies that rule; it never fails the run.
                "import_graph": import_graph,
                "single_stack_mode": single_stack_mode,
                "intent_path": DeepArtifact.INTENT.at(dd),
                "alts_path": DeepArtifact.ALTERNATIVES.at(dd),
                "log": log,
                "branch": branch,
            },
            allow_standalone_artifacts=allow_standalone,
        )

        # Publish one cache handle and its root so the next run can reuse completed units.
        ctx.data["reuse_cache"] = build_reuse_cache(ctx)

        # Resolve and persist one test recipe for all host and prompt consumers.
        # Confinement failure leaves no recipe and preserves the unresolved fallback.
        try:
            test_recipe = resolve_test_recipe(
                getattr(config, "file_config", None), config, repo_root=work.repo
            )
        except (RecipeConfinementError, OSError) as exc:
            test_recipe = None
            print_warning(console, f"Could not resolve the test recipe: {exc}")
        if test_recipe is not None:
            try:
                persist_test_recipe(dd, test_recipe)
            except OSError as exc:
                print_warning(console, f"Could not persist the resolved test recipe: {exc}")
            ctx.data["test_recipe"] = test_recipe

        # Finalization stays inside the recorder so frozen public/archive evidence agrees.
        from daydream.deep.review_terminal import finalize_review
        from daydream.review_result import reason_for_exception

        if dirty_snapshot:
            # Diagram runs carry no coverage record, so the reject is reported
            # without a phase entry there; the gate itself is mode-independent.
            if coverage is not None:
                coverage.require_phase("snapshot")
                coverage.record_phase("snapshot", "failed", reasons=["dirty_snapshot"])
                finalize_review(ctx, "failed")
            print_error(console, "Findings Artifact", "Commit-bound export requires a clean analyzed checkout")
            return 1
        if no_diff and coverage is not None:
            coverage.record_phase("no_diff", "complete", noop=True)
            return finalize_review(ctx, "completed", no_diff=True)
        try:
            with review_deadline_scope(
                ctx.pipeline().review_wall_budget_s, diff=diff,
                scale_deadline=config.review_profile is not None
                and config.review_profile.source_kind == "default",
            ):
                exit_code = await run_flow(ctx.registry, _flow_name_for_mode(mode), ctx)
        except Exception as exc:
            if coverage is not None and not coverage.is_finalized:
                coverage.require_phase("pipeline")
                coverage.record_phase("pipeline", "failed", reasons=[reason_for_exception(exc)])
                try:
                    finalize_review(ctx, "failed")
                except Exception as final_exc:
                    exc.add_note(f"Terminal review finalization failed: {type(final_exc).__name__}")
                    print_warning(console, f"Terminal review finalization failed: {type(final_exc).__name__}")
            raise
        if coverage is not None and not coverage.is_finalized:
            try:
                final_status = finalize_review(ctx, "completed" if exit_code == 0 else "failed")
            except (OSError, ValueError) as exc:
                print_error(console, "Findings Artifact", f"Terminal review finalization failed: {exc}")
                return 1
            if final_status:
                return final_status
        if _cleanup_should_run(ctx, exit_code):
            await _perform_cleanup(ctx)
        return exit_code
