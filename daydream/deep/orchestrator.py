"""Compose deep and diagram flows, their preambles, and public entry points."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markup import escape as escape_markup

from daydream import clock, git_ops
from daydream.agent import console
from daydream.artifact_visibility import ArtifactVisibilityError, artifact_dir_for, artifact_session_active
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.deep import review_steps
from daydream.deep.artifacts import DeepArtifact, deep_dir, diff_key
from daydream.deep.dependency import build_import_graph
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
from daydream.deep.merge_steps import (
    _step_findings_out,
    _step_load_items,
    _step_post_review,
)
from daydream.deep.remote_ci_steps import _step_remote_ci
from daydream.deep.review_steps import (
    _step_exploration,
)
from daydream.deep.review_terminal import _step_committed_review
from daydream.deep.state import DeepState
from daydream.extensions import get_registry
from daydream.extensions.api import FlowStep
from daydream.flows.engine import BackendFactory, FlowContext, run_flow
from daydream.github_app import GitHubExecutionInput
from daydream.phases import PushReceipt
from daydream.review_budget import review_deadline_scope
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.test_execution import (
    RecipeConfinementError,
    persist_test_recipe,
    resolve_test_recipe,
)
from daydream.trajectory import DaydreamRunFlow
from daydream.ui import print_error, print_info, print_warning
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.pr_review import PRInfo
    from daydream.run_artifacts import _RunArtifacts
    from daydream.run_config import RunConfig


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
    target_dir: Path,
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


def _diagram_enabled(ctx: FlowContext) -> bool:
    """Run diagram eligibility checks unless mode is off or this is a fix-only resume.

    No eligible diagram means zero agent calls, but the step still records why.
    Diagram-only flows always pass this gate.
    """
    return _resolved_diagram_mode(ctx) != "off"


def _findings_out_enabled(ctx: FlowContext) -> bool:
    # Two-phase findings artifact (Phase A): emit the artifact and STOP —
    # never post to the PR and never apply fixes. Phase B posts later.
    return ctx.config.findings_out is not None


def _resolve_mode(config: RunConfig) -> str:
    """Map a RunConfig onto the single deep-flow mode key (#330).

    ``review`` / ``comment`` stop after post-review; ``diagram`` (issue #1113) reuses the
    spine's preamble but runs the two-step ``diagram`` flow. Only ``output_mode="loop"``
    selects the fixing mode ``loop``.
    """
    if config.flow_name == "review":
        return "review"
    if config.output_mode in ("diagram", "review", "comment"):
        return config.output_mode
    return "loop"


def _mode_of(ctx: FlowContext) -> str:
    """The active mode, set by ``run_deep``'s dispatch preamble."""
    deep_state = DeepState(ctx.data)
    return str(deep_state.mode)


def _fix_cycle_enabled(ctx: FlowContext) -> bool:
    """The apply-fix gate + verify/fix/test/commit run in loop mode only."""
    return _mode_of(ctx) == "loop"


def _cleanup_applies(ctx: FlowContext) -> bool:
    """Return whether this mode writes ``.review-output.md``."""
    return _mode_of(ctx) in ("loop", "review", "comment")


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
    if mode == "diagram":
        return DaydreamRunFlow.DIAGRAM
    if mode in ("review", "comment"):
        return DaydreamRunFlow.TTT
    return DaydreamRunFlow.DEEP


def _flow_name_for_mode(mode: str) -> str:
    """The registered flow name a mode runs (issue #1113).

    Every PR-process mode runs the ``deep`` flow; only ``diagram`` has its own.
    ``loop`` / ``comment`` / ``review`` are modes, not registered
    flow names, so the raw mode string must never be passed to ``run_flow``.
    """
    return "diagram" if mode == "diagram" else "deep"


# Cleanup follows every successful flow exit, including an early declined gate.
# Failures retain evidence.
STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="review", run=_step_committed_review),
    FlowStep(name="load-items", run=_step_load_items),
    FlowStep(name="diagram", run=_step_diagram, config_phase="diagram", enabled=_diagram_enabled),
    FlowStep(name="findings-out", run=_step_findings_out, enabled=_findings_out_enabled),
    FlowStep(name="post-review", run=_step_post_review),
    # Fix cycle: loop mode only (review/comment stop after post-review).
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
    FlowStep(name="exploration", run=_step_exploration),
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
    """Plan the review, then execute the registered deep flow for review, comment, fixing or diagram-only.

    Every review starts fresh. Runner-managed calls supply run_artifacts; standalone calls
    require explicit allow_standalone=True and no active session. Return the pipeline exit code.
    """
    started = clock.monotonic()
    run_context = resolve_run_context(run_context)
    if run_artifacts is None and not allow_standalone:
        raise ArtifactVisibilityError("standalone deep flow requires allow_standalone=True")
    if run_artifacts is None and artifact_session_active():
        raise ArtifactVisibilityError("standalone deep flow cannot run inside an active artifact session")
    artifact_session = None if run_artifacts is None else run_artifacts.session
    mode = _resolve_mode(config)
    github_execution = github_execution or GitHubExecutionInput()
    # Late imports to avoid circular dependency with runner.
    from daydream.git_ops import GitError, GitTimeoutError
    from daydream.hunk_index import write_hunk_index
    from daydream.phases.inputs import (
        _git_branch,
    )
    from daydream.run_artifacts import _open_recorder, _resolve_review_profile
    from daydream.run_config import _default_backend_name

    target_dir = work.repo

    # Findings export is commit-bound, in every mode that writes an artifact:
    # capture the target once, before any work happens, and diff explicit SHA
    # endpoints. Diagram mode captures the same identity but keeps its own
    # two-dot diff selection below.
    from daydream.review_result import AnalyzedRevision

    captured_pr = work.review_pr
    pr_base_sha = None if captured_pr is None else captured_pr.pr_base_sha
    source_id = None if work.captured_source is None else work.captured_source.source_id
    dirty_snapshot = False
    diff: str | None
    try:
        if work.captured_source is not None:
            work.captured_source.validate()
            captured_head = work.captured_source.head_sha
            captured_base = work.captured_source.base_sha
        else:
            captured_head = git_ops.head_sha(target_dir)
            captured_base = git_ops.resolve_diff_merge_base(target_dir, work.base_branch, captured_head)
    except GitTimeoutError as exc:
        print_error(console, "Git Timeout", f"git timed out under load: {exc}")
        return 1
    except GitError as exc:
        print_error(console, "Git Error", str(exc))
        return 1
    else:
        if config.findings_out is not None and work.captured_source is None:
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
            if work.captured_source is not None:
                diff = work.captured_source.diff
            elif config.findings_out is not None and mode != "diagram":
                paths = [".", *(f":(exclude){p.rstrip('/')}" for p in config.ignore_paths or [])]
                diff = git_ops.diff_paths(target_dir, captured_base, captured_head, paths)
            else:
                diff = git_ops.diff(work.repo, work.base_branch, exclude=config.ignore_paths)
        except GitTimeoutError as exc:
            print_error(console, "Git Timeout", f"git timed out under load: {exc}")
            return 1
        except GitError:
            diff = None
    if diff is not None and run_artifacts is not None and config.dataset_capture:
        from daydream.dataset_capture import retain_original_task

        try:
            retain_original_task(
                run_artifacts, work=work, config=config,
                analyzed_revision=AnalyzedRevision(captured_head, captured_base, diff_key(diff), pr_base_sha,
                                                   source_id=source_id).to_dict(),
                diff=diff,
            )
        except Exception as exc:
            print_warning(console, f"Original run evidence could not be retained ({type(exc).__name__}).")
    branch = work.head_branch or _git_branch(target_dir)

    if diff is None:
        print_error(console, "Git Error", "Unable to determine base branch for diff")
        return 1
    no_diff = not diff.strip()
    if no_diff:
        print_warning(console, f"No diff found -- nothing to {'diagram' if mode == 'diagram' else 'review'}")
        if mode == "diagram":
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
    index_path = write_hunk_index(daydream_dir, diff)
    # Diff is immutable from here on; compute the tiering verdict once and reuse
    # it at both the exploration step's gate and the alternatives step's gate.
    # The latency route is resolved from this full immutable diff before prompt
    # bounding, so prompt capacity cannot hide changed-code surface signals.
    tier = review_steps.select_tier(review_steps.count_changed_files(diff))
    dd = deep_dir(
        target_dir,
        session=artifact_session,
        allow_standalone=allow_standalone,
    )
    current_diff_sha = diff_key(diff)
    # Diagram-only runs preserve prior deep-review artifacts and write no diff-key,
    # because they do not produce the review artifacts that key attests.
    if mode != "diagram":
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

        changed_files = (list(work.captured_source.files) if work.captured_source is not None
                         else _diff_changed_files(diff))
        # One changed-file import graph for diagram eligibility. The guard also
        # catches unsafe tree-sitter versions that can escape the fail-open builder.
        import_graph: dict[str, set[str]] = {}
        if _diagram_mode_for(config, mode) != "off":
            try:
                import_graph = build_import_graph(changed_files, target_dir)
            except Exception:
                import_graph = {}
        registry = get_registry()
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
                "review_coverage": None,
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
                "dd": dd,
                # Issue #1113: the changed-file import graph, published so the
                # diagram step's cross-module rule can read it. ``{}`` simply
                # denies that rule; it never fails the run.
                "import_graph": import_graph,
            },
            allow_standalone_artifacts=allow_standalone,
        )

        coverage = None
        if mode != "diagram":
            from daydream.deep.review_terminal import plan_committed_review
            from daydream.prompt_budget import SanctionedInputUnavailable

            try:
                await plan_committed_review(
                    ctx, run_id=recorder.session_id,
                    revision=AnalyzedRevision(captured_head, captured_base, current_diff_sha, pr_base_sha,
                                              source_id=source_id).to_dict(),
                    index_path=index_path, files=changed_files, started=started,
                )
            except (ValueError, SanctionedInputUnavailable, GitError) as exc:
                print_error(console, "Review Plan", str(exc))
                return 1
            coverage = DeepState(ctx.data).review_coverage
            planned = ctx.data["committed_review"].settings["assignment_policy"]["planned_assignments"]
            print_info(console, f"Review: {len(changed_files)} changed file(s) in {planned} discovery assignment(s), "
                                "then a fresh challenge of every claim.")
            if mode == "loop":
                print_info(console, "Fixing is selected: confirmed findings go to the fix gate.")

        # Resolve and persist one test recipe for all host and prompt consumers.
        # Without package facts, direct configuration still requires readiness;
        # an unresolved command never dispatches a test agent.
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
        from daydream.deep.review_terminal import finalize_review, refresh_review_report
        from daydream.review_result import ReasonCode, reason_for_exception

        if dirty_snapshot:
            # Diagram runs carry no coverage record, so the reject is reported
            # without a phase entry there; the gate itself is mode-independent.
            if coverage is not None:
                finalize_review(ctx, "failed", reason=ReasonCode.DIRTY_SNAPSHOT)
            print_error(console, "Findings Artifact", "Commit-bound export requires a clean analyzed checkout")
            return 1
        if no_diff and coverage is not None:
            return finalize_review(ctx, "completed")
        try:
            with review_deadline_scope(
                ctx.pipeline().review_wall_budget_s, diff=diff,
                scale_deadline=config.review_profile is not None
                and config.review_profile.source_kind == "default",
            ):
                if mode == "diagram":
                    exit_code = await run_flow(ctx.registry, "diagram", ctx)
                else:
                    exit_code = await run_flow(ctx.registry, _flow_name_for_mode(mode), ctx)
        except BaseException as exc:
            if coverage is not None and not coverage.is_finalized:
                try:
                    if isinstance(exc, Exception):
                        finalize_review(ctx, "failed", reason=reason_for_exception(exc))
                    else:
                        # A cancelled review seals what it admitted before the cancellation escapes;
                        # the reviewer holds admissions in memory until then.
                        finalize_review(ctx, "cancelled")
                except Exception as final_exc:
                    exc.add_note(f"Terminal review finalization failed: {type(final_exc).__name__}")
                    print_warning(console, f"Terminal review finalization failed: {type(final_exc).__name__}")
            elif coverage is not None and coverage.is_finalized:
                try:
                    refresh_review_report(ctx)
                except Exception as report_exc:
                    exc.add_note(f'Review report refresh failed: {type(report_exc).__name__}')
                    print_warning(console, f'Review report unavailable: {type(report_exc).__name__}')
            raise
        if coverage is not None and not coverage.is_finalized:
            try:
                final_status = finalize_review(ctx, "completed" if exit_code == 0 else "failed")
            except (OSError, ValueError) as exc:
                print_error(console, "Findings Artifact", f"Terminal review finalization failed: {exc}")
                return 1
            if final_status:
                return final_status
        elif coverage is not None and mode == 'loop':
            try:
                refresh_review_report(ctx)
            except (OSError, ValueError, TypeError, RecursionError) as exc:
                ctx.data['review_report_diagnostic'] = f'Review report unavailable: {type(exc).__name__}'
                print_warning(console, ctx.data['review_report_diagnostic'])
        if _cleanup_should_run(ctx, exit_code):
            await _perform_cleanup(ctx)
        return exit_code
