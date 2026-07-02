"""Shallow flow steps, lifted verbatim from ``runner._run_loop_shallow``.

Each step is a verbatim lift of a contiguous block of the old imperative
body (Pattern: Step extraction): locals became ``ctx.data[...]`` entries,
``_resolve_backend(config, "<phase>", cache)`` became
``ctx.backend_for("<phase>")``, early ``return <int>`` became
``return Stop(<int>)``, and the ``while`` loop's ``break`` became
``return BreakLoop()``. ``phase_scope`` wrappers, error handling, and print
calls moved with their blocks unchanged, so the flow is behavior-neutral.

The ``--loop`` iteration is the ``iterate`` :class:`LoopGroup`
(review -> parse -> fix -> test -> commit-iteration), run once in
single-pass mode. The ``start_at`` resume semantics are encoded VERBATIM as
``enabled`` predicates — NOT a generic ordered cut: ``start_at="fix"`` still
runs parse, exactly like the old body. The old ``while...else`` branch is
the ``loop-exhausted`` step, gated by ``ctx.data["loop_broke"]`` (set on the
paths that used to ``break``).

``register_builtins`` registers :data:`STEPS` and the ``shallow`` flow
definition (:data:`FLOW`); ``_run_loop_shallow`` keeps the preamble (skill
resolution, review-file check, cleanup gate, trajectory recorder + info
block, pre-fix snapshot/HEAD capture) and delegates here via ``run_flow``.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from daydream import git_ops
from daydream.agent import (
    MissingSkillError,
    console,
    get_assume,
    get_non_interactive,
    resolve_gate,
)
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.exploration import ExplorationContext, safe_explore
from daydream.exploration_runner import count_changed_files, pre_scan, select_tier
from daydream.extensions.api import BreakLoop, FlowStep, LoopGroup, Stop
from daydream.git_ops import GitError
from daydream.phases import (
    _git_diff,
    phase_commit_iteration,
    phase_commit_push,
    phase_commit_push_auto,
    phase_fix,
    phase_parse_feedback,
    phase_review,
    phase_test_and_heal,
    revert_uncommitted_changes,
    severity_sorted,
)
from daydream.trajectory import DaydreamPhase, phase_scope
from daydream.ui import (
    SummaryData,
    phase_subtitle,
    print_dim,
    print_error,
    print_info,
    print_iteration_divider,
    print_phase_hero,
    print_success,
    print_summary,
    print_warning,
)

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext

FlowEntry = str | LoopGroup


async def _step_exploration(ctx: FlowContext) -> None:
    """Pre-scan exploration before the first review (``start_at="review"`` only)."""
    from daydream.runner import _compute_diff_ref

    config = ctx.config
    target_dir = ctx.work.repo

    # Pre-scan exploration before the first review; only when starting at
    # "review" (later start phases skip review, so it'd be wasted).
    if config.start_at == "review" and config.exploration_context is None:
        diff_text = _git_diff(target_dir, exclude=config.ignore_paths) or ""
        tier = select_tier(count_changed_files(diff_text))
        if tier == "skip":
            print_dim(console, "Skipping exploration -- trivial diff")
            config.exploration_context = ExplorationContext()
        else:
            print_phase_hero(console, "EXPLORE", phase_subtitle("EXPLORE"))
            explore_backend = ctx.backend_for("exploration")
            print_dim(console, f"Exploration model: {explore_backend.model}")
            config.exploration_context = await safe_explore(
                pre_scan,
                explore_backend,
                target_dir,
                diff_text,
                config.exploration_depth,
                diff_ref=_compute_diff_ref(target_dir),
            )

    if not config.loop:
        # Single-pass mode.
        # Materialise exploration to disk so phase prompts can reference files.
        # (Loop mode materialises after the dirty-tree check in loop-preflight,
        # keeping today's placement: the untracked exploration files must not
        # trip the clean-repo gate.)
        if config.exploration_context is not None:
            exp_parent = target_dir / ".daydream"
            exp_parent.mkdir(exist_ok=True)
            ctx.data["exploration_dir"] = config.exploration_context.write_to_dir(exp_parent / "exploration")


async def _step_loop_preflight(ctx: FlowContext) -> Stop | None:
    """Loop mode: refuse a dirty tree, then materialise exploration to disk."""
    config = ctx.config
    target_dir = ctx.work.repo

    # Loop mode reverts uncommitted changes on failure; refuse a dirty tree.
    try:
        porcelain = git_ops.status_porcelain(target_dir)
    except GitError:
        porcelain = None
    if porcelain is None or porcelain.strip():
        print_error(
            console,
            "Dirty Working Tree",
            "Loop mode requires a clean repo because failed iterations"
            " discard uncommitted changes.",
        )
        return Stop(1)

    # Materialise exploration to disk so phase prompts can reference files.
    if config.exploration_context is not None:
        exp_parent = target_dir / ".daydream"
        exp_parent.mkdir(exist_ok=True)
        ctx.data["exploration_dir"] = config.exploration_context.write_to_dir(exp_parent / "exploration")
    return None


async def _step_review(ctx: FlowContext) -> Stop | None:
    """Run the review skill (loop mode: incremental diff + iteration divider)."""
    from daydream.runner import _print_missing_skill_error

    config = ctx.config
    work = ctx.work
    target_dir = work.repo
    skill = ctx.data["skill"]

    if config.loop:
        iteration = ctx.data["iteration"]
        if iteration > 1:
            (target_dir / REVIEW_OUTPUT_FILE).unlink(missing_ok=True)
            print_iteration_divider(console, iteration, config.max_iterations)

    assert skill is not None, "skill must be set when starting at review phase"
    try:
        async with phase_scope(DaydreamPhase.REVIEW):
            # ``diff_base`` is None until a loop iteration commits, so the
            # single-pass call is identical to the old no-diff_base form.
            await phase_review(
                ctx.backend_for("review"), work, skill, diff_base=ctx.data["diff_base"],
                exploration_dir=ctx.data["exploration_dir"],
                exclude=config.ignore_paths,
            )
    except MissingSkillError as e:
        _print_missing_skill_error(e.skill_name)
        return Stop(1)
    return None


async def _step_parse(ctx: FlowContext) -> Stop | BreakLoop | None:
    """Parse review feedback; canonicalize + severity-sort the items."""
    from daydream.runner import to_canonical_shallow

    config = ctx.config
    work = ctx.work

    if config.loop:
        try:
            async with phase_scope(DaydreamPhase.PARSE):
                items = await phase_parse_feedback(ctx.backend_for("parse"), work)

            if not items:
                print_info(console, f"Clean review on iteration {ctx.data['iteration']}")
                ctx.data["loop_broke"] = True
                return BreakLoop()  # should_continue=False (clean)

            # Canonicalize onto the lens/severity axes, then fix high→low.
            items = severity_sorted(to_canonical_shallow(items))
        except ValueError as exc:
            print_error(console, "Phase 2 Error", str(exc))
            print_error(console, "Parse Failed", "Failed to parse feedback. Exiting.")
            return Stop(1)
        ctx.data["items"] = items
        ctx.data["feedback_items"].extend(items)
    else:
        try:
            async with phase_scope(DaydreamPhase.PARSE):
                feedback_items = await phase_parse_feedback(ctx.backend_for("parse"), work)
        except ValueError as exc:
            print_error(console, "Phase 2 Error", str(exc))
            print_error(console, "Parse Failed", "Failed to parse feedback. Exiting.")
            return Stop(1)
        # Canonicalize onto the lens/severity axes, then fix high→low.
        ctx.data["feedback_items"] = severity_sorted(to_canonical_shallow(feedback_items))
    return None


async def _step_fix(ctx: FlowContext) -> None:
    """Apply a fix per feedback item (HEAL)."""
    config = ctx.config
    work = ctx.work
    target_dir = work.repo

    if config.loop:
        items = ctx.data["items"]
        print_phase_hero(console, "HEAL", phase_subtitle("HEAL"))
        print_dim(console, f"Model: {ctx.backend_for('fix').model}")
        fixes_count = 0
        async with phase_scope(DaydreamPhase.FIX):
            for i, item in enumerate(items, 1):
                await phase_fix(ctx.backend_for("fix"), work, item, i, len(items))
                fixes_count += 1
        ctx.data["iteration_fixes"] = fixes_count

        # Capture the recommended patch NOW, before tests run and a failure
        # reverts the tree below. Overwritten each iteration so the file holds
        # the cumulative fix against the pre-first-fix base. Best-effort.
        git_ops.capture_recommended_patch_with_base(
            target_dir,
            ctx.data["pre_fix_snapshot"],
            ctx.data["pre_fix_head"],
            target_dir / ".daydream" / "recommended.patch",
        )
    else:
        feedback_items = ctx.data["feedback_items"]
        if feedback_items:
            print_phase_hero(console, "HEAL", phase_subtitle("HEAL"))
            print_dim(console, f"Model: {ctx.backend_for('fix').model}")
            async with phase_scope(DaydreamPhase.FIX):
                for i, item in enumerate(feedback_items, 1):
                    await phase_fix(ctx.backend_for("fix"), work, item, i, len(feedback_items))
                    ctx.data["fixes_applied"] += 1
        else:
            print_info(console, "No feedback items found, skipping fix phase")


async def _step_test(ctx: FlowContext) -> BreakLoop | None:
    """Run the test suite; loop mode reverts and breaks on failure."""
    from daydream.runner import _get_head_sha

    config = ctx.config
    work = ctx.work
    target_dir = work.repo

    if config.loop:
        items = ctx.data["items"]
        async with phase_scope(DaydreamPhase.TEST):
            passed, retries = await phase_test_and_heal(ctx.backend_for("test"), work, feedback_items=items)

        ctx.data["test_retries"] += retries
        ctx.data["tests_passed"] = passed

        if not passed:
            print_warning(console, f"Tests failed on iteration {ctx.data['iteration']}, reverting changes")
            if revert_uncommitted_changes(target_dir):
                print_info(console, "Reverted to last committed state")
            else:
                print_warning(console, "Failed to revert changes")
            ctx.data["loop_broke"] = True
            return BreakLoop()  # should_continue=False (failed)

        # Fixes count toward the summary only when the iteration's tests pass
        # (a failed iteration reverts them above).
        ctx.data["fixes_applied"] += ctx.data["iteration_fixes"]

        # Record the pre-commit SHA so the next iteration reviews this
        # iteration's changes.
        ctx.data["diff_base"] = _get_head_sha(target_dir)
    else:
        # Capture the recommended patch NOW, before tests run and a failure
        # returns early below. Best-effort; never blocks the run.
        git_ops.capture_recommended_patch_with_base(
            target_dir,
            ctx.data["pre_fix_snapshot"],
            ctx.data["pre_fix_head"],
            target_dir / ".daydream" / "recommended.patch",
        )

        async with phase_scope(DaydreamPhase.TEST):
            tests_passed, test_retries = await phase_test_and_heal(
                ctx.backend_for("test"), work, feedback_items=ctx.data["feedback_items"]
            )
        ctx.data["tests_passed"] = tests_passed
        ctx.data["test_retries"] = test_retries
    return None


async def _step_commit_iteration(ctx: FlowContext) -> None:
    """Commit so the next review sees a clean tree."""
    await phase_commit_iteration(ctx.backend_for("fix"), ctx.work, ctx.data["iteration"])


async def _step_loop_exhausted(ctx: FlowContext) -> None:
    """The old ``while...else`` branch: max iterations reached without a break."""
    if ctx.data["loop_broke"]:
        return  # a break (clean review / test failure) skipped the else branch

    config = ctx.config
    feedback_items = ctx.data["feedback_items"]
    if feedback_items:
        # Deduplicate by (file, line, description) to avoid inflated counts
        unique_issues = {
            (item.get("file"), item.get("line"), item.get("description"))
            for item in feedback_items
        }
        print_warning(
            console,
            f"Reached max iterations ({config.max_iterations}), "
            f"{len(unique_issues)} unique issues found across all iterations",
        )
        ctx.data["tests_passed"] = False


async def _step_summary(ctx: FlowContext) -> None:
    """Print the run summary."""
    config = ctx.config
    print_summary(
        console,
        SummaryData(
            skill=ctx.data["skill"] or "N/A",
            target=str(ctx.work.repo),
            feedback_count=len(ctx.data["feedback_items"]),
            fixes_applied=ctx.data["fixes_applied"],
            test_retries=ctx.data["test_retries"],
            tests_passed=ctx.data["tests_passed"],
            loop_mode=config.loop,
            iterations_used=ctx.data["iteration"] if config.loop else 1,
        ),
    )


async def _step_commit_gate(ctx: FlowContext) -> Stop:
    """Exploration cleanup, then the commit gate + review-output cleanup."""
    work = ctx.work
    target_dir = work.repo

    exploration_cleanup = target_dir / ".daydream" / "exploration"
    if exploration_cleanup.is_dir():
        shutil.rmtree(exploration_cleanup)

    # Commit gate. Unattended auto-commits (safe_default=True) so a green run's
    # commit isn't silently dropped on non-TTY stdin; --yes auto-commits, a
    # forced --no skips, an interactive run gets the y/N prompt.
    if ctx.data["tests_passed"]:
        commit_decision = resolve_gate(
            assume=get_assume(),
            interactive=not get_non_interactive(),
            safe_default=True,
        )
        if commit_decision is True:
            await phase_commit_push_auto(ctx.backend_for("review"), work)
        elif commit_decision is None:
            await phase_commit_push(ctx.backend_for("review"), work)
        # commit_decision is False -> forced decline, skip commit.

        if ctx.data["cleanup_enabled"]:
            review_output_path = target_dir / REVIEW_OUTPUT_FILE
            if review_output_path.exists():
                review_output_path.unlink()
                print_success(console, f"Cleaned up {REVIEW_OUTPUT_FILE}")

        return Stop(0)
    else:
        return Stop(1)


def _is_loop_mode(ctx: FlowContext) -> bool:
    return ctx.config.loop


def _review_enabled(ctx: FlowContext) -> bool:
    # Loop mode always reviews; single-pass only when starting at "review".
    return ctx.config.loop or ctx.config.start_at == "review"


def _parse_fix_enabled(ctx: FlowContext) -> bool:
    # VERBATIM start_at encoding: start_at="fix" still runs parse (old body).
    return ctx.config.loop or ctx.config.start_at in ("review", "parse", "fix")


def _iterate_max_iterations(ctx: FlowContext) -> int:
    return ctx.config.max_iterations if ctx.config.loop else 1


STEPS: tuple[FlowStep, ...] = (
    # "exploration" is already a registered phase (the review flow's step, a
    # different body: phase_scope + ctx.data["diff"]); phase names are a
    # single registry namespace, so the shallow variant gets a unique name.
    FlowStep(name="shallow-exploration", run=_step_exploration, config_phase="exploration"),
    FlowStep(name="loop-preflight", run=_step_loop_preflight, enabled=_is_loop_mode),
    FlowStep(name="review", run=_step_review, enabled=_review_enabled),
    FlowStep(name="parse", run=_step_parse, enabled=_parse_fix_enabled),
    FlowStep(name="fix", run=_step_fix, enabled=_parse_fix_enabled),
    FlowStep(name="test", run=_step_test),
    # config_phase "fix" mirrors the old body's use of the fix backend for the commit.
    FlowStep(name="commit-iteration", run=_step_commit_iteration, config_phase="fix", enabled=_is_loop_mode),
    FlowStep(name="loop-exhausted", run=_step_loop_exhausted, enabled=_is_loop_mode),
    FlowStep(name="summary", run=_step_summary),
    # config_phase "review" mirrors the old body's use of the review backend for the commit.
    FlowStep(name="commit-gate", run=_step_commit_gate, config_phase="review"),
)

FLOW: tuple[FlowEntry, ...] = (
    "shallow-exploration",
    "loop-preflight",
    LoopGroup(
        name="iterate",
        steps=("review", "parse", "fix", "test", "commit-iteration"),
        max_iterations=_iterate_max_iterations,
    ),
    "loop-exhausted",
    "summary",
    "commit-gate",
)
