"""Main orchestration logic for the review and fix loop.

The runner is unified around a single :func:`run` entry point. ``run`` opens
the workspace via :func:`daydream.workspace.open_workspace` and then dispatches
to a private helper based on ``config.pr_number`` / ``config.output_mode`` /
``config.shallow``::

    pr_number set            -> _run_pr_feedback (PR feedback flow)
    output_mode == "comment" -> _run_comment    (review + post inline PR comments)
    output_mode == "review"  -> _run_review     (review report only, no posting)
    output_mode == "loop":
        config.shallow       -> _run_loop_shallow (single-stack review-fix-test)
        else                 -> _run_loop_deep    (deep multi-stack pipeline, default)

``run_feedback`` is the entry point used by the ``daydream feedback <pr#>``
subcommand and is a thin wrapper that sets ``pr_number`` and re-enters
:func:`run`.
"""

import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from daydream import git_ops
from daydream.agent import (
    MissingSkillError,
    console,
    set_quiet_mode,
)
from daydream.backends import Backend, create_backend
from daydream.config import (
    DEFAULT_EXPLORATION_MODEL,
    PHASE_DEFAULT_MODELS,
    REVIEW_OUTPUT_FILE,
    REVIEW_SKILLS,
    SKILL_MAP,
    ReviewSkillChoice,
)
from daydream.exploration import ExplorationContext, safe_explore
from daydream.exploration_runner import count_changed_files, pre_scan, select_tier
from daydream.git_ops import GitError
from daydream.phases import (
    FixResult,
    _detect_default_branch,
    _git_branch,
    _git_diff,
    _git_log,
    check_review_file_exists,
    phase_alternative_review,
    phase_commit_iteration,
    phase_commit_push,
    phase_commit_push_auto,
    phase_fetch_pr_feedback,
    phase_fix,
    phase_generate_plan,
    phase_parse_feedback,
    phase_respond_pr_feedback,
    phase_review,
    phase_test_and_heal,
    phase_understand_intent,
    revert_uncommitted_changes,
)
from daydream.trajectory import DaydreamRunFlow, TrajectoryRecorder, default_trajectory_path
from daydream.ui import (
    SummaryData,
    phase_subtitle,
    print_dim,
    print_error,
    print_info,
    print_iteration_divider,
    print_menu,
    print_phase_hero,
    print_skipped_phases,
    print_success,
    print_summary,
    print_warning,
    prompt_user,
)
from daydream.workspace import WorkContext, open_workspace

# Stage 4 — output mode for the consolidated CLI surface. ``loop`` runs the
# full review→fix→test cycle, ``comment`` posts inline PR comments and exits,
# ``review`` writes a report to terminal/markdown and exits.
OutputMode = Literal["loop", "comment", "review"]


@dataclass
class RunConfig:
    """Configuration for a daydream run.

    Attributes:
        target: Target directory path for the review. If None, prompts user.
        skill: Review skill to use ("python", "react", "elixir", "go", "rust",
            or "ios"). If None and shallow, prompts user.
        cleanup: Remove review output file after completion. If None, prompts user.
        quiet: Suppress verbose output from the agent.
        start_at: Phase to start at ("review", "parse", "fix", "test", "ttt",
            "per-stack", or "merge").
        pr_number: GitHub PR number for PR feedback mode. If None, normal mode.
        bot: Bot username whose comments to fetch (e.g. "coderabbitai[bot]").
        backend: Default backend to use ("claude" or "codex"). Default is "claude".
        review_backend: Override backend for the review phase. If None, uses backend.
        fix_backend: Override backend for the fix phase. If None, uses backend.
        test_backend: Override backend for the test phase. If None, uses backend.
        review_model: Model override for the review phase. When None, the
            resolver falls back to ``PHASE_DEFAULT_MODELS[backend_name]["review"]``
            and then to the backend's own default.
        parse_model: Model override for the parse phase. When None, resolves via
            ``PHASE_DEFAULT_MODELS[backend_name]["parse"]`` then backend default.
        fix_model: Model override for the fix phase. When None, resolves via
            ``PHASE_DEFAULT_MODELS[backend_name]["fix"]`` then backend default.
        test_model: Model override for the test phase. When None, resolves via
            ``PHASE_DEFAULT_MODELS[backend_name]["test"]`` then backend default.
        loop: Enable continuous review/fix/test iterations. Default is False.
        max_iterations: Maximum number of loop iterations before exiting. Default is 5.
        exploration_model: Model override for exploration subagents. When set, a separate
            backend is created for the exploration phase using this model. Defaults to
            :data:`config.DEFAULT_EXPLORATION_MODEL`.
        ignore_paths: Paths to exclude from diffs (passed to `git :(exclude)` pathspecs
            and surfaced in review prompts). Default is an empty list.
        trajectory_path: Path to write the ATIF v1.6 trajectory JSON. Default-resolved
            by run flows to ``<target>/.daydream/runs/<session_id>/trajectory.json``
            when None.
        pr_repo: GitHub repository in ``owner/repo`` format. Auto-detected from ``gh``
            in deep (default) mode. Stored in trajectory metadata for eval linkage.
        archive: Archive run artifacts to centralized store. Default True.
        run_eval: Run deterministic evaluation on archived artifacts. Default False.
        branch: Specific branch to review. If None, uses cwd's HEAD.
        base: Base ref to compare against. If None, auto-resolves.
        output_mode: ``"loop"`` (review→fix→test, default), ``"comment"``
            (review + post inline PR comments), or ``"review"`` (review report only).
        force_worktree: Force ephemeral worktree even when ``branch`` is None.
        shallow: Single-stack review (skip multi-stack auto-detection).
        extra_copy: Extra paths to copy into ephemeral worktrees.
        plan: Generate an implementation plan and embed it in PR comments.

    """

    target: str | None = None
    skill: str | None = None  # "python", "react", "elixir", "go", "rust", "ios"
    cleanup: bool | None = None
    quiet: bool = True
    start_at: str = "review"
    pr_number: int | None = None
    bot: str | None = None
    backend: str = "claude"
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
    review_model: str | None = None
    parse_model: str | None = None
    fix_model: str | None = None
    test_model: str | None = None
    loop: bool = False
    max_iterations: int = 5
    exploration_context: ExplorationContext | None = None
    exploration_depth: int = 1
    exploration_model: str | None = None
    ignore_paths: list[str] = field(default_factory=list)
    trajectory_path: Path | None = None
    pr_repo: str | None = None
    archive: bool = True
    run_eval: bool = False

    branch: str | None = None
    base: str | None = None
    output_mode: OutputMode = "loop"
    force_worktree: bool = False
    shallow: bool = False
    extra_copy: list[Path] = field(default_factory=list)
    plan: bool = False


def _print_missing_skill_error(skill_name: str) -> None:
    """Print error message for missing skill with installation instructions."""
    print_error(console, "Missing Skill", f"Skill '{skill_name}' is not available")

    if skill_name.startswith("beagle"):
        print_info(console, "The Beagle plugin is required but not installed or enabled.")
        console.print()
        print_dim(console, "To install Beagle:")
        print_dim(console, "  1. Open Claude Code in your terminal")
        print_dim(console, "  2. Run: /install-plugin beagle@existential-birds")
        print_dim(console, "  3. Restart Claude Code")
        console.print()
        print_dim(console, "Or enable it manually in ~/.claude/settings.json:")
        print_dim(console, '  "enabledPlugins": {')
        print_dim(console, '    "beagle@existential-birds": true')
        print_dim(console, "  }")
    else:
        print_info(console, f"The plugin providing '{skill_name}' is not installed.")
        print_dim(console, "Check your ~/.claude/settings.json for enabled plugins.")

    console.print()


def _make_archive_callback(
    config: RunConfig, target_dir: Path,
) -> Callable[[TrajectoryRecorder, str], None] | None:
    """Build the on_write archive callback, or None if archiving is disabled."""
    if not config.archive:
        return None

    def _cb(recorder: TrajectoryRecorder, status: str) -> None:
        from daydream.archive import archive_run

        archive_run(
            recorder=recorder,
            target_dir=target_dir,
            config=config,
            status=status,
            run_eval=config.run_eval,
        )

    return _cb


def _resolve_backend(
    config: RunConfig,
    phase: str,
    cache: dict[tuple[str, str | None], Backend] | None = None,
) -> Backend:
    """Get or create the backend for a given phase, respecting per-phase overrides.

    Resolution order for the model:
        1. ``getattr(config, f"{phase}_model", None)`` — explicit per-phase flag.
        2. ``PHASE_DEFAULT_MODELS[backend_name].get(phase)`` — per-backend phase
           default table.
        3. ``None`` — let :func:`create_backend` apply its own backend default.

    The backend kind is resolved first via
    ``getattr(config, f"{phase}_backend", None) or config.backend``; the model
    lookup then keys into that backend name's table.

    Args:
        config: Run configuration with backend and per-phase model settings.
        phase: Phase name (e.g. ``"review"``, ``"parse"``, ``"fix"``, ``"test"``,
            ``"intent"``, ``"wonder"``, ``"envision"``, ``"merge"``,
            ``"exploration"``, ``"pr_feedback"``).
        cache: Optional dict to cache backends by ``(backend_name, model)``.
            When provided, backends are reused only when both the backend kind
            and the resolved model match — so the same backend kind with two
            different models yields two distinct instances.

    Returns:
        Backend instance for the phase.

    """
    backend_override = getattr(config, f"{phase}_backend", None)
    backend_name = backend_override or config.backend

    phase_flag = getattr(config, f"{phase}_model", None)
    table_default = PHASE_DEFAULT_MODELS.get(backend_name, {}).get(phase)
    resolved_model = phase_flag or table_default  # ``None`` falls through to backend default

    cache_key = (backend_name, resolved_model)
    if cache is not None:
        if cache_key not in cache:
            cache[cache_key] = create_backend(backend_name, model=resolved_model)
        return cache[cache_key]

    return create_backend(backend_name, model=resolved_model)


def _compute_diff_ref(cwd: Path) -> str:
    """Compute the diff ref to hand to exploration specialists.

    Returns ``"{base_branch}...HEAD"`` when a default branch is detected, else
    falls back to ``"HEAD"`` so specialists can still run ``git diff HEAD -- <file>``.
    """
    base_branch = _detect_default_branch(cwd)
    if base_branch:
        return f"{base_branch}...HEAD"
    return "HEAD"


def _get_head_sha(cwd: Path) -> str | None:
    """Get the current HEAD commit SHA.

    Returns:
        The full SHA string, or None if the command fails.

    """
    try:
        return git_ops.head_sha(cwd)
    except GitError:
        return None


# --- Public entry points ----------------------------------------------------


async def run(config: RunConfig | None = None) -> int:
    """Execute a daydream run end-to-end.

    Opens the workspace via :func:`open_workspace` and dispatches to the
    appropriate flow helper based on ``config.pr_number`` / ``config.output_mode``
    / ``config.shallow``. Centralising workspace lifecycle means every flow gets
    a real :class:`WorkContext` (in-place or ephemeral) with consistent
    base/branch resolution.

    Args:
        config: Optional configuration. Defaults to a fresh :class:`RunConfig`
            (interactive prompts for target dir, skill, cleanup).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    if config is None:
        config = RunConfig()

    print_phase_hero(console, "DAYDREAM", phase_subtitle("DAYDREAM"))

    # Resolve target directory (from config or prompt). Done outside the
    # workspace context manager so that path-validation errors short-circuit
    # before we incur any git work.
    if config.target is not None:
        target_dir = Path(config.target).resolve()
    else:
        target_input = prompt_user(console, "Enter target directory", ".")
        target_dir = Path(target_input).resolve()

    if not target_dir.is_dir():
        print_error(console, "Invalid Path", f"'{target_dir}' is not a valid directory")
        return 1

    # Quiet mode tweak: Codex backends need their shell output visible because
    # those commands ARE the user-facing signal. Done before any backend is
    # constructed so per-phase backends inherit the right setting.
    quiet = config.quiet
    if quiet:
        codex_in_use = config.backend == "codex" or any(
            b == "codex"
            for b in (config.review_backend, config.fix_backend, config.test_backend)
            if b is not None
        )
        if codex_in_use:
            quiet = False
    set_quiet_mode(quiet)

    # ``--comment`` and ``--review`` skip the test phase, so they also skip
    # the .env copy mechanism in ephemeral mode (workspace.copy_files_into_ephemeral).
    skip_tests = config.output_mode != "loop"

    # Stage 4.2: removed the silent fallback to ``make_in_place_workcontext``.
    # ``open_workspace`` runs ``assert_is_worktree`` and surfaces
    # ``NotAWorktreeError`` (a ``GitError``) caught below, so the user sees a
    # loud error instead of a confusing "no diff found" downstream. The
    # ``WrongBranchError`` check lives in ``_dispatch`` (loop modes only) and
    # propagates up to :func:`daydream.cli.main` for the user-facing message.
    try:
        async with open_workspace(
            source=target_dir,
            branch=config.branch,
            base=config.base,
            force_ephemeral=config.force_worktree,
            extra_copy=config.extra_copy,
            skip_tests=skip_tests,
        ) as work:
            return await _dispatch(work, config)
    except git_ops.WrongBranchError:
        # Propagate to ``cli.main`` for the actionable error panel.
        raise
    except git_ops.GitError as exc:
        print_error(console, "Workspace Error", str(exc))
        return 1


async def run_feedback(config: RunConfig, pr: int) -> int:
    """Entry point for the ``daydream feedback <pr#>`` subcommand.

    Sets ``config.pr_number`` and re-enters :func:`run` so the dispatch
    routes to :func:`_run_pr_feedback`. Kept as a thin wrapper so cli.py
    has a single named entry point per invocation shape.

    Args:
        config: Run configuration populated by the feedback subparser.
        pr: PR number to ingest.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    config.pr_number = pr
    return await run(config)


# --- Dispatch ---------------------------------------------------------------


async def _dispatch(work: WorkContext, config: RunConfig) -> int:
    """Pick the flow helper for the resolved workspace + config.

    Order matters: ``pr_number`` overrides everything (set by the
    ``daydream feedback <pr#>`` subcommand). Then output_mode picks comment vs
    review vs loop. Inside loop, ``config.shallow`` selects the single-stack
    pipeline; otherwise the deep multi-stack pipeline runs (default).
    """
    if config.pr_number is not None:
        return await _run_pr_feedback(work, config)

    if config.output_mode == "comment":
        return await _run_comment(work, config)

    if config.output_mode == "review":
        return await _run_review(work, config)

    # output_mode == "loop"
    # Stage 4.2 — guard against the silent-failure case where the user runs
    # ``daydream`` from a worktree that's checked out to the base branch with
    # no ``--branch`` and no ``--worktree``. There's nothing to review against
    # itself; raise loudly so cli.main() (and ``run()``'s except clause) can
    # render the actionable message.
    if (
        config.branch is None
        and not config.force_worktree
        and work.head_branch is not None
        and work.head_branch == work.base_branch
    ):
        raise git_ops.WrongBranchError(
            f"cwd is on the base branch {work.base_branch!r} -- "
            "there's nothing to review against itself.\n"
            "Either:\n"
            f"  - check out a feature branch in this worktree and re-run, or\n"
            f"  - run with --branch <feature-branch> to review the server's version, or\n"
            f"  - run with --worktree to force ephemeral isolation."
        )

    if config.shallow:
        return await _run_loop_shallow(work, config)
    # Default: deep multi-stack pipeline. Pass ``--shallow`` to opt into the
    # single-stack flow.
    return await _run_loop_deep(work, config)


# --- Helper: PR feedback flow ----------------------------------------------


async def _run_pr_feedback(work: WorkContext, config: RunConfig) -> int:
    """Today's PR feedback body, refactored to receive ``work`` from the dispatch.

    Fetches bot review comments, parses them, applies fixes one-by-one,
    commits/pushes, and posts a "fixed" reply on each addressed comment.
    """
    if config.pr_number is None or config.bot is None:
        print_error(
            console,
            "Invalid PR config",
            "PR number and --bot are required (use: daydream feedback <pr#> --bot <name>).",
        )
        return 1

    pr_number = config.pr_number
    bot = config.bot
    target_dir = work.repo

    backend_cache: dict[tuple[str, str | None], Backend] = {}
    review_backend = _resolve_backend(config, "review", backend_cache)
    fix_backend = _resolve_backend(config, "fix", backend_cache)

    session_id = str(uuid.uuid4())
    trajectory_path = config.trajectory_path or default_trajectory_path(target_dir, session_id)
    async with TrajectoryRecorder(
        path=trajectory_path,
        run_flow=DaydreamRunFlow.PR,
        target_dir=target_dir,
        agent_model_name="",
        session_id=session_id,
        explicit_path=config.trajectory_path is not None,
        pr_number=config.pr_number,
        pr_repo=config.pr_repo,
        on_write=_make_archive_callback(config, target_dir),
    ):
        console.print()
        print_info(console, f"PR feedback mode: PR #{pr_number}")
        print_info(console, f"Bot: {bot}")
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Model: {review_backend.model}")
        console.print()

        # Phase 1: Fetch PR feedback
        await phase_fetch_pr_feedback(review_backend, work, pr_number, bot)

        # Phase 2: Parse feedback (reused from normal flow)
        try:
            feedback_items = await phase_parse_feedback(review_backend, work)
        except ValueError:
            print_error(console, "Parse Failed", "Failed to parse PR feedback. Exiting.")
            return 1

        if not feedback_items:
            print_info(console, "No actionable feedback found in PR comments.")
            return 0

        # Phase 3: Fix issues sequentially to avoid concurrent access to a
        # single mutable backend instance.
        results: list[FixResult] = []
        total_items = len(feedback_items)
        for idx, item in enumerate(feedback_items, start=1):
            try:
                await phase_fix(fix_backend, work, item, idx, total_items)
                results.append((item, True, None))
            except Exception as e:
                results.append((item, False, f"{type(e).__name__}: {e}"))

        # If all fixes failed, abort
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]

        if not successful:
            print_error(
                console,
                "All Fixes Failed",
                f"All {len(failed)} fix(es) failed. Aborting before commit.",
            )
            return 1

        # Phase 4: Commit and push (no user prompt)
        try:
            await phase_commit_push_auto(
                review_backend, work, items=[item for item, _ok, _err in results if _ok],
            )
        except Exception as e:
            print_error(console, "Commit/Push Failed", str(e))
            return 1

        # Phase 5: Respond to PR comments
        try:
            await phase_respond_pr_feedback(review_backend, work, pr_number, bot, results)
        except Exception as e:
            print_warning(console, f"Failed to respond to PR comments: {e}")
            print_info(console, "Fixes were already pushed successfully.")

        # Summary
        console.print()
        print_success(
            console,
            f"PR #{pr_number}: {len(successful)} fix(es) applied"
            + (f", {len(failed)} failed" if failed else ""),
        )

        return 0


# --- Helper: comment mode (--comment) --------------------------------------


async def _run_comment(work: WorkContext, config: RunConfig) -> int:
    """Review + post inline PR comments + exit.

    Pre-flight: when a branch is explicitly requested but no open PR exists
    for it, refuse early with an actionable error rather than running a
    review that has nowhere to land.
    """
    if config.branch is not None:
        try:
            prs = git_ops.gh_pr_list_for_branch(work.source, config.branch)
        except GitError as exc:
            print_error(console, "GitHub Error", str(exc))
            return 1
        if not prs:
            print_error(
                console,
                "No Open PR",
                f"no open PR for branch {config.branch} — push first or use --review",
            )
            return 1

    return await _run_review_or_comment(work, config, post_to_pr=True)


# --- Helper: review mode (--review) ----------------------------------------


async def _run_review(work: WorkContext, config: RunConfig) -> int:
    """Review + write a report and exit. No PR posting, no fix, no test."""
    return await _run_review_or_comment(work, config, post_to_pr=False)


async def _run_review_or_comment(
    work: WorkContext, config: RunConfig, *, post_to_pr: bool,
) -> int:
    """Shared body for ``--comment`` and ``--review``.

    Lifted from today's ``run_trust`` body. The only difference between the
    two modes is whether the alternative-review issues are posted to the PR
    via :func:`daydream.pr_review.post_review_to_pr_from_alt_issues`.
    """
    backend = _resolve_backend(config, "review")
    target_dir = work.repo

    # Gather git context using the resolved base branch from work (no
    # double-detection — base resolution is locked at workspace open time).
    try:
        diff = git_ops.diff(work.repo, work.base_branch, exclude=config.ignore_paths)
    except GitError:
        diff = None
    log = _git_log(target_dir)
    branch = work.head_branch or _git_branch(target_dir)

    if diff is None:
        print_error(console, "Git Error", "Unable to determine base branch for diff")
        return 1
    if not diff.strip():
        print_warning(console, "No diff found — nothing to review")
        return 0

    # Write diff to file to avoid exceeding prompt size limits
    daydream_dir = target_dir / ".daydream"
    daydream_dir.mkdir(exist_ok=True)
    diff_path = daydream_dir / "diff.patch"
    diff_path.write_text(diff)

    flow = DaydreamRunFlow.TTT
    session_id = str(uuid.uuid4())
    trajectory_path = config.trajectory_path or default_trajectory_path(target_dir, session_id)
    async with TrajectoryRecorder(
        path=trajectory_path,
        run_flow=flow,
        target_dir=target_dir,
        agent_model_name="",
        session_id=session_id,
        explicit_path=config.trajectory_path is not None,
        pr_number=config.pr_number,
        pr_repo=config.pr_repo,
        on_write=_make_archive_callback(config, target_dir),
    ):
        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Branch: {branch}")
        print_info(console, f"Model: {backend.model}")
        console.print()

        # Pre-scan exploration: populate config.exploration_context before phase 1.
        # Skip when already pre-populated (e.g. injected by caller or tests).
        if config.exploration_context is None:
            tier = select_tier(count_changed_files(diff or ""))
            if tier == "skip":
                print_dim(console, "Skipping exploration -- trivial diff")
                config.exploration_context = ExplorationContext()
            else:
                print_phase_hero(console, "EXPLORE", phase_subtitle("EXPLORE"))
                explore_model = config.exploration_model or DEFAULT_EXPLORATION_MODEL
                explore_backend = create_backend(config.backend, model=explore_model)
                print_dim(console, f"Exploration model: {explore_backend.model}")
                config.exploration_context = await safe_explore(
                    pre_scan,
                    explore_backend,
                    target_dir,
                    diff,
                    config.exploration_depth,
                    diff_ref=_compute_diff_ref(target_dir),
                )

        # Materialise exploration to disk so phase prompts can reference files.
        exploration_dir: Path | None = None
        if config.exploration_context is not None:
            exploration_dir = config.exploration_context.write_to_dir(daydream_dir / "exploration")

        # Phase 1: Understand intent
        intent_summary = await phase_understand_intent(
            backend, work, diff_path, log, branch,
            exploration_dir=exploration_dir,
        )

        # Phase 2: Alternative review
        issues = await phase_alternative_review(
            backend, work, diff_path, intent_summary,
            exploration_dir=exploration_dir,
        )

        if not issues:
            print_success(console, "No issues found — the implementation looks good!")
            return 0

        # Phase 3: Generate plan.
        # For ``--comment`` mode, ENVISION is skipped by default (extra
        # latency for a prompt-only flow). ``--plan`` opts back in and
        # feeds the structured plan into the PR comment prompt.
        plan_data: dict[str, Any] | None = None
        skip_plan = post_to_pr and not config.plan
        try:
            if not skip_plan:
                _, plan_data = await phase_generate_plan(
                    backend, work, diff_path, intent_summary, issues,
                    exploration_dir=exploration_dir,
                    auto_select_all=post_to_pr,
                )
        finally:
            exploration_cleanup = target_dir / ".daydream" / "exploration"
            if exploration_cleanup.is_dir():
                shutil.rmtree(exploration_cleanup)

        # Optionally post findings as inline PR review comments. Only the
        # ``--comment`` path enters this branch; ``--review`` exits with the
        # plan written to disk.
        if post_to_pr:
            from daydream.pr_review import post_review_to_pr_from_alt_issues

            await post_review_to_pr_from_alt_issues(
                target_dir, issues, console=console, plan_data=plan_data,
            )

        return 0


# --- Helper: shallow loop (single-stack review-fix-test) -------------------


async def _run_loop_shallow(work: WorkContext, config: RunConfig) -> int:
    """Single-stack review → fix → test → loop body.

    This is today's ``run`` body lifted into a helper. The workspace
    bootstrapping and target-dir resolution have moved up to :func:`run`;
    everything else is unchanged.
    """
    target_dir = work.repo

    # Resolve skill only when the review phase will run.
    skill: str | None = None
    if config.start_at == "review":
        if config.skill is not None:
            if config.skill in SKILL_MAP:
                skill = SKILL_MAP[config.skill]
            elif config.skill in REVIEW_SKILLS.values():
                skill = config.skill
            else:
                print_error(console, "Invalid Skill", f"'{config.skill}' is not a valid skill")
                return 1
        else:
            console.print()
            print_menu(console, "Select review skill", [
                ("1", "Python/FastAPI backend (review-python)"),
                ("2", "React/TypeScript (review-frontend)"),
                ("3", "Elixir/Phoenix (review-elixir)"),
                ("4", "Go backend (review-go)"),
                ("5", "Rust (review-rust)"),
                ("6", "iOS/SwiftUI (review-ios)"),
            ])

            skill_choice = prompt_user(console, "Choice", "1")

            try:
                skill_enum = ReviewSkillChoice(skill_choice)
            except ValueError:
                print_error(console, "Invalid Choice", f"'{skill_choice}' is not a valid option")
                return 1

            skill = REVIEW_SKILLS[skill_enum]

    # Early validation: check review file exists when starting at parse or fix.
    if config.start_at in ("parse", "fix"):
        try:
            check_review_file_exists(target_dir)
        except FileNotFoundError as e:
            print_error(console, "Missing Review File", str(e))
            return 1

    # Cleanup setting (from config or prompt).
    if config.cleanup is not None:
        cleanup_enabled = config.cleanup
    else:
        cleanup_response = prompt_user(console, "Cleanup review output after completion? [y/N]", "n")
        cleanup_enabled = cleanup_response.lower() in ("y", "yes")

    # Backends (per-phase overrides).
    backend_cache: dict[tuple[str, str | None], Backend] = {}
    review_backend = _resolve_backend(config, "review", backend_cache)
    fix_backend = _resolve_backend(config, "fix", backend_cache)
    test_backend = _resolve_backend(config, "test", backend_cache)

    session_id = str(uuid.uuid4())
    trajectory_path = config.trajectory_path or default_trajectory_path(target_dir, session_id)
    async with TrajectoryRecorder(
        path=trajectory_path,
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=target_dir,
        agent_model_name="",
        session_id=session_id,
        explicit_path=config.trajectory_path is not None,
        pr_number=config.pr_number,
        pr_repo=config.pr_repo,
        on_write=_make_archive_callback(config, target_dir),
    ):
        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Model: {review_backend.model}")
        if skill:
            print_info(console, f"Review skill: {skill}")
        if config.start_at != "review":
            print_skipped_phases(console, config.start_at)
        console.print()

        # Pre-scan exploration: populate config.exploration_context before
        # the first phase_review() call. Only runs when starting at "review"
        # (later start phases skip review, so exploration would be wasted).
        if config.start_at == "review" and config.exploration_context is None:
            diff_text = _git_diff(target_dir, exclude=config.ignore_paths) or ""
            tier = select_tier(count_changed_files(diff_text))
            if tier == "skip":
                print_dim(console, "Skipping exploration -- trivial diff")
                config.exploration_context = ExplorationContext()
            else:
                print_phase_hero(console, "EXPLORE", phase_subtitle("EXPLORE"))
                explore_model = config.exploration_model or DEFAULT_EXPLORATION_MODEL
                explore_backend = create_backend(config.backend, model=explore_model)
                print_dim(console, f"Exploration model: {explore_backend.model}")
                config.exploration_context = await safe_explore(
                    pre_scan,
                    explore_backend,
                    target_dir,
                    diff_text,
                    config.exploration_depth,
                    diff_ref=_compute_diff_ref(target_dir),
                )

        feedback_items: list[dict[str, Any]] = []
        fixes_applied = 0
        test_retries = 0
        tests_passed = True
        iteration = 0
        diff_base: str | None = None
        exploration_dir: Path | None = None

        async def _run_loop_iteration() -> tuple[list[dict[str, Any]], int, int, bool, bool]:
            """Execute one iteration of the review-parse-fix-test loop.

            Returns:
                Tuple of (items, fixes_count, retries, tests_passed, should_continue).
                should_continue is False if the loop should break (clean review or test failure).
            """
            nonlocal diff_base

            if iteration > 1:
                (target_dir / REVIEW_OUTPUT_FILE).unlink(missing_ok=True)
                print_iteration_divider(console, iteration, config.max_iterations)

            # Phase 1: Review
            assert skill is not None, "skill must be set when starting at review phase"
            await phase_review(
                review_backend, work, skill, diff_base=diff_base,
                exploration_dir=exploration_dir,
                exclude=config.ignore_paths,
            )

            # Phase 2: Parse feedback
            items = await phase_parse_feedback(review_backend, work)

            if not items:
                print_info(console, f"Clean review on iteration {iteration}")
                return [], 0, 0, True, False  # should_continue=False (clean)

            # Phase 3: Fix
            print_phase_hero(console, "HEAL", phase_subtitle("HEAL"))
            fixes_count = 0
            for i, item in enumerate(items, 1):
                await phase_fix(fix_backend, work, item, i, len(items))
                fixes_count += 1

            # Phase 4: Test
            passed, retries = await phase_test_and_heal(test_backend, work, feedback_items=items)

            if not passed:
                print_warning(console, f"Tests failed on iteration {iteration}, reverting changes")
                if revert_uncommitted_changes(target_dir):
                    print_info(console, "Reverted to last committed state")
                else:
                    print_warning(console, "Failed to revert changes")
                return items, fixes_count, retries, False, False  # should_continue=False (failed)

            # Record the pre-commit SHA so the next iteration reviews
            # the changes introduced by this iteration's commit.
            diff_base = _get_head_sha(target_dir)

            # Commit iteration changes so the next review sees a clean tree
            await phase_commit_iteration(fix_backend, work, iteration)

            return items, fixes_count, retries, True, True  # should_continue=True

        if config.loop:
            # Guard: loop mode reverts uncommitted changes on failure,
            # so refuse to start if the working tree is dirty.
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
                return 1

            # Materialise exploration to disk so phase prompts can reference files.
            if config.exploration_context is not None:
                exp_parent = target_dir / ".daydream"
                exp_parent.mkdir(exist_ok=True)
                exploration_dir = config.exploration_context.write_to_dir(exp_parent / "exploration")

            # --- Loop mode: repeat review-parse-fix-test ---
            while iteration < config.max_iterations:
                iteration += 1

                try:
                    items, fixes_count, retries, passed, should_continue = await _run_loop_iteration()
                except MissingSkillError as e:
                    _print_missing_skill_error(e.skill_name)
                    return 1
                except ValueError as exc:
                    print_error(console, "Phase 2 Error", str(exc))
                    print_error(console, "Parse Failed", "Failed to parse feedback. Exiting.")
                    return 1

                feedback_items.extend(items)
                test_retries += retries
                tests_passed = passed
                if passed:
                    fixes_applied += fixes_count

                if not should_continue:
                    break

            else:
                # while loop exhausted without break — max iterations reached
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
                    # Mark as failed when max iterations reached with unresolved issues
                    tests_passed = False

        else:
            # --- Single-pass mode (existing behavior) ---

            # Materialise exploration to disk so phase prompts can reference files.
            if config.exploration_context is not None:
                exp_parent = target_dir / ".daydream"
                exp_parent.mkdir(exist_ok=True)
                exploration_dir = config.exploration_context.write_to_dir(exp_parent / "exploration")

            # Phase 1: Review
            if config.start_at == "review":
                assert skill is not None, "skill must be set when starting at review phase"
                try:
                    await phase_review(
                        review_backend, work, skill,
                        exploration_dir=exploration_dir,
                        exclude=config.ignore_paths,
                    )
                except MissingSkillError as e:
                    _print_missing_skill_error(e.skill_name)
                    return 1

            # Phase 2: Parse feedback
            if config.start_at in ("review", "parse", "fix"):
                try:
                    feedback_items = await phase_parse_feedback(review_backend, work)
                except ValueError as exc:
                    print_error(console, "Phase 2 Error", str(exc))
                    print_error(console, "Parse Failed", "Failed to parse feedback. Exiting.")
                    return 1

            # Phase 3: Fix
            if config.start_at in ("review", "parse", "fix"):
                if feedback_items:
                    print_phase_hero(console, "HEAL", phase_subtitle("HEAL"))
                    for i, item in enumerate(feedback_items, 1):
                        await phase_fix(fix_backend, work, item, i, len(feedback_items))
                        fixes_applied += 1
                else:
                    print_info(console, "No feedback items found, skipping fix phase")

            # Phase 4: Test
            tests_passed, test_retries = await phase_test_and_heal(
                test_backend, work, feedback_items=feedback_items
            )
            iteration = 1

        # Print summary
        print_summary(
            console,
            SummaryData(
                skill=skill or "N/A",
                target=str(target_dir),
                feedback_count=len(feedback_items),
                fixes_applied=fixes_applied,
                test_retries=test_retries,
                tests_passed=tests_passed,
                loop_mode=config.loop,
                iterations_used=iteration if config.loop else 1,
            ),
        )

        # Clean up exploration files before exit
        exploration_cleanup = target_dir / ".daydream" / "exploration"
        if exploration_cleanup.is_dir():
            shutil.rmtree(exploration_cleanup)

        # Commit if tests passed
        if tests_passed:
            await phase_commit_push(review_backend, work)

            if cleanup_enabled:
                review_output_path = target_dir / REVIEW_OUTPUT_FILE
                if review_output_path.exists():
                    review_output_path.unlink()
                    print_success(console, f"Cleaned up {REVIEW_OUTPUT_FILE}")

            return 0
        else:
            return 1


# --- Helper: deep loop (multi-stack pipeline) ------------------------------


async def _run_loop_deep(work: WorkContext, config: RunConfig) -> int:
    """Delegate to the deep-mode orchestrator."""
    from daydream.deep.orchestrator import run_deep

    return await run_deep(config, work)
