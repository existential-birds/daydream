"""CLI entry point for daydream."""

import argparse
import signal
import sys
import warnings
from pathlib import Path

import anyio

from daydream import git_ops
from daydream.agent import (
    console,
    get_current_backends,
    set_shutdown_requested,
)
from daydream.runner import RunConfig, run, run_feedback
from daydream.trajectory import get_signal_recorder
from daydream.ui import (
    ShutdownPanel,
    get_shutdown_panel,
    print_error,
    set_shutdown_panel,
)


def _signal_handler(signum: int, frame: object) -> None:
    """Handle termination signals: flush partial trajectory then request shutdown.

    D-07: SIGINT/SIGTERM flushes a ``<path>.partial`` trajectory with
    ``extra.partial=true`` so consumers know the run was interrupted.

    Uses :func:`get_signal_recorder` (a module-level stack) rather than the
    ContextVar. Signal handlers fire in the main thread at bytecode boundaries
    and are not synced with the asyncio task context where the ContextVar was
    set, so ContextVar reads from here are non-deterministic.
    """
    signal_name = signal.Signals(signum).name
    set_shutdown_requested(True)

    # Flush partial trajectory before tearing down (D-07). write_partial is
    # synchronous and exception-safe — it cannot crash the shutdown path even
    # if the disk is full or path is unwritable.
    recorder = get_signal_recorder()
    if recorder is not None:
        recorder.write_partial()

    panel = ShutdownPanel(console)
    set_shutdown_panel(panel)
    panel.start(f"Received {signal_name}, shutting down")

    if get_current_backends():
        panel.add_step("Terminating running agent(s)...")

    raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    """Install signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _auto_detect_pr_number() -> int | None:
    """Auto-detect PR number from the current branch via gh CLI.

    Returns:
        The PR number if found, or None if detection fails.

    """
    try:
        data = git_ops.gh_pr_view(Path.cwd(), None)
    except git_ops.GitError:
        return None
    if not data:
        return None
    number = data.get("number")
    return int(number) if isinstance(number, int) else None


def _detect_repo_slug() -> str | None:
    """Detect the GitHub owner/repo slug for the current repository via gh CLI.

    Returns:
        String like ``"owner/repo"``, or None if detection fails.
    """
    try:
        slug = git_ops.gh_repo_view(Path.cwd())
    except git_ops.GitError:
        return None
    if slug is None:
        return None
    owner, name = slug
    return f"{owner}/{name}"


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared (non-output-mode) arguments to a parser or subparser.

    Used by both the top-level parser and the ``feedback`` subparser so flags
    like ``--backend``, ``--model``, ``--trajectory`` work in both places.
    """
    parser.add_argument(
        "--trajectory",
        default=None,
        metavar="PATH",
        type=Path,
        dest="trajectory_path",
        help=(
            "Write ATIF v1.6 trajectory JSON to this path "
            "(default: <target>/.daydream/runs/<session_id>/trajectory.json)"
        ),
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        default=False,
        dest="no_archive",
        help="Disable automatic archival to ~/.daydream/archive/",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        dest="run_eval",
        help="Run deterministic evaluation analysis and store evaluation.json in archive",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["claude", "codex"],
        default="claude",
        help="Agent backend: claude (default) or codex",
    )
    parser.add_argument(
        "--review-backend",
        choices=["claude", "codex"],
        default=None,
        help="Override backend for review phase",
    )
    parser.add_argument(
        "--fix-backend",
        choices=["claude", "codex"],
        default=None,
        help="Override backend for fix phase",
    )
    parser.add_argument(
        "--test-backend",
        choices=["claude", "codex"],
        default=None,
        help="Override backend for test phase",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default: backend-specific). Examples: opus, sonnet, haiku, gpt-5.3-codex",
    )
    parser.add_argument(
        "--exploration-model",
        default=None,
        dest="exploration_model",
        help=(
            "Model for exploration subagents (default: claude-sonnet-4-6). "
            "Use a smaller model to save cost."
        ),
    )
    parser.add_argument(
        "--review-model",
        default=None,
        type=str,
        dest="review_model",
        metavar="MODEL",
        help="Override model for the REVIEW phase (default: per-backend table; see README).",
    )
    parser.add_argument(
        "--parse-model",
        default=None,
        type=str,
        dest="parse_model",
        metavar="MODEL",
        help="Override model for the PARSE phase (default: per-backend table; see README).",
    )
    parser.add_argument(
        "--fix-model",
        default=None,
        type=str,
        dest="fix_model",
        metavar="MODEL",
        help="Override model for the FIX phase (default: per-backend table; see README).",
    )
    parser.add_argument(
        "--test-model",
        default=None,
        type=str,
        dest="test_model",
        metavar="MODEL",
        help="Override model for the TEST phase (default: per-backend table; see README).",
    )


def _build_summarize_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream summarize <path>``.

    Like the ``feedback`` subcommand, ``summarize`` is dispatched manually
    from ``main()`` before the main parser runs so its positional argument
    doesn't collide with the top-level ``TARGET``.
    """
    parser = argparse.ArgumentParser(
        prog="daydream summarize",
        description=(
            "Print run-info markdown (rollup + per-phase breakdown table) "
            "for a trajectory file or run directory."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        metavar="PATH",
        help=(
            "Either a trajectory JSON file or a run directory containing "
            "trajectory.json (and optional trajectories/ siblings)."
        ),
    )
    return parser


def _run_summarize(args: argparse.Namespace) -> int:
    """Dispatch ``daydream summarize`` to the summarize module."""
    from daydream.summarize import summarize

    return summarize(args.path)


def _build_feedback_parser() -> argparse.ArgumentParser:
    """Build the parser for the ``daydream feedback <pr#>`` subcommand.

    Kept as its own parser (not an argparse subparser of the main one) so that
    the main parser's positional ``TARGET`` doesn't collide with the subcommand
    name. We dispatch to this parser from ``_parse_args`` based on argv[0].
    """
    parser = argparse.ArgumentParser(
        prog="daydream feedback",
        description="Fetch bot review comments on a PR, apply fixes, push, and respond.",
    )
    parser.add_argument(
        "pr_number",
        type=int,
        metavar="PR",
        help="Pull request number to process",
    )
    parser.add_argument(
        "--bot",
        required=True,
        metavar="BOT_NAME",
        help="Bot username to filter PR comments (e.g. coderabbitai[bot])",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        metavar="TARGET",
        help="Target directory (default: current directory)",
    )
    _add_shared_arguments(parser)
    return parser


def _build_main_parser() -> argparse.ArgumentParser:
    """Build the main argparse parser for the consolidated CLI surface."""
    parser = argparse.ArgumentParser(
        prog="daydream",
        description="Automated code review and fix loop. "
                    "Use `daydream feedback <pr#>` to process PR bot comments.",
    )

    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        metavar="TARGET",
        help="Target directory (default: prompt interactively). "
             "Use `daydream feedback <pr#>` for the PR feedback flow.",
    )

    # ---- Output mode (mutually exclusive; default = fix-loop) ----
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--comment",
        action="store_true",
        default=False,
        dest="comment",
        help="Review and post inline PR comments, then exit (no fix, no test).",
    )
    output_group.add_argument(
        "--review",
        action="store_true",
        default=False,
        dest="review",
        help="Review and write a report to terminal/markdown, then exit.",
    )

    # ---- Selection ----
    parser.add_argument(
        "--branch",
        default=None,
        metavar="BRANCH",
        help="Branch to review (default: cwd's local HEAD).",
    )
    parser.add_argument(
        "--base",
        default=None,
        metavar="BASE",
        help="Base ref to compare against (default: PR base if any, else origin/HEAD).",
    )

    # ---- Modifiers ----
    parser.add_argument(
        "--worktree",
        action="store_true",
        default=False,
        dest="force_worktree",
        help="Force ephemeral worktree even when --branch is omitted.",
    )
    parser.add_argument(
        "--shallow",
        action="store_true",
        default=False,
        dest="shallow",
        help="Single-stack review (skip multi-stack auto-detection).",
    )
    parser.add_argument(
        "--copy",
        action="append",
        default=[],
        metavar="PATH",
        dest="extra_copy",
        type=Path,
        help="Extra path to copy into ephemeral worktree (repeatable).",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        default=False,
        dest="plan",
        help="Generate an implementation plan and embed it in PR comments (use with --comment).",
    )

    # ---- Skill selection (overrides auto-detect) ----
    parser.add_argument(
        "-s", "--skill",
        choices=["python", "react", "elixir", "go", "rust", "ios"],
        default=None,
        help="Force a specific review skill (default: auto-detect from changed files)",
    )

    # ---- Cleanup / phase resume ----
    cleanup_group = parser.add_mutually_exclusive_group()
    cleanup_group.add_argument(
        "--cleanup",
        action="store_true",
        default=None,
        dest="cleanup",
        help="Cleanup review output after completion",
    )
    cleanup_group.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Keep review output after completion",
    )
    parser.add_argument(
        "--start-at",
        choices=["review", "parse", "fix", "test", "ttt", "per-stack", "merge"],
        default="review",
        dest="start_at",
        help=(
            "Start at a specific phase (default: review). "
            "Choices: review | parse | fix | test | ttt | per-stack | merge. "
            "ttt, per-stack, and merge are valid only in deep (non-shallow) mode."
        ),
    )

    parser.add_argument(
        "--ignore-path",
        action="append",
        default=[],
        metavar="PATH",
        dest="ignore_paths",
        help="Exclude path from diff (repeatable, e.g. --ignore-path .planning --ignore-path vendor)",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Repeat review-fix-test cycle until zero issues or max iterations",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        metavar="N",
        dest="max_iterations",
        help="Maximum loop iterations (default: 5, only meaningful with --loop)",
    )

    _add_shared_arguments(parser)

    return parser


def _parse_args(argv: list[str] | None = None) -> RunConfig:
    """Parse command line arguments and return a RunConfig.

    Implements the consolidated CLI surface: a positional ``target`` directory,
    the ``feedback`` subcommand, output-mode flags (``--comment`` / ``--review``),
    selection flags (``--branch`` / ``--base``), and modifiers (``--worktree`` /
    ``--shallow`` / ``--copy``). Deep is the default; ``--shallow`` opts into
    single-stack mode.

    Args:
        argv: Optional list of arguments. Defaults to ``sys.argv[1:]`` when None.

    Returns:
        RunConfig: Configuration object populated from command line arguments.

    """
    raw_argv = sys.argv[1:] if argv is None else list(argv)

    # Manual subcommand dispatch: argparse subparsers eat the first positional
    # which conflicts with our positional TARGET. So we pop "feedback" off the
    # front ourselves and route to a dedicated parser.
    if raw_argv and raw_argv[0] == "feedback":
        feedback_parser = _build_feedback_parser()
        feedback_args = feedback_parser.parse_args(raw_argv[1:])
        return _build_feedback_config(feedback_args)

    # ``summarize`` is dispatched in main() before _parse_args is called, so
    # we never reach this branch from the summarize path. Kept here only as
    # a guard for callers that hand crafted-argv to _parse_args directly.

    parser = _build_main_parser()
    args = parser.parse_args(raw_argv)

    # ----- Reject purely numeric TARGET (likely meant `daydream feedback N`) -----
    if args.target is not None and args.target.lstrip("-").isdigit():
        parser.error(
            f"target '{args.target}' looks like a PR number — "
            f"did you mean: daydream feedback {args.target}?"
        )

    # ----- Resolve output mode -----
    output_mode: str = "loop"
    if args.comment:
        output_mode = "comment"
    elif args.review:
        output_mode = "review"

    # ttt/per-stack/merge are deep-pipeline resume stages; they don't apply
    # to shallow runs.
    if args.shallow and args.start_at in ("ttt", "per-stack", "merge"):
        parser.error(f"--start-at {args.start_at} is not valid with --shallow")

    # parse and test resume points are ambiguous under deep (default) mode —
    # the deep pipeline has two parse points and no single test phase.
    if not args.shallow and args.start_at in ("parse", "test"):
        parser.error(
            f"--start-at {args.start_at} is not supported in deep mode "
            "(use --shallow, or --start-at fix to resume after the merged report)"
        )

    # Validate --loop incompatibilities
    if args.loop and output_mode != "loop":
        parser.error("--loop cannot be combined with --review/--comment")
    if args.loop and args.start_at != "review":
        parser.error("--loop requires starting at review phase (incompatible with --start-at)")

    # Warn if --max-iterations without --loop
    if args.max_iterations != 5 and not args.loop:
        warnings.warn("--max-iterations has no effect without --loop", stacklevel=2)

    # Detect repo slug for trajectory metadata in deep (default) mode
    pr_repo: str | None = None
    if not args.shallow:
        pr_repo = _detect_repo_slug()

    return RunConfig(
        target=args.target,
        skill=args.skill,
        model=args.model,
        exploration_model=args.exploration_model,
        review_model=args.review_model,
        parse_model=args.parse_model,
        fix_model=args.fix_model,
        test_model=args.test_model,
        cleanup=args.cleanup,
        quiet=True,
        start_at=args.start_at,
        pr_number=None,
        bot=None,
        backend=args.backend,
        review_backend=args.review_backend,
        fix_backend=args.fix_backend,
        test_backend=args.test_backend,
        ignore_paths=args.ignore_paths,
        loop=args.loop,
        max_iterations=args.max_iterations,
        trajectory_path=args.trajectory_path,
        pr_repo=pr_repo,
        archive=not args.no_archive,
        run_eval=args.run_eval,
        branch=args.branch,
        base=args.base,
        output_mode=output_mode,  # type: ignore[arg-type]
        force_worktree=args.force_worktree,
        shallow=args.shallow,
        extra_copy=list(args.extra_copy),
        plan=args.plan,
    )


def _build_feedback_config(args: argparse.Namespace) -> RunConfig:
    """Build a RunConfig from the ``feedback`` subcommand namespace.

    The feedback subcommand handles PR bot comments — fetching them, applying
    fixes, and posting responses. ``pr_number`` and ``bot`` are populated here
    and consumed by :func:`runner.run_feedback`.
    """
    pr_number: int = args.pr_number
    if pr_number <= 0:
        # argparse already enforces type=int, but guard against negatives.
        raise SystemExit(f"feedback subcommand: PR number must be positive (got {pr_number})")

    pr_repo = _detect_repo_slug()

    return RunConfig(
        target=args.target,
        skill=None,
        model=args.model,
        exploration_model=None,
        review_model=args.review_model,
        parse_model=args.parse_model,
        fix_model=args.fix_model,
        test_model=args.test_model,
        cleanup=None,
        quiet=True,
        start_at="review",
        pr_number=pr_number,
        bot=args.bot,
        backend=args.backend,
        review_backend=args.review_backend,
        fix_backend=args.fix_backend,
        test_backend=args.test_backend,
        ignore_paths=[],
        loop=False,
        max_iterations=5,
        trajectory_path=args.trajectory_path,
        pr_repo=pr_repo,
        archive=not args.no_archive,
        run_eval=args.run_eval,
        output_mode="loop",
    )


def _handle_label_command(argv: list[str]) -> None:
    """Handle ``daydream label <session_id> --accepted|--rejected|--mixed``."""
    import argparse as _argparse

    parser = _argparse.ArgumentParser(
        prog="daydream label",
        description="Label a run outcome for RL/fine-tuning",
    )
    parser.add_argument("session_id", help="Session ID (full or prefix) to label")
    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument("--accepted", action="store_const", const="accepted", dest="label")
    label_group.add_argument("--rejected", action="store_const", const="rejected", dest="label")
    label_group.add_argument("--mixed", action="store_const", const="mixed", dest="label")

    args = parser.parse_args(argv)

    from daydream.archive import get_archive_dir
    from daydream.archive.index import update_labels
    from daydream.ui import create_console, print_info, print_warning

    console = create_console()
    archive_dir = get_archive_dir()

    # Resolve session ID once so both stores target the same run.
    resolved_id = _resolve_session_id(archive_dir, args.session_id)
    if resolved_id is None:
        console.print(f"[red]Session {args.session_id} not found in archive[/red]", highlight=False)
        sys.exit(1)

    manifest_updated = _update_manifest_labels(archive_dir, resolved_id, args.label)

    try:
        success = update_labels(archive_dir, resolved_id, [args.label])
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]", highlight=False)
        sys.exit(1)

    if not success:
        console.print(f"[red]Session {resolved_id} not found in index[/red]", highlight=False)
        sys.exit(1)

    if not manifest_updated:
        print_warning(console, f"Index updated but manifest.json not found for {resolved_id}")
    else:
        print_info(console, f"Labeled {resolved_id} as {args.label}")


def _resolve_session_id(archive_dir: Path, session_id: str) -> str | None:
    """Resolve a full or prefix session ID against the archive runs directory.

    Returns:
        The full session ID if exactly one match is found, None otherwise.
    """
    runs_dir = archive_dir / "runs"
    if not runs_dir.is_dir():
        return None

    exact = runs_dir / session_id
    if exact.is_dir():
        return session_id

    candidates = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(session_id)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _update_manifest_labels(archive_dir: Path, session_id: str, label: str) -> bool:
    """Update manifest.json on disk with the new label.

    Args:
        session_id: Already-resolved full session ID.

    Returns:
        True if the manifest was found and updated, False otherwise.
    """
    import json as _json
    from datetime import datetime, timezone

    manifest_path = archive_dir / "runs" / session_id / "manifest.json"
    if not manifest_path.is_file():
        return False

    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    if "outcome" in manifest:
        manifest["outcome"]["labels"] = [label]
        manifest["outcome"]["labeled_at"] = now
    else:
        manifest["outcome"] = {"labels": [label], "labeled_at": now}
    manifest_path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
    return True


def main() -> None:
    """Run the CLI entry point.

    Returns:
        None: This function does not return; it exits via sys.exit().

    Raises:
        SystemExit: Always raised with exit code 0 on success, 130 on keyboard
            interrupt, or 1 on fatal error.

    """
    _install_signal_handlers()

    # Route subcommands before main arg parse
    argv = sys.argv[1:]
    try:
        if argv and argv[0] == "label":
            _handle_label_command(argv[1:])
            return

        # ``summarize`` is sync — short-circuit before anyio.run kicks in.
        if argv and argv[0] == "summarize":
            summarize_parser = _build_summarize_parser()
            summarize_args = summarize_parser.parse_args(argv[1:])
            sys.exit(_run_summarize(summarize_args))

        is_feedback_subcommand = bool(argv) and argv[0] == "feedback"
        config = _parse_args()
        if is_feedback_subcommand:
            assert config.pr_number is not None  # _build_feedback_config guarantees
            exit_code = anyio.run(run_feedback, config, config.pr_number)
        else:
            exit_code = anyio.run(run, config)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        panel = get_shutdown_panel()
        if panel is not None:
            # Complete agent termination step if it was added
            panel.complete_last_step()
            # Add final step
            panel.add_step("Aborted by user", status="completed")
            panel.finish()
            set_shutdown_panel(None)
        sys.exit(130)
    except git_ops.WrongBranchError as exc:
        # Stage 4.2 — loud branch validation. ``runner.run`` re-raises this
        # so ``cli.main`` owns the user-facing rendering for the
        # silent-failure case where cwd is on the base branch.
        console.print()
        print_error(console, "Wrong Branch", str(exc))
        sys.exit(1)
    except Exception as e:
        # Clean up any active shutdown panel
        panel = get_shutdown_panel()
        if panel is not None:
            panel.finish()
            set_shutdown_panel(None)
        console.print()
        print_error(console, "Fatal Error", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
