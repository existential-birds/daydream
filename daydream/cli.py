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
    print_warning,
    set_shutdown_panel,
)


def _warn_deprecated(flag: str, replacement: str) -> None:
    """Emit a deprecation warning for an old-CLI flag.

    Sends the same message through both ``warnings.warn`` (for tooling /
    pytest's ``pytest.warns``) and ``ui.print_warning`` (so end users see a
    visible panel — Python's warnings module is silent by default at the CLI).

    Args:
        flag: The deprecated flag as written on the command line (e.g. ``--ttt``).
        replacement: One-line user-facing replacement guidance.

    """
    msg = f"{flag} is deprecated: {replacement} Removed in next release."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    print_warning(console, msg)


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
        help="Write ATIF v1.6 trajectory JSON to this path (default: <target>/.daydream/trajectory-<ts>-<id>.json)",
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
    # Includes both the new flags (--comment, --review) and the deprecated
    # synonyms (--ttt, --review-only) so argparse handles conflict detection
    # for free.
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
    output_group.add_argument(
        "--ttt", "--trust-the-technology",
        action="store_true",
        default=False,
        dest="trust_the_technology",
        help=argparse.SUPPRESS,  # deprecated; mapped to --comment
    )
    output_group.add_argument(
        "--review-only",
        action="store_true",
        default=False,
        dest="review_only",
        help=argparse.SUPPRESS,  # deprecated; mapped to --review
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

    # ---- Deprecated language / skill flags (mapped to forced_skill + shallow) ----
    skill_group = parser.add_mutually_exclusive_group()
    skill_group.add_argument(
        "-s", "--skill",
        choices=["python", "react", "elixir", "go", "rust", "ios"],
        default=None,
        help="Review skill: python, react, elixir, go, rust, ios",
    )
    skill_group.add_argument(
        "--python",
        action="store_const",
        const="python",
        dest="skill",
        help=argparse.SUPPRESS,
    )
    skill_group.add_argument(
        "--typescript",
        action="store_const",
        const="react",
        dest="skill",
        help=argparse.SUPPRESS,
    )
    skill_group.add_argument(
        "--elixir",
        action="store_const",
        const="elixir",
        dest="skill",
        help=argparse.SUPPRESS,
    )
    skill_group.add_argument(
        "--go",
        action="store_const",
        const="go",
        dest="skill",
        help=argparse.SUPPRESS,
    )
    skill_group.add_argument(
        "--rust",
        action="store_const",
        const="rust",
        dest="skill",
        help=argparse.SUPPRESS,
    )
    skill_group.add_argument(
        "--ios",
        action="store_const",
        const="ios",
        dest="skill",
        help=argparse.SUPPRESS,
    )

    # ---- Other deprecated flags ----
    parser.add_argument(
        "--pr",
        nargs="?",
        const=-1,
        default=None,
        type=int,
        metavar="NUMBER",
        help=argparse.SUPPRESS,  # deprecated: use `daydream feedback <n>`
    )
    parser.add_argument(
        "--bot",
        default=None,
        metavar="BOT_NAME",
        help="Bot username to filter PR comments (required with --pr).",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        default=False,
        dest="deep",
        help=argparse.SUPPRESS,  # deprecated no-op: deep is now the default
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
            "Choices: review | parse | fix | test | ttt (deep-only) | "
            "per-stack (deep-only) | merge (deep-only). Deep-only stages require --deep."
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

    Implements the consolidated CLI surface described in the worktree-isolation
    plan: a positional ``target`` directory, the ``feedback`` subcommand,
    output-mode flags (``--comment`` / ``--review``), selection flags
    (``--branch`` / ``--base``), modifiers (``--worktree`` / ``--shallow`` /
    ``--copy``), and one-release deprecation warnings for the old surface
    (``--python`` / ``--ttt`` / ``--pr`` / ``--deep`` / ``--review-only``).

    The runner-side dispatch on the new ``output_mode`` field lands in
    Stage 4.1b; for now the legacy fields stay populated so the existing run
    flows continue to work unchanged.

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

    parser = _build_main_parser()
    args = parser.parse_args(raw_argv)

    # ----- Reject purely numeric TARGET (likely meant `daydream feedback N`) -----
    if args.target is not None and args.target.lstrip("-").isdigit():
        parser.error(
            f"target '{args.target}' looks like a PR number — "
            f"did you mean: daydream feedback {args.target}?"
        )

    # ----- Map deprecated flags to new fields with warnings -----
    forced_skill: str | None = None
    output_mode: str = "loop"

    if args.skill is not None:
        # Came from --python / --typescript / etc. — treat as deprecated
        # language flag (also covers `-s python` for backward compat; harmless
        # to warn there too since the whole language-pinning mode is going away).
        _warn_deprecated(
            f"--{args.skill}",
            f"use --shallow (deep is now default; {args.skill} auto-detected from changed files).",
        )
        forced_skill = args.skill
        args.shallow = True

    if args.trust_the_technology:
        _warn_deprecated("--ttt", "use --comment.")
        output_mode = "comment"

    if args.review_only:
        _warn_deprecated("--review-only", "use --review.")
        output_mode = "review"

    if args.comment:
        output_mode = "comment"
    elif args.review:
        output_mode = "review"

    if args.deep:
        _warn_deprecated("--deep", "deep is now the default; --deep is unnecessary.")

    # Validate --trust-the-technology mutual exclusions.
    # Note: --ttt vs --review-only is now caught by the output_group mutex
    # (both share that group), so we no longer check it here explicitly.
    if args.trust_the_technology:
        if args.skill:
            parser.error("--trust-the-technology and skill flags are mutually exclusive")
        if args.loop:
            parser.error("--trust-the-technology and --loop are mutually exclusive")
        if args.pr is not None:
            parser.error("--trust-the-technology and --pr are mutually exclusive")

    # Validate --deep mutual exclusions (D-06) and --start-at parse rejection (D-04)
    if args.deep:
        if args.pr is not None:
            parser.error("--deep and --pr are mutually exclusive")
        if args.loop:
            parser.error("--deep and --loop are mutually exclusive")
        if args.trust_the_technology:
            parser.error(
                "--deep and --trust-the-technology are mutually exclusive "
                "(deep runs TTT internally)"
            )
        if args.review_only:
            parser.error("--deep and --review-only are mutually exclusive")
        if args.skill:
            parser.error(
                "--deep and skill flags are mutually exclusive "
                "(deep mode detects stacks and invokes per-stack skills internally)"
            )
        if args.start_at == "parse":
            parser.error(
                "--start-at parse is ambiguous under --deep "
                "(two parse points in the deep pipeline); "
                "use --start-at fix to resume after the merged report"
            )
        if args.start_at == "test":
            parser.error(
                "--start-at test is not supported under --deep "
                "(deep resume stages are ttt|per-stack|merge|fix); "
                "use --start-at fix to resume after the merged report"
            )

    # D-05: deep-only stages require --deep
    if args.start_at in ("ttt", "per-stack", "merge") and not args.deep:
        parser.error(f"--start-at {args.start_at} requires --deep")

    # Validate mutual exclusion: --start-at and --review-only
    if args.start_at != "review" and args.review_only:
        parser.error("--start-at and --review-only are mutually exclusive")

    # Validate --pr mutual exclusions
    if args.pr is not None:
        if not args.bot:
            parser.error("--bot is required when using --pr")
        if args.review_only:
            parser.error("--pr and --review-only are mutually exclusive")
        if args.start_at != "review":
            parser.error("--pr and --start-at are mutually exclusive")
        if args.skill:
            parser.error("--pr and skill flags are mutually exclusive")
        # Deprecated: surface the new path before we route.
        _warn_deprecated("--pr", f"use `daydream feedback {args.pr}` instead.")

    # Validate --bot without --pr
    if args.bot and args.pr is None:
        parser.error("--bot requires --pr")

    # Validate --loop mutual exclusions
    if args.loop:
        if args.review_only:
            parser.error("--loop and --review-only are mutually exclusive")
        if args.start_at != "review":
            parser.error("--loop requires starting at review phase (incompatible with --start-at)")

    # Auto-detect PR number if --pr used without a number
    pr_number = args.pr
    if pr_number == -1:
        pr_number = _auto_detect_pr_number()
        if pr_number is None:
            parser.error("Could not auto-detect PR number from current branch. Specify --pr NUMBER explicitly.")
    if pr_number is not None and pr_number <= 0:
        parser.error("--pr must be a positive integer")

    # Warn if --max-iterations without --loop
    if args.max_iterations != 5 and not args.loop:
        warnings.warn("--max-iterations has no effect without --loop", stacklevel=2)

    # Detect repo slug for trajectory metadata when reviewing a PR
    pr_repo: str | None = None
    if pr_number is not None or args.deep:
        pr_repo = _detect_repo_slug()

    return RunConfig(
        target=args.target,
        skill=args.skill,
        model=args.model,
        cleanup=args.cleanup,
        quiet=True,
        review_only=args.review_only,
        start_at=args.start_at,
        pr_number=pr_number,
        bot=args.bot,
        backend=args.backend,
        review_backend=args.review_backend,
        fix_backend=args.fix_backend,
        test_backend=args.test_backend,
        ignore_paths=args.ignore_paths,
        loop=args.loop,
        max_iterations=args.max_iterations,
        trust_the_technology=args.trust_the_technology,
        deep=args.deep,
        trajectory_path=args.trajectory_path,
        pr_repo=pr_repo,
        archive=not args.no_archive,
        run_eval=args.run_eval,
        # Stage 4 — new consolidated CLI surface.
        branch=args.branch,
        base=args.base,
        output_mode=output_mode,  # type: ignore[arg-type]
        force_worktree=args.force_worktree,
        shallow=args.shallow,
        extra_copy=list(args.extra_copy),
        forced_skill=forced_skill,
    )


def _build_feedback_config(args: argparse.Namespace) -> RunConfig:
    """Build a RunConfig from the ``feedback`` subcommand namespace.

    The feedback subcommand replaces today's ``--pr <n>`` flow. For Stage 4.1a
    the runner-side dispatch hasn't moved yet, so we populate the legacy
    ``pr_number``/``bot`` fields and let ``main()`` keep calling
    ``runner.run_pr_feedback`` via the existing path.
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
        cleanup=None,
        quiet=True,
        review_only=False,
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
        trust_the_technology=False,
        deep=False,
        trajectory_path=args.trajectory_path,
        pr_repo=pr_repo,
        archive=not args.no_archive,
        run_eval=args.run_eval,
        # Stage 4 — new consolidated CLI surface. feedback flow is currently
        # routed via the legacy pr_number/bot fields; output_mode stays "loop"
        # because the feedback flow is its own thing. Stage 4.1b will fold this
        # into a single run() dispatch.
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
