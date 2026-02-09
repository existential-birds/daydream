"""CLI entry point for daydream."""

import argparse
import json
import signal
import subprocess
import sys

import anyio

from daydream.agent import (
    console,
    get_current_backends,
    set_shutdown_requested,
)
from daydream.runner import RunConfig, run
from daydream.ui import (
    ShutdownPanel,
    get_shutdown_panel,
    print_error,
    set_shutdown_panel,
)


def _signal_handler(signum: int, frame: object) -> None:
    """Handle termination signals by requesting shutdown."""
    signal_name = signal.Signals(signum).name
    set_shutdown_requested(True)

    # Create and start the shutdown panel
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
        # Safe: hardcoded command with no user input
        result = subprocess.run(  # noqa: S603, S607
            ["gh", "pr", "view", "--json", "number"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("number")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        # FileNotFoundError occurs when gh CLI is not installed
        pass
    return None


def _parse_args() -> RunConfig:
    """Parse command line arguments and return a RunConfig.

    Returns:
        RunConfig: Configuration object populated from command line arguments.

    """
    parser = argparse.ArgumentParser(
        prog="daydream",
        description="Automated code review and fix loop",
    )

    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        metavar="TARGET",
        help="Target directory (default: prompt interactively)",
    )

    skill_group = parser.add_mutually_exclusive_group()
    skill_group.add_argument(
        "-s", "--skill",
        choices=["python", "react", "elixir"],
        default=None,
        help="Review skill: python, react, elixir",
    )
    skill_group.add_argument(
        "--python",
        action="store_const",
        const="python",
        dest="skill",
        help="Use Python/FastAPI review skill",
    )
    skill_group.add_argument(
        "--typescript",
        action="store_const",
        const="react",
        dest="skill",
        help="Use React/TypeScript review skill",
    )
    skill_group.add_argument(
        "--elixir",
        action="store_const",
        const="elixir",
        dest="skill",
        help="Use Elixir/Phoenix review skill",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Save debug log",
    )

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
        "--review-only",
        action="store_true",
        default=False,
        help="Skip fixes, only review and parse feedback",
    )

    parser.add_argument(
        "--start-at",
        choices=["review", "parse", "fix", "test"],
        default="review",
        dest="start_at",
        help="Start at a specific phase (default: review)",
    )

    parser.add_argument(
        "--pr",
        nargs="?",
        const=-1,
        default=None,
        type=int,
        metavar="NUMBER",
        help="PR feedback mode: fetch and fix bot review comments (auto-detect PR if number omitted)",
    )

    parser.add_argument(
        "--bot",
        default=None,
        metavar="BOT_NAME",
        help="Bot username to filter PR comments (required with --pr)",
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

    args = parser.parse_args()

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

    # Validate --bot without --pr
    if args.bot and args.pr is None:
        parser.error("--bot requires --pr")

    # Auto-detect PR number if --pr used without a number
    pr_number = args.pr
    if pr_number == -1:
        pr_number = _auto_detect_pr_number()
        if pr_number is None:
            parser.error("Could not auto-detect PR number from current branch. Specify --pr NUMBER explicitly.")
    if pr_number is not None and pr_number <= 0:
        parser.error("--pr must be a positive integer")

    return RunConfig(
        target=args.target,
        skill=args.skill,
        model=args.model,
        debug=args.debug,
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
        loop=args.loop,
        max_iterations=args.max_iterations,
    )


def main() -> None:
    """Run the CLI entry point.

    Returns:
        None: This function does not return; it exits via sys.exit().

    Raises:
        SystemExit: Always raised with exit code 0 on success, 130 on keyboard
            interrupt, or 1 on fatal error.

    """
    _install_signal_handlers()
    config = _parse_args()
    try:
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
