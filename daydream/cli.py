"""CLI entry point for daydream."""

import argparse
import signal
import sys

import anyio

from daydream.agent import (
    console,
    get_current_client,
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

    if get_current_client() is not None:
        panel.add_step("Terminating running agent...")

    raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    """Install signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


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
        choices=["python", "frontend"],
        default=None,
        help="Review skill: python, frontend",
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
        const="frontend",
        dest="skill",
        help="Use React/TypeScript review skill",
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
        "--model",
        choices=["opus", "sonnet", "haiku"],
        default="opus",
        help="Claude model to use (default: opus)",
    )

    parser.add_argument(
        "--rlm",
        action="store_true",
        default=False,
        help="Use RLM mode for large codebase review (1M+ tokens)",
    )

    args = parser.parse_args()

    # Validate mutual exclusion: --start-at and --review-only
    if args.start_at != "review" and args.review_only:
        parser.error("--start-at and --review-only are mutually exclusive")

    return RunConfig(
        target=args.target,
        skill=args.skill,
        model=args.model,
        debug=args.debug,
        cleanup=args.cleanup,
        quiet=True,
        review_only=args.review_only,
        start_at=args.start_at,
        rlm_mode=args.rlm,
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
