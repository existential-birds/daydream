"""CLI entry point for daydream."""

import signal
import sys

import anyio

from daydream.agent import (
    console,
    get_current_client,
    set_shutdown_requested,
)
from daydream.runner import run
from daydream.ui import (
    print_dim,
    print_error,
    print_warning,
)


def _signal_handler(signum: int, frame: object) -> None:
    """Handle termination signals by requesting shutdown."""
    signal_name = signal.Signals(signum).name
    print_warning(console, f"Received {signal_name}, shutting down...")
    set_shutdown_requested(True)

    if get_current_client() is not None:
        print_dim(console, "Terminating running agent...")
        raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    """Install signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def main() -> None:
    """Main entry point for the CLI."""
    _install_signal_handlers()
    try:
        exit_code = anyio.run(run)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print()
        print_warning(console, "Aborted by user")
        sys.exit(130)
    except Exception as e:
        console.print()
        print_error(console, "Fatal Error", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
