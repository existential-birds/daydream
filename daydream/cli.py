"""CLI lifecycle and verb dispatch. Command parsers and handlers live in commands/."""

import signal
import sys
from collections.abc import Callable

import anyio
from rich.console import Console

from daydream import git_ops
from daydream.agent import console
from daydream.benchmark.cli import _handle_benchmark_command
from daydream.commands import admin, corpus, improve, review, train
from daydream.diagnostics import format_verbose_exception, sanitize_verbose_message
from daydream.phases import UnconfinedFindingError
from daydream.run_context import active_backends
from daydream.runner import run
from daydream.trajectory import flush_active_signal_recorders
from daydream.ui import ShutdownPanel, get_shutdown_panel, print_error, set_shutdown_panel


def _command_handlers() -> dict[str, Callable[[list[str]], int]]:
    """Resolve synchronous commands at call time so overrides remain visible."""
    return {
        "summarize": admin._handle_summarize_command,
        "corpus": corpus._handle_corpus_command,
        "train": train._handle_train_command,
        "benchmark": _handle_benchmark_command,
        "post-findings": admin._handle_post_findings_command,
        "setup": admin._handle_setup_command,
        "ext": admin._handle_ext_command,
    }


# Bare targets and leading flags select review; every other verb owns its parser.
KNOWN_VERBS = {"review", "improve", *_command_handlers()}


def _first_verb(argv: list[str]) -> str:
    """Return a recognized leading verb, else review for empty argv, flags, or bare targets."""
    if argv and argv[0] in KNOWN_VERBS:
        return argv[0]
    return "review"


def _verbose_token_in_argv(argv: list[str]) -> bool:
    """True iff the exact bare ``--verbose`` token appears before any ``--`` separator.

    Joined forms (``--verbose=true``), the removed ``--log`` spelling, and
    tokens after a ``--`` separator never enable verbose diagnostics.
    """
    for token in argv:
        if token == "--":
            return False
        if token == "--verbose":
            return True
    return False


def _signal_handler(signum: int, _frame: object) -> None:
    """Flush every active sibling trajectory as partial, then request shutdown.

    Use the recorder's run registry, independent of task-local ContextVar routing.
    """
    signal_name = signal.Signals(signum).name

    # Flush every active recorder before tearing down (D-07). The registry
    # isolates ordinary write failures per recorder and never awaits.
    flush_active_signal_recorders()

    panel = ShutdownPanel(console)
    set_shutdown_panel(panel)
    panel.start(f"Received {signal_name}, shutting down")

    if active_backends():
        panel.add_step("Terminating running agent(s)...")

    raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    """Install signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _shutdown_and_exit(
    console: Console,
    title: str,
    message: str,
    *,
    verbose_diagnostic: str | None = None,
) -> None:
    """Finish/clear the shutdown panel, render the error, and exit 1.

    Generic fatal errors may add verbose stderr diagnostics after the panel.
    """
    panel = get_shutdown_panel()
    if panel is not None:
        panel.finish()
        set_shutdown_panel(None)
    console.print()
    print_error(console, title, message)
    if verbose_diagnostic is not None:
        print(verbose_diagnostic, file=sys.stderr)
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Run the verb-first CLI, using ``sys.argv[1:]`` when *argv* is absent.

    Each verb owns its parser and exit code. The default review path accepts
    a bare target or leading flag. This entry point exits 0 on success, 130
    on keyboard interrupt, and 1 on fatal errors.
    """
    _install_signal_handlers()

    # Verb-first dispatch: non-``review`` verbs are short-circuited here (each
    # owns its parser and exit code); everything else flows into ``review._parse_args``.
    argv = list(argv) if argv is not None else sys.argv[1:]
    # Resolve bare --verbose before config/provenance/tracing so startup failures
    # can still include diagnostics. Tokens after -- do not enable it.
    verbose_mode = _verbose_token_in_argv(argv)
    # Reject removed bench explicitly; otherwise it would become a review target.
    if argv and argv[0] == "bench":
        print(
            "error: the 'bench' command is no longer a command; use 'daydream benchmark'",
            file=sys.stderr,
        )
        sys.exit(2)
    verb = _first_verb(argv)
    try:
        if handler := _command_handlers().get(verb):
            sys.exit(handler(argv[1:]))

        # Synchronous Improve cleanup bypasses anyio.run. Use the parser's shared
        # sub-verb set so routing stays aligned.
        if verb == "improve" and (
            argv[1] if len(argv) > 1 else None
        ) in improve.IMPROVE_SYNC_SUB_VERBS:
            config = improve._parse_improve_args(argv)
            if argv[1] == "prune-reanchor":
                sys.exit(improve._handle_prune_reanchor(config))
            sys.exit(improve._handle_list_reanchor(config))

        # ``improve list-reanchored`` is a sync, read-only one-purpose command
        # (mirroring the corpus/ext short-circuits), so it never spins up
        # a flow through ``improve._parse_improve_args``/``anyio.run``.
        if verb == "improve" and len(argv) > 1 and argv[1] == "list-reanchored":
            sys.exit(improve._handle_list_reanchored_command(argv[2:]))

        config = (
            improve._parse_improve_args(argv)
            if verb == "improve"
            else review._parse_args(argv)
        )
        try:
            exit_code = anyio.run(run, config)
        except BaseExceptionGroup as exc:
            # A signal delivered inside a task-group body can be wrapped by
            # AnyIO. Preserve the ordinary interrupt path without hiding an
            # unrelated failure accompanying the interruption.
            interrupts, remainder = exc.split(KeyboardInterrupt)
            if interrupts is not None and remainder is None:
                raise KeyboardInterrupt from None
            raise
        sys.exit(exit_code)
    except KeyboardInterrupt:
        panel = get_shutdown_panel()
        if panel is not None:
            panel.complete_last_step()
            panel.add_step("Aborted by user", status="completed")
            panel.finish()
            set_shutdown_panel(None)
        sys.exit(130)
    except git_ops.WrongBranchError as exc:
        # ``runner.run`` re-raises so cli.main owns the user-facing rendering for
        # the silent-failure case where cwd is on the base branch.
        console.print()
        print_error(console, "Wrong Branch", str(exc))
        sys.exit(1)
    except UnconfinedFindingError as e:
        # Render escaped confinement failures by type; ordinary ValueErrors use the
        # fatal handler. Do not cite fix_failures: only the normal recovery path
        # writes that artifact, so it may not exist here.
        _shutdown_and_exit(
            console,
            "Unconfined Finding",
            f"{e}. Check the finding's file ref.",
        )
    except Exception as e:
        if verbose_mode:
            try:
                diagnostic = format_verbose_exception(e)
            except Exception:
                diagnostic = "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]"
        else:
            diagnostic = None
        # The panel message is redacted and control-neutralized exactly like the
        # verbose diagnostic, so a hostile exception message can never paint
        # the operator's terminal even when verbose diagnostics are off.
        safe_message = sanitize_verbose_message(e)
        _shutdown_and_exit(console, "Fatal Error", safe_message, verbose_diagnostic=diagnostic)


if __name__ == "__main__":
    main()
