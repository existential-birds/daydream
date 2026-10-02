"""Improve commands and their argument parsing."""

import argparse
from pathlib import Path

from daydream.commands import common
from daydream.run_config import RunConfig
from daydream.ui import create_console, print_error, print_info, print_success

# Sub-verbs recognized under the ``improve`` verb. Single source of truth for
# improve sub-verb dispatch: ``_parse_improve_args`` strips any of these from
# argv, and ``main()`` derives its sync short-circuit set from this constant so
# the two can never drift apart.
IMPROVE_SUB_VERBS = frozenset({"plan", "prune-reanchor", "list-reanchor"})


# Sub-verbs that do no agent work and short-circuit to sync handlers in
# ``main()`` — everything except the ``plan`` flow, which routes through
# ``anyio.run(run, ...)``.
IMPROVE_SYNC_SUB_VERBS = IMPROVE_SUB_VERBS - {"plan"}


def _build_improve_parser(
    subverb: str | None = None,
) -> argparse.ArgumentParser:
    """Build the parser for an improve audit or manual sub-verb."""
    suffix = f" {subverb}" if subverb else ""
    parser = argparse.ArgumentParser(
        prog=f"daydream improve{suffix}",
        description="Audit a repository and write prioritized advisory artifacts.",
    )
    if subverb == "plan":
        parser.add_argument(
            "improve_plan_description",
            metavar="DESCRIPTION",
            help="Change to investigate and turn into one implementation plan",
        )
    if subverb == "prune-reanchor":
        parser.add_argument(
            "improve_prune_name",
            metavar="NAME",
            help="name of the -reanchor worktree to remove",
        )
    parser.add_argument("target", metavar="TARGET", help="Repository to audit")
    parser.add_argument(
        "--effort",
        choices=["quick", "standard", "deep"],
        default="standard",
        dest="improve_effort",
        help=(
            "Audit breadth: quick = correctness/security/tests/tech-debt, serial, "
            "HIGH-confidence findings capped near six; standard (default) = all "
            "eight categories in parallel; deep = all eight searched very "
            "thoroughly, including labeled LOW-confidence investigate items. "
            "Does not change the model or reasoning effort — those are per-phase "
            "(see [tool.daydream.phases.<phase>])"
        ),
    )
    parser.add_argument(
        "--focus",
        choices=["security", "performance", "tests", "branch"],
        dest="improve_focus",
        help=(
            "Narrow the audit: a single category, 'branch' to audit only the "
            "diff against the base branch"
        ),
    )
    parser.add_argument(
        "--scope",
        metavar="SERVICE_OR_GLOB",
        dest="improve_scope",
        help=(
            "Restrict the audit to one service, a glob over service roots, or a "
            "named group from [tool.daydream.improve.service_groups]"
        ),
    )
    common._add_shared_arguments(parser)
    return parser


def _parse_improve_args(argv: list[str]) -> RunConfig:
    """Parse an improve invocation into the shared run configuration."""
    improve_argv = argv[1:] if argv and argv[0] == "improve" else argv
    subverb = (
        improve_argv[0]
        if improve_argv and improve_argv[0] in IMPROVE_SUB_VERBS
        else None
    )
    if subverb is not None:
        improve_argv = improve_argv[1:]
    parser = _build_improve_parser(subverb)
    args = parser.parse_args(improve_argv)
    return common.config_from_args(parser, args, flow_name="improve")


def _build_list_reanchored_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream improve list-reanchored <target>``.

    A read-only listing of re-anchored plans from the durable plan index.
    ``--json`` switches the output from the human summary table to a JSON
    array, so the reading is scriptable.
    """
    parser = argparse.ArgumentParser(
        prog="daydream improve list-reanchored",
        description="List every re-anchored plan from the durable plan index.",
    )
    parser.add_argument("target", metavar="TARGET", help="Repository to list")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the rows as a JSON array instead of a human table.",
    )
    return parser


def _handle_list_reanchored_command(argv: list[str]) -> int:
    """Handle ``daydream improve list-reanchored <target>``.

    A one-purpose, read-only listing of re-anchored plans from the durable
    ``daydream_plans/.index.json``. Every row carries the plan number, title,
    status, and landing path. An empty result prints a clear line and exits 0.

    Returns:
        ``0`` always on non-exceptional paths.
    """
    import json

    from rich.markup import escape

    from daydream.improve.plan_index import (
        reanchored_plan_rows,
    )

    parser = _build_list_reanchored_parser()
    args = parser.parse_args(argv)
    rows = reanchored_plan_rows(Path(args.target) / "daydream_plans")

    console = create_console()
    if args.json:
        console.print(
            json.dumps(
                [
                    {
                        "number": entry.number,
                        "title": entry.title,
                        "status": entry.status,
                        "landing_path": entry.landing_path,
                    }
                    for entry in rows
                ],
                indent=2,
            ),
            soft_wrap=True,
            markup=False,
        )
        return 0
    if not rows:
        print_info(console, "No re-anchored plans.")
        return 0
    print_info(console, "Re-anchored plans:")
    # Long landing paths must not wrap mid-string (rich would otherwise insert
    # a newline inside the path at the console width), so rows use soft_wrap.
    for entry in rows:
        console.print(
            f"[neon.cyan]ℹ[/] [neon.fg]{entry.number:03d} "
            f"{escape(entry.title)} | {escape(entry.status)} | "
            f"{escape(entry.landing_path) if entry.landing_path else '(unavailable)'}[/]",
            soft_wrap=True,
        )
    return 0


def _handle_prune_reanchor(config: RunConfig) -> int:
    """Remove one named ``-reanchor`` worktree; sync cleanup, no improve flow.

    Returns ``0`` on removal and ``1`` for every other verdict, so the exit
    code reads as a reliable removal contract for scripts/operators.
    """
    from daydream.improve.reanchor import (
        PRUNE_NOT_FOUND,
        PRUNE_NOT_REANCHOR,
        PRUNE_REMOVED,
        PRUNE_UNSAFE_NAME,
        prune_named_reanchor_worktree,
    )

    console = create_console()
    name = config.improve_prune_name
    assert name is not None  # the prune-reanchor sub-verb guard guarantees this
    repo = Path(config.target) if config.target else Path.cwd()
    outcome = prune_named_reanchor_worktree(repo, name)
    if outcome.verdict == PRUNE_REMOVED:
        plans = "plans" if outcome.plan_count != 1 else "plan"
        suffix = f" (held {outcome.plan_count} re-anchored {plans})"
        print_success(console, f"Removed worktree {name}{suffix}")
        return 0
    messages = {
        PRUNE_NOT_FOUND: f"No such worktree: {name}",
        PRUNE_NOT_REANCHOR: f"{name!r} is not a -reanchor worktree",
        PRUNE_UNSAFE_NAME: f"{name!r} is not a safe worktree name",
    }
    print_error(console, "Prune re-anchor", messages.get(outcome.verdict, f"Could not remove {name}"))
    return 1


def _handle_list_reanchor(config: RunConfig) -> int:
    """List existing ``-reanchor`` worktrees; sync, no improve flow.

    An empty list still exits ``0`` — listing nothing is not an error.
    """
    from daydream.improve.reanchor import (
        list_reanchor_worktrees,
    )

    console = create_console()
    repo = Path(config.target) if config.target else Path.cwd()
    for path in list_reanchor_worktrees(repo):
        print_info(console, path.name)
    return 0
