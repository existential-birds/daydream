"""Review commands and their argument parsing."""

import argparse
import sys
from pathlib import Path
from typing import Any

from daydream.commands import common
from daydream.run_config import DEEP_FLOW_ALIASES, RunConfig, replaced_review_options


class _HelpAllAction(argparse.Action):
    """Rebuild and print full help while preserving the ordinary parser semantics."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        _namespace: argparse.Namespace,
        _values: Any,
        _option_string: str | None = None,
    ) -> None:
        _build_main_parser(full_help=True).print_help()
        parser.exit()


def _build_main_parser(*, full_help: bool = False) -> argparse.ArgumentParser:
    """Build review arguments with identical parsing in both help modes; full_help affects visibility
    only.
    """
    parser = argparse.ArgumentParser(
        prog="daydream",
        description="Automated code review and fix loop.",
        epilog=(
            "Phase A emission: `daydream --review --findings-out PATH` writes a "
            "strict-schema findings artifact (fingerprints + comment placement) "
            "for the privileged `daydream post-findings` poster."
        ) if full_help else None,
    )

    parser.add_argument(
        "--help-all",
        action=_HelpAllAction,
        nargs=0,
        default=argparse.SUPPRESS,
        help="Show all flags, including advanced ones, then exit.",
    )

    parser.add_argument(
        "target",
        nargs="?",
        metavar="TARGET",
        help="Target directory (default: prompt interactively).",
    )

    # Output mode (mutually exclusive; default = report-only review)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--fix",
        action="store_true",
        help="Review, then offer fixes, test, commit and push.",
    )
    output_group.add_argument(
        "--comment",
        action="store_true",
        help="Review and post inline PR comments, then exit (no fix, no test).",
    )
    output_group.add_argument(
        "--review",
        action="store_true",
        help="Review and write a report to terminal/markdown, then exit (the default).",
    )
    # Require the diagram kind so an optional value cannot consume the positional target path.
    output_group.add_argument(
        "--diagram-only",
        choices=["auto", "sequence", "flowchart", "both"],
        help="Run only the grounded-diagram flow and post a standalone PR comment, "
             "then exit (auto = whatever is eligible).",
    )
    parser.add_argument(
        "--diagram",
        choices=["auto", "sequence", "flowchart", "both", "off"],
        help="Control grounded diagrams in the review paths (default: off). "
             "Use auto to render every eligible kind, or --diagram-only for the diagram-only mode."
        if full_help else argparse.SUPPRESS,
    )

    # Selection
    parser.add_argument(
        "--branch",
        metavar="BRANCH",
        help="Branch to review (default: cwd's local HEAD).",
    )
    parser.add_argument(
        "--base",
        metavar="BASE",
        help="Base ref to compare against (default: PR base if any, else origin/HEAD).",
    )

    # Modifiers
    parser.add_argument(
        "--worktree",
        action="store_true",
        dest="force_worktree",
        help="Force ephemeral worktree even when --branch is omitted." if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--shallow",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--flow",
        metavar="NAME",
        dest="flow_name",
        help="Dispatch a registered flow by name (built-in: deep/review, both the "
             "default review; or a daydream_ext custom flow)." if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--precision",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--test-command",
        help="Canonical shell test command run host-side as a real subprocess "
             "(issue #726); its exit status is the pass/fail signal. Also "
             "settable via the test_command key in .daydream.toml or "
             "[tool.daydream] in pyproject.toml. Missing or invalid commands "
             "stop actionable fix runs with configuration guidance before "
             "tests or repairs execute."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--approve-on-clean",
        action="store_true",
        help="Approve the PR when a complete review has no findings "
             "(issue #343): post event: 'APPROVE' instead of 'COMMENT'."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--file-scope-issues",
        action="store_true",
        help="Opt in to filing out-of-scope findings and reverted edits as GitHub "
             "issues (issue #1056; default off). Also settable via [tool.daydream] "
             "scope_issue_filing in a config file."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--copy",
        action="append",
        default=[],
        metavar="PATH",
        dest="extra_copy",
        type=Path,
        help="Extra path to copy into ephemeral worktree (repeatable)." if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--findings-out",
        metavar="PATH",
        help="Write a strict-schema findings artifact (Phase A emission for "
             "`daydream post-findings`; works with the default deep review flow or --review)."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        metavar="N",
        help="Pin the target PR number (trajectory metadata and the --findings-out "
             "artifact target; default: auto-detect from the current branch)."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--approved-head-sha",
        help="Pin the maintainer-approved PR head SHA."
        if full_help else argparse.SUPPRESS,
    )

    # Stack selection (overrides auto-detect)
    parser.add_argument(
        "-s", "--stack",
        choices=["python", "react", "elixir", "go", "rust", "ios"],
        help=argparse.SUPPRESS,
    )

    # Cleanup / phase resume
    cleanup_group = parser.add_mutually_exclusive_group()
    cleanup_group.add_argument(
        "--cleanup",
        action="store_true",
        default=None,
        help="Cleanup review output after completion" if full_help else argparse.SUPPRESS,
    )
    cleanup_group.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Keep review output after completion" if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-review-cache",
        action="store_false",
        default=None,
        dest="review_cache_enabled",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--start-at",
        choices=["review", "parse", "fix", "test", "ttt", "per-stack", "merge"],
        default="review",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--ignore-path",
        action="append",
        default=[],
        metavar="PATH",
        dest="ignore_paths",
        help="Exclude path from diff (repeatable, e.g. --ignore-path .planning --ignore-path vendor)"
        if full_help else argparse.SUPPRESS,
    )

    common._add_shared_arguments(parser, full_help=full_help)

    return parser


# Per-phase model/backend settings remain in RunConfig and config tables; removed CLI flags name
# those replacements.
_REMOVED_PHASE_FLAGS: dict[str, str] = {
    "--review-backend": "[tool.daydream.phases.review] backend = \"...\"",
    "--fix-backend": "[tool.daydream.phases.fix] backend = \"...\"",
    "--test-backend": "[tool.daydream.phases.test] backend = \"...\"",
    "--exploration-model": "[tool.daydream.phases.exploration] model = \"...\"",
    "--review-model": "[tool.daydream.phases.review] model = \"...\"",
    "--per-stack-review-model": "[tool.daydream.phases.review] model = \"...\"",
    "--arbiter-model": "[tool.daydream.phases.arbiter] model = \"...\"",
    "--fix-model": "[tool.daydream.phases.fix] model = \"...\"",
    "--test-model": "[tool.daydream.phases.test] model = \"...\"",
}


def _reject_removed_phase_flags(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    """Reject removed phase flags, including --flag=value, with their config-table replacement."""
    for token in argv:
        flag = token.split("=", 1)[0]
        if flag == "--parse-model":
            parser.error("--parse-model was removed: the review has no parse phase; drop it.")
        replacement = _REMOVED_PHASE_FLAGS.get(flag)
        if replacement is not None:
            parser.error(
                f"{flag} was removed; set it in the config file instead: "
                f"{replacement} (pyproject.toml or .daydream.toml)."
            )


def _parse_args(argv: list[str] | None = None) -> RunConfig:
    """Parse target, output mode, branch/worktree selection, and modifiers; deep is the default."""
    raw_argv = sys.argv[1:] if argv is None else list(argv)

    # Strip explicit review so it parses identically to the default verb.
    if raw_argv and raw_argv[0] == "review":
        raw_argv = raw_argv[1:]

    parser = _build_main_parser()
    _reject_removed_phase_flags(parser, raw_argv)
    args = parser.parse_args(raw_argv)

    output_mode: str = "review"
    if args.fix:
        output_mode = "loop"
    elif args.comment:
        output_mode = "comment"
    elif args.diagram_only is not None:
        output_mode = "diagram"

    # Diagram modifies review, while diagram-only replaces it; the two modes are incompatible.
    if args.diagram is not None and args.diagram_only is not None:
        parser.error("--diagram cannot be combined with --diagram-only")

    # Every message below names flags the parser accepts: unattended fixing must also confirm its gate.
    fix = "--fix" if args.assume == "yes" or not args.non_interactive else "--fix --yes"

    if args.flow_name is not None:
        if args.fix or args.comment or args.review or args.diagram_only is not None:
            parser.error("--flow cannot be combined with --fix/--review/--comment/--diagram-only")
        if args.shallow:
            parser.error("--flow cannot be combined with --shallow")

    # ``--yes`` answers the gates of a fixing run or a custom flow; it never selects fixing.
    review_only = (
        "--comment" if args.comment else "--review" if args.review
        else "--diagram-only" if args.diagram_only is not None
        else "--flow review" if args.flow_name == "review"
        else "--findings-out" if args.findings_out is not None else None
    )
    if args.assume == "yes" and not (
            args.fix or (args.flow_name is not None and args.flow_name not in DEEP_FLOW_ALIASES)):
        hint = (
            f"it does not apply to {review_only}" if review_only is not None
            else f"use {fix} in place of --flow deep" if args.flow_name == "deep" else f"add {fix}"
        )
        parser.error(f"--yes only answers the gates of a fixing run; {hint}")

    # Unattended, the fix gate declines: a requested fix would end as a successful review.
    if args.fix and args.non_interactive and args.assume != "yes":
        parser.error("--fix --non-interactive cannot confirm the fix gate; add --yes")

    if args.findings_out is not None and output_mode not in ("review", "diagram"):
        parser.error("--findings-out exports review or --diagram-only findings; it cannot follow --comment or --fix")

    config = common.config_from_args(
        parser, args, detect_pr=True, output_mode=output_mode,
        # An explicit diagram-only kind overrides a repo's diagram mode.
        diagram=args.diagram_only if args.diagram_only is not None else args.diagram,
        precision_mode=args.precision, scope_issue_filing=args.file_scope_issues,
    )
    replaced = replaced_review_options(config)
    if replaced:
        parser.error("options of the replaced staged reviewer no longer apply. " + "; ".join(replaced))
    return config
