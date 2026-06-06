"""CLI entry point for daydream.

Dispatch is verb-first. :func:`_first_verb` classifies the leading argv token
against :data:`KNOWN_VERBS`; anything that is not an explicit verb — a bare
target path, a leading flag, or empty argv — falls through to the default
``review`` shim, so ``daydream /path`` and ``daydream review /path`` are
equivalent. Each non-``review`` verb is dispatched manually from :func:`main`
before the main argparse parser runs (so its flags don't collide with the
top-level ``TARGET`` positional):

- ``daydream [review] <target>`` — the review/fix loop (default verb)
- ``daydream feedback <pr#>`` — apply bot PR-review comments
- ``daydream summarize <path>`` — print run-info markdown for a trajectory
- ``daydream bench`` — score deep-review findings against the offline benchmark
- ``daydream harvest`` — walk the archive and append one bitemporal
  annotation (outcome label + intrinsic reward) per indexed run
- ``daydream label <session-prefix> --outcome {accepted,contested,rejected,unknown}``
  — record an authoritative human outcome label that overrides automated ones
- ``daydream build-corpus --out <path>`` — project the as-of-pinned
  annotations into a JSONL training corpus plus a lineage manifest
"""

import argparse
import inspect
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anyio

from daydream import git_ops
from daydream.agent import (
    console,
    get_current_backends,
    set_shutdown_requested,
)
from daydream.benchmark.cli import _handle_bench_command
from daydream.config_file import load_file_config
from daydream.runner import RunConfig, run, run_feedback
from daydream.trajectory import get_signal_recorder
from daydream.ui import (
    ShutdownPanel,
    get_shutdown_panel,
    print_error,
    set_shutdown_panel,
)

# Verb-first dispatch table. ``_first_verb`` classifies the leading argv token;
# anything that isn't an explicit verb (bare path, leading flag, empty argv)
# falls through to the ``review`` golden path via the default-verb shim.
KNOWN_VERBS = {"review", "feedback", "summarize", "corpus", "bench"}

# Data-pipeline verbs still routed at the top level. These are slated to move
# under the ``corpus`` namespace; until then they remain explicit verbs so the
# shim does not misroute them to ``review``.
_LEGACY_DATA_VERBS = {"harvest", "build-corpus", "label"}


def _first_verb(argv: list[str]) -> str:
    """Classify the leading argv token into a verb.

    Returns the leading token when it is a recognized verb; otherwise returns
    ``"review"``. The fallthrough covers the three default-verb cases — empty
    argv, a leading flag, and a bare target path — so a plain
    ``daydream /path`` routes through the same parser as ``daydream review
    /path``.

    Args:
        argv: The raw argument list (``sys.argv[1:]``).

    Returns:
        str: The dispatched verb name (a member of :data:`KNOWN_VERBS`, one of
        the legacy data verbs, or ``"review"``).

    """
    if argv and (argv[0] in KNOWN_VERBS or argv[0] in _LEGACY_DATA_VERBS):
        return argv[0]
    return "review"


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


def _auto_detect_pr_number(repo: Path) -> int | None:
    """Auto-detect PR number from the target checkout's branch via gh CLI.

    Args:
        repo: Repository working directory to inspect — the target checkout
            being reviewed, not necessarily the cwd where ``daydream`` was
            launched.

    Returns:
        The PR number if found, or None if detection fails.

    """
    try:
        data = git_ops.gh_pr_view(repo, None)
    except git_ops.GitError:
        return None
    if not data:
        return None
    number = data.get("number")
    return int(number) if isinstance(number, int) else None


def _detect_repo_slug(repo: Path) -> str | None:
    """Detect the GitHub owner/repo slug for a repository via gh CLI.

    Args:
        repo: Repository working directory to inspect — the target checkout
            being reviewed, not necessarily the cwd where ``daydream`` was
            launched. Attributing the slug to the target keeps trajectory and
            archive provenance correct when daydream is run from one repo
            against a checkout of another (the benchmark-harness pattern).

    Returns:
        String like ``"owner/repo"``, or None if detection fails.
    """
    try:
        slug = git_ops.gh_repo_view(repo)
    except git_ops.GitError:
        return None
    if slug is None:
        return None
    owner, name = slug
    return f"{owner}/{name}"


def _add_shared_arguments(parser: argparse.ArgumentParser, *, full_help: bool = True) -> None:
    """Add the shared (non-output-mode) arguments to a parser or subparser.

    Used by both the top-level parser and the ``feedback`` subparser so flags
    like ``--backend``, ``--model``, ``--trajectory`` work in both places. The
    global ``--model``/``--backend`` here feed the source-tiered precedence in
    :func:`daydream.runner._resolved_model` / ``_resolve_backend``
    (CLI > env > config-file > per-backend table).

    Per-phase model/backend overrides are no longer CLI flags — they live in
    ``[tool.daydream.phases.<phase>]`` of the config file (``pyproject.toml`` /
    ``.daydream.toml``). The removed flags are rejected with a curated pointer
    by :func:`_reject_removed_phase_flags`; the underlying ``RunConfig`` fields
    (``review_model``, ``fix_backend``, …) remain and are still populated from
    the config file and read by ``_resolve_backend``.

    Args:
        parser: The parser (or subparser) to add the shared arguments to.
        full_help: When False, advanced flags (``--trajectory``, ``--no-archive``,
            ``--eval``, ``--non-interactive``) are added with their help text
            suppressed so the default ``--help`` stays focused on common flags.
            They still parse and populate ``RunConfig`` unchanged; ``--help-all``
            re-builds the parser with ``full_help=True`` to surface them.
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
        ) if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        default=False,
        dest="no_archive",
        help="Disable automatic archival to ~/.daydream/archive/" if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        dest="run_eval",
        help="Run deterministic evaluation analysis and store evaluation.json in archive"
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["claude", "codex"],
        default=None,
        help="Agent backend: claude or codex "
             "(default: env DAYDREAM_BACKEND, config file, then claude)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        type=str,
        dest="model",
        metavar="MODEL",
        help="Global default model across phases "
             "(default: env DAYDREAM_MODEL, config file, then the per-backend table). "
             "A per-phase config override is beaten by this global --model.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        dest="non_interactive",
        help="Run without prompting; take each prompt's safe default "
             "(confirm intent, decline fixes, exit the test/heal loop)."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--yes",
        action="store_const",
        const="yes",
        default=None,
        dest="assume",
        help="Auto-answer every yes/no gate with yes (apply fixes, commit). "
             "Orthogonal to --non-interactive: --yes pre-decides the answer, "
             "--non-interactive controls whether we may block on stdin.",
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


def _build_build_corpus_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream build-corpus --out <path> [...]``.

    Like ``summarize`` and ``feedback``, ``build-corpus`` is dispatched manually
    from ``main()`` before the main parser runs so its options don't collide
    with the top-level ``TARGET`` positional.
    """
    parser = argparse.ArgumentParser(
        prog="daydream build-corpus",
        description="Project as-of-pinned annotations into JSONL training records (one object per run).",
    )

    # Required output path.
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output .jsonl path",
    )

    # ---- Filters (post-applied AFTER exclusion list) ----
    parser.add_argument(
        "--skill",
        type=str,
        default=None,
        help="Match manifest.skill exactly",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Repeatable; restrict to these repo slugs",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Repeatable; default is just 'accepted' unless --include-all-labels is set",
    )
    parser.add_argument(
        "--min-grounding",
        type=float,
        default=None,
        dest="min_grounding",
        help="Drop runs below this grounding_rate",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=None,
        dest="min_reward",
        help="Alternative admission path: admit runs whose pinned annotation has "
             "composite_reward >= this threshold, even if not 'accepted'",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="complete",
        help="Match manifest.status exactly (default: 'complete')",
    )

    # ---- Stratification ----
    parser.add_argument(
        "--stratify-by",
        type=str,
        choices=["stack"],
        default=None,
        dest="stratify_by",
        help="Stratify the corpus; currently only 'stack' is supported",
    )
    parser.add_argument(
        "--max-stack-share",
        type=float,
        default=0.6,
        dest="max_stack_share",
        help="Per-stack cap fraction in (0, 1] (default: 0.6)",
    )

    # ---- Opt-ins ----
    parser.add_argument(
        "--allow-copyleft",
        action="append",
        default=[],
        dest="allow_copyleft",
        help="Repeatable; permit specific GPL/AGPL repos",
    )
    parser.add_argument(
        "--include-all-labels",
        action="store_true",
        dest="include_all_labels",
        help="Disable the C9 default of accepted-only label filtering",
    )

    # ---- Diagnostic ----
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print summary table, write nothing",
    )
    parser.add_argument(
        "--emit-schema-only",
        action="store_true",
        dest="emit_schema_only",
        help="Write schema.json next to --out, skip records",
    )

    # ---- Bitemporal pin ----
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        dest="as_of",
        metavar="ISO_TS",
        help="ISO-8601 transaction-time pin; resolve each run's annotation "
             "as of this instant for reproducible corpora (default: latest)",
    )

    return parser


def _handle_build_corpus_command(argv: list[str]) -> int:
    """Handle ``daydream build-corpus --out <path> [...]``.

    Drives :func:`daydream.training.corpus.run_build_corpus` synchronously
    (``build-corpus`` does no agent work — just SQLite reads and a JSONL +
    lineage-manifest write). Returns an exit code rather than calling
    :func:`sys.exit`; ``main()`` is responsible for translating the code into a
    process exit. This keeps the handler easy to drive from tests.

    Args:
        argv: The argument vector after the ``build-corpus`` verb.

    Returns:
        ``0`` on success; ``1`` on a validation error.
    """
    from daydream.training.corpus import BuildCorpusConfig, CorpusFilters, run_build_corpus
    from daydream.ui import create_console, print_error

    parser = _build_build_corpus_parser()
    args = parser.parse_args(argv)

    # Validate --max-stack-share is in (0, 1].
    if not (0.0 < args.max_stack_share <= 1.0):
        print_error(create_console(), "Invalid --max-stack-share", "Must be in (0, 1].")
        return 1

    # Validate --min-grounding is in [0, 1] when set.
    if args.min_grounding is not None and not (0.0 <= args.min_grounding <= 1.0):
        print_error(create_console(), "Invalid --min-grounding", "Must be in [0, 1].")
        return 1

    if args.include_all_labels and args.label:
        print_error(create_console(), "Conflicting flags", "--include-all-labels and --label cannot be used together.")
        return 1

    if args.include_all_labels:
        labels: tuple[str, ...] = ()
    else:
        labels = tuple(args.label) if args.label else ("accepted",)

    if args.as_of is not None:
        raw = args.as_of.replace("Z", "+00:00") if args.as_of.endswith("Z") else args.as_of
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            print_error(create_console(), "Invalid --as-of", "Must be an ISO-8601 timestamp.")
            return 1
        if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
            print_error(create_console(), "Invalid --as-of", "Must be a UTC timestamp (ending in Z or +00:00).")
            return 1
        args.as_of = dt.astimezone(timezone.utc).isoformat()

    filters = CorpusFilters(
        skill=args.skill,
        repos=tuple(args.repo),
        labels=labels,
        min_grounding=args.min_grounding,
        status=args.status,
        include_all_labels=args.include_all_labels,
        allow_copyleft=frozenset(args.allow_copyleft),
        min_reward=args.min_reward,
    )
    config = BuildCorpusConfig(
        out_path=args.out,
        filters=filters,
        stratify_by=args.stratify_by,
        max_stack_share=args.max_stack_share,
        dry_run=args.dry_run,
        emit_schema_only=args.emit_schema_only,
        as_of=args.as_of,
    )
    run_build_corpus(config)
    return 0


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


class _HelpAllAction(argparse.Action):
    """Print the full help (advanced flags included) and exit.

    The default ``--help`` is built with ``full_help=False`` so advanced flags
    are suppressed. ``--help-all`` re-builds the parser with ``full_help=True``
    and renders that help instead, surfacing every flag without changing how
    any of them parse.
    """

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):  # noqa: A002
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        _build_main_parser(full_help=True).print_help()
        parser.exit()


def _build_main_parser(*, full_help: bool = False) -> argparse.ArgumentParser:
    """Build the main argparse parser for the consolidated CLI surface.

    Args:
        full_help: When True, advanced flags carry their help text so they show
            up under ``--help-all``. When False (the default for ``--help``),
            advanced flags are added with ``argparse.SUPPRESS`` help so the
            default help stays focused on common flags. Either way the flags
            parse identically and populate ``RunConfig`` unchanged.
    """
    parser = argparse.ArgumentParser(
        prog="daydream",
        description="Automated code review and fix loop. "
                    "Use `daydream feedback <pr#>` to process PR bot comments.",
    )

    parser.add_argument(
        "--help-all",
        action=_HelpAllAction,
        help="Show all flags, including advanced ones, then exit.",
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
        help="Force ephemeral worktree even when --branch is omitted." if full_help else argparse.SUPPRESS,
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
        help="Extra path to copy into ephemeral worktree (repeatable)." if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        default=False,
        dest="plan",
        help="Generate an implementation plan and embed it in PR comments (use with --comment)."
        if full_help else argparse.SUPPRESS,
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
        help="Cleanup review output after completion" if full_help else argparse.SUPPRESS,
    )
    cleanup_group.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Keep review output after completion" if full_help else argparse.SUPPRESS,
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
        ) if full_help else argparse.SUPPRESS,
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

    parser.add_argument(
        "--loop",
        nargs="?",
        const=5,
        default=None,
        type=int,
        metavar="N",
        help="Repeat the review-fix-test cycle until zero issues or N iterations (default N: 5)",
    )

    _add_shared_arguments(parser, full_help=full_help)

    return parser


def _normalize_loop_argv(raw_argv: list[str]) -> list[str]:
    """Disambiguate ``--loop``'s optional count from a following positional.

    ``--loop`` carries an optional integer count (``nargs="?"``), which makes
    argparse greedily try to consume the next token as that count. For the
    golden path ``daydream --loop /some/path`` argparse would then fail trying
    to parse the path as an int. This pre-scan pins the count explicitly when
    ``--loop`` is bare (last token, or followed by a flag or a non-integer
    token), turning it into ``--loop=5`` so the positional is preserved. An
    explicit ``--loop N`` is left untouched.

    Args:
        raw_argv: The argument list after verb-shim stripping.

    Returns:
        A new argv list with bare ``--loop`` rewritten to ``--loop=5``.
    """
    normalized: list[str] = []
    for i, token in enumerate(raw_argv):
        if token == "--loop":
            nxt = raw_argv[i + 1] if i + 1 < len(raw_argv) else None
            if nxt is None or not nxt.isdigit():
                # Bare --loop (end, before a flag, or before a non-int target):
                # pin the default count so the next token stays a positional.
                normalized.append("--loop=5")
                continue
        normalized.append(token)
    return normalized


# Removed per-phase model/backend flags → their config-file replacement. The
# per-phase overrides moved out of the CLI surface into
# ``[tool.daydream.phases.<phase>]`` (pyproject.toml / .daydream.toml). The
# ``RunConfig`` fields they used to set (``review_model``, ``fix_backend``, …)
# remain — still read by ``_resolve_backend`` and settable via the config file —
# they are simply no longer CLI-settable.
_REMOVED_PHASE_FLAGS: dict[str, str] = {
    "--review-backend": "[tool.daydream.phases.review] backend = \"...\"",
    "--fix-backend": "[tool.daydream.phases.fix] backend = \"...\"",
    "--test-backend": "[tool.daydream.phases.test] backend = \"...\"",
    "--exploration-model": "[tool.daydream.phases.exploration] model = \"...\"",
    "--review-model": "[tool.daydream.phases.review] model = \"...\"",
    "--parse-model": "[tool.daydream.phases.parse] model = \"...\"",
    "--fix-model": "[tool.daydream.phases.fix] model = \"...\"",
    "--test-model": "[tool.daydream.phases.test] model = \"...\"",
}


def _reject_removed_phase_flags(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    """Reject any removed per-phase model/backend flag with a config pointer.

    Pre-parse scan (P-reject pattern): if any token in ``argv`` is a removed
    per-phase flag — either the bare ``--flag`` form (``--fix-model value``) or
    the joined ``--flag=value`` form — call ``parser.error`` with a curated
    message naming the ``[tool.daydream.phases.<phase>]`` config replacement.

    Args:
        parser: The parser whose ``error`` is used to exit with the message.
        argv: The argument list (after verb-shim stripping).
    """
    for token in argv:
        flag = token.split("=", 1)[0]
        replacement = _REMOVED_PHASE_FLAGS.get(flag)
        if replacement is not None:
            parser.error(
                f"{flag} was removed; set it in the config file instead: "
                f"{replacement} (pyproject.toml or .daydream.toml)."
            )


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

    # Default-verb shim: an explicit leading ``review`` token is equivalent to
    # the bare ``daydream <target>`` form. Strip it so both parse identically
    # against the main review parser (positional TARGET unchanged).
    if raw_argv and raw_argv[0] == "review":
        raw_argv = raw_argv[1:]

    # Manual subcommand dispatch: argparse subparsers eat the first positional
    # which conflicts with our positional TARGET. So we pop "feedback" off the
    # front ourselves and route to a dedicated parser.
    if raw_argv and raw_argv[0] == "feedback":
        feedback_parser = _build_feedback_parser()
        _reject_removed_phase_flags(feedback_parser, raw_argv[1:])
        feedback_args = feedback_parser.parse_intermixed_args(raw_argv[1:])
        return _build_feedback_config(feedback_args)

    # ``summarize`` is dispatched in main() before _parse_args is called, so
    # we never reach this branch from the summarize path. Kept here only as
    # a guard for callers that hand crafted-argv to _parse_args directly.

    raw_argv = _normalize_loop_argv(raw_argv)

    parser = _build_main_parser()
    _reject_removed_phase_flags(parser, raw_argv)
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

    # ``--yes`` auto-answers the fix/commit gates; ``--review``/``--comment`` run
    # no fix phase, so the flag has nothing to answer there. Reject it rather than
    # silently ignore it.
    if args.assume == "yes" and output_mode != "loop":
        parser.error("--yes has no effect with --review/--comment (no fix phase to auto-apply)")

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

    # ``--loop`` takes an optional count (``--loop`` ⇒ 5, ``--loop N`` ⇒ N).
    # ``args.loop`` is None when the flag is absent.
    loop = args.loop is not None
    max_iterations = args.loop or 5

    # Validate --loop incompatibilities
    if loop and output_mode != "loop":
        parser.error("--loop cannot be combined with --review/--comment")
    if loop and args.start_at != "review":
        parser.error("--loop requires starting at review phase (incompatible with --start-at)")

    # Detect repo slug and PR number for trajectory/archive metadata.
    # Attribute provenance to the target checkout, not the invoking cwd —
    # daydream may run from one repo against a checkout of another.
    target_repo = Path(args.target) if args.target else Path.cwd()
    pr_repo = _detect_repo_slug(target_repo)
    pr_number = _auto_detect_pr_number(target_repo)

    # Low-precedence model/backend source: [tool.daydream] / .daydream.toml at
    # the target repo root, consulted by ``_resolve_backend`` below CLI and env.
    file_config = load_file_config(target_repo)

    return RunConfig(
        target=args.target,
        skill=args.skill,
        model=args.model,
        file_config=file_config,
        # Per-phase model/backend overrides are config-file-only (no CLI flags).
        # Left None here so the config file is the sole low-precedence source.
        exploration_model=None,
        review_model=None,
        parse_model=None,
        fix_model=None,
        test_model=None,
        cleanup=args.cleanup,
        quiet=True,
        start_at=args.start_at,
        pr_number=pr_number,
        bot=None,
        backend=args.backend,
        review_backend=None,
        fix_backend=None,
        test_backend=None,
        ignore_paths=args.ignore_paths,
        loop=loop,
        max_iterations=max_iterations,
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
        non_interactive=args.non_interactive,
        assume=args.assume,
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

    target_repo = Path(args.target) if args.target else Path.cwd()
    pr_repo = _detect_repo_slug(target_repo)
    file_config = load_file_config(target_repo)

    return RunConfig(
        target=args.target,
        skill=None,
        model=args.model,
        file_config=file_config,
        # Per-phase model/backend overrides are config-file-only (no CLI flags).
        exploration_model=None,
        review_model=None,
        parse_model=None,
        fix_model=None,
        test_model=None,
        cleanup=None,
        quiet=True,
        start_at="review",
        pr_number=pr_number,
        bot=args.bot,
        backend=args.backend,
        review_backend=None,
        fix_backend=None,
        test_backend=None,
        ignore_paths=[],
        loop=False,
        max_iterations=5,
        trajectory_path=args.trajectory_path,
        pr_repo=pr_repo,
        archive=not args.no_archive,
        run_eval=args.run_eval,
        output_mode="loop",
        non_interactive=args.non_interactive,
        assume=args.assume,
    )


def _build_harvest_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream harvest [...]``.

    Drives the single deferred annotate pass from
    :mod:`daydream.training.harvest`. Every indexed run gets one fresh
    bitemporal annotation (outcome label + intrinsic reward + ``valid_at``);
    re-running appends a new generation rather than skipping annotated rows.
    """
    parser = argparse.ArgumentParser(
        prog="daydream harvest",
        description=(
            "Walk the archive and append one bitemporal annotation "
            "(outcome label + intrinsic reward) for every indexed run "
            "(RL/fine-tuning corpus prep)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Build annotations but do not write observations or the resume log.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        dest="session",
        metavar="PREFIX",
        help="Restrict the queue to session_ids starting with PREFIX.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.daydream/harvest-cache/"),
        dest="cache_dir",
        metavar="PATH",
        help="Directory backing the gh-api backfill cache (default: ~/.daydream/harvest-cache/).",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        dest="archive_dir",
        metavar="PATH",
        help="Override the archive root (default: daydream.archive.get_archive_dir()).",
    )
    parser.add_argument(
        "--repo-clone-root",
        type=Path,
        default=None,
        dest="repo_clone_root",
        metavar="PATH",
        help="Directory for cached repo clones (default: <cache-dir>/repos/).",
    )
    parser.add_argument(
        "--fix-applied-window-days",
        type=int,
        default=30,
        dest="fix_applied_window_days",
        metavar="N",
        help="Upstream-commit lookback window for the fix-applied cascade (default: 30).",
    )
    parser.add_argument(
        "--gh-spacing-sec",
        type=float,
        default=0.8,
        dest="gh_spacing_sec",
        metavar="SEC",
        help="Sleep between rows to spread gh api calls (default: 0.8).",
    )
    return parser


def _handle_harvest_command(argv: list[str]) -> int:
    """Handle ``daydream harvest [...]``.

    Drives :func:`daydream.training.harvest.run_harvest` (looked up via the
    module attribute so test monkeypatches take effect). ``run_harvest`` is a
    coroutine in production, so it is driven through :func:`anyio.run`; a
    synchronous test double that returns a summary directly is used as-is.
    Returns an exit code; ``main`` translates it to a process exit. Per-row
    harvest errors do not escalate to a non-zero exit — the summary's
    ``errors`` counter surfaces them.

    Args:
        argv: The argument vector after the ``harvest`` verb.

    Returns:
        ``0`` on success; ``1`` on a validation error.
    """
    import daydream.archive as _archive
    import daydream.training.harvest as _harvest
    from daydream.ui import create_console, print_info

    parser = _build_harvest_parser()
    args = parser.parse_args(argv)

    console = create_console()
    if args.fix_applied_window_days < 1:
        print_error(console, "Invalid --fix-applied-window-days", "Must be >= 1.")
        return 1
    if args.gh_spacing_sec < 0.0:
        print_error(console, "Invalid --gh-spacing-sec", "Must be >= 0.0.")
        return 1

    archive_dir = args.archive_dir.expanduser() if args.archive_dir is not None else _archive.get_archive_dir()
    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None

    repo_clone_root = args.repo_clone_root.expanduser() if args.repo_clone_root is not None else None

    config = _harvest.HarvestConfig(
        archive_dir=archive_dir,
        dry_run=args.dry_run,
        cache_dir=cache_dir,
        repo_clone_root=repo_clone_root,
        session_filter=args.session,
        fix_applied_window_days=args.fix_applied_window_days,
        gh_request_spacing_sec=args.gh_spacing_sec,
    )
    run_harvest = _harvest.run_harvest
    summary: dict[str, int]
    if inspect.iscoroutinefunction(run_harvest):
        summary = anyio.run(run_harvest, config)
    else:
        # A synchronous test double (monkeypatched stub) is driven directly —
        # anyio.run rejects non-coroutine callables. Production run_harvest is
        # always async, so mypy only sees the coroutine type here.
        summary = run_harvest(config)  # type: ignore[assignment]
    print_info(console, str(summary))
    return 0


def _build_label_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream label <session-prefix> --outcome ...``.

    Records a human-sourced outcome label that wins over automated rubric
    labels in every precedence projection (and is never deduped). ``unknown``
    is an allowed human outcome (per spec) — a deliberate "I looked and can't
    decide" signal distinct from an unlabeled run.
    """
    parser = argparse.ArgumentParser(
        prog="daydream label",
        description=(
            "Set an authoritative human outcome label on an archived run "
            "(overrides automated rubric labels)."
        ),
    )
    parser.add_argument(
        "session",
        type=str,
        metavar="SESSION_PREFIX",
        help="Full or prefix session_id to label (must match exactly one run).",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        required=True,
        dest="outcome",
        choices=["accepted", "contested", "rejected", "unknown"],
        help="Human outcome label to record.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        dest="archive_dir",
        metavar="PATH",
        help="Override the archive root (default: daydream.archive.get_archive_dir()).",
    )
    return parser


def _handle_label_command(argv: list[str]) -> int:
    """Handle ``daydream label <session-prefix> --outcome {...}``.

    Resolves the archive dir, echoes the label being overridden (the
    "show what it's overriding" affordance), then writes a human-sourced
    observation via :func:`daydream.archive.index.update_labels`. The runs
    cache and every precedence projection settle on the human value.

    Args:
        argv: The argument vector after the ``label`` verb.

    Returns:
        ``0`` on success; ``1`` when no session matches the prefix or the
        prefix is ambiguous.
    """
    import daydream.archive as _archive
    from daydream.archive import index as _index
    from daydream.ui import create_console, print_info

    parser = _build_label_parser()
    args = parser.parse_args(argv)

    console = create_console()
    archive_dir = args.archive_dir.expanduser() if args.archive_dir is not None else _archive.get_archive_dir()

    prior = _index.latest_label_observation(archive_dir, args.session)
    if prior is not None and prior.get("labels"):
        print_info(console, f"Current label for {args.session}: {prior['labels']}")
    else:
        print_info(console, f"No prior label for {args.session}.")

    try:
        updated = _index.update_labels(archive_dir, args.session, [args.outcome])
    except ValueError as exc:
        print_error(console, "Ambiguous session prefix", str(exc))
        return 1

    if not updated:
        print_error(console, "No matching session", f"No archived run matches prefix '{args.session}'.")
        return 1

    print_info(console, f"Set human label for {args.session}: {args.outcome}")
    return 0


def main() -> None:
    """Run the CLI entry point.

    Dispatch is verb-first (see :func:`_first_verb` and :data:`KNOWN_VERBS`):
    the leading token selects a verb, and anything that is not an explicit
    verb — a bare target path, a leading flag, or empty argv — routes through
    the default ``review`` shim. Each non-``review`` verb owns its own parser
    and exit code; ``review`` flows into :func:`_parse_args`.

    Verbs:
        - ``review`` (default) — the review/fix loop (bare ``daydream <target>``)
        - ``feedback`` — apply bot PR-review comments
        - ``summarize`` — print run-info markdown for a trajectory
        - ``bench`` — score deep-review findings against the offline benchmark
        - ``harvest`` / ``build-corpus`` / ``label`` — data-pipeline verbs

    Returns:
        None: This function does not return; it exits via sys.exit().

    Raises:
        SystemExit: Always raised with exit code 0 on success, 130 on keyboard
            interrupt, or 1 on fatal error.

    """
    _install_signal_handlers()

    # Verb-first dispatch. ``_first_verb`` classifies the leading token; bare
    # paths, leading flags, and empty argv all fall through to ``review``.
    # The non-``review`` verbs are short-circuited here (their handlers own
    # their own parsers and exit codes); ``review`` flows into ``_parse_args``
    # which applies the default-verb shim (an explicit leading ``review`` is
    # equivalent to a bare target).
    argv = sys.argv[1:]
    verb = _first_verb(argv)
    try:
        # ``summarize`` is sync — short-circuit before anyio.run kicks in.
        if verb == "summarize":
            summarize_parser = _build_summarize_parser()
            summarize_args = summarize_parser.parse_args(argv[1:])
            sys.exit(_run_summarize(summarize_args))

        # ``build-corpus`` is sync — no agent invocations, just SQLite +
        # filesystem. Short-circuit before anyio.run.
        if verb == "build-corpus":
            sys.exit(_handle_build_corpus_command(argv[1:]))

        # ``harvest`` drives the annotate-pass orchestrator via its own anyio.run.
        if verb == "harvest":
            sys.exit(_handle_harvest_command(argv[1:]))

        # ``bench`` scores deep-review findings against the offline benchmark.
        # ``run_bench`` is sync — short-circuit before anyio.run.
        if verb == "bench":
            sys.exit(_handle_bench_command(argv[1:]))

        # ``label`` records an authoritative human outcome label — sync,
        # SQLite-only. Short-circuit before anyio.run.
        if verb == "label":
            sys.exit(_handle_label_command(argv[1:]))

        config = _parse_args()
        if verb == "feedback":
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
