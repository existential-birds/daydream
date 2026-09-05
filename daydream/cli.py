"""CLI entry point for daydream.

Dispatch is verb-first. :func:`_first_verb` classifies the leading argv token
against :data:`KNOWN_VERBS`; anything that is not an explicit verb — a bare
target path, a leading flag, or empty argv — falls through to the default
``review`` shim, so ``daydream /path`` and ``daydream review /path`` are
equivalent. Each non-``review`` verb is dispatched manually from :func:`main`
before the main argparse parser runs (so its flags don't collide with the
top-level ``TARGET`` positional):

- ``daydream [review] <target>`` — the review/fix loop (default verb)
- ``daydream improve <target>`` — audit a repository and write advisory artifacts
    - ``improve plan <description> <target>`` — investigate and write one plan
- ``daydream summarize <path>`` — print run-info markdown for a trajectory
- ``daydream post-findings <artifact>`` — validate a Phase A findings artifact
  against event-derived facts and post new findings to the PR (the privileged,
  unattended Phase B poster for the Actions trigger surface)
- ``daydream corpus <sub-verb>`` — the data-pipeline namespace:
    - ``corpus harvest`` — walk the archive and append one bitemporal
      annotation (outcome label + intrinsic reward) per indexed run
    - ``corpus build --out <path>`` — project the as-of-pinned annotations
      into a JSONL training corpus plus a lineage manifest
    - ``corpus label <session-prefix> --outcome {accepted,contested,rejected,unknown}``
      — record an authoritative human outcome label that overrides automated ones
    - ``corpus hydrate-hub`` — turn a pinned private-Hub trajectory snapshot into a
      verified, sanitized, harvestable local staging archive and publish it additively
      back to the Hub under ``curated/<curation-id>/``
    - ``corpus adjudicate <build|show|label|export|report|...|publish-final>`` — per-finding
      human-label workflow: build the deterministic adjudication queue, show
      unresolved items grouped by disposition, record provenance-complete
      human observations, export the projector-shape rows (with ``--dry-run``
      validation), and report coverage / inter-rater / conflict strata
- ``daydream train --corpus <path> --out <dir>`` — run the four-stage
  training pipeline (stage0 offline gate → stage1 SFT → stage2 RFT →
  stage3 adapter) and write a stage manifest (``--dry-run`` is the GPU-free
  CI path)
- ``daydream ext validate`` — load the ``daydream_ext`` extension and
  resolve-check the registry (flows, phases, prompts)
"""

import argparse
import inspect
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import anyio
from rich.console import Console

from daydream import git_ops
from daydream.agent import (
    console,
    get_current_backends,
)
from daydream.benchmark.cli import _handle_benchmark_command
from daydream.config_file import DaydreamFileConfig, load_file_config
from daydream.phases import UnconfinedFindingError
from daydream.runner import RunConfig, run
from daydream.trajectory import get_signal_recorder
from daydream.ui import (
    ShutdownPanel,
    create_console,
    get_shutdown_panel,
    print_error,
    print_info,
    print_success,
    set_shutdown_panel,
)

if TYPE_CHECKING:
    from daydream.extensions import Registry

# Verb-first dispatch table. ``_first_verb`` classifies the leading argv token;
# anything that isn't an explicit verb (bare path, leading flag, empty argv)
# falls through to the ``review`` golden path via the default-verb shim.
KNOWN_VERBS = {
    "review",
    "improve",
    "summarize",
    "corpus",
    "train",
    "post-findings",
    "setup",
    "benchmark",
    "ext",
}

# Sub-verbs recognized under the ``improve`` verb. Single source of truth for
# improve sub-verb dispatch: ``_parse_improve_args`` strips any of these from
# argv, and ``main()`` derives its sync short-circuit set from this constant so
# the two can never drift apart.
IMPROVE_SUB_VERBS = frozenset({"plan", "prune-reanchor", "list-reanchor"})
# Sub-verbs that do no agent work and short-circuit to sync handlers in
# ``main()`` — everything except the ``plan`` flow, which routes through
# ``anyio.run(run, ...)``.
IMPROVE_SYNC_SUB_VERBS = IMPROVE_SUB_VERBS - {"plan"}


def _first_verb(argv: list[str]) -> str:
    """Classify the leading argv token into a verb.

    Returns the leading token when it is a recognized verb; otherwise returns
    ``"review"``. The fallthrough covers the three default-verb cases — empty
    argv, a leading flag, and a bare target path — so a plain
    ``daydream /path`` routes through the same parser as ``daydream review
    /path``.
    """
    if argv and argv[0] in KNOWN_VERBS:
        return argv[0]
    return "review"


def _signal_handler(signum: int, _frame: object) -> None:
    """Handle termination signals: flush partial trajectory then request shutdown.

    D-07: SIGINT/SIGTERM flushes a ``<path>.partial`` trajectory with
    ``extra.partial=true`` so consumers know the run was interrupted.

    Uses :func:`get_signal_recorder` (a module-level stack) rather than the
    ContextVar. Signal handlers fire in the main thread at bytecode boundaries
    and are not synced with the asyncio task context where the ContextVar was
    set, so ContextVar reads from here are non-deterministic.
    """
    signal_name = signal.Signals(signum).name

    # Flush partial trajectory before tearing down (D-07); write_partial is sync
    # and exception-safe, so it can't crash the shutdown path.
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
    """
    try:
        slug = git_ops.gh_repo_view(repo)
    except git_ops.GitError:
        return None
    if slug is None:
        return None
    owner, name = slug
    return f"{owner}/{name}"


def _resolve_target_provenance(target: str | None) -> tuple[Path, str | None, DaydreamFileConfig]:
    """Resolve provenance for the target checkout: repo path, slug, file config.

    Provenance is attributed to the target checkout, not the invoking cwd —
    daydream may run from one repo against a checkout of another. The returned
    file config is the low-precedence model/backend source consulted by
    ``_resolve_backend``.
    """
    target_repo = Path(target) if target else Path.cwd()
    pr_repo = _detect_repo_slug(target_repo)
    file_config = load_file_config(target_repo)
    return target_repo, pr_repo, file_config


def _add_shared_arguments(parser: argparse.ArgumentParser, *, full_help: bool = True) -> None:
    """Add the shared (non-output-mode) arguments to a parser.

    The global ``--model``/``--backend`` here feed the source-tiered precedence in
    :func:`daydream.runner._resolved_model` / ``_resolve_backend``
    (CLI > config-file phase override > config-file global > per-backend default).

    Per-phase model/backend overrides are no longer CLI flags — they live in
    ``[tool.daydream.phases.<phase>]`` of the config file (``pyproject.toml`` /
    ``.daydream.toml``). The removed flags are rejected with a curated pointer
    by :func:`_reject_removed_phase_flags`; the underlying ``RunConfig`` fields
    (``review_model``, ``fix_backend``, …) remain and are still populated from
    the config file and read by ``_resolve_backend``.

    Args:
        parser: The parser (or subparser) to add the shared arguments to.
        full_help: When False, advanced flags (``--trajectory``, ``--no-archive``,
            ``--no-eval``, ``--non-interactive``) are added with their help text
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
            "Write ATIF v1.7 trajectory JSON to this path "
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
        "--no-eval",
        action="store_false",
        default=True,
        dest="run_eval",
        help="Skip the deterministic evaluation analysis during archive "
        "(eval runs by default: it is file-based and cheap)"
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dump-artifacts",
        default=None,
        metavar="DIR",
        dest="dump_artifacts",
        help="Copy the full run bundle (ATIF trajectory, review output, deep artifacts, "
             "diffs, findings, manifest, evaluation) into DIR for CI upload. Opt-in "
             "because the logs may contain sensitive data. Works on every flow."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--review-profile",
        default=None,
        metavar="PATH",
        dest="review_profile_path",
        help=(
            "Explicit review-profile TOML path (highest-precedence source; "
            "beats DAYDREAM_REVIEW_PROFILE, the repo-committed "
            "file_config.review_profile, and the packaged default)"
        ),
    )
    parser.add_argument(
        "--trajectory-hub-repo",
        default=None,
        metavar="REPO",
        dest="trajectory_hub_repo",
        help="Upload each run's archive bundle to this HuggingFace dataset repo "
             "(owner/repo), one folder per run keyed by session id. Opt-in and "
             "requires HF_TOKEN; creates the repo private if it does not exist."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["claude", "codex", "pi", "osprey"],
        default=None,
        help="Agent backend: claude, codex, pi, or osprey "
             "(default: config file, then claude)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        type=str,
        dest="model",
        metavar="MODEL",
        help="Global default model across phases "
             "(default: config file, then the per-backend table). "
             "This global --model takes precedence over any per-phase config-file override.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        type=str,
        dest="reasoning_effort",
        metavar="EFFORT",
        help="Global reasoning-effort override (e.g. low, medium, high). "
             "Consumed by every backend through its native knob: Codex as "
             "-c model_reasoning_effort=<EFFORT>, Claude as --effort, Pi as "
             "--thinking. Takes precedence over any per-phase config-file override.",
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

    ``summarize`` is dispatched manually from ``main()`` before the main parser
    runs so its positional argument doesn't collide with the top-level
    ``TARGET``.
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
    """Build the parser for ``daydream corpus build --out <path> [...]``.

    The ``corpus build`` sub-verb is dispatched manually from ``main()`` (via
    :func:`_handle_corpus_command`) before the main parser runs so its options
    don't collide with the top-level ``TARGET`` positional.
    """
    parser = argparse.ArgumentParser(
        prog="daydream corpus build",
        description="Project as-of-pinned annotations into JSONL training records (one object per run).",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output .jsonl path",
    )

    # Filters (post-applied AFTER exclusion list)
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
    parser.add_argument(
        "--pipeline-status",
        type=str,
        default="succeeded",
        dest="pipeline_status",
        help="Match pipeline_status exactly (succeeded/failed/partial/cancelled/"
             "unknown; default: 'succeeded' — excludes failed/partial runs that "
             "archived as complete)",
    )

    # Stratification
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

    # Opt-ins
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

    # Diagnostic
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

    # Bitemporal pin
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
    """Handle ``daydream corpus build --out <path> [...]``.

    Drives :func:`daydream.training.corpus.run_build_corpus` synchronously
    (``corpus build`` does no agent work — just SQLite reads and a JSONL +
    lineage-manifest write). Returns an exit code rather than calling
    :func:`sys.exit`; ``main()`` is responsible for translating the code into a
    process exit. This keeps the handler easy to drive from tests.

    Returns:
        ``0`` on success; ``1`` on a validation error.
    """
    from daydream.training.corpus import BuildCorpusConfig, CorpusFilters, run_build_corpus
    from daydream.ui import create_console, print_error

    parser = _build_build_corpus_parser()
    args = parser.parse_args(argv)

    if not (0.0 < args.max_stack_share <= 1.0):
        print_error(create_console(), "Invalid --max-stack-share", "Must be in (0, 1].")
        return 1

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

    filters = CorpusFilters(
        repos=tuple(args.repo),
        labels=labels,
        min_grounding=args.min_grounding,
        status=args.status,
        include_all_labels=args.include_all_labels,
        allow_copyleft=frozenset(args.allow_copyleft),
        min_reward=args.min_reward,
    )
    try:
        # BuildCorpusConfig is the single validation boundary for --as-of
        # (UTC-only, canonical +00:00 spelling out).
        config = BuildCorpusConfig(
            out_path=args.out,
            filters=filters,
            pipeline_status=args.pipeline_status,
            stratify_by=args.stratify_by,
            max_stack_share=args.max_stack_share,
            dry_run=args.dry_run,
            emit_schema_only=args.emit_schema_only,
            as_of=args.as_of,
        )
    except ValueError as exc:
        print_error(create_console(), "Invalid --as-of", str(exc))
        return 1
    run_build_corpus(config)
    return 0


def _build_build_corpus_v2_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream corpus build-v2 [...]``.

    Mirrors the v1 ``corpus build`` parser in style; dispatches to the
    deterministic per-finding corpus v2 projector over a curated bundle.
    """
    parser = argparse.ArgumentParser(
        prog="daydream corpus build-v2",
        description="Project curated-bundle per-finding resolutions into deterministic, "
        "frozen-split corpus-v2 training records.",
    )

    parser.add_argument(
        "--bundle-root",
        type=Path,
        required=True,
        metavar="DIR",
        help="Hydrated curated-bundle root (must contain _SUCCESS, SHA256SUMS, "
        "curation-manifest.json)",
    )
    parser.add_argument(
        "--annotation-bundle-root",
        type=Path,
        default=None,
        dest="annotation_bundle_dir",
        metavar="DIR",
        help="Annotation-bundle root (must contain _SUCCESS, SHA256SUMS, "
        "lineage.json, annotations.jsonl); self-verified and linked to "
        "--bundle-root before the projection runs",
    )
    parser.add_argument(
        "--license-policy",
        type=Path,
        default=None,
        dest="license_policy",
        metavar="PATH",
        help="Digest-pinned license policy JSON; every record's per-repo license "
        "decision is resolved from it (required)",
    )
    parser.add_argument(
        "--annotations-snapshot",
        type=Path,
        default=None,
        dest="annotations_snapshot",
        metavar="PATH",
        help=argparse.SUPPRESS,  # deprecated: refused in the handler
    )
    parser.add_argument(
        "--repo-slug",
        type=str,
        default=None,
        dest="repo_slug",
        metavar="SLUG",
        help=argparse.SUPPRESS,  # URL-identity smuggling: refused in the handler
    )
    parser.add_argument(
        "--allow-copyleft",
        action="append",
        default=[],
        dest="allow_copyleft",
        metavar="OWNER/REPO",
        help="Repeatable; permit a specific copyleft (GPL/AGPL) repo by exact "
        "owner/repo slug (case-insensitive)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output path; corpus-v2.jsonl, the split manifests, and lineage.json "
        "are written beside it (its parent directory)",
    )

    parser.add_argument(
        "--max-stack-share",
        type=float,
        default=None,
        dest="max_stack_share",
        help="Maximum projected share of any single detected stack, in (0, 1]",
    )

    parser.add_argument(
        "--max-repo-share",
        type=float,
        default=None,
        dest="max_repo_share",
        help="Maximum projected share of any single repository slug, in (0, 1]",
    )

    parser.add_argument(
        "--max-profile-share",
        type=float,
        default=None,
        dest="max_profile_share",
        help="Maximum projected share of any single native profile, in (0, 1]",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print the projection summary, write nothing",
    )

    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        dest="as_of",
        metavar="ISO_TS",
        help="ISO-8601 transaction-time pin; evidence dated after this instant "
        "refuses the build (default: latest)",
    )

    return parser


def _handle_build_corpus_v2_command(argv: list[str]) -> int:
    """Handle ``daydream corpus build-v2 --bundle-root <dir> [...]``.

    Drives :func:`daydream.training.corpus_v2.run_build_corpus_v2` synchronously
    (no agent work, no network — a pure projection over the curated bundle plus
    the annotation bundle). Mirrors :func:`_handle_build_corpus_command`'s
    structure: returns an exit code; ``main()`` translates it into a process
    exit. Errors are fail-closed: a refused build exits non-zero with the
    exception message and writes nothing.
    """
    import tempfile
    from dataclasses import replace

    from daydream.training.corpus_v2 import BuildCorpusV2Config, run_build_corpus_v2
    from daydream.ui import create_console, print_error, print_success

    parser = _build_build_corpus_v2_parser()
    args = parser.parse_args(argv)

    for flag, value in (
        ("--max-stack-share", args.max_stack_share),
        ("--max-repo-share", args.max_repo_share),
        ("--max-profile-share", args.max_profile_share),
    ):
        if value is not None and not (0.0 < value <= 1.0):
            print_error(create_console(), f"Invalid {flag}", "Must be in (0, 1].")
            return 1

    if args.annotations_snapshot is not None:
        print_error(
            create_console(),
            "Unsupported --annotations-snapshot",
            "The side-car snapshot was replaced by the two-bundle contract; "
            "pass the self-verified annotation bundle via --annotation-bundle-root.",
        )
        return 1
    if args.annotation_bundle_dir is None:
        print_error(
            create_console(),
            "Missing --annotation-bundle-root",
            "A corpus v2 build requires a pinned annotation bundle "
            "(_SUCCESS + SHA256SUMS + lineage.json + annotations.jsonl).",
        )
        return 1
    if args.repo_slug is not None:
        print_error(
            create_console(),
            "Unsupported --repo-slug",
            "A raw remote URL (or any override slug) is never a repo identity; "
            "per-repo identity comes from the curation manifest, which the "
            "bundle gate verifies.",
        )
        return 1
    if args.license_policy is None:
        print_error(
            create_console(),
            "Missing --license-policy",
            "A corpus v2 build requires a pinned license policy file; per-repo "
            "license decisions are resolved from it (fail-closed).",
        )
        return 1

    # Validate the policy file before any build work (M10): a malformed or
    # unknown-version policy must refuse without creating the output directory.
    from daydream.training.corpus_v2.license import load_license_policy

    try:
        load_license_policy(args.license_policy)
    except (OSError, ValueError, TypeError) as exc:
        print_error(create_console(), "Invalid --license-policy", str(exc))
        return 1

    # --out names the corpus JSONL; the projector writes its canonical file set
    # (corpus.jsonl, corpus-v2.jsonl, split manifests, lineage.json) into that
    # directory, finishing with _SUCCESS — so the whole set, twin included, is
    # covered by the fail-closed completeness gate.
    out_dir = args.out.parent
    try:
        # BuildCorpusV2Config is the single validation boundary for --as-of
        # (UTC-only, canonical +00:00 spelling out) — normalize_as_of runs in
        # __post_init__, so an unparseable pin refuses here, not as a traceback.
        config = BuildCorpusV2Config(
            out_dir=out_dir,
            bundle_dir=args.bundle_root,
            annotation_bundle_dir=args.annotation_bundle_dir,
            license_policy_path=args.license_policy,
            allow_copyleft=frozenset(s.casefold() for s in args.allow_copyleft),
            as_of=args.as_of,
            max_stack_share=args.max_stack_share,
            max_repo_share=args.max_repo_share,
            max_profile_share=args.max_profile_share,
        )
    except ValueError as exc:
        print_error(create_console(), "Invalid --as-of", str(exc))
        return 1
    try:
        if args.dry_run:
            with tempfile.TemporaryDirectory() as td:
                summary = run_build_corpus_v2(replace(config, out_dir=Path(td)))
        else:
            summary = run_build_corpus_v2(config)
    except (OSError, ValueError, TypeError) as exc:
        print_error(create_console(), "Corpus v2 build refused", str(exc))
        return 1
    print_success(
        create_console(),
        f"Corpus v2 build complete: {summary['emitted']} records "
        f"({summary['adjudication']} to adjudication) in {out_dir}",
    )
    return 0


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
        default=None,
        dest="improve_focus",
        help=(
            "Narrow the audit: a single category, 'branch' to audit only the "
            "diff against the base branch"
        ),
    )
    parser.add_argument(
        "--scope",
        default=None,
        metavar="SERVICE_OR_GLOB",
        dest="improve_scope",
        help=(
            "Restrict the audit to one service, a glob over service roots, or a "
            "named group from [tool.daydream.improve.service_groups]"
        ),
    )
    _add_shared_arguments(parser)
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
    args = _build_improve_parser(subverb).parse_args(improve_argv)
    _, pr_repo, file_config = _resolve_target_provenance(args.target)
    return RunConfig(
        target=args.target,
        backend=args.backend,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        file_config=file_config,
        review_profile_path=args.review_profile_path,
        trajectory_path=args.trajectory_path,
        pr_repo=pr_repo,
        archive=not args.no_archive,
        run_eval=args.run_eval,
        dump_artifacts=args.dump_artifacts,
        trajectory_hub_repo=args.trajectory_hub_repo,
        non_interactive=args.non_interactive,
        assume=args.assume,
        flow_name="improve",
        improve_effort=args.improve_effort,
        improve_focus=args.improve_focus,
        improve_scope=args.improve_scope,
        improve_plan_description=getattr(
            args, "improve_plan_description", None
        ),
        improve_prune_name=getattr(args, "improve_prune_name", None),
    )


class _HelpAllAction(argparse.Action):
    """Print the full help (advanced flags included) and exit.

    The default ``--help`` is built with ``full_help=False`` so advanced flags
    are suppressed. ``--help-all`` re-builds the parser with ``full_help=True``
    and renders that help instead, surfacing every flag without changing how
    any of them parse.
    """

    def __init__(  # noqa: A002 - `help` is argparse.Action's API parameter name
        self,
        option_strings: list[str],
        dest: str = argparse.SUPPRESS,
        default: Any = argparse.SUPPRESS,
        help: str | None = None,
    ):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

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
        help="Show all flags, including advanced ones, then exit.",
    )

    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        metavar="TARGET",
        help="Target directory (default: prompt interactively).",
    )

    # Output mode (mutually exclusive; default = fix-loop)
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
    # Issue #1113. The value is REQUIRED (no ``nargs="?"``): with an optional
    # value, ``--diagram-only /path`` would consume the target positional as
    # the kind and then fail as an invalid choice.
    output_group.add_argument(
        "--diagram-only",
        choices=["auto", "sequence", "flowchart", "both"],
        default=None,
        dest="diagram_only",
        help="Run only the grounded-diagram flow and post a standalone PR comment, "
             "then exit (auto = whatever is eligible).",
    )
    parser.add_argument(
        "--diagram",
        choices=["auto", "sequence", "flowchart", "both", "off"],
        default=None,
        dest="diagram",
        help="Control grounded diagrams in the review paths (default: auto -- render "
             "every eligible kind). Use --diagram-only for the diagram-only mode."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        dest="log_mode",
        help="Bypass Rich UI and emit redacted agent events as plain text to stdout."
        if full_help else argparse.SUPPRESS,
    )

    # Selection
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

    # Modifiers
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
        "--flow",
        default=None,
        metavar="NAME",
        dest="flow_name",
        help="Dispatch a registered flow by name (built-in: deep/shallow/review; "
             "or a daydream_ext custom flow). Built-in names behave like their "
             "dedicated flag." if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--precision",
        action="store_true",
        default=False,
        dest="precision",
        help="Enable precision mode: run a skeptical suppression pass over borderline "
             "findings after the arbiter (issue #232; fail-closed). Also settable "
             "via [tool.daydream] precision_mode in a config file."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--approve-on-clean",
        action="store_true",
        default=False,
        dest="approve_on_clean",
        help="Approve the PR when a deep review has zero high/medium findings "
             "(issue #343): post event: 'APPROVE' instead of 'COMMENT'. Also "
             "settable via [tool.daydream] approve_on_clean in a config file."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--file-scope-issues",
        action="store_true",
        default=False,
        dest="file_scope_issues",
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
        default=None,
        metavar="PATH",
        dest="findings_out",
        help="Write a strict-schema findings artifact (Phase A emission for "
             "`daydream post-findings`; works with the default deep review flow or --review)."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pr-number",
        default=None,
        type=int,
        metavar="N",
        dest="pr_number",
        help="Pin the target PR number (trajectory metadata and the --findings-out "
             "artifact target; default: auto-detect from the current branch)."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--approved-head-sha",
        default=None,
        dest="approved_head_sha",
        help="Pin the maintainer-approved PR head SHA."
        if full_help else argparse.SUPPRESS,
    )

    # Stack selection (overrides auto-detect)
    parser.add_argument(
        "-s", "--stack",
        choices=["python", "react", "elixir", "go", "rust", "ios"],
        default=None,
        dest="stack",
        help="Force a specific stack (default: auto-detect from changed files)",
    )

    # Cleanup / phase resume
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
            "Choices: review | fix | ttt | per-stack | merge. "
            "parse/test are legacy shallow-loop stages with no mapping in the "
            "unified pipeline and are rejected. "
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

    _add_shared_arguments(parser, full_help=full_help)

    return parser


# Removed per-phase model/backend flags → their config-file replacement in
# ``[tool.daydream.phases.<phase>]``. The RunConfig fields they set remain
# (still read by ``_resolve_backend``, settable via config) — just not CLI-settable.
_REMOVED_PHASE_FLAGS: dict[str, str] = {
    "--review-backend": "[tool.daydream.phases.review] backend = \"...\"",
    "--fix-backend": "[tool.daydream.phases.fix] backend = \"...\"",
    "--test-backend": "[tool.daydream.phases.test] backend = \"...\"",
    "--exploration-model": "[tool.daydream.phases.exploration] model = \"...\"",
    "--review-model": "[tool.daydream.phases.review] model = \"...\"",
    "--per-stack-review-model": "[tool.daydream.phases.per_stack_review] model = \"...\"",
    "--arbiter-model": "[tool.daydream.phases.arbiter] model = \"...\"",
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
    output-mode flags (``--comment`` / ``--review``), selection flags
    (``--branch`` / ``--base``), and modifiers (``--worktree`` / ``--shallow`` /
    ``--copy``). Deep is the default; ``--shallow`` opts into single-stack mode.
    """
    raw_argv = sys.argv[1:] if argv is None else list(argv)

    # Default-verb shim: strip an explicit leading ``review`` so it parses
    # identically to the bare ``daydream <target>`` form.
    if raw_argv and raw_argv[0] == "review":
        raw_argv = raw_argv[1:]

    parser = _build_main_parser()
    _reject_removed_phase_flags(parser, raw_argv)
    args = parser.parse_args(raw_argv)

    # Resolve output mode
    output_mode: str = "loop"
    if args.comment:
        output_mode = "comment"
    elif args.review:
        output_mode = "review"
    elif args.diagram_only is not None:
        output_mode = "diagram"

    # Issue #1113: the two diagram flags mean different things (one modifies the
    # review paths, one replaces them), so combining them is a request with two
    # incompatible answers. Reject rather than pick one.
    if args.diagram is not None and args.diagram_only is not None:
        parser.error("--diagram cannot be combined with --diagram-only")

    # ``--yes`` answers the fix/commit gates, which --review/--comment/
    # --diagram-only don't run; reject rather than silently ignore.
    if args.assume == "yes" and output_mode != "loop":
        parser.error(
            "--yes has no effect with --review/--comment/--diagram-only "
            "(no fix phase to auto-apply)"
        )

    findings_out_allowed = (
        output_mode in ("review", "diagram")
        or (output_mode == "loop" and not args.shallow)
    )
    if args.findings_out is not None and not findings_out_allowed:
        parser.error(
            "--findings-out requires --review, --diagram-only, or the default deep "
            "review flow (not --comment/--shallow)"
        )

    # Issue #1113: the diagram flow writes none of the artifacts a resume stage
    # attests, so every --start-at value is meaningless for it. Reject at the
    # CLI rather than silently accepting and ignoring it.
    if args.diagram_only is not None and args.start_at != "review":
        parser.error(f"--start-at {args.start_at} is not valid with --diagram-only")

    # ttt/per-stack/merge are deep-pipeline resume stages; not valid for shallow.
    if args.shallow and args.start_at in ("ttt", "per-stack", "merge"):
        parser.error(f"--start-at {args.start_at} is not valid with --shallow")

    # parse/test are legacy shallow-loop resume points (#330). The unified
    # pipeline has two parse phases and no single test phase, so "resume at
    # parse/test" has no mapping — their artifacts and phases are gone. Reject
    # them loudly in every mode rather than silently treating the run as fresh
    # (which, combined with --yes, would re-review and apply+commit fixes the
    # user did not ask to re-run).
    if args.start_at in ("parse", "test"):
        parser.error(
            f"--start-at {args.start_at} has no mapping in the unified pipeline "
            "(the legacy shallow-loop phases are gone); "
            "use --start-at fix to resume after the merged report"
        )

    if args.flow_name is not None:
        if args.comment or args.review or args.diagram_only is not None:
            parser.error("--flow cannot be combined with --review/--comment/--diagram-only")
        if args.shallow:
            parser.error("--flow cannot be combined with --shallow")

    target_repo, pr_repo, file_config = _resolve_target_provenance(args.target)
    # Explicit --pr-number pins the target PR; otherwise auto-detect from branch.
    pr_number = args.pr_number if args.pr_number is not None else _auto_detect_pr_number(target_repo)

    return RunConfig(
        target=args.target,
        stack=args.stack,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        file_config=file_config,
        review_profile_path=args.review_profile_path,
        # Per-phase overrides are config-file-only; left None so config is the
        # sole low-precedence source.
        exploration_model=None,
        review_model=None,
        parse_model=None,
        fix_model=None,
        test_model=None,
        cleanup=args.cleanup,
        quiet=True,
        start_at=args.start_at,
        pr_number=pr_number,
        approved_head_sha=args.approved_head_sha,
        backend=args.backend,
        review_backend=None,
        fix_backend=None,
        test_backend=None,
        ignore_paths=args.ignore_paths,
        trajectory_path=args.trajectory_path,
        pr_repo=pr_repo,
        archive=not args.no_archive,
        run_eval=args.run_eval,
        branch=args.branch,
        base=args.base,
        output_mode=output_mode,  # type: ignore[arg-type]
        # Issue #1113: --diagram-only's value IS the run's diagram mode, so an
        # explicit request wins over a repo file's ``mode = "off"``.
        diagram=args.diagram_only if args.diagram_only is not None else args.diagram,
        findings_out=args.findings_out,
        dump_artifacts=args.dump_artifacts,
        trajectory_hub_repo=args.trajectory_hub_repo,
        force_worktree=args.force_worktree,
        shallow=args.shallow,
        flow_name=args.flow_name,
        precision_mode=args.precision,
        approve_on_clean=args.approve_on_clean,
        scope_issue_filing=args.file_scope_issues,
        extra_copy=list(args.extra_copy),
        non_interactive=args.non_interactive,
        assume=args.assume,
        log_mode=args.log_mode,
    )


def _build_harvest_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream corpus harvest [...]``.

    Drives the single deferred annotate pass from
    :mod:`daydream.training.harvest`. Every indexed run gets one fresh
    bitemporal annotation (outcome label + intrinsic reward + ``valid_at``);
    re-running appends a new generation rather than skipping annotated rows.
    """
    parser = argparse.ArgumentParser(
        prog="daydream corpus harvest",
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
        "--gh-spacing-sec",
        type=float,
        default=0.8,
        dest="gh_spacing_sec",
        metavar="SEC",
        help="Sleep between rows to spread gh api calls (default: 0.8).",
    )
    return parser


def _handle_harvest_command(argv: list[str]) -> int:
    """Handle ``daydream corpus harvest [...]``.

    Drives :func:`daydream.training.harvest.run_harvest` (looked up via the
    module attribute so test monkeypatches take effect). ``run_harvest`` is a
    coroutine in production, so it is driven through :func:`anyio.run`; a
    synchronous test double that returns a summary directly is used as-is.
    Returns an exit code; ``main`` translates it to a process exit. An
    aborted summary (``aborted >= 1``) or any per-row harvest errors
    (``errors > 0``) exits ``1``; a clean partial completion with unresolved
    findings still exits ``0`` (unresolved findings in the data are not
    process failure).

    Returns:
        ``0`` on success; ``1`` on a validation error or an aborted/errored
        harvest summary.
    """
    import daydream.archive as _archive
    import daydream.training.harvest as _harvest
    from daydream.ui import create_console, print_info

    parser = _build_harvest_parser()
    args = parser.parse_args(argv)

    console = create_console()
    if args.gh_spacing_sec < 0.0:
        print_error(console, "Invalid --gh-spacing-sec", "Must be >= 0.0.")
        return 1

    archive_dir = args.archive_dir.expanduser() if args.archive_dir is not None else _archive.get_archive_dir()
    cache_dir = args.cache_dir.expanduser()

    repo_clone_root = args.repo_clone_root.expanduser() if args.repo_clone_root is not None else None

    config = _harvest.HarvestConfig(
        archive_dir=archive_dir,
        dry_run=args.dry_run,
        cache_dir=cache_dir,
        repo_clone_root=repo_clone_root,
        session_filter=args.session,
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
    if summary.get("aborted", 0) >= 1 or summary.get("errors", 0) > 0:
        return 1
    return 0


def _build_hydrate_hub_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream corpus hydrate-hub [...]``.

    Drives :func:`daydream.archive.hydrate.run_hydrate_hub`: pin a private-Hub
    snapshot revision, download it resumably into a staging directory, run the
    fail-closed ingest gate, and publish the sanitized output additively under
    ``curated/<curation-id>/`` — reporting success only after the clean-room
    verification cycle passes.
    """
    parser = argparse.ArgumentParser(
        prog="daydream corpus hydrate-hub",
        description=(
            "Hydrate a pinned private-Hub trajectory snapshot into a verified, "
            "sanitized, harvestable staging archive and publish it additively "
            "back to the Hub under curated/<curation-id>/ (issue #982)."
        ),
    )
    parser.add_argument(
        "--source-repo",
        type=str,
        required=True,
        dest="source_repo",
        metavar="REPO_ID",
        help="Source private Hub dataset repo (e.g. org/ds) holding the snapshot.",
    )
    parser.add_argument(
        "--source-revision",
        type=str,
        required=True,
        dest="source_revision",
        metavar="REV",
        help=(
            "Pinned source commit: an exact 40-char SHA or unique hex prefix. "
            "Moving branches/tags require --exploratory."
        ),
    )
    parser.add_argument(
        "--destination-repo",
        type=str,
        required=True,
        dest="destination_repo",
        metavar="REPO_ID",
        help="Destination private Hub repo for the sanitized output (must be private).",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        required=True,
        dest="stage_dir",
        metavar="PATH",
        help="Local staging directory for downloads, ingest, and the rebuildable index.",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        dest="exploratory",
        help="Opt in to a moving branch/tag source revision (output is non-canonical).",
    )
    parser.add_argument(
        "--license-policy",
        type=Path,
        default=None,
        dest="license_policy",
        metavar="PATH",
        help="Digest-pinned license policy JSON; REQUIRED for publication "
        "(omitting it on a non-dry run refuses before any Hub access); the "
        "per-repo license admission gate runs at hydration and rejected "
        "sessions are excluded before publication; optional for --dry-run "
        "planning (issue #1094, previously #1080)",
    )
    parser.add_argument(
        "--allow-copyleft",
        action="append",
        default=[],
        dest="allow_copyleft",
        metavar="OWNER/REPO",
        help="Repeatable; permit a specific copyleft (GPL/AGPL) repo by exact "
        "owner/repo slug (case-insensitive); only meaningful with --license-policy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Plan only: discover and normalize sessions, download, ingest, and tally "
            "discovered/admitted/rejected candidates — no Hub publication."
        ),
    )
    return parser


def _build_calibrate_reward_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream corpus calibrate-reward``.

    Drives :func:`daydream.training.calibration.run_calibration`: validate a
    pinned calibration bundle (wire format in ``docs/calibration.md``)
    fail-closed, compute deterministic calibration statistics with bootstrap
    CIs, and emit a byte-reproducible calibration artifact — without touching
    any reward default (issue #999).
    """
    parser = argparse.ArgumentParser(
        prog="daydream corpus calibrate-reward",
        description=(
            "Validate a pinned calibration bundle fail-closed and emit a deterministic, "
            "versioned reward-calibration artifact with Stage-0 marginal "
            "analysis (issue #999). Never mutates reward defaults."
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        dest="corpus_dir",
        metavar="PATH",
        help="Calibration bundle directory holding corpus.jsonl + lineage.json + SHA256SUMS.",
    )
    parser.add_argument(
        "--gold-labels",
        type=Path,
        required=True,
        dest="gold_labels",
        metavar="PATH",
        help='Gold labels JSON keyed by record_id: {"<record_id>": {"accepted": bool}}.',
    )
    parser.add_argument(
        "--breakdowns",
        type=Path,
        required=True,
        dest="breakdowns",
        metavar="PATH",
        help="Intrinsic per-axis breakdowns JSON keyed by record_id (composite excluded).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        dest="out_dir",
        metavar="PATH",
        help="Output directory for the calibration artifact and report.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        dest="run_id",
        metavar="ID",
        help="Unique run identifier recorded in the artifact (collision-guarded).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        dest="seed",
        metavar="N",
        help="Resampling seed; the artifact is byte-reproducible given this seed.",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        action="append",
        required=True,
        dest="candidates",
        metavar="AXIS=V1,V2,...",
        help=(
            "Candidate grid for one breakdown axis, e.g. w_fp=0.1,0.2,0.3. "
            "Repeatable; every value comes from the flag — never from defaults."
        ),
    )
    parser.add_argument(
        "--stage0-scores",
        type=Path,
        dest="stage0_scores",
        metavar="PATH",
        help=(
            'Optional Stage-0 score JSON keyed by record_id: '
            '{"<record_id>": {"score": float, "model_digest": str}}.'
        ),
    )
    parser.add_argument(
        "--model-digest",
        type=str,
        dest="model_digest",
        metavar="DIGEST",
        help="Digest of the Stage-0 model, required when --stage0-scores is given.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=9,
        dest="grid_points",
        metavar="N",
        help="Grid resolution per candidate axis (default: 9).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        dest="bootstrap_resamples",
        metavar="N",
        help="Bootstrap resample count for AUC CIs (default: 1000).",
    )
    return parser


def _parse_candidates(raw: list[str]) -> dict[str, list[float]]:
    """Parse ``AXIS=V1,V2,...`` candidate flags into a grid mapping.

    Raises:
        ValueError: on a flag missing ``=``, holding non-float points, or
            repeating an axis already given in an earlier flag.
    """
    candidates: dict[str, list[float]] = {}
    for spec in raw:
        axis, sep, points = spec.partition("=")
        if not sep or not axis.strip() or not points.strip():
            raise ValueError(
                f"--candidate {spec!r} must be AXIS=V1,V2,... (comma-separated floats)"
            )
        try:
            values = [float(p) for p in points.split(",")]
        except ValueError as exc:
            raise ValueError(f"--candidate {spec!r} has non-float grid points") from exc
        axis = axis.strip()
        if axis in candidates:
            raise ValueError(
                f"--candidate {spec!r} repeats axis {axis!r} (already given as "
                f"{candidates[axis]}); pass all points for one axis in a single flag"
            )
        candidates[axis] = values
    return candidates


def _run_calibration(config: Any) -> Any:
    """Thin module-attribute wrapper around the calibration orchestrator.

    Lives at module level so tests can monkeypatch ``cli._run_calibration``
    (the same seam discipline as harvest's ``run_harvest`` lookup).
    """
    from daydream.training.calibration import run_calibration

    return run_calibration(config)


def _handle_calibrate_reward_command(argv: list[str]) -> int:
    """Handle ``daydream corpus calibrate-reward [...]``.

    Fail-closed: every validation gate in
    :mod:`daydream.training.calibration` surfaces as a ``CalibrationError``
    printed to stderr with exit 1, before any artifact is written. Success
    prints the artifact path plus headline metrics and exits 0.

    Returns:
        ``0`` on a validated, artifact-emitting run; ``1`` on any
        validation or gate failure.
    """
    from daydream.training import calibration as _calibration
    from daydream.ui import create_console, print_error, print_info

    parser = _build_calibrate_reward_parser()
    console = create_console()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            raise
        print_error(
            console,
            "Invalid arguments",
            "corpus calibrate-reward requires --corpus-dir, --gold-labels, "
            "--breakdowns, --out, --run-id, --seed, and at least one --candidate.",
        )
        return 1

    try:
        candidates = _parse_candidates(args.candidates)
    except ValueError as exc:
        print_error(console, "Invalid --candidate", str(exc))
        return 1

    config = _calibration.CalibrationConfig(
        corpus_dir=args.corpus_dir,
        gold_labels=args.gold_labels,
        breakdowns=args.breakdowns,
        out_dir=args.out_dir,
        run_id=args.run_id,
        seed=args.seed,
        candidates=candidates,
        stage0_scores=args.stage0_scores,
        model_digest=args.model_digest,
        grid_points=args.grid_points,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    try:
        summary = _run_calibration(config)
    except _calibration.CalibrationError as exc:
        print_error(console, "Calibration failed", str(exc))
        return 1

    print_info(
        console,
        f"calibration artifact: {config.out_dir / 'calibration.json'} "
        f"(run {summary.get('run_id', config.run_id)}, "
        f"{summary.get('record_count', '?')} records, "
        f"schema {summary.get('schema_version', '?')})",
    )
    return 0


def _run_hydrate_hub(config: Any) -> Any:
    """Thin module-attribute wrapper around the hydrate orchestrator.

    Lives at module level so tests can monkeypatch ``cli._run_hydrate_hub``
    (the same seam discipline as harvest's ``run_harvest`` lookup).
    """
    from daydream.archive.hydrate import run_hydrate_hub

    return run_hydrate_hub(config)


def _handle_hydrate_hub_command(argv: list[str]) -> int:
    """Handle ``daydream corpus hydrate-hub [...]``.

    Fail-closed, fatal semantics (unlike ``hub.py``'s warn-everything): a
    missing ``HF_TOKEN``, ``GITHUB_TOKEN``, or a moving-branch revision without
    ``--exploratory`` exits 1 before any Hub access. Success prints the
    immutable output commit SHA plus a value-free summary (counts and
    reason-code tallies only). Any ``HydrationError`` is ``redact_text``-
    processed before display.

    Returns:
        ``0`` on a verified run (or a completed ``--dry-run`` plan); ``1`` on
        any validation or hydration failure.
    """
    import os

    from daydream.archive import hydrate as _hydrate
    from daydream.trajectory import redact_text
    from daydream.ui import create_console, print_warning

    parser = _build_hydrate_hub_parser()
    console = create_console()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            raise
        print_error(
            console,
            "Invalid arguments",
            "corpus hydrate-hub requires --source-repo, --source-revision, "
            "--destination-repo, and --stage-dir.",
        )
        return 1

    # Moving-branch rejection is checkable locally (no client, no token): a
    # revision that is neither a full SHA nor a hex prefix is a symbolic ref.
    # Case-fold only for the precheck regexes: a symbolic ref is forwarded
    # case-sensitively so a case-sensitive branch/tag never silently maps to a
    # differently-cased name (hydrate folds hex only inside its hex branches).
    revision = args.source_revision.strip()
    if not args.exploratory and not (
        _hydrate._FULL_SHA_RE.fullmatch(revision.lower())
        or _hydrate._HEX_PREFIX_RE.fullmatch(revision.lower())
    ):
        print_error(
            console,
            "Moving source revision",
            f"ref {revision!r} is a moving branch/tag, not a pinned commit; pass "
            "--exploratory to accept it (output is non-canonical), or pin an exact "
            "40-char commit SHA.",
        )
        return 1

    # Token precheck: fail closed before any Hub access; name the env var,
    # never the token.
    if not os.environ.get("HF_TOKEN"):
        print_error(
            console,
            "HF_TOKEN is not set",
            "hydration requires a read token for the private Hub repo. "
            "Export HF_TOKEN and retry.",
        )
        return 1

    # Issue #1094: a non-dry hydrate-hub publication requires a pinned license
    # policy; refuse before any Hub access or staging work. A dry-run may omit
    # it (planning affordance). Mirrors build-v2's policy-required pattern.
    if args.license_policy is None and not args.dry_run:
        print_error(
            console,
            "Missing --license-policy",
            "A hydrate-hub publication requires a pinned license policy file; "
            "per-repo license admission decisions are resolved from it "
            "(fail-closed). A --dry-run may omit it.",
        )
        return 1

    # Issue #1094: license-evidence enrichment (live GitHub license API calls)
    # is a hard runtime requirement on every non-dry publication. Fail closed
    # before any Hub access or staging work, naming the variable and the fix —
    # an unset token would otherwise surface only as a redacted generic 401
    # failure after download/ingest/dedupe have run. A --dry-run may omit it
    # (planning affordance; the resolver itself fail-fasts with this same
    # message if a policy-driven dry run reaches enrichment without one).
    if not args.dry_run and not os.environ.get("GITHUB_TOKEN"):
        print_error(
            console,
            "GITHUB_TOKEN is not set",
            "license evidence enrichment requires a GitHub API token for the "
            "license endpoint. Export GITHUB_TOKEN and retry.",
        )
        return 1

    # Pre-validate the license policy up front so a malformed or missing
    # --license-policy is named by this handler (and redaction-processed) instead
    # of escaping as an unredacted generic "Fatal Error" from main(). Mirrors the
    # build-v2 handler's fail-closed validation.
    if args.license_policy is not None:
        from daydream.training.corpus_v2.license import load_license_policy

        try:
            load_license_policy(args.license_policy)
        except (OSError, ValueError, TypeError) as exc:
            print_error(console, "Invalid --license-policy", str(exc))
            return 1

    stage_dir = args.stage_dir.expanduser()
    config = _hydrate.HydrateHubConfig(
        source_repo=args.source_repo,
        source_revision=revision,
        destination_repo=args.destination_repo,
        stage_dir=stage_dir,
        exploratory=args.exploratory,
        license_policy_path=(
            str(args.license_policy) if args.license_policy is not None else None
        ),
        allow_copyleft=frozenset(s.casefold() for s in args.allow_copyleft),
    )
    if args.dry_run:
        return _hydrate_hub_dry_run(config, console)

    try:
        summary = globals()["_run_hydrate_hub"](config)
    except _hydrate.HydrationError as exc:
        print_error(console, "Hydration failed", redact_text(str(exc)))
        return 1
    if not summary.verified:
        print_error(
            console,
            "Hydration not verified",
            "the clean-room verification cycle did not pass; no success marker "
            "was published.",
        )
        return 1
    print_success(
        console,
        f"hydration verified: curation {summary.curation_id} published at commit "
        f"{summary.output_commit_sha}",
    )
    print_info(
        console,
        f"dry-run discovered {summary.dry_run_discovered} "
        f"candidate(s); admitted {summary.dry_run_admitted} batch(es); "
        f"rejected {summary.dry_run_rejected} batch(es); "
        f"verify admitted {summary.verify_admitted} batch(es)",
    )
    if summary.license_admission:
        buckets = summary.license_admission
        print_info(
            console,
            "license admission: "
            f"admitted {buckets['admitted']}; c5-excluded {buckets['c5_excluded']}; "
            f"copyleft-unopted {buckets['c8_copyleft_unopted']}; "
            f"evidence-missing {buckets['license_evidence_missing']}",
        )
    if summary.dry_run_incomplete_manifests:
        print_warning(
            console,
            "hydration yield reduced: incomplete manifest(s) discovered and "
            "dropped: " + redact_text("; ".join(summary.dry_run_incomplete_manifests)),
        )
    return 0


def _hydrate_hub_dry_run(config: Any, console: Any) -> int:
    """Plan-only hydrate pass: pin, download, ingest, tally — no publication.

    Prints the per-code tallies plus per-repository license decision counts
    (value-free slugs and counts, issue #1094), with a fail-closed accounting
    invariant: every license-adjudicated candidate (imported plus license-gate
    rejections) lands in exactly one per-repository license bucket.
    """
    from daydream.archive import hydrate as _hydrate
    from daydream.trajectory import redact_text
    from daydream.ui import print_warning

    try:
        client = _hydrate._make_client(config.source_repo)
        source_commit = _hydrate.resolve_source_revision(
            client, config.source_revision, exploratory=config.exploratory
        )
        _hydrate.download_snapshot(client, revision=source_commit, stage_dir=config.stage_dir / "downloads")
        _hydrate.ingest_bundles(config.stage_dir, revision=source_commit)
        _hydrate.dedupe_admitted(config.stage_dir, revision=source_commit)
        binding = None
        if config.license_policy_path is not None:
            # Issue #1094: the dry-run derives and reports the v2 candidate id
            # from the same binding inputs the publication will use — enrich
            # (production resolver), gate, then the post-gate binding.
            from daydream.archive.license_enrich import (
                _make_license_resolver,
                enrich_license_evidence,
            )

            enrich_license_evidence(
                config.stage_dir, revision=source_commit, resolver=_make_license_resolver(),
            )
            _hydrate.restamp_admitted_digests(config.stage_dir, revision=source_commit)
            _hydrate.apply_license_gate(
                config.stage_dir,
                revision=source_commit,
                license_policy_path=config.license_policy_path,
                allow_copyleft=config.allow_copyleft,
            )
            binding = _hydrate.resolve_curation_identity(
                config.stage_dir,
                source_commit=source_commit,
                license_policy_path=config.license_policy_path,
                allow_copyleft=config.allow_copyleft,
            )
        ledger = _hydrate.build_import_ledger(
            config.stage_dir, revision=source_commit, source_commit=source_commit,
            binding=binding,
        )
        license_admission = (
            _hydrate.license_admission_summary(ledger)
            if config.license_policy_path is not None
            else {}
        )
        # Issue #1094 Task 8: per-repo auditable decision counts. Value-free
        # (slugs + counts only). License accounting is enforced here — every
        # license-adjudicated candidate (imported sessions plus license-gate
        # rejections; ingest/fixture rejections are never adjudicated by the
        # license gate and are reported via the ledger's non-license rejection
        # tallies) must land in exactly one per-repo bucket, mirroring
        # license_admission_summary's population.
        per_repo = (
            _hydrate.license_admission_by_repo(config.stage_dir, ledger)
            if config.license_policy_path is not None
            else {}
        )
        if config.license_policy_path is not None:
            per_repo_total = sum(sum(b.values()) for b in per_repo.values())
            adjudicated_total = sum(license_admission.values())
            if per_repo_total != adjudicated_total:
                raise _hydrate.HydrationError(
                    redact_text(
                        f"per-repository accounting mismatch for revision "
                        f"{source_commit!r}: license-adjudicated population "
                        f"{adjudicated_total} candidate(s) (imported plus "
                        f"license-gate rejections), per-repository buckets "
                        f"total {per_repo_total}"
                    )
                )
    except _hydrate.HydrationError as exc:
        print_error(console, "Hydration dry-run failed", redact_text(str(exc)))
        return 1
    tallies = ledger.get("tallies", {})
    rejections = ledger.get("rejections", [])
    reason_tally: dict[str, int] = {}
    for rejection in rejections:
        code = str(rejection.get("reason_code"))
        reason_tally[code] = reason_tally.get(code, 0) + 1
    print_info(
        console,
        f"dry-run plan for curation {ledger.get('curation_id')}: pinned {source_commit}; "
        f"discovered {tallies.get('discovered', 0)} candidate(s) of "
        f"{tallies.get('run_shaped_manifests', 0)} run-shaped manifest(s); "
        f"admitted {tallies.get('imported', 0)} batch(es); "
        f"rejected {len(rejections)} batch(es); "
        f"accounted {tallies.get('accounted', 0)} candidate(s); "
        f"reason codes: {reason_tally or 'none'}; no publication performed",
    )
    if license_admission:
        print_info(
            console,
            "license admission: "
            f"admitted {license_admission['admitted']}; "
            f"c5-excluded {license_admission['c5_excluded']}; "
            f"copyleft-unopted {license_admission['c8_copyleft_unopted']}; "
            f"evidence-missing {license_admission['license_evidence_missing']}",
        )
    for repo_slug in sorted(per_repo):
        buckets = per_repo[repo_slug]
        print_info(
            console,
            f"license admission by repo: {repo_slug} -> "
            f"admitted {buckets['admitted']}, "
            f"c5-excluded {buckets['c5_excluded']}, "
            f"copyleft-unopted {buckets['c8_copyleft_unopted']}, "
            f"evidence-missing {buckets['license_evidence_missing']}",
        )
    incomplete = [str(item) for item in tallies.get("incomplete_manifests", [])]
    if incomplete:
        print_warning(
            console,
            "dry-run yield reduced: incomplete manifest(s) discovered and "
            "dropped: " + redact_text("; ".join(incomplete)),
        )
    return 0


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

    from daydream.improve.plans import reanchored_plan_rows
    from daydream.ui import create_console, print_info

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


def _build_label_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream corpus label <session-prefix> --outcome ...``.

    Records a human-sourced outcome label that wins over automated rubric
    labels in every precedence projection (and is never deduped). ``unknown``
    is an allowed human outcome (per spec) — a deliberate "I looked and can't
    decide" signal distinct from an unlabeled run.
    """
    parser = argparse.ArgumentParser(
        prog="daydream corpus label",
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
    """Handle ``daydream corpus label <session-prefix> --outcome {...}``.

    Resolves the archive dir, echoes the label being overridden (the
    "show what it's overriding" affordance), then writes a human-sourced
    observation via :func:`daydream.archive.index.update_labels`. The runs
    cache and every precedence projection settle on the human value.

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


class _TrainParser(argparse.ArgumentParser):
    """Train parser that maps usage errors to exit code 1.

    The train verb reports refusals (including conflicting inputs) as a
    validation failure with exit 1 rather than argparse's default 2, so
    harnesses see one failure code for "the run was refused".
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _build_train_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream train --corpus <path> --out <dir> [...]``.

    Dispatched manually from ``main()`` (verb-first) so its flags don't
    collide with the top-level ``TARGET`` positional.
    """
    parser = _TrainParser(
        prog="daydream train",
        description=(
            "Run the four-stage training pipeline (stage0 gate → stage1 SFT → "
            "stage2 RFT → stage3 adapter) and write a stage manifest."
        ),
    )
    corpus_group = parser.add_mutually_exclusive_group(required=True)
    corpus_group.add_argument(
        "--corpus",
        type=Path,
        metavar="PATH",
        help="Input JSONL training corpus (one record per line; C5/C8 fail-closed)",
    )
    corpus_group.add_argument(
        "--corpus-v2",
        type=Path,
        dest="corpus_v2",
        metavar="DIR",
        help="Frozen corpus-v2 projection directory; verifies _SUCCESS/lineage/"
             "split digests and re-applies C5/C8",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory for stageN/ artifacts and manifest.json",
    )
    parser.add_argument(
        "--stage",
        action="append",
        dest="stages",
        choices=["stage0", "stage1", "stage2", "stage3"],
        default=None,
        help="Repeatable; run only these stages in the order given (default: all four)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-8B",
        dest="base_model",
        help="HuggingFace base model id the LoRA adapter trains against",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed (split freeze + training determinism; default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Execute everything that needs no GPU (corpus load, stage0 gate, "
             "manifest) and mark the GPU stages skipped_dry — the CI path",
    )
    return parser


def _handle_train_command(argv: list[str]) -> int:
    """Handle ``daydream train [...]``.

    Drives :func:`daydream.training.coordinator.run_pipeline` synchronously
    (pure filesystem + SQLite-free work; no agent backend, no GPU on the dry
    path). Returns an exit code rather than calling ``sys.exit``; ``main()``
    translates the code into the process exit.

    Returns:
        ``0`` on success; ``1`` on a gate refusal or validation error.
    """
    from daydream.training.coordinator import PipelineConfig, run_pipeline

    parser = _build_train_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(
        corpus=args.corpus,
        corpus_v2=args.corpus_v2,
        out_dir=args.out,
        stages=tuple(args.stages) if args.stages else ("stage0", "stage1", "stage2", "stage3"),
        base_model=args.base_model,
        seed=args.seed,
    )
    try:
        run_pipeline(config, dry_run=args.dry_run)
    except (OSError, RuntimeError, ValueError) as exc:
        print_error(create_console(), "Training run refused", str(exc))
        return 1
    print_success(create_console(), f"Training run complete: {args.out / 'manifest.json'}")
    return 0


# Sub-verbs of the ``corpus`` namespace mapped to their handler callables.
# ``build`` is the public name for the build-corpus projection.
def _handle_adjudicate_command(argv: list[str]) -> int:
    """Handle ``daydream corpus adjudicate <sub-verb> [...]`` (issue #984).

    Thin wrapper over the per-finding adjudication workflow handlers
    (``build``/``show``/``label`` in :mod:`daydream.training.adjudication.cli`);
    exit codes propagate unchanged, and argparse rejects bare or unknown
    sub-verbs with usage + exit 2.
    """
    from daydream.training.adjudication.cli import handle_adjudicate

    return handle_adjudicate(argv)


_CORPUS_SUBVERBS: dict[str, Callable[[list[str]], int]] = {
    "harvest": _handle_harvest_command,
    "build": _handle_build_corpus_command,
    "build-v2": _handle_build_corpus_v2_command,
    "label": _handle_label_command,
    "hydrate-hub": _handle_hydrate_hub_command,
    "calibrate-reward": _handle_calibrate_reward_command,
    "adjudicate": _handle_adjudicate_command,
}


_CORPUS_USAGE = (
    "usage: daydream corpus {harvest,build,build-v2,label,hydrate-hub,calibrate-reward,adjudicate} ...\n"
    "\n"
    "Data-pipeline sub-verbs:\n"
    "  harvest   walk the archive and append one bitemporal annotation per indexed run\n"
    "  build     project the as-of-pinned annotations into a JSONL training corpus\n"
    "  build-v2  project curated-bundle resolutions into corpus-v2 records (pinned --license-policy required)\n"
    "  label     record an authoritative human outcome label that overrides automated ones\n"
    "  hydrate-hub  hydrate a pinned Hub snapshot into a sanitized, verified staging archive\n"
    "  calibrate-reward  validate a calibration bundle and emit a deterministic reward-calibration artifact\n"
    "  adjudicate  per-finding human-label workflow: build/show/label/export/report, then"
    "\n"
    "  materialize/harvest/publish the annotation snapshot"
)

_EXT_USAGE = (
    "usage: daydream ext {validate} ...\n"
    "\n"
    "Extension sub-verbs:\n"
    "  validate   load the daydream_ext extension and resolve-check the registry"
)


def _print_namespace_help(usage: str, *, error: bool = False) -> None:
    """Print a namespace usage block.

    Args:
        usage: The usage text to print.
        error: When ``True`` write to stderr (unknown sub-verb error path);
            when ``False`` (default) write to stdout (bare invocation / help
            request path).
    """
    from rich.console import Console

    from daydream.ui import NEON_THEME

    Console(stderr=error, theme=NEON_THEME).print(usage)


def _dispatch_namespace(argv: list[str], subverbs: dict[str, Callable[[list[str]], int]], usage: str) -> int:
    """Dispatch a namespace sub-verb to its handler.

    A bare invocation (no sub-verb) prints usage to stdout and exits 2; an
    unknown sub-verb prints usage to stderr and exits 2. Exit codes propagate
    unchanged from the handlers.
    """
    if not argv:
        _print_namespace_help(usage)
        return 2
    handler = subverbs.get(argv[0])
    if handler is None:
        _print_namespace_help(usage, error=True)
        return 2
    return int(handler(argv[1:]))


def _handle_corpus_command(argv: list[str]) -> int:
    """Dispatch a ``corpus`` sub-verb to its handler.

    ``corpus harvest|build|label`` routes to the existing data-pipeline
    handlers (``build`` → the build-corpus projection).

    Returns:
        int: The sub-handler's exit code; ``2`` for a bare (no-arg)
        invocation or an unknown sub-verb.
    """
    return _dispatch_namespace(argv, _CORPUS_SUBVERBS, _CORPUS_USAGE)


def _ext_resolve_failure(registry: "Registry") -> str | None:
    """Resolve-check the registry; return the first failure message, or None.

    Runs ``run_flow``'s pre-flight pass (every flow entry — including
    loop-group bodies — must name a registered phase), then checks that every
    step's config key is a string and that every prompt and renderer resolves
    to a callable.
    """
    from daydream.extensions import UnresolvedExtensionError
    from daydream.flows.engine import _resolve_steps

    for flow_name in registry.flow_names():
        try:
            _resolve_steps(registry, flow_name, registry.flow(flow_name))
        except UnresolvedExtensionError as exc:
            return str(exc)
    for name in registry.phase_names():
        phase_key = registry.phase(name).phase_key
        if not isinstance(phase_key, str):
            return f"phase '{name}' has a non-string config key: {phase_key!r}"
    for name in registry.prompt_names():
        if not callable(registry.prompt(name)):
            return f"prompt slot '{name}' does not resolve to a callable"
    for name in registry.renderer_names():
        if not callable(registry.renderer(name)):
            return f"renderer slot '{name}' does not resolve to a callable"
    return None


def _handle_ext_validate_command() -> int:
    """Handle ``daydream ext validate``.

    Builds the per-run registry for real (builtins seeded, extension module
    discovered, version-gated, and applied), reports the extension source and
    API version, reports tool-supervisor registration, then resolve-checks the
    registry. Runs anywhere —
    validation is registry-shaped, not repo-shaped, so no target directory is
    required.

    Returns:
        int: ``0`` when the registry resolves clean; ``1`` when the extension
        fails to load or a registered piece does not resolve.
    """
    import importlib.util
    import os

    from daydream.extensions import (
        EXTENSION_API_VERSION,
        MIN_SUPPORTED_EXTENSION_API_VERSION,
        ExtensionError,
        build_registry,
    )

    try:
        registry = build_registry()
    except ExtensionError as exc:
        print_error(console, "Extension Error", str(exc))
        return 1

    ext_dir = os.environ.get("DAYDREAM_EXT_DIR")
    if ext_dir:
        source = f"extension source: $DAYDREAM_EXT_DIR = {ext_dir}"
    elif importlib.util.find_spec("daydream_ext") is not None:
        source = "extension source: import daydream_ext"
    else:
        source = "extension source: no extension found (builtins only)"
    console.print(source, soft_wrap=True)
    console.print(
        f"extension API version {EXTENSION_API_VERSION} "
        f"(supported: {MIN_SUPPORTED_EXTENSION_API_VERSION}..{EXTENSION_API_VERSION})"
    )
    supervisor_status = "registered" if registry.tool_supervisor_if_registered() is not None else "none"
    console.print(f"tool supervisor: {supervisor_status}")

    failure = _ext_resolve_failure(registry)
    if failure is not None:
        print_error(console, "Extension Error", failure)
        return 1

    console.print(
        f"registry OK: {len(registry.phase_names())} phases, "
        f"{len(registry.flow_names())} flows, "
        f"{len(registry.prompt_names())} prompts, "
        f"{len(registry.renderer_names())} renderers"
    )
    return 0


def _handle_ext_command(argv: list[str]) -> int:
    """Dispatch an ``ext`` sub-verb (mirrors the ``corpus`` namespace shape).

    A bare ``daydream ext`` prints help to stdout and exits 2; an unknown
    sub-verb (or trailing arguments — ``validate`` takes none) prints help to
    stderr and exits 2.

    Returns:
        int: The sub-handler's exit code; ``2`` for a bare (no-arg)
        invocation or an unknown sub-verb.
    """
    if argv[:1] == ["validate"] and argv[1:]:
        _print_namespace_help(_EXT_USAGE, error=True)
        return 2
    return _dispatch_namespace(argv, {"validate": lambda _argv: _handle_ext_validate_command()}, _EXT_USAGE)


def _build_post_findings_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream post-findings <artifact> --pr ... --head-sha ... --repo ...``.

    The privileged Phase B poster is an unattended CI verb: it takes the
    findings artifact produced by ``--findings-out`` plus the event-derived
    target facts, and posts when validation passes — no prompting.
    """
    parser = argparse.ArgumentParser(
        prog="daydream post-findings",
        description=(
            "Validate a Phase A findings artifact against event-derived facts and "
            "post new findings to the PR (privileged Phase B poster; unattended)."
        ),
    )
    parser.add_argument(
        "artifact",
        type=Path,
        metavar="ARTIFACT",
        help="Path to the findings artifact written by --findings-out.",
    )
    parser.add_argument(
        "--pr",
        type=int,
        required=True,
        dest="pr_number",
        metavar="N",
        help="Event-derived target PR number.",
    )
    parser.add_argument(
        "--head-sha",
        type=str,
        required=True,
        dest="head_sha",
        metavar="SHA",
        help="Event-derived PR head SHA the artifact must declare.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        dest="repo",
        metavar="OWNER/REPO",
        help="Event-derived repository slug the artifact must declare.",
    )
    parser.add_argument(
        "--bot-login",
        type=str,
        default=None,
        dest="bot_login",
        metavar="LOGIN",
        help="Bot login (App slug) for prior-finding author filtering. "
        "Defaults to $DAYDREAM_BOT_HANDLE.",
    )
    parser.add_argument(
        "--approve-on-clean",
        action="store_true",
        default=False,
        dest="approve_on_clean",
        help="Approve the PR when the posted findings contain no high/medium "
        "severity issues (issue #343): post event: 'APPROVE' instead of "
        "'COMMENT'. Also settable via [tool.daydream] approve_on_clean in a "
        "repo config file.",
    )
    return parser


def _handle_post_findings_command(argv: list[str]) -> int:
    """Handle ``daydream post-findings <artifact> --pr N --head-sha SHA --repo OWNER/REPO``.

    Delegates to :func:`daydream.pr_review.post_findings_from_artifact` —
    validate (confused-deputy gate, before any GitHub write), reconcile
    against prior comments, minimize stale findings, post new ones. Sync: no
    agent work, no ATIF trajectory.

    Returns:
        ``0`` on success (including "no new findings"); ``1`` on validation,
        inventory, or post failure.
    """
    from daydream import pr_review
    from daydream.ui import create_console, print_warning

    parser = _build_post_findings_parser()
    args = parser.parse_args(argv)
    if "/" not in args.repo:
        parser.error(f"--repo must be an OWNER/REPO slug, got {args.repo!r}")

    console = create_console()
    # Best-effort config read: the poster previously never consulted the repo
    # config, so a malformed .daydream.toml/pyproject.toml in the CI checkout
    # must not abort the unattended post — warn and fall back to the CLI flag.
    approve = args.approve_on_clean
    try:
        approve = approve or bool(load_file_config(Path.cwd()).approve_on_clean)
    except ValueError as exc:
        print_warning(console, f"Ignoring malformed repo config: {exc}")
    return pr_review.post_findings_from_artifact(
        args.artifact,
        pr_number=args.pr_number,
        head_sha=args.head_sha,
        repo=args.repo,
        console=console,
        bot_login=args.bot_login,
        approve_on_clean=approve,
    )


def _build_setup_parser() -> argparse.ArgumentParser:
    """Build the parser for ``daydream setup <target> --repo o/r | --org name [--verify] [--force]``.

    The ``setup`` verb takes an operator from nothing to a live self-hosted
    review bot: register the GitHub App via the manifest browser flow, deposit
    credentials as Actions secrets, and land the workflows via a reviewable PR.
    ``--verify`` runs the read-only doctor instead. ``--repo`` and ``--org`` are
    mutually exclusive (exactly one is the deposit scope).
    """
    parser = argparse.ArgumentParser(
        prog="daydream setup",
        description=(
            "Set up a self-hosted Daydream review bot: register the GitHub App, "
            "deposit credentials as Actions secrets, and land the workflows via a PR."
        ),
    )
    parser.add_argument(
        "target",
        type=Path,
        metavar="TARGET",
        help="Path to the repository working directory to set the bot up in.",
    )
    scope_group = parser.add_mutually_exclusive_group(required=True)
    scope_group.add_argument(
        "--repo",
        type=str,
        dest="repo",
        metavar="OWNER/REPO",
        help="Deposit secrets/variables and install at repository scope.",
    )
    scope_group.add_argument(
        "--org",
        type=str,
        dest="org",
        metavar="NAME",
        help="Deposit secrets/variables and install at organization scope.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the read-only setup doctor instead of performing setup.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-register the App even if credentials are already deposited.",
    )
    return parser


def _handle_setup_command(argv: list[str]) -> int:
    """Handle ``daydream setup <target> --repo o/r | --org name [--verify] [--force]``.

    Dispatches to :func:`daydream.bot_setup.run_verify` (when ``--verify``) or
    :func:`daydream.bot_setup.run_setup`. Sync: the App-from-manifest leg runs a
    local browser handshake, not agent work, so there is no ATIF trajectory.
    ``GitHubAppError``/``GitError`` are caught here and surfaced via
    :func:`print_error` — never a traceback to the user.

    Returns:
        ``0`` on success; ``1`` on a verify failure or any setup error.
    """
    from daydream import bot_setup
    from daydream.git_ops import GitError
    from daydream.github_app import GitHubAppError

    parser = _build_setup_parser()
    args = parser.parse_args(argv)
    if args.repo is not None and "/" not in args.repo:
        parser.error(f"--repo must be an OWNER/REPO slug, got {args.repo!r}")

    scope = bot_setup.Scope(repo=args.repo, org=args.org)
    target = args.target

    try:
        if args.verify:
            result = bot_setup.run_verify(target, scope=scope)
            bot_setup.print_verify_result(result)
            return 0 if result.ok else 1
        return bot_setup.run_setup(target, scope=scope, force=args.force, anthropic_key=None)
    except (GitHubAppError, GitError) as exc:
        print_error(console, "Setup failed", str(exc))
        return 1


def _handle_prune_reanchor(config: RunConfig) -> int:
    """Remove one named ``-reanchor`` worktree; sync cleanup, no improve flow.

    Returns ``0`` on removal and ``1`` for every other verdict, so the exit
    code reads as a reliable removal contract for scripts/operators.
    """
    from daydream.improve.plans import (
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
    if outcome.verdict == PRUNE_NOT_FOUND:
        print_error(console, "Prune re-anchor", f"No such worktree: {name}")
    elif outcome.verdict == PRUNE_NOT_REANCHOR:
        print_error(
            console, "Prune re-anchor", f"{name!r} is not a -reanchor worktree"
        )
    elif outcome.verdict == PRUNE_UNSAFE_NAME:
        print_error(
            console, "Prune re-anchor", f"{name!r} is not a safe worktree name"
        )
    else:  # PRUNE_GIT_FAILURE
        print_error(console, "Prune re-anchor", f"Could not remove {name}")
    return 1


def _handle_list_reanchor(config: RunConfig) -> int:
    """List existing ``-reanchor`` worktrees; sync, no improve flow.

    An empty list still exits ``0`` — listing nothing is not an error.
    """
    from daydream.improve.plans import list_reanchor_worktrees

    console = create_console()
    repo = Path(config.target) if config.target else Path.cwd()
    for path in list_reanchor_worktrees(repo):
        print_info(console, path.name)
    return 0


def _shutdown_and_exit(console: Console, title: str, message: str) -> None:
    """Finish the shutdown panel, print ``title``/``message``, and exit 1.

    Shared by :func:`main`'s error handlers so the panel-finish sequence
    (``get_shutdown_panel`` -> ``finish`` -> ``set_shutdown_panel(None)`` ->
    ``console.print`` -> ``print_error`` -> ``sys.exit(1)``) lives in one place.
    """
    panel = get_shutdown_panel()
    if panel is not None:
        panel.finish()
        set_shutdown_panel(None)
    console.print()
    print_error(console, title, message)
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Run the CLI entry point.

    ``argv`` defaults to ``sys.argv[1:]``; tests may pass an explicit list.

    Dispatch is verb-first (see :func:`_first_verb` and :data:`KNOWN_VERBS`):
    the leading token selects a verb, and anything that is not an explicit
    verb — a bare target path, a leading flag, or empty argv — routes through
    the default ``review`` shim. Each non-``review`` verb owns its own parser
    and exit code; ``review`` flows into :func:`_parse_args`.

    Verbs:
        - ``review`` (default) — the review/fix loop (bare ``daydream <target>``)
        - ``summarize`` — print run-info markdown for a trajectory
        - ``corpus`` — data-pipeline namespace (``harvest`` / ``build`` / ``label`` /
          ``hydrate-hub``)
        - ``train`` — four-stage training pipeline (``--dry-run`` for GPU-free CI)
        - ``post-findings`` — validate a findings artifact and post new
          findings to the PR (privileged Phase B poster; unattended)
        - ``setup`` — register the review-bot GitHub App, deposit credentials,
          and land the workflows via a PR (``--verify`` for the doctor)
        - ``ext`` — extension namespace (``validate`` loads the
          ``daydream_ext`` extension and resolve-checks the registry)
        - ``improve`` — repository audit + advisory plans; sub-verbs
          ``prune-reanchor`` / ``list-reanchor`` short-circuit to sync
          filesystem cleanup, everything else runs the async audit flow

    Raises:
        SystemExit: Always raised with exit code 0 on success, 130 on keyboard
            interrupt, or 1 on fatal error.

    """
    _install_signal_handlers()

    # Verb-first dispatch: non-``review`` verbs are short-circuited here (each
    # owns its parser and exit code); everything else flows into ``_parse_args``.
    argv = list(argv) if argv is not None else sys.argv[1:]
    # ``bench`` was the legacy benchmark verb, removed in favor of
    # ``daydream benchmark``. Reject it explicitly instead of letting
    # ``_first_verb`` fall through to the review path with ``bench`` as a
    # bare target.
    if argv and argv[0] == "bench":
        print(
            "error: the 'bench' command is no longer a command; use 'daydream benchmark'",
            file=sys.stderr,
        )
        sys.exit(2)
    verb = _first_verb(argv)
    try:
        # ``summarize`` is sync — short-circuit before anyio.run kicks in.
        if verb == "summarize":
            summarize_parser = _build_summarize_parser()
            summarize_args = summarize_parser.parse_args(argv[1:])
            sys.exit(_run_summarize(summarize_args))

        # ``corpus`` namespaces the data-pipeline sub-verbs; all sync (SQLite +
        # filesystem, no agent work), so short-circuit before anyio.run.
        if verb == "corpus":
            sys.exit(_handle_corpus_command(argv[1:]))

        # ``train`` is sync (filesystem-only coordination, no agent work and
        # no GPU on the dry path), so short-circuit before anyio.run.
        if verb == "train":
            sys.exit(_handle_train_command(argv[1:]))

        if verb == "benchmark":
            sys.exit(_handle_benchmark_command(argv[1:]))

        if verb == "post-findings":
            sys.exit(_handle_post_findings_command(argv[1:]))

        if verb == "setup":
            sys.exit(_handle_setup_command(argv[1:]))

        # ``ext`` is sync (registry build + resolve-check, no agent work), so
        # short-circuit before anyio.run.
        if verb == "ext":
            sys.exit(_handle_ext_command(argv[1:]))

        # ``improve prune-reanchor`` / ``list-reanchor`` are pure filesystem
        # cleanup (no agent work), so short-circuit to a sync handler instead of
        # routing every improve invocation through anyio.run(run, ...). Peek
        # ``argv[1]`` exactly as ``_parse_improve_args`` does; the set comes from
        # :data:`IMPROVE_SYNC_SUB_VERBS`, derived from the shared
        # :data:`IMPROVE_SUB_VERBS` so dispatch can't drift.
        if verb == "improve" and (
            argv[1] if len(argv) > 1 else None
        ) in IMPROVE_SYNC_SUB_VERBS:
            config = _parse_improve_args(argv)
            if argv[1] == "prune-reanchor":
                sys.exit(_handle_prune_reanchor(config))
            sys.exit(_handle_list_reanchor(config))

        # ``improve list-reanchored`` is a sync, read-only one-purpose command
        # (mirroring the corpus/ext short-circuits), so it never spins up
        # a flow through ``_parse_improve_args``/``anyio.run``.
        if verb == "improve" and len(argv) > 1 and argv[1] == "list-reanchored":
            sys.exit(_handle_list_reanchored_command(argv[2:]))

        config = (
            _parse_improve_args(argv)
            if verb == "improve"
            else _parse_args()
        )
        exit_code = anyio.run(run, config)
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
        # Defense-in-depth fallback for the fix preflight confinement rejection
        # (the primary path routes it through ``_step_fix``'s recovery; this
        # renders actionably if it ever escapes). Matched by type, not by
        # message, so a ValueError from elsewhere is never misattributed to an
        # unconfined finding; any other ValueError falls through to the generic
        # handler below. The message must not cite the fix_failures artifact:
        # that file is written only by ``_step_fix``'s recovery path, so when
        # the exception escapes to here, no artifact exists.
        _shutdown_and_exit(
            console,
            "Unconfined Finding",
            f"{e}. Check the finding's file ref.",
        )
    except Exception as e:
        _shutdown_and_exit(console, "Fatal Error", str(e))


if __name__ == "__main__":
    main()
