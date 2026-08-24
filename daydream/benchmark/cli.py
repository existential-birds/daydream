"""CLI helpers for the ``daydream bench`` subcommand.

These helpers are called from :func:`daydream.cli.main` when ``bench`` is the
first argv token. They live here rather than in the top-level ``daydream.cli``
module to keep that file below the 1 000-line threshold and to co-locate the
bench argument-parsing logic with the rest of the benchmark package.

``bench`` carries two sub-verbs: ``daydream bench harvest`` builds a corpus
from a review bot's PR history (see :mod:`daydream.benchmark.harvest`) and
``daydream bench manifest`` folds a harvested corpus into a compact,
git-tracked ``manifest.json`` (see :mod:`daydream.benchmark.corpus_manifest`).
Every other argv shape is a benchmark run over one corpus — a withmartian
checkout (``--benchmark-repo``) or a harvested dir (``--harvest-dir``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dotenv

if TYPE_CHECKING:
    from daydream.benchmark import BenchConfig


def _load_bench_dotenv() -> None:
    """Load a ``.env`` from the invocation cwd so benchmark credentials can live there.

    Reads ``.env`` from the operator's current working directory (``usecwd=True``;
    the library default walks up from this module's file instead). ``override``
    is left at its default ``False`` so inline environment variables still win
    over the file. A missing or malformed ``.env`` is a silent no-op.
    """
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))


def _format_elapsed(seconds: float) -> str:
    """Render an elapsed duration as a compact human string.

    Returns:
        ``"{n}s"`` for durations under a minute, else ``"{m}m{s}s"`` with the
        seconds component not zero-padded (e.g. ``252`` -> ``"4m12s"``).
    """
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m{total % 60}s"


def _build_bench_parser() -> argparse.ArgumentParser:
    """Build the parser for the ``daydream bench`` subcommand.

    Kept as its own parser (not an argparse subparser of the main one) so the
    main parser's positional ``TARGET`` doesn't collide with the verb. We
    dispatch to this parser from ``main`` based on argv[0].
    """
    parser = argparse.ArgumentParser(
        prog="daydream bench",
        description="Score daydream's deep-review findings against the code-review-benchmark offline set.",
    )
    parser.add_argument(
        "--benchmark-repo",
        type=Path,
        default=None,
        dest="benchmark_repo",
        metavar="PATH",
        help="Path to the external code-review-benchmark checkout "
        "(optional when [tool.daydream.bench] benchmark-repo is set)",
    )
    parser.add_argument(
        "--harvest-dir",
        type=Path,
        default=None,
        dest="harvest_dir",
        metavar="PATH",
        help="Root of a harvested bot-review corpus (see 'daydream bench harvest'); "
        "mutually exclusive with --benchmark-repo and requires an in-process judge "
        "route (anthropic-direct or openai-compatible)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        dest="cache_dir",
        metavar="PATH",
        help="Directory for per-PR blobless clones (default: <benchmark-repo>/.daydream-bench/cache)",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        dest="trajectory_dir",
        metavar="PATH",
        help="Directory for per-PR ATIF trajectory files (default: <benchmark-repo>/.daydream-bench/trajectories)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        dest="model",
        help="Judge model id (e.g. anthropic/claude-opus-4-5-20251101). If omitted, the route-specific "
        "environment fallback is used; one of the two is required for --score. "
        "Whatever resolves drives both the judge and the per-model results dir.",
    )
    parser.add_argument(
        "--judge-route",
        type=str,
        choices=["martian", "anthropic-direct", "openai-compatible"],
        default=None,
        dest="judge_route",
        help="Benchmark scoring route (default: martian, or [tool.daydream.bench] judge-route)",
    )
    parser.add_argument(
        "--reviewer",
        type=str,
        default=None,
        dest="reviewer",
        metavar="NAME",
        help="Expand a [tool.daydream.bench.reviewers.<NAME>] preset into backend/model/provider "
        "and derive --tool-label as daydream-<NAME>; explicit --reviewer-*/--tool-label flags override",
    )
    parser.add_argument(
        "--reviewer-backend",
        type=str,
        choices=["claude", "codex", "pi", "osprey"],
        default=None,
        dest="reviewer_backend",
        help="Backend for the reviewer under test (default: daydream's built-in default)",
    )
    parser.add_argument(
        "--reviewer-model",
        type=str,
        default=None,
        dest="reviewer_model",
        help="Model id for the reviewer under test (default: the backend's default)",
    )
    parser.add_argument(
        "--reviewer-provider",
        type=str,
        default=None,
        dest="reviewer_provider",
        help="Provider for the reviewer under test, forwarded as PI_PROVIDER (pi backend only)",
    )
    parser.add_argument(
        "--tool-label",
        type=str,
        default=None,
        dest="tool_label",
        help="Results key for this reviewer; MUST be distinct per reviewer backend or runs overwrite each other "
        "(default: daydream, or daydream-<NAME> when --reviewer is set)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        dest="only",
        metavar="SELECTOR",
        help="Restrict the run to PRs whose source repo or golden URL contains this substring",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        dest="limit",
        metavar="N",
        help="Cap the number of PRs processed",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        dest="trials",
        metavar="N",
        help="Run each reviewer config N times (default: 1). N>1 isolates each trial "
        "and enables distribution reporting (mean/median/stddev/bootstrap CI).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        dest="force",
        help="Re-run PRs even if a daydream review already exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        dest="verbose",
        help="Stream the review subprocess output live instead of a quiet spinner",
    )
    parser.add_argument(
        "--score",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="score",
        help="Drive the step2/2.5/3 scoring pipeline (default: on; use --no-score to skip)",
    )
    parser.add_argument(
        "--min-confidence",
        choices=["LOW", "MEDIUM", "HIGH"],
        default=None,
        dest="min_confidence",
        help="Drop findings below this confidence from benchmark submission (default: submit all)",
    )
    parser.add_argument(
        "--min-severity",
        choices=["low", "medium", "high"],
        default=None,
        dest="min_severity",
        help="Drop findings below this severity from benchmark submission (default: submit all)",
    )
    return parser


def _resolve_reviewer_preset(
    name: str, bench_cfg: dict, parser: argparse.ArgumentParser
) -> dict[str, Any]:
    """Look up a named reviewer preset in the bench config table.

    Args:
        bench_cfg: The ``[tool.daydream.bench]`` table from ``load_file_config``.
        parser: The bench parser, used to emit a usage error (``SystemExit``)
            when the preset is unknown.

    Returns:
        The preset dict with ``backend``/``model``/``provider`` keys.
    """
    reviewers = bench_cfg.get("reviewers", {})
    preset = reviewers.get(name) if isinstance(reviewers, dict) else None
    if not isinstance(preset, dict):
        parser.error(
            f"unknown --reviewer '{name}' (define [tool.daydream.bench.reviewers.{name}] in config)"
        )
    return preset


def _bench_config_from_argv(argv: list[str]) -> "BenchConfig":
    """Parse ``daydream bench`` argv into a :class:`BenchConfig`.

    Exactly one of ``--benchmark-repo`` / ``--harvest-dir`` must resolve (flag or
    ``[tool.daydream.bench]`` key); optional path flags fall back to a
    ``.daydream-bench`` dir under whichever corpus root that is. No directories
    are created at parse time.
    """
    from daydream.benchmark import BenchConfig
    from daydream.config_file import load_file_config

    parser = _build_bench_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.trials is not None and args.trials <= 0:
        parser.error("--trials must be a positive integer")
    if (
        args.tool_label is None
        and args.reviewer is None
        and (args.reviewer_backend is not None or args.reviewer_model is not None or args.reviewer_provider is not None)
    ):
        parser.error(
            "--reviewer-backend/--reviewer-model/--reviewer-provider require --tool-label "
            "(or a --reviewer preset) so per-backend results stay isolated"
        )
    bench = load_file_config(Path.cwd()).bench
    # P1: CLI flag > config file > built-in default.
    benchmark_repo = (
        args.benchmark_repo
        if args.benchmark_repo is not None
        else Path(bench["benchmark-repo"])
        if "benchmark-repo" in bench
        else None
    )
    model = args.model if args.model is not None else bench.get("model")
    judge_route = args.judge_route if args.judge_route is not None else bench.get("judge-route", "martian")
    if judge_route not in {"martian", "anthropic-direct", "openai-compatible"}:
        parser.error("--judge-route must be one of: martian, anthropic-direct, openai-compatible")
    min_confidence = args.min_confidence if args.min_confidence is not None else bench.get("min-confidence")
    min_severity = args.min_severity if args.min_severity is not None else bench.get("min-severity")
    trials = args.trials if args.trials is not None else bench.get("trials", 1)
    if not isinstance(trials, int) or trials <= 0:
        parser.error("--trials must be a positive integer")
    if min_confidence is not None and min_confidence.lower() not in {"low", "medium", "high"}:
        parser.error("--min-confidence must be one of: LOW, MEDIUM, HIGH")
    if min_severity is not None and min_severity.lower() not in {"low", "medium", "high"}:
        parser.error("--min-severity must be one of: low, medium, high")
    harvest_dir = (
        args.harvest_dir
        if args.harvest_dir is not None
        else Path(bench["harvest-dir"])
        if "harvest-dir" in bench
        else None
    )
    if benchmark_repo is not None and harvest_dir is not None:
        parser.error("--benchmark-repo and --harvest-dir are mutually exclusive (a run has exactly one corpus)")
    corpus_root = harvest_dir if harvest_dir is not None else benchmark_repo
    if corpus_root is None:
        parser.error(
            "one of --benchmark-repo / --harvest-dir is required "
            "(pass the flag or set [tool.daydream.bench] benchmark-repo / harvest-dir)"
        )
    if harvest_dir is not None and args.score and judge_route == "martian":
        # The martian route shells `uv run python -m code_review_benchmark.step*`
        # with cwd=<corpus root>; that package only exists inside the withmartian
        # checkout, so a harvested corpus can only be scored in-process. Gated on
        # --score because the route is never driven when scoring is off.
        parser.error(
            "--judge-route martian requires --benchmark-repo; score a harvested corpus "
            "with an in-process route (anthropic-direct or openai-compatible)"
        )
    bench_root = corpus_root / ".daydream-bench"
    cache_dir = args.cache_dir if args.cache_dir is not None else bench_root / "cache"
    trajectory_dir = args.trajectory_dir if args.trajectory_dir is not None else bench_root / "trajectories"
    # P1: a --reviewer preset is the config layer under explicit --reviewer-*/--tool-label flags.
    preset: dict[str, Any] = {}
    if args.reviewer is not None:
        preset = _resolve_reviewer_preset(args.reviewer, bench, parser)
    reviewer_backend = args.reviewer_backend if args.reviewer_backend is not None else preset.get("backend")
    reviewer_model = args.reviewer_model if args.reviewer_model is not None else preset.get("model")
    reviewer_provider = args.reviewer_provider if args.reviewer_provider is not None else preset.get("provider")
    tool_label = (
        args.tool_label
        if args.tool_label is not None
        else f"daydream-{args.reviewer}"
        if args.reviewer is not None
        else "daydream"
    )
    return BenchConfig(
        benchmark_repo=benchmark_repo,
        cache_dir=cache_dir,
        force=args.force,
        score=args.score,
        only=args.only,
        limit=args.limit,
        trajectory_dir=trajectory_dir,
        judge_route=judge_route,
        model=model,
        reviewer_backend=reviewer_backend,
        reviewer_model=reviewer_model,
        reviewer_provider=reviewer_provider,
        tool_label=tool_label,
        verbose=args.verbose,
        min_confidence=min_confidence,
        min_severity=min_severity,
        trials=trials,
        harvest_dir=harvest_dir,
    )


def _build_bench_harvest_parser() -> argparse.ArgumentParser:
    """Build the parser for the ``daydream bench harvest`` sub-verb."""
    parser = argparse.ArgumentParser(
        prog="daydream bench harvest",
        description="Harvest a review bot's historic PR reviews into a benchmark corpus. "
        "(Unrelated to 'daydream corpus harvest', which annotates archived daydream runs.)",
    )
    parser.add_argument("--repo", required=True, metavar="OWNER/REPO", help="Repository to scan")
    parser.add_argument(
        "--bot",
        required=True,
        metavar="LOGIN",
        help="Bot login, e.g. 'coderabbitai[bot]'; the '[bot]' suffix is optional",
    )
    parser.add_argument("--out", required=True, type=Path, metavar="DIR", help="Corpus output directory")
    parser.add_argument("--limit", type=int, default=200, metavar="N", help="Max PRs to scan (default: 200)")
    parser.add_argument(
        "--state",
        default="all",
        choices=["all", "open", "closed", "merged"],
        help="PR state filter (default: all)",
    )
    return parser


def _handle_bench_harvest_command(argv: list[str]) -> int:
    """Handle ``daydream bench harvest --repo O/R --bot LOGIN --out DIR [...]``."""
    from daydream.benchmark.harvest import run_harvest

    parser = _build_bench_harvest_parser()
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be a positive integer")
    return run_harvest(args.repo, args.bot, args.out, limit=args.limit, state=args.state)


def _build_bench_manifest_parser() -> argparse.ArgumentParser:
    """Build the parser for the ``daydream bench manifest`` sub-verb."""
    parser = argparse.ArgumentParser(
        prog="daydream bench manifest",
        description="Fold a harvested bot-review corpus into a compact, git-tracked manifest.json.",
    )
    parser.add_argument(
        "--harvest-dir",
        required=True,
        type=Path,
        metavar="DIR",
        dest="harvest_dir",
        help="Root of a harvested bot-review corpus (must contain index.json and harvest/)",
    )
    return parser


def _handle_bench_manifest_command(argv: list[str]) -> int:
    """Handle ``daydream bench manifest --harvest-dir DIR``.

    Regenerates ``DIR/manifest.json`` from ``index.json`` + ``harvest/pr-*.json``.
    Returns ``2`` (not an exception) when the corpus is incomplete or
    internally inconsistent, mirroring the harvest sub-verb's failure
    convention.
    """
    from daydream.agent import console
    from daydream.benchmark.corpus_manifest import write_corpus_manifest
    from daydream.ui import print_info, print_warning

    parser = _build_bench_manifest_parser()
    args = parser.parse_args(argv)
    try:
        path = write_corpus_manifest(args.harvest_dir)
    except (FileNotFoundError, ValueError) as exc:
        print_warning(console, f"cannot build manifest: {exc}")
        return 2
    print_info(console, f"Wrote {path}")
    return 0


def _handle_bench_command(argv: list[str]) -> int:
    """Handle ``daydream bench --benchmark-repo <path> [...]``.

    ``daydream bench harvest [...]`` and ``daydream bench manifest [...]``
    dispatch to :func:`_handle_bench_harvest_command` /
    :func:`_handle_bench_manifest_command` instead.

    Parses argv into a :class:`BenchConfig` and drives
    :func:`daydream.benchmark.run_bench` synchronously. Returns an exit code
    rather than calling :func:`sys.exit`; ``main`` translates it to a process
    exit. When ``--score`` is set, ``run_bench`` verifies the judge credential
    up front and raises :class:`~daydream.benchmark.score.JudgeEnvError` if it is missing; that error
    is allowed to surface to the top-level CLI boundary as a non-zero exit.
    """
    from daydream.benchmark import run_bench

    if argv and argv[0] == "harvest":
        return _handle_bench_harvest_command(argv[1:])

    if argv and argv[0] == "manifest":
        return _handle_bench_manifest_command(argv[1:])

    _load_bench_dotenv()
    config = _bench_config_from_argv(argv)
    return run_bench(config)


def _build_benchmark_parser() -> argparse.ArgumentParser:
    """Build the ``daydream benchmark`` subcommand parser.

    Sub-verbs: ``init``, ``status``, ``validate``, ``build-harbor``, ``upgrade``, ``import-prs``,
    ``curate``, ``calibrate-judge``, ``run``, ``clean``, ``objective``.
    """
    parser = argparse.ArgumentParser(
        prog="daydream benchmark",
        description=(
            "Private PR benchmark workspace: init/status/validate/build-harbor/upgrade/"
            "import-prs/curate/calibrate-judge/run/clean/objective."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand")

    init_p = sub.add_parser("init", help="create a private benchmark workspace")
    init_p.add_argument("dir", type=Path, help="workspace directory (must be empty/absent)")
    init_p.add_argument("--repo", required=True, help="OWNER/REPO repository")
    init_p.add_argument(
        "--reviewer-host", action="append", default=[], help="reviewer egress host (repeatable)"
    )
    init_p.add_argument(
        "--judge-host", action="append", default=[], help="judge egress host (repeatable)"
    )

    status_p = sub.add_parser("status", help="show read-only derived workspace state")
    status_p.add_argument("dir", type=Path, help="workspace directory")

    validate_p = sub.add_parser("validate", help="validate the workspace (0/2/1 exit codes)")
    validate_p.add_argument("dir", type=Path, help="workspace directory")
    validate_p.add_argument("--compiled", action="store_true", help="validate emitted tasks with Harbor 0.21")

    build_p = sub.add_parser("build-harbor", help="package a validated workspace for Harbor 0.21")
    build_p.add_argument("dir", type=Path, help="workspace directory")
    build_p.add_argument("--daydream-wheel", required=True, type=Path, help="wheel for this Daydream version")

    upgrade_p = sub.add_parser(
        "upgrade", help="deterministically upgrade legacy case documents (finding_id + schema_version)"
    )
    upgrade_p.add_argument("dir", type=Path, help="workspace directory")
    upgrade_p.add_argument("--dry-run", action="store_true", help="report the upgrade without writing")

    import_prs_p = sub.add_parser(
        "import-prs", help="import explicit private GitHub PR evidence into the workspace"
    )
    import_prs_p.add_argument("dir", type=Path, help="workspace directory")
    import_prs_p.add_argument(
        "--pr",
        action="append",
        default=[],
        metavar="N|URL",
        help="PR number or https://github.com/OWNER/REPO/pull/N (repeatable)",
    )
    import_prs_p.add_argument(
        "--pr-file", action="append", default=[], type=Path, metavar="FILE",
        help="file listing PR numbers/URLs, one per line (repeatable)",
    )
    import_prs_p.add_argument(
        "--head", action="append", default=[], metavar="PR=<40-hex>",
        help=(
            "explicit head SHA of PR N (PR=<40-hex>, repeatable); a bare 40-hex "
            "is accepted for back-compat and treated as the sole requested PR"
        ),
    )
    import_prs_p.add_argument(
        "--refresh", action="store_true",
        help="re-fetch already-imported PRs",
    )

    curate_p = sub.add_parser("curate", help="curate a case's golden review")
    curate_p.add_argument("dir", type=Path, help="workspace directory")
    curate_p.add_argument(
        "--case", metavar="CASE-ID", help="case id to curate"
    )
    curate_p.add_argument(
        "--apply-gold",
        type=Path,
        default=None,
        metavar="FILE",
        help="apply a reviewed gold YAML draft (derive all forbidden fields, never ready)",
    )

    calibrate_p = sub.add_parser("calibrate-judge", help="calibrate the configured semantic-match judge")
    calibrate_p.add_argument("dir", type=Path, help="workspace directory")
    calibrate_p.add_argument(
        "--yes", action="store_true", help="confirm the paid 72-call calibration run"
    )

    run_p = sub.add_parser(
        "run", help="supervise a Harbor run behind the Oracle self-match gate"
    )
    run_p.add_argument("dir", type=Path, help="workspace directory")
    run_p.add_argument(
        "--oracle", action="store_true", help="run the Oracle self-match pass"
    )
    run_p.add_argument(
        "--yes", action="store_true", help="confirm the paid run without prompting"
    )

    clean_p = sub.add_parser(
        "clean", help="remove ledger-derived disposable artifacts (issue #782)"
    )
    clean_p.add_argument("dir", type=Path, help="workspace directory")
    clean_p.add_argument(
        "--cache", action="store_true",
        help="remove the disposable clone + build stage under cache/"
    )
    clean_p.add_argument(
        "--jobs", action="store_true",
        help="remove ledgered Harbor job dirs + their recorded Docker images"
    )
    clean_p.add_argument(
        "--trajectories", action="store_true",
        help="remove contained agent/trajectory.json files in ledgered job dirs"
    )
    clean_p.add_argument(
        "--derived", action="store_true",
        help="union of --cache --jobs --trajectories (preserves curated source/gold)"
    )
    clean_p.add_argument(
        "--all", action="store_true",
        help="delete every deletable artifact including curated source/gold (needs --yes)"
    )
    clean_p.add_argument(
        "--yes", action="store_true", help="confirm --all without prompting"
    )

    objective_p = sub.add_parser(
        "objective", help="resolve an exact completed run as machine-readable JSON"
    )
    objective_p.add_argument("dir", type=Path, help="workspace directory")
    objective_p.add_argument("--run-id", required=True, help="exact ledgered run id")
    objective_p.add_argument(
        "--json",
        default=None,
        metavar="PATH|-",
        help="write the strict objective JSON to this path ('-' writes to stdout)",
    )

    return parser


def _handle_benchmark_import_prs(args) -> int:
    """Import explicit private PRs: parse targets, preflight, then run the import.

    Expected errors (mis-tokenized targets, preflight failure) print a message
    to stderr and return exit ``1`` — never a bare traceback.
    """
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.workspace import WorkspaceCorrupt

    try:
        targets = gi.parse_import_targets(args.pr, args.pr_file, args.head)
    except gi.ImportTargetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        return gi.run_import_prs(
            args.dir,
            targets.pr_numbers,
            pr_heads=targets.pr_heads,
            refresh=args.refresh,
        )
    except gi.PreflightError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except WorkspaceCorrupt as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _handle_benchmark_init(dir_path: Path, repo: str, reviewer_hosts: list[str], judge_hosts: list[str]) -> int:
    """Run ``init_workspace`` and report classification + egress boundary."""
    from daydream.benchmark.workspace import InitError, init_workspace

    try:
        manifest = init_workspace(dir_path, repo, reviewer_hosts, judge_hosts)
    except InitError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    classification = manifest.privacy.classification
    egress = " ".join(manifest.privacy.reviewer_allowed_hosts + manifest.privacy.judge_allowed_hosts)
    print(f"classification: {classification}")
    print(f"egress boundary: {egress}")
    return 0


def _handle_benchmark_status(dir_path: Path) -> int:
    """Print the read-only derived workspace state + unresolved identity."""
    from daydream.benchmark.workspace import WorkspaceCorrupt, workspace_status

    try:
        status = workspace_status(dir_path)
    except WorkspaceCorrupt as exc:
        print(str(exc), file=sys.stderr)
        return 1
    unresolved = "unresolved" if not status.repository_identity_resolved else "resolved"
    print(f"workspace state: {status.workspace_state}")
    print(f"repository identity: {unresolved}")
    if status.last_preflight_verified_at:
        print(f"repository identity/access verification: ran ({status.last_preflight_verified_at})")
    else:
        print("repository identity/access verification: not yet run")
    print(f"ledger entries: {len(status.ledger.pull_requests)}")
    for summary in status.case_snapshots:
        head = summary.get("head_prefix") or "-"
        print(
            f"  case {summary.get('case_id', '')}: "
            f"snapshot {summary.get('snapshot_status', 'imported')} @ {head}"
        )
    return 0


def _handle_benchmark_validate(args) -> int:
    """Print the human-readable classification and return the numeric code."""
    if args.compiled:
        from daydream.benchmark.harbor.build import CompileError
        from daydream.benchmark.harbor.package import validate_compiled
        from daydream.benchmark.workspace import WorkspaceCorrupt

        try:
            code = validate_compiled(args.dir)
        except (CompileError, WorkspaceCorrupt) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("validation: compiled-ready")
        return code
    from daydream.benchmark.workspace import validate_workspace

    code, label = validate_workspace(args.dir)
    print(f"validation: {label}")
    return code


def _handle_benchmark_build_harbor(args) -> int:
    """Package a validated authoring workspace as a runnable Harbor dataset."""
    from daydream.benchmark.harbor.build import CompileError
    from daydream.benchmark.harbor.package import build_harbor
    from daydream.benchmark.workspace import WorkspaceCorrupt

    try:
        lock = build_harbor(args.dir, wheel=args.daydream_wheel)
    except (CompileError, WorkspaceCorrupt) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"built Harbor dataset with {len(lock.get('cases', {}))} case(s)")
    return 0


def _handle_benchmark_upgrade(args) -> int:
    """Deterministically upgrade legacy v1 case documents to v2 in place.

    Prints the per-case report plus any surfaced errors. Returns ``0`` on a
    successful upgrade (including an idempotent no-op second run) and ``1``
    when a case errored.
    """
    from daydream.benchmark import migrate

    report = migrate.migrate_workspace(args.dir, dry_run=args.dry_run)
    for c in report.cases:
        print(
            f"case {c.case_id}: finding_ids_recomputed={c.finding_ids_recomputed} "
            f"changed={c.changed}"
        )
    for e in report.errors:
        print(f"error: {e}", file=sys.stderr)
    if report.errors:
        return 1
    return 0


def _is_interactive_tty() -> bool:
    """True only when both sys.stdin and stdout are real interactive terminals."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _handle_benchmark_calibrate(args) -> int:
    """Calibrate the configured judge against the fixed 24-pair fixture.

    Requires ``--yes`` or an interactive TTY before any paid judge call. Lazy
    imports the calibrate module (mirroring the other handlers); build failures
    and refusals print to stderr and return exit ``1`` — never a bare traceback.
    """
    if not args.yes and not _is_interactive_tty():
        print(
            "calibrate-judge: requires TTY confirmation or --yes before any paid judge call",
            file=sys.stderr,
        )
        return 1
    from daydream.benchmark.harbor import calibrate

    env = {
        name: os.environ.get(name)
        for name in (
            "DAYDREAM_JUDGE_PROVIDER",
            "DAYDREAM_JUDGE_MODEL",
            "DAYDREAM_JUDGE_API_KEY",
            "DAYDREAM_JUDGE_BASE_URL",
            "DAYDREAM_JUDGE_ALLOWED_HOSTS",
        )
    }

    # Issue #885/R12: thread the control-plane candidate profile digest so a
    # candidate-scoped calibration can be produced (run.py's oracle preflight
    # compares the receipt against inputs that fold the digest). Fail-closed on
    # an invalid candidate, matching run.py's handling. None for default runs.
    env["DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST"] = _candidate_profile_digest()

    return calibrate.run_calibration(
        args.dir,
        yes=args.yes,
        env=env,
        http=None,
    )


def _handle_benchmark_run(args) -> int:
    """Supervise a Harbor run in the workspace behind the Oracle gate.

    Threads the reviewer/judge env vars into the supervisor (mirroring
    :func:`_handle_benchmark_calibrate`); expected supervisor errors already
    print to stderr inside ``run_run`` and return a nonzero code — this
    handler does not wrap them in a traceback.
    """
    from daydream.benchmark.harbor import run as run_mod

    env = {
        name: os.environ.get(name)
        for name in (
            "DAYDREAM_REVIEW_BACKEND",
            "DAYDREAM_REVIEW_MODEL",
            "DAYDREAM_REVIEW_API_KEY",
            "DAYDREAM_REVIEW_BASE_URL",
            "DAYDREAM_JUDGE_PROVIDER",
            "DAYDREAM_JUDGE_MODEL",
            "DAYDREAM_JUDGE_API_KEY",
            "DAYDREAM_JUDGE_BASE_URL",
        )
    }

    # Issue #885/R12: thread the control-plane candidate profile digest into
    # the supervisor env so run.py's ledger/receipt provenance can attribute
    # the run to exactly the tested candidate. run.py reads
    # DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST from this env dict; the in-container
    # entrypoint cannot set it (different process, runs after the ledger row).
    # Fail-closed: an invalid candidate raises ProfileError here, before any
    # paid run -- matching the entrypoint's own fail-closed validation.
    env["DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST"] = _candidate_profile_digest()

    return run_mod.run_run(args.dir, oracle=args.oracle, yes=args.yes, env=env)


def _candidate_profile_digest() -> str | None:
    """Canonical digest of the control-plane Harbor candidate (R12), or None.

    Reads ``DAYDREAM_REVIEW_PROFILE_CANDIDATE`` from the trusted control-plane
    environment and resolves it exactly as the in-container entrypoint will,
    so the ledger/receipt ``profile_digest`` matches the tested profile. No
    candidate -> ``None`` (legacy default-profile runs stay byte-stable).
    Raises ``ProfileError`` on an invalid candidate (fail-closed).
    """
    from daydream import review_profile as rp

    if not os.environ.get("DAYDREAM_REVIEW_PROFILE_CANDIDATE"):
        return None
    resolved = rp.resolve_harbor_profile()  # env=None -> os.environ (trusted)
    return resolved.digest


def _handle_benchmark_clean(args) -> int:
    """Handle ``daydream benchmark clean <dir> [--cache] [--jobs] [...]``.

    Resolves the ``--derived`` union into the three selection flags *before*
    calling ``clean_workspace`` (the contract the routing test pins) and runs
    contracted deletion entirely inside ``clean_workspace``. An explicit
    ``--all`` without ``--yes`` needs a TTY (mirroring ``run --yes`` /

    ``calibrate-judge``); expected ``RunError``/``WorkspaceCorrupt`` print to
    stderr and return exit ``1`` — never a bare traceback.
    """
    from daydream.benchmark.harbor import clean as clean_mod
    from daydream.benchmark.harbor import run as run_mod
    from daydream.benchmark.storage import WorkspaceCorrupt

    if args.all and not args.yes and not _is_interactive_tty():
        print(
            "clean --all: requires TTY confirmation or --yes before deleting "
            "curated source/gold",
            file=sys.stderr,
        )
        return 1
    cache = args.cache or args.derived
    jobs = args.jobs or args.derived
    trajectories = args.trajectories or args.derived
    try:
        report = clean_mod.clean_workspace(
            args.dir,
            cache=cache,
            jobs=jobs,
            trajectories=trajectories,
            all_=args.all,
            yes=args.yes,
        )
    except (run_mod.RunError, WorkspaceCorrupt) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for line in report.summary_lines():
        print(line)
    return report.exit_code


def _handle_benchmark_curate(args) -> int:
    """Curate a case: derive everything, never attests to ready.

    On an interactive TTY, ``curate`` dispatches into the resumable terminal
    client (:func:`daydream.benchmark.curate_tui.run_curate_tui`); otherwise it
    requires ``--apply-gold <file>`` (a reviewed gold YAML draft) and routes
    it through :func:`daydream.benchmark.curation.apply_gold_fragment`. Expected
    workspace errors print to stderr and return exit ``1`` — never a bare traceback.
    """
    from pydantic import ValidationError

    from daydream import git_ops
    from daydream.benchmark import curation as cu
    from daydream.benchmark.storage import WorkspaceCorrupt, load_yaml_strict

    if args.apply_gold is None:
        if _is_interactive_tty():
            from daydream.benchmark.curate_tui import run_curate_tui
            return run_curate_tui(args.dir, args.case)
        print(
            "curate: interactive curation requires a TTY; pass --apply-gold <file> to apply "
            "a reviewed gold draft",
            file=sys.stderr,
        )
        return 1
    try:
        fragment = load_yaml_strict(args.apply_gold)
        cu.apply_gold_fragment(args.dir, args.case, fragment)
    except (cu.CurationError, WorkspaceCorrupt, git_ops.GitError, ValidationError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _handle_benchmark_objective(args) -> int:
    """Resolve an exact completed run and emit its machine-readable objective.

    ``--json`` serializes the opaque privacy-safe objective via
    ``objective.objective_to_json`` and writes it through
    ``storage.atomic_write_json`` (or prints it directly on ``-``); a parse/
    compat failure leaves an existing output file byte-identical. Without
    ``--json``, prints a concise local summary (run_id, comparison_eligible,
    micro F1, task/infra counts) to stdout. Expected ``ObjectiveError`` prints
    to stderr and returns exit ``1`` — never a bare traceback.
    """
    from daydream.benchmark.harbor import objective
    from daydream.benchmark.storage import atomic_write_json

    try:
        run = objective.read_completed_run(args.dir, args.run_id, env=dict(os.environ))
    except objective.ObjectiveError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json is not None:
        blob = objective.objective_to_json(run)
        if args.json == "-":
            print(json.dumps(blob, indent=2))
        else:
            atomic_write_json(Path(args.json), blob)

    obj = run.objective
    if obj is not None:
        print(
            f"objective {run.run_id}: comparison_eligible={obj.comparison_eligible} "
            f"micro_f1={obj.f1:.4f} tasks={obj.task_count} "
            f"scored={obj.scored_task_count} infra={obj.infra_error_task_count}"
        )
    else:
        print(f"objective {run.run_id}: no objective (unscored run)")
    return 0


def _handle_benchmark_command(argv: list[str]) -> int:
    """Handle ``daydream benchmark init|status|validate|build-harbor|upgrade|import-prs|curate``.

    Returns an exit code; ``daydream.cli.main`` translates it to a
    process exit. Expected workspace errors (``InitError``/``WorkspaceCorrupt``/
    ``ImportTargetError``/``PreflightError``/``CurationError``) are printed to
    stderr and mapped to exit ``1`` — never a bare traceback. ``run`` dispatches
    to :func:`_handle_benchmark_run` (the supervised Harbor runner) and
    ``objective`` to :func:`_handle_benchmark_objective` (the read-only
    machine-readable run resolution).
    """

    parser = _build_benchmark_parser()
    args = parser.parse_args(argv)
    sub = args.subcommand
    if sub is None:
        parser.print_help()
        return 0
    if sub == "init":
        return _handle_benchmark_init(args.dir, args.repo, args.reviewer_host, args.judge_host)
    if sub == "status":
        return _handle_benchmark_status(args.dir)
    if sub == "validate":
        return _handle_benchmark_validate(args)
    if sub == "build-harbor":
        return _handle_benchmark_build_harbor(args)
    if sub == "upgrade":
        return _handle_benchmark_upgrade(args)
    if sub == "import-prs":
        return _handle_benchmark_import_prs(args)
    if sub == "curate":
        return _handle_benchmark_curate(args)
    if sub == "calibrate-judge":
        return _handle_benchmark_calibrate(args)
    if sub == "run":
        return _handle_benchmark_run(args)
    if sub == "clean":
        return _handle_benchmark_clean(args)
    if sub == "objective":
        return _handle_benchmark_objective(args)
    parser.print_help(file=sys.stderr)
    return 2
