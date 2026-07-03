"""CLI helpers for the ``daydream bench`` subcommand.

These helpers are called from :func:`daydream.cli.main` when ``bench`` is the
first argv token. They live here rather than in the top-level ``daydream.cli``
module to keep that file below the 1 000-line threshold and to co-locate the
bench argument-parsing logic with the rest of the benchmark package.
"""

from __future__ import annotations

import argparse
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

    Args:
        seconds: Elapsed wall-clock seconds.

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
        choices=["martian", "anthropic-direct"],
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
        choices=["claude", "codex", "pi"],
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
        name: The preset name passed via ``--reviewer``.
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

    Optional path flags fall back to a ``.daydream-bench`` cache derived from
    ``--benchmark-repo``. No directories are created at parse time.

    Args:
        argv: The argument vector after the ``bench`` verb.

    Returns:
        An immutable :class:`BenchConfig` for the requested run.
    """
    from daydream.benchmark import BenchConfig
    from daydream.config_file import load_file_config

    parser = _build_bench_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
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
    if judge_route not in {"martian", "anthropic-direct"}:
        parser.error("--judge-route must be one of: martian, anthropic-direct")
    if benchmark_repo is None:
        parser.error("--benchmark-repo is required (pass the flag or set [tool.daydream.bench] benchmark-repo)")
    bench_root = benchmark_repo / ".daydream-bench"
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
        min_confidence=args.min_confidence if args.min_confidence is not None else bench.get("min-confidence"),
        min_severity=args.min_severity if args.min_severity is not None else bench.get("min-severity"),
    )


def _handle_bench_command(argv: list[str]) -> int:
    """Handle ``daydream bench --benchmark-repo <path> [...]``.

    Parses argv into a :class:`BenchConfig` and drives
    :func:`daydream.benchmark.run_bench` synchronously. Returns an exit code
    rather than calling :func:`sys.exit`; ``main`` translates it to a process
    exit. When ``--score`` is set, ``run_bench`` verifies the judge credential
    up front and raises :class:`~daydream.benchmark.score.JudgeEnvError` if it is missing; that error
    is allowed to surface to the top-level CLI boundary as a non-zero exit.

    Args:
        argv: The argument vector after the ``bench`` verb.

    Returns:
        The exit code from :func:`run_bench`.
    """
    from daydream.benchmark import run_bench

    _load_bench_dotenv()
    config = _bench_config_from_argv(argv)
    return run_bench(config)
