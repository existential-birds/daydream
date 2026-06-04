"""CLI helpers for the ``daydream bench`` subcommand.

These helpers are called from :func:`daydream.cli.main` when ``bench`` is the
first argv token. They live here rather than in the top-level ``daydream.cli``
module to keep that file below the 1 000-line threshold and to co-locate the
bench argument-parsing logic with the rest of the benchmark package.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daydream.benchmark import BenchConfig


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
        required=True,
        dest="benchmark_repo",
        metavar="PATH",
        help="Path to the external code-review-benchmark checkout",
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
        default="anthropic/claude-opus-4.5",
        dest="model",
        help="Judge model id (also names the per-model results dir)",
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
        "--score",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="score",
        help="Drive the step2/2.5/3 scoring pipeline (default: on; use --no-score to skip)",
    )
    return parser


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

    args = _build_bench_parser().parse_args(argv)
    bench_root = args.benchmark_repo / ".daydream-bench"
    cache_dir = args.cache_dir if args.cache_dir is not None else bench_root / "cache"
    trajectory_dir = args.trajectory_dir if args.trajectory_dir is not None else bench_root / "trajectories"
    return BenchConfig(
        benchmark_repo=args.benchmark_repo,
        cache_dir=cache_dir,
        force=args.force,
        score=args.score,
        only=args.only,
        limit=args.limit,
        trajectory_dir=trajectory_dir,
        model=args.model,
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

    config = _bench_config_from_argv(argv)
    return run_bench(config)
