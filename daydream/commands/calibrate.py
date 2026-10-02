"""Calibrate commands and their argument parsing."""

import argparse
from pathlib import Path

from daydream.ui import create_console, print_error, print_info


def _build_calibrate_reward_parser() -> argparse.ArgumentParser:
    """Parse pinned-bundle calibration and candidate controls without changing reward defaults."""
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
        metavar="PATH",
        help="Calibration bundle directory holding corpus.jsonl + lineage.json + SHA256SUMS.",
    )
    parser.add_argument(
        "--gold-labels",
        type=Path,
        required=True,
        metavar="PATH",
        help='Gold labels JSON keyed by record_id: {"<record_id>": {"accepted": bool}}.',
    )
    parser.add_argument(
        "--breakdowns",
        type=Path,
        required=True,
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
        required=True,
        metavar="ID",
        help="Unique run identifier recorded in the artifact (collision-guarded).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        metavar="N",
        help="Resampling seed; the artifact is byte-reproducible given this seed.",
    )
    parser.add_argument(
        "--candidate",
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
        metavar="PATH",
        help=(
            'Optional Stage-0 score JSON keyed by record_id: '
            '{"<record_id>": {"score": float, "model_digest": str}}.'
        ),
    )
    parser.add_argument(
        "--model-digest",
        metavar="DIGEST",
        help="Digest of the Stage-0 model, required when --stage0-scores is given.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=9,
        metavar="N",
        help="Grid resolution per candidate axis (default: 9).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        metavar="N",
        help="Bootstrap resample count for AUC CIs (default: 1000).",
    )
    return parser


def _parse_candidates(raw: list[str]) -> dict[str, list[float]]:
    """Parse AXIS=V1,V2,...; reject missing separators, non-float points, and repeated axes."""
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


def _handle_calibrate_reward_command(argv: list[str]) -> int:
    """Validate before writing; errors go to stderr with exit 1. Success prints artifact path and
    metrics with exit 0.
    """
    from daydream.training import calibration as _calibration

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

    config = _calibration.CalibrationConfig(**(vars(args) | {"candidates": candidates}))
    try:
        summary = _calibration.run_calibration(config)
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
