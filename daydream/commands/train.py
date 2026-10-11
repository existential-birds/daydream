"""Train commands and their argument parsing."""

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from daydream.commands import common
from daydream.ui import create_console, print_error, print_success


class _TrainParser(argparse.ArgumentParser):
    """Report usage errors as validation refusal (exit 1), matching training harness expectations."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _build_train_parser() -> argparse.ArgumentParser:
    """Build the train parser independently of review target parsing."""
    parser = _TrainParser(
        prog="daydream train",
        description=(
            "Run the four-stage training pipeline (stage0 gate → stage1 SFT → "
            "stage2 RFT → stage3 adapter) and write a stage manifest."
        ),
    )
    parser.add_argument(
        "--projection",
        type=Path,
        required=True,
        metavar="DIR",
        help="Frozen projection directory; verifies _SUCCESS/lineage/"
             "split digests and re-applies C5 benchmark exclusions",
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
        help="Repeatable; run only these stages in the order given (default: all four)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace base model id the LoRA adapter trains against",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed (split freeze + training determinism; default: 0)",
    )
    common._add_dry_run_argument(
        parser,
        "Execute everything that needs no GPU (corpus load, stage0 gate, "
        "manifest) and mark the GPU stages skipped_dry — the CI path",
    )
    return parser


def _handle_train_command(argv: list[str]) -> int:
    """Run the coordinator synchronously; gate/validation failures return 1. Dry runs use files without
    an agent backend, SQLite, or GPU.
    """
    from daydream.training.coordinator import PipelineConfig, run_pipeline

    parser = _build_train_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(
        projection=args.projection,
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
