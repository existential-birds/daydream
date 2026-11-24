"""Record-based corpus commands and their argument parsing."""

import argparse
import getpass
import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import anyio

from daydream.commands import common, dataset
from daydream.dataset import LocalRecordStore, SnapshotRecords, StoreError, semantic_evidence_digest
from daydream.json_utils import canonical_json
from daydream.ui import create_console, print_error, print_info, print_success


def _add_record_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", type=Path, default=Path.home() / ".daydream" / "dataset",
                        help="Private JSONL record store (default: ~/.daydream/dataset).")
    parser.add_argument("--snapshot-id", required=True,
                        help="Immutable local snapshot ID selected with dataset snapshot.")


def _build_build_corpus_parser() -> argparse.ArgumentParser:
    from daydream.training.corpus_projection.selection import SHARE_DIMENSION_NAMES

    parser = argparse.ArgumentParser(prog="daydream corpus build",
        description="Project frozen run and observation records into deterministic training examples, offline.")
    _add_record_inputs(parser)
    parser.add_argument("--out", type=Path, required=True,
                        help="Corpus JSONL path; manifests and lineage go beside it.")
    for dimension in SHARE_DIMENSION_NAMES:
        parser.add_argument(f"--max-{dimension}-share", type=float,
                            help=f"Maximum projected share of one {dimension}, in (0, 1].")
    parser.add_argument("--dry-run", action="store_true", help="Print the projection summary without writing output.")
    parser.add_argument("--as-of", help="Optional UTC temporal pin; cannot broaden snapshot eligibility.")
    return parser


def _handle_build_corpus_command(argv: list[str]) -> int:
    import tempfile
    from dataclasses import replace

    from daydream.training.corpus_projection import BuildFrozenCorpusConfig, build_frozen_corpus
    from daydream.training.corpus_projection.projector import ShareCapRangeError
    from daydream.training.corpus_projection.selection import SHARE_DIMENSION_NAMES

    args = _build_build_corpus_parser().parse_args(argv)
    console = create_console()
    try:
        config = BuildFrozenCorpusConfig(out_dir=args.out.parent,
            store_dir=args.store.expanduser(), snapshot_id=args.snapshot_id,
            as_of=args.as_of,
            **{f"max_{name}_share": getattr(args, f"max_{name}_share") for name in SHARE_DIMENSION_NAMES})
        if args.dry_run:
            with tempfile.TemporaryDirectory() as td:
                summary = build_frozen_corpus(replace(config, out_dir=Path(td)))
        else:
            summary = build_frozen_corpus(config)
    except ShareCapRangeError as exc:
        print_error(console, f"Invalid --max-{exc.dimension}-share", "Must be in (0, 1].")
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print_error(console, "Corpus build refused", str(exc))
        return 1
    print_success(console, f"Corpus build complete: {summary['emitted']} records "
                  f"({summary['adjudication']} to adjudication) in {args.out.parent}")
    return 0


def _build_harvest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daydream corpus harvest",
        description="Acquire PR evidence for frozen records and append immutable annotations and enrichment.")
    _add_record_inputs(parser)
    parser.add_argument("--dry-run", action="store_true", help="Build annotations without writing observations.")
    parser.add_argument("--session", help="Restrict runs to this run ID prefix.")
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.daydream/harvest-cache/"))
    parser.add_argument("--repo-clone-root", type=Path, help="Cached clone directory (default: <cache-dir>/repos/).")
    parser.add_argument("--gh-spacing-sec", type=float, default=0.8, help="GitHub request spacing in seconds.")
    return parser


def _handle_harvest_command(argv: list[str]) -> int:
    from daydream.training import harvest

    args = _build_harvest_parser().parse_args(argv)
    console = create_console()
    if args.gh_spacing_sec < 0.0:
        print_error(console, "Invalid --gh-spacing-sec", "Must be >= 0.0.")
        return 1
    try:
        config = harvest.HarvestConfig(store_dir=args.store.expanduser(), snapshot_id=args.snapshot_id,
            dry_run=args.dry_run, cache_dir=args.cache_dir.expanduser(),
            repo_clone_root=args.repo_clone_root.expanduser() if args.repo_clone_root else None,
            session_filter=args.session, gh_request_spacing_sec=args.gh_spacing_sec)
        services = harvest.make_harvest_services(config)
        summary = anyio.run(partial(harvest.run_harvest, services=services), config)
    except (OSError, ValueError, TypeError) as exc:
        print_error(console, "Corpus harvest refused", str(exc))
        return 1
    print_info(console, str(summary))
    return 1 if summary.get("aborted", 0) >= 1 or summary.get("errors", 0) > 0 else 0


def _build_label_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daydream corpus label",
        description="Append an authoritative human outcome observation to a captured run.")
    parser.add_argument("session", help="Full or unique run ID prefix.")
    parser.add_argument("--outcome", required=True, choices=["accepted", "contested", "rejected", "unknown"])
    parser.add_argument("--store", type=Path, default=Path.home() / ".daydream" / "dataset")
    parser.add_argument("--author", default=getpass.getuser(), help="Human labeler identity.")
    return parser


def _handle_label_command(argv: list[str]) -> int:
    from daydream.training.record_evidence import latest_annotation

    args = _build_label_parser().parse_args(argv)
    console = create_console()
    try:
        store = LocalRecordStore(args.store.expanduser())
        records = store.read_records()
        matches = [run for run in records["runs"] if run["run_id"].startswith(args.session)]
        if len(matches) != 1:
            raise StoreError("ambiguous_run_prefix" if matches else "unknown_run_reference")
        run = matches[0]
        prior = latest_annotation(SnapshotRecords({}, records["runs"], records["observations"],
                                                   records["observations"]), run["run_id"])
        if prior is not None:
            print_info(console, f"Current label for {run['run_id']}: {prior['labels']}")
        else:
            print_info(console, f"No prior label for {run['run_id']}.")
        evidence = {"outcome": args.outcome}
        now = datetime.now(timezone.utc).isoformat()
        observation = {"schema_version": "daydream.observation.v4", "run_id": run["run_id"], "item_uid": None,
            "valid_at": now, "observed_at": now, "source": "manual", "author": args.author, "role": "rater",
            "policy_version": "human-outcome-v1", "rubric_version": "human-outcome-v1",
            "semantic_evidence": evidence, "evidence_digest": semantic_evidence_digest(evidence),
            "payload": {"type": "run-label", "label": args.outcome}}
        observation["observation_id"] = hashlib.sha256(canonical_json(observation).encode()).hexdigest()
        store.append_observation(observation)
    except (OSError, ValueError, TypeError) as exc:
        print_error(console, "Corpus label refused", str(exc))
        return 1
    print_info(console, f"Set human label for {run['run_id']}: {args.outcome}")
    return 0


def _handle_adjudicate_command(argv: list[str]) -> int:
    from daydream.training.adjudication.cli import handle_adjudicate

    return handle_adjudicate(argv)


def _build_calibrate_system1_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daydream corpus calibrate-system1",
        description=(
            "Replay archived review runs or a frozen record snapshot through Jev / System 1 "
            "in shadow mode to calibrate gate thresholds."
        ),
    )
    parser.add_argument("--run-dir", type=Path, action="append", default=[],
                        help="Archived run directory or archive root (repeatable).")
    parser.add_argument("--store", type=Path, help="Private JSONL record store (requires --snapshot-id).")
    parser.add_argument("--snapshot-id", help="Frozen snapshot ID inside --store.")
    parser.add_argument("--model", help="System 1 Decisions model override (default: DAYDREAM_SYSTEM1_MODEL or Jev).")
    parser.add_argument("--out", type=Path, help="Write JSON calibration report to this path instead of stdout.")
    parser.add_argument("--target-recall-floor", type=float, default=0.99)
    parser.add_argument("--claim-recall-floor", type=float, default=0.98)
    parser.add_argument("--cross-recall-floor", type=float, default=0.99)
    return parser


def _handle_calibrate_system1_command(argv: list[str]) -> int:
    import json
    from dataclasses import replace

    from daydream.eval.system1_calibration import (
        ShadowDataset,
        load_shadow_cases_from_runs,
        load_shadow_cases_from_store,
        run_shadow_calibration,
    )
    from daydream.json_utils import atomic_write_json
    from daydream.system1 import System1Client, System1Config

    args = _build_calibrate_system1_parser().parse_args(argv)
    console = create_console()
    if not args.run_dir and not args.store:
        print_error(console, "Calibration input required", "Pass at least one --run-dir or --store with --snapshot-id.")
        return 2
    if bool(args.store) != bool(args.snapshot_id):
        print_error(console, "Invalid snapshot selection", "--store and --snapshot-id must be provided together.")
        return 2
    for flag, value in (
        ("--target-recall-floor", args.target_recall_floor),
        ("--claim-recall-floor", args.claim_recall_floor),
        ("--cross-recall-floor", args.cross_recall_floor),
    ):
        if not 0.0 < value <= 1.0:
            print_error(console, f"Invalid {flag}", "Recall floor must be in (0, 1].")
            return 2

    try:
        config = replace(System1Config.from_environment(), enabled=True)
        if args.model is not None:
            config = replace(config, model=args.model)
        if not config.api_key:
            print_error(console, "Missing OpenRouter API key", "Set OPENROUTER_API_KEY to run shadow calibration.")
            return 1
        datasets: list[ShadowDataset] = []
        if args.run_dir:
            datasets.append(load_shadow_cases_from_runs(args.run_dir))
        if args.store and args.snapshot_id:
            datasets.append(load_shadow_cases_from_store(args.store.expanduser(), args.snapshot_id))
        report = anyio.run(
            partial(
                run_shadow_calibration,
                target_recall_floor=args.target_recall_floor,
                claim_recall_floor=args.claim_recall_floor,
                cross_recall_floor=args.cross_recall_floor,
            ),
            ShadowDataset.merge(datasets),
            System1Client(config),
        )
        payload = report.to_dict()
        if args.out is not None:
            atomic_write_json(args.out.expanduser(), payload, trailing_newline=True)
            print_success(console, f"Wrote System 1 calibration report to {args.out}")
        else:
            print(json.dumps(payload, indent=2))
    except (OSError, ValueError, TypeError) as exc:
        print_error(console, "System 1 calibration failed", str(exc))
        return 1
    return 0


_CORPUS_SUBVERBS: dict[str, Callable[[list[str]], int]] = {
    "dataset": dataset.handle_dataset,
    "harvest": _handle_harvest_command,
    "build": _handle_build_corpus_command,
    "label": _handle_label_command,
    "adjudicate": _handle_adjudicate_command,
    "calibrate-system1": _handle_calibrate_system1_command,
}

_CORPUS_USAGE = (
    "usage: daydream corpus {dataset,harvest,build,label,adjudicate,calibrate-system1} ...\n\n"
    "Data-pipeline sub-verbs:\n"
    "  dataset            publish/status/download JSONL evidence or select a frozen snapshot\n"
    "  harvest            append PR enrichment and annotations for selected run records\n"
    "  build              project a frozen record snapshot offline\n"
    "  label              append an authoritative human outcome label\n"
    "  adjudicate         per-finding queue, preview, label, export, report, materialize, and harvest-snapshot\n"
    "  calibrate-system1  replay archived runs through Jev / System 1 in shadow mode to calibrate thresholds"
)


def _handle_corpus_command(argv: list[str]) -> int:
    return common._dispatch_namespace(argv, _CORPUS_SUBVERBS, _CORPUS_USAGE)
