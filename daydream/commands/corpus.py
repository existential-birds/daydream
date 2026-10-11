"""Record-based corpus commands and their argument parsing."""

import argparse
import getpass
import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import anyio

from daydream.commands import calibrate, common, dataset
from daydream.dataset import LocalRecordStore, SnapshotRecords, StoreError, semantic_evidence_digest
from daydream.json_utils import canonical_json
from daydream.ui import create_console, print_error, print_info, print_success


def _add_record_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", type=Path, default=Path.home() / ".daydream" / "dataset",
                        help="Private JSONL record store (default: ~/.daydream/dataset).")
    parser.add_argument("--snapshot-id", required=True,
                        help="Immutable local snapshot ID selected with dataset snapshot.")


def _build_build_corpus_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daydream corpus build",
        description="Project frozen run and observation records into deterministic training examples, offline.")
    _add_record_inputs(parser)
    parser.add_argument("--out", type=Path, required=True,
                        help="Corpus JSONL path; manifests and lineage go beside it.")
    for dimension in ("stack", "repo", "profile"):
        parser.add_argument(f"--max-{dimension}-share", type=float,
                            help=f"Maximum projected share of one {dimension}, in (0, 1].")
    common._add_dry_run_argument(parser, "Print the projection summary without writing output.")
    parser.add_argument("--as-of", help="Optional UTC temporal pin; cannot broaden snapshot eligibility.")
    return parser


def _handle_build_corpus_command(argv: list[str]) -> int:
    import tempfile
    from dataclasses import replace

    from daydream.training.corpus_projection import BuildFrozenCorpusConfig, build_frozen_corpus

    args = _build_build_corpus_parser().parse_args(argv)
    console = create_console()
    for dimension in ("stack", "repo", "profile"):
        value = getattr(args, f"max_{dimension}_share")
        if value is not None and not 0.0 < value <= 1.0:
            print_error(console, f"Invalid --max-{dimension}-share", "Must be in (0, 1].")
            return 1
    try:
        config = BuildFrozenCorpusConfig(out_dir=args.out.parent,
            store_dir=args.store.expanduser(), snapshot_id=args.snapshot_id,
            as_of=args.as_of,
            max_stack_share=args.max_stack_share, max_repo_share=args.max_repo_share,
            max_profile_share=args.max_profile_share)
        if args.dry_run:
            with tempfile.TemporaryDirectory() as td:
                summary = build_frozen_corpus(replace(config, out_dir=Path(td)))
        else:
            summary = build_frozen_corpus(config)
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
    common._add_dry_run_argument(parser, "Build annotations without writing observations.")
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
        observation = {"schema_version": "daydream.observation.v3", "run_id": run["run_id"], "item_uid": None,
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


_CORPUS_SUBVERBS: dict[str, Callable[[list[str]], int]] = {
    "dataset": dataset.handle_dataset,
    "harvest": _handle_harvest_command,
    "build": _handle_build_corpus_command,
    "label": _handle_label_command,
    "calibrate-reward": calibrate._handle_calibrate_reward_command,
    "adjudicate": _handle_adjudicate_command,
}

_CORPUS_USAGE = (
    "usage: daydream corpus {dataset,harvest,build,label,calibrate-reward,adjudicate} ...\n\n"
    "Data-pipeline sub-verbs:\n"
    "  dataset   publish/status/download JSONL evidence or select a frozen snapshot\n"
    "  harvest   append PR enrichment and annotations for selected run records\n"
    "  build     project a frozen record snapshot offline\n"
    "  label     append an authoritative human outcome label\n"
    "  calibrate-reward  emit deterministic reward-calibration artifacts from derived corpus output\n"
    "  adjudicate  per-finding queue, preview, label, export, report, materialize, and harvest-snapshot"
)


def _handle_corpus_command(argv: list[str]) -> int:
    return common._dispatch_namespace(argv, _CORPUS_SUBVERBS, _CORPUS_USAGE)
