"""Corpus commands and their argument parsing."""

import argparse
from collections.abc import Callable
from functools import partial
from pathlib import Path

import anyio

from daydream.commands import calibrate, common, dataset, hydrate
from daydream.ui import create_console, print_error, print_info, print_success


def _build_build_corpus_parser() -> argparse.ArgumentParser:
    """Parse deterministic per-finding projection over a curated bundle."""
    parser = argparse.ArgumentParser(
        prog="daydream corpus build",
        description="Project curated-bundle per-finding resolutions into deterministic, "
        "frozen-split projection training records.",
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
        dest="annotation_bundle_dir",
        metavar="DIR",
        help="Annotation-bundle root (must contain _SUCCESS, SHA256SUMS, "
        "lineage.json, annotations.jsonl); self-verified and linked to "
        "--bundle-root before the projection runs",
    )
    common._add_license_arguments(
        parser,
        policy_help="Digest-pinned license policy JSON; every record's per-repo license "
        "decision is resolved from it (required)",
        copyleft_help="Repeatable; permit a specific copyleft (GPL/AGPL) repo by exact "
        "owner/repo slug (case-insensitive)",
    )
    parser.add_argument(
        "--annotations-snapshot",
        type=Path,
        metavar="PATH",
        help=argparse.SUPPRESS,  # deprecated: refused in the handler
    )
    parser.add_argument(
        "--repo-slug",
        metavar="SLUG",
        help=argparse.SUPPRESS,  # URL-identity smuggling: refused in the handler
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output path; corpus.jsonl, the split manifests, and lineage.json "
        "are written beside it (its parent directory)",
    )

    parser.add_argument(
        "--max-stack-share",
        type=float,
        help="Maximum projected share of any single detected stack, in (0, 1]",
    )

    parser.add_argument(
        "--max-repo-share",
        type=float,
        help="Maximum projected share of any single repository slug, in (0, 1]",
    )

    parser.add_argument(
        "--max-profile-share",
        type=float,
        help="Maximum projected share of any single native profile, in (0, 1]",
    )

    common._add_dry_run_argument(parser, "Print the projection summary, write nothing")

    parser.add_argument(
        "--as-of",
        metavar="ISO_TS",
        help="ISO-8601 transaction-time pin; evidence dated after this instant "
        "refuses the build (default: latest)",
    )

    return parser


def _handle_build_corpus_command(argv: list[str]) -> int:
    """Run frozen projection synchronously without agents or network; refused builds return nonzero
    with the gate error.
    """
    import tempfile
    from dataclasses import replace

    from daydream.training.corpus_projection import BuildFrozenCorpusConfig, build_frozen_corpus

    parser = _build_build_corpus_parser()
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
            "A corpus build requires a pinned annotation bundle "
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
            "A corpus build requires a pinned license policy file; per-repo "
            "license decisions are resolved from it (fail-closed).",
        )
        return 1

    # Validate policy before creating output; malformed or unknown-version policies must refuse.
    from daydream.training.corpus_projection.license import load_license_policy

    try:
        load_license_policy(args.license_policy)
    except (OSError, ValueError, TypeError) as exc:
        print_error(create_console(), "Invalid --license-policy", str(exc))
        return 1

    # --out selects corpus JSONL inside the canonical file set. Publish _SUCCESS
    # last so corpus, manifests, lineage, and its twin share the completeness gate.
    out_dir = args.out.parent
    try:
        # BuildFrozenCorpusConfig validates UTC-only --as-of and canonicalizes +00:00; surface
        # refusal here.
        config = BuildFrozenCorpusConfig(
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
                summary = build_frozen_corpus(replace(config, out_dir=Path(td)))
        else:
            summary = build_frozen_corpus(config)
    except (OSError, ValueError, TypeError) as exc:
        print_error(create_console(), "Corpus build refused", str(exc))
        return 1
    print_success(
        create_console(),
        f"Corpus build complete: {summary['emitted']} records "
        f"({summary['adjudication']} to adjudication) in {out_dir}",
    )
    return 0


def _build_harvest_parser() -> argparse.ArgumentParser:
    """Parse a fresh bitemporal annotation pass."""
    parser = argparse.ArgumentParser(
        prog="daydream corpus harvest",
        description=(
            "Walk the archive and append one bitemporal annotation "
            "(outcome label + intrinsic reward) for every indexed run "
            "(RL/fine-tuning corpus prep)."
        ),
    )
    common._add_dry_run_argument(parser, "Build annotations but do not write observations or the resume log.")
    parser.add_argument(
        "--session",
        metavar="PREFIX",
        help="Restrict the queue to session_ids starting with PREFIX.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.daydream/harvest-cache/"),
        metavar="PATH",
        help="Directory backing the gh-api backfill cache (default: ~/.daydream/harvest-cache/).",
    )
    common._add_archive_dir_argument(parser)
    parser.add_argument(
        "--repo-clone-root",
        type=Path,
        metavar="PATH",
        help="Directory for cached repo clones (default: <cache-dir>/repos/).",
    )
    parser.add_argument(
        "--gh-spacing-sec",
        type=float,
        default=0.8,
        metavar="SEC",
        help="Sleep between rows to spread gh api calls (default: 0.8).",
    )
    return parser


def _handle_harvest_command(argv: list[str]) -> int:
    """Run harvest through anyio and return its process result.

    Validation, aborted rows, and per-row errors fail. A clean partial run with
    unresolved findings succeeds; unresolved data is not a process failure.
    """
    import daydream.archive as _archive
    import daydream.training.harvest as _harvest

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
    services = _harvest.make_harvest_services(config)
    summary = anyio.run(partial(_harvest.run_harvest, services=services), config)
    print_info(console, str(summary))
    if summary.get("aborted", 0) >= 1 or summary.get("errors", 0) > 0:
        return 1
    return 0


def _build_label_parser() -> argparse.ArgumentParser:
    """Parse an authoritative human outcome; unknown is explicitly undecided. Human observations append
    independently of automated labels.
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
        metavar="SESSION_PREFIX",
        help="Full or prefix session_id to label (must match exactly one run).",
    )
    parser.add_argument(
        "--outcome",
        required=True,
        choices=["accepted", "contested", "rejected", "unknown"],
        help="Human outcome label to record.",
    )
    common._add_archive_dir_argument(parser)
    return parser


def _handle_label_command(argv: list[str]) -> int:
    """Show the prior label, append the human observation, and refresh projections; missing/ambiguous
    prefixes return 1.
    """
    import daydream.archive as _archive
    from daydream.archive import index as _index

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


def _handle_adjudicate_command(argv: list[str]) -> int:
    """Delegate adjudication verbs and their exit codes; argparse rejects bare/unknown verbs with 2."""
    from daydream.training.adjudication.cli import handle_adjudicate

    return handle_adjudicate(argv)


_CORPUS_SUBVERBS: dict[str, Callable[[list[str]], int]] = {
    "dataset": dataset.handle_dataset,
    "harvest": _handle_harvest_command,
    "build": _handle_build_corpus_command,
    "label": _handle_label_command,
    "hydrate-hub": hydrate._handle_hydrate_hub_command,
    "calibrate-reward": calibrate._handle_calibrate_reward_command,
    "adjudicate": _handle_adjudicate_command,
}


_CORPUS_USAGE = (
    "usage: daydream corpus {dataset,harvest,build,label,hydrate-hub,calibrate-reward,adjudicate} ...\n"
    "\n"
    "Data-pipeline sub-verbs:\n"
    "  dataset   publish/status/download validated JSONL records from a private Hub dataset\n"
    "  harvest   walk the archive and append one bitemporal annotation per indexed run\n"
    "  build  project curated-bundle resolutions into projection records (pinned --license-policy required)\n"
    "  label     record an authoritative human outcome label that overrides automated ones\n"
    "  hydrate-hub  hydrate a pinned Hub snapshot into a sanitized, verified staging archive\n"
    "  calibrate-reward  validate a calibration bundle and emit a deterministic reward-calibration artifact\n"
    "  adjudicate  per-finding human-label workflow: build/show/label/export/report, then"
    "\n"
    "  checkpoint/resume/materialize/harvest/publish/download the annotation snapshot"
)


def _handle_corpus_command(argv: list[str]) -> int:
    """Dispatch corpus sub-verbs through the shared namespace usage/exit contract."""
    return common._dispatch_namespace(argv, _CORPUS_SUBVERBS, _CORPUS_USAGE)
