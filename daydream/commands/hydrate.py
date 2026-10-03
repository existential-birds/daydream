"""Hydrate commands and their argument parsing."""

import argparse
from pathlib import Path
from typing import Any

from daydream.commands import common
from daydream.trajectory import redact_text
from daydream.ui import create_console, print_error, print_info, print_success, print_warning


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
        required=True,
        metavar="REPO_ID",
        help="Source private Hub dataset repo (e.g. org/ds) holding the snapshot.",
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        metavar="REV",
        help=(
            "Pinned source commit: an exact 40-char SHA or unique hex prefix. "
            "Moving branches/tags require --exploratory."
        ),
    )
    parser.add_argument(
        "--destination-repo",
        required=True,
        metavar="REPO_ID",
        help="Destination private Hub repo for the sanitized output (must be private).",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="Local staging directory for downloads, ingest, and the rebuildable index.",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Opt in to a moving branch/tag source revision (output is non-canonical).",
    )
    common._add_license_arguments(
        parser,
        policy_help="Digest-pinned license policy JSON; REQUIRED for publication "
        "(omitting it on a non-dry run refuses before any Hub access); the "
        "per-repo license admission gate runs at hydration and rejected "
        "sessions are excluded before publication; optional for --dry-run "
        "planning (issue #1094, previously #1080)",
        copyleft_help="Repeatable; permit a specific copyleft (GPL/AGPL) repo by exact "
        "owner/repo slug (case-insensitive); only meaningful with --license-policy",
    )
    common._add_dry_run_argument(
        parser,
        "Plan only: discover and normalize sessions, download, ingest, and tally "
        "discovered/admitted/rejected candidates — no Hub publication.",
    )
    return parser


def _handle_hydrate_hub_command(argv: list[str]) -> int:
    """Validate credentials and revision policy before Hub access, failing with exit 1.

    Success (including dry run) returns 0; verified runs print the immutable commit
    and value-free counts/reason tallies. Hydration errors are redacted for display.
    """
    import os

    from daydream.archive import hydrate as _hydrate

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
    # it (planning affordance). Mirrors build's policy-required pattern.
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
    # build handler's fail-closed validation.
    if args.license_policy is not None:
        from daydream.training.corpus_projection.license import load_license_policy

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
        summary = _hydrate.run_hydrate_hub(config)
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
        _print_license_admission(console, summary.license_admission)
    if summary.dry_run_incomplete_manifests:
        _print_incomplete_manifests(
            console, summary.dry_run_incomplete_manifests, prefix="hydration"
        )
    return 0


def _print_license_admission(console: Any, buckets: Any) -> None:
    """Print the value-free license-admission tally shared by hydrate paths."""
    print_info(
        console,
        "license admission: "
        f"admitted {buckets['admitted']}; c5-excluded {buckets['c5_excluded']}; "
        f"copyleft-unopted {buckets['c8_copyleft_unopted']}; "
        f"evidence-missing {buckets['license_evidence_missing']}",
    )


def _print_incomplete_manifests(console: Any, manifests: Any, *, prefix: str) -> None:
    """Print the reduced-yield warning shared by hydrate paths."""
    print_warning(
        console,
        f"{prefix} yield reduced: incomplete manifest(s) discovered and "
        "dropped: " + redact_text("; ".join(manifests)),
    )


def _hydrate_hub_dry_run(config: Any, console: Any) -> int:
    """Plan-only hydrate pass: pin, download, ingest, tally — no publication.

    Prints the per-code tallies plus per-repository license decision counts
    (value-free slugs and counts, issue #1094), with a fail-closed accounting
    invariant: every license-adjudicated candidate (imported plus license-gate
    rejections) lands in exactly one per-repository license bucket.
    """
    from daydream.archive import hydrate as _hydrate

    try:
        client = _hydrate._make_client(config.source_repo)
        source_commit, binding, discovery, ingest_results = _hydrate.prepare_hydration(config, client)
        ledger = _hydrate.build_import_ledger(
            config.stage_dir, revision=source_commit, source_commit=source_commit,
            discovery=discovery,
            ingest_results=ingest_results,
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
        _print_license_admission(console, license_admission)
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
        _print_incomplete_manifests(console, incomplete, prefix="dry-run")
    return 0
