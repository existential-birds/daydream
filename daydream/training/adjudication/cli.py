"""CLI handlers for the ``corpus adjudicate`` sub-verbs (issue #984).

``corpus adjudicate`` is the per-finding human-label workflow over the
deterministic adjudication queue:

- ``build``  — rebuild the queue from a hydrated index (``sessions.jsonl``
  under ``--index-root``), writing ``queue.json`` into ``--state-dir`` while
  preserving any existing observations (resume-safe; digest drift reopens).
- ``show``   — print unresolved items grouped by disposition plus completed
  counts.
- ``label``  — record a human observation for one ``--record-id`` or the next
  ``--batch N`` unresolved items in deterministic (record_id) order.
- ``export`` — merge the digest-pinned preview ledger with the observation
  store into the projector entry shape and write ``--out`` (or validate the
  rows only with ``--dry-run``).
- ``report`` — print outcome-bearing vs silver/task-only coverage, class
  balance, unresolved count, inter-rater agreement, and strata; with
  ``--as-of``, flag evidence observed after the pin; with ``--conflicts``,
  list disagreeing-rater findings oldest-first.
- ``materialize`` — materialize the preview annotation snapshot
  (``sessions.jsonl`` + ``preview-manifest.json``) for one curation pin.
- ``publish-state`` — publish adjudication state additively to the private
  Hub under ``annotations/<curation-id>/<snapshot-id>/`` with an optional
  batch checkpoint (``--batch-complete``).
- ``resume-state`` — restore published adjudication state onto a fresh VM
  from the Hub-side checkpoint, digest-verified.
- ``harvest-snapshot`` — canonical harvest of the materialized preview
  snapshot: drift gate, precedence merge, exactly-once
  ``label_observations`` append, and ``annotations.jsonl`` emission.
- ``publish-final`` — construct the final annotation staging bundle from
  pipeline state (no hand-authored files) and publish it additively to the
  private Hub under ``annotations/<curation-id>/<snapshot-id>/final/``;
  ``--dry-run`` builds and validates the bundle without constructing a client.
- ``import-local-observations`` — read-only import of surviving local
  archive/backup roots' immutable ``label_observations`` histories:
  read-only inventory, identity linkage against the roots' run metadata,
  content-digest dedupe, version gate, run-level classification,
  reason-coded accounting (bucket sum == source row count), and — unless
  ``--dry-run`` — an append-only merge into the ``--archive-dir`` archive
  (the hydrated stage's index.db — the single merge target) followed by
  fail-closed redaction + secret scan. ``--json`` prints the
  digest-stable import report (also written to
  ``--state-dir/import-report.json`` on a real run, alongside
  ``import-ledger.json``). Dry-run writes nothing (S2).

Every handler returns an int exit code (never calls ``sys.exit`` itself);
argparse converts malformed invocations into ``SystemExit(2)``. Unknown
record ids and missing state files fail closed with exit 1, naming the
offending identifier. No handler mutates anything outside ``--state-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daydream.archive.hydrate import HubUnavailableError, HydrationError, PublicDestinationError, _make_client
from daydream.archive.importer import (
    canonical_payload_digest,
    merge_imported_observations,
    redact_imported_metadata,
    redact_metadata_value,
    run_pure_import,
)
from daydream.archive.index import _get_connection
from daydream.archive.known_versions import STALE_LEGACY
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.export import validate_export_rows, write_export_rows
from daydream.training.adjudication.harvest import build_export_entries
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.observations import (
    append_observation,
    load_observations,
)
from daydream.training.adjudication.precedence import has_rater_conflict
from daydream.training.adjudication.preview import run_preview
from daydream.training.adjudication.publish import (
    publish_annotation_state,
    resume_annotation_state,
)
from daydream.training.adjudication.queue import build_queue
from daydream.training.adjudication.report import build_report
from daydream.training.labeler_versions import (
    ADJUDICATION_LABELER_VERSION,
    REPLY_CLASSIFIER_VERSION,
    RUBRIC_SCHEMA_VERSION,
)

__all__ = [
    "handle_adjudicate",
    "handle_build",
    "handle_export",
    "handle_harvest_snapshot",
    "handle_label",
    "handle_materialize",
    "handle_publish_state",
    "handle_report",
    "handle_resume_state",
    "handle_show",
]

_ANNOTATION_HUB_REPO = "existentialbirds/daydream-trajectories"

_HUMAN_ROLES = frozenset({"rater", "adjudicator"})

_QUEUE_FILENAME = "queue.json"
_OBSERVATIONS_FILENAME = "observations.jsonl"
_SESSIONS_FILENAME = "sessions.jsonl"
_PREVIEW_LEDGER_FILENAME = "preview-ledger.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise ValueError(f"{what} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_queue(state_dir: Path) -> list[dict[str, Any]]:
    queue = _load_json(state_dir / _QUEUE_FILENAME, "adjudication queue")
    if not isinstance(queue, list):
        raise ValueError(f"adjudication queue {state_dir / _QUEUE_FILENAME} is not a JSON list")
    return queue


def _resolved_record_ids(
    queue: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]]
) -> set[str]:
    """Record ids with a human observation matching the item's current digest.

    Matching the ``evidence_digest`` means a stale judgment (the item has
    since been reopened by digest drift) does not count as resolved — the
    reopened item stays in the open set.
    """
    human_by_record: dict[str, str] = {}
    for obs in observations:
        if obs.get("role") in _HUMAN_ROLES:
            human_by_record[str(obs["record_id"])] = str(obs["evidence_digest"])
    resolved: set[str] = set()
    for item in queue:
        record_id = str(item["record_id"])
        if human_by_record.get(record_id) == str(item["evidence_digest"]):
            resolved.add(record_id)
    return resolved


def _open_items(queue: Sequence[Mapping[str, Any]], resolved: set[str]) -> list[Mapping[str, Any]]:
    return [item for item in queue if str(item["record_id"]) not in resolved]


def _write_queue(state_dir: Path, items: list[dict[str, Any]]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / _QUEUE_FILENAME
    content = json.dumps(items, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=state_dir, prefix=f".{_QUEUE_FILENAME}.")
    with open(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    Path(tmp_name).replace(path)
    return path


def _positive_int(raw: str) -> int:
    """argparse type converter: reject values < 1 with a usage error (exit 2).

    Replaces a bare ``assert`` (stripped under ``python -O``) so ``--batch 0``
    or a negative batch is a malformed invocation, never a silent no-op.
    """
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError("--batch must be a positive integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError("--batch must be a positive integer")
    return value


def _add_pin_flags(parser: argparse.ArgumentParser) -> None:
    """Add the K2 preview-pin flags (shared by the snapshot-pipeline verbs).

    Versions come from ``labeler_versions`` constants, not flags. Missing
    components surface as exit 1 from the handler with the field named
    (``snapshot_id`` validates the assembled pin), never exit 2 — a missing
    pin component is a data problem, not a malformed invocation.
    """
    parser.add_argument("--curation-id", type=str, default=None, metavar="ID")
    parser.add_argument("--sanitized-hub-commit", type=str, default=None, metavar="SHA")
    parser.add_argument("--source-hub-commit", type=str, default=None, metavar="SHA")
    parser.add_argument("--archive-index-digest", type=str, default=None, metavar="HEX")
    parser.add_argument("--evidence-observed-at", type=str, default=None, metavar="ISO_TS")
    parser.add_argument("--as-of", type=str, default=None, metavar="ISO_TS")


def _pin_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Assemble the full K2 pin; versions come from ``labeler_versions``.

    ``as_of`` is the one component that may be empty or absent — the unpinned
    edge (``snapshot.snapshot_id`` hashes it as the empty string), so omitting
    ``--as-of`` materializes an unpinned snapshot instead of failing.
    """
    pin = {
        "curation_id": args.curation_id or "",
        "sanitized_hub_commit": args.sanitized_hub_commit or "",
        "source_hub_commit": args.source_hub_commit or "",
        "archive_index_digest": args.archive_index_digest or "",
        "evidence_observed_at": args.evidence_observed_at or "",
        "as_of": args.as_of or "",
        "labeler_version": ADJUDICATION_LABELER_VERSION,
        "rubric_version": RUBRIC_SCHEMA_VERSION,
        "classifier_version": REPLY_CLASSIFIER_VERSION,
    }
    missing = sorted(
        field for field, value in pin.items() if not value and field != "as_of"
    )
    if missing:
        raise ValueError(f"pin is missing required component(s): {missing}")
    return pin


def _add_state_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", type=Path, required=True, metavar="PATH",
                        help="Adjudication state directory (queue.json + observations.jsonl)")


def _build_adjudicate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daydream corpus adjudicate",
        description="Per-finding human adjudication queue + label workflow (issue #984).",
    )
    sub = parser.add_subparsers(dest="adjudicate_subverb", required=True)

    p_build = sub.add_parser("build", help="Build the adjudication queue from a hydrated index.")
    p_build.add_argument("--index-root", type=Path, required=True, metavar="PATH",
                         help="Hydrated index root containing sessions.jsonl")
    _add_state_dir(p_build)

    p_show = sub.add_parser("show", help="Show unresolved queue items grouped by disposition.")
    _add_state_dir(p_show)

    p_label = sub.add_parser("label", help="Record human observation(s) for queue item(s).")
    _add_state_dir(p_label)
    target = p_label.add_mutually_exclusive_group(required=True)
    target.add_argument("--record-id", type=str, default=None, metavar="HEX",
                        help="Record id (64-hex digest) of a single queue item")
    target.add_argument("--batch", type=_positive_int, default=None, metavar="N",
                        help="Label the next N unresolved items in deterministic order")
    p_label.add_argument("--disposition", type=str, required=True,
                         choices=sorted({"accepted", "rejected", "ambiguous", "unknown"}),
                         help="Human disposition for the finding(s)")
    p_label.add_argument("--rationale", type=str, required=True,
                         help="Why this disposition was chosen (stored provenance)")
    p_label.add_argument("--labeler", type=str, required=True,
                         help="Human labeler identity (stored provenance)")
    p_label.add_argument("--role", type=str, default="rater", choices=sorted(_HUMAN_ROLES),
                         help="Human role: rater (default) or adjudicator (conflict resolution)")
    p_label.add_argument("--valid-at", type=str, default=None, metavar="ISO_TS",
                         help="ISO-8601 valid-time pin (default: now)")

    p_export = sub.add_parser(
        "export", help="Merge the preview ledger + observations into the projector export shape."
    )
    p_export.add_argument("--index-root", type=Path, required=True, metavar="PATH",
                          help="Hydrated index root containing sessions.jsonl")
    _add_state_dir(p_export)
    p_export.add_argument("--out", type=Path, default=None, metavar="PATH",
                          help="Export JSONL path (required unless --dry-run)")
    p_export.add_argument("--dry-run", action="store_true",
                          help="Validate the export rows without writing anything")

    p_report = sub.add_parser(
        "report", help="Print adjudication coverage, class balance, inter-rater, strata."
    )
    p_report.add_argument("--index-root", type=Path, required=True, metavar="PATH",
                          help="Hydrated index root containing sessions.jsonl")
    _add_state_dir(p_report)
    p_report.add_argument("--conflicts", action="store_true",
                          help="List disagreeing-rater findings oldest-first instead of the report")
    p_report.add_argument("--as-of", type=str, default=None, metavar="ISO_TS",
                          help="ISO-8601 transaction-time pin; flag evidence observed "
                               "after this instant (default: no as_of comparison)")

    p_materialize = sub.add_parser(
        "materialize",
        help="Materialize the preview annotation snapshot (sessions.jsonl + manifest).",
    )
    p_materialize.add_argument("--index-root", type=Path, required=True, metavar="PATH",
                               help="Hydrated index root containing sessions.jsonl")
    p_materialize.add_argument("--out-dir", type=Path, required=True, metavar="PATH",
                               help="Directory for sessions.jsonl + preview-manifest.json")
    _add_pin_flags(p_materialize)

    p_publish = sub.add_parser(
        "publish-state",
        help="Publish adjudication state additively to the private Hub.",
    )
    _add_state_dir(p_publish)
    p_publish.add_argument("--manifest", type=Path, required=True, metavar="PATH",
                           help="Preview manifest pinning the snapshot")
    p_publish.add_argument("--hub-repo", type=str, default=_ANNOTATION_HUB_REPO, metavar="REPO",
                           help=f"Private Hub dataset repo (default: {_ANNOTATION_HUB_REPO})")
    p_publish.add_argument("--batch-complete", action="store_true",
                           help="Write the checkpoints/batch-latest.json checkpoint")

    p_resume = sub.add_parser(
        "resume-state",
        help="Restore published adjudication state onto a fresh VM (digest-verified).",
    )
    p_resume.add_argument("--manifest", type=Path, required=True, metavar="PATH",
                          help="Preview manifest pinning the snapshot")
    p_resume.add_argument("--stage-dir", type=Path, required=True, metavar="PATH",
                          help="Directory to restore the published state files into")
    p_resume.add_argument("--hub-repo", type=str, default=_ANNOTATION_HUB_REPO, metavar="REPO",
                          help=f"Private Hub dataset repo (default: {_ANNOTATION_HUB_REPO})")

    p_harvest = sub.add_parser(
        "harvest-snapshot",
        help="Canonical harvest: drift gate, precedence merge, label_observations append.",
    )
    p_harvest.add_argument("--index-root", type=Path, required=True, metavar="PATH",
                           help="Hydrated index root containing sessions.jsonl")
    p_harvest.add_argument("--materialize-dir", type=Path, required=True, metavar="PATH",
                           help="Directory produced by `materialize` (manifest + sessions.jsonl)")
    p_harvest.add_argument("--archive-dir", type=Path, required=True, metavar="PATH",
                           help="Archive directory holding the SQLite label-observations index")
    p_harvest.add_argument("--state-dir", type=Path, required=True, metavar="PATH",
                           help="Adjudication state directory (observations.jsonl source)")

    p_publish_final = sub.add_parser(
        "publish-final",
        help="Construct and publish the final annotation bundle (immutable snapshot).",
    )
    p_publish_final.add_argument("--index-root", type=Path, required=True, metavar="PATH",
                                 help="Hydrated index root containing sessions.jsonl")
    p_publish_final.add_argument("--materialize-dir", type=Path, required=True, metavar="PATH",
                                 help="Directory produced by `materialize` (manifest + sessions.jsonl)")
    _add_state_dir(p_publish_final)
    p_publish_final.add_argument("--archive-dir", type=Path, required=True, metavar="PATH",
                                 help="Archive directory holding the SQLite label-observations index")
    p_publish_final.add_argument("--curation-bundle-dir", type=Path, required=True, metavar="PATH",
                                 help="Curation bundle root digested into lineage.json's batch_fileset_digest")
    p_publish_final.add_argument("--hub-repo", type=str, default=_ANNOTATION_HUB_REPO, metavar="REPO",
                                 help=f"Private Hub dataset repo (default: {_ANNOTATION_HUB_REPO})")
    p_publish_final.add_argument("--dry-run", action="store_true",
                                 help="Build and validate the staging bundle without publishing to the Hub")

    p_import = sub.add_parser(
        "import-local-observations",
        help="Import surviving local archive/backup label-observation histories.",
    )
    p_import.add_argument("--archive-root", type=Path, action="append", required=True,
                          metavar="PATH",
                          help="Source archive/backup root holding index.db (repeatable)")
    p_import.add_argument("--index-root", type=Path, required=True, metavar="PATH",
                          help="Pinned hydrated index / materialized snapshot root the "
                               "import links session identity and finding evidence against")
    p_import.add_argument("--archive-dir", type=Path, required=True, metavar="PATH",
                          help="Hydrated archive directory holding the SQLite index the "
                               "import merges into — the single merge target")
    _add_state_dir(p_import)
    p_import.add_argument("--json", action="store_true",
                          help="Print the digest-stable import report as JSON")
    p_import.add_argument("--dry-run", action="store_true",
                          help="Plan the import without writing any state")
    p_import.add_argument("--publish", action="store_true",
                          help="Publish the merged state to the private Hub after the import "
                               "(checkpoint written for fresh-VM resume)")
    p_import.add_argument("--manifest", type=Path, metavar="PATH",
                          help="Preview manifest pinning curation_id + snapshot_id "
                               "(required with --publish)")
    p_import.add_argument("--hub-repo", type=str, default=_ANNOTATION_HUB_REPO, metavar="REPO",
                          help=f"Private Hub dataset repo (default: {_ANNOTATION_HUB_REPO})")

    return parser


def handle_build(argv: list[str]) -> int:
    """Handle ``corpus adjudicate build --index-root <path> --state-dir <path>``."""
    from daydream.ui import create_console, print_error, print_success

    args = _build_adjudicate_parser().parse_args(["build", *argv])
    sessions_path = args.index_root / _SESSIONS_FILENAME
    try:
        if not sessions_path.is_file():
            raise ValueError(f"hydrated index sessions file not found: {sessions_path}")
        raw = [json.loads(line) for line in sessions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (ValueError, json.JSONDecodeError) as exc:
        print_error(create_console(), "adjudicate build failed", str(exc))
        return 1
    observations = load_observations(args.state_dir / _OBSERVATIONS_FILENAME)
    prior: dict[str, Mapping[str, Any]] = {}
    for obs in observations:
        # Include model-suggested observations so build_queue can propagate
        # their stored review-required flag onto the rebuilt item (show then
        # renders the item as needing review).
        prior[str(obs["record_id"])] = obs
    try:
        items = build_queue(raw, prior_observations=prior)
    except ValueError as exc:
        print_error(create_console(), "adjudicate build failed", str(exc))
        return 1
    path = _write_queue(args.state_dir, items)
    reopened = sum(1 for item in items if item["status"] == "reopened")
    print_success(
        create_console(),
        f"Adjudication queue: {len(items)} item(s) "
        f"({reopened} reopened by evidence drift) -> {path}",
    )
    return 0


def handle_show(argv: list[str]) -> int:
    """Handle ``corpus adjudicate show --state-dir <path>``."""
    from daydream.ui import create_console, print_error

    args = _build_adjudicate_parser().parse_args(["show", *argv])
    try:
        queue = _load_queue(args.state_dir)
    except ValueError as exc:
        print_error(create_console(), "adjudicate show failed", str(exc))
        return 1
    observations = load_observations(args.state_dir / _OBSERVATIONS_FILENAME)
    open_items: list[Mapping[str, Any]] = _open_items(queue, _resolved_record_ids(queue, observations))

    by_disposition: dict[str, list[Mapping[str, Any]]] = {}
    for item in open_items:
        by_disposition.setdefault(str(item["disposition"]), []).append(item)
    for disposition in sorted(by_disposition):
        print(f"{disposition}: {len(by_disposition[disposition])}")
        for item in by_disposition[disposition]:
            status = str(item["status"])
            print(f"  {str(item['record_id'])[:12]}  {status}  {str(item['fingerprint'])}")
    print(f"unresolved: {len(open_items)} / {len(queue)}")
    return 0


def handle_label(argv: list[str]) -> int:
    """Handle ``corpus adjudicate label --state-dir <path> ...``."""
    from daydream.ui import create_console, print_error, print_success

    args = _build_adjudicate_parser().parse_args(["label", *argv])
    try:
        queue = _load_queue(args.state_dir)
    except ValueError as exc:
        print_error(create_console(), "adjudicate label failed", str(exc))
        return 1
    observations = load_observations(args.state_dir / _OBSERVATIONS_FILENAME)
    open_items: list[Mapping[str, Any]] = _open_items(queue, _resolved_record_ids(queue, observations))

    targets: list[Mapping[str, Any]]
    if args.record_id is not None:
        matches: list[Mapping[str, Any]] = [item for item in queue if str(item["record_id"]) == args.record_id]
        if not matches:
            print_error(
                create_console(),
                "adjudicate label failed",
                f"unknown --record-id: {args.record_id} (not in queue "
                f"{args.state_dir / _QUEUE_FILENAME})",
            )
            return 1
        open_ids = {str(item["record_id"]) for item in open_items}
        targets = [item for item in matches if str(item["record_id"]) in open_ids] or matches
    else:
        batch = args.batch
        assert batch is not None  # mutually exclusive group guarantees --record-id xor --batch
        targets = open_items[:batch]

    if not targets:
        print_success(create_console(), "Nothing to label: queue drained.")
        return 0

    valid_at = args.valid_at or _now_iso()
    observed_at = _now_iso()
    obs_path = args.state_dir / _OBSERVATIONS_FILENAME
    for item in targets:
        append_observation(obs_path, {
            "record_id": str(item["record_id"]),
            "disposition": args.disposition,
            "evidence_digest": str(item["evidence_digest"]),
            "evidence": item["evidence"],
            "labeler": args.labeler,
            "role": args.role,
            "rationale": args.rationale,
            "valid_at": valid_at,
            "observed_at": observed_at,
            "rubric_version": str(item["rubric_version"]),
        })
    remaining = [item for item in open_items if item not in targets]
    if remaining:
        next_id = str(remaining[0]["record_id"])
        print_success(create_console(), f"Labeled {len(targets)} item(s); next: {next_id[:12]}")
    else:
        print_success(create_console(), f"Labeled {len(targets)} item(s); queue drained.")
    return 0


def _load_sessions_for_index(index_root: Path) -> list[dict[str, Any]]:
    """Load the hydrated index's sessions.jsonl (fail-closed on missing/invalid)."""
    sessions_path = index_root / _SESSIONS_FILENAME
    if not sessions_path.is_file():
        raise ValueError(f"hydrated index sessions file not found: {sessions_path}")
    try:
        return [
            json.loads(line)
            for line in sessions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable hydrated index at {sessions_path}: {exc}") from exc


def handle_export(argv: list[str]) -> int:
    """Handle ``corpus adjudicate export --index-root <path> --state-dir <path>``."""
    from daydream.ui import create_console, print_error, print_success

    parser = _build_adjudicate_parser()
    args = parser.parse_args(["export", *argv])
    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run")
    ledger_path = args.state_dir / _PREVIEW_LEDGER_FILENAME
    try:
        if not ledger_path.is_file():
            run_preview(args.index_root, ledger_path)
        rows = build_export_entries(
            args.index_root, ledger_path,
            observations_path=args.state_dir / _OBSERVATIONS_FILENAME,
        )
        validate_export_rows(rows)
    except (ValueError, FileNotFoundError, HydrationError) as exc:
        print_error(create_console(), "adjudicate export failed", str(exc))
        return 1
    if args.dry_run:
        print_success(
            create_console(),
            f"Dry run OK: {len(rows)} export row(s) validated; nothing written.",
        )
        return 0
    assert args.out is not None
    sha = write_export_rows(rows, args.out)
    print_success(
        create_console(),
        f"Exported {len(rows)} row(s) -> {args.out} (sha256 {sha[:12]})",
    )
    return 0


def _report_items(
    index_root: Path,
    state_dir: Path,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Queue items enriched for the report: observation lists, effective
    dispositions, and the outcome-bearing fields ``tier``/``posterior_eligible``/
    ``evidence_after_as_of``, computed with the same authority as the export
    rows (``classify_tier`` + ``effective_adjudication`` gold-eligibility) so
    the 80% gate sees real adjudication state on the CLI path. The queue is
    the **complete** set (``include_decisive=True``), matching the final
    bundle's coverage report, so a human observation on an automatically
    adjudicated decisive record lands in the queue instead of raising, and
    the CLI's ``outcome_coverage`` cannot be structurally ~0 while the
    published gate passes. Shares one implementation with the final bundle's
    coverage report (``final_bundle._enrich_report_items``) so both gates
    agree."""
    from daydream.training.adjudication.final_bundle import _enrich_report_items

    items = build_queue(_load_sessions_for_index(index_root), include_decisive=True)
    observations = load_observations(state_dir / _OBSERVATIONS_FILENAME)
    return _enrich_report_items(items, observations, as_of=as_of)


def handle_report(argv: list[str]) -> int:
    """Handle ``corpus adjudicate report --index-root <path> --state-dir <path>``."""
    from daydream.ui import create_console, print_error

    args = _build_adjudicate_parser().parse_args(["report", *argv])
    try:
        enriched = _report_items(args.index_root, args.state_dir, as_of=args.as_of)
    except ValueError as exc:
        print_error(create_console(), "adjudicate report failed", str(exc))
        return 1
    if args.conflicts:
        _print_conflicts(enriched)
        return 0
    report = build_report(enriched)
    coverage = report["outcome_coverage"]
    balance = report["class_balance"]
    inter_rater = report["inter_rater"]
    gate = report["admission_gate"]
    print(f"outcome-bearing coverage: adjudicated {coverage['adjudicated']} / {coverage['total']}")
    print(f"silver/task-only: {report['silver_task_only_count']}")
    print(f"class balance: accepted={balance['accepted']} rejected={balance['rejected']}")
    print(f"unresolved: {report['unresolved']}")
    print(f"inter-rater: {inter_rater['items']} item(s), {inter_rater['agreeing']} agreeing")
    print(
        f"evidence after as_of: {len(report['evidence_after_as_of'])} "
        f"record(s){': ' + ', '.join(report['evidence_after_as_of']) if report['evidence_after_as_of'] else ''}"
    )
    print(
        f"admission gate: {gate['outcome_bearing_total']}/{gate['total']} outcome-bearing "
        f"(80% gate {'PASS' if gate['passes_80pct'] else 'FAIL'}, "
        f"class balance {'ok' if gate['class_balance_ok'] else 'unbalanced'})"
    )
    print("strata:")
    for (stack, profile), count in report["strata"].items():
        print(f"  ({stack}, {profile}): {count}")
    return 0


def _print_conflicts(enriched: list[dict[str, Any]]) -> None:
    """List disagreeing-rater findings oldest-first (by earliest observation)."""
    conflicts: list[tuple[str, str, list[dict[str, Any]]]] = []
    for item in enriched:
        human = [o for o in item["observations"] if o.get("role") in _HUMAN_ROLES]
        by_digest: dict[str, list[dict[str, Any]]] = {}
        for obs in human:
            by_digest.setdefault(str(obs["evidence_digest"]), []).append(obs)
        for digest, group in by_digest.items():
            if len({str(o["disposition"]) for o in group}) > 1 and has_rater_conflict(group):
                ordered = sorted(group, key=lambda o: str(o.get("observed_at", "")))
                conflicts.append((str(item["record_id"]), digest, ordered))
    conflicts.sort(key=lambda c: (str(c[2][0].get("observed_at", "")), c[0]))
    if not conflicts:
        print("no rater conflicts")
        return
    for record_id, digest, raters in conflicts:
        item = next(i for i in enriched if str(i["record_id"]) == record_id)
        print(f"conflict {record_id[:12]} ({item.get('stack')}, {item.get('profile')}) digest {digest[:12]}")
        for obs in raters:
            print(
                f"  {obs['labeler']}: {obs['disposition']} "
                f"(observed_at {obs.get('observed_at', '')})"
            )


def handle_materialize(argv: list[str]) -> int:
    """Handle ``corpus adjudicate materialize --index-root <path> --out-dir <path> ...``."""
    from daydream.ui import create_console, print_error, print_success

    parser = _build_adjudicate_parser()
    args = parser.parse_args(["materialize", *argv])
    try:
        pin = _pin_from_args(args)
        summary = run_materialize(args.index_root, args.out_dir, pin=pin)
    except (ValueError, HubUnavailableError, HydrationError) as exc:
        print_error(create_console(), "adjudicate materialize failed", str(exc))
        return 1
    print_success(
        create_console(),
        f"Materialized snapshot {summary['snapshot_id'][:12]}: "
        f"{summary['record_count']} record(s) from index revision "
        f"{summary['index_revision'][:12]} -> {args.out_dir}",
    )
    return 0


def handle_publish_state(argv: list[str]) -> int:
    """Handle ``corpus adjudicate publish-state --state-dir <path> --manifest <path>``."""
    from daydream.ui import create_console, print_error, print_success

    args = _build_adjudicate_parser().parse_args(["publish-state", *argv])
    try:
        client = _make_client(args.hub_repo)
        summary = publish_annotation_state(
            client, args.state_dir, manifest=args.manifest, batch_complete=args.batch_complete,
        )
    except (ValueError, HubUnavailableError, HydrationError, PublicDestinationError, FileNotFoundError) as exc:
        print_error(create_console(), "adjudicate publish-state failed", str(exc))
        return 1
    message = f"Published {len(summary['uploaded'])} file(s) under {summary['prefix']}"
    if "observation_count" in summary:
        message += f" (checkpoint: {summary['observation_count']} observation(s))"
    print_success(create_console(), message)
    return 0


def handle_resume_state(argv: list[str]) -> int:
    """Handle ``corpus adjudicate resume-state --manifest <path> --stage-dir <path>``."""
    from daydream.ui import create_console, print_error, print_success

    args = _build_adjudicate_parser().parse_args(["resume-state", *argv])
    try:
        client = _make_client(args.hub_repo)
        summary = resume_annotation_state(client, manifest=args.manifest, stage_dir=args.stage_dir)
    except (ValueError, HubUnavailableError, HydrationError, PublicDestinationError) as exc:
        print_error(create_console(), "adjudicate resume-state failed", str(exc))
        return 1
    if summary["restored"]:
        print_success(
            create_console(),
            f"Restored {len(summary['restored'])} file(s) "
            f"({summary['observation_count']} observation(s)) -> {args.stage_dir}",
        )
    else:
        print_success(create_console(), "Nothing published yet; empty state restored.")
    return 0


def handle_publish_final(argv: list[str]) -> int:
    """Handle ``corpus adjudicate publish-final`` (issue #1078, M4-M6).

    Both paths share ``build_final_bundle`` entirely: the staging bundle is
    constructed into ``<materialize-dir>/final-bundle`` and fully validated
    before any Hub interaction. With ``--dry-run`` the handler prints the
    per-file record counts + per-disposition summary, the 80%
    human-adjudication admission-gate verdict from the written coverage
    report, and returns 0 without ever constructing a client; otherwise the
    bundle is published via :func:`publish_final_annotation_bundle`
    (private-repo gate, 80% admission-gate refusal, secret scan, SHA256SUMS,
    additive upload, clean-download verify, ``_SUCCESS`` last).
    """
    from daydream.training.adjudication.final_bundle import build_final_bundle
    from daydream.training.adjudication.publish import publish_final_annotation_bundle
    from daydream.ui import create_console, print_error, print_success

    args = _build_adjudicate_parser().parse_args(["publish-final", *argv])
    bundle_dir = args.materialize_dir / "final-bundle"
    try:
        summary = build_final_bundle(
            index_root=args.index_root,
            materialize_dir=args.materialize_dir,
            archive_dir=args.archive_dir,
            out_dir=bundle_dir,
            curation_bundle_dir=args.curation_bundle_dir,
            observations_path=args.state_dir / _OBSERVATIONS_FILENAME,
        )
    except (ValueError, HubUnavailableError, HydrationError, PublicDestinationError, FileNotFoundError) as exc:
        print_error(create_console(), "adjudicate publish-final failed", str(exc))
        return 1
    if args.dry_run:
        counts = " ".join(
            f"{disposition}={count}" for disposition, count in summary["disposition_counts"].items()
        )
        try:
            report = json.loads(
                (bundle_dir / "coverage-report.json").read_text(encoding="utf-8")
            )
            gate = report["admission_gate"]
            coverage = report["outcome_coverage"]
        except (OSError, ValueError, KeyError) as exc:
            print_error(create_console(), "adjudicate publish-final failed", str(exc))
            return 1
        print_success(
            create_console(),
            f"Dry-run: final bundle validated at {bundle_dir} — "
            f"{summary['record_count']} record(s) across "
            f"{', '.join(summary['files'])} ({counts}); "
            f"80% admission gate {'PASS' if gate['passes_80pct'] else 'FAIL'} "
            f"({coverage['adjudicated']}/{coverage['total']} outcome-bearing "
            f"adjudicated); nothing published",
        )
        return 0
    try:
        client = _make_client(args.hub_repo)
        result = publish_final_annotation_bundle(
            client, bundle_dir, manifest=args.materialize_dir / "preview-manifest.json",
            verify_download=True,
        )
        # The snapshot-id label for the success message is read here, inside
        # the publish try/except, so a missing/corrupt manifest can never
        # escape the handler as an uncaught exception after a successful
        # publish — every handler must return an int exit code.
        manifest = json.loads(
            (args.materialize_dir / "preview-manifest.json").read_text(encoding="utf-8")
        )
        snapshot_id = manifest.get("snapshot_id", "")
    except (ValueError, HubUnavailableError, HydrationError, PublicDestinationError, FileNotFoundError) as exc:
        print_error(create_console(), "adjudicate publish-final failed", str(exc))
        return 1
    print_success(
        create_console(),
        f"Published final annotation bundle ({summary['record_count']} record(s)) "
        f"under {result['prefix']} (hub commit {result['hub_commit_sha']}, "
        f"snapshot {snapshot_id})",
    )
    return 0


def handle_harvest_snapshot(argv: list[str]) -> int:
    """Handle ``corpus adjudicate harvest-snapshot --index-root --materialize-dir ...``."""
    from daydream.ui import create_console, print_error, print_success

    args = _build_adjudicate_parser().parse_args(["harvest-snapshot", *argv])
    try:
        summary = run_canonical_harvest(
            args.index_root,
            args.materialize_dir,
            args.archive_dir,
            observations_path=args.state_dir / _OBSERVATIONS_FILENAME,
        )
    except (ValueError, HubUnavailableError, HydrationError, FileNotFoundError) as exc:
        print_error(create_console(), "adjudicate harvest-snapshot failed", str(exc))
        return 1
    print_success(
        create_console(),
        f"Harvested {summary['record_count']} record(s): "
        f"{summary['appended_sessions']} session(s) appended, "
        f"{summary['skipped_sessions']} skipped (already harvested), "
        f"{summary['human_adjudicated']} human-adjudicated",
    )
    return 0


# Full modern ``label_observations`` column list. Legacy-schema roots are
# introspected via ``PRAGMA table_info`` and missing version columns are
# surfaced as the ``"legacy"`` sentinel (never gold-eligible, still imported
# as evidence).
_IMPORT_OBSERVATION_COLUMNS = (
    "session_id",
    "observed_at",
    "labels",
    "pr_state",
    "labeler_version",
    "evidence_sha",
    "rubric_json",
    "valid_at",
    "reward_version",
    "reward_json",
    "composite_reward",
    "reviewer_logins",
    "has_posterior",
    "source",
    "labeler_policy_version",
    "reply_classifier_version",
    "reply_evidence_digest",
    "legacy",
)

_IMPORT_VERSION_COLUMNS = (
    "labeler_policy_version",
    "reply_classifier_version",
    "reply_evidence_digest",
)


def _inventory_import_root(root: Path) -> dict[str, Any]:
    """Read-only inventory of one archive/backup root (M1, Assumption 4).

    Opens ``root/index.db`` in ``mode=ro`` (the source is never written),
    reads every ``label_observations`` row ordered by ``observed_at`` ASC,
    fills version columns missing from a legacy schema with ``"legacy"``, and
    enriches each row with its run's ``repo_slug``/``base_sha``/``head_sha``
    so identity linkage can resolve the session.

    Returns:
        ``{"rows": [...], "source_digest": <sha256 hex of index.db bytes>,
        "runs": {session_id: full runs row dict}}``.

    Raises:
        ValueError: When the root has no ``index.db`` or no
            ``label_observations`` table — always naming the path.
    """
    db_path = root / "index.db"
    if not db_path.is_file():
        raise ValueError(
            f"archive root {root} has no index.db; not a daydream archive/backup root"
        )
    source_digest = hashlib.sha256(db_path.read_bytes()).hexdigest()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "label_observations" not in tables:
            raise ValueError(
                f"archive root {root} has no label_observations table in index.db"
            )
        columns = [str(row[1]) for row in conn.execute("PRAGMA table_info(label_observations)")]
        selected = [column for column in _IMPORT_OBSERVATION_COLUMNS if column in columns]
        rows = [
            dict(row)
            for row in conn.execute(
                f"SELECT {', '.join(selected)} FROM label_observations ORDER BY observed_at ASC"
            )
        ]
        runs = (
            {str(run["session_id"]): dict(run) for run in conn.execute("SELECT * FROM runs")}
            if "runs" in tables
            else {}
        )
    finally:
        conn.close()
    for row in rows:
        for column in _IMPORT_VERSION_COLUMNS:
            if column not in row:
                row[column] = STALE_LEGACY
        if "source" not in row:
            # A pre-``source`` legacy label_observations table: default the
            # precedence marker to the writer's own default ("auto") so no
            # downstream ``row["source"]`` read raises KeyError on a legacy row.
            row["source"] = "auto"
        runs_dir = root / "runs" / str(row["session_id"])
        if runs_dir.is_dir():
            # Derivative content digest for identity linkage: the hydrated
            # index side derives the same digest over its own runs/<sid>
            # directory, so a matching pair links by session_id.
            from daydream.archive.sanitize import _derivative_digest

            row["derivative_digest"] = _derivative_digest(runs_dir)
        run = runs.get(str(row["session_id"]))
        if run is not None:
            for field in ("repo_slug", "base_sha", "head_sha", "remote_url", "source_path"):
                row[field] = run.get(field)
    return {"rows": rows, "source_digest": source_digest, "runs": runs}


def _seed_target_runs(
    state_dir: Path, sessions: set[str], runs: dict[str, dict[str, Any]]
) -> None:
    """Materialize the source runs rows for *sessions* into the state-dir archive.

    ``append_label_observation`` requires the session to exist in ``runs``;
    importing into the state archive therefore materializes the source run
    rows first (deterministic: sessions in sorted order). A session that does
    not yet exist is inserted with the source row's full column set.

    A session that already exists is **never displaced**: its populated target
    columns — ``status``/``archived_at``, the ``profile_*`` fields, the cost
    metrics, and the writer-owned denormalized cache mirrors
    (``outcome_labels``/``labeled_at``/``rubric_json``/``composite_reward``/
    ``has_posterior``) — survive an overlapping re-import of an older backup,
    and only NULL target columns are filled from the source snapshot. The
    observation merge is append-only/no-displacement, so the run-row seeding
    follows the same rule: an older overlapping backup must never overwrite
    newer target state (a deduped no-op append never refreshes the cache, so
    the target row remains the projection authority).
    Credential-bearing ``remote_url``/``source_path`` values are redacted
    fail-closed before insertion (M9/AC6).
    """
    conn = _get_connection(state_dir)
    try:
        existing = {
            str(row["session_id"])
            for row in conn.execute("SELECT session_id FROM runs").fetchall()
        }
        for session_id in sorted(sessions):
            run = dict(runs[session_id])
            # Fail-closed redaction: never persist a credential-bearing URL /
            # absolute path from the source runs onto the state archive.
            for field in ("remote_url", "source_path"):
                if field in run:
                    run[field] = redact_metadata_value(run[field])
            if session_id in existing:
                # No-displacement materialization: fill only the target
                # columns the source snapshot can add (currently NULL); every
                # populated target value — including the writer-owned cache
                # mirrors — wins over the overlapping backup.
                target = conn.execute(
                    "SELECT * FROM runs WHERE session_id = ?", (session_id,)
                ).fetchone()
                fill = [
                    (column, run[column])
                    for column in run
                    if target[column] is None and run[column] is not None
                ]
                if fill:
                    assignments = ", ".join(f"{column} = ?" for column, _ in fill)
                    conn.execute(
                        f"UPDATE runs SET {assignments} WHERE session_id = ?",
                        [value for _, value in fill] + [session_id],
                    )
                continue
            columns = list(run)
            placeholders = ", ".join("?" for _ in columns)
            conn.execute(
                f"INSERT OR REPLACE INTO runs ({', '.join(columns)}) VALUES ({placeholders})",
                [run[column] for column in columns],
            )
        conn.commit()
    finally:
        conn.close()


class _ImportBlockedError(Exception):
    """Fail-closed redaction gate raised by the import merge phase.

    Raised (instead of merging) when the post-redaction secret scan is dirty:
    the payload must not be imported. ``message`` carries the composed blocked
    reason the handler prints verbatim.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _ImportPublishError(Exception):
    """Fail-closed publish-payload gate raised by the import publish phase.

    Raised when the state archive is missing a publishable adjudication-state
    file: the import itself only writes ``index.db``, so publishing a fresh
    state-dir would fail with a bare ``FileNotFoundError``. ``message`` carries
    the composed prerequisite hint the handler prints verbatim.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _inventory_import_roots(
    roots: Sequence[Path], *, console: Any | None = None
) -> dict[str, Any]:
    """Read-only inventory of every archive root for the import pipeline.

    Composes the per-root :func:`_inventory_import_root` reads into the
    pipeline's merge inputs: the dedupe/merge row lists, the per-root source
    records, and the source-runs maps the merge phase seeds from. Per-root
    progress is printed on ``console`` when given (the human-readable run;
    ``--json`` callers pass ``None`` to keep stdout machine-readable).

    Returns:
        ``{"inventories", "sources", "runs_by_session",
        "repo_slug_sha_lookup"}``.

    Raises:
        ValueError/sqlite3.Error/OSError: Any inventory failure (missing
            ``index.db``, unreadable root) — the caller's fail-closed surface.
    """
    inventories: list[list[dict[str, Any]]] = []
    sources: list[dict[str, Any]] = []
    runs_by_session: dict[str, dict[str, Any]] = {}
    repo_slug_sha_lookup: dict[tuple[str, str, str], Any] = {}
    for root in roots:
        inventory = _inventory_import_root(root)
        inventories.append(inventory["rows"])
        sources.append(
            {
                "archive_root": str(root),
                "row_count": len(inventory["rows"]),
                "source_digest": inventory["source_digest"],
            }
        )
        for session_id, run in inventory["runs"].items():
            runs_by_session.setdefault(session_id, run)
            repo_slug, base_sha, head_sha = (
                run.get("repo_slug"),
                run.get("base_sha"),
                run.get("head_sha"),
            )
            if repo_slug and base_sha and head_sha:
                repo_slug_sha_lookup.setdefault(
                    (str(repo_slug), str(base_sha), str(head_sha)), session_id
                )
        if console is not None:  # per-root progress (S3)
            console.print(
                f"import: inventoried {len(inventory['rows'])} label_observations(s) "
                f"from {root}"
            )
    return {
        "inventories": inventories,
        "sources": sources,
        "runs_by_session": runs_by_session,
        "repo_slug_sha_lookup": repo_slug_sha_lookup,
    }


def _link_imported_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Remap identity-linked import rows onto their Hub session ids.

    Only identity-linked rows can merge: unmatched/conflict rows stay in the
    ledger's reason-coded buckets (never silently dropped, never misfiled).
    Each linked row carries an inventory-time payload digest — the merge's
    fail-closed drift gate recomputes exactly this digest before any write.
    """
    linked_rows: list[dict[str, Any]] = []
    for row in result["rows"]:
        link = result["link"]["linked"].get(str(row["session_id"]))
        if link is None:
            continue
        merged_row = dict(row)
        merged_row["session_id"] = link["hub_session_id"]
        # Inventory-time payload digest: the merge's fail-closed drift gate
        # recomputes exactly this digest before any write.
        merged_row["payload_digest"] = canonical_payload_digest(
            merged_row, include_observed_at=merged_row["source"] != "auto"
        )
        linked_rows.append(merged_row)
    return linked_rows


def _load_import_index_sessions(index_root: Path) -> list[dict[str, Any]]:
    """Load the pinned index's sessions for the import identity derivation.

    A materialized snapshot root carries ``sessions.jsonl`` (the hydrated
    session shape); a hydrated staging archive carries ``index.db`` and
    derives its sessions from the ``label_observations`` rows via the shared
    fail-closed materialize adapter. Anything else is a derive failure —
    no empty-literal fallback.
    """
    if (index_root / _SESSIONS_FILENAME).is_file():
        return _load_sessions_for_index(index_root)
    if (index_root / "index.db").is_file():
        from daydream.training.adjudication.materialize import _sessions_from_hydrated_stage

        sessions, _ = _sessions_from_hydrated_stage(index_root)
        return sessions
    raise ValueError(
        f"import index root {index_root} has neither sessions.jsonl nor index.db; "
        "not a hydrated index or materialized snapshot"
    )


def _hydrated_identity_index(
    sessions: list[dict[str, Any]], index_root: Path
) -> dict[str, dict[str, Any]]:
    """Derive the ``link_session_identity`` hydrated-index map from the pinned
    index's sessions — never an empty literal.

    Each entry carries the session's derivative content digest (sha256 over
    ``<index_root>/runs/<session_id>`` when that directory exists, else
    ``None``) plus the identity ``record_id``.
    """
    from daydream.archive.sanitize import _derivative_digest

    hydrated: dict[str, dict[str, Any]] = {}
    for session in sessions:
        session_id = str(session["session_id"])
        runs_dir = index_root / "runs" / session_id
        hydrated[session_id] = {
            "derivative_digest": _derivative_digest(runs_dir) if runs_dir.is_dir() else None,
            "record_id": session_id,
        }
    return hydrated


def _projector_findings_map(
    sessions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Derive the per-finding projected map from the pinned index's sessions.

    ``project_findings`` is the single enumeration authority; each finding
    contributes its ``record_id``, its per-finding evidence digest (so the
    run-level classification's exact-identity match is discriminating even
    for multi-finding sessions), and its finding fingerprint.
    """
    from daydream.training.corpus_v2.projector import project_findings
    from daydream.training.labeler_versions import reply_evidence_digest

    findings_map: dict[str, list[dict[str, Any]]] = {}
    for session in sessions:
        session_id = str(session["session_id"])
        rows: list[dict[str, Any]] = []
        for finding in project_findings(session):
            evidence = finding["evidence"]
            if not isinstance(evidence, list):
                raise ValueError(
                    f"session {session_id!r}: projected finding "
                    f"{finding['finding_fingerprint']!r} carries malformed evidence"
                )
            rows.append(
                {
                    "record_id": finding["record_id"],
                    "evidence_sha": reply_evidence_digest(evidence),
                    "fingerprint": finding["finding_fingerprint"],
                }
            )
        findings_map[session_id] = rows
    return findings_map


def _identity_summary(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-session identity-resolution summary for the import report.

    ``matched_by`` comes from the identity link (``session_id`` /
    ``repo_slug_sha``, or ``None`` for an unmatched session);
    ``validation_outcome`` from the run-level classification (``matched``
    when the evidence digest pinned exactly one projected finding,
    ``ambiguous`` when it could not, ``run_level_only`` when the session has
    no projected findings, and ``unmatched`` when identity itself failed).
    Deterministic (sorted by session id) so the report stays digest-stable.
    """
    link = result["link"]
    run_level = result["run_level"]
    summary: dict[str, dict[str, Any]] = {}
    for session_id in sorted(
        set(link["linked"]) | set(link["unmatched"]) | set(link["identity_conflict"])
    ):
        if session_id in link["linked"]:
            matched_by: str | None = link["linked"][session_id]["matched_by"]
            if session_id in run_level["per_finding"]:
                outcome = "matched"
            elif session_id in run_level["ambiguous_run_mapping"]:
                outcome = "ambiguous"
            else:
                outcome = "run_level_only"
        else:
            matched_by = None
            outcome = "unmatched"
        summary[session_id] = {"matched_by": matched_by, "validation_outcome": outcome}
    return summary


def _write_import_merge(
    archive_dir: Path,
    state_dir: Path,
    linked_rows: list[dict[str, Any]],
    runs_by_session: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Redaction-gated merge of the linked rows into the hydrated archive.

    The merge target is ``--archive-dir`` (the hydrated stage's index.db —
    the single archive the canonical chain reads); ``--state-dir`` remains
    scratch for the scan artifacts and publish payload staging only.

    Fail-closed gates first, before any state write (M9/AC6): the redaction +
    secret scan and the merge's drift / malformed-row / timestamp gate both
    run before the seed/merge, so a blocked or drifted import cannot leave
    partially seeded runs or unredacted observation rows committed in the
    hydrated archive (and every later run re-blocking on the same dirty
    payload). The merge commits the scan's **redacted** payload — never the
    unredacted originals — so credential-bearing metadata cannot reach the
    archive. ``dry_run=True`` never touches the archive, so the gate can run
    unconditionally.

    Returns:
        ``{"planned", "appended", "deduped", "scan"}`` — the merge outcome
        plus the redaction result for the report's ``redaction`` block.

    Raises:
        _ImportBlockedError: When the post-redaction scan is dirty — the
            payload cannot be imported.
        ValueError/sqlite3.Error/OSError: Redaction, drift-gate, seed, or
            merge failures — the caller's fail-closed surface.
    """
    scan = redact_imported_metadata(linked_rows, scan_dir=state_dir / "import-scan")
    # Pre-write gates over the raw linked rows: drift, malformed-row, and
    # timestamp validation all run before the seed/merge commits anything.
    merge_imported_observations(state_dir, linked_rows, dry_run=True)
    if scan["blocked"]:
        # Never leave the dirty scan artifact behind: a later ``scan_run_dir``
        # pass over the state dir would flag the persisted payload as a
        # foreign dirty artifact (M9/AC6).
        (state_dir / "import-scan" / "payload.json").unlink(missing_ok=True)
        message = "; ".join(scan["blocked_reasons"]) + f" ({scan['scan_summary']})"
        raise _ImportBlockedError(message)
    # The merge commits the scan's *redacted* payload — never the unredacted
    # originals — so credential-bearing metadata cannot reach the state
    # archive (M9). Redaction is deterministic over the same in-memory rows
    # the write-path drift gate re-validates; recompute the payload digests
    # over the redacted content so that gate stays consistent.
    redacted_rows = scan["payload"]
    for row in redacted_rows:
        row["payload_digest"] = canonical_payload_digest(
            {k: v for k, v in row.items() if k != "payload_digest"},
            include_observed_at=row["source"] != "auto",
        )
    _seed_target_runs(
        archive_dir,
        {str(row["session_id"]) for row in redacted_rows},
        runs_by_session,
    )
    merged = merge_imported_observations(archive_dir, redacted_rows, dry_run=False)
    return {
        "planned": merged["planned"],
        "appended": merged["appended"],
        "deduped": merged["deduped"],
        "scan": scan,
    }


def _build_import_report(
    sources: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    dry_run: bool,
    merge_state: dict[str, Any],
    identity_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compose the digest-stable import report (S1).

    ``merge_state`` is the merge phase result: the ``merge_imported_observations``
    shape (``planned``/``appended``/``deduped``) for a dry run, or the
    :func:`_write_import_merge` shape — the same keys plus the fail-closed
    ``scan`` — for a real run. Only the real run carries the ``redaction``
    block; dry runs never write it (S2).
    """
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "sources": sources,
        "deduped_count": result["deduped_count"],
        "accounting": dict(result["accounting"]),
        "identity_summary": identity_summary,
        "merge": {
            "planned": len(merge_state["planned"]),
            "appended": merge_state["appended"],
            "deduped": merge_state["deduped"],
        },
    }
    if not dry_run:
        report["redaction"] = {
            "blocked": bool(merge_state["scan"]["blocked"]),
            "reasons": list(merge_state["scan"]["blocked_reasons"]),
        }
    return report


def _publish_import_state(state_dir: Path, hub_repo: str, manifest: Path) -> dict[str, Any]:
    """Publish the imported adjudication state additively to the private Hub.

    The publish composition requires the adjudication-state payload
    (queue.json / observations.jsonl / preview-ledger.json) and the state
    archive index (index.db — the ``--archive-dir`` merge target staged into
    the state dir for publication) to exist; fail closed with a prerequisite
    hint on a fresh state-dir instead of a bare ``FileNotFoundError``.
    Reuses the existing publication: the private repo hard-fail and the S1
    secret scan are ``publish_annotation_state``'s own fail-closed gates, so
    a public destination or a credential-shaped payload is refused before
    any byte reaches the Hub. The published bundle carries ``index.db`` and
    the checkpoint always records its digest, so a fresh-VM resume restores
    the published archive index, not just the queue/report (AC5: the
    VM-local ``--state-dir`` is scratch only).

    Returns:
        ``{"prefix": ..., "uploaded": ...}``.

    Raises:
        _ImportPublishError: When the state archive is missing a publishable
            adjudication-state file.
        ValueError/HubUnavailableError/HydrationError/PublicDestinationError/
        FileNotFoundError: Propagated from the publication steps.
    """
    missing = [
        name
        for name in ("queue.json", "observations.jsonl", "preview-ledger.json", "index.db")
        if not (state_dir / name).is_file()
    ]
    if missing:
        raise _ImportPublishError(
            "state archive is missing publishable adjudication-state file(s): "
            + ", ".join(missing)
            + "; run `corpus adjudicate build`/`label`/`export` to produce the "
            "publication payload before --publish"
        )
    client = _make_client(hub_repo)
    published = publish_annotation_state(
        client, state_dir, manifest=manifest, batch_complete=True,
    )
    return {"prefix": published["prefix"], "uploaded": published["uploaded"]}


def handle_import_local_observations(argv: list[str]) -> int:
    """Handle ``corpus adjudicate import-local-observations`` (KD6, S1/S2/S3).

    Thin composition over the independently testable pipeline units: read-only
    inventory (``_inventory_import_roots``), the pinned-index identity
    derivation (``_load_import_index_sessions`` -> ``_hydrated_identity_index``
    + ``_projector_findings_map`` — real identity inputs, never empty
    literals), the pure import pipeline (``run_pure_import``), identity
    linkage (``_link_imported_rows``), then — unless ``--dry-run`` — the
    redaction-gated append-only merge into the ``--archive-dir`` archive
    (``_write_import_merge``), the digest-stable report
    (``_build_import_report``, carrying the per-session ``identity_summary``),
    and the optional publish (``_publish_import_state``). The state-dir
    ``index.db`` is never written by the import; ``--state-dir`` stays for
    the scan artifacts, report/ledger, and ``--publish`` payload staging.
    This handler owns only arg parsing, the fail-closed error surface, and
    output; ``--json`` prints the report and a real run writes it to
    ``--state-dir/import-report.json`` with the ledger in
    ``--state-dir/import-ledger.json``. Dry-run writes nothing (S2).
    """
    from daydream.ui import create_console, print_error, print_success

    parser = _build_adjudicate_parser()
    args = parser.parse_args(["import-local-observations", *argv])
    if args.publish:
        if args.dry_run:
            parser.error("--publish cannot be combined with --dry-run")
        if args.manifest is None:
            parser.error("--publish requires --manifest")
    console = create_console()
    try:
        inventory = _inventory_import_roots(
            args.archive_root, console=None if args.json else console
        )
        # Real identity inputs, derived from the pinned index — never empty
        # literals: the hydrated-index map for session linkage and the
        # projector's per-finding map for exact run-level evidence matching.
        sessions = _load_import_index_sessions(args.index_root)
        hydrated_index = _hydrated_identity_index(sessions, args.index_root)
        projector_findings = _projector_findings_map(sessions)
        result = run_pure_import(
            inventory["inventories"],
            hydrated_index=hydrated_index,
            repo_slug_sha_lookup=inventory["repo_slug_sha_lookup"],
            projector_findings=projector_findings,
            unmatched_identity_less=True,
        )
        linked_rows = _link_imported_rows(result)
        # Dry-run still exercises the merge's fail-closed drift gate, but the
        # planned appends are counted, never written (S2). The real path runs
        # the redaction + secret scan and the drift / malformed-row gate
        # before any state write (M9/AC6). The merge targets --archive-dir —
        # the hydrated stage's index.db — never the state-dir index.
        merge_state = (
            merge_imported_observations(args.archive_dir, linked_rows, dry_run=True)
            if args.dry_run
            else _write_import_merge(
                args.archive_dir, args.state_dir, linked_rows, inventory["runs_by_session"]
            )
        )
    except _ImportBlockedError as exc:
        print_error(
            console,
            "adjudicate import-local-observations blocked by unredactable metadata",
            exc.message,
        )
        return 1
    except (ValueError, sqlite3.Error, OSError, HubUnavailableError, HydrationError) as exc:
        print_error(console, "adjudicate import-local-observations failed", str(exc))
        return 1

    report = _build_import_report(
        inventory["sources"], result, dry_run=bool(args.dry_run), merge_state=merge_state,
        identity_summary=_identity_summary(result),
    )
    if args.publish:
        try:
            report["publish"] = _publish_import_state(
                args.state_dir, args.hub_repo, args.manifest
            )
        except _ImportPublishError as exc:
            print_error(
                console, "adjudicate import-local-observations publish failed", exc.message
            )
            return 1
        except (ValueError, HubUnavailableError, HydrationError, PublicDestinationError, FileNotFoundError) as exc:
            print_error(console, "adjudicate import-local-observations failed", str(exc))
            return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_success(
            console,
            f"Import {'planned' if args.dry_run else 'complete'}: "
            f"{sum(report['accounting'].values()) + report['deduped_count']} source row(s) across "
            f"{len(inventory['sources'])} root(s), {report['deduped_count']} deduped; "
            + (
                f"{report['merge']['planned']} merge(s) planned; nothing written"
                if args.dry_run
                else f"{report['merge']['appended']} appended, "
                f"{report['merge']['deduped']} deduped by the writer"
            )
            + (f"; published to {report['publish']['prefix']}" if args.publish else ""),
        )

    if not args.dry_run:
        args.state_dir.mkdir(parents=True, exist_ok=True)
        (args.state_dir / "import-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.state_dir / "import-ledger.json").write_text(
            json.dumps(result["ledger"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


_HANDLERS = {
    "build": handle_build,
    "show": handle_show,
    "label": handle_label,
    "export": handle_export,
    "report": handle_report,
    "materialize": handle_materialize,
    "publish-state": handle_publish_state,
    "resume-state": handle_resume_state,
    "harvest-snapshot": handle_harvest_snapshot,
    "publish-final": handle_publish_final,
    "import-local-observations": handle_import_local_observations,
}


def handle_adjudicate(argv: list[str]) -> int:
    """Route ``corpus adjudicate <sub-verb> [...]`` to its handler.

    Bare and unknown sub-verbs are rejected by argparse (usage + exit 2);
    handler exit codes propagate unchanged.
    """
    if not argv:
        _build_adjudicate_parser().parse_args([])
    subverb, rest = argv[0], argv[1:]
    if subverb not in _HANDLERS:
        _build_adjudicate_parser().parse_args([subverb])
    return int(_HANDLERS[subverb](rest))
