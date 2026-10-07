"""Human judgment commands over canonical records and disposable local queues."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore, SnapshotRecords
from daydream.json_utils import atomic_write_bytes, canonical_json
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.export import validate_export_rows, write_export_rows
from daydream.training.adjudication.harvest import build_export_entries
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.observations import _DISPOSITIONS, append_observation, prior_adjudications
from daydream.training.adjudication.precedence import HUMAN_ROLES, has_rater_conflict
from daydream.training.adjudication.preview import run_preview
from daydream.training.adjudication.queue import build_queue
from daydream.training.adjudication.report import build_report, enrich_report_items
from daydream.training.record_evidence import finding_observations, sessions_from_snapshot, validate_output_path
from daydream.ui import create_console, print_error, print_success


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError("--batch must be a positive integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError("--batch must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daydream corpus adjudicate", description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    for name in ("build", "materialize", "preview", "export", "report", "harvest-snapshot"):
        p = sub.add_parser(name)
        p.add_argument("--store", type=Path, required=True)
        p.add_argument("--snapshot-id", required=True)
        if name in ("build", "preview", "export"):
            p.add_argument("--state-dir", type=Path, required=True)
        if name == "materialize":
            p.add_argument("--out-dir", type=Path, required=True)
            p.add_argument("--dry-run", action="store_true")
        if name == "harvest-snapshot":
            p.add_argument("--materialize-dir", type=Path, required=True)
        if name == "export":
            p.add_argument("--out", type=Path)
            p.add_argument("--dry-run", action="store_true")
        if name == "report":
            p.add_argument("--conflicts", action="store_true")
    show = sub.add_parser("show")
    show.add_argument("--state-dir", type=Path, required=True)
    label = sub.add_parser("label")
    label.add_argument("--state-dir", type=Path, required=True)
    target = label.add_mutually_exclusive_group(required=True)
    target.add_argument("--record-id")
    target.add_argument("--batch", type=_positive_int)
    label.add_argument("--disposition", required=True, choices=sorted(_DISPOSITIONS))
    label.add_argument("--rationale", required=True)
    label.add_argument("--labeler", required=True)
    label.add_argument("--role", choices=sorted(HUMAN_ROLES), default="rater")
    label.add_argument("--valid-at")
    return parser


def _write(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (canonical_json(value) + "\n").encode(), mode=0o600)


def _queue(records: SnapshotRecords) -> list[dict[str, object]]:
    return build_queue(
        sessions_from_snapshot(records, overlay_judgments=False),
        prior_observations=prior_adjudications([o for o in finding_observations(records) if o["role"] != "automatic"]),
    )


def _state(path: Path) -> tuple[LocalRecordStore, SnapshotRecords, list[dict[str, Any]]]:
    reference = json.loads((path / "reference.json").read_text())
    if reference.get("schema_version") != "daydream.annotation-queue.v2":
        raise ValueError("invalid annotation queue reference")
    store = LocalRecordStore(Path(reference["store"]))
    records = store.read_snapshot(reference["snapshot_id"])
    queue = json.loads((path / "queue.json").read_text())
    if queue != _queue(records):
        raise ValueError("annotation queue differs from its frozen evidence; rebuild queue")
    return store, records, queue


def _resolved(queue: list[dict[str, Any]], observations: list[dict[str, Any]]) -> set[str]:
    prior = prior_adjudications(observations)
    return {
        str(i["record_id"])
        for i in queue
        if (p := prior.get(str(i["record_id"]))) is not None
        and p["role"] in HUMAN_ROLES
        and p["evidence_digest"] == i["evidence_digest"]
        and not p["conflict"]
        and not p["review_required"]
    }


def _current_observations(store: LocalRecordStore, frozen: SnapshotRecords) -> list[dict[str, Any]]:
    # Queue evidence remains frozen; only judgment history for those selected runs
    # advances during labeling. This never reads mutable run or bundle evidence.
    runs = {r["run_id"] for r in frozen.runs}
    history = tuple(o for o in store.read_records()["observations"] if o["run_id"] in runs)
    return finding_observations(SnapshotRecords(frozen.snapshot, frozen.runs, history, history))


def handle_adjudicate(argv: list[str]) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.verb == "export" and args.out is None and not args.dry_run:
        parser.error("--out is required unless --dry-run")
    try:
        if args.verb in ("show", "label"):
            store, records, queue = _state(args.state_dir)
            observations = _current_observations(store, records)
            resolved = _resolved(queue, observations)
            open_items = [i for i in queue if i["record_id"] not in resolved]
            if args.verb == "show":
                for item in open_items:
                    print(f"{item['record_id'][:12]} {item['disposition']} {item['status']} {item['fingerprint']}")
                print(f"unresolved: {len(open_items)} / {len(queue)}")
                return 0
            targets = (
                [i for i in queue if i["record_id"] == args.record_id] if args.record_id else open_items[: args.batch]
            )
            if args.record_id and not targets:
                raise ValueError("unknown --record-id")
            now = datetime.now(timezone.utc).isoformat()
            for item in targets:
                append_observation(
                    store,
                    {
                        "record_id": item["record_id"],
                        "disposition": args.disposition,
                        "rationale": args.rationale,
                        "labeler": args.labeler,
                        "role": args.role,
                        "valid_at": args.valid_at or now,
                        "observed_at": now,
                        "rubric_version": item["rubric_version"],
                        "evidence_digest": item["evidence_digest"],
                        "evidence_digest_scheme": item["evidence_digest_scheme"],
                        "evidence": item["evidence"],
                    },
                    run_id=item["session_id"],
                    item_uid=item["item_uid"],
                )
            print_success(create_console(), f"Labeled {len(targets)} item(s)")
            return 0
        store = LocalRecordStore(args.store)
        if args.verb in {"build", "preview", "export"}:
            validate_output_path(store.root, args.state_dir)
        if args.verb == "export" and args.out is not None:
            validate_output_path(store.root, args.out)
        records = store.read_snapshot(args.snapshot_id)
        if args.verb == "build":
            queue = _queue(records)
            _write(
                args.state_dir / "reference.json",
                {
                    "schema_version": "daydream.annotation-queue.v2",
                    "store": str(store.root),
                    "snapshot_id": args.snapshot_id,
                },
            )
            _write(args.state_dir / "queue.json", queue)
            print_success(create_console(), f"Adjudication queue: {len(queue)} item(s)")
        elif args.verb == "materialize":
            print(canonical_json(run_materialize(args.store, args.snapshot_id, args.out_dir, dry_run=args.dry_run)))
        elif args.verb == "preview":
            print(canonical_json(run_preview(args.store, args.snapshot_id, args.state_dir / "preview-ledger.json")))
        elif args.verb == "harvest-snapshot":
            print(canonical_json(run_canonical_harvest(args.store, args.snapshot_id, args.materialize_dir)))
        elif args.verb == "export":
            ledger = args.state_dir / "preview-ledger.json"
            run_preview(args.store, args.snapshot_id, ledger)
            rows = build_export_entries(args.store, args.snapshot_id, ledger)
            validate_export_rows(rows)
            if not args.dry_run:
                write_export_rows(rows, args.out)
            print_success(create_console(), f"Validated {len(rows)} export row(s)")
        elif args.verb == "report":
            items = build_queue(sessions_from_snapshot(records, overlay_judgments=False), include_decisive=True)
            enriched = enrich_report_items(items, finding_observations(records), as_of=records.snapshot["valid_before"])
            if args.conflicts:
                for item in sorted(enriched, key=lambda i: str(i["record_id"])):
                    if has_rater_conflict(item["observations"]):
                        print(item["record_id"])
            else:
                report = build_report(enriched)
                report["strata"] = [
                    {"stack": key[0], "profile": key[1], "count": count} for key, count in report["strata"].items()
                ]
                print(json.dumps(report, sort_keys=True, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print_error(create_console(), f"adjudicate {args.verb} failed", str(exc))
        return 1
