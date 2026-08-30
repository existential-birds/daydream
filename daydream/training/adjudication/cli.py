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
  ``--conflicts``, list disagreeing-rater findings oldest-first.

Every handler returns an int exit code (never calls ``sys.exit`` itself);
argparse converts malformed invocations into ``SystemExit(2)``. Unknown
record ids and missing state files fail closed with exit 1, naming the
offending identifier. No handler mutates anything outside ``--state-dir``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daydream.archive.hydrate import HydrationError
from daydream.training.adjudication.export import validate_export_rows, write_export_rows
from daydream.training.adjudication.harvest import build_export_entries
from daydream.training.adjudication.observations import (
    append_observation,
    load_observations,
)
from daydream.training.adjudication.precedence import DECISIVE_DISPOSITIONS, effective_adjudication, has_rater_conflict
from daydream.training.adjudication.preview import run_preview
from daydream.training.adjudication.queue import build_queue
from daydream.training.adjudication.report import build_report

__all__ = [
    "handle_adjudicate",
    "handle_build",
    "handle_export",
    "handle_label",
    "handle_report",
    "handle_show",
]

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
) -> list[dict[str, Any]]:
    """Queue items enriched with their observation lists + effective dispositions."""
    items = build_queue(_load_sessions_for_index(index_root))
    observations = load_observations(state_dir / _OBSERVATIONS_FILENAME)
    queue_ids = {str(item["record_id"]) for item in items}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        record_id = str(obs["record_id"])
        if record_id not in queue_ids:
            raise ValueError(
                f"observation references record_id {record_id!r} which is not in the "
                f"adjudication queue over the hydrated index"
            )
        grouped.setdefault(record_id, []).append(obs)

    enriched: list[dict[str, Any]] = []
    for item in items:
        enriched_item = dict(item)
        record_obs = grouped.get(str(item["record_id"]), [])
        enriched_item["observations"] = record_obs
        if record_obs:
            resolved = effective_adjudication(record_obs)
            if (
                resolved["role"] in ("rater", "adjudicator")
                and resolved["evidence_digest"] == str(item["evidence_digest"])
                and resolved["disposition"] in DECISIVE_DISPOSITIONS
            ):
                enriched_item["disposition"] = resolved["disposition"]
        enriched.append(enriched_item)
    return enriched


def handle_report(argv: list[str]) -> int:
    """Handle ``corpus adjudicate report --index-root <path> --state-dir <path>``."""
    from daydream.ui import create_console, print_error

    args = _build_adjudicate_parser().parse_args(["report", *argv])
    try:
        enriched = _report_items(args.index_root, args.state_dir)
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
    print(f"outcome-bearing coverage: adjudicated {coverage['adjudicated']} / {coverage['total']}")
    print(f"silver/task-only: {report['silver_task_only_count']}")
    print(f"class balance: accepted={balance['accepted']} rejected={balance['rejected']}")
    print(f"unresolved: {report['unresolved']}")
    print(f"inter-rater: {inter_rater['items']} item(s), {inter_rater['agreeing']} agreeing")
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


_HANDLERS = {
    "build": handle_build,
    "show": handle_show,
    "label": handle_label,
    "export": handle_export,
    "report": handle_report,
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
