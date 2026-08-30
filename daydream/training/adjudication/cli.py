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

from daydream.training.adjudication.observations import (
    append_observation,
    load_observations,
)
from daydream.training.adjudication.queue import build_queue

__all__ = ["handle_adjudicate", "handle_build", "handle_label", "handle_show"]

_HUMAN_ROLES = frozenset({"rater", "adjudicator"})

_QUEUE_FILENAME = "queue.json"
_OBSERVATIONS_FILENAME = "observations.jsonl"
_SESSIONS_FILENAME = "sessions.jsonl"


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
    target.add_argument("--batch", type=int, default=None, metavar="N",
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
        if obs.get("role") in _HUMAN_ROLES:
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
        assert args.batch is not None and args.batch >= 1, "--batch must be a positive integer"
        targets = open_items[: args.batch]

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


_HANDLERS = {"build": handle_build, "show": handle_show, "label": handle_label}


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
