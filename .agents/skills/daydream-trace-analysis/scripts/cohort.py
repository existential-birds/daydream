#!/usr/bin/env python3
"""Freeze local captured runs and plan deterministic, whole-run analysis batches.

Standard library only. Sources are read-only JSONL; directory inputs inspect
only immediate *.jsonl files. Selection uses captured_at, never filesystem time.
"""

import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = "daydream.trace-cohort.v1"
SECTIONS = ("original_task", "final_state", "recommended_patch", "trajectories", "findings", "verification", "scoring")
MISSING = object()


def canonical(value):
    """Canonical record bytes shared by frozen evidence consumers."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def parse_json(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON number {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO 8601 string with timezone")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp requires a timezone")
        return parsed.astimezone(datetime.timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid timestamp {value!r}; supply ISO 8601 with timezone") from exc


def pointer_parts(pointer):
    if pointer == "":
        return []
    if not pointer.startswith("/") or re.search(r"~(?![01])", pointer):
        raise ValueError("--where requires an RFC 6901 JSON pointer")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def resolve(record, pointer):
    value = record
    for part in pointer_parts(pointer):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", part) and int(part) < len(value):
            value = value[int(part)]
        else:
            return MISSING
    return value


def context(record):
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    task_section = record.get("original_task")
    task = (task_section.get("value")
            if isinstance(task_section, dict) and task_section.get("status") == "available" else {})
    task = task if isinstance(task, dict) else {}
    result = {key: provenance.get(key) for key in (
        "producer", "resolved_phases", "effective_configuration", "profile", "latency_profile",
        "file_configuration", "repository_context", "flow")}
    result.update({key: task.get(key) for key in (
        "repository", "analyzed_revision", "diff_sha256", "source_diff_sha256", "input_scope", "dirty_tracked")})
    files = task.get("changed_files")
    result["changed_file_count"] = len(files) if isinstance(files, list) else None
    return result


def run_row(record, source, line):
    if not isinstance(record, dict) or record.get("schema_version") != "daydream.run.v1":
        raise ValueError("expected a daydream.run.v1 capture object")
    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        raise ValueError("capture requires a nonempty run_id")
    captured_at = record.get("captured_at")
    if captured_at is not None:
        timestamp(captured_at)
    availability = {}
    for name in SECTIONS:
        section = record.get(name)
        section = section if isinstance(section, dict) else {}
        availability[name] = {"status": section.get("status"), "reason": section.get("reason")}
    encoded = canonical(record)
    return {"run_id": record["run_id"], "source": str(source), "line": line,
            "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "captured_at": captured_at,
            "outcome": record.get("outcome"), "completeness": record.get("completeness"),
            "availability": availability, "context": context(record)}


def load_manifest(path):
    """Validate manifest identity/shape; partial selection with errors is unsafe."""
    manifest = parse_json(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} manifest")
    expected = digest({k: v for k, v in manifest.items() if k != "cohort_id"})
    if manifest.get("cohort_id") != expected:
        raise ValueError("manifest cohort_id does not match its contents")
    if not isinstance(manifest.get("errors"), list) or manifest["errors"]:
        raise ValueError("manifest has selection errors; resolve inputs and select again")
    if (not isinstance(manifest.get("selection"), dict) or not isinstance(manifest.get("exclusions"), list)
            or not isinstance(manifest.get("runs"), list)):
        raise ValueError("manifest requires selection, exclusions, and runs")
    seen = set()
    for row in manifest["runs"]:
        if (not isinstance(row, dict) or not isinstance(row.get("run_id"), str) or not row["run_id"]
                or row["run_id"] in seen):
            raise ValueError("manifest requires unique nonempty run IDs")
        if (not isinstance(row.get("source"), str) or not Path(row["source"]).is_absolute()
                or type(row.get("line")) is not int or row["line"] < 1
                or type(row.get("bytes")) is not int or row["bytes"] < 1
                or not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", row["sha256"])):
            raise ValueError("manifest run requires absolute source, positive line/bytes, and SHA256")
        seen.add(row["run_id"])
    return manifest


def iter_runs(rows):
    """Check and yield frozen records in physical order, scanning each source once."""
    sources = {}
    for row in rows:
        sources.setdefault(row["source"], {}).setdefault(row["line"], []).append(row)
    for path, pending in sources.items():
        source = Path(path)
        with source.open("rb") as stream:
            for line, text in enumerate(stream, 1):
                selected = pending.pop(line, None)
                if selected is None:
                    continue
                record = parse_json(text.decode("utf-8"))
                current = run_row(record, source, line)
                for row in selected:
                    if current != row:
                        raise ValueError(f"{source}:{line}: frozen record or metadata changed")
                    yield row, record
                del record, current
                if not pending:
                    break
        if pending:
            raise ValueError(f"{source}:{min(pending)}: frozen line is missing or changed")


def exclusion_reason(record, row, filters, since, until):
    for clause in filters:
        value = resolve(record, clause["pointer"])
        if value is MISSING:
            return "where_missing"
        if canonical(value) != canonical(clause["value"]):
            return "where"
    when = timestamp(row["captured_at"]) if row["captured_at"] is not None else None
    if since is not None or until is not None:
        if when is None:
            return "captured_at_unknown"
        if since is not None and when < since:
            return "before_since"
        if until is not None and when > until:
            return "after_until"
    return None


def selection_key(row):
    latest = datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)
    age = latest - timestamp(row["captured_at"]) if row["captured_at"] is not None else datetime.timedelta()
    return row["captured_at"] is None, age, row["run_id"], row["source"], row["line"]


def select(args):
    filters = []
    for clause in args.where:
        if "=" not in clause:
            raise ValueError("--where must be JSON_POINTER=JSON_VALUE")
        pointer, value = clause.split("=", 1)
        pointer_parts(pointer)
        filters.append({"pointer": pointer, "value": parse_json(value)})
    since = timestamp(args.since) if args.since is not None else None
    until = timestamp(args.until) if args.until is not None else None
    if since is not None and until is not None and since > until:
        raise ValueError("--since must not be after --until")
    manifest = {"schema": SCHEMA, "runs": [], "exclusions": [], "errors": [], "selection": {
        "inputs": sorted({str(Path(value).expanduser().resolve()) for value in args.inputs}),
        "where": filters, "since": args.since, "until": args.until, "limit": args.limit,
        "order": "captured_at descending; unknown last; run_id/source/line tie-break"}}
    paths = set()
    for value in manifest["selection"]["inputs"]:
        path = Path(value)
        if path.is_dir():
            files = list(path.glob("*.jsonl"))
            if not files:
                manifest["errors"].append({"source": value, "line": None, "error": "no immediate *.jsonl files"})
            paths.update(file.resolve() for file in files)
        else:
            paths.add(path)
    grouped = {}
    for path in sorted(paths):
        try:
            with path.open("rb") as stream:
                found = False
                for line, text in enumerate(stream, 1):
                    if not text.strip():
                        continue
                    found = True
                    try:
                        record = parse_json(text.decode("utf-8"))
                        row = run_row(record, path, line)
                        reason = exclusion_reason(record, row, filters, since, until)
                        # Keep metadata, never the entire corpus of embedded trajectories.
                        grouped.setdefault(row["run_id"], []).append((row, reason))
                    except (ValueError, UnicodeError, RecursionError) as exc:
                        manifest["errors"].append({"source": str(path), "line": line, "error": str(exc)})
                if not found:
                    manifest["errors"].append({"source": str(path), "line": None, "error": "empty input"})
        except OSError as exc:
            manifest["errors"].append({"source": str(path), "line": None, "error": str(exc)})
    for run_id, copies in sorted(grouped.items()):
        if len({row["sha256"] for row, _ in copies}) != 1:
            manifest["errors"].append({"run_id": run_id, "error": "conflicting_run_id"})
            manifest["exclusions"].extend({**row, "reason": "conflicting_run_id"} for row, _ in copies)
            continue
        row, reason = copies[0]
        manifest["exclusions"].extend({**duplicate, "reason": "duplicate"} for duplicate, _ in copies[1:])
        if reason:
            manifest["exclusions"].append({**row, "reason": reason})
        else:
            manifest["runs"].append(row)
    manifest["runs"].sort(key=selection_key)
    if args.limit is not None:
        manifest["exclusions"].extend({**row, "reason": "limit"} for row in manifest["runs"][args.limit:])
        manifest["runs"] = manifest["runs"][:args.limit]
    manifest["cohort_id"] = digest(manifest)
    return manifest


def batches(args):
    manifest = load_manifest(args.manifest)
    if not manifest["runs"]:
        raise ValueError("cannot batch an empty cohort")
    for _, record in iter_runs(manifest["runs"]):
        del record
    planned = [{"batch_id": f"batch-{n + 1}", "bytes": 0, "run_ids": []}
               for n in range(min(args.count, len(manifest["runs"])))]
    for row in sorted(manifest["runs"], key=lambda row: (-row["bytes"], row["run_id"])):
        batch = min(planned, key=lambda batch: batch["bytes"])
        batch["bytes"] += row["bytes"]
        batch["run_ids"].append(row["run_id"])
    plan = {"schema": "daydream.trace-batches.v1", "cohort_id": manifest["cohort_id"], "question": args.question,
            "evidence_chars": args.chars, "max_observations": args.max_observations, "batches": planned}
    plan["plan_id"] = digest(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("select", help="Freeze a cohort; recoverable input errors retain valid runs and exit 2")
    sub.add_argument("inputs", nargs="+", help="Capture JSONL files or flat runs directories")
    sub.add_argument("--where", action="append", default=[], help="Exact JSON_POINTER=JSON_VALUE, repeatable")
    sub.add_argument("--since", help="Inclusive captured_at lower bound with timezone")
    sub.add_argument("--until", help="Inclusive captured_at upper bound with timezone")
    sub.add_argument("--limit", type=int, help="Maximum latest captures after filtering; default all")
    sub = commands.add_parser("batches", help="Balance canonical record bytes without splitting run siblings")
    sub.add_argument("manifest")
    sub.add_argument("--count", type=int, required=True)
    sub.add_argument("--question", required=True)
    sub.add_argument("--chars", type=int, default=350)
    sub.add_argument("--max-observations", type=int, default=8)
    args = parser.parse_args()
    try:
        if args.command == "select":
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be positive")
            result = select(args)
        else:
            if min(args.count, args.chars, args.max_observations) < 1 or not args.question.strip():
                raise ValueError("--count/--chars/--max-observations must be positive and --question nonempty")
            result = batches(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if result.get("errors") else 0
    except (OSError, ValueError, UnicodeError, TypeError, KeyError, RecursionError) as exc:
        parser.exit(2, f"cohort: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
