#!/usr/bin/env python3
"""Read-only, standard-library inspection of Daydream JSONL and ATIF JSON.

All pointers are RFC 6901 paths within the selected physical JSONL record.
No imports from Daydream, command execution, network access, or artifact writes.
"""

import argparse
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def records(path):
    """Support pretty JSON documents and streaming JSONL; retain physical lines."""
    with path.open(encoding="utf-8") as stream:
        first = next(((n, s) for n, s in enumerate(stream, 1) if s.strip()), None)
        if first is None:
            raise ValueError(f"{path}: empty input")
        n, line = first
        if line.strip() == "{":
            # Standalone ATIF is commonly pretty-printed; JSONL remains streamed.
            try:
                yield n, json.loads(line + stream.read())
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n + exc.lineno - 1}: {exc.msg}") from exc
            return
        for number, text in [(n, line)]:
            try:
                yield number, json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc.msg}") from exc
        for number, text in enumerate(stream, n + 1):
            if text.strip():
                try:
                    yield number, json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc.msg}") from exc


def unwrap(value):
    return value.get("value") if isinstance(value, dict) and value.get("status") == "available" else None


def load(path, run_id=None):
    selected = []
    for line, record in records(path):
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line}: expected an object")
        if run_id is None or record.get("run_id", record.get("session_id")) == run_id:
            selected.append((line, record))
    if len(selected) != 1:
        raise ValueError(f"{path}: found {len(selected)} matching records; select one with --run-id")
    line, record = selected[0]
    if record.get("schema_version") == "daydream.run.v1":
        bundle = unwrap(record.get("trajectories")) or {}
        docs = bundle.get("documents", [])
        prefix = "/trajectories/value/documents"
        root_id = bundle.get("root_trajectory_id")
    elif isinstance(record.get("steps"), list):
        docs, prefix, root_id = [record], None, record.get("trajectory_id")
    else:
        raise ValueError(f"{path}:{line}: unsupported schema {record.get('schema_version')!r}")
    return {"source": str(path.resolve()), "line": line, "run_id": record.get("run_id", record.get("session_id")),
            "record": record, "docs": docs, "prefix": prefix, "root_id": root_id}


def base(trace):
    return {k: trace[k] for k in ("source", "line", "run_id")}


def doc_pointer(trace, index):
    return f"{trace['prefix']}/{index}" if trace["prefix"] is not None else ""


def page(items, args):
    return {"total": len(items), "offset": args.offset, "items": items[args.offset:args.offset + args.limit],
            "remaining": max(0, len(items) - args.offset - args.limit)}


def preview(value, chars):
    text = value if isinstance(value, str) else serialized(value)
    return text[:chars]


def walk(value, pointer=""):
    yield pointer, value
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from walk(child, pointer + "/" + escaped)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, pointer + f"/{index}")


def duration(start, end):
    try:
        return round((datetime.datetime.fromisoformat(end.replace("Z", "+00:00")) -
                      datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds(), 6)
    except (AttributeError, TypeError, ValueError):
        return None


def steps(trace, args):
    for di, doc in enumerate(trace["docs"]):
        if args.doc is not None and args.doc != di:
            continue
        # Only direct invocation rows belong to this document. Fork wrapper
        # invocations are repeated in the sibling document and are not counted.
        invocations = doc.get("extra", {}).get("subtrajectories", [])
        for si, step in enumerate(doc.get("steps", [])):
            phase = step.get("extra", {}).get("daydream_phase")
            if args.phase and phase != args.phase:
                continue
            matches = [r.get("invocation_id") for r in invocations
                       if r.get("invocation_id") and step.get("step_id") in r.get("step_ids", [])]
            yield doc, step, {"doc": di, "trajectory_id": doc.get("trajectory_id"),
                              "step_id": step.get("step_id"), "phase": phase,
                              "timestamp": step.get("timestamp"),
                              "invocation_id": matches[0] if len(matches) == 1 else None,
                              "pointer": doc_pointer(trace, di) + f"/steps/{si}"}


def failed(result):
    extra = result.get("extra", {})
    return (extra.get("is_error") is True or extra.get("exit_code") not in (None, 0)
            or extra.get("status") in ("failed", "error", "cancelled", "interrupted"))


def tools(trace, args):
    for doc, step, ref in steps(trace, args):
        results = step.get("observation", {}).get("results", [])
        for ci, call in enumerate(step.get("tool_calls", [])):
            call_id = call.get("tool_call_id")
            matched = [(ri, r) for ri, r in enumerate(results)
                       if call_id is not None and r.get("source_call_id") == call_id]
            yield {**ref, "pointer": ref["pointer"] + f"/tool_calls/{ci}", "tool_call_id": call_id,
                   "function_name": call.get("function_name"), "arguments": call.get("arguments"),
                   "failed": any(failed(r) for _, r in matched),
                   "results": [{"pointer": ref["pointer"] + f"/observation/results/{ri}",
                                "content": r.get("content"), "extra": r.get("extra", {})} for ri, r in matched]}


def tool_view(row, args):
    return {**row, "arguments": preview(row["arguments"], args.chars),
            "result_count": len(row["results"]),
            "results_remaining": max(0, len(row["results"]) - args.evidence_limit),
            "results": [{**r, "content_chars": len(preview(r["content"], sys.maxsize)),
                         "content": preview(r["content"], args.chars)}
                        for r in row["results"][:args.evidence_limit]]}


def unmatched_results(trace, args):
    for doc, step, ref in steps(trace, args):
        ids = {c.get("tool_call_id") for c in step.get("tool_calls", []) if c.get("tool_call_id") is not None}
        for ri, result in enumerate(step.get("observation", {}).get("results", [])):
            if (result.get("source_call_id") not in ids and "subagent_trajectory_ref" not in result):
                yield {**ref, "pointer": ref["pointer"] + f"/observation/results/{ri}",
                       "source_call_id": result.get("source_call_id"), "failed": failed(result),
                       "extra": result.get("extra", {}),
                       "content": preview(result.get("content"), args.chars)}


def summary(trace, args):
    record = trace["record"]
    docs = trace["docs"]
    root = next((d for d in docs if d.get("trajectory_id") == trace["root_id"]), None)
    calls = list(tools(trace, args))
    rows = []
    for di, doc in enumerate(docs):
        own_calls = [r for r in calls if r["doc"] == di]
        rows.append({"doc": di, "pointer": doc_pointer(trace, di), "trajectory_id": doc.get("trajectory_id"),
                     "backend": doc.get("extra", {}).get("backend"), "agent": doc.get("agent"),
                     "steps": len(doc.get("steps", [])), "tool_calls": len(own_calls),
                     "failed_calls": sum(r["failed"] for r in own_calls),
                     "result_chars": sum(len(preview(r["content"], sys.maxsize))
                                         for c in own_calls for r in c["results"]),
                     "metrics": doc.get("final_metrics")})
    diagnostics = [{**ref, "pointer": ref["pointer"] + "/extra/" + key,
                    "excerpt": preview(value, args.chars)}
                   for doc, step, ref in steps(trace, args)
                   for key, value in step.get("extra", {}).items()
                   if "diagnostic" in key or "budget_stop" in key]
    availability = {k: {"status": v.get("status"), "reason": v.get("reason")}
                    for k, v in record.items() if isinstance(v, dict) and "status" in v and "value" in v}
    return {**base(trace), "schema": record.get("schema_version"), "captured_at": record.get("captured_at"),
            "outcome": record.get("outcome"), "completeness": record.get("completeness"),
            "availability": availability, "root_trajectory_id": trace["root_id"],
            "root_metrics": root.get("final_metrics") if root else None,
            "accounting": "Root metrics are reported as recorded; never added to fork totals. Inspect metric scope.",
            "document_count": len(docs), "step_count": sum(len(d.get("steps", [])) for d in docs),
            "tool_call_count": len(calls), "documents": page(rows, args),
            "unmatched_results": page(list(unmatched_results(trace, args)), args),
            "diagnostics": page(diagnostics, args)}


def timeline(trace, args):
    rows = []
    for di, doc in enumerate(trace["docs"]):
        if args.doc is not None and args.doc != di:
            continue
        dp = doc_pointer(trace, di)
        extra = doc.get("extra", {})
        starts = {}
        for ei, event in enumerate(extra.get("phase_events", [])):
            key = (event.get("session_id"), event.get("scope_id"), event.get("phase"))
            if event.get("event") == "phase_start":
                starts[key] = (ei, event)
            elif event.get("event") == "phase_end":
                _, start = starts.pop(key, (None, {}))
                rows.append({"kind": "phase", "doc": di, "trajectory_id": doc.get("trajectory_id"),
                             "pointer": dp + f"/extra/phase_events/{ei}", **event,
                             "started_at": start.get("timestamp"), "ended_at": event.get("timestamp"),
                             "duration_s": duration(start.get("timestamp"), event.get("timestamp"))})
        for ei, event in starts.values():
            rows.append({"kind": "open_phase", "doc": di, "trajectory_id": doc.get("trajectory_id"),
                         "pointer": dp + f"/extra/phase_events/{ei}", **event, "duration_s": None})
        for ii, invocation in enumerate(extra.get("subtrajectories", [])):
            if invocation.get("invocation_id"):
                rows.append({"kind": "invocation", "doc": di, "trajectory_id": doc.get("trajectory_id"),
                             "pointer": dp + f"/extra/subtrajectories/{ii}", **invocation,
                             "duration_s": duration(invocation.get("started_at"), invocation.get("ended_at"))})
    rows = [r for r in rows if not args.phase or r.get("phase") == args.phase]
    rows.sort(key=lambda r: r.get("started_at", r.get("timestamp")) or "")
    # Preserve exact metadata/lifecycle via show, avoiding large nested tables.
    for row in rows:
        for key in ("metadata", "backend_lifecycle"):
            if key in row:
                row[key] = preview(row[key], args.chars)
    return {**base(trace), "timing": "Overlapping rows are not additive wall time; no per-tool timing inferred.",
            **page(rows, args)}


def mine(trace, args):
    calls = list(tools(trace, args))
    groups = collections.defaultdict(list)
    candidates = []
    for row in calls:
        groups[serialized([row["function_name"], row["arguments"]])].append(row)
        if row["failed"]:
            candidates.append({"kind": "tool_failure", "count": 1, "evidence": [tool_view(row, args)]})
    for key, rows in groups.items():
        if len(rows) > 1:
            candidates.append({"kind": "identical_tool_call", "count": len(rows),
                               "fingerprint": hashlib.sha256(key.encode()).hexdigest(),
                               "scope": "across_trajectories" if len({r["doc"] for r in rows}) > 1
                               else "within_trajectory", "function_name": rows[0]["function_name"],
                               "arguments": preview(rows[0]["arguments"], args.chars),
                               "evidence": [tool_view(r, args) for r in rows[:args.evidence_limit]],
                               "evidence_remaining": max(0, len(rows) - args.evidence_limit)})
    for row in unmatched_results(trace, args):
        if row["failed"]:
            candidates.append({"kind": "unmatched_result_failure", "count": 1, "evidence": [row]})
    large = []
    for doc, step, ref in steps(trace, args):
        for pointer, value in walk(step, ref["pointer"]):
            if isinstance(value, str) and len(value) >= args.large_chars:
                large.append({"kind": "large_text", "count": 1, "chars": len(value),
                              "evidence": [{**ref, "pointer": pointer, "excerpt": value[:args.chars]}]})
    large.sort(key=lambda r: -r["chars"])
    candidates.sort(key=lambda r: (-r["count"], r["kind"]))
    return {**base(trace), "interpretation": "Candidates, not proven waste or run failure. Inspect recovery and role.",
            **page(candidates, args), "large_text": page(large, args)}


def search(trace, args):
    hits = []
    needle = args.query.casefold()
    for pointer, value in walk(trace["record"]):
        if isinstance(value, str):
            pos = value.casefold().find(needle)
            if pos >= 0:
                start = max(0, pos - args.chars // 3)
                hits.append({"pointer": pointer, "match_offset": pos, "excerpt_offset": start,
                             "total_chars": len(value), "excerpt": value[start:start + args.chars]})
    return {**base(trace), "query": args.query, **page(hits, args)}


def show(trace, args):
    value = trace["record"]
    if args.pointer:
        if not args.pointer.startswith("/"):
            raise ValueError("--pointer must be an RFC 6901 path starting with /, or empty for root")
        try:
            for part in args.pointer[1:].split("/"):
                key = part.replace("~1", "/").replace("~0", "~")
                value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            message = f"pointer {args.pointer!r} does not resolve; use a pointer from search/tools/timeline"
            raise ValueError(message) from exc
    text = value if isinstance(value, str) else serialized(value)
    return {**base(trace), "pointer": args.pointer, "total_chars": len(text), "offset": args.offset,
            "text": text[args.offset:args.offset + args.chars],
            "remaining": max(0, len(text) - args.offset - args.chars)}


def inventory(args):
    directory = Path(args.input).expanduser()
    paths = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:args.recent]
    rows = []
    for path in paths:
        try:
            for line, record in records(path):
                rows.append({"source": str(path.resolve()), "line": line, "bytes": path.stat().st_size,
                             "run_id": record.get("run_id"), "captured_at": record.get("captured_at"),
                             "outcome": record.get("outcome"), "completeness": record.get("completeness")})
        except (ValueError, OSError, AttributeError) as exc:
            rows.append({"source": str(path), "error": str(exc)})
    return {"selection": "Newest files by mtime; captured_at is displayed, not inferred from filename.",
            "files_examined": len(paths), **page(rows, args)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "summary", "timeline", "tools", "mine", "search", "show"):
        sub = commands.add_parser(command)
        sub.add_argument("input", help="Capture JSONL or standalone ATIF JSON; directory for list")
        sub.add_argument("--limit", type=int, default=12, help="Maximum rows per table")
        sub.add_argument("--evidence-limit", type=int, default=3, help="Maximum examples per group/results per call")
        sub.add_argument("--offset", type=int, default=0, help="Row offset; character offset for show")
        sub.add_argument("--chars", type=int, default=400, help="Maximum characters per excerpt")
        sub.add_argument("--run-id", help="Select a record in a multi-record JSONL file")
        sub.set_defaults(doc=None, phase=None)
        if command in ("tools", "mine", "timeline"):
            sub.add_argument("--doc", type=int, help="Zero-based embedded document index")
            sub.add_argument("--phase", help="Exact recorded phase (arbiter may be deep)")
        if command == "list":
            sub.add_argument("--recent", type=int, default=20, help="Maximum newest files to read")
        if command == "mine":
            sub.add_argument("--large-chars", type=int, default=12000)
        if command == "search":
            sub.add_argument("--query", required=True, help="Case-insensitive literal string")
        if command == "show":
            sub.add_argument("--pointer", required=True, help="JSON pointer from another command")
    args = parser.parse_args()
    if min(args.limit, args.chars, args.evidence_limit) < 1 or args.offset < 0:
        parser.error("--limit/--chars/--evidence-limit must be positive and --offset nonnegative")
    try:
        if args.command == "list":
            if not Path(args.input).expanduser().is_dir() or args.recent < 1:
                raise ValueError("list requires a directory and positive --recent")
            report = inventory(args)
        else:
            trace = load(Path(args.input).expanduser(), args.run_id)
            if args.doc is not None and not 0 <= args.doc < len(trace["docs"]):
                raise ValueError("--doc is outside the embedded document list")
            if args.command == "tools":
                report = {**base(trace), **page([tool_view(r, args) for r in tools(trace, args)], args),
                          "unmatched_results": page(list(unmatched_results(trace, args)), args)}
            else:
                report = {"summary": summary, "timeline": timeline, "mine": mine,
                          "search": search, "show": show}[args.command](trace, args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        parser.exit(2, f"trace: {exc}\n")


if __name__ == "__main__":
    main()
