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
import re
import sys
from pathlib import Path


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def records(path, selected_line=None):
    """Support pretty JSON documents and streaming JSONL; retain physical lines."""
    with path.open(encoding="utf-8") as stream:
        first = next(((n, s) for n, s in enumerate(stream, 1)
                      if n == selected_line or selected_line is None and s.strip()), None)
        if first is None:
            if selected_line is not None:
                raise ValueError(f"{path}:{selected_line}: selected physical line is missing")
            raise ValueError(f"{path}: empty input")
        n, line = first
        if not line.strip():
            raise ValueError(f"{path}:{n}: selected physical line is blank")
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
        if selected_line is not None:
            return
        for number, text in enumerate(stream, n + 1):
            if text.strip():
                try:
                    yield number, json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc.msg}") from exc


def unwrap(value):
    return value.get("value") if isinstance(value, dict) and value.get("status") == "available" else None


def load(path, run_id=None, line=None):
    if line is not None and (type(line) is not int or line < 1):
        raise ValueError("--line must be a positive physical line number")
    selected = []
    for number, record in records(path, line):
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        matches = run_id is None or record.get("run_id", record.get("session_id")) == run_id
        if line is not None and not matches:
            raise ValueError(f"{path}:{number}: --run-id does not match selected physical line")
        if matches:
            selected.append((number, record))
    if len(selected) != 1:
        raise ValueError(f"{path}: found {len(selected)} matching records; select one with --run-id or --line")
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


def steps(trace, args, scope=None):
    """Yield original references, optionally within (document, start, end)."""
    for di, doc in enumerate(trace["docs"]):
        if (args.doc is not None and args.doc != di) or (scope is not None and scope[0] != di):
            continue
        # Only direct invocation rows belong to this document. Fork wrapper
        # invocations are repeated in the sibling document and are not counted.
        invocations = (doc.get("extra") or {}).get("subtrajectories", [])
        document_steps = doc.get("steps", [])
        start, end = scope[1:] if scope is not None else (0, len(document_steps))
        for si in range(start, end):
            step = document_steps[si]
            phase = (step.get("extra") or {}).get("daydream_phase")
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
    extra = result.get("extra") or {}
    return (extra.get("is_error") is True or extra.get("exit_code") not in (None, 0)
            or extra.get("status") in ("failed", "error", "cancelled", "interrupted"))


def step_tools(step, ref):
    results = (step.get("observation") or {}).get("results", [])
    for ci, call in enumerate(step.get("tool_calls") or []):
        call_id = call.get("tool_call_id")
        matched = [(ri, r) for ri, r in enumerate(results)
                   if call_id is not None and r.get("source_call_id") == call_id]
        yield {**ref, "pointer": ref["pointer"] + f"/tool_calls/{ci}", "tool_call_id": call_id,
               "function_name": call.get("function_name"), "arguments": call.get("arguments"),
               "failed": any(failed(r) for _, r in matched),
               "results": [{"pointer": ref["pointer"] + f"/observation/results/{ri}",
                            "content": r.get("content"), "extra": r.get("extra") or {}} for ri, r in matched]}


def tools(trace, args):
    for doc, step, ref in steps(trace, args):
        yield from step_tools(step, ref)


def bounded_rows(rows, limit, anchor_pointer=None):
    """Keep the anchor within the limit, retaining source order for other rows."""
    indices = list(range(min(limit, len(rows))))
    if anchor_pointer is not None:
        anchor = next((i for i, row in enumerate(rows)
                       if any(anchor_pointer == pointer or anchor_pointer.startswith(pointer + "/")
                              for pointer in [row["pointer"], *(r["pointer"] for r in row.get("results", []))])), None)
        if anchor is not None and anchor not in indices:
            indices[-1] = anchor
    return [rows[i] for i in sorted(indices)]


def tool_view(row, args, anchor_pointer=None):
    return {**row, "arguments": preview(row["arguments"], args.chars),
            "result_count": len(row["results"]),
            "results_remaining": max(0, len(row["results"]) - args.evidence_limit),
            "results": [{**r, "content_chars": len(preview(r["content"], sys.maxsize)),
                         "content": preview(r["content"], args.chars)}
                        for r in bounded_rows(row["results"], args.evidence_limit, anchor_pointer)]}


def step_unmatched_results(step, ref):
    ids = {c.get("tool_call_id") for c in step.get("tool_calls") or [] if c.get("tool_call_id") is not None}
    for ri, result in enumerate((step.get("observation") or {}).get("results", [])):
        if result.get("source_call_id") not in ids and not result.get("subagent_trajectory_ref"):
            yield {**ref, "pointer": ref["pointer"] + f"/observation/results/{ri}",
                   "source_call_id": result.get("source_call_id"), "failed": failed(result),
                   "extra": result.get("extra") or {}, "content": result.get("content")}


def unmatched_results(trace, args):
    for doc, step, ref in steps(trace, args):
        for row in step_unmatched_results(step, ref):
            yield {**row, "content": preview(row["content"], args.chars)}


def summary(trace, args):
    record = trace["record"]
    docs = trace["docs"]
    root = next((d for d in docs if d.get("trajectory_id") == trace["root_id"]), None)
    calls = list(tools(trace, args))
    rows = []
    for di, doc in enumerate(docs):
        own_calls = [r for r in calls if r["doc"] == di]
        rows.append({"doc": di, "pointer": doc_pointer(trace, di), "trajectory_id": doc.get("trajectory_id"),
                     "backend": (doc.get("extra") or {}).get("backend"), "agent": doc.get("agent"),
                     "steps": len(doc.get("steps", [])), "tool_calls": len(own_calls),
                     "failed_calls": sum(r["failed"] for r in own_calls),
                     "result_chars": sum(len(preview(r["content"], sys.maxsize))
                                         for c in own_calls for r in c["results"]),
                     "metrics": doc.get("final_metrics")})
    diagnostics = [{**ref, "pointer": ref["pointer"] + "/extra/" + key,
                    "excerpt": preview(value, args.chars)}
                   for doc, step, ref in steps(trace, args)
                   for key, value in (step.get("extra") or {}).items()
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
        extra = doc.get("extra") or {}
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
        for key in ("metadata", "backend_lifecycle", "generation_lifecycle"):
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
            folded_pos = value.casefold().find(needle)
            if folded_pos >= 0:
                # Case folding can expand characters; report offsets in the original text.
                pos = 0
                for char in value:
                    width = len(char.casefold())
                    if folded_pos < width:
                        break
                    folded_pos -= width
                    pos += 1
                start = max(0, pos - args.chars // 3)
                hits.append({"pointer": pointer, "match_offset": pos, "excerpt_offset": start,
                             "total_chars": len(value), "excerpt": value[start:start + args.chars]})
    return {**base(trace), "query": args.query, **page(hits, args)}


def resolve_pointer(value, pointer):
    if pointer:
        if not pointer.startswith("/"):
            raise ValueError("--pointer must be an RFC 6901 path starting with /, or empty for root")
        try:
            for part in pointer[1:].split("/"):
                if re.search(r"~(?![01])", part):
                    raise ValueError("invalid RFC 6901 escape")
                key = part.replace("~1", "/").replace("~0", "~")
                if isinstance(value, list):
                    if not re.fullmatch(r"0|[1-9][0-9]*", key):
                        raise ValueError("invalid RFC 6901 array index")
                    value = value[int(key)]
                else:
                    value = value[key]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            message = f"pointer {pointer!r} does not resolve; use a pointer from search/tools/timeline"
            raise ValueError(message) from exc
    return value


def show(trace, args):
    value = resolve_pointer(trace["record"], args.pointer)
    text = value if isinstance(value, str) else serialized(value)
    return {**base(trace), "pointer": args.pointer, "total_chars": len(text), "offset": args.offset,
            "text": text[args.offset:args.offset + args.chars],
            "remaining": max(0, len(text) - args.offset - args.chars)}


def window(trace, args):
    """Keep recorded context around one exact step without guessing causality."""
    resolve_pointer(trace["record"], args.pointer)
    location = None
    for di, doc in enumerate(trace["docs"]):
        prefix = doc_pointer(trace, di) + "/steps/"
        if args.pointer.startswith(prefix):
            location = di, doc, int(args.pointer[len(prefix):].split("/", 1)[0])
            break
    if location is None:
        raise ValueError("window requires a pointer inside an agent step")
    di, doc, index = location
    document_step_count = len(doc["steps"])
    start, end = max(0, index - args.before), min(document_step_count, index + args.after + 1)
    rows = []
    for _, step, ref in steps(trace, args, scope=(di, start, end)):
        tool_rows = list(step_tools(step, ref))
        result_rows = list(step_unmatched_results(step, ref))
        bounded_tools = []
        for row in bounded_rows(tool_rows, args.evidence_limit, args.pointer):
            bounded = tool_view(row, args, args.pointer)
            for result in bounded["results"]:
                result["extra"] = preview(result["extra"], args.chars)
            bounded_tools.append(bounded)
        rows.append({**ref, "source": step.get("source"),
                     "message": preview(step["message"], args.chars) if step.get("message") is not None else None,
                     "reasoning_content": preview(step["reasoning_content"], args.chars)
                     if step.get("reasoning_content") is not None else None,
                     "tools": {"total": len(tool_rows), "items": bounded_tools,
                               "remaining": max(0, len(tool_rows) - args.evidence_limit)},
                     "unmatched_results": {"total": len(result_rows),
                                           "items": [{**row, "content": preview(row["content"], args.chars),
                                                      "extra": preview(row["extra"], args.chars)} for row in
                                                     bounded_rows(result_rows, args.evidence_limit, args.pointer)],
                                           "remaining": max(0, len(result_rows) - args.evidence_limit)}})
    return {**base(trace), "anchor_pointer": args.pointer,
            "anchor_step_pointer": doc_pointer(trace, di) + f"/steps/{index}",
            "document_step_count": document_step_count, "start_index": start, "end_index": end,
            "interpretation": "Recorded neighbors in one document; invocation and phase changes are explicit. "
                              "Inspect terminal evidence separately; proximity does not establish causality.",
            **page(rows, args)}


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
    for command in ("list", "summary", "timeline", "tools", "mine", "search", "show", "window"):
        sub = commands.add_parser(command)
        sub.add_argument("input", help="Capture JSONL or standalone ATIF JSON; directory for list")
        sub.add_argument("--limit", type=int, default=12, help="Maximum rows per table")
        sub.add_argument("--evidence-limit", type=int, default=3, help="Maximum examples per group/results per call")
        sub.add_argument("--offset", type=int, default=0, help="Row offset; character offset for show")
        sub.add_argument("--chars", type=int, default=400, help="Maximum characters per excerpt")
        sub.add_argument("--run-id", help="Select a record in a multi-record JSONL file")
        if command != "list":
            sub.add_argument("--line", type=int, help="Select an exact physical record line; cohort row line")
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
        if command in ("show", "window"):
            sub.add_argument("--pointer", required=True, help="JSON pointer from another command")
        if command == "window":
            sub.add_argument("--before", type=int, default=2, help="Preceding steps in the same document")
            sub.add_argument("--after", type=int, default=2, help="Following steps in the same document")
    args = parser.parse_args()
    if min(args.limit, args.chars, args.evidence_limit) < 1 or args.offset < 0:
        parser.error("--limit/--chars/--evidence-limit must be positive and --offset nonnegative")
    if args.command == "window" and min(args.before, args.after) < 0:
        parser.error("--before/--after must be nonnegative")
    try:
        if args.command == "list":
            if not Path(args.input).expanduser().is_dir() or args.recent < 1:
                raise ValueError("list requires a directory and positive --recent")
            report = inventory(args)
        else:
            trace = load(Path(args.input).expanduser(), args.run_id, args.line)
            if args.doc is not None and not 0 <= args.doc < len(trace["docs"]):
                raise ValueError("--doc is outside the embedded document list")
            if args.command == "tools":
                report = {**base(trace), **page([tool_view(r, args) for r in tools(trace, args)], args),
                          "unmatched_results": page(list(unmatched_results(trace, args)), args)}
            else:
                report = {"summary": summary, "timeline": timeline, "mine": mine,
                          "search": search, "show": show, "window": window}[args.command](trace, args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        parser.exit(2, f"trace: {exc}\n")


if __name__ == "__main__":
    main()
