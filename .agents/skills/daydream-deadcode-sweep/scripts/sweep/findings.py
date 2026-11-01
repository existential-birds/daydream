#!/usr/bin/env python3
"""Persist sweep candidates and select complete, non-overlapping changes globally."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def rows(paths):
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.lstrip().startswith("{"):
                continue  # legacy completion sentinels
            try:
                value = json.loads(line)
            except ValueError:
                print(f"invalid JSON in {path}", file=sys.stderr)
                continue
            if isinstance(value, dict):
                yield value


def affected(row):
    return sorted(set([row["path"], *row.get("affected_paths", [])]))


def valid(root, row):
    if not all(isinstance(row.get(key), str) and row[key] for key in ("kind", "path", "reason", "evidence")):
        return False
    if type(row.get("est_net_lines")) is not int or row["est_net_lines"] >= 0:
        return False
    if not row.get("symbol", row.get("pair")):
        return False
    if any(not isinstance(row.get(key, []), list) for key in ("affected_paths", "evidence_paths")):
        return False
    return all(isinstance(p, str) and p and not Path(p).is_absolute()
               and Path(p).as_posix() == p and ".." not in Path(p).parts
               and (root / p).resolve().is_relative_to(root.resolve())
               for p in [row["path"], *row.get("affected_paths", []), *row.get("evidence_paths", [])])


def fingerprints(root, row):
    result = {}
    for name in sorted(set(affected(row) + row.get("evidence_paths", []))):
        path = root / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return result


def save(memory, candidates):
    memory.parent.mkdir(parents=True, exist_ok=True)
    temporary = memory.with_suffix(".tmp")
    temporary.write_text(json.dumps({"candidates": list(candidates.values())}, indent=2) + "\n")
    temporary.replace(memory)


def load(root, scan, memory):
    # A damaged durable queue is an error, not permission to discard prior work.
    previous = json.loads(memory.read_text())["candidates"] if memory.exists() else []
    candidates = {}
    for row in [*previous, *rows(sorted(scan.glob("findings-*.jsonl")))]:
        if not valid(root, row):
            print(f"invalid candidate: {row.get('path', '<missing path>')}", file=sys.stderr)
            continue
        identity = [row["kind"], row["path"], row.get("symbol", row.get("pair", ""))]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        prior = candidates.get(key, row)
        # Repeated scout emission must not reset an unchanged applied/rejected decision.
        item = dict(row, id=key, status=prior.get("status", "queued"))
        before = prior.get("fingerprints")
        now = fingerprints(root, item)
        stale = before is not None and before != now
        item["needs_revalidation"] = stale or prior.get("needs_revalidation", False)
        if stale:
            item["status"] = "queued"
        item["fingerprints"] = now
        candidates[key] = item
    return candidates


def plan(root, scan, apply, memory, max_chunks, findings_max):
    candidates = load(root, scan, memory)
    apply.mkdir(parents=True, exist_ok=True)
    eligible = [row for row in candidates.values() if row["status"] not in ("applied", "rejected")
                and row.get("confidence") == "high" and row.get("readiness", "ready") == "ready"]
    eligible.sort(key=lambda row: (row["est_net_lines"], row["id"]))
    selected, owned = [], set()
    for row in eligible:
        paths = set(affected(row))
        if paths & owned:
            continue
        selected.append(row)
        owned.update(paths)
        if len(selected) >= max_chunks * findings_max:
            break
    # Remove only this helper's old plans, never reports or outcomes.
    for old in apply.glob("chunk-*.jsonl"):
        old.unlink()
    labels = []
    for start in range(0, len(selected), findings_max):
        label = f"chunk-{len(labels) + 1}"
        labels.append(label)
        (apply / f"{label}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected[start:start + findings_max]))
    (apply / "chunklist.txt").write_text("".join(label + "\n" for label in labels))
    save(memory, candidates)
    print(f"candidates={len(candidates)} selected={len(selected)} deferred={len(candidates) - len(selected)}")


def finish(root, scan, apply, memory):
    candidates = load(root, scan, memory)
    for outcome in rows(sorted(apply.glob("outcomes-*.jsonl"))):
        key = outcome.get("id")
        if key not in candidates or outcome.get("status") not in ("applied", "rejected", "deferred"):
            continue
        row = candidates[key]
        row.update(status=outcome["status"], outcome_reason=outcome.get("reason", ""),
                   fingerprints=fingerprints(root, row), needs_revalidation=False)
    save(memory, candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "finish"))
    for name in ("root", "scan", "apply", "memory"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--max-chunks", type=int, default=2)
    parser.add_argument("--findings-max", type=int, default=12)
    args = parser.parse_args()
    if args.max_chunks < 1 or args.findings_max < 1:
        parser.error("batch limits must be positive")
    values = (args.root, args.scan, args.apply, args.memory)
    if args.action == "plan":
        plan(*values, args.max_chunks, args.findings_max)
    else:
        finish(*values)


if __name__ == "__main__":
    main()
