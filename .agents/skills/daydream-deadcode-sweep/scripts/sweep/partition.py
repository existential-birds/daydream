#!/usr/bin/env python3
"""Build a shared tracked-file inventory and bounded scout assignments (stdlib).

Usage: partition.py <repo-root> <num-slices> [out.json]
SWEEP_COVERAGE_FILE is optional JSONL: {path, sha256, status: complete|partial}.
SWEEP_FOCUS_MAX (6) / SWEEP_FOCUS_LOC (6000) bound initial reading, not coverage.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CODE_EXT = {".py", ".rs", ".js", ".ts", ".tsx", ".jsx", ".sh", ".ex", ".exs"}
BIG = 1000


def inventory(root):
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
    records = []
    for rel in sorted(set(paths) - {""}):
        path = root / rel
        kind = "test" if "tests" in path.relative_to(root).parts else (
            "code" if path.suffix in CODE_EXT else "doc" if path.suffix in {".md", ".rst"} else "resource"
        )
        rec = {"path": rel, "loc": 0, "kind": kind, "status": "text", "related_paths": []}
        records.append(rec)
        if path.is_symlink():
            rec["status"] = "symlink"
            continue
        try:
            data = path.read_bytes()
        except OSError:
            rec["status"] = "unreadable"
            continue
        rec["sha256"] = hashlib.sha256(data).hexdigest()
        try:
            if b"\0" in data:
                raise UnicodeError
            source = data.decode("utf-8")
        except UnicodeError:
            rec["status"] = "binary"
            continue
        rec["loc"] = len(source.splitlines())
        if path.suffix == ".py":
            try:
                tree = ast.parse(source)
            except SyntaxError:
                rec["parse_error"] = True
                continue
            definitions, imports = [], set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    definitions.append({"name": node.name, "line": node.lineno, "end_line": node.end_lineno})
                elif isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add("." * node.level + (node.module or ""))
            rec["definition_count"], rec["import_count"] = len(definitions), len(imports)
            rec["definitions"] = sorted(definitions, key=lambda d: d["line"] - d["end_line"])[:30]
            rec["imports"] = sorted(imports)[:30]
    return records


def coverage_history(path):
    history = {}
    if path and Path(path).is_file():
        for sequence, line in enumerate(Path(path).read_text().splitlines()):
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and isinstance(rec.get("path"), str):
                    history[rec["path"]] = {**rec, "sequence": sequence}
            except ValueError:
                continue  # Preserve useful checkpoints after a truncated final write.
    return history


def partition(root, k):
    records = inventory(root)
    history = coverage_history(os.environ.get("SWEEP_COVERAGE_FILE"))
    by_stem = defaultdict(list)
    for rec in records:
        stem = Path(rec["path"]).stem
        by_stem[stem.removeprefix("test_")].append(rec)
    units = []
    for peers in by_stem.values():
        # Same-name source/test files travel together; references outside this
        # heuristic association are still discoverable through repository search.
        related = len(peers) > 1 and any(rec["kind"] == "test" for rec in peers)
        if related:
            for rec in peers:
                rec["related_paths"] = [p["path"] for p in peers if p is not rec]
            units.append(peers)
        else:
            units.extend([rec] for rec in peers)

    slices = [{"name": f"slice-{i + 1}", "paths": [], "loc": 0, "big_files": [],
               "role": "subsystem simplification"} for i in range(k)]
    for unit in sorted(units, key=lambda peers: -sum(rec["loc"] for rec in peers)):
        target = min(slices, key=lambda sl: (sl["loc"], len(sl["paths"])))
        target["paths"].extend(rec["path"] for rec in unit)
        target["loc"] += sum(rec["loc"] for rec in unit)
    if k >= 4:
        slices[-2]["role"] = "cross-subsystem duplication and shared ownership"
        slices[-1]["role"] = "architecture, representations, and unnecessary layers"

    def priority(rec):
        old = history.get(rec["path"])
        if old is None:
            state = 1
        elif old.get("sha256") != rec.get("sha256"):
            state = 0
        else:
            state = 3 if old.get("status") == "complete" else 2
        return state, (old or {}).get("sequence", -1), -rec["loc"], rec["path"]

    by_path = {rec["path"]: rec for rec in records}
    focus_max = max(1, int(os.environ.get("SWEEP_FOCUS_MAX", "6")))
    focus_loc = max(1, int(os.environ.get("SWEEP_FOCUS_LOC", "6000")))
    for sl in slices:
        sl["paths"].sort()
        owned = [by_path[path] for path in sl["paths"]]
        sl["big_files"] = [{"path": rec["path"], "loc": rec["loc"]} for rec in owned
                           if rec["loc"] >= BIG and rec["kind"] in {"code", "test"}]
        sl["focus_paths"], used = [], 0
        for rec in sorted(owned, key=priority):
            if rec["status"] != "text" or not rec["loc"]:
                continue
            if sl["focus_paths"] and used + rec["loc"] > focus_loc:
                continue
            sl["focus_paths"].append(rec["path"])
            used += rec["loc"]
            if len(sl["focus_paths"]) >= focus_max:
                break
        sl["inventory_summary"] = {
            "kinds": dict(Counter(rec["kind"] for rec in owned)),
            "statuses": dict(Counter(rec["status"] for rec in owned)),
            "focus_loc": used, "focus_loc_budget": focus_loc,
            "completed_unchanged": sum(
                history.get(rec["path"], {}).get("status") == "complete"
                and history[rec["path"]].get("sha256") == rec.get("sha256") for rec in owned
            ),
        }
    return {"root": str(root), "slices": slices, "files": records,
            "big_files": sorted([rec for sl in slices for rec in sl["big_files"]], key=lambda r: -r["loc"]),
            "total_files": len(records), "total_loc": sum(rec["loc"] for rec in records)}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    payload = partition(Path(sys.argv[1]).resolve(), max(1, int(sys.argv[2])))
    text = json.dumps(payload, indent=1)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
