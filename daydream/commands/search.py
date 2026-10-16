"""Bounded, cwd-confined fixed-string search over Git's relevant file set.

No rg dependency: the same scanner is the fallback on every supported platform.
This is an investigation convenience, not a backend access-control boundary.
"""

from __future__ import annotations

import argparse
import json
import stat
import time
from pathlib import Path
from typing import Any

from daydream import git_ops

_MAX_FILES = 2000
_MAX_FILE_BYTES = 1024 * 1024
_MAX_TOTAL_BYTES = 16 * 1024 * 1024
_MAX_MATCHES = 100
_MAX_OUTPUT_BYTES = 16384
_SCAN_SECONDS = 5


def _relative_root(cwd: Path, value: str) -> str:
    path = cwd / value
    resolved = path.resolve()
    if not resolved.is_relative_to(cwd) or ".git" in path.parts or ".git" in resolved.relative_to(cwd).parts:
        raise ValueError(f"search path is outside the working tree or names Git metadata: {value!r}")
    if not path.exists():
        raise ValueError(f"search path does not exist: {value!r}")
    return str(resolved.relative_to(cwd))


def search_repository(cwd: Path, pattern: str, roots: list[str], *, expand: list[str], reason: str) -> dict[str, Any]:
    """Return matches, visited scope and explicit incompleteness, never a silent clean result."""
    cwd = cwd.resolve()
    if not pattern or "\n" in pattern or "\r" in pattern or "\x00" in pattern:
        raise ValueError("pattern must be a nonempty single-line fixed string")
    if expand and not reason.strip():
        raise ValueError("--expand requires a concrete --reason")
    roots = [_relative_root(cwd, value) for value in roots]
    expanded = [_relative_root(cwd, value) for value in expand]
    # Each query has a five-second timeout without retries. Ignored traversal
    # occurs only under explicitly named expansion paths.
    files = git_ops.search_files(cwd, roots)
    if expanded:
        files.extend(git_ops.search_files(cwd, expanded, include_ignored=True))
    files = list(dict.fromkeys(files))
    result: dict[str, Any] = {
        "roots": roots, "expansion": {"paths": expanded, "reason": reason if expanded else ""},
        "enumerated_files": len(files), "visited_files": [], "bytes_read": 0, "matches": [],
        "incomplete": [], "skipped_files": 0,
    }
    deadline = time.monotonic() + _SCAN_SECONDS
    output_bytes = 0
    needle = pattern.encode("utf-8")
    for name in files:
        if time.monotonic() >= deadline:
            result["incomplete"].append("scan time limit")
            break
        if len(result["visited_files"]) >= _MAX_FILES or result["bytes_read"] >= _MAX_TOTAL_BYTES:
            result["incomplete"].append("file/total byte limit")
            break
        path = cwd / name
        # Git can list symlinks, deleted tracked files, and submodule entries.
        # Do not follow them into sibling checkouts, metadata or special files.
        resolved = path.resolve()
        if (".git" in Path(name).parts or not resolved.is_relative_to(cwd)
                or ".git" in resolved.relative_to(cwd).parts or path.is_symlink()):
            result["skipped_files"] += 1
            continue
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                result["skipped_files"] += 1
                continue
            with path.open("rb") as source:
                data = source.read(min(_MAX_FILE_BYTES, _MAX_TOTAL_BYTES - result["bytes_read"]) + 1)
        except OSError:
            result["skipped_files"] += 1
            continue
        result["visited_files"].append(name)
        result["bytes_read"] += len(data)
        if len(data) > _MAX_FILE_BYTES or result["bytes_read"] > _MAX_TOTAL_BYTES:
            result["incomplete"].append("per-file/total byte limit")
            data = data[:_MAX_FILE_BYTES]
        if b"\x00" in data:
            continue
        for number, line in enumerate(data.splitlines(), 1):
            if needle not in line:
                continue
            match = {"path": name, "line": number, "text": line[:240].decode("utf-8", errors="replace")}
            size = len(json.dumps(match).encode("utf-8"))
            if len(result["matches"]) >= _MAX_MATCHES or output_bytes + size > _MAX_OUTPUT_BYTES:
                result["incomplete"].append("match/output limit")
                return result
            result["matches"].append(match)
            output_bytes += size
    result["incomplete"] = list(dict.fromkeys(result["incomplete"]))
    return result


def handle_search_command(argv: list[str]) -> int:
    """Print bounded JSON evidence; exit 2 on incomplete scope or invalid requests."""
    parser = argparse.ArgumentParser(prog="daydream search", description=__doc__)
    parser.add_argument("pattern", help="fixed string (not a regular expression)")
    parser.add_argument("roots", nargs="+", help="named relevant source roots/files, relative to cwd")
    parser.add_argument("--expand", action="append", default=[], help="named ignored dependency/generated path")
    parser.add_argument("--reason", default="", help="concrete candidate question requiring expansion")
    args = parser.parse_args(argv)
    try:
        result = search_repository(Path.cwd(), args.pattern, args.roots, expand=args.expand, reason=args.reason)
    except (ValueError, OSError, git_ops.GitError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    # Inventory/visited counts remain observable without dumping an unbounded manifest.
    result["visited_count"] = len(result.pop("visited_files"))
    print(json.dumps(result, ensure_ascii=True))
    return 2 if result["incomplete"] or result["skipped_files"] else 0
