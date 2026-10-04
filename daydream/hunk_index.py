"""Unified-diff parsing and persisted hunk indexes shared by review consumers.

Posting, quote scrubbing, coverage, verification selection, and diagram grounding
use the same changed-range and added-line projections."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Header regexes for the shared unified-diff parser.
# Unified-diff hunk header: @@ -<old_start>[,<old_count>] +<new_start>[,<new_count>] @@
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

HUNK_INDEX_FILENAME = "hunk-index.json"


def _unquote_git_path(quoted: str) -> str:
    """Decode Git’s quoted C-style path escapes through raw UTF-8 bytes."""
    if not (quoted.startswith('"') and quoted.endswith('"')):
        return quoted
    inner = quoted[1:-1]
    out = bytearray()
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\" and i + 1 < len(inner):
            nxt = inner[i + 1]
            if nxt == "\\":
                out.append(ord("\\"))
                i += 2
            elif nxt == '"':
                out.append(ord('"'))
                i += 2
            elif nxt in "01234567":
                val = 0
                j = i + 1
                while j < len(inner) and j < i + 4 and inner[j] in "01234567":
                    val = val * 8 + int(inner[j])
                    j += 1
                out.append(val)
                i = j
            else:
                out.append(ord("\\"))
                i += 1
        else:
            out.extend(ch.encode("utf-8"))
            i += 1
    return out.decode("utf-8")


def _header_path(raw: str) -> str | None:
    """Resolve a +++ header, handling Git quoting, optional b/, and trailing tabs.

    Return None for /dev/null deletions."""
    if not raw.startswith("+++ "):
        return None
    tail = raw[4:].rstrip("\t")
    if tail.startswith('"') and tail.endswith('"'):
        tail = _unquote_git_path(tail)
    if tail.startswith("b/"):
        return tail[2:]
    if tail == "/dev/null":
        return None
    return tail


def parse_hunks(diff_text: str) -> dict[str, dict[str, Any]]:
    """Return sorted path-keyed hunks, added/removed totals, and in-memory added lines.

    Each hunk has old/new inclusive bounds and added/removed counts. added_lines
    contains new-side + line numbers; added_text maps each to (hunk_index, text).
    These two projections are not persisted. A +++ header requires a preceding ---;
    malformed input without parseable hunks returns {}."""
    result: dict[str, dict[str, Any]] = {}
    current_meta: dict[str, Any] | None = None
    current_hunk: dict[str, Any] | None = None
    new_line = 0
    prev_old_header = False
    for raw in diff_text.splitlines():
        if raw.startswith(("--- ", '--- "')):
            prev_old_header = True
            continue
        if raw.startswith("+++ ") and prev_old_header:
            prev_old_header = False
            path = _header_path(raw)
            if path is None:
                current_meta = None
                current_hunk = None
                new_line = 0
                continue
            current_meta = result.setdefault(
                path,
                {
                    "hunks": [],
                    "added_total": 0,
                    "removed_total": 0,
                    "added_lines": set(),
                    "added_text": {},
                },
            )
            current_hunk = None
            new_line = 0
            continue
        prev_old_header = False
        if current_meta is None:
            continue
        header = _HUNK_HEADER.match(raw)
        if raw.startswith("@@") and header:
            old_start = int(header.group(1))
            old_count = int(header.group(2)) if header.group(2) else 1
            new_start = int(header.group(3))
            new_count = int(header.group(4)) if header.group(4) else 1
            new_line = new_start
            if new_count == 0:
                # Empty new-side range (pure deletion): pr_review skips it.
                current_hunk = None
                continue
            current_hunk = {
                "old_start": old_start,
                "old_end": old_start + old_count - 1,
                "new_start": new_start,
                "new_end": new_start + new_count - 1,
                "added": 0,
                "removed": 0,
            }
            current_meta["hunks"].append(current_hunk)
        elif raw.startswith("+"):
            current_meta["added_total"] += 1
            current_meta["added_lines"].add(new_line)
            if current_hunk is not None:
                current_hunk["added"] += 1
                current_meta["added_text"][new_line] = (
                    len(current_meta["hunks"]) - 1,
                    raw[1:],
                )
            new_line += 1
        elif raw.startswith("-"):
            current_meta["removed_total"] += 1
            if current_hunk is not None:
                current_hunk["removed"] += 1
        elif raw.startswith(" "):
            new_line += 1
    return result


def range_distance(line: int, start: int, end: int) -> int:
    """Return zero inside an inclusive hunk range, else distance to its nearest boundary."""
    if start <= line <= end:
        return 0
    if line < start:
        return start - line
    return line - end


def head_side_ranges(parsed: dict[str, dict[str, Any]]) -> list[tuple[int, int]]:
    """Flatten new-side inclusive hunk ranges across files in diff order."""
    return [
        pair
        for per_file in head_side_ranges_by_file(parsed).values()
        for pair in per_file
    ]


def head_side_ranges_by_file(parsed: dict[str, dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    """Group new-side inclusive ranges by file, preserving within-file hunk order.

    Retain empty file entries, including pure deletions. Accept both parsed and
    persisted indexes; added-line projections are not needed."""
    return {
        path: [(hunk["new_start"], hunk["new_end"]) for hunk in info["hunks"]]
        for path, info in parsed.items()
    }


def added_line_numbers(parsed: dict[str, dict[str, Any]]) -> dict[str, set[int]]:
    """Map every changed file to the union of its new-side + line numbers."""
    return {path: set(info["added_lines"]) for path, info in parsed.items()}


def hunk_index_path(daydream_dir: Path) -> Path:
    """Return the persisted hunk-index path under a run's ``.daydream`` dir."""
    return daydream_dir / HUNK_INDEX_FILENAME


def write_hunk_index(daydream_dir: Path, diff_text: str) -> Path:
    """Write deterministic path-sorted hunks and totals; return the artifact path.

    Omit the in-memory added_lines and added_text projections."""
    parsed = parse_hunks(diff_text)
    persist: dict[str, Any] = {}
    for file_path, info in parsed.items():
        persist[file_path] = {
            "hunks": info["hunks"],
            "added_total": info["added_total"],
            "removed_total": info["removed_total"],
        }
    index_path = hunk_index_path(daydream_dir)
    index_path.write_text(json.dumps(persist, sort_keys=True, indent=2))
    return index_path


def load_hunk_index(daydream_dir: Path) -> dict[str, Any]:
    """Load the persisted index; missing or malformed data yields {}."""
    path = hunk_index_path(daydream_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
