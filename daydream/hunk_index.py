"""Persisted unified-diff hunk index and the single shared diff parser.

Every deep-review agent used to re-derive three deterministic facts -- changed
file/line ranges, symbol definitions, and finding ``file:line`` validity -- by
re-running ``git diff`` / ``sed`` on ``diff.patch``. This module owns the single
unified-diff parser and the persisted ``hunk-index.json`` write/load, and
exposes the three consumer views (``head_side_ranges``, ``added_line_numbers``,
``change_line_count``) so ``pr_review``, ``quote_scrub``, and ``coverage`` all
count from the same source and cannot drift.
"""

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
    """Unquote a ``core.quotepath``-quoted path from ``git diff`` output.

    Git quotes non-ASCII path names as C-style string literals, e.g.
    ``"b/caf\\303\\251.go"`` (octal escapes of the raw UTF-8 bytes). Strips the
    surrounding quotes, decodes the escapes back to bytes, and decodes those as
    UTF-8. Non-quoted input passes through unchanged.
    """
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
    """Resolve the repo-relative path from a ``+++ `` file-header line.

    Mirrors ``quote_scrub._header_path`` so the shared parser keys files
    identically across every consumer: handles the plain ``+++ b/rel/path`` form,
    git's ``core.quotepath`` quoted form, and ``diff.noprefix`` output (no ``b/``
    prefix). A trailing tab (git appends one after space-containing paths) is
    stripped first. Returns ``None`` for the ``+++ /dev/null`` deletion header.
    """
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
    """Parse a unified diff into per-file hunk information.

    Returns ``{path: {"hunks": [{"old_start", "old_end", "new_start",
    "new_end", "added", "removed", "added_lines"}], "added_total": N,
    "removed_total": N, "added_lines": set[int]}}``.

    ``added_lines`` (the new-side line numbers of ``+`` content lines) is the
    quote_scrub counter contract and lives only in the in-memory result; it is
    not persisted to ``hunk-index.json``. A ``+++ `` line counts as a file
    header only when preceded by its ``--- `` pair. Malformed input with no
    parseable hunks returns ``{}`` (a caller with no hunks behaves exactly as
    today, never raising).

    Args:
        diff_text: Unified-diff text (as written to ``diff.patch``).

    Returns:
        A path-keyed dict (sorted deterministically by path) of per-file hunk
        information.
    """
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
                {"hunks": [], "added_total": 0, "removed_total": 0, "added_lines": set()},
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
                "added_lines": set(),
            }
            current_meta["hunks"].append(current_hunk)
        elif raw.startswith("+"):
            current_meta["added_total"] += 1
            current_meta["added_lines"].add(new_line)
            if current_hunk is not None:
                current_hunk["added"] += 1
                current_hunk["added_lines"].add(new_line)
            new_line += 1
        elif raw.startswith("-"):
            current_meta["removed_total"] += 1
            if current_hunk is not None:
                current_hunk["removed"] += 1
        elif raw.startswith(" "):
            new_line += 1
    for info in result.values():
        for hunk in info["hunks"]:
            hunk["added_lines"] = sorted(hunk["added_lines"])
        info["added_lines"] = set(info["added_lines"])
    return result


def head_side_ranges(parsed: dict[str, dict[str, Any]]) -> list[tuple[int, int]]:
    """Flatten every hunk's new-side inclusive range across all files.

    Mirrors the ``pr_review._parse_hunks`` contract: ``(new_start,
    new_start + count - 1)`` for each hunk, in diff order.
    """
    ranges: list[tuple[int, int]] = []
    for info in parsed.values():
        for hunk in info["hunks"]:
            ranges.append((hunk["new_start"], hunk["new_end"]))
    return ranges


def added_line_numbers(parsed: dict[str, dict[str, Any]]) -> dict[str, set[int]]:
    """Map each changed file to the new-side line numbers of its ``+`` lines.

    Mirrors the ``quote_scrub._added_line_numbers`` contract (every file in the
    diff is a key; values are the union of per-hunk new-side added-line
    numbers).
    """
    return {path: set(info["added_lines"]) for path, info in parsed.items()}


def change_line_count(parsed: dict[str, dict[str, Any]], file: str) -> int:
    """Return the count of changed content lines for one file in a parse result.

    The single shared hunk-line counter (the count of changed content lines,
    file headers excluded) consumed by ``coverage.filter_sweepable_files``.
    """
    info = parsed.get(file)
    if info is None:
        return 0
    return info["added_total"] + info["removed_total"]


def hunk_index_path(daydream_dir: Path) -> Path:
    """Return the persisted hunk-index path under a run's ``.daydream`` dir."""
    return daydream_dir / HUNK_INDEX_FILENAME


def write_hunk_index(daydream_dir: Path, diff_text: str) -> Path:
    """Write ``hunk-index.json`` under ``daydream_dir`` from a diff text.

    The persisted shape is exactly the issue's: ``{path: {"hunks": [{"new_start",
    "new_end", "old_start", "old_end", "added", "removed"}], "added_total": N,
    "removed_total": N}}`` (JSON, sorted by path for determinism). ``added_lines``
    is intentionally NOT persisted. Returns the written path.
    """
    parsed = parse_hunks(diff_text)
    persist: dict[str, Any] = {}
    for file_path, info in parsed.items():
        persist[file_path] = {
            "hunks": [
                {
                    "old_start": h["old_start"],
                    "old_end": h["old_end"],
                    "new_start": h["new_start"],
                    "new_end": h["new_end"],
                    "added": h["added"],
                    "removed": h["removed"],
                }
                for h in info["hunks"]
            ],
            "added_total": info["added_total"],
            "removed_total": info["removed_total"],
        }
    index_path = hunk_index_path(daydream_dir)
    index_path.write_text(json.dumps(persist, sort_keys=True, indent=2))
    return index_path


def load_hunk_index(daydream_dir: Path) -> dict[str, Any]:
    """Load the persisted hunk index, or ``{}`` when missing/malformed.

    Fail-open: a missing index degrades to "no changed files", never raises.
    """
    path = hunk_index_path(daydream_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def files_in_index(index: dict[str, Any]) -> list[str]:
    """Return the sorted keys (changed file paths) of a hunk index."""
    return sorted(index)
