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
from typing import Any
from pathlib import Path

# Block / header regexes mirroring ``daydream.deep.prompts`` so a hunk's owning
# path resolves identically across every consumer (and the persisted index is
# derived from the exact unified-diff contract the deep flow writes).
_DIFF_BLOCK_SPLIT = re.compile(r"^(?=diff --git )", re.MULTILINE)
_DIFF_PLUS_HEADER = re.compile(r"^\+\+\+ (.+)$")
_DIFF_MINUS_HEADER = re.compile(r"^--- (.+)$")
_DIFF_GIT_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)")
# Unified-diff hunk header: @@ -<old_start>[,<old_count>] +<new_start>[,<new_count>] @@
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

HUNK_INDEX_FILENAME = "hunk-index.json"


def _strip_prefix(path: str, prefix: str) -> str:
    return path[len(prefix) :] if path.startswith(prefix) else path


def _diff_block_path(block: str) -> str | None:
    """Resolve the single changed path for one ``diff --git`` block.

    Mirrors ``daydream.deep.prompts._diff_block_path``: prefer the post-state
    path (``+++ b/<path>``), fall back to the pre-state path for deletions, and
    to the ``diff --git`` header for binary / mode-only diffs. ``/dev/null``
    sentinels are skipped at every layer.
    """
    if not block.startswith("diff --git "):
        return None
    plus = _DIFF_PLUS_HEADER.search(block)
    if plus and plus.group(1) != "/dev/null":
        return _strip_prefix(plus.group(1), "b/")
    minus = _DIFF_MINUS_HEADER.search(block)
    if minus and minus.group(1) != "/dev/null":
        return _strip_prefix(minus.group(1), "a/")
    git = _DIFF_GIT_HEADER.match(block)
    if git:
        return git.group(2)
    return None


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
    for block in _DIFF_BLOCK_SPLIT.split(diff_text):
        path = _diff_block_path(block)
        if path is None:
            continue
        hunks: list[dict[str, Any]] = []
        added_total = 0
        removed_total = 0
        added_lines: set[int] = set()
        new_line = 0
        current: dict[str, Any] | None = None
        prev_old_header = False
        for raw in block.splitlines():
            if raw.startswith(("--- ", '--- "')):
                prev_old_header = True
                continue
            if raw.startswith("+++ ") and prev_old_header:
                prev_old_header = False
                continue
            prev_old_header = False
            header = _HUNK_HEADER.match(raw)
            if raw.startswith("@@") and header:
                old_start = int(header.group(1))
                old_count = int(header.group(2)) if header.group(2) else 1
                new_start = int(header.group(3))
                new_count = int(header.group(4)) if header.group(4) else 1
                new_line = new_start
                if new_count == 0:
                    # Empty new-side range (pure deletion): pr_review skips it.
                    current = None
                    continue
                current = {
                    "old_start": old_start,
                    "old_end": old_start + old_count - 1,
                    "new_start": new_start,
                    "new_end": new_start + new_count - 1,
                    "added": 0,
                    "removed": 0,
                    "added_lines": set(),
                }
                hunks.append(current)
            elif raw.startswith("+"):
                added_total += 1
                added_lines.add(new_line)
                if current is not None:
                    current["added"] += 1
                    current["added_lines"].add(new_line)
                new_line += 1
            elif raw.startswith("-"):
                removed_total += 1
                if current is not None:
                    current["removed"] += 1
            elif raw.startswith(" "):
                new_line += 1
        for hunk in hunks:
            hunk["added_lines"] = sorted(hunk["added_lines"])
        result[path] = {
            "hunks": hunks,
            "added_total": added_total,
            "removed_total": removed_total,
            "added_lines": set(added_lines),
        }
    return {path: result[path] for path in sorted(result)}


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
    """Return ``added_total + removed_total`` for one file in a parse result.

    Mirrors the ``coverage.hunk_change_line_count`` contract -- the count of
    changed content lines, file headers excluded.
    """
    info = parsed.get(file)
    if info is None:
        return 0
    return info["added_total"] + info["removed_total"]


def _daydream_dir(daydream_dir: Path) -> Path:
    return Path(daydream_dir)


def hunk_index_path(daydream_dir: Path) -> Path:
    """Return the persisted hunk-index path under a run's ``.daydream`` dir."""
    return _daydream_dir(daydream_dir) / HUNK_INDEX_FILENAME


def write_hunk_index(daydream_dir: Path, diff_text: str) -> Path:
    """Write ``hunk-index.json`` under ``daydream_dir`` from a diff text.

    The persisted shape is exactly the issue's: ``{path: {"hunks": [{"new_start",
    "new_end", "old_start", "old_end", "added", "removed"}], "added_total": N,
    "removed_total": N}}`` (JSON, sorted by path for determinism). ``added_lines``
    is intentionally NOT persisted. Returns the written path.
    """
    parsed = parse_hunks(diff_text)
    persist: dict[str, Any] = {}
    for path, info in parsed.items():
        persist[path] = {
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
    path = hunk_index_path(daydream_dir)
    path.write_text(json.dumps(persist, sort_keys=True, indent=2))
    return path


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