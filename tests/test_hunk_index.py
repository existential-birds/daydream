"""Tests for the shared unified-diff hunk parser and its consumer views.

Pins the load-bearing claim of the hunk-index refactor: a single parser in
``daydream.hunk_index`` reproduces all three previously-siloed line-numbering
contracts (pr_review head-side ranges, coverage added/removed totals,
quote_scrub added-line numbers), so the three can never drift apart.
"""
from __future__ import annotations

from pathlib import Path

from daydream.hunk_index import (
    added_line_numbers,
    files_in_index,
    head_side_ranges,
    head_side_ranges_by_file,
    load_hunk_index,
    parse_hunks,
    write_hunk_index,
)


def test_parse_hunks_matches_pr_head_side_ranges() -> None:
    diff = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1,3 +10,5 @@\n old\n+new1\n+new2\n@@ -20 +30,2 @@\n+new3\n"
    )
    parsed = parse_hunks(diff)
    # pr_review._parse_hunks contract: head-side (new_start,new_start+count-1)
    assert head_side_ranges(parsed) == [(10, 14), (30, 31)]
    # coverage.hunk_change_line_count contract: total + lines, headers excluded
    assert sum(fi["added_total"] + fi["removed_total"] for fi in parsed.values()) == 3
    # quote_scrub._added_line_numbers contract: new-side numbers of '+' lines
    assert added_line_numbers(parsed) == {"x.py": {11, 12, 30}}


def test_write_hunk_index_round_trips(tmp_path: Path) -> None:
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n x\n+y\n"
    write_hunk_index(tmp_path, diff)
    idx = load_hunk_index(tmp_path)
    assert files_in_index(idx) == ["a.py"]
    assert idx["a.py"]["added_total"] == 1 and idx["a.py"]["removed_total"] == 0
    assert idx["a.py"]["hunks"][0]["new_end"] == 2
    assert "added_lines" not in idx["a.py"]


def test_head_side_ranges_by_file_groups_the_flat_view_per_path() -> None:
    """#1113: the per-file view answers "is this line in a changed hunk of THIS
    file", which the flattened ``head_side_ranges`` cannot."""
    diff = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1,3 +10,5 @@\n old\n+new1\n+new2\n@@ -20 +30,2 @@\n+new3\n"
        "diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n"
        "@@ -1 +1,2 @@\n z\n+w\n"
    )
    parsed = parse_hunks(diff)
    by_file = head_side_ranges_by_file(parsed)
    assert by_file == {"x.py": [(10, 14), (30, 31)], "y.py": [(1, 2)]}
    # Same ranges as the flat view, only grouped — no range invented or lost.
    flat = [r for ranges in by_file.values() for r in ranges]
    assert sorted(flat) == sorted(head_side_ranges(parsed))


def test_head_side_ranges_by_file_reads_the_persisted_index(tmp_path: Path) -> None:
    """#1113: persistence drops only ``added_lines``, so the per-file accessor
    works identically on a loaded ``hunk-index.json``."""
    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n x\n+y\n"
    )
    write_hunk_index(tmp_path, diff)
    loaded = load_hunk_index(tmp_path)
    assert head_side_ranges_by_file(loaded) == head_side_ranges_by_file(parse_hunks(diff))
    assert head_side_ranges_by_file(loaded) == {"a.py": [(1, 2)]}


def test_head_side_ranges_by_file_keeps_pure_deletion_files_as_empty() -> None:
    """#1113: a file whose only hunk was a pure deletion has no head-side range
    but is still a changed file, so it maps to ``[]`` rather than vanishing."""
    diff = (
        "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ b/gone.py\n"
        "@@ -1,2 +0,0 @@\n-a\n-b\n"
    )
    parsed = parse_hunks(diff)
    assert head_side_ranges_by_file(parsed) == {"gone.py": []}
