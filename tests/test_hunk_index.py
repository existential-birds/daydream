"""Tests for the shared unified-diff hunk parser and its consumer views.

Pins the load-bearing claim of the hunk-index refactor: a single parser in
``daydream.hunk_index`` reproduces all three previously-siloed line-numbering
contracts (pr_review head-side ranges, coverage added/removed totals,
quote_scrub added-line numbers), so the three can never drift apart.
"""

from __future__ import annotations

from daydream.hunk_index import (
    added_line_numbers,
    files_in_index,
    head_side_ranges,
    load_hunk_index,
    parse_hunks,
    write_hunk_index,
)


def test_parse_hunks_matches_pr_head_side_ranges():
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


def test_write_hunk_index_round_trips(tmp_path):
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n x\n+y\n"
    write_hunk_index(tmp_path, diff)
    idx = load_hunk_index(tmp_path)
    assert files_in_index(idx) == ["a.py"]
    assert idx["a.py"]["added_total"] == 1 and idx["a.py"]["removed_total"] == 0
    assert idx["a.py"]["hunks"][0]["new_end"] == 2
    assert "added_lines" not in idx["a.py"]