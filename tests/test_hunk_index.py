"""Tests for the shared unified-diff hunk parser and its consumer views.

Pins the load-bearing claim of the hunk-index refactor: a single parser in
``daydream.hunk_index`` reproduces all three previously-siloed line-numbering
contracts (pr_review head-side ranges, coverage added/removed totals,
quote_scrub added-line numbers), so the three can never drift apart.
"""

from __future__ import annotations

from daydream.hunk_index import added_line_numbers, head_side_ranges, parse_hunks


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