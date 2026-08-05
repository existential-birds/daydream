"""Unit tests for the uncovered-file sweep helpers (issue #309).

Covers ``daydream/deep/coverage.py``: coverage computation against a crafted
``.daydream`` dir, the hunk-size + capacity budget filter, and the sweep prompt
builder. The real-path sweep behavior lives in ``tests/test_deep_orchestrator.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from daydream.deep.coverage import (
    build_uncovered_sweep_prompt,
    compute_uncovered_files,
    diff_block_for_file,
    filter_sweepable_files,
    hunk_change_line_count,
)

_DIFF = (
    "diff --git a/api.py b/api.py\n"
    "index 000..111 100644\n"
    "--- a/api.py\n"
    "+++ b/api.py\n"
    "@@ -1 +1 @@\n"
    "-'world'\n"
    "+'universe'\n"
    "diff --git a/notes.txt b/notes.txt\n"
    "index 000..222 100644\n"
    "--- a/notes.txt\n"
    "+++ b/notes.txt\n"
    "@@ -1 +1,7 @@\n"
    "+line1\n"
    "+line2\n"
    "+line3\n"
    "+line4\n"
    "+line5\n"
    "+line6\n"
)


def _write_fork(run_dir: Path, name: str, read_paths: list[str]) -> None:
    """Write one sibling trajectory whose tool calls read *read_paths*."""
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    (trajectories_dir / name).write_text(
        json.dumps(
            {
                "session_id": run_dir.name,
                "steps": [
                    {
                        "step_id": "s0",
                        "tool_calls": [
                            {
                                "function_name": "Read",
                                "arguments": {"file_path": path},
                            }
                            for path in read_paths
                        ],
                    }
                ],
            }
        )
    )


def _write_main(run_dir: Path) -> None:
    (run_dir / "trajectory.json").write_text(
        json.dumps({"session_id": run_dir.name, "steps": []})
    )


def test_compute_uncovered_files_reports_unread_diff_files(tmp_path: Path) -> None:
    """Files no ``deep-`` reviewer read land in the uncovered list."""
    daydream_dir = tmp_path / ".daydream"
    daydream_dir.mkdir()
    (daydream_dir / "diff.patch").write_text(_DIFF)
    run_dir = daydream_dir / "runs" / "sess-1"
    run_dir.mkdir(parents=True)
    _write_main(run_dir)
    _write_fork(run_dir, "deep-python.json", ["/repo/api.py"])
    # parse forks must NOT count toward coverage (label does not start with deep-).
    _write_fork(run_dir, "parse-python.json", ["/repo/notes.txt"])

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-1")

    assert uncovered == ["notes.txt"]
    assert stats["files_in_diff"] == 2
    assert stats["files_read_by_reviewers"] == 1
    assert stats["coverage_ratio"] == 0.5


def test_compute_uncovered_files_empty_when_everything_read(tmp_path: Path) -> None:
    """A fully-covered diff reports no uncovered files."""
    daydream_dir = tmp_path / ".daydream"
    daydream_dir.mkdir()
    (daydream_dir / "diff.patch").write_text(_DIFF)
    run_dir = daydream_dir / "runs" / "sess-2"
    run_dir.mkdir(parents=True)
    _write_main(run_dir)
    _write_fork(run_dir, "deep-python.json", ["/repo/api.py"])
    _write_fork(run_dir, "deep-generic.json", ["/repo/notes.txt"])

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-2")

    assert uncovered == []
    assert stats["coverage_ratio"] == 1.0


def test_hunk_change_line_count_excludes_headers() -> None:
    """+++/--- file headers are not counted as added/removed lines."""
    assert hunk_change_line_count(diff_block_for_file(_DIFF, "api.py") or "") == 2
    assert hunk_change_line_count(diff_block_for_file(_DIFF, "notes.txt") or "") == 6


def test_filter_sweepable_files_caps_capacity_and_counts_small_hunks() -> None:
    """Small hunks are skipped; excess sweepable files are counted, not dropped."""
    uncovered = ["api.py", "notes.txt"]

    swept, small, capacity = filter_sweepable_files(
        uncovered, _DIFF, min_hunk_lines=5, max_files=10
    )

    assert swept == ["notes.txt"]
    assert small == 1  # api.py has only 2 +/- lines
    assert capacity == 0


def test_filter_sweepable_files_skips_nonexistent_diff_files() -> None:
    """An uncovered file absent from the diff is skipped as non-sweepable."""
    swept, small, capacity = filter_sweepable_files(
        ["ghost.txt"], _DIFF, min_hunk_lines=1, max_files=10
    )

    assert swept == []
    assert small == 1
    assert capacity == 0


def test_filter_sweepable_files_capacity_cap_keeps_diff_order() -> None:
    """With more sweepable files than max_files, only the first N are kept."""
    diff = _DIFF + (
        "diff --git a/third.txt b/third.txt\n"
        "--- a/third.txt\n"
        "+++ b/third.txt\n"
        "@@ -1 +1,6 @@\n"
        "+a\n+b\n+c\n+d\n+e\n+f\n"
    )

    swept, small, capacity = filter_sweepable_files(
        ["notes.txt", "third.txt"], diff, min_hunk_lines=5, max_files=1
    )

    assert swept == ["notes.txt"]
    assert small == 0
    assert capacity == 1


def test_build_uncovered_sweep_prompt_includes_context_and_markers(tmp_path: Path) -> None:
    """The prompt names the file, inlines hunks, points at intent + output."""
    intent = tmp_path / ".daydream" / "deep" / "intent.md"
    output = tmp_path / ".daydream" / "deep" / "uncovered-0-review.md"
    hunks = diff_block_for_file(_DIFF, "notes.txt") or ""
    prompt = build_uncovered_sweep_prompt(
        file="notes.txt",
        hunks=hunks,
        intent_path=intent,
        cwd=tmp_path,
        output_path=output,
    )

    assert "notes.txt" in prompt
    assert "uncovered file sweep" in prompt
    assert str(intent) in prompt
    assert str(output) in prompt
    # Hunks are inlined (not a pointer to diff.patch).
    assert "+line6" in prompt
    assert "changed file notes.txt was NOT read" in prompt
    # The verification gates are embedded, not a skill-file read instruction.
    assert "Gate-0 anti-confabulation" in prompt
    assert "review-verification-protocol/SKILL.md" not in prompt
