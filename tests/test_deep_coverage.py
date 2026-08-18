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
    """Write one sibling trajectory whose *completed* reads cover *read_paths*.

    Each read tool call carries a matching ``observation.results[].source_call_id``
    so the sweep counts it as coverage (a read without a ToolResult observation
    is treated as interrupted and does NOT cover the file).
    """
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    tool_calls = []
    results = []
    for i, path in enumerate(read_paths):
        call_id = f"read-{i}"
        tool_calls.append(
            {
                "tool_call_id": call_id,
                "function_name": "Read",
                "arguments": {"file_path": path},
            }
        )
        results.append({"source_call_id": call_id, "content": "file content"})
    (trajectories_dir / name).write_text(
        json.dumps(
            {
                "session_id": run_dir.name,
                "steps": [
                    {
                        "step_id": "s0",
                        "tool_calls": tool_calls,
                        "observation": {"results": results},
                    }
                ],
            }
        )
    )


def _write_interrupted_read_fork(run_dir: Path, name: str, read_paths: list[str]) -> None:
    """Write a sibling trajectory whose reads carry NO ToolResult observation.

    Models an interrupted Read (a ToolStartEvent with no matching
    ToolResultEvent): the file was not actually read, so the sweep must treat
    it as uncovered (fail-open: it gets swept, never skipped).
    """
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
                                "tool_call_id": f"read-{i}",
                                "function_name": "Read",
                                "arguments": {"file_path": path},
                            }
                            for i, path in enumerate(read_paths)
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


def _write_colliding_read_id_fork(run_dir: Path) -> None:
    """Write a two-step sibling trajectory that reuses one tool-call ID.

    Tool-call IDs are scoped to individual invocations, not trajectory-global.
    Step ``s0`` holds an interrupted Read of ``/repo/api.py`` whose ID collides
    with the completed Read of ``/repo/notes.txt`` in step ``s1``. The
    completed result in ``s1`` must NOT retroactively complete the interrupted
    read in ``s0`` -- the sweep treats ``api.py`` as uncovered.
    """
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    (trajectories_dir / "deep-python.json").write_text(
        json.dumps(
            {
                "session_id": run_dir.name,
                "steps": [
                    {
                        "step_id": "s0",
                        "tool_calls": [
                            {
                                "tool_call_id": "read-0",
                                "function_name": "Read",
                                "arguments": {"file_path": "/repo/api.py"},
                            }
                        ],
                    },
                    {
                        "step_id": "s1",
                        "tool_calls": [
                            {
                                "tool_call_id": "read-0",
                                "function_name": "Read",
                                "arguments": {"file_path": "/repo/notes.txt"},
                            }
                        ],
                        "observation": {
                            "results": [{"source_call_id": "read-0", "content": "file content"}]
                        },
                    },
                ],
            }
        )
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


def test_compute_uncovered_files_boundary_ignores_suffix_collisions(tmp_path: Path) -> None:
    """A read of ``notapi.py`` must NOT cover the changed file ``api.py``.

    Regression for the suffix-collision false positive: coverage is matched at
    a path-component boundary (``endswith("/" + relative)``), so
    ``/repo/notapi.py`` never counts as a read of ``api.py`` and the file stays
    in the sweep.
    """
    daydream_dir = tmp_path / ".daydream"
    daydream_dir.mkdir()
    (daydream_dir / "diff.patch").write_text(_DIFF)
    run_dir = daydream_dir / "runs" / "sess-3"
    run_dir.mkdir(parents=True)
    _write_main(run_dir)
    _write_fork(run_dir, "deep-python.json", ["/repo/notapi.py"])
    _write_fork(run_dir, "deep-generic.json", ["/repo/notes.txt"])

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-3")

    assert "api.py" in uncovered  # /repo/notapi.py must not cover api.py
    assert stats["files_read_by_reviewers"] == 1  # only notes.txt covered
    swept, _, _ = filter_sweepable_files(uncovered, _DIFF, min_hunk_lines=1, max_files=10)
    assert "api.py" in swept  # api.py is swept, not silently skipped


def test_compute_uncovered_files_requires_completed_reads(tmp_path: Path) -> None:
    """An interrupted Read (no ToolResult observation) leaves the file uncovered.

    A reviewer that starts a Read but never receives a result has not read the
    file; counting it as coverage would let the sweep skip a genuinely unread
    file. Fail-open: the file stays uncovered and is swept.
    """
    daydream_dir = tmp_path / ".daydream"
    daydream_dir.mkdir()
    (daydream_dir / "diff.patch").write_text(_DIFF)
    run_dir = daydream_dir / "runs" / "sess-4"
    run_dir.mkdir(parents=True)
    _write_main(run_dir)
    _write_interrupted_read_fork(run_dir, "deep-python.json", ["/repo/api.py"])
    _write_fork(run_dir, "deep-generic.json", ["/repo/notes.txt"])

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-4")

    assert "api.py" in uncovered  # the interrupted read covers nothing
    assert stats["files_read_by_reviewers"] == 1  # only notes.txt's completed read
    swept, _, _ = filter_sweepable_files(uncovered, _DIFF, min_hunk_lines=1, max_files=10)
    assert "api.py" in swept  # the unread file is swept, never skipped


def test_compute_uncovered_files_scopes_completed_ids_to_step(tmp_path: Path) -> None:
    """A tool-call ID reused across steps must not leak completion state.

    Regression (issue #309 finding 5): completion is matched WITHIN a step, so
    a completed read in one step cannot mark an interrupted read in another
    step (sharing the same ID) as completed. The interrupted read stays
    uncovered and the file is swept, never skipped.
    """
    daydream_dir = tmp_path / ".daydream"
    daydream_dir.mkdir()
    (daydream_dir / "diff.patch").write_text(_DIFF)
    run_dir = daydream_dir / "runs" / "sess-5"
    run_dir.mkdir(parents=True)
    _write_main(run_dir)
    _write_colliding_read_id_fork(run_dir)

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-5")

    # The interrupted read of api.py (ID collision with s1's completed read of
    # notes.txt) covers nothing: api.py stays uncovered and is swept.
    assert "api.py" in uncovered
    assert stats["files_read_by_reviewers"] == 1  # only notes.txt's completed read
    swept, _, _ = filter_sweepable_files(uncovered, _DIFF, min_hunk_lines=1, max_files=10)
    assert "api.py" in swept


def test_hunk_change_line_count_excludes_headers() -> None:
    """+++/--- file headers are not counted as added/removed lines."""
    assert hunk_change_line_count(diff_block_for_file(_DIFF, "api.py") or "") == 2
    assert hunk_change_line_count(diff_block_for_file(_DIFF, "notes.txt") or "") == 6


def test_filter_sweepable_files_caps_capacity_and_counts_small_hunks() -> None:
    """Small hunks are skipped; excess sweepable files are named, not dropped."""
    uncovered = ["api.py", "notes.txt"]

    swept, small_files, capacity_files = filter_sweepable_files(
        uncovered, _DIFF, min_hunk_lines=5, max_files=10
    )

    assert swept == ["notes.txt"]
    assert small_files == ["api.py"]  # api.py has only 2 +/- lines
    assert capacity_files == []
    # Integer counts are derived from the lists (issue #309 finding 10).
    assert len(small_files) == 1
    assert len(capacity_files) == 0


def test_filter_sweepable_files_skips_nonexistent_diff_files() -> None:
    """An uncovered file absent from the diff is skipped as non-sweepable."""
    swept, small_files, capacity_files = filter_sweepable_files(
        ["ghost.txt"], _DIFF, min_hunk_lines=1, max_files=10
    )

    assert swept == []
    assert small_files == ["ghost.txt"]
    assert capacity_files == []


def test_filter_sweepable_files_capacity_cap_keeps_diff_order() -> None:
    """With more sweepable files than max_files, only the first N are kept."""
    diff = _DIFF + (
        "diff --git a/third.txt b/third.txt\n"
        "--- a/third.txt\n"
        "+++ b/third.txt\n"
        "@@ -1 +1,6 @@\n"
        "+a\n+b\n+c\n+d\n+e\n+f\n"
    )

    swept, small_files, capacity_files = filter_sweepable_files(
        ["notes.txt", "third.txt"], diff, min_hunk_lines=5, max_files=1
    )

    assert swept == ["notes.txt"]
    assert small_files == []
    assert capacity_files == ["third.txt"]
    assert len(capacity_files) == 1


def test_filter_sweepable_files_zero_max_files_sweeps_nothing() -> None:
    """max_files=0 sweeps nothing; every sweepable file is capacity-skipped."""
    swept, small_files, capacity_files = filter_sweepable_files(
        ["api.py", "notes.txt"], _DIFF, min_hunk_lines=1, max_files=0
    )

    assert swept == []
    assert small_files == []
    assert capacity_files == ["api.py", "notes.txt"]


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
    # Reading the source file is REQUIRED, not optional (issue #309 finding 6):
    # a hunk-only review must not be reported as read coverage.
    assert "Read the source file FIRST" in prompt
    assert "you may only comment on hunks you have read" in prompt
    # The canonical prompt primitives are present, not reduced duplicates: the
    # sweep reviewer is held to the same standard as per-stack reviewers
    # (issue #309 finding 11).
    assert "## Confidence and Convention Rules" in prompt
    assert "Error Handling Semantics (QUAL-04)" in prompt
    assert "## Dependency Impact" in prompt
    assert "Gate-0 anti-confabulation" in prompt
    assert "review-verification-protocol/SKILL.md" not in prompt


def test_build_uncovered_sweep_prompt_exploration_pointer(tmp_path: Path) -> None:
    """The exploration pointer is inlined only when a directory is supplied."""
    intent = tmp_path / ".daydream" / "deep" / "intent.md"
    output = tmp_path / ".daydream" / "deep" / "uncovered-0-review.md"
    hunks = diff_block_for_file(_DIFF, "notes.txt") or ""
    exploration = tmp_path / ".daydream" / "exploration"
    prompt = build_uncovered_sweep_prompt(
        file="notes.txt",
        hunks=hunks,
        intent_path=intent,
        cwd=tmp_path,
        output_path=output,
        exploration_dir=exploration,
    )

    assert "Pre-scan exploration results are available" in prompt
    assert str(exploration) in prompt

    prompt_no_dir = build_uncovered_sweep_prompt(
        file="notes.txt",
        hunks=hunks,
        intent_path=intent,
        cwd=tmp_path,
        output_path=output,
        exploration_dir=None,
    )
    assert "Pre-scan exploration results" not in prompt_no_dir


def test_coverage_receipt_records_inline_and_frontier(tmp_path: Path) -> None:
    """Issue #731: the deterministic coverage-receipts writer round-trips."""
    from daydream.deep.coverage import coverage_receipt_path, write_coverage_receipts

    deep = tmp_path / ".daydream" / "deep"
    receipts = {"python#0": {"assigned_files": ["a.py"], "inline_files": ["a.py"],
                             "frontier_files": ["shared/iface.py"]}}
    write_coverage_receipts(deep, receipts)
    assert json.loads(coverage_receipt_path(deep).read_text()) == receipts
