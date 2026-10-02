"""Unit tests for the uncovered-file sweep helpers (issue #309).

Covers ``daydream/deep/coverage.py``: coverage computation against a crafted
``.daydream`` dir, the hunk-size + capacity budget filter, and the sweep prompt
builder. The real-path sweep behavior lives in ``tests/deep_orchestrator/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream import review_profile as rp, severity
from daydream.deep.coverage import (
    _completed_read_paths,
    bounded_diff_block_for_file,
    build_uncovered_sweep_prompt,
    compute_uncovered_files,
    coverage_receipt_path,
    filter_sweepable_files,
    resolve_per_stack_verdicts,
    write_coverage_receipts,
)
from daydream.eval.analyzer import load_trajectories
from daydream.hunk_index import load_hunk_index, parse_hunks, write_hunk_index
from daydream.repository_paths import strip_dot_slash

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


def _write_fork_calls(
    run_dir: Path,
    name: str,
    calls: list[dict[str, Any]],
    *,
    include_results: bool = True,
    result_extra: dict[str, Any] | None = None,
) -> None:
    """Write one sibling step carrying *calls* as its tool calls.

    With *include_results* (default) each tool call carries a matching
    ``observation.results[].source_call_id`` so the sweep counts it as coverage;
    without it the step has no ``observation`` (an interrupted call, which does
    NOT cover the file). *result_extra*, when given, is attached as every
    emitted result's ``extra`` (absent means the result dict is byte-identical
    to the original shape).
    """
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    tool_calls = []
    results = []
    for i, call in enumerate(calls):
        call_id = f"read-{i}"
        tool_calls.append(
            {
                "tool_call_id": call_id,
                "function_name": call["function_name"],
                "arguments": call.get("arguments", {}),
            }
        )
        result: dict[str, Any] = {"source_call_id": call_id, "content": "file content"}
        if result_extra is not None:
            result["extra"] = result_extra
        results.append(result)
    step: dict[str, Any] = {"step_id": "s0", "tool_calls": tool_calls}
    if include_results:
        step["observation"] = {"results": results}
    (trajectories_dir / name).write_text(
        json.dumps({"session_id": run_dir.name, "steps": [step]})
    )


def _write_fork(run_dir: Path, name: str, read_paths: list[str]) -> None:
    """Write one completed sibling trajectory whose *completed* reads cover *read_paths*.

    Re-expresses the shared envelope as ``Read`` calls carrying
    ``arguments.file_path``.
    """
    _write_fork_calls(
        run_dir,
        name,
        [
            {"function_name": "Read", "arguments": {"file_path": path}}
            for path in read_paths
        ],
    )


def _write_interrupted_read_fork(run_dir: Path, name: str, read_paths: list[str]) -> None:
    """Write a sibling trajectory whose reads carry NO ToolResult observation.

    Models an interrupted Read (a ToolStartEvent with no matching
    ToolResultEvent): the file was not actually read, so the sweep must treat
    it as uncovered (fail-open: it gets swept, never skipped).
    """
    _write_fork_calls(
        run_dir,
        name,
        [{"function_name": "Read", "arguments": {"file_path": path}} for path in read_paths],
        include_results=False,
    )


def _write_main(run_dir: Path) -> None:
    (run_dir / "trajectory.json").write_text(
        json.dumps({"session_id": run_dir.name, "steps": []})
    )


def _seed_coverage_run(
    tmp_path: Path,
    session: str,
    *,
    with_index: bool = True,
    deep: bool = False,
) -> tuple[Path, Path]:
    """Create a ``.daydream`` run for coverage tests; optionally index the diff.

    Returns ``(daydream_dir, run_dir)``. ``deep=True`` also creates the deep
    artifact dir that holds coverage receipts and per-shard records.
    """
    daydream_dir = tmp_path / ".daydream"
    daydream_dir.mkdir()
    if with_index:
        write_hunk_index(daydream_dir, _DIFF)
    run_dir = daydream_dir / "runs" / session
    run_dir.mkdir(parents=True)
    _write_main(run_dir)
    if deep:
        (daydream_dir / "deep").mkdir(parents=True)
    return daydream_dir, run_dir


_ONE_ISSUE = {
    "file": "api.py",
    "id": 1,
    "description": "d",
    "line": 1,
    "severity": "low",
    "confidence": "MEDIUM",
    "rationale": "r",
    "evidence": "e",
}


def _write_records(deep: Path, shard: str, payload: Any) -> None:
    """Write one shard's ``stack-<shard>-records.json`` for coverage tests."""
    (deep / f"stack-{shard}-records.json").write_text(json.dumps(payload))


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
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-1")
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
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-2")
    _write_fork(run_dir, "deep-python.json", ["/repo/api.py"])
    _write_fork(run_dir, "deep-generic.json", ["/repo/notes.txt"])

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-2")

    assert uncovered == []
    assert stats["coverage_ratio"] == 1.0


def test_compute_uncovered_files_missing_index_surfaces_gap(tmp_path: Path) -> None:
    """Issue #336: a missing hunk index is surfaced, never a full-coverage pass.

    When ``hunk-index.json`` is absent, ``load_hunk_index`` fails open to
    ``{}`` and ``files_in_index`` returns ``[]`` -- but that empty diff is NOT
    evidence full coverage. The stats must surface the unenumerated
    changed-file set (ratio ``None``, ``hunk_index_missing`` true) so the sweep
    does not silently render a coverage gap as a clean pass.
    """
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-missing", with_index=False)
    _write_fork(run_dir, "deep-python.json", ["/repo/api.py"])  # reviewers read, but index absent

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-missing")

    assert uncovered == []                     # cannot enumerate without the index
    assert stats["files_in_diff"] == 0         # index absent -> no changed files
    assert stats["coverage_ratio"] is None     # NOT a false 1.0 full-coverage pass
    assert stats["hunk_index_missing"] is True  # gap is surfaced


def test_compute_uncovered_files_boundary_ignores_suffix_collisions(tmp_path: Path) -> None:
    """A read of ``notapi.py`` must NOT cover the changed file ``api.py``.

    Regression for the suffix-collision false positive: coverage is matched at
    a path-component boundary (``endswith("/" + relative)``), so
    ``/repo/notapi.py`` never counts as a read of ``api.py`` and the file stays
    in the sweep.
    """
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-3")
    _write_fork(run_dir, "deep-python.json", ["/repo/notapi.py"])
    _write_fork(run_dir, "deep-generic.json", ["/repo/notes.txt"])

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-3")

    assert "api.py" in uncovered  # /repo/notapi.py must not cover api.py
    assert stats["files_read_by_reviewers"] == 1  # only notes.txt covered
    swept, _, _ = filter_sweepable_files(uncovered, parse_hunks(_DIFF), min_hunk_lines=1, max_files=10)
    assert "api.py" in swept  # api.py is swept, not silently skipped


def test_compute_uncovered_files_requires_completed_reads(tmp_path: Path) -> None:
    """An interrupted Read (no ToolResult observation) leaves the file uncovered.

    A reviewer that starts a Read but never receives a result has not read the
    file; counting it as coverage would let the sweep skip a genuinely unread
    file. Fail-open: the file stays uncovered and is swept.
    """
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-4")
    _write_interrupted_read_fork(run_dir, "deep-python.json", ["/repo/api.py"])
    _write_fork(run_dir, "deep-generic.json", ["/repo/notes.txt"])

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-4")

    assert "api.py" in uncovered  # the interrupted read covers nothing
    assert stats["files_read_by_reviewers"] == 1  # only notes.txt's completed read
    swept, _, _ = filter_sweepable_files(uncovered, parse_hunks(_DIFF), min_hunk_lines=1, max_files=10)
    assert "api.py" in swept  # the unread file is swept, never skipped


def test_compute_uncovered_files_scopes_completed_ids_to_step(tmp_path: Path) -> None:
    """A tool-call ID reused across steps must not leak completion state.

    Regression (issue #309 finding 5): completion is matched WITHIN a step, so
    a completed read in one step cannot mark an interrupted read in another
    step (sharing the same ID) as completed. The interrupted read stays
    uncovered and the file is swept, never skipped.
    """
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-5")
    _write_colliding_read_id_fork(run_dir)

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-5")

    # The interrupted read of api.py (ID collision with s1's completed read of
    # notes.txt) covers nothing: api.py stays uncovered and is swept.
    assert "api.py" in uncovered
    assert stats["files_read_by_reviewers"] == 1  # only notes.txt's completed read
    swept, _, _ = filter_sweepable_files(uncovered, parse_hunks(_DIFF), min_hunk_lines=1, max_files=10)
    assert "api.py" in swept


def test_loop_read_covers_through_the_sweep_and_verdict_seams(tmp_path: Path) -> None:
    """Loop reads count as coverage through both sweep admission and verdict reconciliation."""
    # Issue #1397 requirement 2: one loop-resolution semantics for every
    # consumer that credits reads -- sweep admission and verdict reconciliation.
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-loop")
    _write_fork_calls(
        run_dir,
        "deep-python.json",
        [{
            "function_name": "shell",
            "arguments": {"command": 'for f in api.py notes.txt; do nl -ba "$f"; done'},
        }],
    )

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-loop")

    assert uncovered == []                        # both listed files are covered
    assert stats["files_read_by_reviewers"] == 2

    fork = load_trajectories(daydream_dir, "sess-loop")["forked"][0]
    verdicts = resolve_per_stack_verdicts(
        assigned_files=["api.py", "notes.txt"],
        declared_verdicts=[
            {"path": "api.py", "lines_read": 10, "verdict": "clean"},
            {"path": "notes.txt", "lines_read": 10, "verdict": "clean"},
        ],
        completed_read_paths=_completed_read_paths(fork),
        finding_files=set(),
    )

    assert [v["verdict"] for v in verdicts] == ["clean", "clean"]


@pytest.mark.parametrize("extra", [
    {"is_error": True},
    {"cancelled": True},
    {"status": "interrupted"},
    {"truncated": True},
    {"exit_code": 2},
])
def test_damaged_observations_credit_no_coverage(tmp_path: Path, extra: dict[str, Any]) -> None:
    """Damaged observations -- failed, cancelled, interrupted, truncated, or non-zero-exit -- credit no coverage."""
    # Issue #1397 requirement 5: a paired result marked failed/cancelled/
    # interrupted/truncated/non-zero-exit establishes no coverage.
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-damaged")
    _write_fork_calls(
        run_dir,
        "deep-python.json",
        [{"function_name": "Read", "arguments": {"file_path": "/repo/api.py"}}],
        result_extra=extra,
    )
    _write_fork(run_dir, "deep-generic.json", ["/repo/notes.txt"])

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-damaged")

    assert "api.py" in uncovered                  # the damaged read is not coverage
    assert stats["files_read_by_reviewers"] == 1  # only notes.txt


def test_zero_exit_code_observation_still_credits(tmp_path: Path) -> None:
    """A clean zero-exit-code read still credits coverage."""
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-clean-exit")
    _write_fork_calls(
        run_dir,
        "deep-python.json",
        [{"function_name": "Read", "arguments": {"file_path": "/repo/api.py"}}],
        result_extra={"is_error": False, "exit_code": 0, "truncated": False},
    )

    uncovered, _ = compute_uncovered_files(daydream_dir, "sess-clean-exit")

    assert "api.py" not in uncovered              # a clean result still credits


def test_a_damaged_batched_read_credits_none_of_its_files(tmp_path: Path) -> None:
    """A damaged batched read credits none of the files it looped over."""
    # Requirement 5 explicitly spans batched/loop reads, not just single Reads.
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-damaged-loop")
    _write_fork_calls(
        run_dir,
        "deep-python.json",
        [{
            "function_name": "shell",
            "arguments": {"command": 'for f in api.py notes.txt; do nl -ba "$f"; done'},
        }],
        result_extra={"truncated": True},
    )

    uncovered, _ = compute_uncovered_files(daydream_dir, "sess-damaged-loop")

    assert uncovered == ["api.py", "notes.txt"]


def test_declared_clean_is_downgraded_when_the_only_read_failed(tmp_path: Path) -> None:
    """A declared clean verdict is downgraded when its only read failed."""
    # Requirement 5's second consumer: verdict reconciliation must not let a
    # failed read rubber-stamp a declared clean verdict.
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-failed-verdict")
    _write_fork_calls(
        run_dir,
        "deep-python.json",
        [{"function_name": "Read", "arguments": {"file_path": "/repo/api.py"}}],
        result_extra={"is_error": True},
    )
    fork = load_trajectories(daydream_dir, "sess-failed-verdict")["forked"][0]

    verdicts = resolve_per_stack_verdicts(
        assigned_files=["api.py"],
        declared_verdicts=[{"path": "api.py", "lines_read": 20, "verdict": "clean"}],
        completed_read_paths=_completed_read_paths(fork),
        finding_files=set(),
    )

    assert verdicts[0]["verdict"] == "not_reviewed"


def test_filter_sweepable_files_from_index_with_patch_unreadable(tmp_path: Path) -> None:
    """The sweep budget derives from the persisted index, not ``diff.patch``.

    Materialize a ``.daydream`` dir with ONLY ``hunk-index.json`` (no
    ``diff.patch``) and confirm the sweep still sizes hunks from the index.
    """

    dd = tmp_path / ".daydream"
    dd.mkdir()
    diff = (
        "diff --git a/api.py b/api.py\n--- a/api.py\n+++ b/api.py\n"
        "@@ -1,1 +1,8 @@\n x\n+1\n+2\n+3\n+4\n+5\n+6\n+7\n"
    )
    write_hunk_index(dd, diff)
    idx = load_hunk_index(dd)
    swept, small, cap = filter_sweepable_files(
        ["api.py"], idx, min_hunk_lines=5, max_files=10
    )
    assert swept == ["api.py"]
    assert small == [] and cap == []


def test_filter_sweepable_files_caps_capacity_and_counts_small_hunks() -> None:
    """Small hunks are skipped; excess sweepable files are named, not dropped."""
    uncovered = ["api.py", "notes.txt"]

    swept, small_files, capacity_files = filter_sweepable_files(
        uncovered, parse_hunks(_DIFF), min_hunk_lines=5, max_files=10
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
        ["ghost.txt"], parse_hunks(_DIFF), min_hunk_lines=1, max_files=10
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
        ["notes.txt", "third.txt"], parse_hunks(diff), min_hunk_lines=5, max_files=1
    )

    assert swept == ["notes.txt"]
    assert small_files == []
    assert capacity_files == ["third.txt"]
    assert len(capacity_files) == 1


def test_filter_sweepable_files_zero_max_files_sweeps_nothing() -> None:
    """max_files=0 sweeps nothing; every sweepable file is capacity-skipped."""
    swept, small_files, capacity_files = filter_sweepable_files(
        ["api.py", "notes.txt"], parse_hunks(_DIFF), min_hunk_lines=1, max_files=0
    )

    assert swept == []
    assert small_files == []
    assert capacity_files == ["api.py", "notes.txt"]


def test_build_uncovered_sweep_prompt_includes_context_and_markers(tmp_path: Path) -> None:
    """The prompt names the file and references the live diff, intent and host output."""
    intent = tmp_path / ".daydream" / "deep" / "intent.md"
    output = tmp_path / ".daydream" / "deep" / "uncovered-0-review.md"
    prompt = build_uncovered_sweep_prompt(
        strategy=rp.build_default_profile().strategies["uncovered_review"].content,
        file="notes.txt",
        diff_path=tmp_path / "diff.patch",
        intent_path=intent,
        cwd=tmp_path,
        output_path=output,
    )

    assert "notes.txt" in prompt
    assert "uncovered file sweep" in prompt
    assert str(intent) in prompt
    assert str(output) in prompt
    # Diff contents stay out of the discovery prompt.
    assert "+line6" not in prompt
    assert str(tmp_path / "diff.patch") in prompt
    assert len(prompt.encode("utf-8")) < 32_768
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
    exploration = tmp_path / ".daydream" / "exploration"
    prompt = build_uncovered_sweep_prompt(
        strategy=rp.build_default_profile().strategies["uncovered_review"].content,
        file="notes.txt",
        diff_path=tmp_path / "diff.patch",
        intent_path=intent,
        cwd=tmp_path,
        output_path=output,
        exploration_dir=exploration,
    )

    assert "Read the pre-scan summary at" in prompt
    assert str(exploration) in prompt

    prompt_no_dir = build_uncovered_sweep_prompt(
        strategy=rp.build_default_profile().strategies["uncovered_review"].content,
        file="notes.txt",
        diff_path=tmp_path / "diff.patch",
        intent_path=intent,
        cwd=tmp_path,
        output_path=output,
        exploration_dir=None,
    )
    assert "Read the pre-scan summary at" not in prompt_no_dir
    assert str(exploration) not in prompt_no_dir


def test_coverage_receipt_records_inline_and_frontier(tmp_path: Path) -> None:
    """Issue #731: the deterministic coverage-receipts writer round-trips."""
    deep = tmp_path / ".daydream" / "deep"
    receipts = {"python#0": {"assigned_files": ["a.py"], "inline_files": ["a.py"],
                             "frontier_files": ["shared/iface.py"]}}
    write_coverage_receipts(deep, receipts)
    assert json.loads(coverage_receipt_path(deep).read_text()) == receipts


# --- Issue #731: coverage-evidence receipts gate the sweep ---


def test_inline_hunk_reviewed_evidence_covers_without_read(tmp_path: Path) -> None:
    """Issue #731: inline grounding + a finding reference covers without a read."""
    daydream_dir, _ = _seed_coverage_run(tmp_path, "sess-a", deep=True)
    # python#0 completed (records exist) and grounded api.py inline.
    deep = daydream_dir / "deep"
    write_coverage_receipts(deep, {"python#0": {"assigned_files": ["api.py"],
                                                "inline_files": ["api.py"], "frontier_files": []}})
    _write_records(deep, "python#0", {"issues": [_ONE_ISSUE]})
    receipts = json.loads(coverage_receipt_path(deep).read_text())
    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-a", receipts=receipts)
    assert "api.py" not in uncovered          # inline-hunk evidence -> not swept
    assert stats["coverage_by_evidence"]["inline_hunk_reviewed"] == 1


def test_legacy_records_bare_list_shape(tmp_path: Path) -> None:
    """Legacy bare-list records still fire inline evidence.

    The compatibility reader accepts the historical plain-list shape as well
    as the current wrapped record shape.
    """
    daydream_dir, _ = _seed_coverage_run(tmp_path, "sess-e", deep=True)
    deep = daydream_dir / "deep"
    write_coverage_receipts(deep, {"python#0": {"assigned_files": ["api.py"],
                                                "inline_files": ["api.py"], "frontier_files": []}})
    # Historical shape: a raw record list rather than a dict wrapper.
    _write_records(deep, "python#0", [_ONE_ISSUE])
    receipts = json.loads(coverage_receipt_path(deep).read_text())
    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-e", receipts=receipts)
    assert "api.py" not in uncovered          # bare-list records -> inline evidence fires
    assert stats["coverage_by_evidence"]["inline_hunk_reviewed"] == 1


def test_frontier_counted_once_per_type_across_shards(tmp_path: Path) -> None:
    """Issue #731: a file in N shards counts once per evidence type, not per shard.

    A hub file listed in several shards' ``frontier_files`` (and referenced by
    each shard's parsed findings) satisfies ``dependency_frontier_read`` once.
    """
    daydream_dir, _ = _seed_coverage_run(tmp_path, "sess-f", deep=True)
    deep = daydream_dir / "deep"
    receipt = {"assigned_files": [], "inline_files": [], "frontier_files": ["api.py"]}
    write_coverage_receipts(deep, {"python#0": dict(receipt), "python#1": dict(receipt),
                                   "python#2": dict(receipt)})
    for shard in ("python#0", "python#1", "python#2"):
        _write_records(deep, shard, [_ONE_ISSUE])
    receipts = json.loads(coverage_receipt_path(deep).read_text())
    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-f", receipts=receipts)
    assert "api.py" not in uncovered          # frontier evidence covers it
    assert stats["coverage_by_evidence"]["dependency_frontier_read"] == 1  # once per type


def test_coverage_by_evidence_absent_without_receipts(tmp_path: Path) -> None:
    """Issue #731: the evidence key is absent when receipts are not provided.

    ``coverage_by_evidence`` is a sharding-only surface: the Reads-only path
    (``receipts=None``) must preserve the existing stats artifact shape.
    """
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-g")
    _write_fork(run_dir, "deep-python.json", ["/repo/api.py"])

    _, stats = compute_uncovered_files(daydream_dir, "sess-g")

    assert "coverage_by_evidence" not in stats


def test_assignment_alone_never_counts(tmp_path: Path) -> None:
    """Issue #731: assignment alone is never coverage -- the file is swept."""
    daydream_dir, _ = _seed_coverage_run(tmp_path, "sess-b", deep=True)
    deep = daydream_dir / "deep"
    write_coverage_receipts(deep, {"python#0": {"assigned_files": ["api.py"],
                                                "inline_files": [], "frontier_files": []}})
    _write_records(deep, "python#0", {"issues": []})
    receipts = json.loads(coverage_receipt_path(deep).read_text())
    uncovered, _ = compute_uncovered_files(daydream_dir, "sess-b", receipts=receipts)
    assert "api.py" in uncovered              # assignment alone is never coverage


def test_incomplete_shard_receipt_does_not_cover(tmp_path: Path) -> None:
    """Issue #731: a receipt without a completed records file covers nothing."""
    daydream_dir, _ = _seed_coverage_run(tmp_path, "sess-c", deep=True)
    deep = daydream_dir / "deep"
    write_coverage_receipts(deep, {"python#0": {"assigned_files": ["api.py"],
                                                "inline_files": ["api.py"], "frontier_files": []}})
    # NO stack-python#0-records.json on purpose.
    receipts = json.loads(coverage_receipt_path(deep).read_text())
    uncovered, _ = compute_uncovered_files(daydream_dir, "sess-c", receipts=receipts)
    assert "api.py" in uncovered              # fail-open: missing completion -> swept


def test_omitted_assigned_file_is_still_swept(tmp_path: Path) -> None:
    """Issue #731: a grounded-but-omitted file is swept, never skipped.

    A completed shard grounded api.py + notes.txt inline, but its parsed
    findings only reference api.py (notes.txt omitted). notes.txt must be
    swept -- inline/frontier credit goes only to finding-referenced files.
    """
    daydream_dir, _ = _seed_coverage_run(tmp_path, "sess-d", deep=True)
    deep = daydream_dir / "deep"
    receipts = {"python#0": {"assigned_files": ["api.py", "notes.txt"],
                             "inline_files": ["api.py", "notes.txt"],
                             "frontier_files": []}}
    write_coverage_receipts(deep, receipts)
    _write_records(deep, "python#0", {"issues": [_ONE_ISSUE]})
    uncovered, _ = compute_uncovered_files(daydream_dir, "sess-d", receipts=receipts)
    assert "notes.txt" in uncovered    # omitted by the reviewer -> swept, never skipped
    assert "api.py" not in uncovered   # reviewed inline -> not swept


@pytest.mark.parametrize("tool", [
    {"function_name": "Bash", "arguments": {
        "command": "grep -n '^from|^import' /repo/api.py"}},
    {"function_name": "Grep", "arguments": {"pattern": "^from|^import", "path": "/repo/api.py"}},
])
def test_compute_uncovered_files_import_only_grep_does_not_cover(tmp_path: Path, tool: Any) -> None:
    """An import-only grep (Bash or Grep tool) covers nothing (issue #739 / AC2/AC3)."""
    daydream_dir, run_dir = _seed_coverage_run(tmp_path, "sess-grep")
    # Bash/Grep spellings (Issue #739) can't go through _write_fork, which is
    # hardcoded to Read/file_path; emit the raw calls for the live sweep.
    _write_fork_calls(run_dir, "deep-python.json", [tool])
    _write_fork(run_dir, "deep-generic.json", ["/repo/notes.txt"])

    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-grep")

    assert "api.py" in uncovered  # the import-only grep covers nothing
    assert stats["files_read_by_reviewers"] == 1  # only notes.txt via Read
    swept, _, _ = filter_sweepable_files(uncovered, parse_hunks(_DIFF), min_hunk_lines=1, max_files=10)
    assert "api.py" in swept  # the file is swept, never silently skipped


def test_resolve_per_stack_verdicts_downgrades_clean_without_read() -> None:
    """AC2: a clean verdict for a file with no completed read becomes not_reviewed."""

    declared = [
        {"path": "api.py", "lines_read": 10, "verdict": "clean"},
        {"path": "notes.txt", "lines_read": 5, "verdict": "has_findings"},
        {"path": "lib/util.py", "lines_read": 12, "verdict": "clean"},
    ]
    reads = {"/repo/api.py"}  # api.py read, notes.txt and lib/util.py not
    findings = {"notes.txt"}  # notes.txt has a finding
    out = resolve_per_stack_verdicts(
        assigned_files=["api.py", "notes.txt", "lib/util.py"],
        declared_verdicts=declared,
        completed_read_paths=reads,
        finding_files=findings,
    )
    by_path = {v["path"]: v["verdict"] for v in out}
    assert by_path["api.py"] == "clean"  # read + no finding -> clean stays
    assert by_path["notes.txt"] == "has_findings"  # finding beats read
    # declared clean + no read -> downgraded to not_reviewed
    assert by_path["lib/util.py"] == "not_reviewed"


@pytest.mark.parametrize(
    ("session", "path"),
    [("sess-clean", "api.py"), ("sess-sc", "./api.py")],
    ids=["clean", "dot-slash"],
)
def test_clean_verdict_covers_without_finding(tmp_path: Path, session: str, path: str) -> None:
    """Issue #740 AC1: a clean verdict (empty findings) covers the file for the sweep.

    A completed shard that read api.py and found nothing must still mark
    api.py covered -- a clean review is indistinguishable from an unreviewed
    file only under the old findings-only gate. A verdict path spelled with a
    leading ``./`` must credit the same file (the canonical strip).
    """
    daydream_dir, _ = _seed_coverage_run(tmp_path, session, deep=True)
    deep = daydream_dir / "deep"
    write_coverage_receipts(deep, {"python#0": {"assigned_files": ["api.py"],
                                                "inline_files": ["api.py"], "frontier_files": []}})
    # Clean review: EMPTY issues, evidence-gated clean verdict for api.py.
    _write_records(deep, "python#0", {
        "issues": [],
        "verdicts": [{"path": path, "lines_read": 30, "verdict": "clean", "n_findings": 0}],
    })
    receipts = json.loads(coverage_receipt_path(deep).read_text())
    uncovered, stats = compute_uncovered_files(daydream_dir, session, receipts=receipts)
    assert "api.py" not in uncovered              # clean verdict -> covered, not swept
    assert stats["coverage_by_evidence"]["inline_hunk_reviewed"] == 1


def test_not_reviewed_verdict_never_credits(tmp_path: Path) -> None:
    """Issue #740 AC6: an unread file (not_reviewed verdict) earns no coverage.

    The anti-confabulation gate (#742/#756): a verdict array present but
    resolving the file to not_reviewed must leave it swept, never credited.
    """
    daydream_dir, _ = _seed_coverage_run(tmp_path, "sess-nr", deep=True)
    deep = daydream_dir / "deep"
    write_coverage_receipts(deep, {"python#0": {"assigned_files": ["api.py"],
                                                "inline_files": ["api.py"], "frontier_files": []}})
    # Verdict present but NOT a pass: the file was never read.
    _write_records(deep, "python#0", {
        "issues": [],
        "verdicts": [{"path": "api.py", "lines_read": 0, "verdict": "not_reviewed", "n_findings": 0}],
    })
    receipts = json.loads(coverage_receipt_path(deep).read_text())
    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-nr", receipts=receipts)
    assert "api.py" in uncovered                  # not_reviewed -> swept, never credited
    assert stats["coverage_by_evidence"]["inline_hunk_reviewed"] == 0


# --- Issue #740 round-2: frontier credit decoupled from listing shard's records ---


def test_frontier_credited_when_lister_shard_lacks_records(tmp_path: Path) -> None:
    """Issue #740: a frontier file is credited when ANY completed shard read it.

    The cross-shard union rationale builds ``frontier_evidence`` from every
    completed shard's covered set because a frontier file's read evidence lives
    in a SIBLING shard, never the shard that merely lists it. Previously the
    credit pass ``continue``d the whole per-receipt loop when the LISTING shard
    had no records, suppressing ``dependency_frontier_read`` for files it
    merely named. Now the frontier branch runs against the sibling union
    independent of the lister's own record presence.
    """
    daydream_dir, _ = _seed_coverage_run(tmp_path, "sess-fl", deep=True)
    deep = daydream_dir / "deep"
    # python#0 merely LISTS api.py as a frontier and has NO records (incomplete
    # lister). python#1 actually read api.py (its records cover it) but its own
    # receipt lists no inline/frontier files, so only the sibling-union frontier
    # branch on python#0 can credit api.py.
    write_coverage_receipts(deep, {
        "python#0": {"assigned_files": [], "inline_files": [], "frontier_files": ["api.py"]},
        "python#1": {"assigned_files": [], "inline_files": [], "frontier_files": []},
    })
    # python#1 completed and read api.py -> union evidence covers it.
    _write_records(deep, "python#1", [_ONE_ISSUE])
    # python#0's records file deliberately absent.
    receipts = json.loads(coverage_receipt_path(deep).read_text())
    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-fl", receipts=receipts)
    assert "api.py" not in uncovered          # frontier evidence covers it
    assert stats["coverage_by_evidence"]["dependency_frontier_read"] == 1


def test_frontier_not_credited_without_any_sibling_evidence(tmp_path: Path) -> None:
    """Issue #740: frontier credit still requires real sibling evidence.

    Decoupling the frontier branch from the lister's records must not credit a
    frontier file when NO completed shard read it -- assignment/grounding alone
    never counts. With both shards' records absent, api.py stays swept.
    """
    daydream_dir, _ = _seed_coverage_run(tmp_path, "sess-fn", deep=True)
    deep = daydream_dir / "deep"
    write_coverage_receipts(deep, {
        "python#0": {"assigned_files": [], "inline_files": [], "frontier_files": ["api.py"]},
        "python#1": {"assigned_files": [], "inline_files": [], "frontier_files": []},
    })
    # No completed shard: frontier_evidence stays empty.
    receipts = json.loads(coverage_receipt_path(deep).read_text())
    uncovered, stats = compute_uncovered_files(daydream_dir, "sess-fn", receipts=receipts)
    assert "api.py" in uncovered              # no sibling evidence -> swept
    assert stats["coverage_by_evidence"].get("dependency_frontier_read", 0) == 0


# --- Issue #740 round-2: single canonical ./ strip ---


def test_strip_dot_slash_normalizes_once() -> None:
    """Issue #740: ``strip_dot_slash`` is the single canonical ``./`` strip."""

    assert strip_dot_slash("api.py") == "api.py"
    assert strip_dot_slash("./api.py") == "api.py"
    assert strip_dot_slash("./dir/x.py") == "dir/x.py"
    assert strip_dot_slash("a/b/c.py") == "a/b/c.py"


def test_uncovered_sweep_prompt_carries_severity_rubric(tmp_path: Path) -> None:
    """Issue #972 R1.1: the sweep reviewer assigns severities, so it gets the
    host severity rubric, appended after the profile strategy text."""

    intent = tmp_path / ".daydream" / "deep" / "intent.md"
    output = tmp_path / ".daydream" / "deep" / "uncovered-0-review.md"
    strategy = rp.build_default_profile().strategies["uncovered_review"].content
    prompt = build_uncovered_sweep_prompt(
        strategy=strategy,
        file="notes.txt",
        diff_path=tmp_path / "diff.patch",
        intent_path=intent,
        cwd=tmp_path,
        output_path=output,
    )
    assert severity.SEVERITY_RUBRIC in prompt
    assert prompt.index(severity.SEVERITY_RUBRIC) > prompt.index(strategy.format(file="notes.txt"))


def test_non_pi_sweep_excerpt_streams_past_large_lines(tmp_path: Path) -> None:
    path = tmp_path / "diff.patch"
    path.write_text("diff --git a/large.txt b/large.txt\n--- a/large.txt\n+++ b/large.txt\n"
                    "@@ -0,0 +1 @@\n+" + "x" * 3_690_129 + "\n" + _DIFF)
    excerpt = bounded_diff_block_for_file(path, "notes.txt")
    assert "+line6" in excerpt
    assert "large.txt" not in excerpt
    assert len(excerpt.encode()) < 12_288
    large = bounded_diff_block_for_file(path, "large.txt")
    assert "diff excerpt truncated" in large
    assert len(large.encode()) <= 12_288
    assert "unavailable" in bounded_diff_block_for_file(path, "absent.txt")
