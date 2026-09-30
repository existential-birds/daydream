"""End-to-end proof that a completed batched loop read stops the duplicate sweep.

Issue #1397: a reviewer that reads several files through one literal shell loop
has, in fact, read those files. The uncovered-file sweep must not dispatch a
redundant second pass for them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.eval.analyzer import analyze_coverage, load_trajectories
from daydream.runner import run
from tests.deep_orchestrator.support import (
    _batched_loop_sweep_target,
    _install_uncovered_sweep_stub,
)
from tests.test_deep_orchestrator import MakeConfig, Mute, _silence


async def test_run_deep_credits_a_completed_batched_loop_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """AC1/#1397: the loop-read files are not re-reviewed, the extra file is."""

    target = _batched_loop_sweep_target(tmp_path)
    _silence(monkeypatch)
    mute_side_effects()
    stub = _install_uncovered_sweep_stub(monkeypatch, target)
    stub.per_stack_read_command = (
        'for task_file in loop_one.py loop_two.py; do nl -ba "$task_file"; done'
    )
    stub.per_stack_loop_reads = frozenset({"loop_one.py", "loop_two.py"})
    stub.per_stack_unread = frozenset({"notes.txt"})
    stub.sweep_file = "notes.txt"

    exit_code = await run(make_config(target, assume="yes", output_mode="loop"))
    assert exit_code == 0

    stats = json.loads((target / ".daydream" / "deep" / "coverage-stats.json").read_text())
    pre_sweep = stats["pre_sweep"]
    assert pre_sweep["files_in_diff"] == 4
    # main.py via its own Read + loop_one.py / loop_two.py via the batched loop.
    assert pre_sweep["files_read_by_reviewers"] == 3
    assert pre_sweep["uncovered_files"] == ["notes.txt"]   # loop files are NOT re-reviewed
    assert stats["attempted_files"] == ["notes.txt"]       # exactly one sweep dispatch

    # The loop's read is visible to eval coverage analysis too (shared seam).
    post = analyze_coverage(load_trajectories(target / ".daydream"), target / ".daydream")
    assert "loop_one.py" not in post["uncovered_files"]
    assert "loop_two.py" not in post["uncovered_files"]
