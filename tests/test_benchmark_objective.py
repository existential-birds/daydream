"""Hermetic suite for the explicit-run ledger resolution (issue #888).

Task 1: ObjectiveError, CompletedRun, and explicit-run ledger resolution.
"""
import json

import pytest

from daydream.benchmark.harbor import objective
from daydream.benchmark.harbor import run as run_mod


def _ws(tmp_path):
    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    (ws / "harbor").mkdir()
    (ws / "harbor" / "benchmark.lock.json").write_text(
        json.dumps({"schema_version": 1, "cases": {}})
    )
    (ws / "benchmark.yaml").write_text(json.dumps({
        "schema_version": 1, "benchmark_id": "6c38dc0a-5f5a-4b73-bf36-9a2eb390f63b",
        "created_at": "2026-08-21T12:00:00Z", "source": {}, "privacy": {}, "pull_requests": [],
        "cases": []}))
    return ws


def _append(ws, run_id='run-1'):
    run_mod.ledger_append_running(
        ws, run_id=run_id, compiled_lock_sha256="a" * 64,
        job_dir=str((ws / "harbor" / "jobs" / run_id).resolve()),
    )


def test_objective_resolves_complete_run_by_explicit_run_id(tmp_path):
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    run = objective.read_completed_run(ws, run_id, env={})
    assert run.run_id == run_id
    assert run.state == "complete"


def test_objective_rejects_missing_and_nonterminal_runs(tmp_path):
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="running")
    for bad in ("running", "missing-run"):
        with pytest.raises(objective.ObjectiveError) as e:
            objective.read_completed_run(ws, bad, env={})
        assert bad in str(e.value) and str(ws) in str(e.value)
