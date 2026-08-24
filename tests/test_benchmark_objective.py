"""Hermetic suite for the explicit-run ledger resolution (issue #888).

Task 1: ObjectiveError, CompletedRun, and explicit-run ledger resolution.
Task 2: per-task reward parsing and failure classification.
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


def _reward(tp=0, fp=0, fn=0, reward=0.0, clean_task=0, clean_pass=0,
            gold_count=0, candidate_count=0, verifier_error=0):
    return {"reward": reward, "tp": tp, "fp": fp, "fn": fn,
            "precision": (tp/(tp+fp)) if (tp+fp) else 1.0,
            "recall": (tp/(tp+fn)) if (tp+fn) else 1.0,
            "f1": 0.0, "gold_count": gold_count, "candidate_count": candidate_count,
            "clean_task": clean_task, "clean_pass": clean_pass,
            "verifier_error": verifier_error}


def _seed_trials(ws, run_id, trials):
    job = ws / "harbor" / "jobs" / run_id
    for i, row in enumerate(trials):
        trial = job / f"case-{i}" / "verifier"
        trial.mkdir(parents=True)
        if row is None:
            (trial / "reward-details.json").write_text("{}")   # unscored infra path
        else:
            (trial / "reward.json").write_text(json.dumps(row))


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


def test_objective_parses_scored_and_infra_trials(tmp_path):
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    _seed_trials(ws, run_id, [_reward(tp=2, fp=1, fn=0), None])
    run = objective.read_completed_run(ws, run_id, env={})
    assert run.task_rows[0]["tp"] == 2 and run.task_rows[1] is None
    assert run.objective.infra_error_task_count == 1
    assert run.objective.scored_task_count == 1


def test_objective_clean_task_is_not_infra_failure(tmp_path):
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    _seed_trials(ws, run_id, [_reward(clean_task=1, clean_pass=1)])
    run = objective.read_completed_run(ws, run_id, env={})
    assert run.objective.clean_task_count == 1
    assert run.objective.infra_error_task_count == 0


def test_objective_malformed_numeric_fails_closed(tmp_path):
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    _seed_trials(ws, run_id, [{"reward": 0.5, "tp": "not-an-int", "fp": 0, "fn": 0}])
    with pytest.raises(objective.ObjectiveError) as e:
        objective.read_completed_run(ws, run_id, env={})
    assert "run-1" in str(e.value)
