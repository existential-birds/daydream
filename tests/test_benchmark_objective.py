"""Hermetic suite for the explicit-run ledger resolution (issue #888).

Task 1: ObjectiveRun, CompletedRun, and explicit-run ledger resolution.
Task 2: per-task reward parsing and failure classification.
Task 3: full compatibility-identity binding and the compile-lock cross-check.
Task 4: count-derived metrics equal the authoritative scoring.
"""
import json

import pytest

from daydream.benchmark.harbor import objective
from daydream.benchmark.harbor import run as run_mod

_WHEEL = {"distribution": "daydream", "version": "0.1.0", "sha256": "c" * 64}


def _seed_compiled_lock(ws, wheel=_WHEEL):
    """Write a compiled lock with the ``daydream`` wheel block + a case entry."""
    lock = {"schema_version": 1, "cases": {"case-a": {"key": "case-a"}}, "files": {},
            "daydream": wheel}
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps(lock))


def _ws(tmp_path):
    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    (ws / "harbor").mkdir()
    _seed_compiled_lock(ws)
    (ws / "harbor" / "harbor-job.yaml").write_text("jobs_dir: jobs\nn_attempts: 3\n")
    (ws / "benchmark.yaml").write_text(json.dumps({
        "schema_version": 1, "benchmark_id": "6c38dc0a-5f5a-4b73-bf36-9a2eb390f63b",
        "created_at": "2026-08-21T12:00:00Z", "source": {}, "privacy": {}, "pull_requests": [],
        "cases": []}))
    return ws


def _env(**over):
    """A trusted control-plane env; the default already pins the candidate digest."""
    base = {
        "DAYDREAM_JUDGE_PROVIDER": "anthropic",
        "DAYDREAM_JUDGE_MODEL": "m",
        "DAYDREAM_REVIEW_MODEL": "rm",
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "DAYDREAM_REVIEW_BASE_URL": "http://review.example",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST": "d" * 64,
    }
    base.update(over)
    return base


def _append(ws, run_id='run-1'):
    run_mod.ledger_append_running(
        ws, run_id=run_id,
        compiled_lock_sha256=run_mod._compiled_lock_sha256(ws),
        job_dir=str((ws / "harbor" / "jobs" / run_id).resolve()),
    )


def _complete_ws(tmp_path, run_id="run-1", trials=None):
    """A complete, consistent run whose ledger lock hash matches the seed lock."""
    ws = _ws(tmp_path)
    run_mod.ledger_append_running(
        ws, run_id=run_id,
        compiled_lock_sha256=run_mod._compiled_lock_sha256(ws),
        job_dir=str((ws / "harbor" / "jobs" / run_id).resolve()),
    )
    run_mod.ledger_mark(ws, run_id, state="complete")
    if trials is None:
        trials = [_reward(tp=2, fp=1, fn=0)]
    _seed_trials(ws, run_id, trials)
    return ws


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


def _rows_to_jsonl(rows, path):
    path.write_text("\n".join(("null" if r is None else json.dumps(r)) for r in rows) + "\n")
    return path


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


def test_objective_binds_full_compatibility_identity(tmp_path):
    ws = _complete_ws(tmp_path)
    run = objective.read_completed_run(ws, "run-1", env=_env())
    ident = run.identity
    assert ident.profile_digest == "d" * 64
    assert ident.reviewer_model == "rm"          # from _env
    assert ident.judge_model == "m"              # from _env
    assert ident.daydream_wheel_sha256 == "c" * 64
    assert ident.daydream_version == "0.1.0"
    assert ident.compiled_lock_sha256 == run_mod._compiled_lock_sha256(ws)
    assert ident.attempts == 3


def test_objective_rejects_identity_disagreement(tmp_path):
    ws = _ws(tmp_path)
    # ledger says lock hash "a"*64 but the on-disk lock hashes to something else
    run_mod.ledger_append_running(
        ws, run_id="run-1", compiled_lock_sha256="a" * 64,
        job_dir=str((ws / "harbor" / "jobs" / "run-1").resolve()),
    )
    run_mod.ledger_mark(ws, "run-1", state="complete")
    with pytest.raises(objective.ObjectiveError) as e:
        objective.read_completed_run(ws, "run-1", env={})
    assert "compiled_lock" in str(e.value).lower()


def test_objective_metrics_equal_verifier_core_and_metric_py(tmp_path):
    import types

    from daydream.benchmark.harbor import build, verifier_core

    def _compiled_metric():
        mod = types.ModuleType("metric")
        exec(compile(build.render_metric().decode("utf-8"), "metric.py", "exec"), mod.__dict__)
        return mod

    ws = _complete_ws(tmp_path, trials=[_reward(tp=2, fp=1, fn=0, reward=0.8),
                                        _reward(tp=1, fp=0, fn=1, reward=0.5), None])
    run = objective.read_completed_run(ws, "run-1", env={})
    flat = list(run.task_rows)
    expected = verifier_core.aggregate_metrics(flat)
    assert run.objective.tp == expected["total_tp"]
    assert run.objective.fp == expected["total_fp"]
    assert run.objective.fn == expected["total_fn"]
    assert run.objective.precision == expected["micro_precision"]
    assert run.objective.recall == expected["micro_recall"]
    assert run.objective.f1 == expected["micro_f1"]
    assert run.objective.task_count == expected["task_count"]
    assert run.objective.infra_error_task_count == expected["infra_error_task_count"]
    jsonl = _rows_to_jsonl(flat, tmp_path / "flat.jsonl")
    via_metric = _compiled_metric().aggregate_rewards_file(str(jsonl))
    assert run.objective._as_metric_dict() == via_metric


def test_objective_tokens_cost_absent_when_unrecorded(tmp_path):
    ws = _complete_ws(tmp_path)
    run = objective.read_completed_run(ws, "run-1", env={})
    assert run.objective.tokens is None and run.objective.cost is None
