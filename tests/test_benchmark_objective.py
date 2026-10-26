"""Hermetic suite for the explicit-run ledger resolution (issue #888).

Task 1: ObjectiveRun, CompletedRun, and explicit-run ledger resolution.
Task 2: per-task reward parsing and failure classification.
Task 3: full compatibility-identity binding and the compile-lock cross-check.
Task 4: count-derived metrics equal the authoritative scoring.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.harbor import objective, run as run_mod, verifier_core
from tests.test_benchmark_run import _seed_compiled_lock

harbor = pytest.importorskip("harbor", reason="harbor is an optional benchmark extra")

_WHEEL = {"distribution": "daydream", "version": "0.1.0", "sha256": "c" * 64}


def _ws(tmp_path: Path) -> Any:
    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    (ws / "harbor").mkdir()
    _seed_compiled_lock(ws)
    (ws / "harbor" / "harbor-job.yaml").write_text("jobs_dir: jobs\nn_attempts: 3\n")
    (ws / "benchmark.yaml").write_text(json.dumps({
        "schema_version": 1, "benchmark_id": "6c38dc0a-5f5a-4b73-bf36-9a2eb390f63b",
        "created_at": "2026-08-21T12:00:00Z", "source": {}, "privacy": {}, "pull_requests": [], "cases": []}))
    return ws


def _env(**over: Any) -> Any:
    """A trusted control-plane env; the default already pins the candidate digest."""
    base = {"DAYDREAM_JUDGE_PROVIDER": "anthropic", "DAYDREAM_JUDGE_MODEL": "m", "DAYDREAM_REVIEW_MODEL": "rm",
        "DAYDREAM_REVIEW_BACKEND": "claude", "DAYDREAM_REVIEW_BASE_URL": "http://review.example",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST": "d" * 64,
    }
    base.update(over)
    return base


def _append(ws: Path, run_id: Any='run-1', profile_digest: Any=None) -> None:
    run_mod.ledger_append_running(ws, run_id=run_id, compiled_lock_sha256=run_mod._compiled_lock_sha256(ws),
        job_dir=str((ws / "harbor" / "jobs" / run_id).resolve()), profile_digest=profile_digest,
    )


def _complete_ws(tmp_path: Path, run_id: Any="run-1", trials: Any=None) -> Any:
    """A complete, consistent run whose ledger lock hash matches the seed lock."""
    ws = _ws(tmp_path)
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    if trials is None:
        trials = [_reward(tp=2, fp=1, fn=0)]
    _seed_trials(ws, run_id, trials)
    return ws


def _reward(**fields: Any) -> dict[str, Any]:
    """Build a canonical verifier reward row, defaulting precision/recall from counts."""
    tp, fp, fn = fields.get("tp", 0), fields.get("fp", 0), fields.get("fn", 0)
    fields.setdefault("precision", (tp / (tp + fp)) if (tp + fp) else 1.0)
    fields.setdefault("recall", (tp / (tp + fn)) if (tp + fn) else 1.0)
    return verifier_core.Reward(**fields).to_dict()


def _seed_trials(ws: Path, run_id: Any, trials: Any) -> None:
    job = ws / "harbor" / "jobs" / run_id
    for i, row in enumerate(trials):
        trial = job / f"case-{i}" / "verifier"
        trial.mkdir(parents=True)
        if row is None:
            (trial / "reward-details.json").write_text("{}")   # unscored infra path
        else:
            (trial / "reward.json").write_text(json.dumps(row))


def test_objective_resolves_complete_run_by_explicit_run_id(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    run = objective.read_completed_run(ws, run_id, env={})
    assert run.run_id == run_id
    assert run.state == "complete"

def test_objective_rejects_missing_and_nonterminal_runs(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="running")
    for bad in ("running", "missing-run"):
        with pytest.raises(objective.ObjectiveError) as e:
            objective.read_completed_run(ws, bad, env={})
        assert bad in str(e.value) and str(ws) in str(e.value)

def test_objective_parses_scored_and_infra_trials(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    _seed_trials(ws, run_id, [_reward(tp=2, fp=1, fn=0), None])
    run = objective.read_completed_run(ws, run_id, env={})
    first = run.task_rows[0]
    assert first is not None and run.task_rows[1] is None
    assert first["tp"] == 2
    assert run.objective is not None
    assert run.objective.infra_error_task_count == 1
    assert run.objective.scored_task_count == 1
    assert run.objective._as_metric_dict() == verifier_core.aggregate_metrics(run.task_rows)

def test_objective_malformed_numeric_fails_closed(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    _seed_trials(ws, run_id, [{"reward": 0.5, "tp": "not-an-int", "fp": 0, "fn": 0}])
    with pytest.raises(objective.ObjectiveError) as e:
        objective.read_completed_run(ws, run_id, env={})
    assert "run-1" in str(e.value)

def test_objective_rejects_identity_disagreement(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    # ledger says lock hash "a"*64 but the on-disk lock hashes to something else
    run_mod.ledger_append_running(
        ws, run_id="run-1", compiled_lock_sha256="a" * 64, job_dir=str((ws / "harbor" / "jobs" / "run-1").resolve()),
    )
    run_mod.ledger_mark(ws, "run-1", state="complete")
    with pytest.raises(objective.ObjectiveError) as e:
        objective.read_completed_run(ws, "run-1", env={})
    assert "compiled_lock" in str(e.value).lower()

def test_objective_json_is_opaque(tmp_path: Path) -> None:
    ws = _complete_ws(tmp_path)
    # seed a private repo slug + gold/judge text into the workspace artifacts
    (ws / "benchmark.yaml").write_text(json.dumps(
        {"schema_version": 1, "source": {"repository": "acme/private-repo"}, "cases": [], "pull_requests": []}))
    job = ws / "harbor" / "jobs" / "run-1" / "case-0" / "verifier"
    (job / "reward-details.json").write_text(
        json.dumps({"reasoning": "the gold finding in private-repo", "path": "/src/x.py"}))
    run = objective.read_completed_run(ws, "run-1", env={})
    blob = json.dumps(objective.objective_to_json(run))
    for forbidden in ("acme", "private-repo", "/src/", "reasoning"):
        assert forbidden not in blob
    doc = json.loads(blob)
    assert set(doc) <= {"run_id", "mode", "schema_version", "identity", "objective"}
    assert isinstance(doc["objective"], dict)


def _complete_ws_at(tmp_path: Path, name: Any, run_id: Any, trials: Any, digest: Any="d" * 64, wheel: Any=None) -> Any:
    """A complete, consistent run under a repository workspace of the given name.

    The ledger stores the on-disk compiled-lock hash and the supplied ``digest``
    as the canonical review-profile digest, so ``read_completed_run`` resolves
    the run and binds a differing profile digest per caller-supplied value.
    """
    ws = _ws(tmp_path / name)
    _seed_compiled_lock(ws, wheel=wheel or _WHEEL)
    _append(ws, run_id, profile_digest=digest)
    run_mod.ledger_mark(ws, run_id, state="complete")
    _seed_trials(ws, run_id, trials)
    return ws


def test_aggregate_suite_experiment_id_stable_under_reorder_rejects_dup(tmp_path: Path) -> None:
    a = _complete_ws_at(tmp_path, "a", "r1", [_reward(tp=1, fp=0, fn=0)])
    b = _complete_ws_at(tmp_path, "b", "r2", [_reward(tp=1, fp=0, fn=0)])
    m1 = {"schema_version": 1, "entries": [
        {"workspace": str(a), "run_id": "r1"}, {"workspace": str(b), "run_id": "r2"}]}
    m2 = {"schema_version": 1, "entries": [
        {"workspace": str(b), "run_id": "r2"}, {"workspace": str(a), "run_id": "r1"}]}
    s1 = objective.aggregate_suite(m1, env={})
    s2 = objective.aggregate_suite(m2, env={})
    assert s1.experiment_id == s2.experiment_id
    assert s1.objective == s2.objective
    dup = {"schema_version": 1, "entries": [
        {"workspace": str(a), "run_id": "r1"}, {"workspace": str(a), "run_id": "r1"}]}
    with pytest.raises(objective.ObjectiveError):
        objective.aggregate_suite(dup, env={})

def test_aggregate_suite_fails_closed_on_incompatible_identity(tmp_path: Path) -> None:
    a = _complete_ws_at(tmp_path, "a", "r1", [_reward(tp=1, fp=0, fn=0)], digest="d" * 64)
    b = _complete_ws_at(tmp_path, "b", "r2", [_reward(tp=1, fp=0, fn=0)], digest="e" * 64)
    manifest = {"schema_version": 1, "entries": [
        {"workspace": str(a), "run_id": "r1"}, {"workspace": str(b), "run_id": "r2"}]}
    with pytest.raises(objective.ObjectiveError) as e:
        objective.aggregate_suite(manifest, env={})
    assert "profile_digest" in str(e.value)

def test_suite_pooled_output_equals_authoritative_scoring_end_to_end(tmp_path: Path) -> None:
    # Two compatible workspaces with a realistic, comparison-eligible mix: scored
    # and gold-free clean tasks (an infra-error ``None`` trial would make the entry
    # comparison-ineligible and ``aggregate_suite`` correctly refuses to pool it).
    a = _complete_ws_at(tmp_path, "a", "r1",
                        [_reward(tp=2, fp=1, fn=0, reward=0.8, location_present=1, location_exact=2, location_near=1,
                            location_file=1, location_miss=0, location_credit=0.75,
                            severity_present=1, severity_pairs=4,
                            severity_exact=3, severity_within_1=1, severity_mean_distance=0.25, severity_credit=0.875),
                         _reward(clean_task=1, clean_pass=1)])
    b = _complete_ws_at(tmp_path, "b", "r2", [_reward(tp=1, fp=0, fn=1, reward=0.5)])
    manifest = {"schema_version": 1, "entries": [
        {"workspace": str(a), "run_id": "r1"}, {"workspace": str(b), "run_id": "r2"}]}
    assert [(entry.workspace, entry.run_id) for entry in objective.validate_suite_manifest(manifest)] == [
        (a, "r1"), (b, "r2"),
    ]
    suite = objective.aggregate_suite(manifest, env=_env())
    first = objective.read_completed_run(a, "r1", env=_env())
    second = objective.read_completed_run(b, "r2", env=_env())
    assert first.run_id == "r1" and first.state == "complete"
    assert suite.objective._as_metric_dict() == verifier_core.aggregate_metrics(first.task_rows + second.task_rows)
    assert (suite.objective.tp, suite.objective.fp, suite.objective.fn) == (3, 1, 1)
    assert suite.objective.precision == pytest.approx(3 / 4)
    assert suite.objective.precision != pytest.approx((2 / 3 + 1) / 2)
    assert (suite.objective.task_count, suite.objective.scored_task_count,
            suite.objective.infra_error_task_count) == (3, 3, 0)
    assert (suite.objective.clean_task_count, suite.objective.clean_pass_count) == (1, 1)
    assert first.objective is not None and first.objective.infra_error_task_count == 0
    assert first.objective.tokens is None and first.objective.cost is None
    assert first.identity == second.identity == suite.identity
    identity = objective.identity_to_dict(suite.identity)
    assert objective.objective_to_json(first)["identity"] == identity
    assert (identity["profile_digest"], identity["reviewer_model"], identity["judge_model"], identity["attempts"]) == (
        "d" * 64, "rm", "m", 3,
    )
    assert (identity["daydream_version"], identity["daydream_wheel_sha256"], identity["compiled_lock_sha256"]) == (
        "0.1.0", "c" * 64, run_mod._compiled_lock_sha256(a),
    )
    exported = objective.objective_to_json(first)["objective"]
    assert isinstance(exported, dict) and "tokens" not in exported and "cost" not in exported
    axes = {
        "location_pairs_scored": 4, "severity_pairs_scored": 4,
        "location_exact": 2, "location_near": 1, "location_file": 1, "location_miss": 0,
        "total_location_exact": 2, "total_location_near": 1, "total_location_file": 1, "total_location_miss": 0,
        "severity_exact": 3, "severity_within_1": 1, "total_severity_exact": 3, "total_severity_within_1": 1,
        "location_exact_rate": 0.5, "location_near_rate": 0.25, "location_file_rate": 0.25, "location_miss_rate": 0.0,
        "severity_exact_rate": 0.75, "severity_within_1_rate": 0.25, "severity_mean_distance": 0.25,
        "severity_credit": 0.875, "location_credit": 0.75,
    }
    assert {key: exported[key] for key in axes} == axes

def test_suite_manifest_rejects_duplicate_and_incomplete(tmp_path: Path) -> None:
    dup = {"schema_version": 1, "entries": [
        {"workspace": str(tmp_path / "a"), "run_id": "r1"}, {"workspace": str(tmp_path / "a"), "run_id": "r1"},
    ]}
    with pytest.raises(objective.ObjectiveError) as e:
        objective.validate_suite_manifest(dup)
    assert "duplicate" in str(e.value).lower()
    incomplete = {"schema_version": 1, "entries": [{"workspace": str(tmp_path / "a")}]}
    with pytest.raises(objective.ObjectiveError):
        objective.validate_suite_manifest(incomplete)
    unsupported = {"schema_version": 99, "entries": []}
    with pytest.raises(objective.ObjectiveError):
        objective.validate_suite_manifest(unsupported)
