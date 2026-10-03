"""Hermetic suite for the explicit-run ledger resolution (issue #888).

Task 1: ObjectiveRun, CompletedRun, and explicit-run ledger resolution.
Task 2: per-task reward parsing and failure classification.
Task 3: full compatibility-identity binding and the compile-lock cross-check.
Task 4: count-derived metrics equal the authoritative scoring.
"""
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark import storage
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


def _append(ws: Path, run_id: Any='run-1') -> None:
    run_mod.ledger_append_running(
        ws, run_id=run_id, compiled_lock_sha256=storage.sha256_file(ws / "harbor" / "benchmark.lock.json"),
        job_dir=str((ws / "harbor" / "jobs" / run_id).resolve()),
    )


def _complete_ws(tmp_path: Path, run_id: Any="run-1", trials: Any=None) -> Any:
    """A complete, consistent run whose ledger lock hash matches the seed lock."""
    ws = _ws(tmp_path)
    run_mod.ledger_append_running(
        ws, run_id=run_id, compiled_lock_sha256=storage.sha256_file(ws / "harbor" / "benchmark.lock.json"),
        job_dir=str((ws / "harbor" / "jobs" / run_id).resolve()),
    )
    run_mod.ledger_mark(ws, run_id, state="complete")
    if trials is None:
        trials = [_reward(tp=2, fp=1, fn=0)]
    _seed_trials(ws, run_id, trials)
    return ws


def _reward(tp: Any=0, fp: Any=0, fn: Any=0, reward: Any=0.0, clean_task: Any=0, clean_pass: Any=0, gold_count: Any=0,
    candidate_count: Any=0, verifier_error: Any=0, location_present: Any=0, location_exact: Any=0, location_near: Any=0,
    location_file: Any=0, location_miss: Any=0, location_credit: Any=0.0, severity_present: Any=0,
    severity_pairs: Any=0, severity_exact: Any=0, severity_within_1: Any=0, severity_mean_distance: Any=0.0,
    severity_credit: Any=0.0,
) -> dict[str, Any]:
    return {"reward": reward, "tp": tp, "fp": fp, "fn": fn,
            "precision": (tp/(tp+fp)) if (tp+fp) else 1.0, "recall": (tp/(tp+fn)) if (tp+fn) else 1.0,
            "f1": 0.0, "gold_count": gold_count, "candidate_count": candidate_count,
            "clean_task": clean_task, "clean_pass": clean_pass, "verifier_error": verifier_error,
            "location_present": location_present, "location_exact": location_exact, "location_near": location_near,
            "location_file": location_file, "location_miss": location_miss, "location_credit": location_credit,
            "severity_present": severity_present, "severity_pairs": severity_pairs, "severity_exact": severity_exact,
            "severity_within_1": severity_within_1, "severity_mean_distance": severity_mean_distance,
            "severity_credit": severity_credit}


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
    assert run.objective.metrics['infra_error_task_count'] == 1
    assert run.objective.metrics['scored_task_count'] == 1

def test_objective_clean_task_is_not_infra_failure(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    _seed_trials(ws, run_id, [_reward(clean_task=1, clean_pass=1)])
    run = objective.read_completed_run(ws, run_id, env={})
    assert run.objective is not None
    assert run.objective.metrics['clean_task_count'] == 1
    assert run.objective.metrics['infra_error_task_count'] == 0

def test_objective_malformed_numeric_fails_closed(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    run_id = "run-1"
    _append(ws, run_id)
    run_mod.ledger_mark(ws, run_id, state="complete")
    _seed_trials(ws, run_id, [{"reward": 0.5, "tp": "not-an-int", "fp": 0, "fn": 0}])
    with pytest.raises(objective.ObjectiveError) as e:
        objective.read_completed_run(ws, run_id, env={})
    assert "run-1" in str(e.value)

def test_objective_binds_full_compatibility_identity(tmp_path: Path) -> None:
    ws = _complete_ws(tmp_path)
    run = objective.read_completed_run(ws, "run-1", env=_env())
    assert run.identity is not None
    ident = run.identity
    assert ident.profile_digest == "d" * 64
    assert ident.reviewer_model == "rm"          # from _env
    assert ident.judge_model == "m"              # from _env
    assert ident.daydream_wheel_sha256 == "c" * 64
    assert ident.daydream_version == "0.1.0"
    assert ident.compiled_lock_sha256 == storage.sha256_file(ws / "harbor" / "benchmark.lock.json")
    assert ident.attempts == 3

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

def test_objective_metrics_equal_verifier_core(tmp_path: Path) -> None:
    ws = _complete_ws(tmp_path, trials=[_reward(tp=2, fp=1, fn=0, reward=0.8),
                                        _reward(tp=1, fp=0, fn=1, reward=0.5), None])
    run = objective.read_completed_run(ws, "run-1", env={})
    flat = list(run.task_rows)
    assert run.objective is not None
    expected = verifier_core.aggregate_metrics(flat)
    assert run.objective.metrics['total_tp'] == expected["total_tp"]
    assert run.objective.metrics['total_fp'] == expected["total_fp"]
    assert run.objective.metrics['total_fn'] == expected["total_fn"]
    assert run.objective.metrics['micro_precision'] == expected["micro_precision"]
    assert run.objective.metrics['micro_recall'] == expected["micro_recall"]
    assert run.objective.metrics['micro_f1'] == expected["micro_f1"]
    assert run.objective.metrics['task_count'] == expected["task_count"]
    assert run.objective.metrics['infra_error_task_count'] == expected["infra_error_task_count"]

def test_objective_tokens_cost_absent_when_unrecorded(tmp_path: Path) -> None:
    ws = _complete_ws(tmp_path)
    run = objective.read_completed_run(ws, "run-1", env={})
    assert run.objective is not None
    assert run.objective.tokens is None and run.objective.cost is None

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
    run_mod.ledger_append_running(
        ws, run_id=run_id, compiled_lock_sha256=storage.sha256_file(ws / "harbor" / "benchmark.lock.json"),
        job_dir=str((ws / "harbor" / "jobs" / run_id).resolve()), profile_digest=digest,
    )
    run_mod.ledger_mark(ws, run_id, state="complete")
    _seed_trials(ws, run_id, trials)
    return ws


def test_aggregate_suite_pools_micro_metrics_and_never_mean(tmp_path: Path) -> None:
    a = _complete_ws_at(tmp_path, "a", "r1", [_reward(tp=1, fp=1, fn=0)])
    b = _complete_ws_at(tmp_path, "b", "r2", [_reward(tp=1, fp=3, fn=0)])
    manifest = {"schema_version": 1, "entries": [
        {"workspace": str(a), "run_id": "r1"}, {"workspace": str(b), "run_id": "r2"},
    ]}
    suite = objective.aggregate_suite(manifest, env={})
    assert suite.objective.metrics['micro_precision'] == pytest.approx(2 / 6)  # pooled, not mean
    assert suite.objective.metrics['micro_precision'] != pytest.approx((0.5 + 0.25) / 2)
    a_rows = objective.read_completed_run(a, "r1", env={}).task_rows
    b_rows = objective.read_completed_run(b, "r2", env={}).task_rows
    flat = a_rows + b_rows   # flattened per-task dicts across both runs
    assert dict(suite.objective.metrics) == verifier_core.aggregate_metrics(flat)

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
                        [_reward(tp=2, fp=1, fn=0, reward=0.8), _reward(clean_task=1, clean_pass=1)])
    b = _complete_ws_at(tmp_path, "b", "r2", [_reward(tp=1, fp=0, fn=1, reward=0.5)])
    manifest = {"schema_version": 1, "entries": [
        {"workspace": str(a), "run_id": "r1"}, {"workspace": str(b), "run_id": "r2"}]}
    suite = objective.aggregate_suite(manifest, env={})
    a_rows = objective.read_completed_run(a, "r1", env={}).task_rows
    b_rows = objective.read_completed_run(b, "r2", env={}).task_rows
    flat = a_rows + b_rows   # the exact flattened per-task rows across both runs
    assert dict(suite.objective.metrics) == verifier_core.aggregate_metrics(flat)

def test_suite_manifest_validation(tmp_path: Path) -> None:
    good = {"schema_version": 1, "entries": [
        {"workspace": str(tmp_path / "a"), "run_id": "r1"}, {"workspace": str(tmp_path / "b"), "run_id": "r2"},
    ]}
    entries = objective.validate_suite_manifest(good)
    assert [ (e.workspace.name, e.run_id) for e in entries ] == [("a", "r1"), ("b", "r2")]

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

def test_identity_to_dict_is_single_source_for_all_projections(tmp_path: Path) -> None:
    ws = _complete_ws(tmp_path)
    run = objective.read_completed_run(ws, "run-1", env=_env())
    assert run.identity is not None

    per_run = objective.objective_to_json(run)["identity"]
    compat = objective.identity_to_dict(run.identity)
    assert per_run == compat == objective.identity_to_dict(run.identity)

    # The suite aggregate identity for the same workspace must match too.
    manifest = {"schema_version": 1, "entries": [{"workspace": str(ws), "run_id": "run-1"}]}
    suite = objective.aggregate_suite(manifest, env=_env())
    assert suite.identity == run.identity
    assert objective.identity_to_dict(suite.identity) == per_run

def test_objective_metric_dict_includes_axis_keys() -> None:
    # Shape parity (P-NUMERIC-ROW): the objective projection must carry the
    # exact key set the authoritative aggregate_metrics returns.
    pooled = objective._build_objective([], 0, Path("@fixture"), "empty")
    assert set(pooled.metrics) == set(verifier_core.aggregate_metrics([]))
    with pytest.raises(TypeError):
        pooled.metrics["micro_f1"] = 0.0  # type: ignore[index]


def test_objective_json_carries_reported_axes(tmp_path: Path) -> None:
    ws = _complete_ws(tmp_path,
        trials=[_reward(tp=2, fp=1, fn=0, reward=0.8, location_present=1, location_exact=2, location_near=1,
            location_file=1, location_miss=0, location_credit=0.75,
            severity_present=1, severity_pairs=4, severity_exact=3, severity_within_1=1, severity_mean_distance=0.25,
            severity_credit=0.875,
        )],
    )
    run = objective.read_completed_run(ws, "run-1", env={})
    assert run.objective is not None
    blob = objective.objective_to_json(run)["objective"]
    assert isinstance(blob, dict)
    metric = dict(run.objective.metrics)
    axis_keys = ["location_pairs_scored", "severity_pairs_scored",
        "location_exact", "location_near", "location_file", "location_miss",
        "total_location_exact", "total_location_near", "total_location_file", "total_location_miss",
        "severity_exact", "severity_within_1", "total_severity_exact", "total_severity_within_1",
        "location_exact_rate", "location_near_rate", "location_file_rate", "location_miss_rate",
        "severity_exact_rate", "severity_within_1_rate", "severity_mean_distance", "severity_credit", "location_credit",
    ]
    for key in axis_keys:
        assert key in blob, f"objective JSON drops {key!r}"
        assert blob[key] == metric[key]
        assert blob[key] == run.objective.metrics[key]
    assert len(axis_keys) == 23


def test_objective_captures_metrics_before_source_mutation() -> None:
    metrics = verifier_core.aggregate_metrics([])
    admitted = objective.Objective(
        metrics=metrics, clean_pass_count=0, candidate_count=0, gold_count=0,
        verifier_error_task_count=0, malformed_task_count=0, failed_task_count=0,
        comparison_eligible=True,
    )
    metrics["micro_f1"] = 0.25
    assert admitted.metrics["micro_f1"] == 1.0
    with pytest.raises(TypeError):
        admitted.metrics["micro_f1"] = 0.5  # type: ignore[index]


@pytest.mark.parametrize("command", ["objective", "aggregate"])
@pytest.mark.parametrize("destination", ["-", "file"])
def test_native_objective_cli_emits_exact_metrics_and_pure_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], command: str, destination: str,
) -> None:
    from daydream.cli import main

    for name, value in _env().items():
        monkeypatch.setenv(name, value)
    ws = _complete_ws(tmp_path, trials=[_reward(tp=2, fp=1), _reward(fn=1)])
    original = {path.relative_to(ws): path.read_bytes() for path in ws.rglob("*") if path.is_file()}
    if command == "objective":
        argv = ["benchmark", command, str(ws), "--run-id", "run-1"]
    else:
        manifest = tmp_path / "suite.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "entries": [{"workspace": str(ws), "run_id": "run-1"}],
        }))
        argv = ["benchmark", command, str(manifest)]
    output = tmp_path / "objective.json"
    target = "-" if destination == "-" else str(output)
    with pytest.raises(SystemExit) as exit_info:
        main([*argv, "--json", target])
    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    blob = json.loads(captured.out if destination == "-" else output.read_text())
    metrics = blob["objective"]
    assert metrics["task_count"] == metrics["scored_task_count"] == 2
    if command == "objective":
        assert metrics["tp"] == 2 and metrics["fp"] == metrics["fn"] == 1
        assert metrics["precision"] == metrics["recall"] == metrics["f1"] == pytest.approx(2 / 3)
        assert not {"total_tp", "micro_f1", "metrics", "tokens", "cost"} & metrics.keys()
    else:
        assert metrics["total_tp"] == 2 and metrics["total_fp"] == metrics["total_fn"] == 1
        assert metrics["micro_precision"] == metrics["micro_recall"] == metrics["micro_f1"] == pytest.approx(2 / 3)
        assert "tp" not in metrics and "f1" not in metrics
    human = captured.err if destination == "-" else captured.out
    assert "micro_f1=0.6667" in human
    assert str(ws) not in json.dumps(blob)
    assert {path.relative_to(ws): path.read_bytes() for path in ws.rglob("*") if path.is_file()} == original


@pytest.mark.parametrize("command", ["objective", "aggregate"])
def test_native_objective_cli_failed_admission_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], command: str,
) -> None:
    from daydream.cli import main

    for name, value in _env().items():
        monkeypatch.setenv(name, value)
    ws = _complete_ws(tmp_path)
    output = tmp_path / "objective.json"
    output.write_bytes(b"existing approved result\n")
    if command == "objective":
        argv = ["benchmark", command, str(ws), "--run-id", "missing"]
    else:
        manifest = tmp_path / "suite.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "entries": [
                {"workspace": str(ws), "run_id": "run-1"},
                {"workspace": str(ws), "run_id": "missing"},
            ],
        }))
        argv = ["benchmark", command, str(manifest)]
    with pytest.raises(SystemExit) as exit_info:
        main([*argv, "--json", str(output)])
    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing" in captured.err
    assert output.read_bytes() == b"existing approved result\n"


def test_completed_identity_uses_the_lock_bytes_admitted_by_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _complete_ws(tmp_path)
    lock_path = ws / "harbor" / "benchmark.lock.json"
    admitted = lock_path.read_bytes()
    replacement = json.loads(admitted)
    replacement["daydream"] = {"version": "replacement", "sha256": "b" * 64}
    read_bytes = Path.read_bytes

    def replace_after_capture(path: Path) -> bytes:
        payload = read_bytes(path)
        if path == lock_path:
            lock_path.write_text(json.dumps(replacement))
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_capture)
    result = objective.read_completed_run(ws, "run-1", env=_env())
    assert result.identity is not None
    assert result.identity.daydream_wheel_sha256 == "c" * 64
    assert result.identity.daydream_version == "0.1.0"
    assert result.identity.compiled_lock_sha256 == hashlib.sha256(admitted).hexdigest()
