"""Hermetic suite for the `daydream benchmark clean` command (issue #782).

The deletion set is strictly ledger-driven and containment-checked: every
hermetic test stubs the Docker-removal default so CI never shells out to a
real Docker/network call (the real ``docker rmi`` path is issue #13).
"""
import json
from pathlib import Path

import pytest

import daydream.benchmark.harbor.clean as clean_mod
from daydream.benchmark.cli import _build_benchmark_parser, _handle_benchmark_command


@pytest.fixture(autouse=True)
def _stub_harbor_environment(monkeypatch):
    """Keep the clean suite hermetic: stub the Docker-removal default.

    Tests that explicitly inject a ``docker_rm`` callable override this stub;
    a default (no explicit seam) must never shell out to real Docker in CI.
    """
    import daydream.benchmark.harbor.clean as clean_mod

    monkeypatch.setattr(
        clean_mod, "_default_docker_rm", lambda refs: {"returncode": 0}
    )


# ---------------------------------------------------------------------------
# shared hermetic fixtures (Tasks 2-9, 11)
# ---------------------------------------------------------------------------


def _seed_clean_ws(tmp_path):
    """A minimal valid workspace: curated scaffold + runtime, no derived dirt."""
    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    for name in ("imports", "cases", "snapshots"):
        (ws / name).mkdir(parents=True, exist_ok=True)
    (ws / "benchmark.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    return ws


def _docker_env(trial_name, *, removed=False, image_id=None):
    env_id = f"env-{trial_name}"
    return {
        "trial_name": trial_name,
        "environment_id": env_id,
        "backend": "docker",
        "image_id": image_id or f"hb__{trial_name}",
        "image_tags": [],
        "removed": removed,
    }


def _append_ledger_run(ws, run_id, *, state, environments):
    """Append a contained, validated ledger run via the run supervisor helpers."""
    import daydream.benchmark.harbor.run as run_mod

    job_dir = str((ws / "harbor" / "jobs" / run_id).resolve())
    run_mod.ledger_append_running(
        ws, run_id=run_id, compiled_lock_sha256="a" * 64, job_dir=job_dir,
        mode="oracle",
    )
    run_mod.ledger_mark(ws, run_id, state=state, environments=environments)


def _append_ledger_run_raw(ws, run_id, *, job_dir, state, environments):
    """Append a caller-supplied job_dir row (to inject a non-contained path)."""
    from daydream.benchmark import storage

    path = ws / "runtime" / "harbor.json"
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
    else:
        doc = {"schema_version": 1, "runs": []}
    doc["runs"].append({
        "run_id": run_id,
        "mode": "oracle",
        "state": state,
        "compiled_lock_sha256": "a" * 64,
        "job_dir": job_dir,
        "harbor_job_id": None,
        "environments": environments,
        "error": None,
    })
    storage.atomic_write_json(path, doc, mode=0o600)


# ---------------------------------------------------------------------------
# Task 1: clean subcommand surface + --derived union
# ---------------------------------------------------------------------------


def test_clean_parser_exposes_flags_and_derived_union():
    parser = _build_benchmark_parser()
    args = parser.parse_args(["clean", "/ws", "--cache", "--jobs", "--trajectories"])
    assert args.subcommand == "clean"
    assert args.dir == Path("/ws")
    assert args.cache is True and args.jobs is True and args.trajectories is True
    assert args.derived is False and args.all is False and args.yes is False
    derived = parser.parse_args(["clean", "/ws", "--derived"])
    assert derived.derived is True and derived.cache is False


def test_handle_clean_routes_to_clean_workspace(tmp_path, monkeypatch):
    import daydream.benchmark.harbor.clean as clean_mod

    captured = {}

    def fake_clean(root, *, cache, jobs, trajectories, all_, yes):
        captured.update(root=Path(root), cache=cache, jobs=jobs,
                        trajectories=trajectories, all_=all_, yes=yes)
        return clean_mod.CleanReport()

    monkeypatch.setattr(clean_mod, "clean_workspace", fake_clean)
    code = _handle_benchmark_command(["clean", str(tmp_path), "--derived", "--yes"])
    assert code == 0
    assert captured["root"] == tmp_path
    assert captured["cache"] is True and captured["jobs"] is True
    assert captured["trajectories"] is True
    assert captured["all_"] is False and captured["yes"] is True


# ---------------------------------------------------------------------------
# Task 2: CleanReport + empty-selection no-op
# ---------------------------------------------------------------------------


def test_clean_report_has_exit_code_and_summary():
    r = clean_mod.CleanReport()
    assert r.exit_code == 0
    lines = r.summary_lines()
    assert isinstance(lines, list)
    assert len(lines) >= 1   # a summary line exists


def test_clean_no_flags_deletes_nothing_and_preserves_gold(tmp_path):
    ws = _seed_clean_ws(tmp_path)          # benchmark.yaml + imports/cases/snapshots/
    report = clean_mod.clean_workspace(ws)  # no selection flags
    assert report.exit_code == 0
    for name in ("benchmark.yaml", "imports", "cases", "snapshots"):
        assert (ws / name).exists(), f"{name} must be preserved on a no-flag clean"
    assert report.cache_deleted == 0 and report.job_dirs_deleted == 0
    assert report.trajectory_deleted == 0 and report.images_removed == 0


# ---------------------------------------------------------------------------
# Task 3: --cache
# ---------------------------------------------------------------------------


def test_clean_cache_deletes_only_cache_targets(tmp_path):
    ws = _seed_clean_ws(tmp_path)
    (ws / "cache" / "repository.git").mkdir(parents=True)
    (ws / "cache" / "harbor-build-stage").mkdir(parents=True)
    report = clean_mod.clean_workspace(ws, cache=True)
    assert report.exit_code == 0
    assert report.cache_deleted == 2
    assert not (ws / "cache" / "repository.git").exists()
    assert not (ws / "cache" / "harbor-build-stage").exists()
    for name in ("benchmark.yaml", "imports", "cases", "snapshots"):
        assert (ws / name).exists(), f"curated {name} must survive --cache"
    # the empty cache/ dir itself remains (clean removes targets, not the scaffold)
    assert (ws / "cache").is_dir()


def test_clean_cache_absent_target_is_already_clean(tmp_path):
    ws = _seed_clean_ws(tmp_path)  # no cache/ targets yet
    report = clean_mod.clean_workspace(ws, cache=True)
    assert report.exit_code == 0 and report.cache_absent == 2
    assert report.cache_deleted == 0
    # container job dirs / trajectories untouched when only --cache is given
    assert report.job_dirs_deleted == 0 and report.trajectory_deleted == 0


# ---------------------------------------------------------------------------
# Task 4: --trajectories
# ---------------------------------------------------------------------------


def test_clean_trajectories_deletes_job_trajectories_only(tmp_path):
    ws = _seed_clean_ws(tmp_path)
    run_id = "00000000-0000-0000-0000-0000000000a1"
    job = ws / "harbor" / "jobs" / run_id
    traj = job / "case-abc" / "agent" / "trajectory.json"
    traj.parent.mkdir(parents=True)
    traj.write_text("{}")
    (job / "case-abc" / "verifier").mkdir(parents=True)
    (job / "case-abc" / "verifier" / "reward.json").write_text("{}")
    _append_ledger_run(ws, run_id, state="complete",
                       environments=[_docker_env("case-abc__1", removed=False)])
    report = clean_mod.clean_workspace(ws, trajectories=True)
    assert report.exit_code == 0 and report.trajectory_deleted == 1
    assert not traj.exists()
    assert (job / "case-abc" / "verifier" / "reward.json").exists()  # dir remains
    assert job.is_dir()  # --trajectories does not remove the job dir itself
    for name in ("benchmark.yaml", "imports", "cases", "snapshots"):
        assert (ws / name).exists()


def test_clean_trajectories_absent_dir_is_already_clean(tmp_path):
    ws = _seed_clean_ws(tmp_path)
    _append_ledger_run(ws, "00000000-0000-0000-0000-0000000000a2",
                       state="complete", environments=[])
    report = clean_mod.clean_workspace(ws, trajectories=True)
    assert report.exit_code == 0 and report.trajectory_absent == 1
    assert report.job_dirs_deleted == 0


# ---------------------------------------------------------------------------
# Task 5: --jobs (ledger-driven job-dir deletion + cleaned state)
# ---------------------------------------------------------------------------


def test_clean_jobs_deletes_ledgered_job_dir_and_marks_cleaned(tmp_path):
    ws = _seed_clean_ws(tmp_path)
    run_id = "00000000-0000-0000-0000-0000000000b1"
    job = ws / "harbor" / "jobs" / run_id
    (job / "case-abc" / "verifier").mkdir(parents=True)
    (job / "case-abc" / "verifier" / "reward.json").write_text("{}")
    _append_ledger_run(ws, run_id, state="complete",
                       environments=[_docker_env("case-abc__1", removed=False)])
    report = clean_mod.clean_workspace(ws, jobs=True)
    assert report.exit_code == 0 and report.job_dirs_deleted == 1
    assert not job.exists()
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert ledger["runs"][0]["state"] == "cleaned"
    for name in ("benchmark.yaml", "imports", "cases", "snapshots"):
        assert (ws / name).exists()


def test_clean_jobs_already_cleaned_run_is_noop(tmp_path):
    ws = _seed_clean_ws(tmp_path)
    run_id = "00000000-0000-0000-0000-0000000000b2"
    _append_ledger_run(ws, run_id, state="cleaned", environments=[])
    report = clean_mod.clean_workspace(ws, jobs=True)
    assert report.exit_code == 0 and report.runs_already_clean == 1
    assert report.job_dirs_deleted == 0


# ---------------------------------------------------------------------------
# Task 6: --jobs recorded-image removal via the docker_rm seam
# ---------------------------------------------------------------------------


def test_clean_jobs_removes_recorded_images_and_marks_removed(tmp_path):
    ws = _seed_clean_ws(tmp_path)
    run_id = "00000000-0000-0000-0000-0000000000c1"
    (ws / "harbor" / "jobs" / run_id / "t").mkdir(parents=True)
    env = _docker_env("case-abc__1", removed=False, image_id="hb__deadbeef")
    _append_ledger_run(ws, run_id, state="complete", environments=[env])
    removed_refs = []

    def fake_docker_rm(refs):
        removed_refs.extend(refs)
        return {"returncode": 0}

    clean_mod.clean_workspace(ws, jobs=True, docker_rm=fake_docker_rm)
    assert removed_refs == ["hb__deadbeef"]            # exact recorded ref, never guessed
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert ledger["runs"][0]["environments"][0]["removed"] is True
    assert ledger["runs"][0]["state"] == "cleaned"


def test_clean_jobs_removed_true_env_skipped(tmp_path):
    ws = _seed_clean_ws(tmp_path)
    run_id = "00000000-0000-0000-0000-0000000000c2"
    (ws / "harbor" / "jobs" / run_id / "t").mkdir(parents=True)
    _append_ledger_run(ws, run_id, state="complete",
                       environments=[_docker_env("c", removed=True)])
    called = []
    clean_mod.clean_workspace(
        ws, jobs=True, docker_rm=lambda refs: called.append(refs) or {"returncode": 0}
    )
    assert called == []                                 # already-removed image not re-attempted
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert ledger["runs"][0]["state"] == "cleaned"


# ---------------------------------------------------------------------------
# Task 7: partial Docker failure tolerance
# ---------------------------------------------------------------------------


def test_job_dir_kept_when_image_removal_fails(tmp_path):
    ws = _seed_clean_ws(tmp_path)
    run_id = "00000000-0000-0000-0000-0000000000d1"
    job = ws / "harbor" / "jobs" / run_id
    (job / "t").mkdir(parents=True)
    _append_ledger_run(ws, run_id, state="complete",
                       environments=[_docker_env("c", removed=False)])

    def fail_docker_rm(refs):
        return {"returncode": 1}

    report = clean_mod.clean_workspace(ws, jobs=True, docker_rm=fail_docker_rm)
    assert report.exit_code == 1 and report.images_failed == 1
    assert job.exists()                                # job dir preserved on image failure
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert ledger["runs"][0]["environments"][0]["removed"] is False
    assert ledger["runs"][0]["state"] == "complete"    # not transitioned to cleaned


def test_partial_failure_continues_to_other_runs(tmp_path):
    ws = _seed_clean_ws(tmp_path)
    for i, (rid, img) in enumerate([("a", "hb__x-a"), ("b", "hb__x-b")]):
        job = ws / "harbor" / "jobs" / rid
        (job / "t").mkdir(parents=True)
        _append_ledger_run(
            ws, rid, state="complete",
            environments=[_docker_env(f"c{i}", removed=False, image_id=img)],
        )

    def selective(refs):
        return {"returncode": 1 if refs == ["hb__x-a"] else 0}

    report = clean_mod.clean_workspace(ws, jobs=True, docker_rm=selective)
    assert report.images_failed == 1 and report.images_removed == 1
    assert report.job_dirs_deleted == 1                 # successful run's dir removed