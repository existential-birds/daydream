"""Hermetic suite for the `daydream benchmark clean` command (issue #782).

The deletion set is strictly ledger-driven and containment-checked: every
hermetic test stubs the Docker-removal default so CI never shells out to a
real Docker/network call (the real ``docker rmi`` path is issue #13).
"""
import json
from pathlib import Path
from typing import Any

import pytest

import daydream.benchmark.harbor.clean as clean_mod
import daydream.benchmark.harbor.run as run_mod
from daydream.benchmark import storage
from daydream.benchmark.cli import _handle_benchmark_command


@pytest.fixture(autouse=True)
def _stub_harbor_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the clean suite hermetic: stub the Docker-removal default.

    Tests that explicitly inject a ``docker_rm`` callable override this stub;
    a default (no explicit seam) must never shell out to real Docker in CI.
    """
    monkeypatch.setattr(clean_mod, "_default_docker_rm", lambda refs: {"returncode": 0})

# shared hermetic fixtures (Tasks 2-9, 11)


@pytest.fixture
def ws(tmp_path: Path) -> Any:
    return _make_ws(tmp_path / "ws")


def _make_ws(ws: Path) -> Path:
    """A minimal valid workspace: curated scaffold + runtime, no derived dirt."""
    (ws / "runtime").mkdir(parents=True)
    for name in ("imports", "cases", "snapshots"):
        (ws / name).mkdir(parents=True, exist_ok=True)
    (ws / "benchmark.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    return ws


def _docker_env(trial_name: Any, *, removed: Any=False, image_id: Any=None) -> dict[str, Any]:
    env_id = f"env-{trial_name}"
    return {"trial_name": trial_name, "environment_id": env_id, "backend": "docker",
        "image_id": image_id or f"hb__{trial_name}", "image_tags": [], "removed": removed,
    }


def _append_ledger_run(ws: Path, run_id: Any, *, state: Any, environments: Any) -> None:
    """Append a contained, validated ledger run via the run supervisor helpers."""
    job_dir = str((ws / "harbor" / "jobs" / run_id).resolve())
    run_mod.ledger_append_running(ws, run_id=run_id, compiled_lock_sha256="a" * 64, job_dir=job_dir, mode="oracle")
    run_mod.ledger_mark(ws, run_id, state=state, environments=environments)


def _append_ledger_run_raw(ws: Path, run_id: Any, *, job_dir: Any, state: Any, environments: Any) -> None:
    """Append a caller-supplied job_dir row (to inject a non-contained path)."""
    path = ws / "runtime" / "harbor.json"
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
    else:
        doc = {"schema_version": 1, "runs": []}
    doc["runs"].append({
        "run_id": run_id, "mode": "oracle", "state": state, "compiled_lock_sha256": "a" * 64, "job_dir": job_dir,
        "harbor_job_id": None, "environments": environments, "error": None,
    })
    storage.atomic_write_json(path, doc, mode=0o600)


def test_handle_clean_routes_to_clean_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    def fake_clean(root: Any, *, cache: pytest.Cache, jobs: Any, trajectories: Any, all_: Any, yes: Any) -> Any:
        captured.update(root=Path(root), cache=cache, jobs=jobs, trajectories=trajectories, all_=all_, yes=yes)
        return clean_mod.CleanReport()

    monkeypatch.setattr(clean_mod, "clean_workspace", fake_clean)
    code = _handle_benchmark_command(["clean", str(tmp_path), "--derived", "--yes"])
    assert code == 0
    assert captured["root"] == tmp_path
    assert captured["cache"] is True and captured["jobs"] is True
    assert captured["trajectories"] is True
    assert captured["all_"] is False and captured["yes"] is True

def _absent_image_counts_absent_without_blocking(ws: Path) -> None:
    """An already-absent image (the docker_rm seam reports ``absent``) is
    counted as absent, not failed, and does not block the run from cleaning:
    without this, an externally-pruned image re-fails forever and the run
    never transitions to ``cleaned``.
    """
    run_id = "00000000-0000-0000-0000-0000000000e2"
    job = ws / "harbor" / "jobs" / run_id
    (job / "t").mkdir(parents=True)
    env = _docker_env("c", removed=False, image_id="hb__absent")
    _append_ledger_run(ws, run_id, state="complete", environments=[env])

    def absent(refs: Any) -> dict[str, Any]:
        return {"returncode": 1, "absent": True}

    report = clean_mod.clean_workspace(ws, jobs=True, docker_rm=absent)
    assert report.exit_code == 0
    assert report.images_absent == 1 and report.images_failed == 0
    assert report.job_dirs_deleted == 1 and report.runs_cleaned == 1
    assert not job.exists()
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    # persisted so a later pass does not re-attempt (and re-fail) the image
    assert ledger["runs"][0]["environments"][0]["removed"] is True
    assert ledger["runs"][0]["state"] == "cleaned"

def _keeps_dir_when_environments_empty(ws: Path) -> None:
    """A ledger row with no recorded environments cannot be cleaned: deleting
    its job dir would permanently orphan the images that run spawned. Keep the
    dir and the run's prior state, and surface the incomplete cleanup.
    """
    run_id = "00000000-0000-0000-0000-0000000000d3"
    job = ws / "harbor" / "jobs" / run_id
    (job / "t").mkdir(parents=True)
    _append_ledger_run(ws, run_id, state="complete", environments=[])
    called: list[list[str]] = []

    def _docker_rm(refs: list[str]) -> dict[str, Any]:
        called.append(refs)
        return {"returncode": 0}

    report = clean_mod.clean_workspace(ws, jobs=True, docker_rm=_docker_rm)
    assert called == []                              # no image refs to address
    assert report.exit_code == 1 and report.images_failed == 1
    assert job.is_dir()
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert ledger["runs"][0]["state"] == "complete"  # never marked cleaned


def _symlink_escape_target_rejected(ws: Path) -> None:
    victim = ws.parent / "outside-secret"
    victim.mkdir()
    (ws / "cache").mkdir(parents=True)
    (ws / "cache" / "repository.git").symlink_to(victim, target_is_directory=True)
    with pytest.raises(clean_mod.WorkspaceCorrupt):
        clean_mod.clean_workspace(ws, cache=True)
    assert victim.exists()   # outside target untouched

def _non_contained_ledger_job_dir_rejected(ws: Path) -> None:
    evil = ws.parent / "evil"
    evil.mkdir()
    _append_ledger_run_raw(ws, "bad", job_dir=str(evil), state="complete", environments=[])
    with pytest.raises(clean_mod.RunError):
        clean_mod.clean_workspace(ws, jobs=True)
    assert evil.exists()

def _refuses_without_yes_and_no_tty(ws: Path) -> None:
    report = clean_mod.clean_workspace(ws, all_=True, yes=False, confirm=lambda _: False)
    assert report.exit_code == 1
    for name in ("benchmark.yaml", "imports", "cases", "snapshots"):
        assert (ws / name).exists()   # nothing deleted on refusal


def _preserves_curated_when_derived_stage_raises(ws: Path) -> None:
    """Curated source/gold is unrecoverable, so it must be deleted last: a
    failure in a derived stage (here the jobs image-removal seam) must not have
    already destroyed the irreplaceable workspace when the stage raises.
    """
    run_id = "00000000-0000-0000-0000-0000000000f1"
    job = ws / "harbor" / "jobs" / run_id
    (job / "t").mkdir(parents=True)
    _append_ledger_run(ws, run_id, state="complete", environments=[_docker_env("c", removed=False)])

    def boom(refs: list[str]) -> dict[str, Any]:
        raise RuntimeError("derived selection failed")

    with pytest.raises(RuntimeError):
        clean_mod.clean_workspace(ws, all_=True, yes=True, docker_rm=boom)
    for name in ("benchmark.yaml", "imports", "cases", "snapshots"):
        assert (ws / name).exists(), f"curated {name} must survive a derived-stage failure"

def _preserves_curated_when_derived_stage_soft_fails(ws: Path) -> None:
    """A soft derived-stage failure (a ``docker_rm`` non-zero returncode, not
    an exception) must also preserve curated source/gold: the pass exits
    non-zero, and an unrecoverable workspace must not be destroyed first.
    """
    run_id = "00000000-0000-0000-0000-0000000000f2"
    job = ws / "harbor" / "jobs" / run_id
    (job / "t").mkdir(parents=True)
    _append_ledger_run(ws, run_id, state="complete", environments=[_docker_env("c", removed=False)])

    def fail_soft(refs: Any) -> dict[str, Any]:
        return {"returncode": 1}

    report = clean_mod.clean_workspace(ws, all_=True, yes=True, docker_rm=fail_soft)
    assert report.exit_code == 1 and report.images_failed == 1
    assert report.gold_deleted == 0 and report.recoverable is True
    for name in ("benchmark.yaml", "imports", "cases", "snapshots"):
        assert (ws / name).exists(), f"curated {name} must survive a soft derived failure"

def test_clean_derived_union_deletes_all_derived(ws: Path) -> None:
    """One ledger-owned workspace exercises selected cleanup, resume and confirmed curated deletion."""
    (ws / "cache" / "repository.git").mkdir(parents=True)
    (ws / "cache" / "harbor-build-stage").mkdir()
    primary, missing, already = ("00000000-0000-0000-0000-0000000000e1",
        "00000000-0000-0000-0000-0000000000e2", "00000000-0000-0000-0000-0000000000e3")
    job = ws / "harbor" / "jobs" / primary
    trajectory = job / "case" / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text("{}")
    verifier = job / "case" / "verifier" / "reward.json"
    verifier.parent.mkdir()
    verifier.write_text('{"reward": 1}')
    _append_ledger_run(ws, primary, state="complete", environments=[
        _docker_env("active", image_id="hb__owned"), _docker_env("retired", removed=True, image_id="hb__already")])
    _append_ledger_run(ws, missing, state="cleanup_pending", environments=[])
    _append_ledger_run(ws, already, state="cleaned", environments=[])
    curated = [ws / name for name in ("benchmark.yaml", "imports", "cases", "snapshots")]
    original_manifest = curated[0].read_bytes()
    before = (ws / "runtime" / "harbor.json").read_bytes()
    assert clean_mod.clean_workspace(ws) == clean_mod.CleanReport()
    assert trajectory.exists() and verifier.exists() and (ws / "runtime" / "harbor.json").read_bytes() == before
    trajectories = clean_mod.clean_workspace(ws, trajectories=True)
    assert (trajectories.exit_code, trajectories.trajectory_deleted, trajectories.trajectory_absent) == (0, 1, 2)
    assert trajectories.job_dirs_deleted == 0 and trajectories.images_removed == 0
    assert not trajectory.exists() and verifier.read_text() == '{"reward": 1}' and job.is_dir()
    cache = clean_mod.clean_workspace(ws, cache=True)
    assert (cache.exit_code, cache.cache_deleted, cache.cache_absent) == (0, 2, 0)
    assert cache.job_dirs_deleted == cache.trajectory_deleted == 0 and (ws / "cache").is_dir()
    assert not (ws / "cache" / "repository.git").exists() and not (ws / "cache" / "harbor-build-stage").exists()
    removed: list[list[str]] = []
    def docker_rm(refs: list[str]) -> dict[str, Any]:
        removed.append(refs)
        return {"returncode": 0}
    report = clean_mod.clean_workspace(ws, cache=True, jobs=True, trajectories=True, docker_rm=docker_rm)
    assert (report.exit_code, report.cache_absent, report.trajectory_absent, report.job_dirs_deleted,
            report.job_dirs_absent, report.runs_cleaned, report.runs_already_clean,
            report.images_removed) == (0, 2, 2, 1, 1, 2, 1, 1)
    assert removed == [["hb__owned"]] and not job.exists()
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert {row["run_id"]: row["state"] for row in ledger["runs"]} == {
        primary: "cleaned", missing: "cleaned", already: "cleaned"
    }
    assert [env["removed"] for env in ledger["runs"][0]["environments"]] == [True, True]
    assert all(path.exists() for path in curated) and curated[0].read_bytes() == original_manifest
    frozen_ledger = (ws / "runtime" / "harbor.json").read_bytes()
    repeat = clean_mod.clean_workspace(ws, cache=True, jobs=True, trajectories=True, docker_rm=docker_rm)
    assert (repeat.exit_code, repeat.cache_absent, repeat.trajectory_absent, repeat.runs_already_clean) == (0, 2, 3, 3)
    assert repeat.cache_deleted == repeat.trajectory_deleted == repeat.job_dirs_deleted == repeat.images_removed == 0
    assert removed == [["hb__owned"]] and (ws / "runtime" / "harbor.json").read_bytes() == frozen_ledger
    assert all(path.exists() for path in curated)
    final = clean_mod.clean_workspace(ws, all_=True, yes=True, docker_rm=docker_rm)
    assert final.exit_code == 0 and final.gold_deleted == 4 and final.recoverable is False
    assert not any(path.exists() for path in curated) and removed == [["hb__owned"]]


def test_clean_jobs_and_all_fail_closed_without_destroying_unrecoverable_state(tmp_path: Path) -> None:
    """Every cleanup refusal or derived failure keeps images, escaped targets and curated gold."""
    for check in (_absent_image_counts_absent_without_blocking, _keeps_dir_when_environments_empty,
                  _symlink_escape_target_rejected, _non_contained_ledger_job_dir_rejected,
                  _refuses_without_yes_and_no_tty, _preserves_curated_when_derived_stage_raises,
                  _preserves_curated_when_derived_stage_soft_fails):
        check(_make_ws(tmp_path / check.__name__ / "ws"))
