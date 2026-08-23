"""Hermetic suite for the `daydream benchmark clean` command (issue #782).

The deletion set is strictly ledger-driven and containment-checked: every
hermetic test stubs the Docker-removal default so CI never shells out to a
real Docker/network call (the real ``docker rmi`` path is issue #13).
"""
import json
from pathlib import Path

import pytest

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