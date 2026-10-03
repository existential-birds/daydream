"""Current hydrated inventory changes independently of immutable observations."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from daydream.archive import hydrate
from daydream.archive.index import append_label_observation, label_observation_history, query_runs
from tests.test_archive_hydrate import _ingested_stage, _seed_admitted_runs, _write_policy


def _observe(stage: Path, sid: str) -> list[dict[str, object]]:
    append_label_observation(
        stage, sid, labels=["finding-valid"], pr_state="MERGED", labeler_version="human",
        evidence_sha="e" * 40, source="human", observed_at="2025-01-01T00:00:00+00:00",
    )
    return label_observation_history(stage, sid)


def test_license_exclusion_removes_current_run_and_preserves_readmission_history(tmp_path: Path) -> None:
    stage = _ingested_stage(tmp_path)
    hydrate.dedupe_admitted(stage, revision="a" * 40)
    history = _observe(stage, "sess-a")
    hydrate.apply_license_gate(
        stage, revision="a" * 40, license_policy_path=_write_policy(tmp_path), allow_copyleft=frozenset(),
    )
    entry = json.loads(next(stage.glob("_dedupe/*/dedupe.jsonl")).read_text().splitlines()[-1])
    assert (entry["session_id"], entry["reason_code"]) == ("sess-a", "repo_identity_missing")
    assert query_runs(stage) == []
    assert label_observation_history(stage, "sess-a") == history
    ledger = hydrate.build_import_ledger(stage, revision="a" * 40, source_commit="a" * 40)
    assert ledger["imported"] == []
    assert ledger["rejections"][0]["reason_code"] == "repo_identity_missing"
    with pytest.raises(ValueError, match="Unknown session"):
        _observe(stage, "sess-a")
    shutil.copytree(stage / "excluded" / "sess-a", stage / "runs" / "sess-a")
    hydrate.rebuild_index(stage)
    assert [row["session_id"] for row in query_runs(stage)] == ["sess-a"]
    assert label_observation_history(stage, "sess-a") == history


def test_license_population_keeps_survivors_and_drops_stale_resolution_members(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    _seed_admitted_runs(stage, [
        ("sess-admitted", "octo/good", {"spdx_id": "MIT"}), ("sess-rejected", None, None),
    ])
    hydrate.rebuild_index(stage)
    history = _observe(stage, "sess-rejected")
    hydrate.apply_license_gate(
        stage, revision="a" * 40, license_policy_path=_write_policy(tmp_path), allow_copyleft=frozenset(),
    )
    assert [row["session_id"] for row in query_runs(stage)] == ["sess-admitted"]
    assert label_observation_history(stage, "sess-rejected") == history
    assert hydrate.build_resolution_map(stage, repo_commits={"octo/good": "b" * 40}) == {
        "octo/good": {"repo_slug": "octo/good", "pinned_sha": "b" * 40, "session_ids": ["sess-admitted"]},
    }


def test_rebuild_failure_preserves_complete_previous_population_and_history(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    _seed_admitted_runs(stage, [("sess-a", "octo/good", {"spdx_id": "MIT"})])
    hydrate.rebuild_index(stage)
    history = _observe(stage, "sess-a")
    before = query_runs(stage)
    path = stage / "runs" / "sess-a" / "manifest.json"
    data = json.loads(path.read_text())
    data["branch"] = "changed-before-invalid-row"
    path.write_text(json.dumps(data))
    invalid = stage / "runs" / "sess-z"
    invalid.mkdir()
    (invalid / "manifest.json").write_text(json.dumps({"session_id": "sess-z", "archived_at": None}))
    with pytest.raises(sqlite3.IntegrityError):
        hydrate.rebuild_index(stage)
    assert query_runs(stage) == before
    assert label_observation_history(stage, "sess-a") == history


def test_later_fixture_exclusion_removes_current_row_but_preserves_history(tmp_path: Path) -> None:
    stage = _ingested_stage(tmp_path)
    hydrate.dedupe_admitted(stage, revision="a" * 40)
    history = _observe(stage, "sess-a")
    path = stage / "runs" / "sess-a" / "manifest.json"
    data = json.loads(path.read_text())
    data["source_path"] = "/tmp/pytest-of-user/test_x"
    path.write_text(json.dumps(data))
    hydrate.dedupe_admitted(stage, revision="a" * 40)
    entry = json.loads(next(stage.glob("_dedupe/*/dedupe.jsonl")).read_text().splitlines()[-1])
    assert (entry["session_id"], entry["status"], entry["reason_code"]) == (
        "sess-a", "excluded", "fixture_pytest_path",
    )
    assert query_runs(stage) == []
    assert label_observation_history(stage, "sess-a") == history


def test_collision_restores_current_baseline_and_keeps_observation_history(tmp_path: Path) -> None:
    stage = _ingested_stage(tmp_path)
    hydrate.dedupe_admitted(stage, revision="a" * 40)
    history = _observe(stage, "sess-a")
    path = stage / "runs" / "sess-a" / "trajectory.json"
    before = path.read_bytes()
    path.write_bytes(b'{"different": true}')
    hydrate.dedupe_admitted(stage, revision="a" * 40)
    entry = json.loads(next(stage.glob("_dedupe/*/dedupe.jsonl")).read_text().splitlines()[-1])
    assert (entry["session_id"], entry["status"]) == ("sess-a", "collision")
    assert path.read_bytes() == before
    assert [row["session_id"] for row in query_runs(stage)] == ["sess-a"]
    assert label_observation_history(stage, "sess-a") == history
