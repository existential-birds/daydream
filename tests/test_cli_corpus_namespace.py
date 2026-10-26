"""Public record corpus commands, benchmark gates, and retired command refusals."""
import json
from pathlib import Path

import pytest

from daydream.dataset import LocalRecordStore, SnapshotRecords
from daydream.training.record_evidence import latest_annotation
from tests.harness.dataset import observation, run_record
from tests.harness.record_projection import seed_projection_store
from tests.harness.scripts import cli_main


def test_bare_corpus_prints_supported_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["corpus"]) == 2
    output = capsys.readouterr().out
    assert all(verb in output for verb in ("dataset", "harvest", "build", "label", "adjudicate"))
    assert "hydrate-hub" not in output
    assert "checkpoint" not in output


@pytest.mark.parametrize('verb', [
    "publish-final",
])
def test_retired_annotation_commands_rejected_without_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    directories = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_dir()}
    assert cli_main(["corpus", "adjudicate", verb]) == 2
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_dir()} == directories


def test_retired_hydration_command_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    assert cli_main(["corpus", "hydrate-hub"]) == 2
    assert set(tmp_path.rglob("*")) == before


def test_run_label_appends_human_history_and_refuses_ambiguous_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record("run-first"))
    store.commit_run(run_record("run-second"))
    assert cli_main(["corpus", "label", "run-", "--store", str(store.root), "--outcome", "accepted"]) == 1
    assert store.read_records()["observations"] == ()
    store.append_observation(observation("automated", run_id="run-first", item_uid=None, role="automatic",
        payload={"type": "run-label", "label": "rejected"}))
    args = ["corpus", "label", "run-first", "--store", str(store.root), "--author", "alice"]
    assert cli_main([*args, "--outcome", "accepted"]) == 0
    assert "rejected" in capsys.readouterr().out
    assert cli_main([*args, "--outcome", "unknown"]) == 0
    assert "accepted" in capsys.readouterr().out
    store.append_observation(observation("later-automated", run_id="run-first", item_uid=None, role="automatic",
        observed_at="2100-01-01T00:00:00Z", payload={"type": "run-label", "label": "accepted"}))
    current = store.read_records()
    resolved = latest_annotation(SnapshotRecords({}, current["runs"], current["observations"],
                                                  current["observations"]), "run-first")
    assert resolved is not None and resolved["labels"] == [] and resolved["human_labeler"] == "alice"
    history = current["observations"]
    assert len(history) == 4
    assert {(r["role"], r["run_id"], r["author"], r["payload"]["label"]) for r in history
            if r["role"] != "automatic"} == {
        ("rater", "run-first", "alice", "accepted"), ("rater", "run-first", "alice", "unknown")}
    assert cli_main([
        "corpus", "label", "run-second", "--store", str(store.root), "--author", "bob", "--outcome", "contested",
    ]) == 0
    assert "No prior label for run-second." in capsys.readouterr().out
    updated_history = store.read_records()["observations"]
    assert len(updated_history) == 5
    assert [(r["role"], r["run_id"], r["author"], r["payload"]["label"], r["semantic_evidence"]["outcome"])
            for r in updated_history if r["run_id"] == "run-second"] == [
        ("rater", "run-second", "bob", "contested", "contested")]
    assert tuple(r for r in updated_history if r["run_id"] == "run-first") == history


def test_snapshot_cli_freezes_temporal_membership(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation())
    assert cli_main(["corpus", "dataset", "snapshot", "--store", str(store.root),
                     "--observed-before", "2026-10-04T12:00:00Z", "--valid-before", "2026-10-04T10:00:00Z"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    selected = store.read_snapshot(snapshot["snapshot_id"])
    assert len(selected.observations) == 1
    assert selected.eligible_observations == ()
    store.append_observation(observation("later", observed_at="2026-10-05T00:00:00Z"))
    assert store.read_snapshot(snapshot["snapshot_id"]).observations == selected.observations


def test_record_build_cli_without_license_configuration(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path, dispositions=("accepted",))
    snapshot = store.select_snapshot(observed_before="2100-01-01T00:00:00Z")
    output = tmp_path / "out"
    assert cli_main(["corpus", "build", "--store", str(store.root), "--snapshot-id", snapshot["snapshot_id"],
                     "--out", str(output / "corpus.jsonl")]) == 0
    assert (output / "_SUCCESS").exists()
    lineage = json.loads((output / "lineage.json").read_text())
    assert "license_policy" not in lineage
    assert not (output / "license-report.json").exists()


@pytest.mark.parametrize("flag", ["--license-policy", "--allow-copyleft"])
def test_record_build_cli_rejects_removed_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str,
) -> None:
    store = seed_projection_store(tmp_path, dispositions=("accepted",))
    snapshot = store.select_snapshot(observed_before="2100-01-01T00:00:00Z")
    output = tmp_path / "out"
    args = ["corpus", "build", "--store", str(store.root), "--snapshot-id", snapshot["snapshot_id"],
            "--out", str(output / "corpus.jsonl")]
    assert cli_main([*args, flag, "unused"]) == 2
    assert f"unrecognized arguments: {flag}" in capsys.readouterr().err
    assert not output.exists()
    assert cli_main(args) == 0
    assert (output / "_SUCCESS").exists()


def test_bare_harvest_is_unknown_verb_treated_as_review_target(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["harvest", "--dry-run"]) == 2
    assert "--dry-run" in capsys.readouterr().err


def test_frozen_evidence_consumer_can_enter_fresh_cli_process(tmp_path: Path) -> None:
    import subprocess
    import sys

    code = ("from daydream.training.record_evidence import sessions_from_snapshot; "
            "from daydream.cli import main; main(['corpus', 'harvest', '--help'])")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage: daydream corpus harvest" in result.stdout
