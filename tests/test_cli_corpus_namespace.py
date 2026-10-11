"""Public record corpus commands, benchmark gates, and retired command refusals."""
import json
from pathlib import Path

import pytest

from daydream.dataset import LocalRecordStore
from tests.harness.dataset import observation, run_record
from tests.harness.record_projection import seed_projection_store
from tests.harness.scripts import cli_main


def test_bare_corpus_prints_supported_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["corpus"]) == 2
    output = capsys.readouterr().out
    assert all(verb in output for verb in ("dataset", "harvest", "build", "label", "adjudicate"))
    assert "hydrate-hub" not in output
    assert "checkpoint" not in output


@pytest.mark.parametrize("verb", ["publish-state", "resume-state", "publish-final", "download-final"])
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


def test_run_label_appends_human_history_and_refuses_ambiguous_prefix(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record("run-first"))
    store.commit_run(run_record("run-second"))
    assert cli_main(["corpus", "label", "run-", "--store", str(store.root), "--outcome", "accepted"]) == 1
    assert store.read_records()["observations"] == ()
    assert cli_main(["corpus", "label", "run-first", "--store", str(store.root),
                     "--outcome", "accepted", "--author", "alice"]) == 0
    assert cli_main(["corpus", "label", "run-first", "--store", str(store.root),
                     "--outcome", "unknown", "--author", "alice"]) == 0
    history = store.read_records()["observations"]
    assert {r["payload"]["label"] for r in history} == {"accepted", "unknown"}
    assert all(r["role"] == "rater" and r["run_id"] == "run-first" for r in history)


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
