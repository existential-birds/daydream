"""Run labels preserve authoritative precedence and the prior-label CLI display."""
from pathlib import Path

import pytest

from daydream.dataset import LocalRecordStore, SnapshotRecords
from daydream.training.record_evidence import latest_annotation
from tests.harness.dataset import observation, run_record
from tests.harness.scripts import cli_main


def test_label_shows_prior_and_human_unknown_overrides_later_automation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    store.append_observation(observation("automated", item_uid=None, role="automatic",
        payload={"type": "run-label", "label": "rejected"}))
    args = ["corpus", "label", "run-1", "--store", str(store.root), "--author", "alice"]
    assert cli_main([*args, "--outcome", "accepted"]) == 0
    assert "rejected" in capsys.readouterr().out
    assert cli_main([*args, "--outcome", "unknown"]) == 0
    assert "accepted" in capsys.readouterr().out
    store.append_observation(observation("later-automated", item_uid=None, role="automatic",
        observed_at="2100-01-01T00:00:00Z", payload={"type": "run-label", "label": "accepted"}))
    current = store.read_records()
    resolved = latest_annotation(SnapshotRecords({}, current["runs"], current["observations"],
                                                  current["observations"]), "run-1")
    assert resolved is not None and resolved["labels"] == [] and resolved["human_labeler"] == "alice"
    assert len(current["observations"]) == 4


def test_label_unknown_run_refuses_without_observations(tmp_path: Path) -> None:
    store = LocalRecordStore(tmp_path / "records")
    store.commit_run(run_record())
    assert cli_main(["corpus", "label", "absent", "--store", str(store.root), "--outcome", "accepted"]) == 1
    assert store.read_records()["observations"] == ()
