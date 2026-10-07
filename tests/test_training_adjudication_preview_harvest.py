"""Preview and export pin canonical finding evidence without mutable indexes."""

import json
from pathlib import Path

import pytest

from daydream.dataset import LocalRecordStore
from daydream.training.adjudication.harvest import build_export_entries
from daydream.training.adjudication.preview import run_preview
from tests.harness.adjudication import append_replies, record_store, snapshot_id
from tests.harness.scripts import cli_main


def test_preview_and_export_identity_digest_stability_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    store = record_store(tmp_path / "records")
    pin = snapshot_id(store)
    ledger = tmp_path / "state" / "preview-ledger.json"
    args = ["corpus", "adjudicate", "preview", "--store", str(store.root), "--snapshot-id", pin,
            "--state-dir", str(ledger.parent)]
    assert cli_main(args) == 0
    first = json.loads(capsys.readouterr().out)
    before = ledger.read_bytes()
    assert cli_main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert before == ledger.read_bytes() and first == second
    rows = build_export_entries(store.root, pin, ledger)
    assert len(rows) == 2 and len({r["record_id"] for r in rows}) == 2
    assert all(r["tier"] == "task-only" for r in rows)


def _edited_snapshot(store: LocalRecordStore) -> str:
    append_replies(store)
    return snapshot_id(store)


def test_preview_detects_immutable_snapshot_evidence_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    store = record_store(tmp_path / "records")
    state = tmp_path / "state"
    args = ["corpus", "adjudicate", "preview", "--store", str(store.root), "--state-dir", str(state)]
    assert cli_main([*args, "--snapshot-id", snapshot_id(store)]) == 0
    first = json.loads(capsys.readouterr().out)
    first_id = json.loads((state / "preview-ledger.json").read_text())["items"][0]["record_id"]
    assert cli_main([*args, "--snapshot-id", _edited_snapshot(store)]) == 0
    changed = json.loads(capsys.readouterr().out)
    assert changed["drifted_record_ids"] == [first_id]
    assert changed["snapshot_id"] != first["snapshot_id"] and changed["ledger_digest"] != first["ledger_digest"]


def test_export_rejects_old_ledger_digest_before_any_write(tmp_path: Path) -> None:
    from daydream.training.adjudication.canonical import AnnotationDriftError

    store = record_store(tmp_path / "records")
    ledger = tmp_path / "ledger.json"
    run_preview(store.root, snapshot_id(store), ledger)
    before = ledger.read_bytes()
    with pytest.raises(AnnotationDriftError) as error:
        build_export_entries(store.root, _edited_snapshot(store), ledger)
    assert len(error.value.requeued_record_ids) == 1
    assert ledger.read_bytes() == before
