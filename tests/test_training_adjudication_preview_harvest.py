"""Preview and export pin canonical finding evidence without mutable indexes."""

from pathlib import Path

from daydream.dataset import LocalRecordStore
from daydream.training.adjudication.harvest import build_export_entries
from daydream.training.adjudication.preview import run_preview
from tests.harness.adjudication import append_replies, record_store, snapshot_id


def test_preview_and_export_identity_digest_stability_gate(tmp_path: Path) -> None:
    store = record_store(tmp_path / "records")
    pin = snapshot_id(store)
    ledger = tmp_path / "ledger.json"
    first = run_preview(store.root, pin, ledger)
    before = ledger.read_bytes()
    second = run_preview(store.root, pin, ledger)
    assert before == ledger.read_bytes() and first == second
    rows = build_export_entries(store.root, pin, ledger)
    assert len(rows) == 2 and len({r["record_id"] for r in rows}) == 2
    assert all(r["tier"] == "task-only" for r in rows)


def _edited_snapshot(store: LocalRecordStore) -> str:
    append_replies(store)
    return snapshot_id(store)


def test_preview_detects_immutable_snapshot_evidence_drift(tmp_path: Path) -> None:
    store = record_store(tmp_path / "records")
    ledger = tmp_path / "ledger.json"
    first = run_preview(store.root, snapshot_id(store), ledger)
    changed = run_preview(store.root, _edited_snapshot(store), ledger)
    assert len(changed["drifted_record_ids"]) == 1
    assert changed["snapshot_id"] != first["snapshot_id"] and changed["ledger_digest"] != first["ledger_digest"]


def test_export_rejects_old_ledger_digest_before_any_write(tmp_path: Path) -> None:
    import pytest

    from daydream.training.adjudication.canonical import AnnotationDriftError

    store = record_store(tmp_path / "records")
    ledger = tmp_path / "ledger.json"
    run_preview(store.root, snapshot_id(store), ledger)
    before = ledger.read_bytes()
    with pytest.raises(AnnotationDriftError) as error:
        build_export_entries(store.root, _edited_snapshot(store), ledger)
    assert len(error.value.requeued_record_ids) == 1
    assert ledger.read_bytes() == before
