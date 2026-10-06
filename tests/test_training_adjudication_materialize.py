"""Frozen record materialization is deterministic, complete and read-only."""

from pathlib import Path

import pytest

from daydream.training.adjudication.materialize import run_materialize
from tests.harness.adjudication import record_store, snapshot_id


def test_materialize_emits_all_dispositions_deterministically(tmp_path: Path) -> None:
    store = record_store(tmp_path / "records", ("accepted", "rejected", "ambiguous", "unanswered", "missing"))
    pin = snapshot_id(store)
    before = store.read_records()
    first, second = tmp_path / "first", tmp_path / "second"
    assert run_materialize(store.root, pin, first)["record_count"] == 5
    run_materialize(store.root, pin, second)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    assert store.read_records() == before


def test_materialize_dry_run_and_unknown_snapshot_leave_no_outputs(tmp_path: Path) -> None:
    store = record_store(tmp_path / "records")
    out = tmp_path / "out"
    assert run_materialize(store.root, snapshot_id(store), out, dry_run=True)["record_count"] == 2
    assert not out.exists()
    with pytest.raises(ValueError):
        run_materialize(store.root, "a" * 64, out)
    assert not out.exists()
