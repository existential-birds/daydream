"""Record annotation exports reject evidence drift before output mutation."""

from pathlib import Path

import pytest

from daydream.training.adjudication.canonical import AnnotationDriftError, run_canonical_harvest
from daydream.training.adjudication.materialize import run_materialize
from tests.harness.adjudication import append_replies, record_store, snapshot_id


def test_record_harvest_refuses_evidence_drift_before_any_write(tmp_path: Path) -> None:
    store = record_store(tmp_path / "records")
    out = tmp_path / "out"
    run_materialize(store.root, snapshot_id(store), out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    append_replies(store)
    with pytest.raises(AnnotationDriftError) as error:
        run_canonical_harvest(store.root, snapshot_id(store), out)
    assert len(error.value.requeued_record_ids) == 1
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
