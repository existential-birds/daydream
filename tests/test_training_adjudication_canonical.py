"""Record annotation exports reject evidence drift before output mutation."""

from pathlib import Path

import pytest

from daydream.training.adjudication.canonical import AnnotationDriftError, run_canonical_harvest
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.labeler_versions import reply_evidence_digest
from tests.harness.adjudication import record_store, snapshot_id


def test_record_harvest_refuses_evidence_drift_before_any_write(tmp_path: Path) -> None:
    store = record_store(tmp_path / "records")
    out = tmp_path / "out"
    run_materialize(store.root, snapshot_id(store), out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    observation = store.read_records()["observations"][0]
    evidence = [{"reply_id": 9, "body": "edited"}]
    store.append_observation(
        {
            **observation,
            "observation_id": "edited",
            "observed_at": "2026-10-06T12:00:00Z",
            "semantic_evidence": evidence,
            "evidence_digest": reply_evidence_digest(evidence),
        }
    )
    with pytest.raises(AnnotationDriftError) as error:
        run_canonical_harvest(store.root, snapshot_id(store), out)
    assert len(error.value.requeued_record_ids) == 1
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
