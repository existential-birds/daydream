"""Offline pinned membership, deterministic bytes, and chronological evidence eligibility."""

from pathlib import Path

import pytest

from daydream.dataset import semantic_evidence_digest
from daydream.training.corpus_projection.projector import build_frozen_corpus
from tests.harness.dataset import observation
from tests.harness.record_projection import (
    projection_config,
    seed_projection_store,
)
from tests.test_corpus_projection import _read_jsonl


def test_observation_temporal_eligibility_and_embedded_posterior_leak_are_distinct(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path, dispositions=("accepted",))
    run = store.read_records()["runs"][0]
    item = run["findings"]["value"]["items"][0]
    evidence = [{"valid_at": "2026-10-06T00:00:00Z", "body": "late"}]
    # An observation whose own valid time is future cannot replace eligible human evidence.
    store.append_observation(observation("late-valid", run_id=run["run_id"], item_uid=item["item_uid"],
        role="adjudicator", valid_at="2026-10-06T00:00:00Z", observed_at="2026-10-04T13:00:00Z",
        semantic_evidence=evidence, evidence_digest=semantic_evidence_digest(evidence),
        payload={"type": "finding-judgment", "disposition": "rejected", "rationale": "late"}))
    config = projection_config(store, tmp_path)
    build_frozen_corpus(config)
    assert _read_jsonl(config.out_dir / "corpus.jsonl")[0]["disposition"] == "accepted"
    # Eligible envelopes still must not smuggle posterior reply evidence through the pin.
    store.append_observation(observation("leaking-valid", run_id=run["run_id"], item_uid=item["item_uid"],
        role="automatic", observed_at="2026-10-04T14:00:00Z", semantic_evidence=evidence,
        evidence_digest=semantic_evidence_digest(evidence),
        payload={"type": "finding-judgment", "disposition": "rejected", "rationale": "invalid pin"}))
    with pytest.raises(ValueError, match="posterior"):
        build_frozen_corpus(projection_config(store, tmp_path, out_dir=tmp_path / "leaking"))
    assert not (tmp_path / "leaking").exists()
