"""Typed human observations preserve append-only history and model review gates."""

from pathlib import Path

import pytest

from daydream.training.adjudication.observations import append_observation
from daydream.training.record_evidence import sessions_from_snapshot
from tests.harness.adjudication import judgment, record_store, snapshot_id


def test_typed_judgment_append_is_immutable_idempotent_and_model_review_required(tmp_path: Path) -> None:
    store = record_store(tmp_path / "records", ("unanswered",))
    typed = judgment(store, role="model-suggested", author="gpt-6")
    row = {**typed, **typed["payload"], "labeler": typed["author"], "evidence": typed["semantic_evidence"]}
    assert append_observation(store, row, run_id="run-1", item_uid="item:0")
    assert not append_observation(store, row, run_id="run-1", item_uid="item:0")
    effective = sessions_from_snapshot(store.read_snapshot(snapshot_id(store)))[0]["resolutions"][0]
    assert effective["review_required"] and not effective["gold_eligible"]
    row.update(role="adjudicator")
    with pytest.raises(ValueError, match="model/LLM"):
        append_observation(store, row, run_id="run-1", item_uid="item:0")
