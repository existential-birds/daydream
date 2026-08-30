"""Observation store: append-only, idempotent, provenance-complete."""
from pathlib import Path

import pytest

from daydream.training.adjudication.observations import append_observation, load_observations

R1 = "a" * 64  # record_id-shaped

def _obs(record_id: str, labeler: str, disposition: str = "accepted",
         digest: str = "e" * 64, **kw) -> dict:
    base = {
        "record_id": record_id, "disposition": disposition,
        "evidence_digest": digest, "labeler": labeler, "role": "rater",
        "rationale": "matches reply meaning", "valid_at": "2026-08-30T12:00:00+00:00",
        "observed_at": "2026-08-30T12:00:01+00:00",
        "rubric_version": "984-adjudicate-r1", "review_required": False,
    }
    return {**base, **kw}

def test_append_is_append_only_and_load_round_trips(tmp_path: Path) -> None:
    store = tmp_path / "observations.jsonl"
    append_observation(store, _obs(R1, "alice"))
    append_observation(store, _obs(R1, "bob", disposition="rejected"))
    obs = load_observations(store)
    assert [o["labeler"] for o in obs] == ["alice", "bob"]  # both kept, newest last

def test_reappend_identical_observation_is_idempotent(tmp_path: Path) -> None:
    store = tmp_path / "observations.jsonl"
    o = _obs(R1, "alice")
    append_observation(store, o)
    append_observation(store, o)  # re-run of an interrupted labeling session
    assert len(load_observations(store)) == 1

def test_observation_missing_required_field_raises(tmp_path: Path) -> None:
    o = _obs(R1, "alice")
    del o["evidence_digest"]
    with pytest.raises(ValueError, match="evidence_digest"):
        append_observation(tmp_path / "o.jsonl", o)

def test_invalid_disposition_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bogus"):
        append_observation(tmp_path / "o.jsonl", _obs(R1, "alice", disposition="bogus"))
