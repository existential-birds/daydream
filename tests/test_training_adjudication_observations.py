"""Observation store: append-only, idempotent, provenance-complete."""
from pathlib import Path
from typing import Any

import pytest

from daydream.training.adjudication.observations import append_observation, load_observations
from daydream.training.labeler_versions import ADJUDICATION_LABELER_VERSION

R1 = "a" * 64  # record_id-shaped

def _obs(record_id: str, labeler: str, disposition: str = "accepted",
         digest: str = "e" * 64, **kw: Any) -> dict[str, Any]:
    base = {
        "record_id": record_id, "disposition": disposition,
        "evidence_digest": digest, "labeler": labeler, "role": "rater",
        "rationale": "matches reply meaning", "valid_at": "2026-08-30T12:00:00+00:00",
        "observed_at": "2026-08-30T12:00:01+00:00",
        "rubric_version": ADJUDICATION_LABELER_VERSION, "review_required": False,
        "evidence": [{"reply_id": "r1"}],
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

def test_observation_missing_evidence_raises(tmp_path: Path) -> None:
    # The resolver hard-requires evidence, so the store must reject
    # evidence-less rows up front (fail-closed, not a late harvest crash).
    o = _obs(R1, "alice")
    del o["evidence"]
    with pytest.raises(ValueError, match="evidence"):
        append_observation(tmp_path / "o.jsonl", o)


def test_invalid_disposition_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bogus"):
        append_observation(tmp_path / "o.jsonl", _obs(R1, "alice", disposition="bogus"))

def test_model_suggested_label_is_review_required_and_rejected_as_human(tmp_path: Path) -> None:
    assert ADJUDICATION_LABELER_VERSION == "984-adjudicate-r1"  # era-pinned equality
    store = tmp_path / "o.jsonl"
    # Caller omits review_required entirely; the writer must force it on.
    model_obs = _obs(R1, "claude-classifier", role="model-suggested")
    model_obs.pop("review_required")
    append_observation(store, model_obs)
    obs = load_observations(store)
    assert obs[0]["role"] == "model-suggested"
    assert obs[0]["review_required"] is True
    # Stored rubric_version stays pinned to the canonical version axis.
    assert obs[0]["rubric_version"] == ADJUDICATION_LABELER_VERSION

def test_role_adjudicator_with_model_labeler_raises(tmp_path: Path) -> None:
    # An unreviewed LLM classifier is never a human labeler: an observation
    # claiming adjudicator authority under a model-shaped labeler is rejected.
    with pytest.raises(ValueError, match="labeler"):
        append_observation(
            tmp_path / "o.jsonl", _obs(R1, "claude-classifier", role="adjudicator")
        )
