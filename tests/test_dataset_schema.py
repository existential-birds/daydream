"""Raw contracts reject incomplete evidence and preserve independent reply digests."""
import hashlib
import json
from typing import Any

import jsonschema
import pytest

from daydream.dataset import parse_observation, parse_run, run_record_schema
from daydream.dataset.scoring import capture_scoring
from daydream.training.labeler_versions import reply_evidence_digest
from daydream.training.reward import ScoringInputs
from tests.harness.dataset import observation, run_record


def test_observation_roundtrip_separates_redacted_text_from_semantic_reply_digest() -> None:
    replies = [{"reply_id": "reply-9", "body_sha256": "a" * 64, "disposition": "rejected"}]
    text = "Use [REDACTED] instead."
    raw = observation(semantic_evidence=replies, evidence_digest=reply_evidence_digest(replies),
        evidence_digest_scheme="reply-evidence-v1", correction={
            "status": "available", "source_reply_id": "reply-9", "body_sha256": "a" * 64, "text": text,
            "captured_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "redaction_provenance": {"policy": "shared-redactor-v1", "redacted": True}})
    restored = parse_observation(raw)
    assert restored["semantic_evidence"] == replies
    assert restored["correction"] is not None and restored["correction"]["text"] == text
    assert restored["correction"]["body_sha256"] == "a" * 64
    assert restored["correction"]["captured_sha256"] != restored["correction"]["body_sha256"]
    assert restored["evidence_digest"] == reply_evidence_digest(replies)
    for changed in ({"correction": {**raw["correction"], "text": "changed"}},
                    {"semantic_evidence": [{"reply_id": "changed"}]}):
        with pytest.raises(ValueError):
            parse_observation({**raw, **changed})


@pytest.mark.parametrize("role", ["rater", "adjudicator"])
def test_model_identity_cannot_roundtrip_as_human_decision(role: str) -> None:
    raw = observation(author="gpt-6", role=role)
    with pytest.raises(ValueError, match="record"):
        parse_observation(raw)
    assert parse_observation({**raw, "role": "model-suggested", "review_required": False})["review_required"]


@pytest.mark.parametrize("field", ["original_task", "trajectories", "findings", "verification", "scoring"])
def test_available_run_sections_require_complete_semantic_payload(field: str) -> None:
    raw = run_record()
    raw[field] = {"status": "available", "value": {}}
    with pytest.raises(ValueError, match="invalid RunRecord"):
        parse_run(raw)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, run_record_schema())
    raw[field] = {"status": "unavailable", "reason": "not acquired"}
    jsonschema.validate(parse_run(raw), run_record_schema())


@pytest.mark.parametrize("registered_child", [False, True])
def test_trajectory_membership_preserves_invocations_and_unproduced_registered_children(registered_child: bool) -> None:
    summary: dict[str, Any] = ({"trajectory_id": "run-1-child", "invocations": []} if registered_child else
               {"trajectory_id": "run-1", "invocation_id": "attempt-1", "step_ids": [1], "phase": "review"})
    raw = run_record(outcome="interrupted", trajectories={"status": "available", "value": {
        "root_trajectory_id": "run-1", "status": "partial", "cutoff_at": "2026-10-04T12:01:00Z",
        "documents": [{"schema_version": "ATIF-v1.7", "session_id": "run-1", "trajectory_id": "run-1",
                       "agent": {"name": "daydream", "version": "test"},
                       "steps": [{"step_id": 1, "source": "user", "message": "review"}],
                       "extra": {"subtrajectories": [summary]}}]}})
    restored = parse_run(raw)
    assert restored["outcome"] == "interrupted"
    assert restored["trajectories"]["value"]["documents"][0]["extra"]["subtrajectories"] == [summary]
    if not registered_child:
        summary["step_ids"] = [9]
        with pytest.raises(ValueError, match="invalid RunRecord"):
            parse_run(raw)


@pytest.mark.parametrize("version", [None, "daydream.run.v2"])
def test_read_requires_supported_explicit_version_and_withholds_input(version: str | None) -> None:
    raw = run_record(run_id="PRIVATE-CREDENTIAL")
    if version is None:
        raw.pop("schema_version")
    else:
        raw["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version") as error:
        parse_run(raw)
    assert "PRIVATE-CREDENTIAL" not in str(error.value)


def test_persisted_reward_retains_uncomputable_correctness_and_rejects_missing_policy() -> None:
    scoring = capture_scoring(ScoringInputs(None, True, 6), "review")
    raw = run_record(scoring={"status": "available", "value": scoring})
    restored = parse_run(json.dumps(parse_run(raw)))["scoring"]["value"]
    assert restored["persisted_breakdown"]["composite"] is None
    assert restored["persisted_breakdown"]["correctness_per_finding"] is None
    assert restored["posterior_cost"] is None
    scoring["reward_policy"]["configuration"] = {}
    with pytest.raises(ValueError, match="invalid RunRecord"):
        parse_run(raw)


def test_rejected_unknown_field_names_do_not_leak_into_schema_diagnostics() -> None:
    with pytest.raises(ValueError) as error:
        parse_run(run_record(**{"PRIVATE-CREDENTIAL-KEY": "PRIVATE-CREDENTIAL-VALUE"}))
    assert "PRIVATE-CREDENTIAL" not in str(error.value)
