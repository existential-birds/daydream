"""Public raw-evidence serialization preserves absent and intentionally empty inputs."""

import hashlib
from typing import Any

import pytest

from daydream.dataset.schema import Evidence, RunRecord, parse_run, serialize_record


def test_run_schema_retains_original_diff_and_explicit_absent_evidence() -> None:
    diff = ""
    run = RunRecord(
        run_id="run-1",
        captured_at="2026-10-04T12:00:00+00:00",
        outcome="success",
        original_task=Evidence(status="available", value={
            "diff": diff,
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "analyzed_revision": {"head_sha": "a" * 40, "merge_base_sha": "b" * 40, "diff_key": "c" * 64},
            "repository": {"repo_slug": None, "remote_url": None, "host": None},
            "pr": {"number": None, "repo": None}, "changed_files": [],
            "input_scope": "committed_and_tracked_worktree", "dirty_tracked": False,
            "untracked_files_included": False,
        }),
    )
    restored = parse_run(serialize_record(run))
    assert restored.original_task.status == "available"
    assert restored.original_task.value["diff"] == ""
    assert restored.verification.status == "unproduced"
    assert restored.verification.value is None
    with pytest.raises(ValueError, match="schema_version"):
        parse_run({**serialize_record(run), "schema_version": "daydream.run.v2"})


def test_observation_roundtrip_separates_redacted_text_from_semantic_reply_digest() -> None:
    from daydream.dataset.schema import parse_observation
    from daydream.training.labeler_versions import reply_evidence_digest

    replies = [{"reply_id": "reply-9", "body_sha256": "a" * 64, "disposition": "rejected"}]
    redacted = "Use [REDACTED] instead."
    observation: dict[str, Any] = {
        "schema_version": "daydream.observation.v1", "observation_id": "obs-1",
        "run_id": "run-1", "item_uid": "item:1", "valid_at": "2026-10-04T12:00:00Z",
        "observed_at": "2026-10-04T13:00:00Z", "source": "github-reply",
        "author": "maintainer", "role": "rater", "policy_version": "980-policy-r1",
        "rubric_version": "984-adjudicate-r1", "classifier_version": "980-classifier-r1",
        "evidence_digest": reply_evidence_digest(replies),
        "evidence_digest_scheme": "reply-evidence-v1", "semantic_evidence": replies,
        "payload": {"type": "finding-judgment", "disposition": "rejected", "rationale": "Wrong target"},
        "correction": {
            "status": "available", "text": redacted,
            "captured_sha256": hashlib.sha256(redacted.encode()).hexdigest(),
            "body_sha256": "a" * 64, "source_reply_id": "reply-9",
            "redaction_provenance": {"policy": "shared-redactor-v1", "redacted": True},
        },
    }
    restored = parse_observation(observation)
    assert serialize_record(restored)["semantic_evidence"] == replies
    assert restored.correction is not None
    assert restored.correction.text == redacted
    assert restored.correction.body_sha256 == "a" * 64
    assert restored.correction.captured_sha256 != restored.correction.body_sha256
    assert restored.evidence_digest == reply_evidence_digest(replies)
    assert restored.payload.type == "finding-judgment"
    with pytest.raises(ValueError, match="correction"):
        parse_observation({**observation, "correction": {**observation["correction"], "text": "changed"}})
    with pytest.raises(ValueError, match="record"):
        parse_observation({**observation, "semantic_evidence": [{"reply_id": "changed"}]})


@pytest.mark.parametrize("role", ["rater", "adjudicator"])
def test_model_identity_cannot_roundtrip_as_human_decision(role: str) -> None:
    from daydream.dataset.schema import parse_observation, semantic_evidence_digest

    evidence = {"reason": "model proposal"}
    observation: dict[str, Any] = {
        "schema_version": "daydream.observation.v1", "observation_id": "obs-model",
        "run_id": "run-1", "item_uid": "item:1",
        "valid_at": "2026-10-04T12:00:00Z", "observed_at": "2026-10-04T13:00:00Z",
        "source": "model", "author": "gpt-6", "role": role, "policy_version": "policy-1",
        "rubric_version": "rubric-1", "evidence_digest": semantic_evidence_digest(evidence),
        "semantic_evidence": evidence,
        "payload": {"type": "finding-judgment", "disposition": "accepted", "rationale": "suggestion"},
    }
    with pytest.raises(ValueError, match="record"):
        parse_observation(observation)
    restored = parse_observation({**observation, "role": "model-suggested", "review_required": False})
    assert restored.review_required is True


@pytest.mark.parametrize("field", ["original_task", "trajectories", "findings", "verification", "scoring"])
def test_available_run_sections_require_complete_semantic_payload(field: str) -> None:
    run = RunRecord(run_id="run-1", captured_at="2026-10-04T12:00:00Z", outcome="failed")
    with pytest.raises(ValueError, match="invalid RunRecord"):
        parse_run({**serialize_record(run), field: {"status": "available", "value": {}}})


def test_published_json_schema_checks_the_available_semantic_envelopes() -> None:
    import jsonschema

    from daydream.dataset.schema import run_record_schema

    run = RunRecord(run_id="run-1", captured_at="2026-10-04T12:00:00Z", outcome="failed")
    raw = serialize_record(run)
    jsonschema.validate(raw, run_record_schema())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**raw, "findings": {"status": "available", "value": {}}}, run_record_schema())


def test_trajectory_membership_preserves_producer_order_and_rejects_unknown_step() -> None:
    from copy import deepcopy

    invocation = {"trajectory_id": "run-1", "invocation_id": "attempt-1", "step_ids": [1],
                  "phase": "review", "started_at": "2026-10-04T12:00:00Z", "ended_at": None}
    raw: dict[str, Any] = {
        "schema_version": "daydream.run.v1", "run_id": "run-1",
        "captured_at": "2026-10-04T12:01:00Z", "outcome": "interrupted",
        "trajectories": {"status": "available", "value": {
            "root_trajectory_id": "run-1", "status": "partial", "cutoff_at": "2026-10-04T12:01:00Z",
            "documents": [{"schema_version": "ATIF-v1.7", "session_id": "run-1", "trajectory_id": "run-1",
                           "agent": {"name": "daydream", "version": "test"},
                           "steps": [{"step_id": 1, "source": "user", "message": "review"}],
                           "extra": {"subtrajectories": [invocation]}}],
        }},
    }
    restored = parse_run(raw)
    assert restored.trajectories.value["documents"][0]["extra"]["subtrajectories"] == [invocation]
    changed = deepcopy(raw)
    changed["trajectories"]["value"]["documents"][0]["extra"]["subtrajectories"][0]["step_ids"] = [9]
    with pytest.raises(ValueError, match="invalid RunRecord"):
        parse_run(changed)


def test_interrupted_trajectory_preserves_registered_child_without_produced_document() -> None:
    raw = {
        "schema_version": "daydream.run.v1", "run_id": "run-1",
        "captured_at": "2026-10-04T12:01:00Z", "outcome": "interrupted",
        "trajectories": {"status": "available", "value": {
            "root_trajectory_id": "run-1", "status": "partial", "cutoff_at": "2026-10-04T12:01:00Z",
            "documents": [{"schema_version": "ATIF-v1.7", "session_id": "run-1", "trajectory_id": "run-1",
                           "agent": {"name": "daydream", "version": "test"},
                           "steps": [{"step_id": 1, "source": "user", "message": "review"}],
                           "extra": {"subtrajectories": [{"trajectory_id": "run-1-child", "invocations": []}]}}],
        }},
    }
    restored = parse_run(raw)
    assert restored.outcome == "interrupted"
    assert restored.trajectories.value["documents"][0]["extra"]["subtrajectories"][0]["trajectory_id"] == "run-1-child"


def test_read_requires_an_explicit_version_and_diagnostics_withhold_input() -> None:
    with pytest.raises(ValueError, match="schema_version") as error:
        parse_run({"run_id": "PRIVATE-CREDENTIAL", "captured_at": "2026-10-04T12:00:00Z", "outcome": "failed"})
    assert "PRIVATE-CREDENTIAL" not in str(error.value)


def test_persisted_reward_retains_uncomputable_correctness_and_rejects_missing_policy() -> None:
    from daydream.training.reward import DEFAULT_WEIGHTS, REWARD_VERSION, ScoringInputs, score_trajectory

    scoring: dict[str, Any] = {
        "verifier_verdicts": None, "format_valid": True, "review_text": "review", "length": 6,
        "reward_policy": {"version": REWARD_VERSION, "configuration": {
            "w_len": DEFAULT_WEIGHTS.w_len, "w_fp": DEFAULT_WEIGHTS.w_fp,
            "len_tau": DEFAULT_WEIGHTS.len_tau, "len_scale": DEFAULT_WEIGHTS.len_scale,
            "verdict_map": dict(DEFAULT_WEIGHTS.verdict_map), "fp_penalty_map": dict(DEFAULT_WEIGHTS.fp_penalty_map),
        }},
        "persisted_breakdown": score_trajectory(ScoringInputs(None, True, 6)).to_dict(), "posterior_cost": None,
    }
    run = RunRecord(run_id="run-1", captured_at="2026-10-04T12:00:00Z", outcome="success",
                    scoring=Evidence(status="available", value=scoring))
    restored = parse_run(serialize_record(run))
    assert restored.scoring.value["persisted_breakdown"]["composite"] is None
    assert restored.scoring.value["persisted_breakdown"]["correctness_per_finding"] is None
    assert restored.scoring.value["posterior_cost"] is None
    with pytest.raises(ValueError, match="invalid RunRecord"):
        parse_run({**serialize_record(run), "scoring": {"status": "available", "value": {
            **scoring, "reward_policy": {"version": REWARD_VERSION, "configuration": {}},
        }}})


def test_rejected_unknown_field_names_do_not_leak_into_schema_diagnostics() -> None:
    raw = serialize_record(RunRecord(run_id="run-1", captured_at="2026-10-04T12:00:00Z", outcome="failed"))
    with pytest.raises(ValueError) as error:
        parse_run({**raw, "PRIVATE-CREDENTIAL-KEY": "PRIVATE-CREDENTIAL-VALUE"})
    assert "PRIVATE-CREDENTIAL" not in str(error.value)
