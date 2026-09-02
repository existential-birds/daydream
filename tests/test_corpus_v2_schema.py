"""Schema tests for the extended corpus-v2 record schema (finding_text / task_identity)."""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

SCHEMA = json.loads((Path(__file__).parent.parent / "daydream/training/schema/v2.json").read_text())


def _base_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "2",
        "record_id": "a" * 64,
        "record_type": "outcome-finding",
        "tier": "gold",
        "session_id": "session-1",
        "trajectory_id": "traj-1",
        "task_segment": "seg-1",
        "finding_fingerprint": "b" * 64,
        "disposition": "accepted",
        "profile": {
            "profile_schema_version": "1",
            "profile_name": "default",
            "profile_source_kind": "builtin",
            "profile_digest": "c" * 64,
        },
        "stack": "python",
        "outcome_label": "correct",
        "lineage": {
            "hub_commit": "d" * 40,
            "curation_id": "cur-1",
            "content_digests": ["e" * 64],
            "labeler_policy_version": "1",
            "reply_classifier_version": "1",
            "rubric_schema_version": "1",
            "as_of": "2026-01-01T00:00:00Z",
            "valid_at": "2026-01-01T00:00:00Z",
            "split": "train",
            "exclusion_reason": None,
            "repo_slug": "owner/repo",
            "license_decision": {
                "status": "admitted",
                "reason_code": None,
                "spdx_id": "MIT",
                "policy_version": "1",
                "evidence_ref": "evidence",
                "repo_slug": "owner/repo",
            },
        },
    }
    record.update(overrides)
    return record


TASK_IDENTITY = {
    "repo_slug": "owner/repo",
    "source": "curation",
    "base_sha": "1" * 40,
    "head_sha": "2" * 40,
    "diff_digest": "3" * 64,
    "diff_ref": {"content_digest": "4" * 64, "path": "batches/s1/diff.patch"},
    "replay_verification": {"status": "passed"},
}


def test_process_trace_record_validates() -> None:
    record = _base_record(
        record_type="process-trace",
        tier="silver",
        disposition="ambiguous",
        prompt="prompt text",
        completion="completion text",
        task_identity=TASK_IDENTITY,
    )
    jsonschema.validate(record, SCHEMA)


def test_task_only_record_validates() -> None:
    record = _base_record(
        record_type="task-only",
        tier="task-only",
        disposition="missing",
        finding_fingerprint=None,
        task_identity=TASK_IDENTITY,
    )
    jsonschema.validate(record, SCHEMA)


def test_gold_outcome_finding_with_finding_text_and_task_identity_validates() -> None:
    record = _base_record(
        finding_text="The loop mutates the list while iterating.",
        finding_text_sha256="5" * 64,
        task_identity=TASK_IDENTITY,
    )
    record["lineage"]["diff_digest"] = "6" * 64
    record["lineage"]["diff_ref"] = {"content_digest": "4" * 64, "path": "batches/s1/diff.patch"}
    jsonschema.validate(record, SCHEMA)


def test_task_identity_with_bad_base_sha_is_rejected() -> None:
    bad_identity = dict(TASK_IDENTITY, base_sha="short")
    record = _base_record(record_type="task-only", tier="task-only", task_identity=bad_identity)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(record, SCHEMA)
