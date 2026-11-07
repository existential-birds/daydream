"""Schema tests for the extended projection record schema (finding_text / task_identity)."""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

SCHEMA = json.loads((Path(__file__).parent.parent / "daydream/training/schema/record-schema.json").read_text())


def _base_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "3", "record_id": "a" * 64, "record_type": "outcome-finding", "tier": "gold",
        "session_id": "session-1", "trajectory_id": "traj-1", "task_segment": "seg-1", "finding_fingerprint": "b" * 64,
        "disposition": "accepted",
        "profile": {"profile_schema_version": "1", "profile_name": "default", "profile_source_kind": "builtin",
            "profile_digest": "c" * 64,
        }, "stack": "python", "outcome_label": "correct",
        "lineage": {"hub_commit": "d" * 40, "snapshot_id": "f" * 64,
            "source_identity_version": "record-snapshot-v2", "content_digests": ["e" * 64],
            "labeler_policy_version": "1", "reply_classifier_version": "1", "rubric_schema_version": "1",
            "as_of": "2026-01-01T00:00:00Z", "valid_at": "2026-01-01T00:00:00Z", "split": "train",
            "exclusion_reason": None, "repo_slug": "owner/repo",
        },
    }
    record.update(overrides)
    return record


TASK_IDENTITY = {"repo_slug": "owner/repo", "base_sha": "1" * 40, "head_sha": "2" * 40,
    "diff_digest": "3" * 64, "diff_ref": {"record_digest": "4" * 64, "run_id": "s1", "section": "original_task.diff"},
}


def test_task_identity_sha_schema_rejects_short_and_admits_content_stamps() -> None:
    """Short SHAs are rejected, while sha256-style 64-hex content-address stamps
    (v1 corpus shas are 64-hex) are admitted."""
    bad_identity = dict(TASK_IDENTITY, base_sha="short")
    record = _base_record(record_type="task-only", tier="task-only", task_identity=bad_identity)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(record, SCHEMA)
    identity = dict(TASK_IDENTITY, base_sha="f" * 64, head_sha="e" * 64)
    record = _base_record(record_type="task-only", tier="task-only", task_identity=identity)
    jsonschema.validate(record, SCHEMA)
