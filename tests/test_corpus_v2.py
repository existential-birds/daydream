"""Tests for corpus-v2: schema artifact, version constants, and projection."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from daydream.training.schema import (
    TRAINING_SCHEMA_V1_PATH,
    TRAINING_SCHEMA_V2_PATH,
    TRAINING_SCHEMA_V2_VERSION,
)


def _minimal_v2_record() -> dict[str, object]:
    """Smallest dict passing every required field of schema/v2.json."""
    return {
        "schema_version": "2",
        "record_id": "a" * 64,
        "record_type": "outcome-finding",
        "tier": "gold",
        "session_id": "sess-1",
        "trajectory_id": "traj-1",
        "task_segment": "seg-1",
        "finding_fingerprint": "fp-1",
        "disposition": "accepted",
        "profile": {
            "profile_schema_version": None,
            "profile_name": None,
            "profile_source_kind": None,
            "profile_digest": None,
        },
        "stack": "python",
        "lineage": {
            "hub_commit": None,
            "curation_id": None,
            "content_digests": [],
            "labeler_policy_version": None,
            "reply_classifier_version": None,
            "rubric_schema_version": None,
            "as_of": None,
            "valid_at": None,
            "split": "train",
            "exclusion_reason": None,
            "license_decision": None,
        },
    }


def test_v2_schema_ships_alongside_v1_and_validates_golden_record() -> None:
    assert Path(TRAINING_SCHEMA_V2_PATH).exists()
    assert Path(TRAINING_SCHEMA_V2_PATH) != Path(TRAINING_SCHEMA_V1_PATH)  # v1 untouched
    schema = json.loads(Path(TRAINING_SCHEMA_V2_PATH).read_text())
    validator = Draft202012Validator(schema)
    record = _minimal_v2_record()
    errors = sorted(validator.iter_errors(record), key=lambda e: e.json_path)
    assert errors == [], [e.message for e in errors]


def test_v2_version_constant_is_two() -> None:
    assert TRAINING_SCHEMA_V2_VERSION == "2"
