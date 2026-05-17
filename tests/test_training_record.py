"""Tests for daydream.training.export record and span builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from daydream.training.export import _build_record, _build_spans

SCHEMA_PATH = Path(__file__).parent.parent / "daydream" / "training" / "schema" / "v1.json"


def _make_manifest_row(**overrides: Any) -> dict[str, Any]:
    """Build a minimal manifest_row dict for _build_record tests."""
    row: dict[str, Any] = {
        "session_id": "test-session-001",
        "skill": "beagle-python:review-python",
        "repo_slug": "someorg/python-app",
        "branch": "feat/foo",
        "base_branch": "main",
        "head_sha": "abc123def456",
        "grounding_rate": 0.87,
        "outcome_labels": json.dumps(["accepted"]),
        "archive_path": "",
    }
    row.update(overrides)
    return row


def test_spans_emits_reason_and_act_for_agent_step() -> None:
    trajectory = {
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "reasoning_content": "thinking",
                "message": "",
                "tool_calls": [{"name": "Bash", "arguments": {}}],
            }
        ]
    }
    assert _build_spans(trajectory) == [
        {"step_id": 1, "kind": "REASON", "content_path": "steps[0].reasoning_content"},
        {"step_id": 1, "kind": "ACT", "content_path": "steps[0].tool_calls"},
    ]


def test_spans_prefers_reasoning_content_over_message() -> None:
    trajectory = {
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "reasoning_content": "A",
                "message": "B",
            }
        ]
    }
    spans = _build_spans(trajectory)
    reason_spans = [s for s in spans if s["kind"] == "REASON"]
    assert len(reason_spans) == 1
    assert reason_spans[0]["content_path"] == "steps[0].reasoning_content"


def test_spans_falls_back_to_message_when_no_reasoning_content() -> None:
    trajectory = {
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "reasoning_content": None,
                "message": "some text",
            }
        ]
    }
    spans = _build_spans(trajectory)
    reason_spans = [s for s in spans if s["kind"] == "REASON"]
    assert len(reason_spans) == 1
    assert reason_spans[0]["content_path"].endswith(".message")


def test_spans_skips_user_and_system_steps() -> None:
    trajectory = {
        "steps": [
            {"step_id": 1, "source": "user", "message": "please review"},
            {"step_id": 2, "source": "system", "message": "ready"},
            {
                "step_id": 3,
                "source": "agent",
                "reasoning_content": "thinking",
                "message": "",
            },
        ]
    }
    spans = _build_spans(trajectory)
    assert len(spans) == 1
    assert spans[0]["step_id"] == 3
    assert spans[0]["kind"] == "REASON"


def test_spans_skips_copied_context_steps() -> None:
    trajectory = {
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "reasoning_content": "thinking",
                "message": "hello",
                "tool_calls": [{"name": "Bash", "arguments": {}}],
                "is_copied_context": True,
            }
        ]
    }
    assert _build_spans(trajectory) == []


def test_spans_empty_for_trajectory_with_no_agent_steps() -> None:
    assert _build_spans({"steps": []}) == []
    assert _build_spans({}) == []


def test_record_validates_against_v1_schema() -> None:
    manifest_row = _make_manifest_row()
    trajectory = {
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "reasoning_content": "thinking through the change",
                "message": "",
                "tool_calls": [{"name": "Read", "arguments": {"file_path": "x.py"}}],
            }
        ]
    }
    record = _build_record(manifest_row, trajectory, stack="python")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(record, schema)


def test_record_fix_diff_ref_reflects_archive_state(tmp_path: Path) -> None:
    manifest_row = _make_manifest_row(archive_path=str(tmp_path))
    trajectory: dict[str, Any] = {"steps": []}

    # No diff.patch on disk yet.
    record = _build_record(manifest_row, trajectory, stack="python")
    assert record["fix_diff_ref"]["available"] is False
    assert record["fix_diff_ref"]["archive_relative_path"] == "diff.patch"

    # Write the diff and confirm the next call sees it.
    (tmp_path / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    record = _build_record(manifest_row, trajectory, stack="python")
    assert record["fix_diff_ref"]["available"] is True
