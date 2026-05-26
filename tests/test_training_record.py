"""Tests for daydream.training.corpus record and span builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from daydream.training.corpus import _build_record, _build_spans

SCHEMA_PATH = Path(__file__).parent.parent / "daydream" / "training" / "schema" / "v1.json"

# Minimal ATIF v1.6-shaped trajectory used by record builder tests that only
# care about manifest-driven fields (review_output, code_context, etc.).
_MIN_TRAJECTORY: dict[str, Any] = {"steps": []}


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


def test_record_code_context_sourced_from_manifest_dict(tmp_path: Path) -> None:
    """``base_sha`` + ``changed_files`` are read from the manifest dict."""
    manifest_row = _make_manifest_row(archive_path=str(tmp_path))
    manifest = {
        "code_context": {
            "base_sha": "deadbeef" * 5,
            "head_sha": manifest_row["head_sha"],
            "base_branch": "main",
            "branch": "feat/foo",
            "changed_files": ["src/a.py", "src/b.py"],
        }
    }
    trajectory: dict[str, Any] = {"steps": []}

    record = _build_record(manifest_row, trajectory, stack="python", manifest=manifest)

    assert record["code_context"]["base_sha"] == "deadbeef" * 5
    assert record["code_context"]["changed_files"] == ["src/a.py", "src/b.py"]


def test_record_code_context_falls_back_when_manifest_missing(tmp_path: Path) -> None:
    """A manifest without ``code_context`` yields base_sha=None, changed_files=[]."""
    manifest_row = _make_manifest_row(archive_path=str(tmp_path))
    record = _build_record(manifest_row, {"steps": []}, stack=None, manifest={})

    assert record["code_context"]["base_sha"] is None
    assert record["code_context"]["changed_files"] == []
    # Scalar fields still come from the row.
    assert record["code_context"]["head_sha"] == manifest_row["head_sha"]
    assert record["code_context"]["branch"] == manifest_row["branch"]


def test_record_review_output_loaded_from_archive(tmp_path: Path) -> None:
    """When ``review-output.md`` is present, its text lands in ``review_output``."""
    (tmp_path / "review-output.md").write_text("# Review\nLooks good.\n", encoding="utf-8")
    manifest_row = _make_manifest_row(archive_path=str(tmp_path))

    record = _build_record(manifest_row, {"steps": []}, stack="python")
    assert record["review_output"] == "# Review\nLooks good.\n"


def test_record_review_output_none_when_file_absent(tmp_path: Path) -> None:
    manifest_row = _make_manifest_row(archive_path=str(tmp_path))
    record = _build_record(manifest_row, {"steps": []}, stack="python")
    assert record["review_output"] is None


def test_stack_for_skill_resolves_short_name() -> None:
    """Manifests store short skill names (e.g. 'python'); the stack
    derivation must round-trip them."""
    from daydream.training.corpus import _stack_for_skill

    assert _stack_for_skill("python") == "python"
    assert _stack_for_skill("react") == "react"
    assert _stack_for_skill("beagle-python:review-python") == "python"
    assert _stack_for_skill(None) is None
    assert _stack_for_skill("unknown-stack") is None


def test_build_record_reads_review_output_from_deep_subdir(tmp_path: Path) -> None:
    """Deep-mode archives only have review-output.md under deep/."""
    archive = tmp_path / "archive"
    (archive / "deep").mkdir(parents=True)
    (archive / "deep" / "review-output.md").write_text("# Deep review\n")
    row = {"session_id": "abc", "archive_path": str(archive), "outcome_labels": "[]"}
    record = _build_record(row, _MIN_TRAJECTORY, stack="python", manifest={})
    assert record["review_output"] == "# Deep review\n"


def test_build_record_prefers_root_review_output_over_deep(tmp_path: Path) -> None:
    """When both exist, root wins (deep is the fallback)."""
    archive = tmp_path / "archive"
    (archive / "deep").mkdir(parents=True)
    (archive / "review-output.md").write_text("# Root\n")
    (archive / "deep" / "review-output.md").write_text("# Deep\n")
    row = {"session_id": "abc", "archive_path": str(archive), "outcome_labels": "[]"}
    record = _build_record(row, _MIN_TRAJECTORY, stack="python", manifest={})
    assert record["review_output"] == "# Root\n"


def test_build_record_surfaces_rubric_when_present(tmp_path: Path) -> None:
    row = {"session_id": "abc", "archive_path": str(tmp_path),
           "outcome_labels": '["accepted"]',
           "rubric_json": '{"pr_merge": {"merged": true}, "posterior_source": "pr_review"}'}
    record = _build_record(row, _MIN_TRAJECTORY, stack="python", manifest={})
    assert record["rubric"] == {"pr_merge": {"merged": True}, "posterior_source": "pr_review"}
    assert record["posterior_source"] == "pr_review"


def test_build_record_omits_rubric_when_absent(tmp_path: Path) -> None:
    row = {"session_id": "abc", "archive_path": str(tmp_path), "outcome_labels": "[]"}
    record = _build_record(row, _MIN_TRAJECTORY, stack="python", manifest={})
    assert "rubric" not in record
    assert "posterior_source" not in record


def test_build_record_propagates_local_branch_source(tmp_path: Path) -> None:
    row = {"session_id": "abc", "archive_path": str(tmp_path),
           "outcome_labels": '["accepted"]',
           "rubric_json": '{"posterior_source": "local_branch"}'}
    record = _build_record(row, _MIN_TRAJECTORY, stack="python", manifest={})
    assert record["posterior_source"] == "local_branch"
