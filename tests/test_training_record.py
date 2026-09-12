"""Tests for daydream.training.corpus span and skill-decode helpers."""

from __future__ import annotations

from daydream.training.corpus import _build_spans


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


def test_stack_for_skill_resolves_short_name() -> None:
    """Manifests store short skill names (e.g. 'python'); the stack
    derivation must round-trip them."""
    from daydream.training.corpus import _stack_for_skill

    assert _stack_for_skill("python") == "python"
    assert _stack_for_skill("react") == "react"
    assert _stack_for_skill("beagle-python:review-python") == "python"
    assert _stack_for_skill(None) is None
    assert _stack_for_skill("unknown-stack") is None
