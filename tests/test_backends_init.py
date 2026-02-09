# tests/test_backends_init.py
"""Tests for backend protocol, event types, and factory."""


import pytest

from daydream.backends import (
    ClaudeBackend,
    ContinuationToken,
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    create_backend,
)


def test_text_event_has_text_field():
    event = TextEvent(text="hello")
    assert event.text == "hello"


def test_thinking_event_has_text_field():
    event = ThinkingEvent(text="reasoning...")
    assert event.text == "reasoning..."


def test_tool_start_event_fields():
    event = ToolStartEvent(id="t1", name="Bash", input={"command": "ls"})
    assert event.id == "t1"
    assert event.name == "Bash"
    assert event.input == {"command": "ls"}


def test_tool_result_event_fields():
    event = ToolResultEvent(id="t1", output="file.py", is_error=False)
    assert event.id == "t1"
    assert event.output == "file.py"
    assert event.is_error is False


def test_cost_event_fields():
    event = CostEvent(cost_usd=0.01, input_tokens=100, output_tokens=50)
    assert event.cost_usd == 0.01
    assert event.input_tokens == 100
    assert event.output_tokens == 50


def test_cost_event_nullable_fields():
    event = CostEvent(cost_usd=None, input_tokens=None, output_tokens=None)
    assert event.cost_usd is None


def test_continuation_token_fields():
    token = ContinuationToken(backend="codex", data={"thread_id": "abc"})
    assert token.backend == "codex"
    assert token.data == {"thread_id": "abc"}


def test_result_event_fields():
    token = ContinuationToken(backend="codex", data={})
    event = ResultEvent(structured_output={"key": "val"}, continuation=token)
    assert event.structured_output == {"key": "val"}
    assert event.continuation is token


def test_result_event_nullable():
    event = ResultEvent(structured_output=None, continuation=None)
    assert event.structured_output is None
    assert event.continuation is None


def test_create_backend_claude_default():
    backend = create_backend("claude")
    assert isinstance(backend, ClaudeBackend)
    assert backend.model == "opus"


def test_create_backend_claude_custom_model():
    backend = create_backend("claude", model="sonnet")
    assert isinstance(backend, ClaudeBackend)
    assert backend.model == "sonnet"


def test_create_backend_codex_default():
    backend = create_backend("codex")
    # Import here to avoid circular — just check it's not ClaudeBackend
    from daydream.backends.codex import CodexBackend
    assert isinstance(backend, CodexBackend)
    assert backend.model == "gpt-5.3-codex"


def test_create_backend_codex_custom_model():
    backend = create_backend("codex", model="o3-pro")
    from daydream.backends.codex import CodexBackend
    assert isinstance(backend, CodexBackend)
    assert backend.model == "o3-pro"


def test_create_backend_invalid_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("invalid")
