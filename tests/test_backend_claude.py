# tests/test_backend_claude.py
"""Tests for ClaudeBackend."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.claude import ClaudeBackend

# --- Mock SDK types (same pattern as test_integration.py) ---


@dataclass
class MockTextBlock:
    text: str


@dataclass
class MockToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] | None = None


@dataclass
class MockToolResultBlock:
    tool_use_id: str
    content: str | None = None
    is_error: bool = False


@dataclass
class MockThinkingBlock:
    thinking: str


@dataclass
class MockAssistantMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class MockUserMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class MockResultMessage:
    total_cost_usd: float | None = 0.001
    structured_output: Any = None


class MockClaudeSDKClient:
    """Mock client that yields a configurable message sequence."""

    def __init__(self, options: Any = None):
        self.options = options
        self._prompt: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        yield MockAssistantMessage(content=[MockTextBlock(text="Hello world")])
        yield MockResultMessage(total_cost_usd=0.05, structured_output=None)


class MockClaudeSDKClientWithTools:
    """Mock client that yields tool use messages."""

    def __init__(self, options: Any = None):
        self.options = options
        self._prompt: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        yield MockAssistantMessage(content=[
            MockThinkingBlock(thinking="Let me think..."),
        ])
        yield MockAssistantMessage(content=[
            MockTextBlock(text="I'll run a command."),
        ])
        yield MockAssistantMessage(content=[
            MockToolUseBlock(id="tool-1", name="Bash", input={"command": "ls"}),
        ])
        yield MockUserMessage(content=[
            MockToolResultBlock(tool_use_id="tool-1", content="file.py", is_error=False),
        ])
        yield MockResultMessage(total_cost_usd=0.10)


class MockClaudeSDKClientStructured:
    """Mock client that returns structured output."""

    def __init__(self, options: Any = None):
        self.options = options
        self._prompt: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        yield MockAssistantMessage(content=[MockTextBlock(text="Parsed.")])
        yield MockResultMessage(
            total_cost_usd=0.02,
            structured_output={"issues": [{"id": 1, "description": "Fix X", "file": "a.py", "line": 10}]},
        )


@pytest.fixture
def patch_sdk(monkeypatch):
    """Return a function that patches the SDK imports in claude.py."""
    def _patch(client_class):
        monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", client_class)
        monkeypatch.setattr("daydream.backends.claude.AssistantMessage", MockAssistantMessage)
        monkeypatch.setattr("daydream.backends.claude.UserMessage", MockUserMessage)
        monkeypatch.setattr("daydream.backends.claude.ResultMessage", MockResultMessage)
        monkeypatch.setattr("daydream.backends.claude.TextBlock", MockTextBlock)
        monkeypatch.setattr("daydream.backends.claude.ThinkingBlock", MockThinkingBlock)
        monkeypatch.setattr("daydream.backends.claude.ToolUseBlock", MockToolUseBlock)
        monkeypatch.setattr("daydream.backends.claude.ToolResultBlock", MockToolResultBlock)
    return _patch


@pytest.mark.asyncio
async def test_execute_yields_text_and_result(patch_sdk):
    patch_sdk(MockClaudeSDKClient)
    backend = ClaudeBackend(model="opus")
    events = []
    async for event in backend.execute(Path("/tmp"), "Say hello"):
        events.append(event)

    # Should have TextEvent, CostEvent, ResultEvent
    text_events = [e for e in events if isinstance(e, TextEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    result_events = [e for e in events if isinstance(e, ResultEvent)]

    assert len(text_events) == 1
    assert text_events[0].text == "Hello world"
    assert len(cost_events) == 1
    assert cost_events[0].cost_usd == 0.05
    assert len(result_events) == 1
    assert result_events[0].structured_output is None
    assert result_events[0].continuation is None


@pytest.mark.asyncio
async def test_execute_yields_tool_events(patch_sdk):
    patch_sdk(MockClaudeSDKClientWithTools)
    backend = ClaudeBackend(model="opus")
    events = []
    async for event in backend.execute(Path("/tmp"), "Run ls"):
        events.append(event)

    thinking_events = [e for e in events if isinstance(e, ThinkingEvent)]
    tool_start_events = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_result_events = [e for e in events if isinstance(e, ToolResultEvent)]

    assert len(thinking_events) == 1
    assert thinking_events[0].text == "Let me think..."
    assert len(tool_start_events) == 1
    assert tool_start_events[0].name == "Bash"
    assert tool_start_events[0].input == {"command": "ls"}
    assert len(tool_result_events) == 1
    assert tool_result_events[0].output == "file.py"
    assert tool_result_events[0].is_error is False


@pytest.mark.asyncio
async def test_execute_structured_output(patch_sdk):
    patch_sdk(MockClaudeSDKClientStructured)
    backend = ClaudeBackend(model="opus")
    events = []
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    async for event in backend.execute(Path("/tmp"), "Parse", output_schema=schema):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [{"id": 1, "description": "Fix X", "file": "a.py", "line": 10}]
    }


def test_format_skill_invocation_full_key():
    backend = ClaudeBackend()
    result = backend.format_skill_invocation("beagle-python:review-python")
    assert result == "/beagle-python:review-python"


def test_format_skill_invocation_with_args():
    backend = ClaudeBackend()
    result = backend.format_skill_invocation("beagle-core:fetch-pr-feedback", "--pr 42 --bot mybot")
    assert result == "/beagle-core:fetch-pr-feedback --pr 42 --bot mybot"


