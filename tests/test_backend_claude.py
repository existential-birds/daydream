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
    assert tool_start_events[0].id == tool_result_events[0].id  # Verify ID correlation


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


class MockClaudeSDKClientCapture:
    """Mock client that captures the options it was constructed with."""

    captured_options = None

    def __init__(self, options: Any = None):
        MockClaudeSDKClientCapture.captured_options = options
        self._prompt: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        yield MockAssistantMessage(content=[MockTextBlock(text="OK")])
        yield MockResultMessage(total_cost_usd=0.01)


@pytest.mark.asyncio
async def test_execute_passes_agents_dict_to_options(patch_sdk):
    """Agents dict must reach ClaudeAgentOptions with original keys preserved verbatim."""
    from claude_agent_sdk.types import AgentDefinition

    patch_sdk(MockClaudeSDKClientCapture)
    backend = ClaudeBackend(model="opus")

    pattern_scanner = AgentDefinition(
        description="pattern scanner",
        prompt="scan patterns",
        tools=["Read", "Grep"],
        model="sonnet",
    )
    dependency_tracer = AgentDefinition(
        description="dependency tracer",
        prompt="trace deps",
        tools=["Read", "Grep"],
        model="sonnet",
    )

    agents = {
        "pattern-scanner": pattern_scanner,
        "dependency-tracer": dependency_tracer,
    }

    events = []
    async for event in backend.execute(Path("/tmp"), "Go", agents=agents):
        events.append(event)

    opts = MockClaudeSDKClientCapture.captured_options
    assert opts is not None
    assert opts.agents == {
        "pattern-scanner": pattern_scanner,
        "dependency-tracer": dependency_tracer,
    }
    # No key rewriting
    assert "explorer-0" not in opts.agents
    assert "explorer-1" not in opts.agents


@pytest.mark.asyncio
async def test_execute_passes_none_when_no_agents(patch_sdk):
    """When agents=None, ClaudeAgentOptions should not carry an agents dict."""
    patch_sdk(MockClaudeSDKClientCapture)
    backend = ClaudeBackend(model="opus")

    events = []
    async for event in backend.execute(Path("/tmp"), "Go"):
        events.append(event)

    opts = MockClaudeSDKClientCapture.captured_options
    assert opts is not None
    agents_val = getattr(opts, "agents", None)
    assert agents_val is None


def test_backend_protocol_agents_param_is_dict_typed():
    """The Backend protocol's execute.agents annotation must be dict[str, AgentDefinition]."""
    from daydream.backends import Backend

    annotations = Backend.execute.__annotations__
    assert "agents" in annotations
    annotation = annotations["agents"]
    # Annotation may be a string (from __future__ annotations) or a real type
    annotation_str = annotation if isinstance(annotation, str) else repr(annotation)
    assert "dict[str, AgentDefinition]" in annotation_str


