"""Claude SDK stand-ins must replace the runtime message/block classes because ClaudeBackend dispatches
with isinstance. patch_claude_sdk installs them; scripted_client replays messages and can capture
options and prompt. Optional model, message ID, usage, timing, and stop-reason fields match what the
backend reads.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class MockTextBlock:
    text: str


@dataclass
class MockThinkingBlock:
    thinking: str


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
class MockAssistantMessage:
    content: list[Any] = field(default_factory=list)
    # Defaults omit model, metrics, and continuation until the test supplies them.
    model: str | None = None
    message_id: str = ""
    usage: dict[str, Any] | None = None


@dataclass
class MockUserMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class MockResultMessage:
    total_cost_usd: float | None = 0.001
    structured_output: Any = None
    is_error: bool = False
    result: str | None = None
    subtype: str = "success"
    # Default None suppresses continuation unless the test supplies the real SDK session ID.
    session_id: str | None = None
    usage: dict[str, Any] | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    stop_reason: str | None = None


def patch_claude_sdk(monkeypatch: pytest.MonkeyPatch, client_class: type,) -> None:
    """Patch the SDK classes resolved by ClaudeBackend, including the supplied client_class."""
    monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", client_class)

    def _injected_client(*, options: Any, transport: Any, initialize_timeout_s: float,) -> Any:
        return client_class(options=options)

    monkeypatch.setattr("daydream.backends.claude._RunLocalClaudeSDKClient", _injected_client)
    monkeypatch.setattr("daydream.backends.claude.AssistantMessage", MockAssistantMessage)
    monkeypatch.setattr("daydream.backends.claude.UserMessage", MockUserMessage)
    monkeypatch.setattr("daydream.backends.claude.ResultMessage", MockResultMessage)
    monkeypatch.setattr("daydream.backends.claude.TextBlock", MockTextBlock)
    monkeypatch.setattr("daydream.backends.claude.ThinkingBlock", MockThinkingBlock)
    monkeypatch.setattr("daydream.backends.claude.ToolUseBlock", MockToolUseBlock)
    monkeypatch.setattr("daydream.backends.claude.ToolResultBlock", MockToolResultBlock)


def scripted_client(messages: Sequence[Any], *, captured: dict[str, Any] | None = None) -> type:
    """Build a client class that yields messages in order. When captured is supplied, record
    constructed options under "options" and the queried prompt under "prompt". Use the result with
    patch_claude_sdk.
    """

    class _ScriptedClient:
        def __init__(self, options: Any = None) -> None:
            self.options = options
            self._prompt: str = ""
            if captured is not None:
                captured["options"] = options

        async def __aenter__(self) -> _ScriptedClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            self._prompt = prompt
            if captured is not None:
                captured["prompt"] = prompt

        async def receive_response(self) -> AsyncIterator[Any]:
            for message in messages:
                yield message

    return _ScriptedClient
