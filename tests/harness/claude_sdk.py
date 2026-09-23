"""Shared Claude-SDK message mocks and patch helper for ``ClaudeBackend`` tests.

``ClaudeBackend.execute`` dispatches on the SDK's message/block types by
``isinstance``, so any test that drives it must (a) supply stand-in classes and
(b) monkeypatch the eight names ``daydream.backends.claude`` resolves at
runtime. That block was re-rolled verbatim in four test modules, which meant an
SDK shape change had to be chased through four copies.

This module owns one set of stand-ins plus the two helpers every consumer needs:

* :func:`patch_claude_sdk` — patch the eight SDK names in one call.
* :func:`scripted_client` — build a ``ClaudeSDKClient`` stand-in that replays a
  canned message sequence and, optionally, records the
  ``ClaudeAgentOptions`` it was constructed with.

The mocks are intentionally minimal but carry every optional field
``ClaudeBackend`` reads (``model`` / ``message_id`` / ``usage`` on the
assistant message; ``usage`` / ``duration_ms`` / ``duration_api_ms`` /
``stop_reason`` on the result message) so no consumer needs a subclass.
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
    # Only read via ``getattr`` by ``ClaudeBackend.execute``; the defaults give a
    # message with no model, no metrics, and no continuation unless a test opts in.
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
    # Real ResultMessage always carries one; default None keeps every existing
    # scripted message minting no continuation until a test opts in.
    session_id: str | None = None
    usage: dict[str, Any] | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    stop_reason: str | None = None


def patch_claude_sdk(
    monkeypatch: pytest.MonkeyPatch,
    client_class: type,
) -> None:
    """Patch every SDK name ``daydream.backends.claude`` resolves at runtime.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        client_class: Stand-in for ``ClaudeSDKClient`` (see
            :func:`scripted_client`).
    """
    monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", client_class)

    def _injected_client(
        *,
        options: Any,
        transport: Any,
        initialize_timeout_s: float,
    ) -> Any:
        return client_class(options=options)

    monkeypatch.setattr(
        "daydream.backends.claude._RunLocalClaudeSDKClient", _injected_client
    )
    monkeypatch.setattr("daydream.backends.claude.AssistantMessage", MockAssistantMessage)
    monkeypatch.setattr("daydream.backends.claude.UserMessage", MockUserMessage)
    monkeypatch.setattr("daydream.backends.claude.ResultMessage", MockResultMessage)
    monkeypatch.setattr("daydream.backends.claude.TextBlock", MockTextBlock)
    monkeypatch.setattr("daydream.backends.claude.ThinkingBlock", MockThinkingBlock)
    monkeypatch.setattr("daydream.backends.claude.ToolUseBlock", MockToolUseBlock)
    monkeypatch.setattr("daydream.backends.claude.ToolResultBlock", MockToolResultBlock)


def scripted_client(messages: Sequence[Any], *, captured: dict[str, Any] | None = None) -> type:
    """Build a ``ClaudeSDKClient`` stand-in that replays *messages* verbatim.

    Args:
        messages: Message objects ``receive_response()`` yields, in order.
        captured: When given, the constructed ``ClaudeAgentOptions`` is recorded
            under ``"options"`` and the queried prompt under ``"prompt"`` — the
            observable for tests that assert on what reached the SDK.

    Returns:
        A class suitable for :func:`patch_claude_sdk`'s ``client_class``.
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
