"""Backend loaders for the canonical-script contract test.

Each loader is an async generator that drives one ``Backend.execute`` against
a synthesized message stream equivalent to the canonical script and yields the
resulting ``AgentEvent`` instances.

The two backends consume different low-level message shapes (Claude SDK objects
vs Codex JSONL bytes). What MUST be identical is the ``AgentEvent`` stream's
effect on the trajectory recorder — i.e. the resulting list of ATIF Steps must
match across backends. The loaders synthesize each backend's native message
format from the same canonical dict; the contract test then compares the
recorded Step shape across both.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from daydream.backends import AgentEvent
from daydream.backends.claude import ClaudeBackend
from daydream.backends.codex import CodexBackend

# ---------------------------------------------------------------------------
# Claude loader — synthesize SDK message objects, mock receive_response()
# ---------------------------------------------------------------------------


@dataclass
class _MockTextBlock:
    text: str


@dataclass
class _MockThinkingBlock:
    thinking: str


@dataclass
class _MockToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] | None = None


@dataclass
class _MockToolResultBlock:
    tool_use_id: str
    content: str | None = None
    is_error: bool = False


@dataclass
class _MockAssistantMessage:
    content: list[Any] = field(default_factory=list)
    message_id: str = ""
    model: str = "claude-test-model"
    usage: dict[str, Any] | None = None


@dataclass
class _MockUserMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class _MockResultMessage:
    total_cost_usd: float | None = None
    structured_output: Any = None
    usage: dict[str, Any] | None = None


def _build_claude_messages(script: dict[str, Any]) -> list[Any]:
    """Translate the canonical script into a Claude SDK message sequence.

    Order (verified against ``ClaudeBackend.execute``):

    1. AssistantMessage for turn 1 (text, thinking, tool_use blocks).
    2. UserMessage carrying ToolResultBlock(s) for the script's tool_results.
    3. AssistantMessage for turn 2 (text only). Carries ``usage`` so the
       backend emits a ``MetricsEvent``.
    4. ResultMessage with the same ``usage`` so the backend emits a
       ``CostEvent`` mirroring Codex's ``turn.completed``.
    """
    turns = script["turns"]
    tool_results_by_id: dict[str, dict[str, Any]] = {
        tr["id"]: tr for tr in script.get("tool_results", [])
    }
    final_usage: dict[str, Any] | None = script.get("final_usage")

    messages: list[Any] = []
    for idx, turn in enumerate(turns):
        blocks: list[Any] = []
        if turn.get("text"):
            blocks.append(_MockTextBlock(text=turn["text"]))
        if turn.get("thinking"):
            blocks.append(_MockThinkingBlock(thinking=turn["thinking"]))
        for tc in turn.get("tool_calls", []):
            blocks.append(
                _MockToolUseBlock(id=tc["id"], name=tc["name"], input=tc.get("input") or {})
            )
        # Per-turn usage only on the final turn so MetricsEvent is emitted
        # exactly once (same cardinality as Codex's single turn.completed).
        usage = final_usage if idx == len(turns) - 1 else None
        messages.append(
            _MockAssistantMessage(
                content=blocks,
                message_id=turn["message_id"],
                usage=usage,
            )
        )

        # Inject a UserMessage with the matching tool results immediately
        # after the assistant turn that issued the tool calls.
        result_blocks: list[Any] = []
        for tc in turn.get("tool_calls", []):
            tr = tool_results_by_id.get(tc["id"])
            if tr is None:
                continue
            result_blocks.append(
                _MockToolResultBlock(
                    tool_use_id=tr["id"],
                    content=tr.get("output", ""),
                    is_error=bool(tr.get("is_error", False)),
                )
            )
        if result_blocks:
            messages.append(_MockUserMessage(content=result_blocks))

    messages.append(
        _MockResultMessage(
            total_cost_usd=None,
            structured_output=None,
            usage=final_usage,
        )
    )
    return messages


async def claude_loader(
    script: dict[str, Any], *, read_only: bool = False
) -> AsyncIterator[AgentEvent]:
    """Drive ``ClaudeBackend.execute`` against the canonical script."""
    messages = _build_claude_messages(script)

    class _ScriptedClient:
        def __init__(self, options: Any = None) -> None:
            self.options = options

        async def __aenter__(self) -> "_ScriptedClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self) -> AsyncIterator[Any]:
            for m in messages:
                yield m

    with (
        patch("daydream.backends.claude.ClaudeSDKClient", _ScriptedClient),
        patch("daydream.backends.claude.AssistantMessage", _MockAssistantMessage),
        patch("daydream.backends.claude.UserMessage", _MockUserMessage),
        patch("daydream.backends.claude.ResultMessage", _MockResultMessage),
        patch("daydream.backends.claude.TextBlock", _MockTextBlock),
        patch("daydream.backends.claude.ThinkingBlock", _MockThinkingBlock),
        patch("daydream.backends.claude.ToolUseBlock", _MockToolUseBlock),
        patch("daydream.backends.claude.ToolResultBlock", _MockToolResultBlock),
    ):
        backend = ClaudeBackend(model="claude-test-model")
        async for event in backend.execute(Path("/tmp"), "go", read_only=read_only):
            yield event


# ---------------------------------------------------------------------------
# Codex loader — synthesize JSONL byte stream, mock subprocess
# ---------------------------------------------------------------------------


def _build_codex_jsonl(script: dict[str, Any]) -> list[str]:
    """Translate the canonical script into Codex JSONL event lines.

    Each turn becomes:

    - One reasoning ``item.completed`` carrying the thinking text (when set).
    - One ``mcp_tool_call`` ``item.started`` + ``item.completed`` pair per
      tool call, where the completed event's ``result.content`` carries the
      matching ``tool_results`` entry's output. ``mcp_tool_call`` is the only
      Codex item shape that supports an arbitrary tool name + arguments dict,
      so it's the natural mapping for the canonical script's "Read" call.
    - One agent_message ``item.completed`` with the turn's text.

    The final ``turn.completed`` carries ``final_usage`` so Codex emits one
    ``MetricsEvent`` + ``CostEvent`` — same cardinality as Claude's
    ``ResultMessage`` carrying the same usage dict.
    """
    turns = script["turns"]
    tool_results_by_id: dict[str, dict[str, Any]] = {
        tr["id"]: tr for tr in script.get("tool_results", [])
    }
    final_usage = script.get("final_usage") or {}

    lines: list[str] = [
        json.dumps({"type": "thread.started", "thread_id": "th_canonical"})
    ]

    for turn in turns:
        if turn.get("thinking"):
            reasoning_id = f"reason_{turn['message_id']}"
            lines.append(
                json.dumps(
                    {
                        "type": "item.started",
                        "item": {"type": "reasoning", "id": reasoning_id, "content": []},
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "reasoning",
                            "id": reasoning_id,
                            "text": turn["thinking"],
                        },
                    }
                )
            )
        for tc in turn.get("tool_calls", []):
            lines.append(
                json.dumps(
                    {
                        "type": "item.started",
                        "item": {
                            "type": "mcp_tool_call",
                            "id": tc["id"],
                            "tool": tc["name"],
                            "arguments": tc.get("input") or {},
                        },
                    }
                )
            )
            tr = tool_results_by_id.get(tc["id"])
            output = "" if tr is None else tr.get("output", "")
            is_error = False if tr is None else bool(tr.get("is_error", False))
            completed_item: dict[str, Any] = {
                "type": "mcp_tool_call",
                "id": tc["id"],
                "tool": tc["name"],
                "arguments": tc.get("input") or {},
                "result": {"content": output},
            }
            if is_error:
                completed_item["error"] = output
            lines.append(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": completed_item,
                    }
                )
            )
        if turn.get("text"):
            lines.append(
                json.dumps(
                    {
                        "type": "item.started",
                        "item": {
                            "type": "agent_message",
                            "id": turn["message_id"],
                            "content": [],
                        },
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "id": turn["message_id"],
                            "text": turn["text"],
                        },
                    }
                )
            )

    usage_payload: dict[str, Any] = {}
    if final_usage.get("input_tokens") is not None:
        usage_payload["input_tokens"] = final_usage["input_tokens"]
    if final_usage.get("output_tokens") is not None:
        usage_payload["output_tokens"] = final_usage["output_tokens"]
    if final_usage.get("cached_tokens") is not None:
        usage_payload["cached_input_tokens"] = final_usage["cached_tokens"]
    lines.append(json.dumps({"type": "turn.completed", "usage": usage_payload}))
    return lines


def _make_mock_process(lines: list[str]) -> MagicMock:
    """Build an async-subprocess stand-in that yields *lines* through stdout."""

    class _MockStdout:
        def __init__(self) -> None:
            self._lines = iter(lines)

        async def readline(self) -> bytes:
            try:
                line = next(self._lines)
                return (line + "\n").encode()
            except StopIteration:
                return b""

    process = MagicMock()
    process.stdout = _MockStdout()
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.close = MagicMock()
    process.wait = AsyncMock(return_value=0)
    process.returncode = 0
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


async def codex_loader(
    script: dict[str, Any], *, read_only: bool = False
) -> AsyncIterator[AgentEvent]:
    """Drive ``CodexBackend.execute`` against the canonical script."""
    lines = _build_codex_jsonl(script)
    mock_proc = _make_mock_process(lines)
    backend = CodexBackend(model="codex-test-model")
    with patch(
        "daydream.backends.codex.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        async for event in backend.execute(Path("/tmp"), "go", read_only=read_only):
            yield event
