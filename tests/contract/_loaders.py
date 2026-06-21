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
from unittest.mock import patch

from daydream.backends import AgentEvent
from daydream.backends.claude import ClaudeBackend
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
from tests.harness.codex_replay import make_mock_process
from tests.harness.pi_replay import make_mock_process as make_mock_process_pi

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
    is_error: bool = False
    result: str | None = None
    subtype: str = "success"


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


async def codex_loader(
    script: dict[str, Any], *, read_only: bool = False
) -> AsyncIterator[AgentEvent]:
    """Drive ``CodexBackend.execute`` against the canonical script."""
    lines = _build_codex_jsonl(script)
    mock_proc = make_mock_process(lines)
    backend = CodexBackend(model="codex-test-model")
    with patch(
        "daydream.backends.codex.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        async for event in backend.execute(Path("/tmp"), "go", read_only=read_only):
            yield event


# ---------------------------------------------------------------------------
# Pi loader — synthesize JSONL byte stream, mock subprocess
# ---------------------------------------------------------------------------


def _build_pi_jsonl(script: dict[str, Any]) -> list[str]:
    """Translate the canonical script into Pi JSONL event lines.

    Pi's event vocabulary (plan §4) is turn-oriented: a ``message_end`` carries
    the full assistant content blocks (text / thinking / toolCall), then
    ``tool_execution_start``/``tool_execution_end`` pairs carry each tool's
    result, then ``turn_end`` carries usage + closes the turn. ``agent_end``
    carries no payload of interest here.

    Mapping per turn:

    - ``turn_start``
    - ``message_start`` + ``message_end`` with text / thinking / toolCall
      blocks (toolCall blocks present for completeness; the authoritative
      tool-call/result comes from the ``tool_execution_*`` events).
    - ``tool_execution_start`` + ``tool_execution_end`` per tool call, with the
      ``tool_results`` entry's output on the ``end`` event's
      ``result.content[0].text``.
    - ``turn_end`` carrying the assistant message. Only the final turn's
      message carries ``usage`` so Pi emits exactly one ``MetricsEvent``
      (matching Codex's single-``turn.completed`` cardinality).

    ``final_usage`` is also aggregated onto the ``agent_end``-derived
    ``CostEvent``, but since the parity test compares message / reasoning /
    tool_calls / observation results (not metrics), the exact token split does
    not affect Step parity.
    """
    turns = script["turns"]
    tool_results_by_id: dict[str, dict[str, Any]] = {
        tr["id"]: tr for tr in script.get("tool_results", [])
    }
    final_usage = script.get("final_usage") or {}

    lines: list[str] = [
        json.dumps({"type": "session", "sessionId": "pi_canonical_session"}),
        json.dumps({"type": "agent_start"}),
    ]

    for idx, turn in enumerate(turns):
        is_last = idx == len(turns) - 1
        content: list[dict[str, Any]] = []
        if turn.get("text"):
            content.append({"type": "text", "text": turn["text"]})
        if turn.get("thinking"):
            content.append({"type": "thinking", "thinking": turn["thinking"]})
        for tc in turn.get("tool_calls", []):
            content.append(
                {
                    "type": "toolCall",
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc.get("input") or {},
                }
            )

        usage_payload: dict[str, Any] = {}
        if is_last:
            if final_usage.get("input_tokens") is not None:
                usage_payload["input"] = final_usage["input_tokens"]
            if final_usage.get("output_tokens") is not None:
                usage_payload["output"] = final_usage["output_tokens"]
            if final_usage.get("cached_tokens") is not None:
                usage_payload["cacheRead"] = final_usage["cached_tokens"]
            usage_payload["cost"] = {"total": 0.0}

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "model": "pi-test-model",
        }
        if usage_payload:
            assistant_msg["usage"] = usage_payload

        lines.append(json.dumps({"type": "turn_start"}))
        lines.append(json.dumps({"type": "message_start", "message": assistant_msg}))
        lines.append(json.dumps({"type": "message_end", "message": assistant_msg}))

        for tc in turn.get("tool_calls", []):
            tr = tool_results_by_id.get(tc["id"])
            output = "" if tr is None else tr.get("output", "")
            is_error = False if tr is None else bool(tr.get("is_error", False))
            lines.append(
                json.dumps(
                    {
                        "type": "tool_execution_start",
                        "toolCallId": tc["id"],
                        "toolName": tc["name"],
                        "args": tc.get("input") or {},
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "type": "tool_execution_end",
                        "toolCallId": tc["id"],
                        "toolName": tc["name"],
                        "result": {"content": [{"type": "text", "text": output}]},
                        "isError": is_error,
                    }
                )
            )

        turn_end_msg: dict[str, Any] = {
            "role": "assistant",
            "content": list(content),
            "model": "pi-test-model",
            "stopReason": "stop",
        }
        if usage_payload:
            turn_end_msg["usage"] = usage_payload
        lines.append(json.dumps({"type": "turn_end", "message": turn_end_msg, "toolResults": []}))

    lines.append(json.dumps({"type": "agent_end", "messages": []}))
    return lines


async def pi_loader(
    script: dict[str, Any], *, read_only: bool = False
) -> AsyncIterator[AgentEvent]:
    """Drive ``PiBackend.execute`` against the canonical script."""
    lines = _build_pi_jsonl(script)
    mock_proc = make_mock_process_pi(lines)
    backend = PiBackend(model="pi-test-model")
    with patch(
        "daydream.backends.pi.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        async for event in backend.execute(Path("/tmp"), "go", read_only=read_only):
            yield event
