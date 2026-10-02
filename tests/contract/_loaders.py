"""Backend loaders translate one canonical script into native SDK/JSONL messages, then yield real
Backend.execute events. The backends must preserve the same observable event and trajectory-step
shapes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.backends import AgentEvent
from daydream.backends.claude import ClaudeBackend
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    MockThinkingBlock,
    MockToolResultBlock,
    MockToolUseBlock,
    MockUserMessage,
    patch_claude_sdk,
    scripted_client,
)
from tests.harness.codex_replay import make_mock_process
from tests.harness.pi_replay import make_mock_process as make_mock_process_pi

# Claude loader — synthesize SDK message objects, mock receive_response()


def _tool_results_by_id(script: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {tr["id"]: tr for tr in script.get("tool_results", [])}


def _build_claude_messages(script: dict[str, Any]) -> list[Any]:
    """Emit assistant text/thinking/tool-use, then matching user tool results for each turn. Only the
    final assistant message carries usage; ResultMessage repeats it so metrics and terminal cost
    match the other loaders.
    """
    turns = script["turns"]
    tool_results_by_id = _tool_results_by_id(script)
    final_usage: dict[str, Any] | None = script.get("final_usage")

    messages: list[Any] = []
    for idx, turn in enumerate(turns):
        blocks: list[Any] = []
        if turn.get("text"):
            blocks.append(MockTextBlock(text=turn["text"]))
        if turn.get("thinking"):
            blocks.append(MockThinkingBlock(thinking=turn["thinking"]))
        for tc in turn.get("tool_calls", []):
            blocks.append(MockToolUseBlock(id=tc["id"], name=tc["name"], input=tc.get("input") or {}))
        usage = final_usage if idx == len(turns) - 1 else None
        messages.append(MockAssistantMessage(
                content=blocks, model="claude-test-model", message_id=turn["message_id"], usage=usage,
            )
        )

        # Place matching user tool results immediately after the assistant turn that issued them.
        result_blocks: list[Any] = []
        for tc in turn.get("tool_calls", []):
            tr = tool_results_by_id.get(tc["id"])
            if tr is None:
                continue
            result_blocks.append(MockToolResultBlock(
                    tool_use_id=tr["id"], content=tr.get("output", ""), is_error=bool(tr.get("is_error", False)),
                )
            )
        if result_blocks:
            messages.append(MockUserMessage(content=result_blocks))

    messages.append(MockResultMessage(total_cost_usd=None, structured_output=None, usage=final_usage,))
    return messages


async def claude_loader(script: dict[str, Any], *, read_only: bool = False) -> AsyncIterator[AgentEvent]:
    messages = _build_claude_messages(script)
    client = scripted_client(messages)
    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_claude_sdk(monkeypatch, client)
        backend = ClaudeBackend(model="claude-test-model")
        async for event in backend.execute(Path("/tmp"), "go", read_only=read_only):
            yield event


# Codex loader — synthesize JSONL byte stream, mock subprocess


def _build_codex_jsonl(script: dict[str, Any]) -> list[str]:
    """Translate each turn to reasoning, tool-call/result pairs, and agent text. mcp_tool_call supports
    the canonical arbitrary tool names and argument dictionaries; result.content carries the
    matching output. The final turn.completed supplies one MetricsEvent and CostEvent from
    final_usage.
    """
    turns = script["turns"]
    tool_results_by_id = _tool_results_by_id(script)
    final_usage = script.get("final_usage") or {}

    lines: list[str] = [json.dumps({"type": "thread.started", "thread_id": "th_canonical"})]

    for turn in turns:
        if turn.get("thinking"):
            reasoning_id = f"reason_{turn['message_id']}"
            lines.append(json.dumps(
                    {"type": "item.started", "item": {"type": "reasoning", "id": reasoning_id, "content": []}}
                )
            )
            lines.append(json.dumps({"type": "item.completed",
                        "item": {"type": "reasoning", "id": reasoning_id, "text": turn["thinking"]},
                    }
                )
            )
        for tc in turn.get("tool_calls", []):
            lines.append(json.dumps({"type": "item.started",
                        "item": {"type": "mcp_tool_call", "id": tc["id"], "tool": tc["name"],
                            "arguments": tc.get("input") or {},
                        },
                    }
                )
            )
            tr = tool_results_by_id.get(tc["id"])
            output = "" if tr is None else tr.get("output", "")
            is_error = False if tr is None else bool(tr.get("is_error", False))
            completed_item: dict[str, Any] = {
                "type": "mcp_tool_call", "id": tc["id"], "tool": tc["name"], "arguments": tc.get("input") or {},
                "result": {"content": output},
            }
            if is_error:
                completed_item["error"] = output
            lines.append(json.dumps({"type": "item.completed", "item": completed_item}))
        if turn.get("text"):
            lines.append(json.dumps({"type": "item.started",
                        "item": {"type": "agent_message", "id": turn["message_id"], "content": []},
                    }
                )
            )
            lines.append(json.dumps({"type": "item.completed",
                        "item": {"type": "agent_message", "id": turn["message_id"], "text": turn["text"]},
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


async def codex_loader(script: dict[str, Any], *, read_only: bool = False) -> AsyncIterator[AgentEvent]:
    lines = _build_codex_jsonl(script)
    mock_proc = make_mock_process(lines)
    backend = CodexBackend(model="codex-test-model")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc,):
        async for event in backend.execute(Path("/tmp"), "go", read_only=read_only):
            yield event


# Pi loader — synthesize JSONL byte stream, mock subprocess


def _build_pi_jsonl(script: dict[str, Any]) -> list[str]:
    """Emit turn_start, message_start/message_end, tool_execution_start/tool_execution_end pairs, and
    turn_end. Message toolCall blocks describe the calls; tool_execution events own their results.
    Only the final turn carries usage, matching the other loaders' single MetricsEvent. agent_end
    supplies aggregate cost; exact token splits do not affect the message/reasoning/tool/observation
    step-parity assertions.
    """
    turns = script["turns"]
    tool_results_by_id = _tool_results_by_id(script)
    final_usage = script.get("final_usage") or {}

    lines: list[str] = [
        json.dumps({"type": "session", "sessionId": "pi_canonical_session"}), json.dumps({"type": "agent_start"}),
    ]

    for idx, turn in enumerate(turns):
        is_last = idx == len(turns) - 1
        content: list[dict[str, Any]] = []
        if turn.get("text"):
            content.append({"type": "text", "text": turn["text"]})
        if turn.get("thinking"):
            content.append({"type": "thinking", "thinking": turn["thinking"]})
        for tc in turn.get("tool_calls", []):
            content.append({"type": "toolCall", "id": tc["id"], "name": tc["name"], "arguments": tc.get("input") or {}}
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

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content, "model": "pi-test-model"}
        if usage_payload:
            assistant_msg["usage"] = usage_payload

        lines.append(json.dumps({"type": "turn_start"}))
        lines.append(json.dumps({"type": "message_start", "message": assistant_msg}))
        lines.append(json.dumps({"type": "message_end", "message": assistant_msg}))

        for tc in turn.get("tool_calls", []):
            tr = tool_results_by_id.get(tc["id"])
            output = "" if tr is None else tr.get("output", "")
            is_error = False if tr is None else bool(tr.get("is_error", False))
            lines.append(json.dumps({"type": "tool_execution_start", "toolCallId": tc["id"], "toolName": tc["name"],
                        "args": tc.get("input") or {},
                    }
                )
            )
            lines.append(json.dumps({"type": "tool_execution_end", "toolCallId": tc["id"], "toolName": tc["name"],
                        "result": {"content": [{"type": "text", "text": output}]}, "isError": is_error,
                    }
                )
            )

        turn_end_msg: dict[str, Any] = {
            "role": "assistant", "content": list(content), "model": "pi-test-model", "stopReason": "stop",
        }
        if usage_payload:
            turn_end_msg["usage"] = usage_payload
        lines.append(json.dumps({"type": "turn_end", "message": turn_end_msg, "toolResults": []}))

    lines.append(json.dumps({"type": "agent_end", "messages": []}))
    return lines


async def pi_loader(script: dict[str, Any], *, read_only: bool = False) -> AsyncIterator[AgentEvent]:
    lines = _build_pi_jsonl(script)
    mock_proc = make_mock_process_pi(lines)
    backend = PiBackend(model="pi-test-model")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc,):
        async for event in backend.execute(Path("/tmp"), "go", read_only=read_only):
            yield event
