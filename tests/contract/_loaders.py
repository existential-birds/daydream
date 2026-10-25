"""Build native Codex JSONL for real host phase-replay tests."""

from __future__ import annotations

import json
from typing import Any


def _build_codex_jsonl(script: dict[str, Any]) -> list[str]:
    """Translate each turn to reasoning, tool-call/result pairs, and agent text. mcp_tool_call supports
    the canonical arbitrary tool names and argument dictionaries; result.content carries the
    matching output. The final turn.completed supplies one MetricsEvent and CostEvent from
    final_usage.
    """
    turns = script["turns"]
    tool_results_by_id = {tr["id"]: tr for tr in script.get("tool_results", [])}
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
