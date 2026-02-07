# daydream/backends/codex.py
"""Codex CLI subprocess backend for daydream.

Spawns `codex exec --experimental-json` as an async subprocess,
writes the prompt to stdin, and reads JSONL events from stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)

logger = logging.getLogger(__name__)


class CodexError(Exception):
    """Raised when a Codex turn fails."""


class CodexBackend:
    """Backend that wraps the Codex CLI subprocess.

    Translates Codex JSONL events into the unified AgentEvent stream.
    """

    def __init__(self, model: str = "gpt-5.3-codex"):
        self.model = model
        self._process: asyncio.subprocess.Process | None = None

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Execute a prompt via Codex CLI and yield unified events.

        Args:
            cwd: Working directory for the agent.
            prompt: The prompt to send.
            output_schema: Optional JSON schema for structured output.
            continuation: Optional token for thread resumption.

        Yields:
            AgentEvent instances.

        Raises:
            CodexError: If the Codex turn fails.

        """
        args = [
            "codex", "exec", "--experimental-json",
            "--model", self.model,
            "--sandbox", "danger-full-access",
            "--cd", str(cwd),
        ]

        schema_path: str | None = None
        if output_schema:
            schema_path = self._write_temp_schema(output_schema)
            args.extend(["--output-schema", schema_path])

        if continuation and continuation.backend == "codex":
            args.extend(["resume", continuation.data["thread_id"]])

        thread_id: str | None = None
        last_agent_text: str | None = None
        structured_result: Any = None
        # Track generated IDs for items missing id field (ensures start/completed match)
        pending_item_ids: dict[str, str] = {}

        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Write prompt to stdin and close immediately
            if self._process.stdin:
                self._process.stdin.write(prompt.encode())
                self._process.stdin.close()

            # Read JSONL events line by line
            while True:
                if self._process.stdout is None:
                    break
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    event = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSONL line: %s", line.decode().strip())
                    continue

                event_type = event.get("type", "")

                if event_type == "thread.started":
                    thread_id = event.get("thread_id")

                elif event_type == "item.started":
                    item = event.get("item", {})
                    item_type = item.get("type", "")

                    if item_type == "command_execution":
                        item_id = item.get("id")
                        if not item_id:
                            item_id = str(uuid.uuid4())
                            pending_item_ids[f"command_execution:{item.get('command', '')}"] = item_id
                        yield ToolStartEvent(
                            id=item_id,
                            name="shell",
                            input={"command": item.get("command", "")},
                        )
                    elif item_type == "mcp_tool_call":
                        item_id = item.get("id")
                        if not item_id:
                            item_id = str(uuid.uuid4())
                            pending_item_ids[f"mcp_tool_call:{item.get('tool', '')}"] = item_id
                        yield ToolStartEvent(
                            id=item_id,
                            name=item.get("tool", "unknown"),
                            input=item.get("arguments", {}),
                        )
                    # agent_message and reasoning item.started are no-ops
                    # (text is empty, we wait for item.completed)

                elif event_type == "item.completed":
                    item = event.get("item", {})
                    item_type = item.get("type", "")

                    if item_type == "agent_message":
                        text = self._extract_text(item)
                        if text:
                            last_agent_text = text
                            yield TextEvent(text=text)

                    elif item_type == "reasoning":
                        text = self._extract_text(item)
                        if text:
                            yield ThinkingEvent(text=text)

                    elif item_type == "command_execution":
                        item_id = item.get("id")
                        if not item_id:
                            lookup_key = f"command_execution:{item.get('command', '')}"
                            item_id = pending_item_ids.pop(lookup_key, str(uuid.uuid4()))
                        exit_code = item.get("exit_code", -1)
                        output = item.get("aggregated_output", "")
                        status = item.get("status", "")

                        if status == "declined":
                            yield ToolResultEvent(
                                id=item_id,
                                output="Command declined by sandbox",
                                is_error=True,
                            )
                        else:
                            yield ToolResultEvent(
                                id=item_id,
                                output=output,
                                is_error=exit_code != 0,
                            )

                    elif item_type == "file_change":
                        # file_change has no item.started — emit synthetic pair
                        item_id = item.get("id", str(uuid.uuid4()))
                        file_path = item.get("file_path", "unknown")
                        action = item.get("action", "modified")
                        yield ToolStartEvent(
                            id=item_id,
                            name="patch",
                            input={"file": file_path, "action": action},
                        )
                        yield ToolResultEvent(
                            id=item_id,
                            output=f"{action}: {file_path}",
                            is_error=False,
                        )

                    elif item_type == "mcp_tool_call":
                        item_id = item.get("id")
                        if not item_id:
                            lookup_key = f"mcp_tool_call:{item.get('tool', '')}"
                            item_id = pending_item_ids.pop(lookup_key, str(uuid.uuid4()))
                        result_content = ""
                        if "result" in item:
                            result_content = str(item["result"].get("content", ""))
                        error = item.get("error")
                        yield ToolResultEvent(
                            id=item_id,
                            output=result_content,
                            is_error=bool(error),
                        )

                elif event_type == "turn.completed":
                    usage = event.get("usage", {})
                    yield CostEvent(
                        cost_usd=None,
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                    )

                    # Parse structured output from last agent message if schema was provided
                    if output_schema and last_agent_text:
                        try:
                            structured_result = json.loads(last_agent_text)
                        except json.JSONDecodeError:
                            pass

                    continuation_token = None
                    if thread_id:
                        continuation_token = ContinuationToken(
                            backend="codex",
                            data={"thread_id": thread_id},
                        )

                    yield ResultEvent(
                        structured_output=structured_result,
                        continuation=continuation_token,
                    )

                elif event_type == "turn.failed":
                    error = event.get("error", {})
                    raise CodexError(error.get("message", "Unknown Codex error"))

            await self._process.wait()

        finally:
            self._process = None
            if schema_path:
                Path(schema_path).unlink(missing_ok=True)

    async def cancel(self) -> None:
        """Cancel the running Codex process.

        Sends SIGTERM, waits briefly, then SIGKILL if still running.
        """
        if self._process is not None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        """Format a skill invocation for Codex.

        Codex uses $skill-name syntax. Strips namespace prefix if present.

        Args:
            skill_key: Full skill key (e.g. "beagle-python:review-python").
            args: Optional arguments string.

        Returns:
            Formatted skill invocation string.

        """
        # Strip namespace prefix: "beagle-python:review-python" → "review-python"
        if ":" in skill_key:
            skill_name = skill_key.split(":")[-1]
        else:
            skill_name = skill_key

        result = f"${skill_name}"
        if args:
            result = f"{result} {args}"
        return result

    @staticmethod
    def _extract_text(item: dict[str, Any]) -> str:
        """Extract text from a Codex item's content blocks."""
        content = item.get("content", [])
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _write_temp_schema(schema: dict[str, Any]) -> str:
        """Write JSON schema to a temp file and return the path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="daydream-schema-"
        ) as f:
            json.dump(schema, f)
            return f.name
