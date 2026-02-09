# daydream/backends/codex.py
"""Codex CLI subprocess backend for daydream.

Spawns `codex exec --experimental-json` as an async subprocess,
writes the prompt to stdin, and reads JSONL events from stdout.
"""

from __future__ import annotations

import asyncio
import json
import re
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

_SHELL_WRAPPER_RE = re.compile(r"/bin/(?:zsh|bash|sh)\s+-lc\s+(.+)$", re.DOTALL)
_CD_PREFIX_RE = re.compile(r"^cd\s+\S+\s*&&\s*")


def _raw_log(message: str) -> None:
    """Log raw event to the agent debug log if available."""
    # Lazy import to avoid circular dependency at module load time
    from daydream.agent import _log_debug
    _log_debug(message)


def _unwrap_shell_command(command: str) -> str:
    """Strip shell wrapper from Codex command_execution commands.

    Codex wraps commands in three forms::

        /bin/zsh -lc 'actual command'      (single-quoted)
        /bin/zsh -lc "actual command"      (double-quoted)
        /bin/zsh -lc actual command         (unquoted)

    This extracts just the inner command for display purposes.
    """
    m = _SHELL_WRAPPER_RE.match(command)
    if not m:
        return command
    inner = m.group(1)
    # Strip surrounding quotes if present
    if (inner.startswith('"') and inner.endswith('"')) or (
        inner.startswith("'") and inner.endswith("'")
    ):
        inner = inner[1:-1]
    # Strip leading "cd /some/path &&" if present
    inner = _CD_PREFIX_RE.sub("", inner)
    return inner.strip()


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
        # Accumulate text from item.updated deltas for agent_message items
        # keyed by item id
        updated_text: dict[str, list[str]] = {}

        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
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

                raw_line = line.decode().strip()
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    _raw_log(f"[CODEX_RAW] unparseable: {raw_line[:500]}\n")
                    continue

                event_type = event.get("type", "")
                _raw_log(f"[CODEX_RAW] {raw_line[:1000]}\n")

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
                        raw_cmd = item.get("command", "")
                        yield ToolStartEvent(
                            id=item_id,
                            name="shell",
                            input={"command": _unwrap_shell_command(raw_cmd)},
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

                elif event_type == "item.updated":
                    item = event.get("item", {})
                    item_type = item.get("type", "")
                    item_id = item.get("id", "")

                    if item_type in ("agent_message", "reasoning"):
                        text = self._extract_text(item)
                        if text and item_id:
                            updated_text.setdefault(item_id, []).append(text)

                elif event_type == "item.completed":
                    item = event.get("item", {})
                    item_type = item.get("type", "")

                    if item_type == "agent_message":
                        text = self._extract_text(item)
                        # Fall back to text accumulated from item.updated deltas
                        if not text:
                            item_id = item.get("id", "")
                            parts = updated_text.pop(item_id, [])
                            text = "".join(parts)
                        if text:
                            last_agent_text = text
                            yield TextEvent(text=text)

                    elif item_type == "reasoning":
                        text = self._extract_text(item)
                        if not text:
                            item_id = item.get("id", "")
                            parts = updated_text.pop(item_id, [])
                            text = "".join(parts)
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

                    # Fallback: check for result/output field directly on turn.completed
                    if output_schema and structured_result is None:
                        for key in ("result", "output"):
                            raw = event.get(key)
                            if raw is not None:
                                if isinstance(raw, dict):
                                    structured_result = raw
                                elif isinstance(raw, str) and raw.strip():
                                    try:
                                        structured_result = json.loads(raw)
                                    except json.JSONDecodeError:
                                        pass
                                if structured_result is not None:
                                    break

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

                elif event_type not in ("turn.started",):
                    _raw_log(f"[CODEX_UNHANDLED] {event_type}: {json.dumps(event)[:500]}\n")

            await self._process.wait()

        finally:
            proc = self._process
            if proc is not None and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
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
        """Extract text from a Codex item.

        Codex items may carry text either as a top-level ``text`` field
        or inside ``content`` blocks (with type ``text`` or ``output_text``).
        """
        # Top-level text field (real Codex CLI format)
        top = item.get("text")
        if isinstance(top, str) and top:
            return top
        # Content-block format (legacy / alternative)
        content = item.get("content", [])
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
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
