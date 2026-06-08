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
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)

_SHELL_WRAPPER_RE = re.compile(r"/bin/(?:zsh|bash|sh)\s+-lc\s+(.+)$", re.DOTALL)
_CD_PREFIX_RE = re.compile(r"^cd\s+\S+\s*&&\s*")


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

    def __init__(self, model: str):
        self.model = model
        self._process: asyncio.subprocess.Process | None = None
        self._processes: list[asyncio.subprocess.Process] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Execute a prompt via Codex CLI and yield unified events.

        Args:
            cwd: Working directory for the agent.
            prompt: The prompt to send.
            output_schema: Optional JSON schema for structured output.
            continuation: Optional token for thread resumption.
            agents: Optional subagent mapping. Codex does not support non-empty
                subagent maps and will raise if provided.
            read_only: When True, run under ``--sandbox read-only`` so the agent
                can inspect history (read-only git, cat, ls) but cannot mutate
                the working tree, index, or refs. Accepted residual (Task 0
                spike): ``--sandbox read-only`` does not block ``git commit``;
                the working tree — the failure-handoff incident's danger — is
                fully protected. Default False keeps ``danger-full-access``.

        Yields:
            AgentEvent instances.

        Raises:
            CodexError: If the Codex turn fails.
            NotImplementedError: If ``agents`` is non-empty (Codex backend
                does not support exploration subagents).

        """
        if agents:
            raise NotImplementedError(
                "Codex backend does not support exploration subagents; "
                "use --backend claude for exploration."
            )

        # Enforced read-only profile (failure summarizer): the read-only sandbox
        # blocks all working-tree/index/ref mutation while permitting read-only
        # git inspection. Accepted residual (Task 0 spike): it does not block
        # `git commit`; the working tree — the incident's danger — is protected.
        sandbox_mode = "read-only" if read_only else "danger-full-access"
        args = [
            "codex", "exec", "--experimental-json",
            "--model", self.model,
            "--sandbox", sandbox_mode,
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

        # Event correlation state
        # ─────────────────────────────────────────────────────────────────────
        # Codex JSONL events arrive as item.started → item.updated* → item.completed
        # sequences. Two challenges require explicit correlation tracking:
        #
        # 1. Missing IDs: Some item.started events lack an `id` field, but the
        #    corresponding item.completed must emit a ToolResultEvent with the
        #    same ID as the ToolStartEvent. We generate a UUID at item.started
        #    and store it keyed by item type + unique content (e.g., command text).
        #    On item.completed, we look up and pop that ID to maintain pairing.
        #
        # 2. Incremental text: For agent_message and reasoning items, text may
        #    arrive incrementally via item.updated events (each containing a delta),
        #    while item.completed may have empty text. We accumulate deltas by
        #    item ID during updates, then join them on completion if needed.
        # ─────────────────────────────────────────────────────────────────────
        pending_item_ids: dict[str, str] = {}  # "type:content" → generated UUID
        updated_text: dict[str, list[str]] = {}  # item_id → [text deltas]

        proc: asyncio.subprocess.Process | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._processes.append(proc)
            self._process = proc

            # Write prompt to stdin and close immediately
            if proc.stdin:
                proc.stdin.write(prompt.encode())
                proc.stdin.close()

            # Read JSONL events line by line
            while True:
                if proc.stdout is None:
                    break
                line = await proc.stdout.readline()
                if not line:
                    break

                raw_line = line.decode().strip()
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
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
                            # Codex has no per-message id surface — message_id
                            # stays empty (D-04 correlator unused for Codex).
                            yield TurnEndEvent(message_id="")

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
                            item_id = pending_item_ids.pop(lookup_key, None)
                            if item_id is None:
                                item_id = str(uuid.uuid4())
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
                            item_id = pending_item_ids.pop(lookup_key, None)
                            if item_id is None:
                                item_id = str(uuid.uuid4())
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
                    # Phase 2 (EVNT-07): emit MetricsEvent for the turn. Codex
                    # has no per-message id surface so the message id is the
                    # empty string. Codex does not report cost_usd — D-16 says
                    # leave it as None and DO NOT synthesize cost from a
                    # token-price table (Pitfall 6, ATIF Metrics fields all
                    # optional). Codex DOES emit cached_input_tokens on
                    # turn.completed.usage (codex-rs/protocol/src/protocol.rs
                    # TokenUsage.cached_input_tokens) — surface it so cache-hit
                    # ratios work for the Codex backend (refs #65, K4).
                    # EVNT-02 field names: MetricsEvent uses prompt_tokens /
                    # completion_tokens (ATIF/Metrics-side names). Codex's SDK
                    # boundary keys are input_tokens / output_tokens; rename
                    # here. Skip emission when either is missing — EVNT-02 types
                    # both as int (required, not Optional). CostEvent below
                    # carries the partial-data signal.
                    cached_tokens = usage.get("cached_input_tokens")
                    if usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
                        yield MetricsEvent(
                            message_id="",
                            prompt_tokens=usage["input_tokens"],
                            completion_tokens=usage["output_tokens"],
                            cached_tokens=cached_tokens,
                            cost_usd=None,
                            model_name=self.model,
                        )
                    yield CostEvent(
                        cost_usd=None,
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                        cached_tokens=cached_tokens,
                        model_name=self.model,
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
                    pass

            await proc.wait()

        finally:
            if proc is not None and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            self._processes = [active for active in self._processes if active is not proc]
            self._process = self._processes[-1] if self._processes else None
            if schema_path:
                Path(schema_path).unlink(missing_ok=True)

    async def cancel(self) -> None:
        """Cancel the running Codex process.

        Sends SIGTERM, waits briefly, then SIGKILL if still running.
        """
        processes = list(self._processes)
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

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
