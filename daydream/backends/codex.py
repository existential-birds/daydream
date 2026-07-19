# daydream/backends/codex.py
"""Codex CLI subprocess backend for daydream.

Spawns `codex exec --experimental-json` as an async subprocess,
writes the prompt to stdin, and reads JSONL events from stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import uuid
from collections.abc import AsyncGenerator
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
from daydream.backends._subprocess import cancel_processes, terminate_process
from daydream.pricing import compute_cost_from_totals, load_user_prices, resolve_prices

_SHELL_WRAPPER_RE = re.compile(r"/bin/(?:zsh|bash|sh)\s+-lc\s+(.+)$", re.DOTALL)
_CD_PREFIX_RE = re.compile(r"^cd\s+\S+\s*&&\s*")
_CODEX_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024

_logger = logging.getLogger(__name__)


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
    if (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
        inner = inner[1:-1]
    inner = _CD_PREFIX_RE.sub("", inner)  # Strip leading "cd /some/path &&".
    return inner.strip()


class CodexError(Exception):
    """Raised when a Codex turn fails."""


class CodexBackend:
    """Backend that wraps the Codex CLI subprocess.

    Translates Codex JSONL events into the unified AgentEvent stream.
    """

    concise_fix_prompts = False

    def __init__(self, model: str, reasoning_effort: str | None = None):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.fanout_concurrency = 4
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
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute a prompt via Codex CLI and yield unified events.

        Args:
            agents: Optional subagent mapping. Codex does not support non-empty
                subagent maps and will raise if provided.
            read_only: When True, run under ``--sandbox read-only`` so the agent
                can inspect history (read-only git, cat, ls) but cannot mutate
                the working tree, index, or refs. Accepted residual (Task 0
                spike): ``--sandbox read-only`` does not block ``git commit``;
                the working tree — the failure-handoff incident's danger — is
                fully protected. Default False keeps ``danger-full-access``.

        Raises:
            CodexError: If the Codex turn fails.
            NotImplementedError: If ``agents`` is non-empty (Codex backend
                does not support exploration subagents).
        """
        if agents:
            raise NotImplementedError(
                "Codex backend does not support exploration subagents; use --backend claude for exploration."
            )

        # Read-only sandbox blocks working-tree/index/ref mutation but (accepted
        # residual, Task 0 spike) not `git commit`; the working tree is protected.
        sandbox_mode = "read-only" if read_only else "danger-full-access"
        args = [
            "codex",
            "exec",
            "--experimental-json",
            "--model",
            self.model,
            "--sandbox",
            sandbox_mode,
            "--cd",
            str(cwd),
        ]
        if self.reasoning_effort:
            args.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])

        schema_path: str | None = None
        if output_schema:
            schema_path = self._write_temp_schema(output_schema)
            args.extend(["--output-schema", schema_path])

        if continuation and continuation.backend == "codex":
            args.extend(["resume", continuation.data["thread_id"]])

        thread_id: str | None = None
        last_agent_text: str | None = None
        structured_result: Any = None

        # Event correlation state. Codex events arrive item.started → updated* →
        # completed. (1) Some item.started lack an `id`; we pair start/result via
        # a deterministic FIFO per item_type (order-preserving, content-
        # independent) with the legacy content-key as a secondary fallback. When
        # both miss, the completion is an orphan — we emit an OBSERVABLE warning
        # and assign a deterministic sequence id so the trajectory recorder can
        # still bucket it via unmatched_tool_results without a silent drop.
        # (2) agent_message/reasoning text may stream as item.updated deltas with
        # an empty item.completed, so we accumulate deltas by id and join them.
        pending_fifo: dict[str, list[str]] = {}  # item_type → [ids] in start order
        pending_item_ids: dict[str, str] = {}  # "type:content" → generated id (legacy)
        updated_text: dict[str, list[str]] = {}  # item_id → [text deltas]
        parse_warnings: list[str] = []  # observable parse-failure surface
        unmatched_seq = 0  # monotonic source for orphaned tool-result ids

        def _warn(msg: str, **detail: Any) -> None:
            """Log a parser warning and record it for later trajectory surfacing."""
            parse_warnings.append(msg)
            _logger.warning("codex: %s %s", msg, detail)

        def _claim_tool_id(item_type: str, content_key: str, content_value: Any) -> str:
            """Pop the next correlated id for a no-id item.completed.

            Prefers the FIFO (content-independent order correlation), falls back
            to the legacy content-key, and finally — if both miss — emits an
            OBSERVABLE warning and returns a deterministic orphan id. The orphan
            still lands in ``trajectory.extra.unmatched_tool_results`` because
            the validator hard-fails on a dangling ``source_call_id``; the
            warning ensures the miss is no longer silent.
            """
            nonlocal unmatched_seq
            fifo = pending_fifo.get(item_type, [])
            item_id = fifo.pop(0) if fifo else None
            if item_id is None:
                item_id = pending_item_ids.pop(content_key, None)
            if item_id is None:
                _warn("unmatched tool result", item_type=item_type, key=content_key, value=content_value)
                item_id = f"codex-unmatched-{unmatched_seq}"
                unmatched_seq += 1
            return item_id

        proc: asyncio.subprocess.Process | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=_CODEX_STDOUT_LIMIT_BYTES,
            )
            self._processes.append(proc)

            if proc.stdin:
                proc.stdin.write(prompt.encode())
                proc.stdin.close()

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
                    # Codex occasionally emits non-JSON status lines on stdout;
                    # skip them but leave a debug breadcrumb for triage.
                    _logger.debug("codex: non-JSON line skipped: %r", raw_line[:80])
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
                            # FIFO is the primary correlation path (order-
                            # preserving); content-key stays as legacy fallback.
                            pending_fifo.setdefault("command_execution", []).append(item_id)
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
                            # FIFO is the primary correlation path (order-
                            # preserving); content-key stays as legacy fallback.
                            pending_fifo.setdefault("mcp_tool_call", []).append(item_id)
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
                        # Fall back to text accumulated from item.updated deltas.
                        if not text:
                            item_id = item.get("id", "")
                            parts = updated_text.pop(item_id, [])
                            text = "".join(parts)
                        if text:
                            last_agent_text = text
                            yield TextEvent(text=text)
                            # Codex has no per-message id; message_id stays empty (D-04).
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
                            item_id = _claim_tool_id(
                                "command_execution",
                                f"command_execution:{item.get('command', '')}",
                                item.get("command", ""),
                            )
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
                        # file_change has no item.started — emit a synthetic pair.
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
                            item_id = _claim_tool_id(
                                "mcp_tool_call",
                                f"mcp_tool_call:{item.get('tool', '')}",
                                item.get("tool", ""),
                            )
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
                    # EVNT-07: MetricsEvent per turn (empty message_id — Codex has no
                    # per-message id). #194 reverses D-16: the CLI emits no cost field,
                    # so we now synthesize from tokens via the #61 user-overridable price
                    # table (Claude/Pi parity). None when the model is unknown to the
                    # table (preserves the #156 observable-marker). cached_input_tokens
                    # IS surfaced (#65, K4). Rename input/output_tokens → prompt/completion;
                    # skip if either missing (EVNT-02 requires both). CostEvent below
                    # carries partials.
                    #
                    # #192: reasoning_output_tokens is the reasoning portion of
                    # output_tokens — a SUBSET, NOT additive (codex's own
                    # accounting.rs already counts these inside output_tokens, so
                    # cost synthesis is unchanged). Surfaced for cost attribution
                    # and perf observability (#171/#172/#186). OpenAI emits the
                    # COUNT only — no reasoning content (openai/codex#26428).
                    cached_tokens = usage.get("cached_input_tokens")
                    reasoning_tokens = usage.get("reasoning_output_tokens")
                    in_tok = usage.get("input_tokens")
                    out_tok = usage.get("output_tokens")
                    synth_cost = compute_cost_from_totals(
                        self.model,
                        total_input_tokens=in_tok or 0,
                        cached_input_tokens=cached_tokens or 0,
                        output_tokens=out_tok or 0,
                        prices=resolve_prices(load_user_prices()),
                    )
                    if in_tok is not None and out_tok is not None:
                        yield MetricsEvent(
                            message_id="",
                            prompt_tokens=in_tok,
                            completion_tokens=out_tok,
                            cached_tokens=cached_tokens,
                            cost_usd=synth_cost,
                            reasoning_tokens=reasoning_tokens,
                            model_name=self.model,
                        )
                    yield CostEvent(
                        cost_usd=synth_cost,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cached_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                        model_name=self.model,
                    )

                    if output_schema and last_agent_text:
                        try:
                            structured_result = json.loads(last_agent_text)
                        except json.JSONDecodeError:
                            # Observable failure path — surface the bad payload
                            # instead of silently degrading to None.
                            _warn(
                                "structured output parse failed",
                                source="agent_text",
                                raw=last_agent_text[:200],
                            )

                    # Fallback: result/output field directly on turn.completed.
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
                                        _warn(
                                            "structured output parse failed",
                                            source=f"turn.completed.{key}",
                                            raw=raw[:200],
                                        )
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
                await terminate_process(proc)
            self._processes = [active for active in self._processes if active is not proc]
            if schema_path:
                Path(schema_path).unlink(missing_ok=True)

    async def cancel(self) -> None:
        """Cancel all running Codex processes.

        Sends SIGTERM, waits briefly, then SIGKILL if still running.
        """
        await cancel_processes(self._processes)

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        """Format a skill invocation for Codex.

        Codex uses $skill-name syntax. Strips namespace prefix if present.
        """
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
        # Top-level text field (real Codex CLI format).
        top = item.get("text")
        if isinstance(top, str) and top:
            return top
        # Content-block format (legacy / alternative).
        content = item.get("content", [])
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(block.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _write_temp_schema(schema: dict[str, Any]) -> str:
        """Write JSON schema to a temp file and return the path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="daydream-schema-") as f:
            json.dump(schema, f)
            return f.name
