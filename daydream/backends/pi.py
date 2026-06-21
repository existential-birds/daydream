# daydream/backends/pi.py
"""Pi CLI subprocess backend for daydream.

Spawns ``pi --mode json`` (the ``@earendil-works/pi-coding-agent`` TypeScript
coding agent) as an async subprocess, reads the JSONL event stream from stdout,
and translates it into the unified :data:`daydream.backends.AgentEvent` stream.

Pi is a subprocess + JSONL backend — the same proven shape as
:mod:`daydream.backends.codex`. The Pi backend is a second instance of that
pattern; it emits the same event vocabulary so the existing
:class:`daydream.trajectory.TrajectoryRecorder` produces valid ATIF v1.6
trajectories indistinguishable in shape from the other two backends.

z.ai coding plan wiring (GLM models) is configured once in ``~/.pi/models.json``
(see ``design.md`` §3); daydream never fabricates a base URL or cost table.
Optional ``PI_PROVIDER`` / ``PI_API_KEY`` / ``PI_THINKING`` env overrides are
forwarded as CLI flags. ``PI_BASE_URL`` has no matching CLI flag — set it in
``~/.pi/models.json`` (documented in CLAUDE.md).
"""

from __future__ import annotations

import asyncio
import json
import os
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

# Mirror Codex's generous stdout cap so large JSONL events (big file reads,
# patch payloads) do not trip asyncio's "chunk is longer than limit" guard.
_PI_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024

# Known AgentSessionEvent types (see plan §4). Used to decide whether the first
# stdout line — the session header — also carries a dispatchable event type.
_PI_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "agent_start",
        "agent_end",
        "turn_start",
        "turn_end",
        "message_start",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
    }
)

# Read-only tool subset (plan §3). Excludes the mutating edit/bash/write tools.
_PI_READ_ONLY_TOOLS = "read,find,ls,grep"


class PiError(Exception):
    """Raised when a Pi turn fails (e.g. ``stopReason == "error"``)."""


def _extract_json(text: str) -> Any:
    """Extract a JSON object or array from possibly prose-wrapped model text.

    Thin wrapper around the shared :func:`daydream.json_utils.extract_json`.
    Kept as a module-private alias so existing call sites and tests are stable.
    """
    from daydream.json_utils import extract_json

    return extract_json(text)


def _render_tool_result(result: Any) -> str:
    """Render a Pi ``AgentToolResult`` into a flat string for ``ToolResultEvent``.

    The canonical shape is ``{"content": [{"type": "text", "text": "..."}],
    "details": <any>, "terminate": bool}``. We join every ``text`` block. If the
    shape diverges (older/newer Pi build), fall back to ``details`` then to a
    JSON dump so the trajectory never loses the observation.
    """
    if not isinstance(result, dict):
        return "" if result is None else str(result)
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        joined = "".join(parts)
        if joined:
            return joined
    if isinstance(content, str) and content:
        return content
    details = result.get("details")
    if details is not None and details != "":
        return str(details)
    # Last resort — preserve the payload rather than dropping the observation.
    return json.dumps(result, default=str) if result else ""


def _extract_usage(message: dict[str, Any]) -> dict[str, Any]:
    """Pull token + cost fields out of a Pi ``AssistantMessage``.

    Returns a dict with keys ``input``, ``output``, ``cacheRead`` (ints or None)
    and ``cost_total`` (float or None). Never raises — every field is optional.
    """
    usage = message.get("usage") or {}
    cost = usage.get("cost") or {}
    return {
        "input": usage.get("input"),
        "output": usage.get("output"),
        "cacheRead": usage.get("cacheRead"),
        "cost_total": cost.get("total"),
    }


def _schema_instruction(schema: dict[str, Any]) -> str:
    """Build the prompt appendix emulating ``--output-schema``.

    Pi has no wire-level schema mechanism, so the schema is described in the
    prompt and the final assistant text is parsed as JSON at ``agent_end``
    (mirroring Codex's structured-output fallback).
    """
    return (
        "\n\nRespond with ONLY a single valid JSON object matching this JSON "
        "schema. Do not include any prose, explanations, or markdown fences "
        "outside the JSON.\n" + json.dumps(schema)
    )


def _resolve_skill_dir(skill_key: str) -> Path | None:
    """Best-effort resolution of a Beagle skill key to its on-disk directory.

    Pi has no skill registry (Beagle skills are Claude Code plugins), so
    :meth:`PiBackend.format_skill_invocation` injects a path reference. This
    helper searches the standard plugin locations for a directory whose slug
    matches ``skill_key`` and contains a ``SKILL.md``. The full Beagle
    skill-path resolver is tracked separately (design doc Phase 2); this is the
    sufficient-for-parity implementation. Returns ``None`` when unresolved — the
    caller degrades gracefully and never raises.
    """
    slug = skill_key.split(":")[-1]
    home = Path.home()
    search_roots: list[Path] = [
        home / ".claude" / "skills",
        home / ".agents" / "skills",
        Path.cwd() / ".claude" / "skills",
        Path.cwd() / ".agents" / "skills",
    ]
    extra = os.environ.get("DAYDREAM_SKILLS_DIR")
    if extra:
        search_roots.append(Path(extra))
    for root in search_roots:
        candidate = root / slug
        if (candidate / "SKILL.md").is_file():
            return candidate
    return None


class PiBackend:
    """Backend that wraps the Pi CLI subprocess.

    Translates the Pi JSONL event stream into the unified ``AgentEvent`` stream
    so trajectory recording (ATIF v1.6) works identically to Claude/Codex.
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
        """Execute a prompt via the Pi CLI and yield unified events.

        Args:
            cwd: Working directory for the agent (passed as the process ``cwd``).
            prompt: The prompt to send (Pi's positional argument).
            output_schema: Optional JSON schema for structured output. Pi has no
                native schema flag, so the schema is appended to the prompt and
                the final assistant text is parsed as JSON at ``agent_end``.
            continuation: Optional token for session resumption. When present
                and ``backend == "pi"``, ``--session-id <id>`` resumes that
                session. Mutually exclusive with ``--no-session``.
            agents: Optional subagent mapping. Pi does not support non-empty
                subagent maps and will raise if provided.
            max_turns: NOT enforced by Pi (no direct turn-count flag). Documented
                gap; the argument is accepted for protocol parity only.
            read_only: When True, restrict Pi's tools to the read-only subset
                (``read,find,ls,grep``) so the agent cannot write/edit/bash.
                Cleaner than Codex's read-only sandbox — bash exclusion also
                blocks ``git commit``.

        Yields:
            AgentEvent instances.

        Raises:
            PiError: If a Pi turn ends with ``stopReason == "error"``.
            NotImplementedError: If ``agents`` is non-empty (Pi backend does not
                support exploration subagents).

        """
        if agents:
            raise NotImplementedError(
                "Pi backend does not support exploration subagents; use --backend claude for exploration."
            )

        args: list[str] = [
            "pi",
            "--mode",
            "json",
            "--model",
            self.model,
        ]

        provider = os.environ.get("PI_PROVIDER")
        api_key = os.environ.get("PI_API_KEY")
        thinking = os.environ.get("PI_THINKING")
        if provider:
            args.extend(["--provider", provider])
        if api_key:
            args.extend(["--api-key", api_key])
        if thinking:
            args.extend(["--thinking", thinking])

        if read_only:
            args.extend(["--tools", _PI_READ_ONLY_TOOLS])

        # --session-id and --no-session are mutually exclusive (plan §10).
        # Resume takes precedence; fresh runs are ephemeral.
        resume_id: str | None = None
        if continuation and continuation.backend == "pi":
            resume_id = continuation.data.get("session_id")
        if resume_id:
            args.extend(["--session-id", resume_id])
        else:
            args.append("--no-session")

        full_prompt = prompt
        if output_schema:
            full_prompt = prompt + _schema_instruction(output_schema)
        args.append(full_prompt)

        session_id: str | None = None
        last_assistant_text: str | None = None
        structured_result: Any = None
        finalized = False

        total_input = 0
        total_output = 0
        total_cache_read: int | None = None
        total_cost: float | None = None

        proc: asyncio.subprocess.Process | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(cwd),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_PI_STDOUT_LIMIT_BYTES,
            )
            self._processes.append(proc)
            self._process = proc

            is_first_line = True
            while True:
                if proc.stdout is None:
                    break
                line = await proc.stdout.readline()
                if not line:
                    break

                raw_line = line.decode().strip()
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                if is_first_line:
                    is_first_line = False
                    # Session header — capture the session id for the
                    # continuation token. Header field name is not stable across
                    # Pi builds, so probe the common keys.
                    for key in ("sessionId", "session_id", "session", "id"):
                        val = event.get(key)
                        if isinstance(val, str) and val:
                            session_id = val
                            break
                    # If the header also carries a dispatchable event type, fall
                    # through; otherwise it is a pure header — skip.
                    if event.get("type") not in _PI_EVENT_TYPES:
                        continue

                event_type = event.get("type", "")

                if event_type == "agent_start":
                    pass

                elif event_type == "message_end":
                    msg = event.get("message") or {}
                    if msg.get("role") == "assistant":
                        text_parts: list[str] = []
                        for block in msg.get("content") or []:
                            if not isinstance(block, dict):
                                continue
                            btype = block.get("type")
                            if btype == "text":
                                text = block.get("text", "")
                                if text:
                                    yield TextEvent(text=text)
                                    text_parts.append(text)
                            elif btype == "thinking":
                                thinking_text = block.get("thinking", "")
                                if thinking_text:
                                    yield ThinkingEvent(text=thinking_text)
                        if text_parts:
                            last_assistant_text = "".join(text_parts)

                elif event_type == "tool_execution_start":
                    yield ToolStartEvent(
                        id=event.get("toolCallId") or str(uuid.uuid4()),
                        name=event.get("toolName", "unknown"),
                        input=event.get("args") or {},
                    )

                elif event_type == "tool_execution_end":
                    yield ToolResultEvent(
                        id=event.get("toolCallId") or str(uuid.uuid4()),
                        output=_render_tool_result(event.get("result")),
                        is_error=bool(event.get("isError", False)),
                    )

                elif event_type == "turn_end":
                    msg = event.get("message") or {}
                    stop_reason = msg.get("stopReason")
                    if stop_reason == "error":
                        raise PiError(msg.get("errorMessage") or "Unknown Pi error")
                    usage = _extract_usage(msg)
                    inp = usage["input"]
                    outp = usage["output"]
                    cached = usage["cacheRead"]
                    cost = usage["cost_total"]
                    if isinstance(inp, int) and isinstance(outp, int):
                        model_name = msg.get("responseModel") or msg.get("model") or self.model
                        yield MetricsEvent(
                            message_id="",  # Pi has no per-message id (plan §5).
                            prompt_tokens=inp,
                            completion_tokens=outp,
                            cached_tokens=cached if isinstance(cached, int) else None,
                            cost_usd=cost if isinstance(cost, (int, float)) else None,
                            model_name=model_name,
                        )
                        total_input += inp
                        total_output += outp
                        if isinstance(cached, int):
                            total_cache_read = (total_cache_read or 0) + cached
                        if isinstance(cost, (int, float)):
                            total_cost = (total_cost or 0.0) + cost
                    yield TurnEndEvent(message_id="")

                elif event_type == "agent_end":
                    if output_schema and last_assistant_text:
                        structured_result = _extract_json(last_assistant_text)
                    yield CostEvent(
                        cost_usd=total_cost,
                        input_tokens=total_input,
                        output_tokens=total_output,
                        cached_tokens=total_cache_read,
                        model_name=self.model,
                    )
                    token_session = session_id or resume_id or str(uuid.uuid4())
                    yield ResultEvent(
                        structured_output=structured_result,
                        continuation=ContinuationToken(
                            backend="pi",
                            data={"session_id": token_session},
                        ),
                    )
                    finalized = True

                # turn_start / message_start / message_update /
                # tool_execution_update are streaming-only; the full content is
                # already captured at message_end / tool_execution_end.

            await proc.wait()

            # Guard (plan §10): if the stream ended without agent_end, still
            # finalize exactly once so downstream consumers get Cost/Result.
            if not finalized:
                if output_schema and last_assistant_text:
                    structured_result = _extract_json(last_assistant_text)
                yield CostEvent(
                    cost_usd=total_cost,
                    input_tokens=total_input,
                    output_tokens=total_output,
                    cached_tokens=total_cache_read,
                    model_name=self.model,
                )
                token_session = session_id or resume_id or str(uuid.uuid4())
                yield ResultEvent(
                    structured_output=structured_result,
                    continuation=ContinuationToken(
                        backend="pi",
                        data={"session_id": token_session},
                    ),
                )
                finalized = True

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

    async def cancel(self) -> None:
        """Cancel the running Pi process.

        Sends SIGTERM, waits briefly, then SIGKILL if still running (mirrors
        ``CodexBackend.cancel``).
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
        """Format a skill invocation for Pi via path-reference injection.

        Pi has no skill registry (Beagle skills are Claude Code plugins), so we
        resolve the skill directory on disk and instruct the agent to read its
        ``SKILL.md``. When the directory cannot be resolved, the raw skill key
        is kept as a hint — the review proceeds; this method never raises
        (design doc §5 degradation path).

        Args:
            skill_key: Full skill key (e.g. "beagle-python:review-python").
            args: Optional arguments string.

        Returns:
            Formatted skill invocation string.

        """
        skill_dir = _resolve_skill_dir(skill_key)
        if skill_dir is not None:
            base = (
                f"Read `{skill_dir}/SKILL.md` and follow it as your review "
                f"methodology. Read its companion files as it directs."
            )
        else:
            base = f"Follow the `{skill_key}` skill methodology."
        if args:
            base = f"{base} {args}"
        return base
