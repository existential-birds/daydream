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

z.ai coding plan wiring (GLM models) is configured once in
``~/.pi/agent/models.json`` (see ``design.md`` §3); daydream never fabricates a
base URL or cost table. Optional ``PI_PROVIDER`` / ``PI_API_KEY`` /
``PI_THINKING`` env overrides are forwarded as CLI flags. ``PI_BASE_URL`` is not
a CLI flag — when set, a throwaway models.json override is written and pi is
pointed at it via ``PI_CODING_AGENT_DIR`` (plan §6), so the user's persistent
``~/.pi/agent/models.json`` is never mutated.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
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
from daydream.json_utils import extract_json

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

# Pi reads its agent config (models.json, auth.json, settings.json) from its
# agent dir, which is overridable via PI_CODING_AGENT_DIR (verified in pi's
# src/config.ts: ENV_AGENT_DIR = `${APP_NAME.toUpperCase()}_CODING_AGENT_DIR`,
# getAgentDir() honors it, getModelsPath() = join(getAgentDir(), "models.json")).
# Used to write a throwaway models.json for PI_BASE_URL (plan §6) without
# touching the user's persistent ~/.pi/agent/models.json.
_PI_AGENT_DIR_ENV = "PI_CODING_AGENT_DIR"

# Default provider key the PI_BASE_URL override targets when PI_PROVIDER is
# unset. The z.ai coding plan (DEFAULT_PI_MODEL = "glm-5.2") uses "zai"
# (plan §3; matches a configured ~/.pi/agent/models.json).
_PI_DEFAULT_OVERRIDE_PROVIDER = "zai"


class PiError(Exception):
    """Raised when a Pi turn fails (e.g. ``stopReason == "error"``)."""


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

    Searches the standard plugin locations for a directory whose slug
    matches ``skill_key`` and contains a ``SKILL.md``. Returns ``None``
    when unresolved; the caller degrades gracefully and this function
    never raises.

    Note: this is a parallel skill-location mechanism —
    :func:`daydream.deep.orchestrator.get_installed_skills` answers the
    same question by reading the Claude Code plugin registry. The two
    can disagree on where the Beagle skills live; consolidate when the
    full resolver lands.
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


def _pi_agent_models_path() -> Path:
    """Path to the user's persistent pi models.json (``~/.pi/agent/models.json``).

    Computed at call time (not import) so ``HOME`` overrides in tests are
    honored. Mirrors pi's own resolution: ``getModelsPath() ==
    join(getAgentDir(), "models.json")`` with the default agent dir
    ``~/.pi/agent`` (pi ``src/config.ts``).
    """
    return Path.home() / ".pi" / "agent" / "models.json"


def _write_models_override(base_url: str, provider: str, api_key: str | None) -> Path:
    """Write a throwaway models.json carrying a ``baseUrl`` override.

    ``baseUrl`` is not a Pi CLI flag — it lives in pi's models.json (plan §6).
    Pi locates models.json via its agent dir, which is overridable through the
    ``PI_CODING_AGENT_DIR`` env var, so this builds a temporary agent dir
    containing a models.json that *merges* the user's existing config with the
    ``base_url`` override applied to ``provider`` (and ``api_key`` when given),
    and returns the dir path for the caller to pass as that env var. The user's
    persistent ``~/.pi/agent/models.json`` is never mutated.

    Best-effort merge: an unreadable/absent/invalid existing models.json yields
    a fresh config with just the override entry. Temp-dir/write failures raise
    ``OSError`` so the caller can degrade to "no override" (plan §6:
    best-effort).

    Args:
        base_url: The ``baseUrl`` to set on the provider entry (``PI_BASE_URL``).
        provider: Provider key to override (``PI_PROVIDER`` or the z.ai default).
        api_key: Optional ``apiKey`` to also set (``PI_API_KEY``).

    Returns:
        Path to the temporary agent dir containing the written ``models.json``.

    """
    temp_dir = Path(tempfile.mkdtemp(prefix="daydream-pi-models-"))
    try:
        config: dict[str, Any] = {"providers": {}}
        existing = _pi_agent_models_path()
        try:
            if existing.is_file():
                parsed = json.loads(existing.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    config = parsed
        except (OSError, ValueError):
            # Absent/unreadable/invalid existing config — start fresh.
            config = {"providers": {}}
        providers = config.get("providers")
        if not isinstance(providers, dict):
            providers = {}
            config["providers"] = providers
        entry = providers.get(provider)
        if not isinstance(entry, dict):
            entry = {}
            providers[provider] = entry
        entry["baseUrl"] = base_url
        if api_key:
            entry["apiKey"] = api_key
        (temp_dir / "models.json").write_text(json.dumps(config), encoding="utf-8")
    except OSError:
        # Leave no half-written temp dir behind on failure.
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return temp_dir


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
                session. Fresh runs generate a UUID and also pass
                ``--session-id <uuid>`` so the returned token always points to
                a saved (resumable) session — Pi's ``--no-session`` is
                ephemeral and cannot be resumed.
            agents: Optional subagent mapping. Pi does not support non-empty
                subagent maps and will raise if provided.
            max_turns: NOT enforced by Pi (no direct turn-count flag). Documented
                gap; the argument is accepted for protocol parity only.
            read_only: When True, restricts Pi's tools to the read-only subset
                (``read,find,ls,grep``) so the agent cannot write/edit/bash.

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
        base_url = os.environ.get("PI_BASE_URL")
        if provider:
            args.extend(["--provider", provider])
        if api_key:
            args.extend(["--api-key", api_key])
        if thinking:
            args.extend(["--thinking", thinking])

        # PI_BASE_URL has no CLI flag — it lives in pi's models.json (plan §6).
        # When set, write a throwaway models.json in a temp agent dir and point
        # pi at it via PI_CODING_AGENT_DIR. The user's persistent config is
        # never mutated; failures degrade to "no override" (best-effort).
        models_override_dir: Path | None = None
        sub_env: dict[str, str] | None = None
        if base_url:
            try:
                models_override_dir = _write_models_override(
                    base_url, provider or _PI_DEFAULT_OVERRIDE_PROVIDER, api_key
                )
                sub_env = {
                    **os.environ,
                    _PI_AGENT_DIR_ENV: str(models_override_dir),
                }
            except OSError:
                from daydream.agent import console
                from daydream.ui import print_warning

                print_warning(
                    console,
                    "PI_BASE_URL is set but the temporary models.json override "
                    "could not be written; set baseUrl in "
                    "~/.pi/agent/models.json instead.",
                )

        if read_only:
            args.extend(["--tools", _PI_READ_ONLY_TOOLS])

        # Pi's --no-session is ephemeral (pi docs: "Don't save session
        # (ephemeral)"); resuming it with --session-id <id> "creates it if
        # missing" → an empty session that discards prior turns. So every run
        # uses --session-id with a persistent id: the continuation's id when
        # resuming, or a freshly generated UUID for a new session.
        resume_id: str | None = None
        if continuation and continuation.backend == "pi":
            resume_id = continuation.data.get("session_id")
        effective_session_id = resume_id or str(uuid.uuid4())
        args.extend(["--session-id", effective_session_id])

        full_prompt = prompt
        if output_schema:
            full_prompt = prompt + _schema_instruction(output_schema)
        args.append(full_prompt)

        session_id: str | None = None
        last_assistant_text: str | None = None
        structured_result: Any = None
        # Non-JSON lines (stderr merged into stdout, pi diagnostic output, etc.)
        # captured for error reporting when the process exits non-zero.
        stderr_lines: list[str] = []

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
                # Merge stderr into stdout (matching CodexBackend). stderr=PIPE
                # here would never be drained — a latent deadlock if pi writes
                # more than the OS pipe buffer (~64KB) to stderr. The parse loop
                # below already `continue`s past non-JSON lines, so stray stderr
                # text mixed into stdout is harmlessly skipped.
                stderr=asyncio.subprocess.STDOUT,
                env=sub_env,
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
                    # Capture non-JSON lines — these are stderr merged into
                    # stdout (pi diagnostics, login prompts, errors). Kept for
                    # error reporting when the process exits non-zero.
                    if len(stderr_lines) < 20:
                        stderr_lines.append(raw_line)
                    continue

                if is_first_line:
                    is_first_line = False
                    # Session header — capture the session id for the
                    # continuation token. Header field name is not stable across
                    # Pi builds, so probe the common keys.
                    for key in ("id", "sessionId", "session_id", "session"):
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
                    pass  # Lifecycle marker; nothing to emit.

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
                    # No inline finalization — Cost/Result are emitted once from
                    # the single post-loop path below (plan §10). The loop keeps
                    # draining to EOF so the stdout pipe cannot fill mid-run.
                    pass

                # turn_start / message_start / message_update /
                # tool_execution_update are streaming-only; the full content is
                # already captured at message_end / tool_execution_end.

            await proc.wait()

            # Fail fast on non-zero exit: if pi crashed without emitting a
            # turn_end error event, surface the failure with diagnostic output
            # instead of reporting a successful completion with empty/partial
            # output.
            if proc.returncode != 0:
                stderr_tail = "\n".join(stderr_lines[-10:])
                if stderr_lines:
                    detail = (
                        f"\nPi CLI output (last {len(stderr_lines)} "
                        f"non-JSON lines):\n{stderr_tail}"
                    )
                else:
                    detail = (
                        "\n(no non-JSON output captured — pi may have "
                        "crashed before writing to stdout)"
                    )
                raise PiError(
                    f"Pi CLI exited with return code {proc.returncode}.{detail}"
                )

            # Single finalization path (plan §10): runs exactly once whether
            # the stream closed on agent_end or ended without it (truncated
            # output). Cost/Result are derived from fully-accumulated totals.
            if output_schema and last_assistant_text:
                structured_result = extract_json(last_assistant_text)
            yield CostEvent(
                cost_usd=total_cost,
                input_tokens=total_input,
                output_tokens=total_output,
                cached_tokens=total_cache_read,
                model_name=self.model,
            )
            # Prefer the id pi reports in the session header (authoritative);
            # fall back to the persistent id we passed via --session-id.
            token_session = session_id or effective_session_id
            yield ResultEvent(
                structured_output=structured_result,
                continuation=ContinuationToken(
                    backend="pi",
                    data={"session_id": token_session},
                ),
            )

        finally:
            if proc is not None and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            if models_override_dir is not None:
                shutil.rmtree(models_override_dir, ignore_errors=True)
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
