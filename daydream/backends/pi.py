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

z.ai coding plan wiring (GLM models) is registered via a pi extension at
``~/.pi/extensions/zai-provider/`` that calls ``pi.registerProvider()`` with
``baseUrl``, ``apiKey``, ``api: "openai-completions"``, and the six GLM
models (see https://pi.dev/docs/latest/custom-provider). When no model is
selected by daydream, Pi's configured ``defaultModel`` is respected; only when
Pi has no configured model does daydream pass ``glm-5.2`` with provider
``zai`` as its fallback. Explicit model and ``PI_PROVIDER`` / ``PI_API_KEY`` /
``PI_THINKING`` values remain CLI overrides. The provider extension is
installed once via ``pi install ~/.pi/extensions/zai-provider``; daydream never
fabricates a base URL or writes a models.json override.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
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
from daydream.backends._subprocess import cancel_processes, terminate_process
from daydream.config import DEFAULT_PI_MODEL
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


def _read_pi_default_model(path: Path) -> str | None:
    """Read Pi's configured default model, ignoring malformed settings."""
    try:
        with path.open(encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
    except (OSError, json.JSONDecodeError):
        return None

    model = settings.get("defaultModel") if isinstance(settings, dict) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


def _configured_pi_model(cwd: Path) -> str | None:
    """Return the effective Pi settings default, if one is configured.

    Pi merges project settings over global settings. We mirror only the
    ``defaultModel`` field because that is the setting daydream must not replace
    with its GLM fallback.
    """
    agent_dir = Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent"))
    settings_paths = (cwd / ".pi" / "settings.json", agent_dir / "settings.json")
    for settings_path in settings_paths:
        model = _read_pi_default_model(settings_path)
        if model:
            return model
    return None

# Matches Pi's native ``/skill:<slug>`` command embedded in a prompt (emitted by
# ``format_skill_invocation``). Used in ``execute`` to register the referenced
# skill directories with the subprocess via ``--skill`` flags. The character
# class mirrors Pi's documented skill-name grammar (lowercase a-z, 0-9, hyphens
# only; see pi docs/skills.md "Name Rules"): uppercase and underscore are
# excluded because no valid Pi slug uses them, so the wider ``\w`` class would
# otherwise admit tokens that can never resolve to a real skill.
_SKILL_TOKEN_RE = re.compile(r"/skill:([a-z0-9-]+)")

# Pi CLI ships only a minimal built-in system prompt. Claude Code and Codex
# inject rich guidance (tool efficiency, exploration strategy, conciseness) at
# the CLI layer; Pi does not, so the GLM model burns its tool-call budget on
# exploratory reads during LISTEN. This preamble is appended (via
# ``--append-system-prompt``) to Pi's built-in coding-assistant prompt to
# mirror that guidance. Keep it concise — the model re-reads it every turn.
_PI_SYSTEM_PREAMBLE = """\
You are an efficient coding agent operating under a strict tool-call budget.
You have a LIMITED number of tool calls per turn (typically 50). Every call is
precious — make each one count.

WORK STRATEGY:
- Search before you read. Use grep/find/ls to map relevant locations before
  opening any file. Prefer one targeted grep over three sequential reads.
- Batch related reads. Don't read files one at a time in a loop when a single
  grep would surface every relevant location.
- Read the diff first. If a diff file or git output is in your context, start
  there; only explore files referenced by the diff or their direct imports.
- Don't re-read what you've already read. If a file's content is already in
  your context (prior tool result, the diff, the prompt), reuse it.
- Answer directly when you can. If the existing context (commit log, diff,
  prior tool results) already answers the question, respond without additional
  tool calls.
- Stop exploring once you know enough. The goal is understanding and reporting,
  not exhaustive codebase enumeration. When you have enough, produce your
  answer immediately.

GIT CONTEXT:
You are operating in a git repository. Use `git diff`, `git log`, and
`git show` to understand changes efficiently — they are usually cheaper than
reading whole files.

Be concise in your responses. Do not narrate exploration step by step; report
findings and conclusions."""


_PI_DEFAULT_RETRY_ATTEMPTS = 3
_PI_DEFAULT_RETRY_BASE_DELAY = 2.0

STREAM_DROP_SIGNATURES = (
    "terminated",
    "econnreset",
    "connection reset",
    "socket hang up",
    "premature close",
    "epipe",
)

logger = logging.getLogger(__name__)


def _pi_retry_attempts() -> int:
    raw = os.environ.get("DAYDREAM_PI_RETRY_ATTEMPTS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "DAYDREAM_PI_RETRY_ATTEMPTS=%r is not a valid integer; using default %d",
                raw,
                _PI_DEFAULT_RETRY_ATTEMPTS,
            )
        else:
            if value < 0:
                logger.warning(
                    "DAYDREAM_PI_RETRY_ATTEMPTS=%r is negative; using default %d",
                    raw,
                    _PI_DEFAULT_RETRY_ATTEMPTS,
                )
            else:
                return value
    return _PI_DEFAULT_RETRY_ATTEMPTS


def _pi_retry_base_delay() -> float:
    raw = os.environ.get("DAYDREAM_PI_RETRY_BASE_DELAY_S")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "DAYDREAM_PI_RETRY_BASE_DELAY_S=%r is not a valid float; using default %g",
                raw,
                _PI_DEFAULT_RETRY_BASE_DELAY,
            )
        else:
            if not math.isfinite(value):
                logger.warning(
                    "DAYDREAM_PI_RETRY_BASE_DELAY_S=%r is not finite; using default %g",
                    raw,
                    _PI_DEFAULT_RETRY_BASE_DELAY,
                )
            elif value < 0:
                logger.warning(
                    "DAYDREAM_PI_RETRY_BASE_DELAY_S=%r is negative; using default %g",
                    raw,
                    _PI_DEFAULT_RETRY_BASE_DELAY,
                )
            else:
                return value
    return _PI_DEFAULT_RETRY_BASE_DELAY


def _is_retryable_error_message(message: str) -> bool:
    """Return True if the error message signals a transient overload or rate-limit.

    High-precision literals (429, rate limit/rate_limit, too many requests) are
    matched as plain substrings — they are extremely unlikely to appear in a
    non-transient context. Ambiguous terms require positive overload/capacity
    wording and explicitly reject negated or planning contexts.
    """
    lower = message.lower()
    # Unambiguous literals — plain substring is safe.
    if any(token in lower for token in ("429", "rate limit", "rate_limit", "too many requests")):
        return True
    if re.search(r"\bnot\s+overloaded\b|\bcapacity\s+planning\b", lower):
        return False
    if bool(
        re.search(r"\boverloaded?\b|\boverload(?:ed|ing)?\b", lower)
        or re.search(r"\bcapacity\s+(?:unavailable|exceeded|limit|limited|full|reached)\b", lower)
        or re.search(r"\bthrottl(?:e|ed|ing)\b", lower)
    ):
        return True
    # Stream-drop signatures (terminated, econnreset, premature close, ...).
    #
    # Unlike daydream/benchmark/daydream_run.py:_is_transient — which scans raw
    # stdout and therefore gates STREAM_DROP_SIGNATURES behind
    # _ERROR_CONTEXT_MARKERS so the substrings only count when daydream actually
    # errored — this function is only ever invoked on a PiError `errorMessage`
    # (see the turn_end / stopReason == "error" call site below), where error
    # context is already implied by construction. The asymmetry is deliberate:
    # the benchmark's _ERROR_CONTEXT_MARKERS gate is the harness's own concern
    # for stdout scanning, not a contract production must mirror. Matching these
    # signatures unconditionally here is therefore safe and correct.
    if any(sig in lower for sig in STREAM_DROP_SIGNATURES):
        return True
    return False


def _is_retryable_exit_code(code: int) -> bool:
    """Return True for exit codes that indicate OOM/SIGKILL rather than a logic error."""
    return code in (-9, 137)


class PiError(Exception):
    """Raised when a Pi turn fails (e.g. ``stopReason == "error"``)."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


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


def _skill_slug(skill_key: str) -> str:
    """Strip a Beagle plugin prefix from a skill key, leaving the bare slug.

    ``beagle-python:review-python`` -> ``review-python``; a key that already
    has no prefix (``review-python``) is returned unchanged. Pi has no plugin
    namespace, so its native ``/skill:<slug>`` command and on-disk skill
    directories are keyed by the bare slug.
    """
    return skill_key.split(":")[-1]


def _resolve_skill_dir(skill_key: str) -> Path | None:
    """Best-effort resolution of a Beagle skill key to its on-disk directory.

    Searches, in priority order, for a directory whose slug matches
    ``skill_key`` and contains a ``SKILL.md``:

    1. ``$DAYDREAM_SKILLS_DIR`` — the configurable override; when set it is
       searched first so it wins over the built-in locations.
    2. ``~/.agents/skills`` — the primary default location.
    3. ``~/.claude/skills`` — the legacy Claude Code plugin mirror.
    4. the project-local ``.agents/skills`` then ``.claude/skills`` mirrors.

    Returns ``None`` when unresolved; the caller degrades gracefully and this
    function never raises.

    Note: this is a parallel skill-location mechanism —
    :func:`daydream.deep.orchestrator.get_installed_skills` answers the
    same question by reading the Claude Code plugin registry. The two
    can disagree on where the Beagle skills live; consolidate when the
    full resolver lands.
    """
    slug = _skill_slug(skill_key)
    home = Path.home()
    search_roots: list[Path] = []
    extra = os.environ.get("DAYDREAM_SKILLS_DIR")
    if extra:
        search_roots.append(Path(extra))
    search_roots.extend(
        [
            home / ".agents" / "skills",
            home / ".claude" / "skills",
            Path.cwd() / ".agents" / "skills",
            Path.cwd() / ".claude" / "skills",
        ]
    )
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

    concise_fix_prompts = True  # GLM produces verbose reasoning in fix prompts

    def __init__(self, model: str | None = None, *, cwd: Path | None = None):
        """Initialize the backend with an optional explicit model override."""
        self._model_override = model
        # ``.model`` must be resolved at construction (runner/recorder read it
        # before execute); cache the settings lookup so execute() need not
        # re-read settings.json for the same workspace.
        self._configured_cache: tuple[Path, str | None] | None = None
        configured: str | None = None
        if model is None and cwd is not None:
            configured = _configured_pi_model(cwd)
            self._configured_cache = (cwd, configured)
        self.model = model or configured or DEFAULT_PI_MODEL
        self.fanout_concurrency = 2
        self.retry_attempts = _pi_retry_attempts()
        self.retry_base_delay_s = _pi_retry_base_delay()
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

        Raises:
            PiError: If a Pi turn ends with ``stopReason == "error"``.
            NotImplementedError: If ``agents`` is non-empty (Pi backend does not
                support exploration subagents).
        """
        if agents:
            raise NotImplementedError(
                "Pi backend does not support exploration subagents; use --backend claude for exploration."
            )

        args: list[str] = ["pi", "--mode", "json"]

        configured_model = None
        provider: str | None = None
        if self._model_override is not None:
            self.model = self._model_override
            args.extend(["--model", self.model])
            provider = os.environ.get("PI_PROVIDER", "zai")
        else:
            if self._configured_cache is not None and self._configured_cache[0] == cwd:
                configured_model = self._configured_cache[1]
            else:
                configured_model = _configured_pi_model(cwd)
            self.model = configured_model or DEFAULT_PI_MODEL
            if configured_model is None:
                args.extend(["--model", self.model])
            provider = os.environ.get("PI_PROVIDER")
            if provider is None and configured_model is None:
                provider = "zai"

        api_key = os.environ.get("PI_API_KEY")
        thinking = os.environ.get("PI_THINKING")
        if provider:
            args.extend(["--provider", provider])
        if api_key:
            args.extend(["--api-key", api_key])
        if thinking:
            args.extend(["--thinking", thinking])

        # Pi's built-in system prompt is minimal; append the daydream preamble
        # so the GLM model gets the same tool-efficiency / budget-awareness
        # guidance that Claude Code and Codex inject natively via their CLIs.
        args.extend(["--append-system-prompt", _PI_SYSTEM_PREAMBLE])

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

        # Register any /skill:<slug> commands referenced in the prompt with the
        # subprocess via repeatable --skill <dir> flags so availability does not
        # depend solely on ambient ~/.agents/skills mirrors. Best-effort and
        # de-duplicated: slugs that don't resolve to an on-disk skill dir are
        # skipped (pi may still auto-discover them) and never crash the run.
        registered_dirs: set[str] = set()
        for slug in _SKILL_TOKEN_RE.findall(full_prompt):
            skill_dir = _resolve_skill_dir(slug)
            if skill_dir is None:
                continue
            resolved = str(skill_dir)
            if resolved in registered_dirs:
                continue
            registered_dirs.add(resolved)
            args.extend(["--skill", resolved])

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
                limit=_PI_STDOUT_LIMIT_BYTES,
            )
            self._processes.append(proc)

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
                        error_msg = msg.get("errorMessage") or "Unknown Pi error"
                        raise PiError(error_msg, retryable=_is_retryable_error_message(error_msg))
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
            returncode = proc.returncode
            if returncode is not None and returncode != 0:
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
                    f"Pi CLI exited with return code {returncode}.{detail}",
                    retryable=_is_retryable_exit_code(returncode),
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
                await terminate_process(proc)
            self._processes = [active for active in self._processes if active is not proc]

    async def cancel(self) -> None:
        """Cancel the running Pi process.

        Sends SIGTERM, waits briefly, then SIGKILL if still running (mirrors
        ``CodexBackend.cancel``).
        """
        await cancel_processes(self._processes)

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        """Format a skill invocation as Pi's native ``/skill:<slug>`` command.

        Pi exposes installed skills through a built-in slash command keyed by
        the bare slug (it has no plugin namespace), so a Beagle key like
        ``beagle-python:review-python`` becomes ``/skill:review-python``. The
        plugin prefix is stripped via :func:`_skill_slug`; the matching skill
        directory is registered with the subprocess in :meth:`execute` via a
        ``--skill`` flag so the command resolves even without an ambient
        mirror. This method never raises.
        """
        base = f"/skill:{_skill_slug(skill_key)}"
        if args:
            base = f"{base} {args}"
        return base
