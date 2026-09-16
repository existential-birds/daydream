# daydream/backends/pi.py
"""Pi CLI subprocess backend for daydream.

Spawns ``pi --mode json`` (the ``@earendil-works/pi-coding-agent`` TypeScript
coding agent) as an async subprocess, reads the JSONL event stream from stdout,
and translates it into the unified :data:`daydream.backends.AgentEvent` stream.

Pi is a subprocess + JSONL backend — the same proven shape as
:mod:`daydream.backends.codex`. The Pi backend is a second instance of that
pattern; it emits the same event vocabulary so the existing
:class:`daydream.trajectory.TrajectoryRecorder` produces valid ATIF v1.7
trajectories indistinguishable in shape from the other two backends.

Nous research wiring (DeepSeek models) is configured via pi's
``~/.pi/agent/`` provider registry (``models.json`` custom ``nous`` provider
pointing at ``https://inference-api.nousresearch.com/v1``, key in
``auth.json``). When no model is selected by daydream, Pi's configured
``defaultModel`` is respected; only when Pi has no configured model does
daydream pass ``deepseek/deepseek-v4-flash-0731`` with provider
``nous`` as its fallback. Explicit model and ``PI_PROVIDER`` / ``PI_API_KEY`` /
``PI_THINKING`` values remain CLI overrides. Daydream never fabricates a base
URL or writes a models.json override.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from daydream.backends import (
    AgentEvent,
    BackendExecutionInput,
    ContinuationToken,
    CostEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    PiRequestConfig,
    ReasoningChoicePart,
    RequestEvent,
    ResultEvent,
    TextChoicePart,
    TextEvent,
    ThinkingEvent,
    ToolCallChoicePart,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    _admit_json_value,
    _admit_native_unix_ms,
    _new_generation_id,
    resolve_fanout_concurrency,
    unix_ms_to_ns,
)
from daydream.backends._subprocess import (
    DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S,
    DEFAULT_STREAM_IDLE_TIMEOUT_S,
    stream_idle_timeout_s,
)
from daydream.backends._transport import (
    CliTransport,
    StderrPolicy,
    StdinMode,
    TransportExitError,
)
from daydream.config import DEFAULT_PI_MODEL
from daydream.json_utils import extract_json

# Mirror Codex's generous stdout cap so large JSONL events (big file reads,
# patch payloads) do not trip asyncio's "chunk is longer than limit" guard.
_PI_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024

# Known AgentSessionEvent types. Used to decide whether the first
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

# Read-only tool subset. Excludes the mutating edit/bash/write tools.
_PI_READ_ONLY_TOOLS = "read,find,ls,grep"

_PI_PROVIDER_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "ZAI_API_KEY",
    "nous": "NOUS_API_KEY",
}


def _read_pi_default_model(path: Path) -> str | None:
    """Read Pi's configured default model, ignoring malformed settings."""
    try:
        with path.open(encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
    except (OSError, json.JSONDecodeError):
        return None

    model = settings.get("defaultModel") if isinstance(settings, dict) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


_AMBIENT_AGENT_DIR = object()


def _configured_pi_model(
    cwd: Path, *, agent_dir: Path | None | object = _AMBIENT_AGENT_DIR
) -> str | None:
    """Return the effective Pi settings default, if one is configured.

    Pi merges project settings over global settings. We mirror only the
    ``defaultModel`` field because that is the setting daydream must not replace
    with its DeepSeek fallback.
    """
    if agent_dir is _AMBIENT_AGENT_DIR:
        resolved_agent_dir: Path | None = Path(
            os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent")
        )
    else:
        assert agent_dir is None or isinstance(agent_dir, Path)
        resolved_agent_dir = agent_dir
    settings_paths = [cwd / ".pi" / "settings.json"]
    if resolved_agent_dir is not None:
        settings_paths.append(resolved_agent_dir / "settings.json")
    for settings_path in settings_paths:
        model = _read_pi_default_model(settings_path)
        if model:
            return model
    return None


# Pi CLI ships only a minimal built-in system prompt. Claude Code and Codex
# inject rich guidance (tool efficiency, exploration strategy, conciseness) at
# the CLI layer; Pi does not, so the default DeepSeek model burns its
# tool-call budget on exploratory reads during LISTEN. This preamble is
# appended (via ``--append-system-prompt``) to Pi's built-in coding-assistant
# prompt to mirror that guidance. Keep it concise — the model re-reads it
# every turn.
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


_PI_DEFAULT_RETRY_ATTEMPTS = 20
_PI_DEFAULT_RETRY_BASE_DELAY = 10.0
_PI_DEFAULT_RETRY_MAX_DELAY = 120.0
_PI_DEFAULT_FANOUT_CONCURRENCY = 10

STREAM_DROP_SIGNATURES = (
    "terminated",
    "econnreset",
    "connection reset",
    "socket hang up",
    "premature close",
    "epipe",
)

logger = logging.getLogger(__name__)

# The provider half of the daydream-supplied default pairing (the model half
# is ``DEFAULT_PI_MODEL``). Single source of truth so the argv fallback
# branches, the migration warnings, and the module docs cannot drift.
_PI_DEFAULT_PROVIDER = "nous"

# One-shot migration-warning guard: ``execute`` runs once per phase,
# invocation, and retry attempt, so a stale pre-migration configuration would
# otherwise re-log the identical warning for every call. Keys are per-mismatch.
_warned_migration_mismatches: set[str] = set()


def _warn_migration_mismatch_once(key: str, message: str, *args: object) -> None:
    """Log a stale-configuration warning at most once per ``key`` per process."""
    if key in _warned_migration_mismatches:
        return
    _warned_migration_mismatches.add(key)
    logger.warning(message, *args)


def _pi_retry_attempts() -> int:
    raw = os.environ.get("DAYDREAM_PI_RETRY_ATTEMPTS")
    if raw is None:
        return _PI_DEFAULT_RETRY_ATTEMPTS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "DAYDREAM_PI_RETRY_ATTEMPTS=%r is not a valid integer; using default %d",
            raw,
            _PI_DEFAULT_RETRY_ATTEMPTS,
        )
        return _PI_DEFAULT_RETRY_ATTEMPTS
    if value < 0:
        logger.warning(
            "DAYDREAM_PI_RETRY_ATTEMPTS=%r is negative; using default %d",
            raw,
            _PI_DEFAULT_RETRY_ATTEMPTS,
        )
        return _PI_DEFAULT_RETRY_ATTEMPTS
    return value


def _pi_retry_delay(env_name: str, default: float) -> float:
    """Read one finite, non-negative retry delay from the environment."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a valid float; using default %g",
            env_name,
            raw,
            default,
        )
        return default
    if not math.isfinite(value):
        logger.warning(
            "%s=%r is not finite; using default %g",
            env_name,
            raw,
            default,
        )
        return default
    if value < 0:
        logger.warning(
            "%s=%r is negative; using default %g",
            env_name,
            raw,
            default,
        )
        return default
    return value


def _pi_retry_base_delay() -> float:
    return _pi_retry_delay("DAYDREAM_PI_RETRY_BASE_DELAY_S", _PI_DEFAULT_RETRY_BASE_DELAY)


def _pi_retry_max_delay() -> float:
    return _pi_retry_delay("DAYDREAM_PI_RETRY_MAX_DELAY_S", _PI_DEFAULT_RETRY_MAX_DELAY)


# Shared error-taxonomy tokens, used by both the retryable-message check and
# the stable diagnostic category so the two views of the taxonomy cannot drift
# out of sync.
_RATE_LIMIT_TOKENS = ("429", "rate limit", "rate_limit", "too many requests")
_SERVER_ERROR_TOKENS = (
    "502",
    "503",
    "bad gateway",
    "service unavailable",
    "server error",
)


def _is_stream_truncation_message(normalized_message: str) -> bool:
    """Return True when a normalized message reports an unfinished stream."""
    return "finish_reason" in normalized_message and any(
        token in normalized_message for token in ("stream", "ended", "end")
    )


def _is_retryable_error_message(message: str) -> bool:
    """Return True if the error message signals a transient overload or rate-limit.

    High-precision literals (429, rate limit/rate_limit, too many requests,
    502/503, bad gateway, service unavailable, server error) are matched as
    plain substrings — they are extremely unlikely to appear in a non-transient
    context. Ambiguous
    terms require positive overload/capacity wording and explicitly reject
    negated or planning contexts.
    """
    lower = message.lower()
    # Unambiguous literals — plain substring is safe.
    if any(token in lower for token in _RATE_LIMIT_TOKENS + _SERVER_ERROR_TOKENS):
        return True
    if _is_stream_truncation_message(lower):
        return True
    if any(token in lower for token in ("timed out", "timeout", "deadline exceeded")):
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


def _pi_error_category(message: str) -> str:
    """Classify Pi failures into stable host-owned diagnostic categories."""
    lower = message.casefold()
    if any(token in lower for token in _RATE_LIMIT_TOKENS):
        return "RATE_LIMIT"
    if any(token in lower for token in _SERVER_ERROR_TOKENS):
        return "SERVER_ERROR"
    if any(token in lower for token in ("timed out", "timeout", "deadline exceeded")):
        return "TIMEOUT"
    if _is_stream_truncation_message(lower):
        return "STREAM_TRUNCATION"
    if any(signature in lower for signature in STREAM_DROP_SIGNATURES):
        return "STREAM_DROP"
    if "pi cli exited with return code" in lower:
        return "PROCESS_EXIT"
    if any(
        token in lower
        for token in (
            "auth",
            "credential",
            "api key",
            "api_key",
            "configuration",
            "not configured",
            "provider",
            "model not found",
        )
    ):
        return "AUTH_CONFIG"
    return "UNKNOWN"


class PiError(Exception):
    """Raised when a Pi turn fails (e.g. ``stopReason == "error"``)."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        category: str = "UNKNOWN",
    ):
        super().__init__(message)
        self.retryable = retryable
        self.category = category


def _render_tool_result(result: Any) -> str:
    """Render a Pi ``AgentToolResult`` into a flat string for ``ToolResultEvent``.

    Plain text-only responses stay readable. Structured details and mixed
    content blocks are serialized as JSON so image/resource blocks and their
    accompanying text remain available to consumers.
    """
    if not isinstance(result, dict):
        return result if isinstance(result, str) else ("" if result is None else json.dumps(result))
    content = result.get("content")
    if result.get("details") not in (None, "") or (
        isinstance(content, list)
        and any(not isinstance(block, dict) or block.get("type") != "text" for block in content)
    ):
        return json.dumps(result, ensure_ascii=False)
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
    # Last resort — preserve the payload rather than dropping the observation.
    return json.dumps(result, ensure_ascii=False) if result else ""


def _extract_usage(message: dict[str, Any]) -> dict[str, Any]:
    """Pull token + cost fields out of a Pi ``AssistantMessage``.

    Returns ``input``, ``output``, ``cacheRead``, ``cacheWrite`` (ints or None)
    and ``cost_total`` (float or None). Never raises — every field is optional.
    """
    usage = message.get("usage") or {}
    cost = usage.get("cost") or {}
    return {
        "input": usage.get("input"),
        "output": usage.get("output"),
        "cacheRead": usage.get("cacheRead"),
        "cacheWrite": usage.get("cacheWrite"),
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


class PiBackend:
    """Backend that wraps the Pi CLI subprocess.

    Translates the Pi JSONL event stream into the unified ``AgentEvent`` stream
    so trajectory recording (ATIF v1.7) works identically to Claude/Codex.
    """

    concise_fix_prompts = True  # DeepSeek produces verbose reasoning in fix prompts

    def __init__(
        self,
        model: str | None = None,
        *,
        cwd: Path | None = None,
        reasoning_effort: str | None = None,
        execution_input: BackendExecutionInput | None = None,
    ):
        """Initialize the backend with an optional explicit model override.

        Args:
            reasoning_effort: Resolved per-phase reasoning level, forwarded as
                ``--thinking <level>``. When None, ``PI_THINKING`` is used
                instead — the env var is Pi's ambient default, the same role
                ``model_reasoning_effort`` in ``~/.codex/config.toml`` plays for
                Codex, so an explicitly resolved per-phase level outranks it.
        """
        self._model_override = model
        self.reasoning_effort = reasoning_effort
        self._execution_input = execution_input
        # ``.model`` must be resolved at construction (runner/recorder read it
        # before execute); cache the settings lookup so execute() need not
        # re-read settings.json for the same workspace.
        self._configured_cache: tuple[Path, str | None] | None = None
        configured: str | None = None
        if model is None and cwd is not None:
            configured = _configured_pi_model(
                cwd,
                agent_dir=(
                    execution_input.pi_agent_dir
                    if execution_input is not None
                    else _AMBIENT_AGENT_DIR
                ),
            )
            self._configured_cache = (cwd, configured)
        self.model = model or configured or DEFAULT_PI_MODEL
        if execution_input is not None:
            self.fanout_concurrency = execution_input.fanout_concurrency
            self.retry_policy = execution_input.retry_policy
            self.retry_attempts = self.retry_policy.attempts
            self.retry_base_delay_s = self.retry_policy.base_delay_s
            self.retry_max_delay_s = self.retry_policy.max_delay_s
        else:
            self.fanout_concurrency = resolve_fanout_concurrency(
                "DAYDREAM_PI_FANOUT_CONCURRENCY", _PI_DEFAULT_FANOUT_CONCURRENCY
            )
            self.retry_attempts = _pi_retry_attempts()
            self.retry_base_delay_s = _pi_retry_base_delay()
            self.retry_max_delay_s = _pi_retry_max_delay()
        self._transports: list[CliTransport] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute a prompt via the Pi CLI and yield unified events.

        Args:
            output_schema: Optional JSON schema for structured output. Pi has no
                native schema flag, so the schema is appended to the prompt and
                the final assistant text is parsed as JSON at ``agent_end``.
            continuation: Optional token for session resumption. When present
                and ``backend == "pi"``, ``--session-id <id>`` resumes that
                session when ``persist_session`` is True.
            agents: Optional subagent mapping. Pi does not support non-empty
                subagent maps and will raise if provided.
            max_turns: NOT enforced by Pi (no direct turn-count flag). Documented
                gap; the argument is accepted for protocol parity only.
            read_only: When True, restricts Pi's tools to the read-only subset
                (``read,find,ls,grep``) so the agent cannot write/edit/bash.
            persist_session: When False, pass ``--no-session`` and return no
                continuation. The default preserves resumable sessions.

        Raises:
            PiError: If a Pi turn ends with ``stopReason == "error"``.
            StreamStalledError: If the ``pi`` subprocess emits nothing on stdout
                for the idle window (see
                :func:`daydream.backends._subprocess.stream_idle_timeout_s`).
                Retryable — ``run_agent`` re-arms a fresh subprocess and retries.
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
            provider = (
                self._execution_input.pi_provider
                if self._execution_input is not None
                else os.environ.get("PI_PROVIDER")
            )
            if provider is None:
                provider = _PI_DEFAULT_PROVIDER
                if self.model.casefold().startswith("glm-"):
                    # The default provider moved zai -> nous; a pinned z.ai
                    # GLM model silently loses its provider and fails at
                    # runtime unless the user opts back in explicitly.
                    _warn_migration_mismatch_once(
                        f"glm-pin:{self.model}",
                        "Explicit model %r is a z.ai-hosted GLM model, but the "
                        "Pi backend now defaults to the %s provider and "
                        "PI_PROVIDER is unset; the pinned model will fail at "
                        "runtime unless PI_PROVIDER=zai is set or the model is "
                        "migrated to the %s registry.",
                        self.model,
                        _PI_DEFAULT_PROVIDER,
                        _PI_DEFAULT_PROVIDER,
                    )
        else:
            if self._configured_cache is not None and self._configured_cache[0] == cwd:
                configured_model = self._configured_cache[1]
            else:
                configured_model = _configured_pi_model(
                    cwd,
                    agent_dir=(
                        self._execution_input.pi_agent_dir
                        if self._execution_input is not None
                        else _AMBIENT_AGENT_DIR
                    ),
                )
            self.model = configured_model or DEFAULT_PI_MODEL
            if configured_model is None:
                args.extend(["--model", self.model])
            provider = (
                self._execution_input.pi_provider
                if self._execution_input is not None
                else os.environ.get("PI_PROVIDER")
            )
            if provider is None and configured_model is None:
                provider = _PI_DEFAULT_PROVIDER
            elif configured_model is None and provider and provider != _PI_DEFAULT_PROVIDER:
                # The previous default paired the zai provider with a GLM
                # model; an explicit PI_PROVIDER from that setup now pairs
                # with the DeepSeek fallback model and fails at runtime
                # unless the provider actually serves it.
                _warn_migration_mismatch_once(
                    f"fallback-provider:{provider}",
                    "PI_PROVIDER=%r is set with no configured model, so the Pi "
                    "backend falls back to %r, which is served by the %s "
                    "provider; if %r does not serve that model the run will "
                    "fail at runtime. Unset PI_PROVIDER or set it to %s to "
                    "adopt the new default, or configure a model explicitly.",
                    provider,
                    self.model,
                    _PI_DEFAULT_PROVIDER,
                    provider,
                    _PI_DEFAULT_PROVIDER,
                )

        child_env = (
            self._execution_input.child_environment()
            if self._execution_input is not None
            else os.environ.copy()
        )
        api_key = child_env.get("PI_API_KEY")
        thinking = self.reasoning_effort or (
            self._execution_input.pi_thinking
            if self._execution_input is not None
            else os.environ.get("PI_THINKING")
        )
        if provider:
            args.extend(["--provider", provider])
        if thinking:
            args.extend(["--thinking", thinking])

        child_env.pop("PI_API_KEY", None)
        if api_key:
            native_key_name = _PI_PROVIDER_API_KEY_ENV.get(provider.casefold()) if provider else None
            if native_key_name is None:
                logger.warning(
                    "PI_API_KEY could not be mapped to a native credential "
                    "environment variable for provider %r and is being ignored. "
                    "Set the provider's native credential variable directly.",
                    provider,
                )
            else:
                child_env[native_key_name] = api_key

        # Pi's built-in system prompt is minimal; append the daydream preamble
        # so the default DeepSeek model gets the same tool-efficiency / budget-awareness
        # guidance that Claude Code and Codex inject natively via their CLIs.
        args.extend(["--append-system-prompt", _PI_SYSTEM_PREAMBLE])

        if read_only:
            args.extend(["--tools", _PI_READ_ONLY_TOOLS])

        resume_id: str | None = None
        if persist_session and continuation and continuation.backend == "pi":
            resume_id = continuation.data.get("session_id")
        effective_session_id: str | None = None
        if persist_session:
            effective_session_id = resume_id or str(uuid.uuid4())
            args.extend(["--session-id", effective_session_id])
        else:
            args.append("--no-session")

        args.append("--no-skills")

        full_prompt = prompt
        if output_schema:
            full_prompt = prompt + _schema_instruction(output_schema)

        args.append(full_prompt)

        # P18 Task 1: generation lifecycle correlation state (Pi only —
        # native_generation_interval class). One open generation per
        # assistant message; user/tool-result lifecycle never creates one.
        open_generation_id: str | None = None
        # Host receipt of assistant message_start (Unix ns, host clock).
        generation_start_ns: int | None = None

        session_id: str | None = None
        last_assistant_text: str | None = None
        structured_result: Any = None
        # Non-JSON lines (stderr merged into stdout, pi diagnostic output, etc.)
        # captured for error reporting when the process exits non-zero.
        stderr_lines: list[str] = []

        total_input: int | None = None
        total_output: int | None = None
        total_cache_read: int | None = None
        total_cache_write: int | None = None
        total_cost: float | None = None
        last_model = self.model
        last_provider = provider
        finish_reason: str | None = None
        saw_finish_reason = False
        saw_turn_start = False

        transport: CliTransport | None = None

        def terminal_events() -> tuple[CostEvent, ResultEvent]:
            native_session = session_id or effective_session_id
            return (
                CostEvent(
                    cost_usd=total_cost,
                    input_tokens=total_input,
                    output_tokens=total_output,
                    cached_tokens=total_cache_read,
                    cache_creation_tokens=total_cache_write,
                    model_name=last_model,
                    provider_name=last_provider,
                    measurement_source="terminal",
                    cost_source="reported" if total_cost is not None else None,
                ),
                ResultEvent(
                    structured_output=structured_result,
                    continuation=(
                        ContinuationToken(backend="pi", data={"session_id": native_session})
                        if persist_session and native_session and finish_reason != "error"
                        else None
                    ),
                    model_name=last_model,
                    provider_name=last_provider,
                    session_id=native_session,
                    finish_reason=finish_reason,
                ),
            )

        # P18 Task 1: closed typed effective-config admission from the exact
        # argv built above. max_turns is accepted-but-not-enforced by Pi (no
        # native flag) so it stays None; output_schema is emulated by prompt
        # appendix (schema_emulated=True whenever a schema was supplied).
        yield RequestEvent(
            prompt=full_prompt,
            system_prompt=_PI_SYSTEM_PREAMBLE,
            model_name=self.model,
            provider_name=provider,
            session_id=effective_session_id,
            reasoning_effort=thinking,
            output_schema=output_schema,
            config=PiRequestConfig(
                read_only=read_only,
                persist_session=persist_session,
                continuation_mode="resume" if resume_id is not None else "fresh",
                model_mode="single",
                selected_tools_count=len(_PI_READ_ONLY_TOOLS.split(",")) if read_only else None,
                selected_tools_present=read_only,
                no_skills=True,
                schema_emulated=output_schema is not None,
            ),
            model_source="configured",
            provider_source="configured" if provider is not None else None,
            session_source="host_generated" if resume_id is None else "configured",
        )

        try:
            transport = CliTransport(
                "pi",
                args,
                stdin_mode=StdinMode.DEVNULL,
                stderr_policy=StderrPolicy.MERGE_INTO_STDOUT,
                limit=_PI_STDOUT_LIMIT_BYTES,
                env=child_env,
                cwd=str(cwd),
            )
            self._transports.append(transport)
            await transport.start()

            if self._execution_input is not None:
                response_idle_timeout_s = (
                    self._execution_input.pi_response_idle_timeout_s
                )
                tool_idle_timeout_s = self._execution_input.stream_idle_timeout_s
            else:
                response_idle_timeout_s = stream_idle_timeout_s(
                    default=DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S
                )
                tool_idle_timeout_s = stream_idle_timeout_s(
                    default=DEFAULT_STREAM_IDLE_TIMEOUT_S
                )
            active_tool_calls = 0
            is_first_line = True
            async for raw_line in transport.lines(
                lambda: tool_idle_timeout_s if active_tool_calls > 0 else response_idle_timeout_s
            ):
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

                elif event_type == "turn_start":
                    saw_turn_start = True
                    saw_finish_reason = False

                elif event_type == "message_start":
                    msg = event.get("message") or {}
                    if msg.get("role") == "assistant" and open_generation_id is None:
                        # P18: host receipt of the assistant generation start.
                        # Host-observed only — never relabeled as provider
                        # request start; the native start comes from the
                        # completed message's Unix-ms timestamp at message_end.
                        open_generation_id = _new_generation_id()
                        generation_start_ns = time.time_ns()
                        assert generation_start_ns is not None  # just assigned
                        yield GenerationStartEvent(
                            generation_id=open_generation_id,
                            observed_at_unix_ns=generation_start_ns,
                            boundary_complete=True,
                        )

                elif event_type == "message_end":
                    msg = event.get("message") or {}
                    if msg.get("role") == "assistant":
                        last_model = msg.get("responseModel") or msg.get("model") or last_model
                        last_provider = msg.get("provider") or last_provider
                        text_parts: list[str] = []
                        choice_parts: list[Any] = []
                        for block in msg.get("content") or []:
                            if not isinstance(block, dict):
                                continue
                            btype = block.get("type")
                            if btype == "text":
                                text = block.get("text", "")
                                if text:
                                    yield TextEvent(text=text)
                                    text_parts.append(text)
                                    choice_parts.append(TextChoicePart(text=text))
                            elif btype == "thinking":
                                thinking_text = block.get("thinking", "")
                                if thinking_text:
                                    yield ThinkingEvent(text=thinking_text)
                                    choice_parts.append(ReasoningChoicePart(text=thinking_text))
                            elif btype == "toolCall":
                                # P18: the provider choice already carries the
                                # exact call ID/name/arguments at message_end;
                                # the later tool execution links by this call ID
                                # and never authors or duplicates the choice part.
                                call_id = block.get("id")
                                call_name = block.get("name")
                                if isinstance(call_id, str) and call_id and isinstance(call_name, str) and call_name:
                                    arguments_admitted, _arguments_diag = _admit_json_value(block.get("arguments"))
                                    if arguments_admitted is not None or block.get("arguments") is None:
                                        try:
                                            choice_parts.append(
                                                ToolCallChoicePart(
                                                    call_id=call_id,
                                                    name=call_name,
                                                    arguments=(
                                                        arguments_admitted if arguments_admitted is not None else {}
                                                    ),
                                                )
                                            )
                                        except ValueError:
                                            # Unsafe tool identity never enters the
                                            # provider choice (fixed admission policy).
                                            pass
                        if text_parts:
                            last_assistant_text = "".join(text_parts)
                        # P18: seal the generation exactly once at the matching
                        # assistant message_end, before any tool execution.
                        if open_generation_id is not None:
                            ended_at_ns = time.time_ns()
                            native_start_ms, _start_diag = _admit_native_unix_ms(msg.get("timestamp"))
                            # Chronology: a native start after the host end
                            # receipt is an explicit incomplete boundary, never
                            # clamped or reordered.
                            if (
                                native_start_ms is not None
                                and generation_start_ns is not None
                                and unix_ms_to_ns(native_start_ms) > generation_start_ns
                            ):
                                native_start_ms = None
                            yield GenerationEndEvent(
                                generation_id=open_generation_id,
                                native_started_at_unix_ms=native_start_ms,
                                ended_at_unix_ns=ended_at_ns,
                                end_source="host_observed_message_end",
                                choice_parts=tuple(choice_parts),
                                response_id=(
                                    msg.get("responseId")
                                    if isinstance(msg.get("responseId"), str) and msg.get("responseId")
                                    else None
                                ),
                                model_name=last_model,
                                provider_name=last_provider,
                                finish_reason=msg.get("stopReason") if msg.get("stopReason") is not None else None,
                                boundary_complete=True,
                            )
                            open_generation_id = None
                            generation_start_ns = None

                elif event_type == "tool_execution_start":
                    active_tool_calls += 1
                    yield ToolStartEvent(
                        id=event.get("toolCallId") or str(uuid.uuid4()),
                        name=event.get("toolName", "unknown"),
                        input=event.get("args") or {},
                    )

                elif event_type == "tool_execution_end":
                    active_tool_calls = max(0, active_tool_calls - 1)
                    yield ToolResultEvent(
                        id=event.get("toolCallId") or str(uuid.uuid4()),
                        output=_render_tool_result(event.get("result")),
                        is_error=bool(event.get("isError", False)),
                    )

                elif event_type == "turn_end":
                    active_tool_calls = 0
                    msg = event.get("message") or {}
                    stop_reason = msg.get("stopReason")
                    last_model = msg.get("responseModel") or msg.get("model") or last_model
                    last_provider = msg.get("provider") or last_provider
                    if stop_reason is not None:
                        saw_finish_reason = True
                        finish_reason = stop_reason
                    usage = _extract_usage(msg)
                    inp = usage["input"]
                    outp = usage["output"]
                    cached = usage["cacheRead"]
                    created = usage["cacheWrite"]
                    cost = usage["cost_total"]
                    if isinstance(inp, int):
                        inp += (cached if isinstance(cached, int) else 0) + (created if isinstance(created, int) else 0)
                        total_input = (total_input or 0) + inp
                    if isinstance(outp, int):
                        total_output = (total_output or 0) + outp
                    if isinstance(cached, int):
                        total_cache_read = (total_cache_read or 0) + cached
                    if isinstance(created, int):
                        total_cache_write = (total_cache_write or 0) + created
                    if isinstance(cost, (int, float)):
                        total_cost = (total_cost or 0.0) + cost
                    if isinstance(inp, int) and isinstance(outp, int):
                        yield MetricsEvent(
                            message_id="",  # Pi has no per-message id.
                            prompt_tokens=inp,
                            completion_tokens=outp,
                            cached_tokens=cached if isinstance(cached, int) else None,
                            cost_usd=cost if isinstance(cost, (int, float)) else None,
                            model_name=last_model,
                            provider_name=last_provider,
                            cache_creation_tokens=created if isinstance(created, int) else None,
                            measurement_source="turn_end",
                        )
                    # P18: per-turn identity where the Pi stream actually
                    # exposes it (stopReason on turn_end.message; response
                    # model/provider already tracked). Pi supplies no native
                    # message id, so message_id stays "" with configured/
                    # absent provenance; timing stays host-observed.
                    yield TurnEndEvent(
                        message_id="",
                        finish_reason=stop_reason if stop_reason is not None else None,
                        model_name=last_model,
                        provider_name=last_provider,
                        model_source="native",
                        provider_source="native" if last_provider is not None else None,
                    )
                    if stop_reason == "error":
                        for terminal in terminal_events():
                            yield terminal
                        error_msg = msg.get("errorMessage") or "Unknown Pi error"
                        raise PiError(
                            error_msg,
                            retryable=_is_retryable_error_message(error_msg),
                            category=_pi_error_category(error_msg),
                        )

                elif event_type == "agent_end":
                    # No inline finalization — Cost/Result are emitted once from
                    # the single post-loop path below. The loop keeps
                    # draining to EOF so the stdout pipe cannot fill mid-run.
                    pass

                # turn_start / message_start / message_update /
                # tool_execution_update are streaming-only; the full content is
                # already captured at message_end / tool_execution_end.

            # Reap the child (the transport raises TransportExitError on a
            # non-zero exit; the check below formats the backend-specific
            # message from the code and captured stderr lines).
            try:
                await transport.wait()
            except TransportExitError:
                pass

            if output_schema and last_assistant_text:
                structured_result = extract_json(last_assistant_text)
            for terminal in terminal_events():
                yield terminal

            # Fail fast on non-zero exit: if pi crashed without emitting a
            # turn_end error event, surface the failure with diagnostic output
            # instead of reporting a successful completion with empty/partial
            # output.
            returncode = transport.returncode
            if returncode is not None and returncode != 0:
                stderr_tail = "\n".join(stderr_lines[-10:])
                if stderr_lines:
                    detail = f"\nPi CLI output (last {len(stderr_lines)} non-JSON lines):\n{stderr_tail}"
                else:
                    detail = "\n(no non-JSON output captured — pi may have crashed before writing to stdout)"
                raise PiError(
                    f"Pi CLI exited with return code {returncode}.{detail}",
                    retryable=_is_retryable_exit_code(returncode),
                    category="PROCESS_EXIT",
                )

            if saw_turn_start and not saw_finish_reason:
                raise PiError(
                    "Stream ended without finish_reason",
                    retryable=True,
                    category="STREAM_TRUNCATION",
                )

        finally:
            if transport is not None:
                await transport.terminate()
                if transport in self._transports:
                    self._transports.remove(transport)

    async def cancel(self) -> None:
        """Cancel all running Pi processes.

        Sends SIGTERM, waits briefly, then SIGKILL if still running (mirrors
        ``CodexBackend.cancel``).
        """
        await CliTransport.cancel_all(self._transports)
