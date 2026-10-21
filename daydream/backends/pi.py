# daydream/backends/pi.py
"""Translate pi --mode json (@earendil-works/pi-coding-agent) into AgentEvent.

Ordinary calls respect Pi's project/global defaultModel. Without one, use DEFAULT_PI_MODEL
with the nous provider; explicit model and PI_PROVIDER/API_KEY/THINKING
remain overrides. Provider endpoints and credentials belong to Pi's registry
and auth files under ~/.pi/agent; Daydream never writes a models.json override.
Bound reviews let the actual Pi CLI resolve native defaults in the original
configuration context while its stock read/search tools use captured source.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, ExitStack
from dataclasses import dataclass
from functools import partial
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Literal

import anyio

from daydream import clock, host_tools as host_policy
from daydream.backends import (
    AgentEvent,
    BackendExecutionInput,
    ContinuationToken,
    CostEvent,
    DiagnosticEvent,
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
    _parsed_nonnegative_float,
    _parsed_nonnegative_int,
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
    process_exit_message,
    raise_for_exit,
    reap,
    teardown,
)
from daydream.backends.lifecycle import current_backend_lifecycle
from daydream.config import DEFAULT_PI_MODEL, DEFAULT_TOOL_CALL_BUDGET, DEFAULT_WALL_BUDGET_S
from daydream.host_tools import HostToolBinding
from daydream.redaction import _PEM_HEADER, _PEM_KEY_REDACTED_MARKER, redact_structured_text
from daydream.retry_policy import classify_failure, parse_message_retry_hint

# Mirror Codex's generous stdout cap so large JSONL events (big file reads,
# patch payloads) do not trip asyncio's "chunk is longer than limit" guard.
_PI_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024
_PI_FINAL_PAYLOAD_MAX_BYTES = 128 * 1024
_PI_ERROR_MESSAGE_MAX_CHARS = 4096
_PI_RAW_STOP_REASON_MAX_CHARS = 256
_PI_RESPONSE_HEADERS = frozenset({"x-request-id", "request-id", "x-amzn-requestid", "x-amz-request-id",
                                  "cf-ray", "retry-after", "retry-after-ms"})
_PI_PEM_START = re.compile(rf"-----BEGIN (?P<kind>{_PEM_HEADER})-----")


def _pi_diagnostic_text(value: str, limit: int) -> tuple[str, bool]:
    """Redact before bounding; diagnostic failures cannot replace the native error."""
    try:
        redacted = redact_structured_text(value)
        if not isinstance(redacted, str):
            raise TypeError("diagnostic redactor returned non-text")
    except Exception:  # noqa: BLE001 - fixed marker only, never raw fallback
        redacted = "[DIAGNOSTIC_REDACTION_FAILED]"
    return redacted[:limit], len(redacted) > limit


def _pi_response_metadata(event: dict[str, Any]) -> dict[str, Any] | None:
    """Independently admit the helper's closed response fields before recording."""
    status = event.get("status")
    if type(status) is not int or not 100 <= status <= 599:
        return None
    headers = event.get("headers")
    selected: dict[str, str] = {}
    truncated = False
    if isinstance(headers, dict):
        for name in _PI_RESPONSE_HEADERS:
            value = headers.get(name)
            if isinstance(value, str):
                selected[name], clipped = _pi_diagnostic_text(value, 512)
                truncated |= clipped
    return {"status": status, "headers": selected, **({"headers_truncated": True} if truncated else {})}

# Known AgentSessionEvent types. Used to decide whether the first
# stdout line — the session header — also carries a dispatchable event type.
_PI_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "agent_start",
        "agent_end",
        "agent_settled",
        "daydream_host_tools_ready",
        "daydream_host_tools_failed",
        "daydream_host_tool_fatal",
        "daydream_provider_response",
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


def _unsupported_hosted_policy(message: str) -> ValueError:
    from daydream.hosted_review import UnsupportedHostedPolicy

    return UnsupportedHostedPolicy(message)

_PI_PROVIDER_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "ZAI_API_KEY",
    "nous": "NOUS_API_KEY",
}


def _pi_process_exit_message(stderr_lines: list[str], returncode: int) -> str:
    """Bind pi's captured stderr into the shared PROCESS_EXIT message."""
    return process_exit_message(display="Pi", returncode=returncode, lines=stderr_lines)


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
    """Resolve defaultModel with project settings taking precedence over global settings."""
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


@dataclass(frozen=True)
class _PiReviewContext:
    source: Path
    agent_dir: Path
    provider: str | None
    thinking: str | None
    inputs: tuple[tuple[Path, str | None], ...]


def _selection_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def native_pi_settings(cwd: Path, *, agent_dir: Path | None) -> dict[str, str]:
    """Project only nonsecret model defaults with Pi's project/global precedence."""
    result: dict[str, str] = {}
    paths = ([] if agent_dir is None else [agent_dir / "settings.json"]) + [cwd / ".pi" / "settings.json"]
    for path in paths:
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(settings, dict):
            for key in ("defaultModel", "defaultProvider", "defaultThinkingLevel"):
                value = settings.get(key)
                if isinstance(value, str) and value.strip():
                    result[key] = value.strip()
    return result


# Pi's minimal built-in prompt needs explicit exploration/tool guidance.
# Append this preamble each turn; keep it concise to limit repeated context.
# The allowance sentence states what this invocation actually has: a tool-call
# ceiling the host does not impose must never be described as strict, because
# a cap that does not exist is not a discipline, it is a lie about the budget.
_PI_PREAMBLE_TEMPLATE = """\
You are an efficient coding agent.
{allowance}
Each call must resolve a specific unanswered question; the allowance is a ceiling,
not a target.

WORK STRATEGY:
- Use supplied exact file paths and diff context directly. When a location is
  unknown, use targeted grep/find/ls to locate it before opening files.
- Batch related reads. Don't read files one at a time in a loop when a single
  grep would surface every relevant location.
{starting_method}
- Don't re-read what you've already read. If a file's content is already in
  your context, reuse it for understanding. For staged reviews, diff or prompt
  contents are enough when they support a decision; use ordinary file tools or
  supplied captured context when another source is useful.
- Answer directly when you can. If the existing context (commit log, diff,
  prior tool results) already answers the question, respond without additional
  tool calls.
- Stop exploring once you know enough. The goal is understanding and reporting,
  not exhaustive codebase enumeration. When you have enough, produce your
  answer immediately.

GIT CONTEXT:
You are operating in a git repository. When a Git execution tool is enabled,
use `git diff`, `git log`, and `git show` to understand changes efficiently.
Read-only Pi has no Git execution tool. Captured before-side context for deleted
or renamed files is supplied as an ordinary file when available; never infer
missing old-source contents.

Be concise in your responses. Do not narrate exploration step by step; report
findings and conclusions."""


def render_pi_preamble(wall_budget_s: float | None, tool_call_budget: int | None, *, staged: bool = False) -> str:
    """Render the preamble with the invocation's actual allowances.

    Requirement 17: the numbers are rendered, never hardcoded, and the tool-call
    sentence says exactly what is true. ``DEFAULT_TOOL_CALL_BUDGET`` is ``None``,
    so the common case has no ceiling to honour — an uncapped turn is described as
    uncapped rather than as a strict budget, and the wall allowance the host did
    impose is stated instead.
    """
    wall = f"{float(wall_budget_s):g} seconds" if wall_budget_s is not None else "unbounded"
    if tool_call_budget is None:
        allowance = (
            f"Attempt dispatch-time snapshot: effective wall allowance is {wall}; tool-call allocation is uncapped. "
            "Every call must earn the time it spends."
        )
    else:
        allowance = (
            f"Attempt dispatch-time snapshot: effective wall allowance is {wall}; "
            f"tool-call allocation is {int(tool_call_budget)} calls. "
            "Each call counts, including failed calls, parallel batch members and structured submissions. "
            "Reserve one call for structured_output when native submission is required."
        )
    return _PI_PREAMBLE_TEMPLATE.format(allowance=allowance, starting_method=(
        '- Follow the active stage method and persistent exact access guide when choosing where to start.'
        if staged else '- Read the diff first. If a diff file or git output is in your context, start\n'
        '  there; only explore files referenced by the diff or their direct imports.'))


_PI_SYSTEM_PREAMBLE = render_pi_preamble(DEFAULT_WALL_BUDGET_S, DEFAULT_TOOL_CALL_BUDGET)


def _pi_submission_assignment(review_instructions: str | None) -> dict[str, Any] | None:
    """Project only the stage builder's explicit assignment, never conversation history.

    Oversized or malformed metadata is omitted whole: full original assignments
    remain in the system instructions, and identities are never truncated.
    """
    prefix = "Daydream final-submission assignment: "
    lines = [line[len(prefix):] for line in (review_instructions or "").splitlines() if line.startswith(prefix)]
    if len(lines) != 1 or len(lines[0].encode("utf-8")) > 4096:
        return None
    try:
        value = json.loads(lines[0])
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict) or set(value) != {"objective", "target_ids", "candidate_ids"}:
        return None
    objective = value["objective"]
    if not isinstance(objective, str) or not objective or len(objective) > 256:
        return None
    for field in ("target_ids", "candidate_ids"):
        identities = value[field]
        if (not isinstance(identities, list) or len(identities) > 128
                or any(not isinstance(identity, str) or not identity or len(identity) > 256
                       for identity in identities)):
            return None
    return value


def pi_system_preamble(
    wall_budget_s: float | None = DEFAULT_WALL_BUDGET_S, tool_call_budget: int | None = None, *, staged: bool = False
) -> str:
    """Render this invocation's preamble, defaulting to the module allowances.

    A turn granted a ceiling other than ``DEFAULT_WALL_BUDGET_S`` must not be told
    about the default: the rendered allowance has to be the one the host actually
    imposes, or the model plans against time it does not have. Omitted wall
    allowance retains the adapter's standalone default; explicit ``None`` means
    the caller imposes no wall deadline. Tool-call ``None`` is likewise uncapped.
    """
    wall = wall_budget_s
    calls = DEFAULT_TOOL_CALL_BUDGET if tool_call_budget is None else tool_call_budget
    if not staged and wall == DEFAULT_WALL_BUDGET_S and calls == DEFAULT_TOOL_CALL_BUDGET:
        return _PI_SYSTEM_PREAMBLE
    return render_pi_preamble(wall, calls, staged=staged)


_PI_FINALIZATION_PREAMBLE = """\
Serialize the completed task using only the supplied context and completed evidence.
Return exactly the requested output format. Tools are disabled. Do not investigate,
research, test, or infer missing evidence. An empty findings result is successful
when no defect is substantiated; preserve truthful incomplete coverage.
"""


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
    return _parsed_nonnegative_int(
        os.environ, "DAYDREAM_PI_RETRY_ATTEMPTS", _PI_DEFAULT_RETRY_ATTEMPTS
    )


def _pi_retry_base_delay() -> float:
    return _parsed_nonnegative_float(
        os.environ, "DAYDREAM_PI_RETRY_BASE_DELAY_S", _PI_DEFAULT_RETRY_BASE_DELAY
    )


def _pi_retry_max_delay() -> float:
    return _parsed_nonnegative_float(
        os.environ, "DAYDREAM_PI_RETRY_MAX_DELAY_S", _PI_DEFAULT_RETRY_MAX_DELAY
    )


# Permanent conditions precede transient matches. Bare "provider" is too
# broad ("provider rate limit"); credential/configuration tokens still match.
_PERMANENT_TOKENS = (
    "auth",
    "credential",
    "api key",
    "api_key",
    "configuration",
    "not configured",
    "model not found",
)
_SCHEMA_TOKENS = ("schema validation", "additionalproperties", "structured output")
_RATE_LIMIT_TOKENS = ("429", "rate limit", "rate_limit", "too many requests")
_SERVER_ERROR_TOKENS = (
    "502",
    "503",
    "bad gateway",
    "service unavailable",
    "server error",
)
# Ambiguous overload/throttle wording. Unlike the high-precision literals above,
# these need positive overload/capacity meaning and an explicit rejection of
# negated or planning contexts: "not overloaded" and "capacity planning" are
# healthy messages that must not be read as transient failures.
_NEGATED_OVERLOAD_RE = re.compile(r"\bnot\s+overloaded\b|\bcapacity\s+planning\b")
_OVERLOAD_RE = re.compile(r"\boverloaded?\b|\boverload(?:ed|ing)?\b")
_CAPACITY_RE = re.compile(
    r"\bcapacity\s+(?:unavailable|exceeded|limit|limited|full|reached)\b"
    # Require an affirmative sentence, not a negation, question or conditional.
    r"|(?:^|[.:!?])\s*(?:the\s+)?(?:model|service|provider)\s+is\s+temporarily\s+at\s+capacity(?=[.!]|$)"
)
_THROTTLE_RE = re.compile(r"\bthrottl(?:e|ed|ing)\b")


def _is_transient_overload_message(lower: str) -> bool:
    """Return True for an overload/throttle/capacity message that a retry may help."""
    if _NEGATED_OVERLOAD_RE.search(lower):
        return False
    return bool(
        _OVERLOAD_RE.search(lower)
        or _CAPACITY_RE.search(lower)
        or _THROTTLE_RE.search(lower)
    )


def _is_stream_truncation_message(normalized_message: str) -> bool:
    """Return True when a normalized message reports an unfinished stream."""
    return "finish_reason" in normalized_message and any(
        token in normalized_message for token in ("stream", "ended", "end")
    )


def _is_retryable_exit_code(code: int | None) -> bool:
    """Return True for exit codes that indicate OOM/SIGKILL rather than a logic error."""
    return code in (-9, 137)


def _pi_error_category(message: str) -> str:
    """Classify failures with permanent conditions taking precedence over transient tokens.

    For example, model not found with HTTP 503 remains AUTH_CONFIG.
    """
    lower = message.casefold()
    if any(token in lower for token in _PERMANENT_TOKENS):
        return "AUTH_CONFIG"
    if any(token in lower for token in _SCHEMA_TOKENS):
        return "SCHEMA"
    if any(token in lower for token in _RATE_LIMIT_TOKENS):
        return "RATE_LIMIT"
    if any(token in lower for token in _SERVER_ERROR_TOKENS):
        return "SERVER_ERROR"
    if _is_transient_overload_message(lower):
        return "SERVER_ERROR"
    if any(token in lower for token in ("timed out", "timeout", "deadline exceeded")):
        return "TIMEOUT"
    if _is_stream_truncation_message(lower):
        return "STREAM_TRUNCATION"
    if any(signature in lower for signature in STREAM_DROP_SIGNATURES):
        return "STREAM_DROP"
    if "pi cli exited with return code" in lower:
        return "PROCESS_EXIT"
    return "UNKNOWN"


class _PiFailureFacts(Exception):
    """Classifier probe carrying Pi's category + message and no opt-in flag."""

    def __init__(self, message: str, category: str) -> None:
        super().__init__(message)
        self.category = category


def _pi_retryable_for(*, category: str, message: str) -> bool:
    """Use run_agent's shared failure classifier to keep retryability consistent."""
    return classify_failure(_PiFailureFacts(message, category)).retries_allowed


class PiError(Exception):
    """Raised when a Pi turn fails (e.g. ``stopReason == "error"``)."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        category: str = "UNKNOWN",
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.category = category
        self.retry_after = retry_after


def _render_tool_result(result: Any) -> str:
    """Flatten plain text; preserve details and mixed/image/resource blocks as JSON."""
    if not isinstance(result, dict):
        return result if isinstance(result, str) else ("" if result is None else json.dumps(result))
    content = result.get("content")
    if result.get("details") not in (None, "") or (
        isinstance(content, list)
        and any(not isinstance(block, dict) or block.get("type") != "text" for block in content)
    ):
        return json.dumps(result, ensure_ascii=False)
    if isinstance(content, list):
        joined = "".join(block.get("text", "") for block in content)
        if joined:
            return joined
    if isinstance(content, str) and content:
        return content
    # Last resort — preserve the payload rather than dropping the observation.
    return json.dumps(result, ensure_ascii=False) if result else ""


def _tool_completion_metadata(event: dict[str, Any], *, output_control: bool = False) -> dict[str, Any]:
    """Retain explicit native completion fields; rendered notices are not authority."""
    result = event.get('result')
    result = result if isinstance(result, dict) else {}
    details = None if output_control else result.get('details')
    details = details if isinstance(details, dict) else {}
    truncation = details.get('truncation')
    truncation = truncation if isinstance(truncation, dict) else {}
    structured = result.get('structuredContent')
    structured = structured if isinstance(structured, dict) else {}
    exit_code = structured.get('exit_code', result.get('exit_code', event.get('exit_code')))
    status = event.get('status', result.get('status'))
    return {'exit_code': exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None,
        'status': status if isinstance(status, str) else None,
        'cancelled': event.get('cancelled') is True or result.get('cancelled') is True,
        'truncated': (truncation.get('truncated') is True or truncation.get('firstLineExceedsLimit') is True
                      or structured.get('truncated') is True or result.get('truncated') is True)}


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
    """Append the schema for excluded modes that retain prose serialization."""
    return (
        "\n\nRespond with ONLY a single valid JSON object matching this JSON "
        "schema. Do not include any prose, explanations, or markdown fences "
        "outside the JSON.\n" + json.dumps(schema)
    )


def _stage_supplies_schema(prompt: str, schema: dict[str, Any]) -> bool:
    """A matching host stage contract is already the authoritative schema text."""
    marker = 'Host review stage:\n'
    start = prompt.find(marker)
    if start < 0:
        return False
    try:
        stage, _ = json.JSONDecoder().raw_decode(prompt[start + len(marker):].lstrip())
    except (ValueError, TypeError):
        return False
    contract = stage.get('response_contract') if isinstance(stage, dict) else None
    return isinstance(contract, dict) and contract.get('schema') == schema


def _write_prompt_attachment(text: str, attachments: ExitStack | AsyncExitStack) -> Path:
    """Create a closed private UTF-8 file owned by the invocation's cleanup scope."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="daydream-pi-prompt-", suffix=".txt", delete=False,
    ) as prompt_file:
        path = Path(prompt_file.name).absolute()
        attachments.callback(path.unlink, missing_ok=True)
        prompt_file.write(text)
    return path


class PiBackend:
    """Translate the Pi JSONL event stream into normalized AgentEvent records."""

    supports_finalization = True
    supports_tools_disabled = True
    supports_host_tools = True
    supports_review_instructions = True
    supports_budget_preamble = True
    # Validation opt-out selects text transport for downstream partial-object validation.
    supports_structured_output_opt_out = True
    concise_fix_prompts = True  # DeepSeek produces verbose reasoning in fix prompts

    def __init__(
        self,
        model: str | None = None,
        *,
        cwd: Path | None = None,
        reasoning_effort: str | None = None,
        execution_input: BackendExecutionInput | None = None,
    ):
        """Resolve the model before recording; explicit reasoning_effort overrides PI_THINKING."""
        self._model_override = model
        self.reasoning_effort = reasoning_effort
        self._execution_input = execution_input
        self._review_context: _PiReviewContext | None = None
        # ``.model`` must be resolved at construction (runner/recorder read it
        # before execute); cache the settings lookup so execute() need not
        # re-read settings.json for the same workspace.
        self._configured_cache: tuple[Path, str | None] | None = None
        configured: str | None = None
        if model is None and cwd is not None:
            configured = (execution_input.pi_settings.get("defaultModel")
                if execution_input is not None and execution_input.pi_settings is not None else _configured_pi_model(
                (execution_input.pi_settings_root or cwd) if execution_input is not None else cwd,
                agent_dir=(
                    execution_input.pi_agent_dir
                    if execution_input is not None
                    else _AMBIENT_AGENT_DIR
                ),
            ))
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

    @property
    def _hosted_read_only(self) -> bool:
        return self._execution_input is not None and self._execution_input.pi_hosted_read_only

    def _check_hosted_resources(self, agent_dir: Path) -> None:
        """Reject package resolution inputs before Pi's resource loader can run them.

        Pi resolves configured packages before applying --no-extensions. An
        isolated, resource-free operator directory is therefore required in
        addition to the CLI's native resource and forced-untrusted switches.
        """
        try:
            for name in ("settings.json", "models.json", "auth.json"):
                path = agent_dir / name
                if path.is_symlink() or path.exists() and not stat.S_ISREG(path.lstat().st_mode):
                    raise _unsupported_hosted_policy(
                        "Hosted Pi requires regular operator-owned configuration/authentication files; "
                        f"{name} cannot be a link or special file")
            settings_path = agent_dir / "settings.json"
            settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
            if not isinstance(settings, dict) or any(settings.get(name) for name in (
                "packages", "extensions", "skills", "prompts", "themes",
            )):
                raise _unsupported_hosted_policy(
                    "Hosted Pi requires operator settings without packages, extensions, skills, prompts or themes")
            for name in ("extensions", "skills", "prompts", "themes", ".pi", ".agents",
                         "SYSTEM.md", "APPEND_SYSTEM.md"):
                path = agent_dir / name
                if path.exists() or path.is_symlink():
                    raise _unsupported_hosted_policy(
                        f"Hosted Pi requires an isolated resource-free operator directory; remove {name}")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _unsupported_hosted_policy("Hosted Pi operator settings could not be validated") from exc

    async def validate_hosted_policy(self) -> None:
        """Verify the installed supported native baseline before review work begins."""
        if not self._hosted_read_only or self._review_context is None:
            raise _unsupported_hosted_policy("Hosted Pi preflight requires an operator-bound review context")
        self._check_review_context()
        assert self._execution_input is not None
        binding = HostToolBinding(deadline=clock.monotonic() + host_policy._SETUP_TIMEOUT_S,
                                  accept=lambda _: {})
        await self._require_host_tools_version(self._review_context.source,
                                               self._execution_input.child_environment(), binding)

    def bind_review_context(self, original_source: Path) -> dict[str, object]:
        """Bind native configuration context without resolving its model selection.

        Pi starts in the original project; stock tools and context files bind to
        execute's captured source. Configuration drift requires a new run, while
        native authentication remains live. The returned identity is opaque.
        """
        if self._review_context is not None:
            self._check_review_context()
            return self._context_identity(self._review_context)
        environment = self._execution_input.child_environment() if self._execution_input else os.environ.copy()
        agent_dir = (self._execution_input.pi_agent_dir if self._execution_input else None) or Path(
            environment.get("PI_CODING_AGENT_DIR", str(Path(environment.get("HOME", str(Path.home()))) / ".pi/agent"))
        )
        original_source = original_source.resolve(strict=True)
        if not agent_dir.is_absolute():
            agent_dir = original_source / agent_dir
        if self._hosted_read_only:
            assert self._execution_input is not None
            if (self._execution_input.pi_agent_dir is None or not self._execution_input.pi_agent_dir.is_absolute()
                    or any(not isinstance(value, str) or not value.strip()
                           for value in (self._execution_input.pi_provider, self._model_override))):
                raise _unsupported_hosted_policy(
                    "Hosted Pi requires an explicit isolated agent directory and nonblank provider/model strings")
            agent_dir = agent_dir.resolve(strict=True)
            if agent_dir.is_relative_to(original_source) or original_source.is_relative_to(agent_dir):
                raise _unsupported_hosted_policy("Hosted Pi operator configuration and source must be disjoint")
            if any(environment.get(name) for name in (
                "NODE_OPTIONS", "NODE_PATH", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "BASH_ENV", "ENV",
            )):
                raise _unsupported_hosted_policy("Hosted Pi environment must not contain executable loader overrides")
            self._check_hosted_resources(agent_dir)
            # Native settings/resource discovery never sees the reviewed project.
            original_source = agent_dir
            inputs = tuple(agent_dir / name for name in ("settings.json", "models.json"))
            provider: str | None = self._execution_input.pi_provider
            thinking = self.reasoning_effort or self._execution_input.pi_thinking
        else:
            inputs = (original_source / ".pi/settings.json", *(agent_dir / name
                      for name in ("settings.json", "models.json", "trust.json")))
            provider = environment.get("PI_PROVIDER") or None
            thinking = self.reasoning_effort or environment.get("PI_THINKING")
        self._review_context = _PiReviewContext(
            original_source, agent_dir, provider, thinking,
            tuple((path, _selection_digest(path)) for path in inputs),
        )
        self.model = self._model_override or "pi-native"
        return self._context_identity(self._review_context)

    @staticmethod
    def _context_identity(context: _PiReviewContext) -> dict[str, object]:
        identity = json.dumps([(str(path), digest) for path, digest in context.inputs]).encode()
        return {"configuration_identity": hashlib.sha256(identity).hexdigest()}

    def _check_review_context(self) -> None:
        context = self._review_context
        if context is not None and self._hosted_read_only:
            self._check_hosted_resources(context.agent_dir)
        if context is not None and any(_selection_digest(path) != digest for path, digest in context.inputs):
            if self._hosted_read_only:
                raise _unsupported_hosted_policy("Hosted Pi operator configuration changed after review binding")
            raise PiError("Pi configuration changed after review binding; start a new run", category="AUTH_CONFIG")

    async def _require_host_tools_version(self, cwd: Path, environment: dict[str, str],
                                         binding: HostToolBinding) -> None:
        """Pin the fail-closed extension loader used by the readiness barrier."""
        if self._hosted_read_only:
            await self._require_native_version("node", "v24.18.1", cwd, environment, binding)
        await self._require_native_version("pi", "1.4.0", cwd, environment, binding)

    async def _require_native_version(self, command: str, expected: str, cwd: Path,
                                      environment: dict[str, str], binding: HostToolBinding) -> None:
        probe = CliTransport(command, [command, "--version"], limit=4096, env=environment, cwd=str(cwd),
                             stderr_policy=StderrPolicy.DRAIN_TASK)
        self._transports.append(probe)
        probe_failed = False
        try:
            allowance = min(host_policy._SETUP_TIMEOUT_S, binding.deadline - clock.monotonic())
            async with asyncio.timeout(max(0.0, allowance)):
                await binding.start(probe.start(), lambda _: teardown(
                    probe, self._transports, cleanup_timeout_s=host_policy._CLEANUP_TIMEOUT_S))
                lines = []
                async for line in probe.lines(lambda: allowance):
                    lines.append(line)
                    if len(lines) > 1:
                        break
                code = await reap(probe, timeout_s=allowance)
        except (TimeoutError, OSError, ValueError):
            probe_failed = True
            if self._hosted_read_only:
                raise _unsupported_hosted_policy(
                    f"Could not verify hosted read-only native baseline: {command} {expected}; "
                    "install official Pi 1.4.0 and Node 24.18.1") from None
            raise PiError("Could not verify required Pi 1.4.0 host-tool capabilities", category="HOST_TOOLS") from None
        except BaseException:
            probe_failed = True
            raise
        else:
            if code != 0 or lines != [expected]:
                probe_failed = True
                if self._hosted_read_only:
                    raise _unsupported_hosted_policy(
                        f"Hosted read-only policy requires verified official Pi 1.4.0 / Node 24.18.1; "
                        f"{command} did not report {expected}")
                raise PiError("Native host tools require verified Pi 1.4.0", category="HOST_TOOLS")
        finally:
            try:
                await teardown(probe, self._transports, cleanup_timeout_s=host_policy._CLEANUP_TIMEOUT_S)
            except BaseException:
                binding.mark_cleanup_failed()
                if not probe_failed:
                    raise

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
        finalization: bool = False,
        review_instructions: str | None = None,
        tools_disabled: bool = False,
        validate_structured_output: bool = True,
        wall_budget_s: float | None = DEFAULT_WALL_BUDGET_S,
        tool_call_budget: int | None = None,
        host_tools: HostToolBinding | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield Pi events; a turn error raises PiError and nonempty agents are unsupported.

        Tools-enabled schema calls use the exact-schema output tool; excluded modes retain assistant-text serialization.
        Pi tokens resume via --session-id when persistence is enabled;
        persist_session=False uses --no-session and suppresses continuation.

        read_only allows read/find/ls/grep. finalization disables tools, substitutes
        serialization instructions, and caps thinking at low while preserving lower
        settings. tools_disabled keeps normal instructions/thinking and uses stdin.
        Pi cannot enforce max_turns: callers must enforce an absolute deadline.
        wall_budget_s/tool_call_budget state the allowances the caller is actually
        enforcing, so the preamble describes this turn rather than the defaults.
        Stdout silence raises retryable StreamStalledError; run_agent starts a fresh
        subprocess for each retry.
        """
        if self._execution_input is not None:
            response_idle_timeout_s = self._execution_input.pi_response_idle_timeout_s
            tool_idle_timeout_s = self._execution_input.stream_idle_timeout_s
        else:
            response_idle_timeout_s = stream_idle_timeout_s(default=DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S)
            tool_idle_timeout_s = stream_idle_timeout_s(default=DEFAULT_STREAM_IDLE_TIMEOUT_S)
        lifecycle = current_backend_lifecycle()
        if lifecycle is not None:
            lifecycle.activate(response_idle_timeout_s=response_idle_timeout_s, tool_idle_timeout_s=tool_idle_timeout_s)
        if agents:
            raise NotImplementedError(
                "Pi backend does not support exploration subagents; use --backend claude for exploration."
            )

        if self._hosted_read_only:
            from daydream.hosted_review import HOSTED_PI_TOOLS

            if (not read_only or persist_session or finalization or tools_disabled or host_tools is None
                    or output_schema is None or not validate_structured_output or continuation is not None
                    or self._review_context is None):
                raise _unsupported_hosted_policy(
                    "Hosted Pi requires a fresh, nonpersistent read-only invocation with bound native host submission")
            host_tools.restrict(deadline=None, tool_call_budget=None, allowed_tools=HOSTED_PI_TOOLS)
            if host_tools.allowed_tools != HOSTED_PI_TOOLS:
                raise _unsupported_hosted_policy("Hosted Pi requires the exact operator read/search/submission grant")

        args: list[str] = ["pi", "--mode", "json"]
        if self._hosted_read_only:
            args.extend(["--no-approve", "--no-mcp", "--no-prompt-templates", "--no-themes"])
        native_output = (output_schema is not None and validate_structured_output
                         and not (finalization or tools_disabled))
        if host_tools is not None:
            if not native_output:
                raise PiError("Host admission requires a tools-enabled native terminal schema",
                              category="HOST_TOOLS", retryable=False)
            assert output_schema is not None
            host_tools.configure_schema(output_schema)
            # run_agent passes the residual allocation on retries; the binding
            # stores an original cumulative limit, so account for earlier starts.
            host_tools.restrict(deadline=(clock.monotonic() + wall_budget_s) if wall_budget_s is not None else None,
                                tool_call_budget=(host_tools.calls + tool_call_budget)
                                if tool_call_budget is not None else None)

        review_context = self._review_context
        self._check_review_context()
        process_cwd = review_context.source if review_context is not None else cwd
        configured_model = None
        if self._model_override is None and review_context is None:
            if self._configured_cache is not None and self._configured_cache[0] == cwd:
                configured_model = self._configured_cache[1]
            else:
                configured_model = (self._execution_input.pi_settings.get("defaultModel")
                    if self._execution_input is not None and self._execution_input.pi_settings is not None
                    else _configured_pi_model(
                    (self._execution_input.pi_settings_root or cwd) if self._execution_input is not None else cwd,
                    agent_dir=(
                        self._execution_input.pi_agent_dir
                        if self._execution_input is not None
                        else _AMBIENT_AGENT_DIR
                    ),
                ))
        self.model = self._model_override or ("pi-native" if review_context else configured_model or DEFAULT_PI_MODEL)
        settings_root = (None if self._execution_input is None or review_context is not None
                         else self._execution_input.pi_settings_root)
        frozen_settings = ({} if self._execution_input is None else self._execution_input.pi_settings or {})
        if self._model_override is not None or (review_context is None
                and (configured_model is None or settings_root is not None)):
            args.extend(["--model", self.model])
        provider = (
            self._execution_input.pi_provider
            if self._execution_input is not None
            else os.environ.get("PI_PROVIDER")
        )
        if review_context is not None:
            provider = review_context.provider
        if provider is None and settings_root is not None:
            provider = frozen_settings.get("defaultProvider")
            if (provider is None and configured_model is not None
                    and native_pi_settings(cwd, agent_dir=None).get("defaultProvider") is not None):
                raise PiError("Case Pi provider conflicts with unavailable operator provider", category="AUTH_CONFIG")
        if review_context is None and provider is None and configured_model is None:
            provider = _PI_DEFAULT_PROVIDER
            if self._model_override is not None and self.model.casefold().startswith("glm-"):
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
        elif (
            review_context is None and self._model_override is None and configured_model is None
            and provider and provider != _PI_DEFAULT_PROVIDER
        ):
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
        if review_context is not None:
            thinking = review_context.thinking
            child_env["PI_CODING_AGENT_DIR"] = str(review_context.agent_dir)
            args.append("--no-context-files")
        if thinking is None and settings_root is not None:
            thinking = frozen_settings.get("defaultThinkingLevel")
            if thinking is None and native_pi_settings(cwd, agent_dir=None).get("defaultThinkingLevel") is not None:
                raise PiError("Case Pi thinking conflicts with unavailable operator effort", category="AUTH_CONFIG")
        if finalization:
            thinking = thinking if thinking in {"off", "minimal", "low"} else "low"
        if provider:
            args.extend(["--provider", provider])
        if thinking:
            args.extend(["--thinking", thinking])

        child_env.pop("PI_API_KEY", None)
        if api_key:
            api_key_provider = (provider or _PI_DEFAULT_PROVIDER) if review_context else provider
            native_key_name = _PI_PROVIDER_API_KEY_ENV.get(api_key_provider.casefold()) if api_key_provider else None
            if native_key_name is None:
                logger.warning(
                    "PI_API_KEY could not be mapped to a native credential "
                    "environment variable for provider %r and is being ignored. "
                    "Set the provider's native credential variable directly.",
                    api_key_provider,
                )
            else:
                child_env[native_key_name] = api_key

        # Pi's built-in system prompt is minimal; append the daydream preamble
        # so the default DeepSeek model gets the same tool-efficiency / budget-awareness
        # guidance that Claude Code and Codex inject natively via their CLIs.
        system_prompt = (
            _PI_FINALIZATION_PREAMBLE
            if finalization
            else pi_system_preamble(wall_budget_s, tool_call_budget, staged=bool(review_instructions
                                                and "Staged review contract:" in review_instructions))
        )
        if review_instructions and not finalization:
            system_prompt += (
                "\n\nBOUNDED REPOSITORY REVIEW:\n" + review_instructions
                + "\nKeep searches repository-scoped to the working directory and assigned "
                "files or their directly relevant dependencies. Do not search filesystem "
                "roots, host caches, or unrelated repositories. Use explicitly supplied "
                "review artifacts at their exact paths. Once a candidate is resolved, do not reopen it without new "
                "contradictory evidence. After covering the assigned changes and resolving "
                "concrete candidates, emit the requested result immediately; an empty "
                "findings result is valid. Do not spend remaining time reconsidering "
                "closed candidates or searching for a reason to avoid an empty result."
            )
        args.extend([
            "--system-prompt" if finalization else "--append-system-prompt",
            system_prompt,
        ])

        if finalization or tools_disabled:
            args.append("--no-tools")
        elif read_only:
            args.extend(["--tools", _PI_READ_ONLY_TOOLS + (",structured_output" if native_output else "")])

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
        if native_output:
            full_prompt += "\n\nSubmit the final result using structured_output alone as your final action."
        elif output_schema and not _stage_supplies_schema(prompt, output_schema):
            full_prompt = prompt + _schema_instruction(output_schema)

        # P18 Task 1: generation lifecycle correlation state (Pi only —
        # native_generation_interval class). One open generation per
        # assistant message; user/tool-result lifecycle never creates one.
        open_generation_id: str | None = None
        last_generation_id: str | None = None
        # Host receipt of assistant message_start (Unix ns, host clock).
        generation_start_ns: int | None = None

        session_id: str | None = None
        structured_result: Any = None
        # Finalized toolResult sets transcript order; track association/settlement, never proposal/completion order.
        submission_calls: dict[str, str] = {}
        incomplete_calls: set[str] = set()
        pending_submission: Any = None
        pending_submission_generation: str | None = None
        pending_usage: dict[str, Any] | None = None
        # Pi routes extension protocol events through stderr, so preserve the
        # merged stream's ordering. Only non-JSON text enters this bounded tail.
        stderr_lines: deque[str] = deque(maxlen=20)
        diagnostic_tail_truncated = False
        pending_response: DiagnosticEvent | None = None
        pem_end: str | None = None

        def capture_diagnostic(line: str) -> None:
            nonlocal diagnostic_tail_truncated, pem_end
            # Suppress private blocks before line clipping/eviction, including
            # an unterminated key. Flat redaction cannot see adjacent lines.
            parts = []
            while True:
                if pem_end is not None:
                    _, ended, line = line.partition(pem_end)
                    if not ended:
                        break
                    pem_end = None
                start = _PI_PEM_START.search(line)
                if start is None:
                    parts.append(line)
                    break
                parts.append(line[:start.start()] + _PEM_KEY_REDACTED_MARKER)
                pem_end = f"-----END {start['kind']}-----"
                line = line[start.end():]
            if not any(parts):
                return
            safe, clipped = _pi_diagnostic_text("".join(parts), 1024)
            diagnostic_tail_truncated |= clipped or len(stderr_lines) == stderr_lines.maxlen
            stderr_lines.append(safe)

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
        saw_native_settlement = False
        host_ready = False
        seen_host_starts: dict[str, tuple[str | None, str]] = {}
        seen_host_results: dict[str, tuple[str | None, str]] = {}
        ambiguous_host_identity = False
        host_submission_indeterminate = False
        native_host_failure = False
        host_failure_call_id: str | None = None
        host_failure_generation_id: str | None = None
        terminal_total_emitted = False
        host_server = None

        transport: CliTransport | None = None

        def observe_usage(message: dict[str, Any], source: Literal["message_end", "turn_end"]) -> MetricsEvent | None:
            nonlocal total_input, total_output, total_cache_read, total_cache_write, total_cost
            nonlocal last_model, last_provider
            last_model = message.get("responseModel") or message.get("model") or last_model
            last_provider = message.get("provider") or last_provider
            usage = _extract_usage(message)
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
            if not isinstance(inp, int) or not isinstance(outp, int):
                return None
            return MetricsEvent(
                message_id="",  # Pi has no per-message id.
                prompt_tokens=inp,
                completion_tokens=outp,
                cached_tokens=cached if isinstance(cached, int) else None,
                cost_usd=cost if isinstance(cost, (int, float)) else None,
                model_name=last_model,
                provider_name=last_provider,
                cache_creation_tokens=created if isinstance(created, int) else None,
                measurement_source=source,
            )

        def associate_submission(call_id: Any, expected: str | None, state: str) -> None:
            if (not isinstance(call_id, str) or not call_id or len(call_id.encode()) > 2048
                    or submission_calls.get(call_id) != expected
                    or (expected is None and len(submission_calls) >= 4096)):
                raise PiError("Invalid structured output association", retryable=False, category="STREAM_TRUNCATION")
            submission_calls[call_id] = state

        def host_event_identity(event: dict[str, Any]) -> tuple[str | None, str]:
            try:
                encoded = json.dumps(event, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()
            except (ValueError, RecursionError):
                raise PiError("Invalid native host-tool evidence", category="HOST_TOOLS") from None
            return last_generation_id, hashlib.sha256(encoded).hexdigest()

        def identity_diagnostic(event: dict[str, Any]) -> DiagnosticEvent:
            # Preserve the first authoritative ATIF pair. A reused identity's
            # actual native request/rejection belongs to diagnostic evidence,
            # never a fabricated second correlation or silent disappearance.
            payload = {key: event.get(key) for key in ("toolName", "args", "result", "isError") if key in event}
            admitted, diagnostic = _admit_json_value(payload)
            metadata: dict[str, Any] = {"call_id": event.get("toolCallId"),
                                        "native_event_type": event.get("type"),
                                        "generation_id": last_generation_id}
            try:
                size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
            except (ValueError, RecursionError):
                size = 32769
            if diagnostic is None and size <= 32768:
                metadata["native_payload"] = admitted
            else:
                metadata["native_payload_unavailable"] = "bounded_admission_failed"
            return DiagnosticEvent(code="pi_host_tool_identity_conflict",
                                   message="Native tool identity was reused or its completion disagreed",
                                   metadata=metadata)

        def terminal_events() -> tuple[CostEvent, ResultEvent]:
            nonlocal terminal_total_emitted
            terminal_total_emitted = True
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
        # native flag) so it stays None. Native schemas validate tool arguments, not provider decoding.
        if lifecycle is not None:
            lifecycle.request_prepared()
        yield RequestEvent(
            prompt=full_prompt,
            system_prompt=system_prompt,
            model_name=self._model_override if review_context else self.model,
            provider_name=provider,
            session_id=effective_session_id,
            reasoning_effort=thinking,
            output_schema=output_schema,
            config=PiRequestConfig(
                finalization=finalization,
                read_only=read_only,
                persist_session=persist_session,
                continuation_mode="resume" if resume_id is not None else "fresh",
                model_mode="single",
                selected_tools_count=(
                    0 if finalization or tools_disabled else
                    len(_PI_READ_ONLY_TOOLS.split(",")) + int(native_output)
                    if read_only else None
                ),
                selected_tools_present=read_only and not (finalization or tools_disabled),
                no_tools=finalization or tools_disabled,
                no_skills=True,
                schema_emulated=output_schema is not None and not native_output,
                no_extensions=read_only and (native_output or host_tools is not None or review_context is not None),
            ),
            model_source=None if review_context and self._model_override is None else "configured",
            provider_source="configured" if provider is not None else None,
            session_source="host_generated" if resume_id is None else "configured",
        )

        attachments = AsyncExitStack()
        native_failed = False
        cleaned_up = False
        process_running_at_cleanup: bool | None = None

        async def cleanup() -> None:
            nonlocal cleaned_up, process_running_at_cleanup
            if cleaned_up:
                return
            cleaned_up = True
            if host_tools is not None:
                host_tools.close_authority()
            try:
                if transport is not None:
                    process_running_at_cleanup = transport.returncode is None
                    if lifecycle is not None:
                        lifecycle.teardown_started()
                    async def collect_tail() -> None:
                        nonlocal diagnostic_tail_truncated
                        assert transport is not None
                        size = 0
                        try:
                            async with asyncio.timeout(host_policy._CLEANUP_TIMEOUT_S):
                                count = 0
                                async for line in transport.lines(lambda: 0.1):
                                    size += len(line.encode("utf-8")) + 1
                                    count += 1
                                    if size > 64 * 1024 or count > 1024:
                                        diagnostic_tail_truncated = True
                                        break
                                    try:
                                        json.loads(line)
                                    except json.JSONDecodeError:
                                        capture_diagnostic(line)
                        except Exception:  # noqa: BLE001 - diagnostic-only tail cannot replace the original failure
                            diagnostic_tail_truncated = True

                    tail = asyncio.create_task(collect_tail()) if native_failed else None
                    try:
                        await teardown(transport, self._transports,
                                       cleanup_timeout_s=host_policy._CLEANUP_TIMEOUT_S if host_tools else None)
                    except BaseException:
                        if host_tools is not None:
                            host_tools.mark_cleanup_failed()
                        if host_tools is None or not native_failed:
                            raise
                    else:
                        if lifecycle is not None:
                            lifecycle.teardown_completed()
                    finally:
                        if tail is not None:
                            with anyio.CancelScope(shield=True):
                                await asyncio.gather(tail, return_exceptions=True)
            finally:
                try:
                    await attachments.aclose()
                finally:
                    if host_tools is not None and lifecycle is not None:
                        lifecycle.host_tools_closed(native_settled=host_tools.native_settled,
                                                    cleanup_complete=host_tools.cleanup_complete,
                                                    calls=host_tools.calls, rejections=host_tools.rejections)

        try:
            extensions: list[tuple[str, str, dict[str, Any]]] = []
            if host_tools is not None:
                host_server = await attachments.enter_async_context(host_tools.session())
                await self._require_host_tools_version(process_cwd, child_env, host_tools)
                host_packet = host_server.packet()
                if self._hosted_read_only:
                    host_packet["hosted_model"] = {"provider": provider, "id": self.model}
                extensions.append(("DAYDREAM_PI_HOST_PACKET", "pi_host_tools.ts", host_packet))
            if native_output or review_context is not None:
                packet: dict[str, Any] = {"output_schema": output_schema} if native_output else {}
                if review_context is not None:
                    packet["source_root"] = str(cwd)
                if self._hosted_read_only:
                    packet["hosted_read_only"] = True
                if host_server is not None:
                    packet["host_tools"] = host_server.packet()
                if tool_call_budget is not None:
                    packet["tool_call_budget"] = tool_call_budget
                    assignment = _pi_submission_assignment(review_instructions)
                    if assignment is not None:
                        packet["submission_assignment"] = assignment
                extensions.append(("DAYDREAM_PI_OUTPUT_PACKET", "pi_structured_output.ts", packet))
            for variable in ("DAYDREAM_PI_HOST_PACKET", "DAYDREAM_PI_OUTPUT_PACKET"):
                child_env.pop(variable, None)
            if read_only and (native_output or host_tools is not None or review_context is not None):
                args.append("--no-extensions")
            diagnostic_extension = attachments.enter_context(as_file(
                files("daydream.backends").joinpath("pi_diagnostics.ts")))
            args.extend(["--extension", str(diagnostic_extension)])
            for variable, resource, packet in extensions:
                child_env[variable] = str(_write_prompt_attachment(json.dumps(packet, ensure_ascii=False), attachments))
                extension = attachments.enter_context(as_file(files("daydream.backends").joinpath(resource)))
                args.extend(["--extension", str(extension)])
            if not tools_disabled:
                args.append(f"@{_write_prompt_attachment(full_prompt, attachments)}")
            if review_instructions and not finalization:
                # Pi resolves existing system-prompt paths directly (without @).
                # Review instructions are caller-controlled and need the same
                # size-independent transport as the user prompt.
                index = args.index("--append-system-prompt") + 1
                args[index] = str(_write_prompt_attachment(system_prompt, attachments))
            self._check_review_context()
            transport = CliTransport(
                "pi",
                args,
                stdin_mode=StdinMode.PIPE if tools_disabled else StdinMode.DEVNULL,
                stdin_data=full_prompt.encode("utf-8") if tools_disabled else None,
                stderr_policy=StderrPolicy.MERGE_INTO_STDOUT,
                limit=_PI_STDOUT_LIMIT_BYTES,
                env=child_env,
                cwd=str(process_cwd),
                spawn_pending=lifecycle.spawn_pending if lifecycle is not None else None,
                spawn_completed=lifecycle.spawn_completed if lifecycle is not None else None,
                raw_line_received=lifecycle.raw_line_received if lifecycle is not None else None,
                stdout_eof=lifecycle.stdout_eof if lifecycle is not None else None,
                process_exited=lifecycle.process_exited if lifecycle is not None else None,
            )
            self._transports.append(transport)
            if host_tools is not None:
                await host_tools.start(transport.start(), lambda _: teardown(
                    transport, self._transports, cleanup_timeout_s=host_policy._CLEANUP_TIMEOUT_S))
            else:
                await transport.start()

            active_tool_calls = 0
            is_first_line = True
            setup_deadline = clock.monotonic() + host_policy._SETUP_TIMEOUT_S + 2 if host_tools else None

            def line_timeout() -> float | None:
                ordinary = tool_idle_timeout_s if active_tool_calls > 0 else response_idle_timeout_s
                if host_tools is None:
                    return ordinary
                bound = host_tools.deadline
                if not host_ready and setup_deadline is not None:
                    bound = min(bound, setup_deadline)
                remaining = max(0.0, bound - clock.monotonic())
                return min(ordinary, remaining) if ordinary is not None else remaining

            async for raw_line in transport.lines(line_timeout):
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    if native_output and raw_line.lstrip().startswith(("{", "[")):
                        raise PiError("Malformed native event", retryable=True, category="STREAM_TRUNCATION")
                    capture_diagnostic(raw_line)
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
                if event_type == "daydream_provider_response":
                    if pending_response is not None:
                        yield pending_response
                    response_metadata = _pi_response_metadata(event)
                    pending_response = (DiagnosticEvent(
                        code="pi_provider_response", message="Provider HTTP response", metadata={
                            **response_metadata, "generation_id": None, "observed_at_unix_ns": time.time_ns(),
                        }) if response_metadata is not None else None)
                    continue
                if event_type == "daydream_host_tools_ready":
                    if host_server is None or not host_server.ready:
                        raise PiError("Native host readiness was not established", category="HOST_TOOLS")
                    host_ready = True
                    continue
                if event_type == "daydream_host_tools_failed":
                    if self._hosted_read_only:
                        if event.get("code") == "hosted_model_selection":
                            raise _unsupported_hosted_policy(
                                "Hosted Pi model selection must exactly match the operator's provider/model; "
                                f"configure the native registry with {provider}/{self.model}. "
                                "Fuzzy model fallback is unsupported.")
                        raise _unsupported_hosted_policy(
                            "Hosted Pi native readiness failed; required stock tools and packaged helpers "
                            "must be supported by official Pi 1.4.0 / Node 24.18.1")
                    raise PiError("Required native host-tool binding is unavailable", category="HOST_TOOLS")
                if event_type == "daydream_host_tool_fatal" and host_tools is not None:
                    # Native details only explain the stop. Binding-local state
                    # remains the authority for acceptance and retry decisions.
                    native_host_failure = True
                    host_failure_generation_id = last_generation_id
                    native_id = event.get("id")
                    if isinstance(native_id, str) and 0 < len(native_id.encode()) <= 512:
                        host_failure_call_id = native_id
                    native_message, _ = _pi_diagnostic_text(str(event.get("error", "")), 1024)
                    yield DiagnosticEvent(code="pi_host_tool_native_error", message=native_message,
                        metadata={"call_id": host_failure_call_id,
                                  "outcome": event.get("outcome") if event.get("outcome") in
                                  {"fatal", "indeterminate"} else "unavailable",
                                  "generation_id": last_generation_id})
                    continue
                if lifecycle is not None and isinstance(event_type, str) and event_type in _PI_EVENT_TYPES:
                    lifecycle.native_event()
                if event_type in {"agent_start", "turn_start", "message_start", "tool_execution_start"}:
                    saw_native_settlement = False

                if host_tools is not None and event_type in {"tool_execution_start", "tool_execution_end"}:
                    native_id = event.get("toolCallId")
                    if event_type == "tool_execution_start":
                        if not isinstance(native_id, str) or not native_id or len(native_id.encode()) > 512:
                            raise PiError("Invalid native tool identity", category="HOST_TOOLS")
                        seen = seen_host_starts
                    else:
                        if not isinstance(native_id, str) or native_id not in seen_host_starts:
                            raise PiError("Uncorrelated native tool result", category="HOST_TOOLS")
                        seen = seen_host_results
                    identity = host_event_identity(event)
                    if native_id in seen:
                        if seen[native_id] != identity:
                            ambiguous_host_identity = True
                            yield identity_diagnostic(event)
                        continue
                    seen[native_id] = identity

                if event_type == "agent_start":
                    pass  # Lifecycle marker; nothing to emit.

                elif event_type == "turn_start":
                    saw_turn_start = True
                    saw_finish_reason = False
                    last_generation_id = None

                elif event_type == "message_start":
                    msg = event.get("message") or {}
                    if msg.get("role") == "assistant" and open_generation_id is None:
                        # P18: host receipt of the assistant generation start.
                        # Host-observed only — never relabeled as provider
                        # request start; the native start comes from the
                        # completed message's Unix-ms timestamp at message_end.
                        open_generation_id = _new_generation_id()
                        last_generation_id = open_generation_id
                        generation_start_ns = time.time_ns()
                        if lifecycle is not None:
                            lifecycle.generation_started()
                        assert generation_start_ns is not None  # just assigned
                        yield GenerationStartEvent(
                            generation_id=open_generation_id,
                            observed_at_unix_ns=generation_start_ns,
                            boundary_complete=True,
                        )
                        if pending_response is not None:
                            pending_response.metadata["generation_id"] = open_generation_id
                            yield pending_response
                            pending_response = None

                elif event_type == "message_update":
                    # Pi's JSON mode emits delta-only assistantMessageEvent frames;
                    # no outer message role or content is needed for admission.
                    update = event.get("assistantMessageEvent")
                    if lifecycle is not None and isinstance(update, dict):
                        content_index, delta = update.get("contentIndex"), update.get("delta")
                        if type(content_index) is int and content_index >= 0 and isinstance(delta, str) and delta:
                            update_type = update.get("type")
                            if update_type == "text_delta":
                                lifecycle.progress("text")
                            elif update_type == "thinking_delta":
                                lifecycle.progress("thinking")
                            elif update_type == "toolcall_delta":
                                lifecycle.progress("toolcall")

                elif event_type == "message_end":
                    msg = event.get("message") or {}
                    if msg.get("role") == "assistant":
                        # The finalized message already contains native usage.
                        # Keep normal accounting at turn_end, but retain this
                        # evidence if tool correlation/transport fails before it.
                        pending_usage = msg
                        if lifecycle is not None:
                            lifecycle.generation_ended()
                        last_model = msg.get("responseModel") or msg.get("model") or last_model
                        last_provider = msg.get("provider") or last_provider
                        choice_parts: list[Any] = []
                        for block in msg.get("content") or []:
                            if not isinstance(block, dict):
                                continue
                            btype = block.get("type")
                            if btype == "text":
                                text = block.get("text", "")
                                if text:
                                    yield TextEvent(text=text)
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
                                if msg.get("stopReason") == "length":
                                    if (not isinstance(call_id, str) or not call_id or len(call_id.encode()) > 2048
                                            or any(ord(char) < 32 for char in call_id)
                                            or call_id in incomplete_calls or len(incomplete_calls) >= 4096):
                                        raise PiError("Incomplete tool input association lost", retryable=False,
                                                      category="STREAM_TRUNCATION")
                                    incomplete_calls.add(call_id)
                                if native_output and call_name == "structured_output":
                                    associate_submission(call_id, None, "proposed")
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
                    elif (native_output and msg.get("role") == "toolResult"
                          and msg.get("toolName") == "structured_output"):
                        is_error = msg.get("isError")
                        if type(is_error) is not bool:
                            raise PiError("Invalid output status", retryable=False, category="STREAM_TRUNCATION")
                        associate_submission(msg.get("toolCallId"), "error" if is_error else "success", "finalized")
                        if not is_error:
                            details = msg.get("details")
                            try:
                                size = len(json.dumps(details, ensure_ascii=False, allow_nan=False,
                                                      separators=(",", ":")).encode())
                            except (ValueError, RecursionError):
                                size = _PI_FINAL_PAYLOAD_MAX_BYTES + 1
                            admitted, diagnostic = _admit_json_value(details)
                            if details is None or diagnostic is not None or size > _PI_FINAL_PAYLOAD_MAX_BYTES:
                                raise PiError("Invalid output details", retryable=False, category="STREAM_TRUNCATION")
                            pending_submission = admitted
                            pending_submission_generation = last_generation_id

                elif event_type == "tool_execution_start":
                    active_tool_calls += 1
                    if lifecycle is not None:
                        lifecycle.active_tools(active_tool_calls)
                    call_id = event.get("toolCallId")
                    yield ToolStartEvent(
                        id=call_id or str(uuid.uuid4()),
                        name=event.get("toolName", "unknown"),
                        input=event.get("args") or {},
                        input_incomplete=isinstance(call_id, str) and call_id in incomplete_calls,
                    )
                    if native_output and event.get("toolName") == "structured_output":
                        if event.get("parentToolCallId") is not None:
                            raise PiError("Nested output call", retryable=False, category="STREAM_TRUNCATION")
                        associate_submission(event.get("toolCallId"), "proposed", "started")

                elif event_type == "tool_execution_end":
                    active_tool_calls = max(0, active_tool_calls - 1)
                    if lifecycle is not None:
                        lifecycle.active_tools(active_tool_calls)
                    call_id = event.get("toolCallId")
                    result = event.get("result")
                    if (host_tools is not None and isinstance(result, dict)
                            and isinstance(result.get("details"), dict)
                            and result["details"].get("host_submission_status") in {"fatal", "indeterminate"}
                            and isinstance(call_id, str)):
                        host_failure_call_id = call_id
                        host_failure_generation_id = last_generation_id
                    elif (host_tools is not None and host_tools.fatal_error is not None
                          and event.get("toolName") == "structured_output" and isinstance(call_id, str)):
                        host_failure_call_id = call_id
                        host_failure_generation_id = last_generation_id
                    if (host_tools is not None and event.get("toolName") == "structured_output"
                            and event.get("isError") is True and isinstance(result, dict)
                            and result.get("details") == {"host_submission_status": "indeterminate"}):
                        host_submission_indeterminate = True
                        host_tools.close_authority()
                    yield ToolResultEvent(
                        id=call_id or str(uuid.uuid4()), output=_render_tool_result(result),
                        is_error=bool(event.get("isError", False)),
                        **_tool_completion_metadata(event,
                            output_control=native_output and event.get("toolName") == "structured_output"),
                    )
                    if isinstance(call_id, str):
                        incomplete_calls.discard(call_id)
                    if native_output and event.get("toolName") == "structured_output":
                        if type(event.get("isError")) is not bool:
                            raise PiError("Invalid output status", retryable=False, category="STREAM_TRUNCATION")
                        associate_submission(event.get("toolCallId"), "started",
                                             "error" if event["isError"] else "success")

                elif event_type == "tool_execution_update":
                    if (lifecycle is not None and active_tool_calls > 0
                            and isinstance(event.get("partialResult"), dict)):
                        lifecycle.progress("tool")

                elif event_type == "turn_end":
                    if not native_output and host_tools is None:
                        active_tool_calls = 0
                        if lifecycle is not None:
                            lifecycle.active_tools(0)
                    msg = event.get("message") or {}
                    stop_reason = msg.get("stopReason")
                    last_model = msg.get("responseModel") or msg.get("model") or last_model
                    last_provider = msg.get("provider") or last_provider
                    if stop_reason is not None:
                        saw_finish_reason = True
                        finish_reason = stop_reason
                    pending_usage = None
                    metrics = observe_usage(msg, "turn_end")
                    if metrics is not None:
                        yield metrics
                    # Use native stopReason/model/provider when exposed. Pi has no message
                    # id; keep it empty and retain host-observed timing.
                    if stop_reason == "error":
                        supplied_error = msg.get("errorMessage")
                        native_error = supplied_error if isinstance(supplied_error, str) and supplied_error else None
                        error_msg = native_error or "Unknown Pi error"
                        category = _pi_error_category(error_msg)
                        retryable = _pi_retryable_for(category=category, message=error_msg)
                        # Redact whole fields before clipping, so a bound cannot turn a
                        # complete credential/PEM block into an unrecognizable fragment.
                        diagnostic_message, message_truncated = (
                            _pi_diagnostic_text(native_error, _PI_ERROR_MESSAGE_MAX_CHARS)
                            if native_error is not None else ("", False)
                        )
                        diagnostic_metadata: dict[str, Any] = {
                            "category": category, "retryable": retryable, "finish_reason": "error",
                            "message_source": "native" if native_error is not None else "unavailable",
                        }
                        if message_truncated:
                            diagnostic_metadata["message_truncated"] = True
                        raw_stop_reason = msg.get("rawStopReason")
                        if isinstance(raw_stop_reason, str) and raw_stop_reason:
                            redacted_stop, stop_truncated = _pi_diagnostic_text(
                                raw_stop_reason, _PI_RAW_STOP_REASON_MAX_CHARS,
                            )
                            diagnostic_metadata["raw_stop_reason"] = redacted_stop
                            if stop_truncated:
                                diagnostic_metadata["raw_stop_reason_truncated"] = True
                        if last_generation_id is not None:
                            diagnostic_metadata["generation_id"] = last_generation_id
                        # Observe before TurnEnd/Result close the assistant step. The
                        # recorder applies its existing JSON normalization/redaction too.
                        yield DiagnosticEvent(
                            code="pi_terminal_error",
                            message=diagnostic_message,
                            metadata=diagnostic_metadata,
                        )
                    yield TurnEndEvent(
                        message_id="",
                        finish_reason=stop_reason if stop_reason is not None else None,
                        model_name=last_model,
                        provider_name=last_provider,
                        model_source="native",
                        provider_source="native" if last_provider is not None else None,
                    )
                    if stop_reason == "error" and not (host_tools is not None and host_tools.fatal_error is not None):
                        for terminal in terminal_events():
                            yield terminal
                        raise PiError(
                            error_msg,
                            retryable=retryable,
                            category=category,
                            retry_after=parse_message_retry_hint(error_msg),
                        )

                elif event_type == "agent_end":
                    # No inline finalization — Cost/Result are emitted once from
                    # the single post-loop path below. The loop keeps
                    # draining to EOF so the stdout pipe cannot fill mid-run.
                    pass
                elif event_type == "agent_settled":
                    saw_native_settlement = True
                    if host_tools is not None:
                        host_tools.native_settled = True
                        host_tools.close_authority()

                # Streaming updates only retain content-free progress here;
                # full output events still come from message_end/tool_execution_end.

            # Reap the child, then yield its terminal events before the shared
            # exit check formats the backend-specific message from the code and
            # captured stderr lines (the events must stay between the two).
            returncode = await reap(transport, timeout_s=host_policy._CLEANUP_TIMEOUT_S if host_tools else None)
            if pending_response is not None:
                yield pending_response
                pending_response = None

            native_unsettled = (native_output or host_tools is not None) and (
                active_tool_calls != 0 or open_generation_id is not None
                or bool(incomplete_calls)
                or any(state != "finalized" for state in submission_calls.values())
                or finish_reason in {"aborted", "length", "error"}
                or not (saw_turn_start and saw_finish_reason and saw_native_settlement)
                or (host_tools is not None and not host_ready)
                or (host_tools is not None and (
                    not host_tools.accepted or pending_submission != host_tools.accepted_result
                    or pending_submission_generation != last_generation_id))
                or ambiguous_host_identity
                or host_submission_indeterminate
            )
            if host_server is not None and host_tools is not None:
                await attachments.aclose()
                if not host_tools.cleanup_complete:
                    native_unsettled = True
                yield DiagnosticEvent(code="pi_host_tools_lifecycle", message="Invocation host endpoint outcomes",
                                      metadata={"authority_closed": not host_tools.authority_open,
                                                "native_settled": host_tools.native_settled,
                                                "endpoint_cleanup_complete": host_tools.cleanup_complete,
                                                "calls": host_tools.calls, "rejections": host_tools.rejections})
            if native_output and returncode == 0 and not native_unsettled:
                structured_result = pending_submission
            if pending_usage is not None:
                # EOF can arrive before turn_end. Include its already-finalized
                # usage before the single terminal total, rather than first
                # publishing a stale aggregate and correcting it on failure.
                message = pending_usage
                pending_usage = None
                metrics = observe_usage(message, "message_end")
                if metrics is not None:
                    yield metrics
            for terminal in terminal_events():
                yield terminal

            if host_tools is not None and host_tools.fatal_error is not None:
                raise host_tools.fatal_error
            if native_host_failure:
                raise PiError("Native host-tool admission did not receive a confirmed continuation outcome",
                              retryable=False, category="HOST_TOOLS")
            if host_submission_indeterminate:
                raise PiError("Native host submission did not receive a confirmed outcome",
                              retryable=False, category="HOST_TOOLS")

            # Fail fast on non-zero exit: if pi crashed without emitting a
            # turn_end error event, surface the failure with diagnostic output
            # instead of reporting a successful completion with empty/partial
            # output.
            raise_for_exit(
                returncode,
                error_type=PiError,
                category="PROCESS_EXIT",
                build_message=partial(_pi_process_exit_message, list(stderr_lines)),
                retryable=_is_retryable_exit_code(returncode),
            )

            if native_unsettled or (saw_turn_start and not saw_finish_reason):
                raise PiError(
                    "Native host-tool identity conflict prevents complete correlation" if ambiguous_host_identity
                    else "Native stream did not settle" if native_unsettled else "Stream ended without finish_reason",
                    retryable=True,
                    category="STREAM_TRUNCATION",
                )

        except Exception:
            native_failed = True
            await cleanup()
            if pending_response is not None:
                yield pending_response
            if transport is not None:
                yield DiagnosticEvent(code="pi_process_failure", message="Pi process failure context",
                    metadata={"process_running_at_cleanup": process_running_at_cleanup,
                              "process_returncode": transport.returncode,
                              "diagnostic_tail": list(stderr_lines),
                              "diagnostic_tail_truncated": diagnostic_tail_truncated})
            if pending_usage is not None:
                # Do not consume later native events after an association error.
                # Flush only usage already observed on the finalized assistant
                # message, once; successful turns cleared it at turn_end.
                message = pending_usage
                pending_usage = None
                metrics = observe_usage(message, "message_end")
                if metrics is not None:
                    yield metrics
                yield terminal_events()[0]
            if host_tools is not None and host_tools.fatal_error is not None:
                fatal = host_tools.fatal_error
                if not terminal_total_emitted:
                    yield terminal_events()[0]
                fatal_message, _ = _pi_diagnostic_text(str(fatal), 1024)
                cause = fatal.__cause__
                cause_metadata = ({"cause_type": type(cause).__name__,
                                   "cause_message": _pi_diagnostic_text(str(cause), 1024)[0]}
                                  if cause is not None else {})
                yield DiagnosticEvent(code="pi_host_tool_fatal", message=fatal_message,
                    metadata={"outcome": fatal.outcome, "code": fatal.code, "retryable": False,
                              "call_id": host_failure_call_id,
                              "generation_id": host_failure_generation_id or last_generation_id,
                              **cause_metadata})
                raise PiError(str(fatal), retryable=False, category="HOST_TOOLS") from fatal
            raise
        except BaseException:
            native_failed = True
            raise
        finally:
            await cleanup()

    async def cancel(self) -> None:
        """Terminate and reap every active Pi transport."""
        await CliTransport.cancel_all(self._transports)
