"""Translate Osprey's versioned headless JSONL into backend events.

Osprey owns the agent loop, tools, MCP, policy, hooks, caps, and provider telemetry.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import AsyncGenerator, Callable, Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    OspreyRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    resolve_fanout_concurrency,
)
from daydream.backends._subprocess import stream_idle_timeout_s
from daydream.backends._transport import (
    CliTransport,
    teardown,
    write_temp_json_schema,
)
from daydream.trajectory import redact_text

logger = logging.getLogger(__name__)

_OSPREY_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024
_OSPREY_OBSERVATION_UPDATE_BYTES = 64 * 1024
_OSPREY_OBSERVATION_INLINE_BYTES = 256 * 1024
_OSPREY_OBSERVATION_ADMISSION_BYTES = 2 * 1024 * 1024
_JSONL_PROTOCOL_VERSION = 2
_MAX_DIAGNOSTIC_LINES = 10
_MAX_DIAGNOSTIC_LINE_CHARS = 500

_SUCCESS_OUTCOMES = frozenset({"completed", "terminal_tool"})
_KNOWN_IGNORED_EVENTS = frozenset(
    {
        # Producer/ATIF lifecycle and telemetry without a Daydream event equivalent.
        "completion_gate",
        "finalization_record",
        "verification_checkpoint",
        "context_compaction",
        "driver_retry",
        "todo_updated",
        "moa_reference_start",
        "moa_reference_delta",
        "moa_reference_end",
        "moa_aggregating",
        "model_change",
        "persona_switch",
        "workflow_started",
        "workflow_phase_started",
        "workflow_phase_completed",
        "workflow_phase_failed",
        "workflow_agent_started",
        "workflow_agent_settled",
        "workflow_notice",
        "workflow_control",
        "workflow_script_saved",
        "workflow_completed",
        "workflow_failed",
        "effort_notice",
        "ultracode_status_note",
        "startup_notice",
        "extension_warning",
        "extension_management",
        "reusable_work_suggestion",
    }
)


class OspreyError(Exception):
    """Bounded failure from the Osprey subprocess or JSONL contract."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "OSPREY",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


class OspreyUnsupportedOption(OspreyError):
    """A requested daydream option has no verified headless CLI mapping."""

    def __init__(self, option: str, detail: str) -> None:
        super().__init__(
            f"Osprey headless boundary does not support {option}: {detail}",
            category="UNSUPPORTED_OPTION",
        )


class OspreyProtocolError(OspreyError):
    """The Osprey JSONL stream is malformed or violates its ordering contract."""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="PROTOCOL")


class OspreyTerminalError(OspreyError):
    """Osprey emitted a non-success terminal outcome."""

    def __init__(self, outcome: str, detail: str = "") -> None:
        suffix = f": {detail}" if detail else ""
        super().__init__(
            f"Osprey session ended with outcome {outcome!r}{suffix}",
            category="TERMINAL_OUTCOME",
        )
        self.outcome = outcome


def _required_string(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value:
        raise OspreyProtocolError(f"event {event.get('event')!r} requires non-empty string {key!r}")
    return value


def _required_int(event: dict[str, Any], key: str) -> int:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OspreyProtocolError(f"event {event.get('event')!r} requires integer {key!r}")
    return value


def _optional_int(event: dict[str, Any], key: str) -> int | None:
    value: object = event.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise OspreyProtocolError(f"event {event.get('event')!r} has invalid integer {key!r}")
    return value


def _required_non_negative_int(event: dict[str, Any], key: str) -> int:
    value = _required_int(event, key)
    if value < 0:
        raise OspreyProtocolError(f"event {event.get('event')!r} has negative {key!r}")
    return value


def _optional_non_negative_int(event: dict[str, Any], key: str) -> int | None:
    value = _optional_int(event, key)
    if value is not None and value < 0:
        raise OspreyProtocolError(f"event {event.get('event')!r} has negative {key!r}")
    return value


def _parse_cost(value: Any, *, event_name: str, field: str = "cost_usd") -> float | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OspreyProtocolError(f"event {event_name!r} has non-string {field!r}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise OspreyProtocolError(f"event {event_name!r} has invalid {field!r}") from exc
    if not math.isfinite(parsed):
        raise OspreyProtocolError(f"event {event_name!r} has non-finite {field!r}")
    if parsed < 0:
        raise OspreyProtocolError(f"event {event_name!r} has negative {field!r}")
    return parsed


def _bounded_diagnostics(lines: Iterable[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        if len(cleaned) >= _MAX_DIAGNOSTIC_LINES:
            break
        cleaned.append(_bounded_diagnostic_line(line))
    if not cleaned:
        return "no diagnostic output captured"
    return "\n".join(cleaned)


def _bounded_diagnostic_line(line: str) -> str:
    return redact_text(line)[:_MAX_DIAGNOSTIC_LINE_CHARS]


def _osprey_process_exit_message(stderr_lines: list[str], returncode: int) -> str:
    """Inline bounded diagnostics without Codex/Pi's count header."""
    return f"Osprey CLI exited with return code {returncode}: {_bounded_diagnostics(stderr_lines)}"


def _stderr_diagnostic_sink(diagnostics: list[str]) -> Callable[[str], None]:
    """Build the redacting, capped sink the transport's stderr drain feeds."""

    def sink(line: str) -> None:
        if len(diagnostics) < _MAX_DIAGNOSTIC_LINES:
            diagnostics.append(_bounded_diagnostic_line(line))

    return sink


@dataclass
class _OspreyProtocolState:
    """Ordering and correlation state owned by one Osprey JSONL stream."""

    active_turn_id: str | None = None
    pending_tool_calls: set[str] = field(default_factory=set)

    def start_turn(self, turn_id: str) -> None:
        if self.active_turn_id is not None:
            raise OspreyProtocolError("turn_start before prior turn_end")
        self.active_turn_id = turn_id

    def end_turn(self, turn_id: str) -> None:
        if self.active_turn_id is None:
            raise OspreyProtocolError("turn_end without turn_start")
        if turn_id != self.active_turn_id:
            raise OspreyProtocolError(f"turn_end {turn_id!r} does not match active turn {self.active_turn_id!r}")
        self.active_turn_id = None

    def start_tool_call(self, call_id: str) -> None:
        if call_id in self.pending_tool_calls:
            raise OspreyProtocolError(f"duplicate pending tool_call_id {call_id!r}")
        self.pending_tool_calls.add(call_id)

    def finish_tool_call(self, call_id: str) -> None:
        # Preserve unmatched results: the trajectory recorder has an explicit
        # bucket for them. Matching calls are retired.
        self.pending_tool_calls.discard(call_id)


@dataclass(frozen=True, kw_only=True, repr=False)
class OspreyConfig(OspreyRequestConfig):
    """Own native options; only a fresh exact-base request config crosses telemetry.

    Field groups on OspreyRequestConfig declare scalar argv order once. Private
    paths, labels and variables are invocation inputs, never request metadata.
    """

    model: str | None = None
    reasoning_effort: str | None = None
    osprey_binary: str = ""
    persona: str | None = None
    toolset: str | None = None
    sandbox: bool | None = False
    immutable_surface: bool | None = False
    ultracode: bool | None = False
    allowed_roots: Iterable[Path | str] = ()
    atif_output: Path | None = None
    atif_system_prompt_plaintext: bool = False
    tool_result_raw_dir: Path | None = None
    vars: Iterable[tuple[str, str]] = ()
    effort: str | None = None
    provider: str | None = None
    base_url: str | None = None
    osprey_home: Path | None = None

    def __post_init__(self) -> None:
        if self.provider is not None:
            raise OspreyUnsupportedOption(
                "provider",
                "the current CLI resolves providers from Osprey configuration/environment; it has no provider flag",
            )
        if self.base_url is not None:
            raise OspreyUnsupportedOption(
                "base_url",
                "the current CLI resolves custom endpoints from Osprey "
                "configuration/environment; it has no base-url flag",
            )

        if self.approval_mode is not None:
            if self.approval_mode in {"on-request", "on-failure", "unless-trusted"}:
                raise OspreyUnsupportedOption(
                    "approval",
                    f"{self.approval_mode!r} requires an interactive approver and headless mode rejects it",
                )
            if self.approval_mode != "deny-untrusted":
                raise OspreyUnsupportedOption("approval", f"unsupported headless value {self.approval_mode!r}")
        super().__post_init__()
        object.__setattr__(self, "osprey_binary", self.osprey_binary or os.environ.get("OSPREY_BINARY", "osprey"))
        object.__setattr__(self, "allowed_roots", tuple(str(root) for root in self.allowed_roots))
        object.__setattr__(self, "vars", tuple(self.vars))

    def prepare(
        self,
        prompt: str,
        *,
        output_schema_path: str | Path | None = None,
        continuation: ContinuationToken | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
        tool_search_mode: str | None = None,
    ) -> tuple[list[str], OspreyRequestConfig]:
        """Compile argv and immutable metadata from the same admitted call controls."""
        if tool_search_mode is not None:
            raise OspreyUnsupportedOption(
                "tool_search_mode",
                "Osprey resolves [agent].tool_search from its config; the "
                "current headless CLI exposes no corresponding flag",
            )
        if not persist_session:
            raise OspreyUnsupportedOption(
                "persist_session=False",
                "the current headless CLI has resume/fork but no ephemeral-session flag",
            )
        if continuation is not None and continuation.backend not in {"osprey", ""}:
            continuation = None
        resume_id: str | None = None
        mode = "fresh"
        if continuation is not None:
            resume_id = continuation.data.get("session_id")
            if not isinstance(resume_id, str) or not resume_id:
                raise OspreyProtocolError("osprey continuation token requires session_id")
            mode = continuation.data.get("mode", "resume")
            if mode not in ("resume", "fork"):
                raise OspreyProtocolError(f"unknown osprey continuation mode {mode!r}")
        variables = tuple(self.vars)
        # Project the closed safe schema, never serialize this private subclass.
        request = OspreyRequestConfig(**{
            **{item.name: getattr(self, item.name) for item in fields(OspreyRequestConfig)},
            "temperature": float(self.temperature) if self.temperature is not None else None,
            "finalization": None,
            "read_only": read_only,
            "max_turns": max_turns,
            "persist_session": None,
            "continuation_mode": mode,
            "model_mode": "single",
            "persona_present": self.persona is not None,
            "toolset_present": self.toolset is not None,
            "observation_update_bytes": _OSPREY_OBSERVATION_UPDATE_BYTES,
            "observation_inline_bytes": _OSPREY_OBSERVATION_INLINE_BYTES,
            "observation_admission_bytes": _OSPREY_OBSERVATION_ADMISSION_BYTES,
            "vars_count": len(variables) if variables else None,
        })

        args = [
            self.osprey_binary,
            "agent",
            "--events-jsonl",
            "--observation-budget-update-bytes",
            str(_OSPREY_OBSERVATION_UPDATE_BYTES),
            "--observation-budget-inline-bytes",
            str(_OSPREY_OBSERVATION_INLINE_BYTES),
            "--observation-budget-admission-bytes",
            str(_OSPREY_OBSERVATION_ADMISSION_BYTES),
        ]

        def add_value(flag: str, value: object | None) -> None:
            if value is not None:
                args.extend([flag, str(value)])

        def add_group(group: str) -> None:
            for item in fields(request):
                if item.metadata.get("osprey_group") == group:
                    add_value("--" + item.name.replace("_", "-"), getattr(request, item.name))

        if self.persona:
            args.extend(["--persona", self.persona])
        if self.toolset:
            args.extend(["--toolset", self.toolset])
        if self.model:
            args.extend(["--model", self.model])
        add_value("--temperature", self.temperature)
        add_value("--atif-output", self.atif_output)
        if self.atif_system_prompt_plaintext:
            args.append("--atif-system-prompt-plaintext")
        if request.immutable_surface:
            args.append("--immutable-runtime-surface")
        add_value("--max-turns", request.max_turns)
        add_group("timeout")

        if request.read_only:
            args.append("--read-only")
        if request.approval_mode is not None:
            args.extend(["--approval", request.approval_mode])
        if request.sandbox:
            args.append("--sandbox")
        for root in self.allowed_roots:
            args.extend(["--allowed-root", str(root)])

        if request.compress_context is not None:
            args.append(f"--compress-context={str(request.compress_context).lower()}")
        add_group("result")
        add_value("--tool-result-raw-dir", self.tool_result_raw_dir)
        for key, value in variables:
            args.extend(["--var", f"{key}={value}"])
        if resume_id is not None:
            args.extend(["--fork-from" if request.continuation_mode == "fork" else "--resume", resume_id])
        if output_schema_path is not None:
            args.extend(["--output-schema", str(output_schema_path)])
        add_group("limit")
        add_value("--effort", self.effort or self.reasoning_effort)
        if request.ultracode:
            args.append("--ultracode")
        args.append(prompt)
        return args, request


class OspreyBackend:
    """Translate ``osprey agent --events-jsonl`` into daydream events."""

    name = "osprey"

    def __init__(self, config: OspreyConfig | None = None) -> None:
        self.config = config or OspreyConfig()
        # Requested model stays in config; native session identity updates only this hint.
        self.model = self.config.model or "unknown"
        self.fanout_concurrency = resolve_fanout_concurrency("DAYDREAM_OSPREY_FANOUT_CONCURRENCY", 4)
        self._transports: list[CliTransport] = []

    @property
    def reasoning_effort(self) -> str | None:
        return self.config.reasoning_effort

    @property
    def sandbox(self) -> bool:
        return bool(self.config.sandbox)

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
        """Run Osprey and translate its version-2 JSONL stream."""
        if agents:
            raise NotImplementedError(
                "Osprey backend does not accept daydream exploration agents; use Osprey max-subagents instead"
            )
        config = self.config
        schema_path: str | None = None
        if output_schema is not None:
            schema_path = write_temp_json_schema(output_schema, prefix="daydream-osprey-schema-")

        session_id: str | None = None
        session_model: str | None = None
        provider: str | None = None
        saw_header = False
        saw_session_start = False
        saw_session_end = False
        failed_message: str | None = None
        total_cost: float | None = None
        stderr_lines: list[str] = []
        terminal_outcome: str | None = None
        terminal_exit_code: int | None = None
        terminal_structured_output: Any = None
        turn_text_emitted = False
        turn_started_at: str | None = None
        thinking_parts: list[str] = []
        protocol_state = _OspreyProtocolState()

        transport: CliTransport | None = None

        try:
            command, request_config = config.prepare(
                prompt,
                output_schema_path=schema_path,
                continuation=continuation,
                max_turns=max_turns,
                read_only=read_only,
                persist_session=persist_session,
            )
            child_env = os.environ.copy()
            if config.osprey_home is not None:
                child_env["OSPREY_HOME"] = str(config.osprey_home)
            transport = CliTransport(
                "osprey",
                command,
                stderr_sink=_stderr_diagnostic_sink(stderr_lines),
                # Repair non-UTF-8 tool output inside valid JSON rather than aborting.
                decode_errors="replace",
                limit=_OSPREY_STDOUT_LIMIT_BYTES,
                env=child_env,
                cwd=str(cwd),
            )
            self._transports.append(transport)
            await transport.start()
            stream = transport.lines(stream_idle_timeout_s)
            while True:
                try:
                    raw_line = await anext(stream)
                except StopAsyncIteration:
                    break
                except ValueError as exc:
                    raise OspreyProtocolError(
                        f"Osprey stdout JSONL line exceeded the {_OSPREY_STDOUT_LIMIT_BYTES}-byte limit"
                    ) from exc
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise OspreyProtocolError(
                        f"Osprey emitted a non-JSON line in JSONL mode: {_bounded_diagnostics([raw_line])}"
                    ) from exc
                if not isinstance(event, dict):
                    raise OspreyProtocolError("Osprey JSONL event must be an object")
                event_name = event.get("event")
                if not isinstance(event_name, str) or not event_name:
                    raise OspreyProtocolError("Osprey JSONL event requires string event")

                if not saw_header:
                    if event_name != "protocol":
                        raise OspreyProtocolError("Osprey JSONL stream must begin with protocol header")
                    version = event.get("version")
                    if version != _JSONL_PROTOCOL_VERSION:
                        raise OspreyProtocolError(
                            f"unsupported Osprey JSONL protocol version {version!r}; expected {_JSONL_PROTOCOL_VERSION}"
                        )
                    saw_header = True
                    continue
                if event_name == "protocol":
                    raise OspreyProtocolError("Osprey JSONL stream contains duplicate protocol header")
                if not saw_session_start:
                    if event_name != "session_start":
                        raise OspreyProtocolError("session_start must follow the protocol header")
                    session_id = _required_string(event, "session_id")
                    started_at = _required_string(event, "started_at")
                    session_model = _required_string(event, "model")
                    # The native header supplies the authoritative model identity.
                    self.model = session_model
                    provider = _required_string(event, "provider")
                    saw_session_start = True
                    # Admit only command facts; arbitrary persona/toolset labels stay private.
                    yield RequestEvent(
                        prompt=prompt,
                        model_name=session_model,
                        provider_name=provider,
                        session_id=session_id,
                        reasoning_effort=config.effort or config.reasoning_effort,
                        output_schema=output_schema,
                        timestamp=started_at,
                        config=request_config,
                        model_source="native",
                        provider_source="native",
                        session_source="native",
                        timestamp_source="native",
                    )
                    continue
                if saw_session_end:
                    raise OspreyProtocolError("JSONL event appeared after session_end")
                if session_id is None:
                    raise OspreyProtocolError("session_start did not provide a session_id")

                if event_name != "thinking_delta" and thinking_parts:
                    thinking_text = "".join(thinking_parts)
                    thinking_parts.clear()
                    if thinking_text.strip():
                        yield ThinkingEvent(thinking_text)

                if event_name == "session_end":
                    outcome = _required_string(event, "outcome")
                    if "exit_code" not in event:
                        raise OspreyProtocolError("session_end requires exit_code")
                    exit_code = _optional_int(event, "exit_code")
                    terminal_outcome = outcome
                    terminal_exit_code = exit_code
                    terminal_structured_output = event.get("structured_output")
                    saw_session_end = True
                    final_cost = _parse_cost(event.get("total_cost_usd"), event_name=event_name, field="total_cost_usd")
                    if final_cost is not None:
                        total_cost = final_cost
                    yield CostEvent(
                        cost_usd=total_cost,
                        input_tokens=_optional_non_negative_int(event, "total_prompt_tokens"),
                        output_tokens=_optional_non_negative_int(event, "total_completion_tokens"),
                        cached_tokens=_optional_non_negative_int(event, "total_cached_tokens"),
                        cache_creation_tokens=_optional_non_negative_int(event, "total_cache_write_tokens"),
                        reasoning_tokens=_optional_non_negative_int(event, "total_thinking_tokens"),
                        model_name=session_model,
                        provider_name=provider,
                        measurement_source="session",
                        cost_source="reported" if total_cost is not None else None,
                    )
                    yield ResultEvent(
                        structured_output=terminal_structured_output,
                        continuation=(
                            ContinuationToken(
                                backend="osprey",
                                data={
                                    "session_id": session_id,
                                    "provider": provider,
                                    "model": session_model,
                                    "outcome": terminal_outcome,
                                    "exit_code": terminal_exit_code,
                                },
                            )
                            if terminal_outcome in _SUCCESS_OUTCOMES
                            else None
                        ),
                        model_name=session_model,
                        provider_name=provider,
                        session_id=session_id,
                        finish_reason=terminal_outcome,
                        duration_ms=_optional_non_negative_int(event, "session_wallclock_ms"),
                    )
                    continue

                if event_name == "session_start":
                    raise OspreyProtocolError("duplicate session_start event")
                if event_name == "text_delta":
                    content = event.get("content")
                    if not isinstance(content, str):
                        raise OspreyProtocolError("text_delta requires string content")
                    turn_text_emitted = True
                    yield TextEvent(content)
                elif event_name == "thinking_delta":
                    content = event.get("content")
                    if not isinstance(content, str):
                        raise OspreyProtocolError("thinking_delta requires string content")
                    thinking_parts.append(content)
                elif event_name == "tool_call":
                    call_id = _required_string(event, "tool_call_id")
                    tool_name = _required_string(event, "tool_name")
                    arguments = event.get("arguments")
                    if not isinstance(arguments, dict):
                        raise OspreyProtocolError("tool_call requires object arguments")
                    protocol_state.start_tool_call(call_id)
                    yield ToolStartEvent(call_id, tool_name, arguments)
                elif event_name == "tool_result":
                    call_id = _required_string(event, "tool_call_id")
                    _required_string(event, "tool_name")
                    status = _required_string(event, "status")
                    if status not in {"success", "error"}:
                        raise OspreyProtocolError(f"tool_result has invalid status {status!r}")
                    content = event.get("content")
                    if not isinstance(content, str):
                        raise OspreyProtocolError("tool_result requires string content")
                    duration_ms = _required_int(event, "duration_ms")
                    protocol_state.finish_tool_call(call_id)
                    yield ToolResultEvent(call_id, content, status == "error", duration_ms=duration_ms, status=status)
                elif event_name == "tool_update":
                    _required_string(event, "tool_call_id")
                    if not isinstance(event.get("content"), str):
                        raise OspreyProtocolError("tool_update requires string content")
                elif event_name == "turn_start":
                    turn_id = _required_string(event, "turn_id")
                    turn_started_at = _required_string(event, "timestamp")
                    protocol_state.start_turn(turn_id)
                    turn_text_emitted = False
                elif event_name == "turn_end":
                    turn_id = _required_string(event, "turn_id")
                    protocol_state.end_turn(turn_id)
                    usage_reported = event.get("usage_reported")
                    if not isinstance(usage_reported, bool):
                        raise OspreyProtocolError("turn_end requires boolean usage_reported")
                    if usage_reported:
                        _required_non_negative_int(event, "duration_ms")
                    else:
                        _optional_non_negative_int(event, "duration_ms")
                    if usage_reported:
                        prompt_tokens = _required_non_negative_int(event, "prompt_tokens")
                        completion_tokens = _required_non_negative_int(event, "completion_tokens")
                        cached_tokens = _optional_non_negative_int(event, "cached_tokens")
                        reasoning_tokens = _optional_non_negative_int(event, "thinking_tokens")
                        cost = _parse_cost(event.get("cost_usd"), event_name=event_name)
                        turn_model = event.get("model")
                        if turn_model is not None and not isinstance(turn_model, str):
                            raise OspreyProtocolError("turn_end model must be string or null")
                        if cost is not None:
                            total_cost = (total_cost or 0.0) + cost
                        yield MetricsEvent(
                            message_id=turn_id,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cached_tokens=cached_tokens,
                            cost_usd=cost,
                            reasoning_tokens=reasoning_tokens,
                            model_name=turn_model or session_model,
                            provider_name=provider,
                            cache_creation_tokens=_optional_non_negative_int(event, "cache_write_tokens"),
                            duration_ms=_required_non_negative_int(event, "duration_ms"),
                            started_at=turn_started_at,
                            measurement_source="turn_end",
                        )
                    # Allow native per-turn model overrides. Provider remains session-scoped;
                    # a session outcome is not a model finish reason.
                    turn_model = event.get("model")
                    yield TurnEndEvent(
                        message_id=turn_id,
                        model_name=(turn_model if isinstance(turn_model, str) and turn_model else session_model),
                        provider_name=provider,
                        model_source="native",
                        provider_source="native",
                    )
                    turn_started_at = None
                elif event_name == "message_end":
                    messages = event.get("messages")
                    if not isinstance(messages, list):
                        raise OspreyProtocolError("message_end requires an array messages")
                    if not turn_text_emitted:
                        for message in messages:
                            if not isinstance(message, dict) or message.get("type") != "result":
                                continue
                            data = message.get("data")
                            if not isinstance(data, dict) or not isinstance(data.get("content"), str):
                                raise OspreyProtocolError("message_end result requires string data.content")
                            turn_text_emitted = True
                            yield TextEvent(data["content"])
                elif event_name == "failed":
                    failed_message = _required_string(event, "message")
                elif event_name == "cancelled":
                    failed_message = "cancelled"
                elif event_name == "loop_terminated":
                    failed_message = _required_string(event, "reason")
                elif event_name in _KNOWN_IGNORED_EVENTS:
                    continue
                else:
                    raise OspreyProtocolError(f"unknown Osprey JSONL event {event_name!r}")

            returncode = await transport.wait()
            # Close descendant-held stderr before joining its drain, or EOF may never
            # arrive. This also completes diagnostics; finally's teardown is idempotent.
            await teardown(transport, self._transports)
            # No retryable= kwarg: osprey's PROCESS_EXIT is not retryable.
            if returncode != 0:
                raise OspreyError(_osprey_process_exit_message(stderr_lines, returncode), category="PROCESS_EXIT")
            if not saw_header:
                raise OspreyProtocolError("Osprey produced no protocol header")
            if not saw_session_start:
                raise OspreyProtocolError("Osprey produced no session_start event")
            if not saw_session_end:
                raise OspreyProtocolError("Osprey stream ended without session_end")
            if terminal_exit_code is not None and terminal_exit_code != returncode:
                raise OspreyProtocolError("session_end exit_code does not match subprocess return code")
            if terminal_outcome not in _SUCCESS_OUTCOMES:
                detail = failed_message or f"exit_code={terminal_exit_code!r}"
                raise OspreyTerminalError(terminal_outcome or "unknown", detail)
            if protocol_state.active_turn_id is not None:
                raise OspreyProtocolError("successful session_end has an active turn")
            if protocol_state.pending_tool_calls:
                raise OspreyProtocolError("successful session_end has pending tool calls")
        finally:
            if transport is not None:
                await teardown(transport, self._transports)
            if schema_path is not None:
                Path(schema_path).unlink(missing_ok=True)

    async def cancel(self) -> None:
        """Terminate all active Osprey process groups and reap their pipes."""
        await CliTransport.cancel_all(self._transports)
