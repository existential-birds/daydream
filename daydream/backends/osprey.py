"""Osprey headless JSONL backend for daydream.

This module is deliberately a thin subprocess adapter.  Osprey owns the agent
loop, tool-search catalog, MCP lifecycle, policy checks, hooks, caps, and
provider telemetry; daydream only translates the versioned stream into its
existing backend event union.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import tempfile
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field
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
    resolve_fanout_concurrency,
)
from daydream.backends._subprocess import (
    cancel_processes,
    readline_with_idle_timeout,
    stream_idle_timeout_s,
    terminate_process,
)
from daydream.trajectory import redact_text

logger = logging.getLogger(__name__)

_OSPREY_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024
_JSONL_PROTOCOL_VERSION = 2
_MAX_DIAGNOSTIC_LINES = 10
_MAX_DIAGNOSTIC_LINE_CHARS = 500

_SUCCESS_OUTCOMES = frozenset({"completed", "terminal_tool"})
_KNOWN_IGNORED_EVENTS = frozenset(
    {
        # Osprey's stream contains these provider-neutral lifecycle and
        # telemetry records.  Daydream has no corresponding event class, so
        # they are retained in the producer/ATIF path and not reinterpreted.
        "completion_gate",
        "finalization_record",
        "verification_checkpoint",
        "context_compaction",
        "driver_retry",
        "message_end",
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
    value = event.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise OspreyProtocolError(f"event {event.get('event')!r} has invalid integer {key!r}")
    assert isinstance(value, int)
    return value


def _parse_cost(value: Any, *, event_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OspreyProtocolError(f"event {event_name!r} has non-string cost_usd")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise OspreyProtocolError(f"event {event_name!r} has invalid cost_usd") from exc
    if not math.isfinite(parsed):
        raise OspreyProtocolError(f"event {event_name!r} has non-finite cost_usd")
    return parsed


def _bounded_diagnostics(lines: Iterable[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        if len(cleaned) >= _MAX_DIAGNOSTIC_LINES:
            break
        cleaned.append(redact_text(line)[:_MAX_DIAGNOSTIC_LINE_CHARS])
    if not cleaned:
        return "no diagnostic output captured"
    return "\n".join(cleaned)


@dataclass(frozen=True)
class _OspreyCommandOptions:
    """Per-invocation inputs to the verified Osprey argv surface."""

    prompt: str
    output_schema_path: str | Path | None
    continuation: ContinuationToken | None
    max_turns: int | None
    read_only: bool
    persist_session: bool
    tool_search_mode: str | None


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
            raise OspreyProtocolError(
                f"turn_end {turn_id!r} does not match active turn {self.active_turn_id!r}"
            )
        self.active_turn_id = None

    def start_tool_call(self, call_id: str) -> None:
        if call_id in self.pending_tool_calls:
            raise OspreyProtocolError(f"duplicate pending tool_call_id {call_id!r}")
        self.pending_tool_calls.add(call_id)

    def finish_tool_call(self, call_id: str) -> None:
        # Preserve unmatched results: the trajectory recorder has an explicit
        # bucket for them. Matching calls are retired.
        self.pending_tool_calls.discard(call_id)


class OspreyBackend:
    """Translate ``osprey agent --events-jsonl`` into daydream events."""

    name = "osprey"
    concise_fix_prompts = False

    def __init__(
        self,
        model: str | None = None,
        *,
        cwd: Path | None = None,
        reasoning_effort: str | None = None,
        osprey_binary: str | None = None,
        persona: str | None = None,
        toolset: str | None = None,
        temperature: float | None = None,
        approval: str | None = None,
        sandbox: bool = False,
        allowed_roots: Iterable[Path | str] = (),
        atif_output: Path | None = None,
        atif_system_prompt_plaintext: bool = False,
        immutable_runtime_surface: bool = False,
        turn_timeout: int | None = None,
        stream_idle_timeout_secs: int | None = None,
        streaming_timeout_secs: int | None = None,
        empty_completion_threshold: int | None = None,
        driver_max_retries: int | None = None,
        compress_context: bool | None = None,
        compress_min_bytes: int | None = None,
        tool_result_cap: int | None = None,
        tool_result_head: int | None = None,
        tool_result_tail: int | None = None,
        tool_result_max_lines: int | None = None,
        tool_result_raw_dir: Path | None = None,
        retry_failure_threshold: int | None = None,
        no_progress_family_threshold: int | None = None,
        no_progress_family_window: int | None = None,
        no_progress_artifact_threshold: int | None = None,
        no_progress_suppression_window: int | None = None,
        vars: Iterable[tuple[str, str]] = (),
        max_subagents: int | None = None,
        llm_rpm: int | None = None,
        effort: str | None = None,
        ultracode: bool = False,
        tool_search_mode: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        osprey_home: Path | None = None,
    ) -> None:
        if provider is not None:
            raise OspreyUnsupportedOption(
                "provider",
                "the current CLI resolves providers from Osprey configuration/environment; it has no provider flag",
            )
        if base_url is not None:
            raise OspreyUnsupportedOption(
                "base_url",
                "the current CLI resolves custom endpoints from Osprey "
                "configuration/environment; it has no base-url flag",
            )
        if tool_search_mode is not None and tool_search_mode not in {"auto", "on", "off"}:
            raise OspreyUnsupportedOption("tool_search_mode", f"invalid mode {tool_search_mode!r}")

        self._model_override = model
        self.model = model or "osprey"
        self.reasoning_effort = reasoning_effort
        self.osprey_binary = osprey_binary or os.environ.get("OSPREY_BINARY", "osprey")
        self.cwd = cwd
        self.persona = persona
        self.toolset = toolset
        self.temperature = temperature
        self.approval = approval
        self.sandbox = sandbox
        self.allowed_roots = tuple(str(root) for root in allowed_roots)
        self.atif_output = atif_output
        self.atif_system_prompt_plaintext = atif_system_prompt_plaintext
        self.immutable_runtime_surface = immutable_runtime_surface
        self.turn_timeout = turn_timeout
        self.stream_idle_timeout_secs = stream_idle_timeout_secs
        self.streaming_timeout_secs = streaming_timeout_secs
        self.empty_completion_threshold = empty_completion_threshold
        self.driver_max_retries = driver_max_retries
        self.compress_context = compress_context
        self.compress_min_bytes = compress_min_bytes
        self.tool_result_cap = tool_result_cap
        self.tool_result_head = tool_result_head
        self.tool_result_tail = tool_result_tail
        self.tool_result_max_lines = tool_result_max_lines
        self.tool_result_raw_dir = tool_result_raw_dir
        self.retry_failure_threshold = retry_failure_threshold
        self.no_progress_family_threshold = no_progress_family_threshold
        self.no_progress_family_window = no_progress_family_window
        self.no_progress_artifact_threshold = no_progress_artifact_threshold
        self.no_progress_suppression_window = no_progress_suppression_window
        self.vars = tuple(vars)
        self.max_subagents = max_subagents
        self.llm_rpm = llm_rpm
        self.effort = effort
        self.ultracode = ultracode
        self.tool_search_mode = tool_search_mode
        self.osprey_home = osprey_home
        self.fanout_concurrency = resolve_fanout_concurrency(
            "DAYDREAM_OSPREY_FANOUT_CONCURRENCY", 4
        )
        self._processes: list[asyncio.subprocess.Process] = []

    def build_command(
        self,
        prompt: str,
        *,
        output_schema_path: str | Path | None = None,
        continuation: ContinuationToken | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
        tool_search_mode: str | None = None,
    ) -> list[str]:
        return self._build_command(
            _OspreyCommandOptions(
                prompt=prompt,
                output_schema_path=output_schema_path,
                continuation=continuation,
                max_turns=max_turns,
                read_only=read_only,
                persist_session=persist_session,
                tool_search_mode=tool_search_mode,
            )
        )

    def _build_command(self, options: _OspreyCommandOptions) -> list[str]:
        """Build only flags verified against the current Osprey CLI source."""
        prompt = options.prompt
        output_schema_path = options.output_schema_path
        continuation = options.continuation
        max_turns = options.max_turns
        read_only = options.read_only
        persist_session = options.persist_session
        tool_search_mode = options.tool_search_mode
        selected_tool_search = (
            tool_search_mode if tool_search_mode is not None else self.tool_search_mode
        )
        if selected_tool_search is not None:
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

        args = [self.osprey_binary, "agent", "--events-jsonl"]
        if self.persona:
            args.extend(["--persona", self.persona])
        if self.toolset:
            args.extend(["--toolset", self.toolset])
        if self._model_override:
            args.extend(["--model", self._model_override])
        if self.temperature is not None:
            args.extend(["--temperature", str(self.temperature)])
        if self.atif_output is not None:
            args.extend(["--atif-output", str(self.atif_output)])
        if self.atif_system_prompt_plaintext:
            args.append("--atif-system-prompt-plaintext")
        if self.immutable_runtime_surface:
            args.append("--immutable-runtime-surface")
        if max_turns is not None:
            args.extend(["--max-turns", str(max_turns)])
        if self.turn_timeout is not None:
            args.extend(["--turn-timeout", str(self.turn_timeout)])
        if self.stream_idle_timeout_secs is not None:
            args.extend(["--stream-idle-timeout-secs", str(self.stream_idle_timeout_secs)])
        if self.streaming_timeout_secs is not None:
            args.extend(["--streaming-timeout-secs", str(self.streaming_timeout_secs)])
        if self.empty_completion_threshold is not None:
            args.extend(["--empty-completion-threshold", str(self.empty_completion_threshold)])
        if self.driver_max_retries is not None:
            args.extend(["--driver-max-retries", str(self.driver_max_retries)])

        if read_only:
            args.append("--read-only")
        if self.approval is not None:
            if self.approval in {"on-request", "on-failure", "unless-trusted"}:
                raise OspreyUnsupportedOption(
                    "approval",
                    f"{self.approval!r} requires an interactive approver and headless mode rejects it",
                )
            if self.approval != "deny-untrusted":
                raise OspreyUnsupportedOption("approval", f"unsupported headless value {self.approval!r}")
            args.extend(["--approval", self.approval])
        if self.sandbox:
            args.append("--sandbox")
        for root in self.allowed_roots:
            args.extend(["--allowed-root", root])

        def add_value(flag: str, value: object | None) -> None:
            if value is not None:
                args.extend([flag, str(value)])

        if self.compress_context is not None:
            args.append(f"--compress-context={str(self.compress_context).lower()}")
        add_value("--compress-min-bytes", self.compress_min_bytes)
        add_value("--tool-result-cap", self.tool_result_cap)
        add_value("--tool-result-head", self.tool_result_head)
        add_value("--tool-result-tail", self.tool_result_tail)
        add_value("--tool-result-max-lines", self.tool_result_max_lines)
        add_value("--tool-result-raw-dir", self.tool_result_raw_dir)
        add_value("--retry-failure-threshold", self.retry_failure_threshold)
        add_value("--no-progress-family-threshold", self.no_progress_family_threshold)
        add_value("--no-progress-family-window", self.no_progress_family_window)
        add_value("--no-progress-artifact-threshold", self.no_progress_artifact_threshold)
        add_value("--no-progress-suppression-window", self.no_progress_suppression_window)
        for key, value in self.vars:
            args.extend(["--var", f"{key}={value}"])
        if continuation is not None:
            data = continuation.data
            session_id = data.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise OspreyProtocolError("osprey continuation token requires session_id")
            mode = data.get("mode", "resume")
            if mode == "fork":
                args.extend(["--fork-from", session_id])
            elif mode == "resume":
                args.extend(["--resume", session_id])
            else:
                raise OspreyProtocolError(f"unknown osprey continuation mode {mode!r}")
        if output_schema_path is not None:
            args.extend(["--output-schema", str(output_schema_path)])
        add_value("--max-subagents", self.max_subagents)
        add_value("--llm-rpm", self.llm_rpm)
        add_value("--effort", self.effort or self.reasoning_effort)
        if self.ultracode:
            args.append("--ultracode")
        args.append(prompt)
        return args

    @staticmethod
    def _write_temp_schema(schema: dict[str, Any]) -> str:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", prefix="daydream-osprey-schema-", delete=False
        )
        completed = False
        try:
            json.dump(schema, handle, separators=(",", ":"))
            handle.write("\n")
            completed = True
            return handle.name
        finally:
            try:
                handle.close()
            finally:
                if not completed:
                    # delete=False lets Osprey reopen this path, so remove the file
                    # ourselves if serialization or writing does not complete.
                    Path(handle.name).unlink(missing_ok=True)

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
        schema_path: str | None = None
        if output_schema is not None:
            schema_path = self._write_temp_schema(output_schema)

        proc: asyncio.subprocess.Process | None = None
        session_id: str | None = None
        session_model: str | None = None
        provider: str | None = None
        saw_header = False
        saw_session_start = False
        saw_session_end = False
        failed_message: str | None = None
        turn_durations_ms: list[int] = []
        total_cost: float | None = None
        saw_metric_cost = False
        non_json_lines: list[str] = []
        terminal_outcome: str | None = None
        terminal_exit_code: int | None = None
        terminal_structured_output: Any = None
        turn_text_emitted = False
        protocol_state = _OspreyProtocolState()

        try:
            command = self.build_command(
                prompt,
                output_schema_path=schema_path,
                continuation=continuation,
                max_turns=max_turns,
                read_only=read_only,
                persist_session=persist_session,
            )
            child_env = os.environ.copy()
            if self.osprey_home is not None:
                child_env["OSPREY_HOME"] = str(self.osprey_home)
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=_OSPREY_STDOUT_LIMIT_BYTES,
                env=child_env,
                start_new_session=True,
            )
            self._processes.append(proc)
            while True:
                if proc.stdout is None:
                    break
                line = await readline_with_idle_timeout(
                    proc.stdout, cli="osprey", timeout_s=stream_idle_timeout_s()
                )
                if not line:
                    break
                raw = line.decode(errors="replace").strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError as exc:
                    if len(non_json_lines) < _MAX_DIAGNOSTIC_LINES:
                        non_json_lines.append(raw)
                    raise OspreyProtocolError(
                        "Osprey emitted a non-JSON line in JSONL mode: "
                        f"{_bounded_diagnostics(non_json_lines)}"
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
                    _required_string(event, "started_at")
                    session_model = _required_string(event, "model")
                    provider = _required_string(event, "provider")
                    saw_session_start = True
                    continue
                if saw_session_end:
                    raise OspreyProtocolError("JSONL event appeared after session_end")
                if session_id is None:
                    raise OspreyProtocolError("session_start did not provide a session_id")

                if event_name == "session_end":
                    outcome = _required_string(event, "outcome")
                    if "exit_code" not in event:
                        raise OspreyProtocolError("session_end requires exit_code")
                    exit_code = _optional_int(event, "exit_code")
                    terminal_outcome = outcome
                    terminal_exit_code = exit_code
                    terminal_structured_output = event.get("structured_output")
                    saw_session_end = True
                    final_cost = _parse_cost(event.get("total_cost_usd"), event_name=event_name)
                    if final_cost is not None and not saw_metric_cost:
                        total_cost = final_cost
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
                    yield ThinkingEvent(content)
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
                    _required_int(event, "duration_ms")
                    protocol_state.finish_tool_call(call_id)
                    yield ToolResultEvent(call_id, content, status == "error")
                elif event_name == "tool_update":
                    _required_string(event, "tool_call_id")
                    if not isinstance(event.get("content"), str):
                        raise OspreyProtocolError("tool_update requires string content")
                elif event_name == "turn_start":
                    turn_id = _required_string(event, "turn_id")
                    _required_string(event, "timestamp")
                    protocol_state.start_turn(turn_id)
                    turn_text_emitted = False
                elif event_name == "turn_end":
                    turn_id = _required_string(event, "turn_id")
                    protocol_state.end_turn(turn_id)
                    usage_reported = event.get("usage_reported")
                    if not isinstance(usage_reported, bool):
                        raise OspreyProtocolError("turn_end requires boolean usage_reported")
                    duration = (
                        _required_int(event, "duration_ms")
                        if usage_reported
                        else _optional_int(event, "duration_ms")
                    )
                    if duration is not None:
                        turn_durations_ms.append(duration)
                    if usage_reported:
                        prompt_tokens = _required_int(event, "prompt_tokens")
                        completion_tokens = _required_int(event, "completion_tokens")
                        cached_tokens = _optional_int(event, "cached_tokens")
                        reasoning_tokens = _optional_int(event, "thinking_tokens")
                        if reasoning_tokens is not None and reasoning_tokens < 0:
                            raise OspreyProtocolError(
                                "turn_end thinking_tokens must be non-negative"
                            )
                        cost = _parse_cost(event.get("cost_usd"), event_name=event_name)
                        turn_model = event.get("model")
                        if turn_model is not None and not isinstance(turn_model, str):
                            raise OspreyProtocolError("turn_end model must be string or null")
                        if cost is not None:
                            total_cost = (total_cost or 0.0) + cost
                            saw_metric_cost = True
                        yield MetricsEvent(
                            message_id=turn_id,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cached_tokens=cached_tokens,
                            cost_usd=cost,
                            reasoning_tokens=reasoning_tokens,
                            model_name=turn_model or session_model,
                        )
                    yield TurnEndEvent(message_id=turn_id)
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
                                raise OspreyProtocolError(
                                    "message_end result requires string data.content"
                                )
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

            await proc.wait()
            returncode = proc.returncode
            if returncode not in (None, 0):
                raise OspreyError(
                    f"Osprey CLI exited with return code {returncode}: {_bounded_diagnostics(non_json_lines)}",
                    category="PROCESS_EXIT",
                )
            if not saw_header:
                raise OspreyProtocolError("Osprey produced no protocol header")
            if not saw_session_start:
                raise OspreyProtocolError("Osprey produced no session_start event")
            if not saw_session_end:
                raise OspreyProtocolError("Osprey stream ended without session_end")
            if terminal_exit_code is not None and terminal_exit_code != returncode:
                raise OspreyProtocolError(
                    "session_end exit_code does not match subprocess return code"
                )
            if terminal_outcome not in _SUCCESS_OUTCOMES:
                detail = failed_message or f"exit_code={terminal_exit_code!r}"
                raise OspreyTerminalError(terminal_outcome or "unknown", detail)
            if protocol_state.active_turn_id is not None:
                raise OspreyProtocolError("successful session_end has an active turn")
            if protocol_state.pending_tool_calls:
                raise OspreyProtocolError("successful session_end has pending tool calls")
            if total_cost is not None and not saw_metric_cost:
                yield CostEvent(
                    cost_usd=total_cost,
                    input_tokens=None,
                    output_tokens=None,
                    cached_tokens=None,
                    model_name=session_model,
                )
            yield ResultEvent(
                structured_output=terminal_structured_output,
                continuation=ContinuationToken(
                    backend="osprey",
                    data={
                        "session_id": session_id,
                        "provider": provider,
                        "model": session_model,
                        "outcome": terminal_outcome,
                        "exit_code": terminal_exit_code,
                    },
                ),
            )
        finally:
            if proc is not None:
                await terminate_process(proc)
            self._processes = [active for active in self._processes if active is not proc]
            if schema_path is not None:
                Path(schema_path).unlink(missing_ok=True)

    async def cancel(self) -> None:
        """Terminate all active Osprey process groups and reap their pipes."""
        await cancel_processes(self._processes)
