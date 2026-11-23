"""Antigravity CLI subprocess backend for daydream.

Spawns one ``agy`` process per invocation in headless ``stream-json`` mode, writes the
prompt as one NDJSON user line on stdin, and reads NDJSON events from stdout. Every
invocation runs a Daydream-owned custom agent behind a deny-by-default hook from a
private per-run directory (see ``_antigravity_run``); the default agent never runs.
The wire contract is the one recorded on ``agy`` 1.5.15: shapes outside it are
reported as parser coverage and never count as success.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, deque
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, aclosing
from functools import partial
from pathlib import Path
from typing import Any

from daydream.backends import (
    AgentEvent,
    AntigravityRequestConfig,
    BackendExecutionInput,
    ContinuationToken,
    DiagnosticEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    fanout_concurrency_from_environment,
)
from daydream.backends._antigravity_run import MCP_PERMISSION_SETTINGS, RunDirectory, create_run_directory
from daydream.backends._mcp_host import HOST_SERVER, McpHostEndpoint, exempt_loopback_from_proxy
from daydream.backends._subprocess import StreamStalledError, stream_idle_timeout_s
from daydream.backends._transport import CliTransport, process_exit_message, reap, teardown
from daydream.backends.lifecycle import current_backend_lifecycle
from daydream.host_tools import HostToolBinding, UnsubmittedTurn
from daydream.redaction import redact_structured_text

_AGY = "agy"
_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024
#: Oldest build whose headless contract Daydream has verified; older builds are refused.
VERSION_FLOOR = (1, 5, 15)
#: Newest verified build; newer ones proceed with a diagnostic (agy updates itself in the background).
VERIFIED_VERSION = (1, 5, 15)
_VERSION_LINE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_VERSION_TIMEOUT_S = 30.0
_PROCESS_REAP_TIMEOUT_S = 5.0
_CLEANUP_TIMEOUT_S = 10.0
# The child runs with this directory as HOME: a dedicated agy home with its own sign-in.
_HOME_ENV = "DAYDREAM_ANTIGRAVITY_HOME"
_STDERR_ERROR_PREFIX = "AGY_ERROR:"
_EXCERPT_MAX_LINES = 20
_EXCERPT_MAX_CHARS = 256
_COMPLETED_TOOL_STATES = frozenset({"DONE", "ERROR"})

_logger = logging.getLogger(__name__)


def _version_text(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def _bounded(value: Any) -> str:
    """Redact one native line or label before applying the excerpt cap."""
    return redact_structured_text(str(value))[:_EXCERPT_MAX_CHARS]


def _token_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _milliseconds(seconds: Any) -> float | None:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
        return None
    return round(float(seconds) * 1000.0, 3)  # agy reports microseconds


class AntigravityError(Exception):
    """Antigravity failure with a retry category and, when agy states it, a retryable flag."""

    def __init__(self, message: str, *, category: str | None = None, retryable: bool | None = None):
        super().__init__(message)
        self.category = category
        if retryable is not None:
            self.retryable = retryable


class AntigravityBackend:
    """Backend that wraps the Antigravity CLI (``agy``) in headless stream-json mode.

    ``persist_session=False`` cannot be honoured: agy always stores the conversation in
    its home. Such a call neither resumes nor returns a continuation.
    """

    supports_host_tools = True

    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str | None = None,
        *,
        execution_input: BackendExecutionInput | None = None,
    ):
        # Without a model agy uses its configured one; the generic label is replaced by init.model.
        self.model = model or "antigravity"
        self._model = model
        self.reasoning_effort = reasoning_effort
        self._execution_input = execution_input
        if execution_input is not None:
            self.fanout_concurrency = execution_input.fanout_concurrency
            self.retry_policy = execution_input.retry_policy
        else:
            # Subscription quotas are narrow; the shared environment setting still overrides this.
            self.fanout_concurrency = fanout_concurrency_from_environment(os.environ, 2)
        self._transports: list[CliTransport] = []
        self._version: tuple[int, int, int] | None = None
        self._version_lock = asyncio.Lock()

    def bind_review_context(self, original_source: Path) -> dict[str, object]:
        """Opaque identity of this adapter's native selection, without credentials or environment."""
        del original_source
        selection = json.dumps([self._model, self.reasoning_effort]).encode()
        return {"configuration_identity": hashlib.sha256(selection).hexdigest()}

    def _child_environment(self) -> dict[str, str]:
        """The run-owned environment, with the dedicated agy home as HOME when one is selected."""
        environment = (self._execution_input.child_environment() if self._execution_input is not None
                       else os.environ.copy())
        if home := environment.get(_HOME_ENV):
            environment["HOME"] = home
        return environment

    async def _checked_version(self, environment: dict[str, str]) -> tuple[int, int, int]:
        """Run ``agy --version`` once per backend instance and refuse builds below the floor."""
        async with self._version_lock:
            if self._version is None:
                self._version = await self._observe_version(environment)
            return self._version

    async def _observe_version(self, environment: dict[str, str]) -> tuple[int, int, int]:
        refusal = (f"Daydream requires agy {_version_text(VERSION_FLOOR)} or newer; install or update the "
                   "Antigravity CLI and make sure `agy` is on PATH")
        # agy prints the bare version on stdout; its stderr (update notices) is not part of the answer.
        transport = CliTransport(_AGY, [_AGY, "--version"], stderr_sink=lambda _line: None,
                                 limit=64 * 1024, env=environment)
        self._transports.append(transport)
        try:
            try:
                await transport.start()
            except OSError as exc:
                raise AntigravityError(f"agy could not be started ({type(exc).__name__}). {refusal}",
                                       category="AUTH_CONFIG", retryable=False) from exc
            lines = [line async for line in transport.lines(lambda: _VERSION_TIMEOUT_S) if line]
            try:
                returncode = await reap(transport, timeout_s=_PROCESS_REAP_TIMEOUT_S)
            except TimeoutError:
                returncode = None
        finally:
            await teardown(transport, self._transports, cleanup_timeout_s=_CLEANUP_TIMEOUT_S)
        match = _VERSION_LINE.fullmatch(lines[0]) if len(lines) == 1 else None
        if returncode != 0 or match is None:
            raise AntigravityError(f"`agy --version` did not report a bare X.Y.Z version. {refusal}",
                                   category="AUTH_CONFIG", retryable=False)
        version = (int(match[1]), int(match[2]), int(match[3]))
        if version < VERSION_FLOOR:
            raise AntigravityError(f"agy {_version_text(version)} is older than the verified floor. {refusal}",
                                   category="AUTH_CONFIG", retryable=False)
        return version

    async def execute(
        self, cwd: Path, prompt: str, output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None, agents: dict[str, Any] | None = None,
        read_only: bool = False, persist_session: bool = True,
        review_instructions: str | None = None, host_tools: HostToolBinding | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield Antigravity events; the process is reaped before its host endpoint closes.

        host_tools serves the binding's tools and terminal ``structured_output`` from a fresh
        loopback MCP server for this attempt, registered only in the per-run directory;
        ``review_instructions`` become the custom agent's system prompt section. The result
        is then the host-accepted submission, never prose or agy's own schema output.
        """
        lifecycle = current_backend_lifecycle()
        async with AsyncExitStack() as attempt:
            if host_tools is not None:
                if continuation is not None or agents or not read_only or output_schema is None:
                    raise ValueError("Antigravity host tools require a fresh read-only invocation with a terminal "
                                     "schema, without continuation or agents")
                host_tools.configure_schema(output_schema)
                if lifecycle is not None:
                    binding, collector = host_tools, lifecycle
                    # Recorded after the attempt's endpoint closes, on every exit path.
                    attempt.callback(lambda: collector.host_tools_closed(
                        native_settled=binding.native_settled, cleanup_complete=binding.cleanup_complete,
                        calls=binding.calls, rejections=binding.rejections))
            elif review_instructions is not None:
                raise ValueError("Antigravity review instructions are served only with host tools")
            if agents:
                raise NotImplementedError(
                    "Antigravity backend does not support exploration subagents; use --backend claude for exploration."
                )
            environment = self._child_environment()
            version = await self._checked_version(environment)
            host: McpHostEndpoint | None = None
            if host_tools is not None:
                host = await attempt.enter_async_context(host_tools.attempt(partial(
                    McpHostEndpoint, native_owner="Antigravity")))
            stream = await attempt.enter_async_context(aclosing(self._execute(
                cwd, prompt, output_schema, continuation, read_only, persist_session, review_instructions,
                host, environment, version)))
            async for event in stream:
                if lifecycle is not None:
                    lifecycle.adapter_yield()
                yield event

    async def _execute(
        self, cwd: Path, prompt: str, output_schema: dict[str, Any] | None,
        continuation: ContinuationToken | None, read_only: bool, persist_session: bool,
        review_instructions: str | None, host: McpHostEndpoint | None, environment: dict[str, str],
        version: tuple[int, int, int],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Drive one agy process and settle its result only after SUCCESS, EOF, zero exit and cleanup."""
        lifecycle = current_backend_lifecycle()
        idle_timeout_s = (self._execution_input.stream_idle_timeout_s if self._execution_input is not None
                          else stream_idle_timeout_s())
        if lifecycle is not None:
            lifecycle.activate(response_idle_timeout_s=idle_timeout_s, tool_idle_timeout_s=idle_timeout_s,
                               backend="antigravity")
        resumed: str | None = None
        if persist_session and continuation is not None and continuation.backend == "antigravity":
            conversation = continuation.data.get("conversation_id")
            resumed = conversation if isinstance(conversation, str) and conversation else None

        stderr_excerpt: deque[str] = deque(maxlen=_EXCERPT_MAX_LINES)
        stderr_error: dict[str, Any] = {}

        def _keep_stderr(line: str) -> None:
            """Keep agy's stderr for the exit excerpt and its structured ``AGY_ERROR`` line for classification."""
            if line.startswith(_STDERR_ERROR_PREFIX):
                try:
                    parsed = json.loads(line[len(_STDERR_ERROR_PREFIX):])
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    stderr_error.clear()
                    stderr_error.update(parsed)
            stderr_excerpt.append(_bounded(line))

        coverage: Counter[str] = Counter()  # fixed reasons for shapes outside the recorded contract
        started_tools: set[str] = set()
        metered_steps: set[int] = set()
        conversation_id: str | None = None
        native_model: str | None = None
        initialized = False
        pending_turn: str | None = None  # a finished response whose requested tool steps may still follow
        turn_text = False
        terminal: dict[str, Any] | None = None
        cleaned_up = False
        run: RunDirectory | None = None
        transport: CliTransport | None = None

        def _turn_end(message_id: str) -> TurnEndEvent:
            return TurnEndEvent(message_id=message_id, model_name=native_model,
                                message_id_source="host_generated" if message_id else None,
                                model_source="native" if native_model else None)

        async def _teardown_once(started: CliTransport) -> None:
            nonlocal cleaned_up
            cleaned_up = True
            if lifecycle is not None:
                lifecycle.teardown_started()
            try:
                await teardown(started, self._transports, cleanup_timeout_s=_CLEANUP_TIMEOUT_S)
            except TimeoutError as exc:
                if lifecycle is not None:
                    lifecycle.expiry("cleanup")
                raise AntigravityError(f"agy process cleanup did not complete within {_CLEANUP_TIMEOUT_S:g}s",
                                       category="TIMEOUT", retryable=False) from exc
            if lifecycle is not None:
                lifecycle.teardown_completed()

        try:
            run = create_run_directory(
                read_only=read_only, review_instructions=review_instructions,
                mcp_url=host.url if host is not None else None, mcp_token=host.token if host is not None else None)
            args = [_AGY, "--input-format", "stream-json", "--output-format", "stream-json"]
            if self._model:
                args.extend(["--model", self._model])
            if self.reasoning_effort:
                args.extend(["--effort", self.reasoning_effort])
            args.extend(["--add-dir", str(run.path), "--agent", run.agent])
            native_schema = output_schema is not None and host is None
            if host is not None:
                # The only permission rule added: this attempt's own server. Its URL and bearer
                # live in the per-run directory, never in argv, the prompt or the child environment.
                args.extend(["--settings", MCP_PERMISSION_SETTINGS])
                exempt_loopback_from_proxy(environment)
            elif output_schema is not None:
                args.extend(["--json-schema", json.dumps(output_schema, separators=(",", ":"))])
            if resumed is not None:
                args.extend(["--conversation", resumed])
            if not read_only:
                # The hook still receives, and can deny, every call under this flag.
                args.append("--dangerously-skip-permissions")

            if lifecycle is not None:
                lifecycle.request_prepared()
            yield RequestEvent(
                prompt=prompt, system_prompt=review_instructions, model_name=self._model,
                output_schema=output_schema, reasoning_effort=self.reasoning_effort, session_id=resumed,
                config=AntigravityRequestConfig(
                    read_only=read_only,
                    persist_session=True,  # agy always stores the conversation in its home
                    continuation_mode="resume" if resumed is not None else "fresh",
                    model_mode="single",
                    tools_count=len(run.tools),
                    native_output_schema=native_schema,
                    host_mcp=host is not None,
                    dedicated_home=_HOME_ENV in environment and bool(environment[_HOME_ENV]),
                ),
                model_source="configured" if self._model else None,
                session_source="configured" if resumed is not None else None,
            )
            newer = version > VERIFIED_VERSION
            yield DiagnosticEvent(
                code="antigravity_cli_version",
                message=(f"agy {_version_text(version)} is newer than the newest verified build "
                         f"{_version_text(VERIFIED_VERSION)}; its headless contract is unverified." if newer
                         else f"agy {_version_text(version)} is a verified build."),
                metadata={"observed": _version_text(version), "verified": _version_text(VERIFIED_VERSION),
                          "newer_than_verified": newer},
            )

            user_line = json.dumps({"event": "user", "message": {"content": prompt}}) + "\n"
            transport = CliTransport(
                _AGY, args, stdin_data=user_line.encode(), stderr_sink=_keep_stderr,
                limit=_STDOUT_LIMIT_BYTES, env=environment, cwd=str(cwd),
                spawn_pending=lifecycle.spawn_pending if lifecycle is not None else None,
                spawn_completed=lifecycle.spawn_completed if lifecycle is not None else None,
                raw_line_received=lifecycle.raw_line_received if lifecycle is not None else None,
                stdout_eof=lifecycle.stdout_eof if lifecycle is not None else None,
                process_exited=lifecycle.process_exited if lifecycle is not None else None,
            )
            self._transports.append(transport)
            await transport.start()

            async for raw_line in transport.lines(lambda: idle_timeout_s):
                if host is not None and host.binding.fatal_error is not None:
                    # A terminal host failure stops this conversation; the process is torn down below.
                    raise host.binding.fatal_error
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    coverage["non_json_line"] += 1
                    continue
                kind = event.get("event") if isinstance(event, dict) else None
                body = event.get(kind) if isinstance(kind, str) else None
                if not isinstance(body, dict):
                    coverage["malformed_event"] += 1
                    continue
                if lifecycle is not None:
                    lifecycle.native_event()

                if terminal is not None:
                    # Nothing after the terminal result changes it.
                    coverage["event_after_result"] += 1

                elif kind == "init":
                    if body.get("agent") != run.agent:
                        raise AntigravityError(
                            "agy did not select Daydream's custom agent (init.agent was "
                            f"{_bounded(body.get('agent'))!r}); the invocation was stopped before any model step",
                            category="TRANSPORT", retryable=False)
                    initialized = True
                    model = body.get("model")
                    native_model = model if isinstance(model, str) and model else None
                    conversation = event.get("conversation_id")
                    conversation_id = conversation if isinstance(conversation, str) and conversation else None

                elif kind == "step_update":
                    if not initialized:
                        raise AntigravityError("agy emitted a step before naming its agent in init; the invocation "
                                               "was stopped", category="TRANSPORT", retryable=False)
                    step_type, state, index = body.get("step_type"), body.get("state"), body.get("step_index")
                    if type(index) is not int:
                        coverage["malformed_step"] += 1
                        continue
                    step_id = f"step-{index}"
                    if step_type in ("agent_response", "tool") and host is not None and not host.ready:
                        raise host.fatal("Antigravity did not list the host tools before model work",
                                         code="native_readiness")
                    if step_type == "agent_response" and pending_turn not in (None, step_id):
                        # The previous response's tool steps are over: its turn closes with them inside.
                        assert pending_turn is not None
                        yield _turn_end(pending_turn)
                        pending_turn, turn_text = None, False
                    usage = body.get("usage")
                    if isinstance(usage, dict) and index not in metered_steps:
                        prompt_tokens = _token_count(usage.get("input_tokens"))
                        completion_tokens = _token_count(usage.get("output_tokens"))
                        if prompt_tokens is None or completion_tokens is None:
                            coverage["malformed_usage"] += 1
                        else:
                            # One charge per step. agy reports no cost; cache reads are part of input.
                            metered_steps.add(index)
                            yield MetricsEvent(
                                message_id=step_id, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                cached_tokens=_token_count(usage.get("cache_read_tokens")), cost_usd=None,
                                reasoning_tokens=_token_count(usage.get("thinking_tokens")),
                                model_name=native_model or self._model,
                                duration_ms=_milliseconds(body.get("duration_seconds")),
                                measurement_source="message_end",
                            )

                    if step_type == "agent_response":
                        delta = body.get("text_delta")
                        if isinstance(delta, str) and delta:
                            turn_text = True
                            yield TextEvent(text=delta)
                        if state == "DONE":
                            pending_turn = step_id
                    elif step_type == "tool":
                        info = body.get("tool_info")
                        info = info if isinstance(info, dict) else {}
                        name = body.get("tool_name") or info.get("name")
                        if not isinstance(name, str) or not name:
                            coverage["malformed_step"] += 1
                            continue
                        parameters = info.get("parameters")
                        parameters = parameters if isinstance(parameters, dict) else {}
                        if state != "ACTIVE" and state not in _COMPLETED_TOOL_STATES:
                            coverage["unknown_tool_state"] += 1
                            continue
                        if step_id not in started_tools:
                            started_tools.add(step_id)
                            if host is not None and not (name == "call_mcp_tool"
                                                         and parameters.get("ServerName") == HOST_SERVER):
                                # agy ran this start itself; the endpoint charges this attempt's own host calls.
                                host.observe_native_start(step_id, name)
                            yield ToolStartEvent(id=step_id, name=name, input=parameters)
                        if state in _COMPLETED_TOOL_STATES:
                            error = info.get("error")
                            output = info.get("output")
                            if not isinstance(output, str):
                                message = error.get("message") if isinstance(error, dict) else error
                                output = message if isinstance(message, str) else ""
                            yield ToolResultEvent(
                                id=step_id, output=output, is_error=state == "ERROR" or bool(error),
                                status=state.lower(), duration_ms=_milliseconds(body.get("duration_seconds")))
                    elif step_type not in ("user_input", "finish"):
                        coverage["unknown_step_type"] += 1

                elif kind == "result":
                    terminal = body
                    if lifecycle is not None:
                        lifecycle.terminal_received()
                    conversation = body.get("conversation_id")
                    if isinstance(conversation, str) and conversation:
                        conversation_id = conversation

                else:
                    coverage["unknown_event"] += 1

            if host is not None and host.binding.fatal_error is not None:
                raise host.binding.fatal_error

            try:
                returncode = await reap(transport, timeout_s=_PROCESS_REAP_TIMEOUT_S)
            except TimeoutError as exc:
                if lifecycle is not None:
                    lifecycle.expiry("process_reap")
                raise AntigravityError(
                    f"agy closed its output but its exit was not observed within {_PROCESS_REAP_TIMEOUT_S:g}s",
                    category="TIMEOUT", retryable=False if terminal is not None else None) from exc
            if host is not None and returncode:
                host.binding.close_attempt()  # as in finally: no callback is served during teardown
            # Joins the stderr drain, so agy's structured error line is complete before classification.
            await _teardown_once(transport)

            if coverage:
                _logger.warning("antigravity parser coverage gaps: %s", dict(coverage))
                yield DiagnosticEvent(
                    code="antigravity_parser_coverage",
                    message="The agy stream contained shapes outside the recorded contract; they were not interpreted.",
                    metadata={"reasons": dict(coverage)},
                )

            if terminal is None or terminal.get("status") != "SUCCESS" or not initialized or returncode != 0:
                raise self._failure(terminal, returncode, stderr_error, list(stderr_excerpt))

            denied = terminal.get("denied_actions")
            if isinstance(denied, list) and denied:
                actions = Counter(_bounded(item.get("action") if isinstance(item, dict) else None)[:64]
                                  for item in denied[:64])
                yield DiagnosticEvent(
                    code="antigravity_denied_actions",
                    message="agy denied actions itself during this turn.",
                    metadata={"actions": dict(actions), "total": len(denied)},
                )
                if host is not None and "mcp" in actions and not host.binding.possibly_accepted:
                    # The host never saw this call, and nothing the model resubmits changes a permission denial.
                    raise AntigravityError("agy denied a host tool call itself before acceptance; the turn failed",
                                           category="TRANSPORT", retryable=False)

            response = terminal.get("response")
            if isinstance(response, str) and response.strip() and not turn_text:
                # The final message did not stream as deltas; the terminal response is its only copy.
                if pending_turn is not None:
                    yield _turn_end(pending_turn)
                yield TextEvent(text=response)
                pending_turn = ""
            if pending_turn is not None:
                yield _turn_end(pending_turn)
            result = ResultEvent(
                structured_output=terminal.get("structured_output") if native_schema else None,
                continuation=(ContinuationToken(backend="antigravity", data={"conversation_id": conversation_id})
                              if persist_session and conversation_id else None),
                model_name=native_model or self._model,
                session_id=conversation_id,
                duration_ms=_milliseconds(terminal.get("duration_seconds")),
            )
            if host is not None:
                # Host acceptance is the only result; SUCCESS, EOF, zero exit and cleanup settle it.
                result.structured_output = host.binding.accepted_result
                host.binding.native_settled = host.binding.accepted
            yield result
            if host is not None and not host.binding.closed:
                # Neither accepted nor failed.
                raise UnsubmittedTurn("Antigravity completed its turn without an accepted structured_output "
                                      "submission")
        except StreamStalledError as exc:
            if lifecycle is not None:
                lifecycle.expiry("terminal_drain" if terminal is not None else "preterminal_stream")
            if terminal is None:
                raise
            # The turn may have completed its work: a fresh attempt could repeat a mutation.
            raise AntigravityError("agy reported its result but did not close its output; the turn was not "
                                   "retried", category="TIMEOUT", retryable=False) from exc
        finally:
            primary = sys.exception()
            if host is not None:
                # Closure first: no callback is served while the process is torn down.
                host.binding.close_attempt()
            try:
                if transport is not None and not cleaned_up:
                    try:
                        await _teardown_once(transport)
                    except AntigravityError:
                        if primary is None:
                            raise
            finally:
                if run is not None:
                    # The bearer and policy leave the disk with the attempt.
                    run.remove()

    @staticmethod
    def _failure(terminal: dict[str, Any] | None, returncode: int | None, stderr_error: dict[str, Any],
                 excerpt: list[str]) -> AntigravityError:
        """Classify a turn that did not reach SUCCESS with an observed zero exit."""
        error = terminal.get("error") if terminal is not None else None
        error_text = error if isinstance(error, str) else ""
        if error_text == "interrupted":
            return AntigravityError("Antigravity turn was interrupted before it completed",
                                    category="CANCELLED", retryable=False)
        if "authentication failed or timed out" in error_text.casefold():
            return AntigravityError(
                "Antigravity authentication failed or timed out. Sign in by running `agy` interactively with the "
                f"agy home this run selects (the directory named by {_HOME_ENV} when it is set, otherwise your "
                "own home), or provide API-key or Vertex credentials, then retry.",
                category="AUTH_CONFIG", retryable=False)
        stated = stderr_error.get("retryable")
        retryable = stated if type(stated) is bool else None
        detail = stderr_error.get("short_error") or error_text
        if returncode:
            message = process_exit_message(display="Antigravity", returncode=returncode, lines=excerpt)
            if detail:
                message = f"Antigravity turn failed: {_bounded(detail)}\n{message}"
            return AntigravityError(message, category="PROCESS_EXIT",
                                    retryable=False if retryable is None else retryable)
        if terminal is None:
            # Nothing completed: a truncated stream may be retried unless agy said otherwise.
            return AntigravityError("agy exited without a terminal result event", category="TRANSPORT",
                                    retryable=retryable)
        return AntigravityError(
            f"Antigravity turn did not succeed (status {_bounded(terminal.get('status'))!r}): "
            f"{_bounded(detail) or 'no error text'}",
            category="TRANSPORT", retryable=False if retryable is None else retryable)

    async def cancel(self) -> None:
        """Terminate and reap every active agy transport."""
        await CliTransport.cancel_all(self._transports)
