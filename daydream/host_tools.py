"""Synchronous admission for one live native Pi review assignment.

Pi owns ordinary reads and tool execution. This binding owns one domain callback,
its original deadline and cumulative allowance across short-lived transports.
Acceptance runs on the event-loop thread without an await, so authority closure
and state installation have an unambiguous order.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import anyio
from jsonschema import Draft202012Validator

from daydream import clock
from daydream.redaction import redact_structured_text

if TYPE_CHECKING:
    from daydream.host_tools_server import PiHostToolServer

_MAX_FRAME_BYTES = 256 * 1024
_MAX_ARGUMENT_BYTES = 128 * 1024
_MAX_RESULT_BYTES = 128 * 1024
_MAX_ITEMS = 32768
_MAX_DEPTH = 64
_CALLBACK_TIMEOUT_S = 3.0
_SETUP_TIMEOUT_S = 5.0
_CLEANUP_TIMEOUT_S = 2.0
NATIVE_SUBMIT = "structured_output"
# Conservative byte/token ratio for preflight only; providers tokenize differently.
BYTES_PER_TOKEN_ESTIMATE = 3
# Pi 1.4.0 compaction defaults (dist/core/compaction) when native settings omit them.
PI_DEFAULT_COMPACTION = {"reserve_tokens": 16384, "keep_recent_tokens": 20000}


class HostToolRejected(Exception):
    """Explicitly correctable payload failure before any domain mutation."""

    def __init__(self, message: str, *, code: str = "invalid_submission") -> None:
        super().__init__(_bounded_message(message))
        self.code = _bounded_message(code, limit=64)


HostToolFailureCode = Literal[
    "input_unavailable", "evidence_incomplete", "authority_closed", "deadline_expired", "capacity_exhausted",
    "native_correlation", "native_readiness", "invalid_frame", "callback_failed", "invalid_result",
    "native_capacity",
]


@dataclass(frozen=True)
class HostTool:
    """One charged nonterminal model tool served synchronously by the host.

    ``mutating`` callbacks install domain state at the same closure-first
    boundary as terminal acceptance; an unexpected failure is indeterminate.
    Read-only callbacks never change domain state.
    """

    schema: dict[str, Any]
    callback: Callable[[dict[str, Any]], dict[str, Any]]
    mutating: bool = False


@dataclass(frozen=True)
class NativeCapacityRequirement:
    """Whole-invocation bytes that must fit the native model before provider work."""

    largest_tool_result_bytes: int
    submission_bytes: int


def _bounded_message(message: str, *, limit: int = 1024) -> str:
    return redact_structured_text(message).encode("utf-8", errors="replace")[:limit].decode("utf-8", errors="ignore")


class HostToolFatal(Exception):
    """Trusted terminal host outcome; native details cannot assert this guarantee."""

    def __init__(self, message: str, *, code: HostToolFailureCode, preacceptance: bool = True) -> None:
        suffix = (" Changing the submission cannot resolve this failure." if preacceptance else
                  " Acceptance may already have occurred; do not resubmit.")
        super().__init__(_bounded_message(message, limit=768) + suffix)
        self.code = code
        self.preacceptance = preacceptance
        self.outcome: Literal["fatal", "indeterminate"] = "fatal" if preacceptance else "indeterminate"


class HostToolCallBudgetExceeded(HostToolFatal):
    """The binding's original native call allowance cannot authorize further work."""

    def __init__(self, message: str, *, preacceptance: bool = True) -> None:
        super().__init__(message, code="capacity_exhausted", preacceptance=preacceptance)


def _valid_call_id(call_id: Any) -> bool:
    try:
        return isinstance(call_id, str) and 0 < len(call_id.encode("utf-8", errors="strict")) <= 512
    except UnicodeError:
        return False


def _submission_schema_error(arguments: Any, schema: dict[str, Any] | None) -> str | None:
    if not isinstance(arguments, dict):
        return "Invalid terminal submission schema at $: submit a JSON object"
    if schema is None:
        return None
    error = next(Draft202012Validator(schema).iter_errors(arguments), None)
    if error is None:
        return None
    # Select a union branch only when the submitted types/valid properties
    # identify it. Otherwise explain all shapes, rather than suggesting a repair
    # for an unrelated alternative (for example a supporting selector's basis).
    while error.validator in {"anyOf", "oneOf"} and error.context:
        scores = []
        for option in error.validator_value:
            kind = option.get("type")
            matching_type = int(kind is not None and Draft202012Validator({"type": kind}).is_valid(error.instance))
            fields = option.get("properties", {})
            present = [key for key in fields if isinstance(error.instance, dict) and key in error.instance]
            matching_fields = sum(Draft202012Validator(fields[key]).is_valid(error.instance[key]) for key in present)
            scores.append((matching_type, matching_fields, len(present)))
        best = max(scores)
        if scores.count(best) != 1:
            break
        branch = scores.index(best)
        children = [child for child in error.context if child.schema_path and child.schema_path[0] == branch]
        if not children:
            break
        error = max(children, key=lambda item: len(item.absolute_path))
    # Never use jsonschema's message: it can include complete submitted contents.
    path = "$" + "".join(f"[{part}]" if isinstance(part, int) else f"[{json.dumps(part)}]"
                          for part in error.absolute_path)
    rule, limit = error.validator, error.validator_value
    if rule == "required":
        missing = [field for field in limit if field not in error.instance]
        correction = f"add required fields {json.dumps(missing)}"
    elif rule == "additionalProperties":
        correction = "remove undeclared fields; use only properties in the tool schema"
    elif rule == "type":
        correction = f"use the required JSON type {json.dumps(limit)}"
    elif rule == "enum":
        correction = f"choose an allowed value from {json.dumps(limit)}"
    elif rule in {"anyOf", "oneOf"}:
        shapes = []
        for option in limit:
            shape = str(option.get("type", "schema alternative"))
            if option.get("required"):
                shape += f" with required fields {json.dumps(option['required'])}"
            if "enum" in option:
                shape += f" from {json.dumps(option['enum'])}"
            shapes.append(shape)
        correction = "use one allowed JSON shape: " + " or ".join(shapes)
    elif rule == "pattern":
        correction = f"use a string matching the required pattern {json.dumps(limit)}"
    elif rule in {"maxItems", "maxLength", "maximum"}:
        unit = {"maxItems": "items", "maxLength": "characters", "maximum": "as a numeric value"}[rule]
        correction = f"reduce this field to at most {limit} {unit}"
    elif rule in {"minItems", "minLength", "minimum"}:
        unit = {"minItems": "items", "minLength": "characters", "minimum": "as a numeric value"}[rule]
        correction = f"provide at least {limit} {unit} in this field"
    else:
        correction = f"match this field's {rule} constraint in the tool schema"
    return _bounded_message(f"Invalid terminal submission schema at {path}: {correction}")


class HostToolBinding:
    """One review capability with cumulative limits and attempt-local correlation."""

    def __init__(self, *, deadline: float, accept: Callable[[Any], dict[str, Any]],
                 allowed_tools: Set[str] | None = None, host_tools: Mapping[str, HostTool] | None = None,
                 reserves: Mapping[str, Any] | None = None,
                 native_requirement: NativeCapacityRequirement | None = None) -> None:
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("host tools require a finite absolute monotonic deadline")
        if not callable(accept):
            raise TypeError("host tools require a synchronous acceptance callback")
        self.deadline = float(deadline)
        self.calls = self.rejections = 0
        self.authority_open = self.native_settled = self.cleanup_complete = False
        self.closed = False
        self.fatal_error: HostToolFatal | None = None
        self._accept = accept
        self._allowed_tools = frozenset(allowed_tools) if allowed_tools is not None else None
        self._call_limit: int | None = None
        self._schema: dict[str, Any] | None = None
        self._session_active = self._cleanup_failed = self._acceptance_started = False
        self._accepted_result: dict[str, Any] | None = None
        self._accepted_identity: tuple[str, str] | None = None
        self._admitted: set[str] = set()
        self._pending: dict[str, str] = {}
        self._late_cleanup: set[asyncio.Task[None]] = set()
        self.host_tools: dict[str, HostTool] = dict(host_tools or {})
        if NATIVE_SUBMIT in self.host_tools:
            raise ValueError("the terminal submission tool cannot be a nonterminal host tool")
        self.reserves = dict(reserves or {})
        self.native_requirement = native_requirement
        self.prompt_bytes = 0
        self.native_limits: dict[str, Any] | None = None
        self._host_pending: dict[str, str] = {}

    @property
    def accepted(self) -> bool:
        return self._accepted_result is not None

    @property
    def accepted_result(self) -> dict[str, Any] | None:
        return self._accepted_result

    @property
    def possibly_accepted(self) -> bool:
        """Unexpected callback failures cannot authorize replay of a mutation."""
        return self._acceptance_started

    @property
    def schema(self) -> dict[str, Any] | None:
        return self._schema

    @property
    def allowed_tools(self) -> frozenset[str] | None:
        return self._allowed_tools

    @property
    def remaining_tool_calls(self) -> int | None:
        return None if self._call_limit is None else max(0, self._call_limit - self.calls)

    def configure_schema(self, schema: dict[str, Any]) -> None:
        """Bind readiness and native argument validation to the model-facing schema."""
        if self._session_active or self.possibly_accepted:
            raise ValueError("cannot change an active host submission schema")
        self._schema = json.loads(json.dumps(schema, ensure_ascii=False, allow_nan=False))

    def restrict(self, *, deadline: float | None, tool_call_budget: int | None,
                 allowed_tools: Set[str] | None = None) -> None:
        """Intersect original cumulative policy; retries cannot replenish it."""
        if deadline is not None:
            self.deadline = min(self.deadline, deadline)
        if tool_call_budget is not None:
            self._call_limit = (tool_call_budget if self._call_limit is None
                                else min(self._call_limit, tool_call_budget))
        if allowed_tools is not None:
            self._allowed_tools = (frozenset(allowed_tools) if self._allowed_tools is None
                                   else self._allowed_tools & allowed_tools)

    def _check_open(self, *, preacceptance: bool | None = None) -> None:
        if self.fatal_error is not None:
            raise self.fatal_error
        if clock.monotonic() >= self.deadline:
            raise self._fatal("Host-tool absolute deadline expired", code="deadline_expired",
                              preacceptance=preacceptance)
        if self.closed or not self.authority_open:
            raise self._fatal("Host-tool authority is closed", code="authority_closed", preacceptance=preacceptance)

    def ensure_open(self) -> None:
        """Recheck authority immediately before the domain installs prepared state."""
        self._check_open(preacceptance=not self.accepted)

    def fail(self, error: HostToolFatal) -> HostToolFatal:
        """Retain the first terminal host condition and synchronously revoke authority."""
        if self.fatal_error is None:
            self.fatal_error = error
        self.close_authority()
        return self.fatal_error

    def _fatal(self, message: str, *, code: HostToolFailureCode,
               preacceptance: bool | None = None) -> HostToolFatal:
        self.rejections += 1
        return self.fail(HostToolFatal(message, code=code, preacceptance=(not self.possibly_accepted
                                      if preacceptance is None else preacceptance)))

    def _reject(self, message: str, *, code: str = "invalid_submission") -> HostToolRejected | HostToolFatal:
        self.rejections += 1
        if self.possibly_accepted:
            return self.fail(HostToolFatal("Host callback could not confirm a new submission after terminal acceptance",
                                          code="native_correlation", preacceptance=False))
        rejection = HostToolRejected(message, code=code)
        if self._call_limit is not None and self.calls >= self._call_limit:
            fatal = HostToolCallBudgetExceeded(
                "Original cumulative tool allowance exhausted; no calls remain for correction. "
                f"Rejected submission: {rejection}", preacceptance=not self.possibly_accepted)
            fatal.__cause__ = rejection
            return self.fail(fatal)
        return rejection

    def _arguments_key(self, arguments: Any) -> str:
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                 separators=(",", ":"))
            byte_count = len(encoded.encode("utf-8", errors="strict"))
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise self._reject("Invalid structured_output arguments; submit a JSON object "
                               "with valid UTF-8 text") from None
        if byte_count > _MAX_ARGUMENT_BYTES:
            raise self._reject(f"The entire structured_output argument object exceeds {_MAX_ARGUMENT_BYTES} UTF-8 "
                               "bytes; reduce the submitted arguments before retrying")
        return encoded

    def admit(self, call_id: Any, name: Any, arguments: Any, *, terminal: bool = False,
              malformed: bool | str = False) -> None:
        """Charge real starts before native validation; only sole final calls can submit."""
        self._check_open()
        if not _valid_call_id(call_id) or not isinstance(name, str) or not name or len(name) > 256:
            raise self._fatal("Invalid native tool identity or name prevents host correlation",
                              code="native_correlation")
        if call_id in self._admitted:
            raise self._fatal("Native tool identity was already charged; repeated starts cannot be correlated",
                              code="native_correlation")
        self.calls += 1
        self._admitted.add(call_id)
        if self._call_limit is not None and self.calls > self._call_limit:
            self.rejections += 1
            raise self.fail(HostToolCallBudgetExceeded(
                "Original cumulative tool allowance exhausted; no further native calls are authorized",
                preacceptance=not self.possibly_accepted))
        if self._allowed_tools is not None and name not in self._allowed_tools:
            raise self._reject("This tool is not granted by the operator's read-only policy; use only granted tools",
                               code="permission_denied")
        if malformed:
            raise self._reject(malformed if isinstance(malformed, str) else
                               "Invalid bounded host frame; reduce the submitted argument size and nesting")
        if name in self.host_tools:
            self._host_pending[call_id] = name
        if name == NATIVE_SUBMIT:
            if terminal is not True:
                raise self._reject("Submit structured_output alone in a completed native generation")
            key = self._arguments_key(arguments)
            if schema_error := _submission_schema_error(arguments, self._schema):
                raise self._reject(schema_error)
            self._pending[call_id] = key

    def submit(self, call_id: Any, arguments: Any) -> dict[str, Any]:
        """Synchronously accept once, immediately after the final authority/deadline check."""
        if self._accepted_identity is None:
            self._check_open()
        key = self._arguments_key(arguments)
        if self._accepted_identity == (call_id, key):
            assert self._accepted_result is not None
            return self._accepted_result
        self._check_open()
        if self.possibly_accepted:
            raise self._fatal("This invocation has already entered terminal acceptance", code="native_correlation")
        if not isinstance(call_id, str) or self._pending.pop(call_id, None) != key:
            raise self._fatal("Host callback has no matching completed native terminal call", code="native_correlation")
        if schema_error := _submission_schema_error(arguments, self._schema):
            raise self._reject(schema_error)
        # All payload/identity checks precede this final synchronous boundary.
        # There is deliberately no await or detached thread between the check
        # and domain installation. The callback must preflight its result before
        # changing state; uncertain unexpected failures prohibit automatic replay.
        self._check_open()
        self._acceptance_started = True
        try:
            result = self._accept(arguments)
        except HostToolRejected as exc:
            self._acceptance_started = False
            raise self._reject(str(exc), code=exc.code) from exc
        except HostToolFatal as exc:
            if exc.preacceptance:
                self._acceptance_started = False
            raise self.fail(exc)
        except Exception as exc:
            raise self.fail(HostToolFatal("Host acceptance callback failed unexpectedly", code="callback_failed",
                                          preacceptance=False)) from exc
        except BaseException:
            self.close_authority()
            raise
        try:
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            if not isinstance(result, dict) or len(encoded.encode("utf-8", errors="strict")) > _MAX_RESULT_BYTES:
                raise ValueError("invalid canonical result")
            admitted: dict[str, Any] = json.loads(encoded)
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise self.fail(HostToolFatal("Host acceptance returned an invalid canonical result", code="invalid_result",
                                          preacceptance=False)) from exc
        self._accepted_result = admitted
        self._accepted_identity = (call_id, key)
        self._pending.clear()
        return admitted

    def capacity(self) -> dict[str, Any]:
        """Current host ledger, distinct from any dispatch-time snapshot in instructions."""
        return {
            "basis": "current host ledger after charging this call; supersedes dispatch-time snapshots",
            "remaining_tool_calls": self.remaining_tool_calls,
            "remaining_s": round(max(0.0, self.deadline - clock.monotonic()), 3),
            "protected": dict(self.reserves),
        }

    def call(self, call_id: Any, name: Any, arguments: Any) -> dict[str, Any]:
        """Serve one charged nonterminal host tool exactly once for its admitted start."""
        self._check_open()
        if self.possibly_accepted:
            raise self._fatal("This invocation has already entered terminal acceptance", code="native_correlation")
        if (not isinstance(call_id, str) or not isinstance(name, str)
                or self._host_pending.pop(call_id, None) != name or name not in self.host_tools):
            raise self._fatal("Host tool call has no matching unused admitted start", code="native_correlation")
        tool = self.host_tools[name]
        if schema_error := _submission_schema_error(arguments, tool.schema):
            raise self._reject(schema_error.replace("terminal submission", f"{name} argument"))
        self._check_open()
        try:
            result = tool.callback(arguments)
        except HostToolRejected as exc:
            raise self._reject(str(exc), code=exc.code) from exc
        except HostToolFatal as exc:
            raise self.fail(exc) from exc
        except Exception as exc:
            raise self.fail(HostToolFatal(f"Host {name} callback failed unexpectedly", code="callback_failed",
                                          preacceptance=not tool.mutating)) from exc
        try:
            encoded = json.dumps({**result, "capacity": self.capacity()}, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":"))
            if not isinstance(result, dict) or len(encoded.encode("utf-8", errors="strict")) > _MAX_RESULT_BYTES:
                raise ValueError("invalid host tool result")
            response: dict[str, Any] = json.loads(encoded)
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise self.fail(HostToolFatal(f"Host {name} returned an invalid bounded result", code="invalid_result",
                                          preacceptance=not tool.mutating)) from exc
        return response

    def check_native_limits(self, reported: Any) -> dict[str, Any]:
        """Record native model/compaction limits and refuse a whole invocation that cannot fit."""
        model = reported.get("model") if isinstance(reported, dict) else None
        settings = reported.get("compaction") if isinstance(reported, dict) else None
        def positive(value: Any) -> int | None:
            return value if type(value) is int and value > 0 else None
        evidence: dict[str, Any] = {
            "context_window": positive(model.get("context_window")) if isinstance(model, dict) else None,
            "max_tokens": positive(model.get("max_tokens")) if isinstance(model, dict) else None,
            "compaction_enabled": settings.get("enabled") if isinstance(settings, dict)
            and type(settings.get("enabled")) is bool else None,
        }
        for key, default in PI_DEFAULT_COMPACTION.items():
            value = positive(settings.get(key)) if isinstance(settings, dict) else None
            evidence[key] = value if value is not None else default
            evidence[f"{key}_source"] = "native_settings" if value is not None else "pi_1.4.0_default"
        requirement = self.native_requirement
        if requirement is not None:
            def tokens(size: int) -> int:
                return -(-size // BYTES_PER_TOKEN_ESTIMATE)
            context = (tokens(self.prompt_bytes + requirement.largest_tool_result_bytes + requirement.submission_bytes)
                       + evidence["reserve_tokens"] + evidence["keep_recent_tokens"])
            output = tokens(requirement.submission_bytes)
            evidence.update(required_context_tokens=context, required_output_tokens=output)
            self.native_limits = evidence
            if evidence["context_window"] is None or evidence["max_tokens"] is None:
                raise self._fatal("Native model context/output limits are unavailable before provider work; "
                                  "declare contextWindow and maxTokens in the native model registry",
                                  code="native_capacity")
            if evidence["context_window"] < context or evidence["max_tokens"] < output:
                raise self._fatal(
                    f"Native model limits cannot fit this review invocation: needs contextWindow >= {context} "
                    f"(instructions, largest tool result, bounded submission, compaction reserve and kept "
                    f"context) and maxTokens >= {output}; observed contextWindow "
                    f"{evidence['context_window']} and maxTokens {evidence['max_tokens']}. Select a model "
                    "with larger native limits or adjust native compaction settings.", code="native_capacity")
        self.native_limits = evidence
        return evidence

    @asynccontextmanager
    async def session(self) -> AsyncIterator[PiHostToolServer]:
        """Open one attempt's endpoint; unaccepted retries retain counters and deadline."""
        from daydream.host_tools_server import PiHostToolServer

        if self.fatal_error is not None:
            raise self.fatal_error
        if self.closed:
            raise self._fatal("Host-tool authority is closed", code="authority_closed")
        if self._session_active or self.possibly_accepted or self._cleanup_failed:
            raise ValueError("host-tool binding cannot start another native attempt")
        if clock.monotonic() >= self.deadline:
            raise self._fatal("Host-tool absolute deadline expired before native setup", code="deadline_expired")
        self._session_active = True
        self.cleanup_complete = self.native_settled = False
        self._admitted.clear()
        self._pending.clear()
        self._host_pending.clear()
        server = PiHostToolServer(self)
        try:
            try:
                async with asyncio.timeout(min(_SETUP_TIMEOUT_S, max(0, self.deadline - clock.monotonic()))):
                    await server.start()
            except TimeoutError as exc:
                if clock.monotonic() >= self.deadline:
                    raise self._fatal("Host-tool absolute deadline expired during native setup",
                                      code="deadline_expired") from exc
                raise
            self.authority_open = True
            self._check_open()
            yield server
        finally:
            self.close_attempt()
            with anyio.CancelScope(shield=True):
                try:
                    async with asyncio.timeout(_CLEANUP_TIMEOUT_S):
                        await server.close()
                        await asyncio.gather(*self._late_cleanup, return_exceptions=True)
                except TimeoutError:
                    self.mark_cleanup_failed()
            self._session_active = False
            self.cleanup_complete = not self._cleanup_failed and not self._late_cleanup and server.closed

    async def start[T](self, operation: Coroutine[Any, Any, T],
                       cleanup: Callable[[T], Coroutine[Any, Any, None]]) -> T:
        """Keep process ownership when native startup returns after cancellation."""
        task = asyncio.create_task(operation)
        try:
            return await asyncio.shield(task)
        except BaseException:
            self.close_attempt()
            joined = False
            try:
                with anyio.CancelScope(shield=True):
                    async with asyncio.timeout(_CLEANUP_TIMEOUT_S):
                        await cleanup(await asyncio.shield(task))
                        joined = True
            except Exception:
                pass  # Preserve the original startup failure.
            finally:
                if not joined and not task.cancelled() and (not task.done() or task.exception() is None):
                    self.mark_cleanup_failed()

                    def finish(completed: asyncio.Task[T]) -> None:
                        if not completed.cancelled() and completed.exception() is None:
                            self.track_cleanup(cleanup(completed.result()))

                    task.add_done_callback(finish)
            raise

    def track_cleanup(self, cleanup: Coroutine[Any, Any, Any]) -> None:
        """Own late transport cleanup without writing frozen evidence."""
        task = asyncio.create_task(cleanup)
        self._late_cleanup.add(task)

        def finished(completed: asyncio.Task[None]) -> None:
            self._late_cleanup.discard(completed)
            if completed.cancelled() or completed.exception() is not None:
                self.mark_cleanup_failed()

        task.add_done_callback(finished)

    def mark_cleanup_failed(self) -> None:
        self._cleanup_failed = True
        self.cleanup_complete = False

    def close_authority(self) -> None:
        """Permanently revoke the capability before joining native resources."""
        self.closed = True
        self.close_attempt()

    def close_attempt(self) -> None:
        """Revoke this transport; a permitted live retry keeps cumulative policy."""
        self.authority_open = False
        self._pending.clear()
        self._host_pending.clear()
