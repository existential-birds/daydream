"""Agent interaction and backend management."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import nullcontext
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import anyio
from rich.console import Console

if TYPE_CHECKING:
    from claude_agent_sdk.types import AgentDefinition
    from rich.text import Text

from daydream import clock
from daydream.agent_output import StructuredOutputFailure as StructuredOutputFailure, resolve_output
from daydream.agent_retry import _plan_retry_delay, _resolve_retry_settings, _retry_hint, _RetryTelemetry
from daydream.artifact_visibility import ArtifactVisibilityError, artifact_session_active, assert_model_cwd_clean
from daydream.backends import (
    AgentEvent,
    AgentEventStream,
    Backend,
    ContinuationToken,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolStartEvent,
    TurnEndEvent,
    is_adapter_instance,
)
from daydream.backends.lifecycle import BackendLifecycleCollector, BackendLifecycleSnapshot
from daydream.config import BUDGET_CLEANUP_GRACE_S
from daydream.diagnostics import exception_text, sanitize_verbose_message
from daydream.extensions import get_registry
from daydream.host_tools import HostToolBinding, HostToolCallBudgetExceeded
from daydream.observability.spans import agent_scope, attempt_scope, record_budget_stop, record_retry_stop
from daydream.outage_circuit import CIRCUIT_CLOSED, CIRCUIT_HALF_OPEN
from daydream.prompt_budget import PreparedSanctionedInputs
from daydream.prompts.grounding import (
    READ_ONLY_PI_SEARCH_GUIDANCE,
    REPOSITORY_SEARCH_GUIDANCE,
    REVIEW_STOPPING_GUIDANCE,
)
from daydream.retry_policy import FailureClass, classify_failure
from daydream.review_budget import (
    BudgetDimension,
    BudgetExhausted,
    BudgetLedger,
    ReviewLimits,
    review_deadline,
    review_limits_for_scope,
)
from daydream.review_evidence import FinalizationContext, ReviewEvidence
from daydream.run_context import (
    RunContext,
    bind_run_context,
    current_run_context,
    resolve_gate as resolve_gate,
    resolve_run_context,
)
from daydream.trajectory import DaydreamPhase, get_current_recorder, redact_text
from daydream.ui import NEON_THEME, print_error, print_warning
from daydream.ui.agent_stream import AgentDisplay, present_result

_logger = logging.getLogger(__name__)


class _ToolSupervisorFailure(Exception):
    """Internal marker that keeps supervisor failures out of backend retries."""

    #: Names this failure for the shared classifier; a supervisor veto is a
    #: permanent, never-retryable decision.
    failure_class = FailureClass.TOOL_POLICY

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(exception_text(original) or "")

    @property
    def subtype(self) -> str:
        """Expose the original error's type name for trajectory recording."""
        return type(self.original).__name__


class _RedactedSupervisorError(RuntimeError):
    """Fallback preserving exception type name and a scrubbed message.

    Use when rebuilding cannot remove secrets from custom str/repr or OSError
    fields; outer handlers may print this exception without another redaction pass.
    """

    def __init__(self, original_type_name: str, message: str) -> None:
        self.original_type_name = original_type_name
        self.retryable: bool = False
        super().__init__(message)


def _scrubbed_supervisor_error(original: BaseException) -> BaseException:
    """Rebuild from scrubbed args, falling back if construction fails or str stays unsafe.

    Preserve retryable for outer retry consumers; custom str/repr and OSError fields
    must never leak the original credential.
    """
    scrubbed_args = tuple(
        redact_text(a) if isinstance(a, str) else a for a in original.args
    )
    try:
        clone = type(original)(*scrubbed_args)
    except (AttributeError, TypeError):
        clone = None
    clone_text = exception_text(clone) if clone is not None else None
    if clone is not None and clone_text is not None and redact_text(clone_text) == clone_text:
        setattr(clone, "retryable", getattr(original, "retryable", False))
        return clone
    try:
        message = redact_text(exception_text(original) or "")
    except Exception:  # noqa: BLE001 - fail closed if redaction itself fails
        message = ""
    stand_in = _RedactedSupervisorError(type(original).__name__, message)
    stand_in.retryable = getattr(original, "retryable", False)
    return stand_in


class _EventStreamScope:
    """Idempotent owner for one backend invocation's event stream."""

    def __init__(
        self, event_iter: AgentEventStream, *, lifecycle: BackendLifecycleCollector,
        record_lifecycle: Callable[[BackendLifecycleSnapshot], None],
        host_tools: HostToolBinding | None = None,
    ) -> None:
        self.event_iter = event_iter
        self._closed = False
        self._lifecycle = lifecycle
        self._record_lifecycle = record_lifecycle
        self._host_tools = host_tools
        self.ledger_stop: str | None = None

    async def __aenter__(self) -> AgentEventStream:
        self._lifecycle_token = self._lifecycle.bind()
        return self.event_iter

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._host_tools is not None and (
            isinstance(exc, _ToolSupervisorFailure) or (exc is not None and not isinstance(exc, Exception))
        ):
            self._host_tools.close_authority()
        try:
            await self.aclose()
        finally:
            await self._lifecycle.stop_sampling()
            # No await/yield between freezing and handing evidence to the scopes
            # that finalize next. GeneratorExit/cancellation cannot drop this.
            try:
                snapshot = self._lifecycle.finalize()
                if snapshot is not None:
                    self._record_lifecycle(snapshot)
            finally:
                self._lifecycle.reset(self._lifecycle_token)

    async def events(self) -> AsyncIterator[AgentEvent]:
        """Observe demand separately from adapter yields and recorder delivery.

        A ledger limit that ends the adapter stream (a deadline-cut read or a
        binding's deadline/allocation failure) ends iteration with ``ledger_stop``
        set, so it takes the in-flight budget-stop path instead of a retry.
        """
        while True:
            self._lifecycle.consumer_requested()
            try:
                event = await anext(self.event_iter)
            except StopAsyncIteration:
                return
            except Exception as exc:
                self.ledger_stop = _ledger_stop(exc, self._host_tools)
                if self.ledger_stop is None:
                    raise
                return
            self._lifecycle.consumer_event()
            yield event

    async def aclose(self) -> None:
        """Close the invocation once without masking its outcome."""
        if self._closed:
            return
        self._closed = True
        if self._host_tools is not None:
            self._host_tools.close_attempt()
        try:
            with anyio.move_on_after(BUDGET_CLEANUP_GRACE_S, shield=True) as scope:
                await self.event_iter.aclose()
            if scope.cancel_called:
                self._lifecycle.expiry("cleanup")
        except Exception:  # noqa: BLE001 - cleanup must not mask the invocation outcome
            pass


_STOP_DIMENSIONS: dict[str | None, BudgetDimension] = {
    "wall_budget_exceeded": "time", "tool_call_budget_exceeded": "tool_calls",
}


def _ledger_stop(exc: Exception, host_tools: HostToolBinding | None) -> str | None:
    """Map a ledger limit that ended an attempt to its budget stop reason, else None.

    A binding failure counts only when it caused this exception before any possible
    acceptance; every other failure keeps its own handling.
    """
    if isinstance(exc, BudgetExhausted):
        return "wall_budget_exceeded"
    fatal = host_tools.fatal_error if host_tools is not None else None
    if fatal is None or not fatal.preacceptance or fatal not in (exc, exc.__cause__):
        return None
    if fatal.code == "deadline_expired":
        return "wall_budget_exceeded"
    return "tool_call_budget_exceeded" if isinstance(fatal, HostToolCallBudgetExceeded) else None


class _LogRedactingConsole(Console):
    """Redact string console payloads in verbose mode, including UI calls outside agent-event emission."""

    def print(self, *objects: Any, **kwargs: Any) -> None:
        context = current_run_context()
        if context is not None and context.policy.log_mode:
            objects = tuple(
                redact_text(obj) if isinstance(obj, str) else obj
                for obj in objects
            )
        super().print(*objects, **kwargs)


console = _LogRedactingConsole(theme=NEON_THEME)


def is_environmental_failure(test_output: str) -> bool:
    """Recognize infrastructure failures case-insensitively to stop futile code-heal turns."""
    if not test_output:
        return False

    output_lower = test_output.lower()

    infra_signatures = [
        "connection refused",
        "localhost:5432",
        ":6379",
        "container is not running",
        "make db-up",
        "econnrefused",
    ]
    return any(signature in output_lower for signature in infra_signatures)


async def run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    *,
    phase: DaydreamPhase,
    output_schema: dict[str, Any] | None = None,
    result_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[Text], Any] | None = None,
    continuation: ContinuationToken | None = None,
    agents: dict[str, AgentDefinition] | None = None,
    read_only: bool = False,
    persist_session: bool = True,
    budget: BudgetLedger | None = None,
    validate_structured_output: bool = True,
    require_full_schema: bool = False,
    sanctioned_inputs: PreparedSanctionedInputs | None = None,
    run_context: RunContext | None = None,
    review_limits: ReviewLimits | None = None,
    finalization_context: FinalizationContext | None = None,
    tools_disabled: bool = False,
    review_system_instructions: str | None = None,
    review_stage_scope_id: str | None = None,
    host_tools: HostToolBinding | None = None,
) -> tuple[str | Any, ContinuationToken | None, str | None]:
    """Run one logical agent, tracing its actual returned or salvaged result.

    Backend retry, supervision, budget and ATIF semantics live in the invocation
    executor. The outer scope owns exactly the result the phase receives.
    ``budget`` is the invocation's one ledger: an absolute deadline and a charged
    tool-call count. Its relative wall allowance is anchored here, after admission,
    so queue time is never charged. Reviewer (investigation) ledgers are cumulative
    across invocations and may use optional bounded review_evidence; failure/cutoff
    blocks automatic serialization. Host tools charge the same ledger.
    """
    if host_tools is not None:
        if not getattr(backend, "supports_host_tools", False):
            raise NotImplementedError(f"{type(backend).__name__} does not support host tools")
        if tools_disabled or continuation is not None or review_limits is not None:
            raise ValueError("host tools require a fresh tools-enabled invocation without review finalization")
        if budget is None:
            budget = host_tools.budget
        elif budget is not host_tools.budget:
            raise ValueError("host tools must charge the invocation's budget ledger")
    budget = (budget if budget is not None else BudgetLedger(phase.value)).anchor().begin()
    investigation = budget.investigation
    if investigation and budget.deadline is None:
        raise ValueError("an investigation ledger requires an absolute deadline")
    if tools_disabled and not getattr(backend, "supports_tools_disabled", False):
        raise NotImplementedError(f"{type(backend).__name__} does not support tools_disabled")
    if review_system_instructions is not None:
        if not tools_disabled and not investigation:
            raise ValueError("review_system_instructions requires tools_disabled=True or an investigation ledger")
        if not investigation and not getattr(backend, "supports_review_instructions", False):
            raise NotImplementedError(f"{type(backend).__name__} does not support review_instructions")
    if sanctioned_inputs is not None:
        # Callers may already have rendered this suffix. Move it after the
        # host's budget/finalization instructions without duplicating inputs.
        rendered_suffix = sanctioned_inputs.render_prompt("")
        if rendered_suffix and prompt.endswith(rendered_suffix):
            prompt = prompt.removesuffix(rendered_suffix)
    context = resolve_run_context(run_context)
    if investigation and review_limits is not None:
        raise ValueError("an investigation ledger replaces review_limits")
    evidence = ReviewEvidence(output_schema) if review_limits is not None else None
    review_instructions = review_system_instructions
    turn = budget
    hard_deadline = budget.deadline
    shared = budget.shared_deadline
    if investigation:
        budget_instructions = (
            "Reviewer dispatch-time snapshot: hard wall allowance: "
            f"{budget.remaining_s() or 0.0:g} seconds until the absolute deadline; "
            f"current invocation allocation: {budget.remaining_tool_calls} tool calls. "
            "Future-stage reserves are already excluded from this allocation; do not subtract them again. "
        )
    if review_limits is not None:
        review_limits = review_limits_for_scope(review_limits)
        started = clock.monotonic()
        bounds = [started + review_limits.investigation_s + review_limits.finalization_s]
        if budget.deadline is not None:
            bounds.append(budget.deadline)
        shared = review_deadline(discovery=review_limits.discovery)
        if shared is not None:
            bounds.append(shared)
        hard_deadline = min(bounds)
        investigation_deadline = max(
            started, min(started + review_limits.investigation_s, hard_deadline - review_limits.finalization_s),
        )
        turn = budget.carve(budget.owner, limit=review_limits.tool_calls, deadline=investigation_deadline)
        budget_instructions = (
            f"Investigation allowance: at most {max(0.0, investigation_deadline - started):g} seconds and "
            f"{turn.limit} tool calls. " + REVIEW_STOPPING_GUIDANCE
        )
    if investigation or review_limits is not None:
        prompt += "\n\n" + budget_instructions
        review_instructions = "\n\n".join(
            item for item in (review_system_instructions, budget_instructions) if item
        )
    search_turn = read_only or review_limits is not None or investigation
    if search_turn and not tools_disabled and phase in {
        DaydreamPhase.REVIEW, DaydreamPhase.DEEP, DaydreamPhase.EXPLORATION,
        DaydreamPhase.INTENT, DaydreamPhase.ALTERNATIVES, DaydreamPhase.VERIFY,
    }:
        # Host composition also covers wholesale extension prompt overrides.
        # Read-only Pi has only its stock bounded tools: no shell, Git or helper.
        search_guidance = REPOSITORY_SEARCH_GUIDANCE
        if read_only and is_adapter_instance(backend, "PiBackend"):
            # Composed review prompts embed the shell guidance; substitute it in place.
            search_guidance = READ_ONLY_PI_SEARCH_GUIDANCE
            prompt = prompt.replace(REPOSITORY_SEARCH_GUIDANCE, search_guidance)
            if review_instructions:
                review_instructions = review_instructions.replace(REPOSITORY_SEARCH_GUIDANCE, search_guidance)
        if search_guidance not in prompt:
            prompt += "\n\n" + search_guidance
        # Instructions can quote untrusted text that repeats the guidance; only the host's own copy ends them.
        if review_instructions and not review_instructions.endswith(search_guidance):
            review_instructions += "\n\n" + search_guidance
    if sanctioned_inputs is not None:
        prompt = sanctioned_inputs.render_prompt(prompt)
    backend_name = type(backend).__name__.removesuffix("Backend").lower()
    with bind_run_context(context), agent_scope(
        phase.value, backend=backend_name, model=backend.model
    ) as observed:
        observed.content("traceloop.entity.input", {"prompt": prompt, "output_schema": output_schema})
        try:
            result = await _run_agent(
                backend, cwd, prompt, phase=phase, budget=turn, output_schema=output_schema,
                result_schema=result_schema, progress_callback=progress_callback,
                continuation=continuation, agents=agents, read_only=read_only,
                persist_session=persist_session,
                validate_structured_output=validate_structured_output,
                require_full_schema=require_full_schema,
                sanctioned_inputs=sanctioned_inputs,
                run_context=context,
                review_evidence=evidence,
                review_instructions=review_instructions,
                tools_disabled=tools_disabled,
                review_stage_scope_id=review_stage_scope_id,
                host_tools=host_tools,
            )
        finally:
            # One stop record per budget stop, including one that ends in an exception.
            if turn.stopped is not None:
                observed.attrs(turn.stopped.attributes())
        if evidence is not None and review_limits is not None and result[2] in {
            "wall_budget_exceeded", "tool_call_budget_exceeded",
        }:
            partial, token, reason = result
            if evidence.valid(partial):
                result = (partial, token, reason)
            elif evidence.checkpoint is not None:
                result = (evidence.checkpoint, token, reason)
            elif hard_deadline is not None and clock.monotonic() < hard_deadline:
                try:
                    captured = (
                        sanctioned_inputs.finalization_text(
                            backend, cwd, read_only,
                            input_priority=finalization_context.input_priority if finalization_context else (),
                        )
                        if sanctioned_inputs is not None else ""
                    )
                    final_prompt = evidence.finalization_prompt(
                        finalization_context or FinalizationContext(task=phase.value), captured,
                    )
                    final_budget = BudgetLedger(budget.owner, limit=0, deadline=min(
                        hard_deadline, clock.monotonic() + review_limits.finalization_s)).anchor()
                    try:
                        finalized, _, final_reason = await _run_agent(
                            backend, cwd, final_prompt, phase=phase, budget=final_budget,
                            output_schema=output_schema, progress_callback=progress_callback,
                            require_full_schema=require_full_schema,
                            read_only=read_only, persist_session=persist_session, finalization=True,
                            sanctioned_inputs=sanctioned_inputs, run_context=context,
                        )
                    finally:
                        # The investigation stop stays on the span's attributes; this one is an event.
                        if final_budget.stopped is not None:
                            record_budget_stop(final_budget.stopped)
                    if final_reason is None and (
                        evidence.valid(finalized) or (output_schema is None and isinstance(finalized, str))
                    ):
                        result = (finalized, None, reason)
                except ArtifactVisibilityError:
                    raise
                except Exception:  # finalization must not erase the original incomplete result
                    _logger.warning("Review finalization failed; retaining incomplete output")
            # Budget-limited outputs use full validation, never shape-only salvage.
            if output_schema is not None and not evidence.valid(result[0]):
                result = ("", None, reason)
        if (result[2] == "wall_budget_exceeded" and shared is not None
                and hard_deadline == shared and clock.monotonic() >= shared):
            result = (result[0], result[1], "pipeline_budget_exceeded")
        if output_schema is not None and not investigation and not isinstance(result[0], str):
            present_result(context.policy, result[0])
        observed.output(result[0])
        observed.finish(1 if result[2] else 0, reason=result[2])
        return result


def _measure_host_dispatch(
    *, prompt: str, system_instructions: str | None, system_instructions_supported: bool,
    attempt: int, stage_scope_id: str | None,
) -> dict[str, int | str] | None:
    """Measure exact UTF-8 sizes of the final host execute arguments, or warn and return None.

    Backend-internal prompts, schemas, history and transport envelopes are
    outside this boundary. An unsent system argument has no byte count.
    """
    try:
        measured: dict[str, int | str] = {
            "attempt": attempt,
            "user_utf8_bytes": len(prompt.encode("utf-8")),
            "system_disposition": (
                "supplied" if system_instructions is not None else
                "absent" if system_instructions_supported else "unsupported"
            ),
        }
        if system_instructions is not None:
            measured["system_utf8_bytes"] = len(system_instructions.encode("utf-8"))
        if stage_scope_id is not None:
            measured["stage_scope_id"] = stage_scope_id
        return measured
    except Exception as exc:  # noqa: BLE001 - measurement must never interrupt agent work
        print_warning(console, f"Host dispatch measurement failed: {type(exc).__name__}")
        return None


async def _run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    *,
    phase: DaydreamPhase,
    budget: BudgetLedger,
    output_schema: dict[str, Any] | None = None,
    result_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[Text], Any] | None = None,
    continuation: ContinuationToken | None = None,
    agents: dict[str, AgentDefinition] | None = None,
    read_only: bool = False,
    persist_session: bool = True,
    validate_structured_output: bool = True,
    require_full_schema: bool = False,
    sanctioned_inputs: PreparedSanctionedInputs | None = None,
    run_context: RunContext,
    review_evidence: ReviewEvidence | None = None,
    review_instructions: str | None = None,
    finalization: bool = False,
    tools_disabled: bool = False,
    review_stage_scope_id: str | None = None,
    host_tools: HostToolBinding | None = None,
) -> tuple[str | Any, ContinuationToken | None, str | None]:
    """Execute attempts under one budget ledger.

    Recorder observation precedes presentation and supervision. Backend failures
    may retry; supervisor failures never do. Each attempt owns and closes its
    stream. A retry after a retryable provider failure refunds that attempt's tool
    charges; elapsed time is never refunded. A deadline/tool/veto stop returns
    partial output and its reason, with bounded cleanup; a pre-dispatch or
    pre-backoff stop discards failed partials.

    Structured primary and extracted results share the salvage gate. Callers that
    validate downstream may opt out; bounded review finalization is handled by
    the public run_agent wrapper.
    """
    output_parts: list[str] = []
    assistant_turn_bytes, assistant_turn_overflow = 0, False
    completed_assistant_text: str | None = None
    completed_assistant_overflow = False
    structured_result: Any = None
    result_continuation: ContinuationToken | None = None
    aborted_reason: str | None = None
    investigation = budget.investigation
    tool_supervisor = get_registry().tool_supervisor_if_registered()

    with run_context.backend_registration(backend):
        try:
            # Open Invocation scope when a recorder is active; nullcontext keeps the
            # with-shape uniform otherwise (CORE-09 no-op). D-19: no ATIF construction
            # here — only inv.observe()/inv.observe_user_step() against the recorder.
            recorder = get_current_recorder()
            settings = _resolve_retry_settings(backend)
            max_attempts = settings.attempts
            base_delay = settings.base_delay_s
            max_delay = settings.max_delay_s

            # One absolute ledger deadline covers attempts and backoff.
            invocation_start = clock.monotonic()
            effective_deadline = budget.deadline
            # Telemetry counters for the single invocation: only dispatched
            # attempts and time actually spent inside them are charged, so
            # backend_s + backoff_s <= elapsed_s holds.
            telemetry = _RetryTelemetry()
            cleanup_elapsed_s = 0.0
            # A deadline can end the ladder on either side of a dispatch. The two
            # shapes are not interchangeable: an in-flight attempt keeps its
            # partials, while the pre-dispatch (loop-top) and pre-backoff breaks
            # reset them, and only the latter two misreport "kept".
            partials_discarded_by_deadline = False
            # Set when the invocation already emitted its one stop record, so the
            # post-loop deadline emitter never writes a second, contradicting one.
            stop_recorded = False
            # Calls a failed attempt charged, refunded only when a retry actually dispatches.
            pending_refund = 0

            def _emit_ladder_stop(
                stop_reason: str, limit_expired: str = "retry_ladder"
            ) -> None:
                """Record retry overhead and its terminal reason without affecting the raised failure.

                Counts include dispatched/in-flight retries and backoff, excluding initial useful
                work. Persist only durations and codes, never monotonic timestamps or deadlines.
                """
                now = clock.monotonic()
                try:
                    stop: dict[str, Any] = {
                        "limit_expired": limit_expired,
                        "elapsed_s": now - invocation_start,
                        "backend_s": telemetry.retry_backend_s + telemetry.pending_retry_s(now),
                        "backoff_s": telemetry.backoff_s,
                        "attempts": telemetry.retry_attempts,
                        "retry_stop_reason": stop_reason,
                        "circuit_state": run_context.outage_circuit.state(),
                        "partial_edit_handling": "discarded",
                    }
                    record_retry_stop(stop)
                    if recorder is not None:
                        recorder.emit_agent_budget_stop(phase, **stop)
                except Exception:  # noqa: BLE001 - telemetry must never break the run
                    _logger.exception("failed to record agent retry-ladder stop")

            _logger.debug("invocation ledger %s: deadline=%s limit=%s", budget.owner, effective_deadline, budget.limit)

            for attempt in range(max_attempts + 1):
                # Reset accumulated state BEFORE the deadline checks: a failed
                # attempt's partial output must never leak into the invocation
                # return when the pre-dispatch break (or the retry branch's
                # pre-backoff break) ends the ladder, so the caller sees only
                # output from the attempt that actually completed the turn.
                output_parts = []
                assistant_turn_bytes, assistant_turn_overflow = 0, False
                completed_assistant_text = None
                completed_assistant_overflow = False
                structured_result = None
                native_output = False
                result_continuation = None
                remaining_calls = budget.remaining_tool_calls
                if remaining_calls is not None:
                    remaining_calls += pending_refund  # what this retry gets once it dispatches
                refund_point = budget.charged
                if review_evidence is not None:
                    review_evidence.reset()
                budget_reason: str | None = None
                # Never dispatch an attempt once the deadline is spent: the
                # ladder stops here with the reset state.
                if budget.expired():
                    aborted_reason = "wall_budget_exceeded"
                    if host_tools is not None:
                        host_tools.close_authority()
                    partials_discarded_by_deadline = True
                    # When retries or backoff spent time, preserve that overhead in a ladder-stop record.
                    if telemetry.spent_retry_overhead:
                        _emit_ladder_stop("retry_deadline_exhausted", limit_expired="invocation_wall_budget")
                        stop_recorded = True
                    break
                # The ledger refuses an attempt whose remaining calls cannot fit its minimum work.
                if remaining_calls is not None and remaining_calls < budget.start_floor:
                    aborted_reason = "tool_call_budget_exceeded"
                    if host_tools is not None:
                        host_tools.close_authority()
                    break
                # Dispatch bookkeeping: the opening attempt is useful work, every
                # later one is retry overhead.
                display = AgentDisplay(
                    console, run_context.policy, progress_callback, structured=output_schema is not None,
                )

                try:
                    if artifact_session_active():
                        assert_model_cwd_clean(cwd)
                    if sanctioned_inputs is not None:
                        sanctioned_inputs.revalidate(backend, cwd, read_only)
                    dispatch_time = clock.monotonic()
                    if effective_deadline is not None and dispatch_time >= effective_deadline:
                        aborted_reason = "wall_budget_exceeded"
                        if host_tools is not None:
                            host_tools.close_authority()
                        partials_discarded_by_deadline = True
                        if telemetry.spent_retry_overhead:
                            _emit_ladder_stop("retry_deadline_exhausted", limit_expired="invocation_wall_budget")
                            stop_recorded = True
                        break
                    telemetry.start_attempt(dispatch_time, retry=attempt > 0)
                    # The retry is a fresh conversation, so it regains the failed attempts' calls.
                    budget.refund(budget.charged - pending_refund)
                    pending_refund = 0
                    refund_point = budget.dispatch()
                    execute_kwargs: dict[str, Any] = {
                        "agents": agents,
                        "read_only": read_only,
                    }
                    if host_tools is not None:
                        execute_kwargs["host_tools"] = host_tools
                    if finalization and getattr(backend, "supports_finalization", False):
                        execute_kwargs["finalization"] = True
                    if tools_disabled and getattr(backend, "supports_tools_disabled", False):
                        execute_kwargs["tools_disabled"] = True
                    # Host-tool adapters serve committed-review instructions as the system prompt.
                    supports_review_instructions = (host_tools is not None
                                                    or bool(getattr(backend, "supports_review_instructions", False)))
                    if review_instructions and supports_review_instructions:
                        execute_kwargs["review_instructions"] = review_instructions
                    if not validate_structured_output and getattr(
                        backend, "supports_structured_output_opt_out", False
                    ):
                        execute_kwargs["validate_structured_output"] = False
                    if not persist_session:
                        execute_kwargs["persist_session"] = False
                    if getattr(backend, "supports_budget_preamble", False):
                        # The absolute ledger deadline already reflects preparation,
                        # earlier stages and retry/backoff time.
                        execute_kwargs["deadline"] = effective_deadline
                        execute_kwargs["tool_call_budget"] = remaining_calls
                    attempt_prompt = prompt
                    if investigation and attempt > 0:
                        retry_budget_update = (
                            "\n\nRetry dispatch-time snapshot (supersedes the earlier invocation allocation): "
                            f"current invocation allocation: {remaining_calls} tool calls. "
                            "Failed-attempt calls were refunded, elapsed time was not; "
                            "future-stage reserves remain excluded."
                        )
                        attempt_prompt += retry_budget_update
                        if "review_instructions" in execute_kwargs:
                            execute_kwargs["review_instructions"] += retry_budget_update
                    event_iter = backend.execute(
                        cwd, attempt_prompt, output_schema, continuation,
                        **execute_kwargs,
                    )
                    invocation_cm: Any = (
                        recorder.invocation(phase=phase) if recorder is not None else nullcontext(None)
                    )
                    lifecycle = BackendLifecycleCollector(
                        attempt=attempt + 1, started_at=dispatch_time,
                        wall_allowance_s=(
                            max(0.0, effective_deadline - dispatch_time) if effective_deadline is not None else None
                        ),
                        deadline=effective_deadline,
                    )

                    def record_lifecycle(snapshot: BackendLifecycleSnapshot) -> None:
                        if inv is not None:
                            inv.record_backend_lifecycle(snapshot)
                        observed.record_backend_lifecycle(snapshot)

                    event_stream_scope = _EventStreamScope(
                        event_iter, lifecycle=lifecycle, record_lifecycle=record_lifecycle,
                        host_tools=host_tools,
                    )

                    async with (
                        attempt_scope(attempt + 1) as observed,
                        invocation_cm as inv,
                        event_stream_scope,
                    ):
                        if inv is not None:
                            inv.observe_backend_lifecycle(lifecycle)
                            inv.observe_user_step(prompt=attempt_prompt)
                        dispatch = _measure_host_dispatch(
                            prompt=attempt_prompt,
                            system_instructions=execute_kwargs.get("review_instructions"),
                            system_instructions_supported=supports_review_instructions,
                            attempt=attempt + 1, stage_scope_id=review_stage_scope_id,
                        )
                        if dispatch is not None:
                            observed.record_host_dispatch(dispatch)
                            if inv is not None:
                                inv.record_host_dispatch(dispatch)

                        # Per-invocation abort controls live here so both backends
                        # are covered without a backend-signature change: the tool-call
                        # ceiling and supervisor veto break in-loop, while the deadline
                        # is read per event and also backstopped by a real-time
                        # move_on_after over the time the effective deadline has left.
                        remaining_s = (
                            max(effective_deadline - clock.monotonic(), 0.0)
                            if effective_deadline is not None
                            else None
                        )
                        wall_scope: Any = (
                            anyio.move_on_after(remaining_s) if remaining_s is not None else nullcontext()
                        )

                        with wall_scope:
                            async for event in event_stream_scope.events():
                                # Host tools charge the ledger at admission, before Pi emits the start.
                                over_limit = (isinstance(event, ToolStartEvent) and host_tools is None
                                              and not budget.charge())
                                # The sole telemetry observer runs before UI callbacks,
                                # supervision and budgets can interrupt event handling.
                                observed.observe(event)
                                # Parser/transport evidence and the invocation ledger must be forwarded
                                # before any branch-specific break; _dispatch is UI-free.
                                if inv is not None:
                                    inv.observe(event)
                                # After expiry, supervise/charge received native starts;
                                # reject completions/checkpoints.
                                deadline_expired = budget.expired()
                                if deadline_expired and not (native_output and isinstance(event, ToolStartEvent)):
                                    budget_reason = "wall_budget_exceeded"
                                    break
                                if review_evidence is not None:
                                    review_evidence.observe(event)
                                if isinstance(event, RequestEvent):
                                    # A host-tool submission is the only output; prose never becomes it.
                                    native_output = host_tools is not None or (
                                        isinstance(event.config, PiRequestConfig)
                                        and event.output_schema is not None
                                        and event.config.schema_emulated is False)
                                elif isinstance(event, TextEvent) and not native_output:
                                    if not investigation:
                                        output_parts.append(event.text)
                                    elif event.text:
                                        size = len(event.text.encode())
                                        if assistant_turn_bytes + size > 128 * 1024:
                                            assistant_turn_overflow = True
                                        else:
                                            output_parts.append(event.text)
                                            assistant_turn_bytes += size
                                elif isinstance(event, TurnEndEvent) and output_schema is not None:
                                    completed_assistant_text = ''.join(output_parts)
                                    completed_assistant_overflow = assistant_turn_overflow
                                    output_parts = []
                                    assistant_turn_bytes, assistant_turn_overflow = 0, False
                                elif isinstance(event, ResultEvent):
                                    structured_result = event.structured_output
                                    result_continuation = event.continuation
                                if isinstance(event, ToolStartEvent):
                                    if tool_supervisor is not None:
                                        try:
                                            # Use the strip-only entry point for start-anchored deny patterns.
                                            supervisor_input = event.input
                                            if event.name == "shell" and isinstance(event.input, dict):
                                                from daydream.backends.codex import supervisor_shell_command

                                                command = event.input.get("command")
                                                if isinstance(command, str):
                                                    supervisor_input = dict(event.input)
                                                    supervisor_input["command"] = supervisor_shell_command(command)
                                            decision = tool_supervisor(event.name, supervisor_input, phase=phase)
                                        except Exception as exc:  # noqa: BLE001 - policy failures must propagate
                                            raise _ToolSupervisorFailure(exc) from exc
                                        if decision.veto:
                                            if recorder is not None:
                                                recorder.emit_tool_veto(
                                                    event.name, decision.reason, phase=phase
                                                )
                                            budget_reason = f"tool_vetoed:{event.name}"
                                            break

                                    if deadline_expired:
                                        budget_reason = "wall_budget_exceeded"
                                        break
                                    if over_limit:
                                        budget_reason = "tool_call_budget_exceeded"
                                        break

                                await display.observe(event)

                            if event_stream_scope.ledger_stop is not None:
                                budget_reason = event_stream_scope.ledger_stop
                            await display.flush()

                        # Abort handling: a spent deadline cut the loop, the wall
                        # backstop cancelled it, a quantitative tool ceiling fired, or
                        # a supervisor veto broke out. Mark the ATIF turn aborted and
                        # let the invocation's event-stream scope close its resources
                        # before returning partial output.
                        wall_cancelled = bool(getattr(wall_scope, "cancelled_caught", False))
                        if budget_reason is None and wall_cancelled:
                            budget_reason = "wall_budget_exceeded"
                        aborted_reason = budget_reason
                        if budget_reason is not None:
                            if host_tools is not None:
                                host_tools.close_authority()
                            observed.abort(budget_reason)
                            # Cleanup can hang on a backend subprocess that never
                            # exits. Bound it with a shielded grace so it cannot
                            # extend or abort the already-captured partial result;
                            # the grace time is measured on the clock seam and
                            # reported separately from the invocation's elapsed_s.
                            cleanup_started_at = clock.monotonic()
                            with anyio.move_on_after(
                                BUDGET_CLEANUP_GRACE_S, shield=True
                            ) as cleanup_scope:
                                # Stamp the abort reason BEFORE aclose(): a hung
                                # backend subprocess can block the close until the
                                # grace fires, and the turn must still carry its
                                # stop_reason and partial mark.
                                if inv is not None:
                                    inv.mark_aborted(budget_reason)
                                    inv.observe(TurnEndEvent())
                                await event_stream_scope.aclose()
                            cleanup_elapsed_s = clock.monotonic() - cleanup_started_at
                            if cleanup_scope.cancel_called:
                                _logger.warning(
                                    "post-expiry cleanup exceeded its %ss grace; "
                                    "continuing with the captured partial result",
                                    BUDGET_CLEANUP_GRACE_S,
                                )
                            await display.aborted(budget_reason)
                        display.finish()

                    # A completed, un-aborted attempt closes the run circuit;
                    # budget aborts share this break but are not a success.
                    if budget_reason is None:
                        run_context.outage_circuit.record_success()
                    break  # success — exit the retry loop

                except _ToolSupervisorFailure:
                    raise
                except Exception as exc:
                    await display.flush()
                    # Classify once per failure, before the per-failure cap: a
                    # permanent condition (bad credentials, an unknown model, a
                    # schema rejection) wins even when the message also carries
                    # a transient token or the exception advertises a higher
                    # retry cap.
                    classification = classify_failure(exc)
                    # A completed turn that only lacks its output is the provider answering, not an outage.
                    completed = getattr(exc, "reason_code", None) == "missing_output"
                    if completed:
                        run_context.outage_circuit.record_success()
                    exception_max_retries = min(
                        max_attempts, getattr(exc, "max_retries", max_attempts)
                    )
                    if host_tools is not None and (host_tools.possibly_accepted or host_tools.fatal_error is not None):
                        # A terminal host failure cannot be corrected by replay; a
                        # callback may also have installed state before settlement failed.
                        exception_max_retries = 0
                    if attempt < exception_max_retries and classification.retries_allowed:
                        # A spent deadline permits no further sleep or dispatch. Discard failed partials
                        # and record any retry overhead already spent.
                        if budget.expired():
                            output_parts = []
                            assistant_turn_bytes, assistant_turn_overflow = 0, False
                            completed_assistant_text = None
                            completed_assistant_overflow = False
                            structured_result = None
                            result_continuation = None
                            aborted_reason = "wall_budget_exceeded"
                            partials_discarded_by_deadline = True
                            if telemetry.spent_retry_overhead:
                                _emit_ladder_stop("retry_deadline_exhausted", limit_expired="invocation_wall_budget")
                                stop_recorded = True
                            break
                        # The circuit gates retries only; stale open state cannot block a first attempt.
                        # Denial propagates the current failure with its retryable attribute intact.
                        circuit_now = clock.monotonic()
                        admission = run_context.outage_circuit.admit_retry(circuit_now)
                        opened_here = (
                            False
                            if completed or admission.allowed
                            and admission.state == CIRCUIT_HALF_OPEN
                            else run_context.outage_circuit.record_failure(circuit_now)
                        )
                        if not admission.allowed:
                            _emit_ladder_stop("circuit_open")
                            raise
                        # The delay and the ladder's remaining bound are decided by
                        # one pure callable: a server hint replaces jitter but never
                        # extends the deadline, and a hint longer than the remaining
                        # deadline stops the ladder exactly as exhaustion does.
                        hint = _retry_hint(exc)
                        delay, delay_stop = _plan_retry_delay(
                            attempt=attempt,
                            base_delay_s=base_delay,
                            max_delay_s=max_delay,
                            deadline_remaining_s=budget.remaining_s(),
                            hint=hint,
                        )
                        if delay_stop is not None:
                            budget.stop("time")
                            _emit_ladder_stop(delay_stop)
                            raise
                        # A provider-advertised wait is distinguished in the notice so an
                        # operator can tell a server-directed recovery from jitter backoff;
                        # the printed value is the already-admitted delay, never the raw header.
                        hint_note = " (server-advertised wait)" if hint is not None else ""
                        retry_msg = (
                            f"Backend error ({type(exc).__name__}), retrying "
                            f"attempt {attempt + 2}/{exception_max_retries + 1} after {delay:.1f}s{hint_note}..."
                        )
                        await display.retry(retry_msg)
                        # The event-stream scope has already closed only this failed
                        # invocation. Backend-wide cancel() is reserved for shutdown.
                        display.tools.discard_all()
                        # Charge the failed attempt's backend time up to the
                        # backoff point: the sleep that follows is retry backoff
                        # (counted in backoff_s) and must not also land in
                        # backend_s, or backend_s + backoff_s would overcount.
                        telemetry.charge_attempt(clock.monotonic())
                        await anyio.sleep(delay)
                        telemetry.backoff_s += delay
                        # Recheck after backoff because a sibling may have opened the circuit.
                        # Do not consume a granted half-open probe twice; the tripper retains its retry.
                        if (
                            not opened_here
                            and admission.state == CIRCUIT_CLOSED
                            and not run_context.outage_circuit.admit_retry(
                                clock.monotonic()
                            ).allowed
                        ):
                            _emit_ladder_stop("circuit_open")
                            raise
                        # Every gate passed. This attempt's calls are refunded only if
                        # the retry dispatches; time is never refunded.
                        pending_refund += budget.charged - refund_point
                        continue
                    if classification.retries_allowed and exception_max_retries > 0:
                        _emit_ladder_stop("retry_attempts_exhausted")
                    raise
                finally:
                    primary_error = sys.exception()
                    try:
                        display.finish_log_text()
                    except Exception as exc:  # noqa: BLE001 - preserve the active attempt failure
                        if primary_error is None:
                            raise
                        primary_error.add_note(f"Log text unavailable ({type(exc).__name__[:64]})")
                    finally:
                        if telemetry.attempt_started_at is not None:
                            # Post-stop cleanup (the bounded backend aclose) is not
                            # dispatched-attempt time: exclude it so backend_s keeps
                            # its documented 'time inside dispatched attempts'
                            # semantics and backend_s + backoff_s can never exceed
                            # elapsed_s (which already subtracts cleanup_elapsed_s).
                            telemetry.charge_attempt(clock.monotonic(), cleanup_elapsed_s=cleanup_elapsed_s)

            if (dimension := _STOP_DIMENSIONS.get(aborted_reason)) is not None:
                budget.stop(dimension)
            # Emit at most one stop record. In-flight deadline stops retain partials;
            # pre-dispatch stops discard them. Recorder failure cannot change the return value.
            if (
                aborted_reason == "wall_budget_exceeded"
                and not stop_recorded
                and recorder is not None
            ):
                try:
                    recorder.emit_agent_budget_stop(
                        phase,
                        limit_expired="invocation_wall_budget",
                        elapsed_s=clock.monotonic() - invocation_start - cleanup_elapsed_s,
                        backend_s=telemetry.backend_s,
                        backoff_s=telemetry.backoff_s,
                        attempts=telemetry.attempts_dispatched,
                        cleanup_elapsed_s=cleanup_elapsed_s,
                        retry_stop_reason=None,
                        circuit_state=run_context.outage_circuit.state(),
                        partial_edit_handling=(
                            "discarded" if partials_discarded_by_deadline else "kept"
                        ),
                    )
                except Exception:  # noqa: BLE001 - telemetry must never break the run
                    _logger.exception("failed to record agent budget stop")

        except _ToolSupervisorFailure as exc:
            original = exc.original
            detail = exception_text(original) or ""
            diagnostic = (
                f"{type(original).__name__}: {detail}" if detail else type(original).__name__
            )
            print_error(console, "Extension Failure", sanitize_verbose_message(diagnostic))
            # Outer handlers may print str(exc) without redaction. Rebuild instead of only
            # changing args: custom str/repr and OSError fields can retain credentials.
            raise _scrubbed_supervisor_error(original) from original
        except Exception as exc:
            category = getattr(exc, "category", None)
            msg = (exception_text(exc) or "").strip()
            diagnostic = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
            if isinstance(category, str):
                diagnostic += f" [{category}]"
            # Error messages can embed secrets (a leaked env var, an API key in a
            # provider error); sanitize (redact AND neutralize terminal control
            # codes) at this host boundary like every other surfaced text, so a
            # hostile exception message cannot paint or escape the operator's
            # terminal in the fatal path.
            print_error(console, "Backend Execution Error", sanitize_verbose_message(diagnostic))
            raise
        except asyncio.CancelledError:
            # The attempt stream already closed during unwinding. Cancellation
            # of one fanout task must not terminate a shared backend's siblings.
            if host_tools is not None:
                host_tools.close_authority()
            raise
        except BaseException:
            # Shutdown path: SIGINT (KeyboardInterrupt) / task cancellation
            # (CancelledError) are BaseException, so the generic `except Exception`
            # above never sees them. Deterministically reap the tracked subprocesses
            # via backend.cancel() before unwinding.
            try:
                # AnyIO task groups keep scopes cancelled at each await; existing bounded grace lets teardown reap
                # without masking the original cancellation.
                with anyio.move_on_after(BUDGET_CLEANUP_GRACE_S, shield=True) as shutdown_scope:
                    await backend.cancel()
                if shutdown_scope.cancel_called:
                    _logger.warning("backend shutdown exceeded its %ss grace", BUDGET_CLEANUP_GRACE_S)
            except Exception:  # cancel() must not mask the original signal
                _logger.exception("backend.cancel() failed during shutdown")
            raise


    if output_schema is not None:
        raw = (''.join(output_parts) if output_parts or assistant_turn_overflow
               else completed_assistant_text or '')
        text_overflow = (assistant_turn_overflow if output_parts or assistant_turn_overflow
                         else completed_assistant_overflow)
    else:
        raw, text_overflow = ''.join(output_parts), False

    if native_output and aborted_reason is not None:
        structured_result = None
        if review_evidence is not None:
            review_evidence.checkpoint = None
    output = resolve_output(
        structured_result, raw, native_output=native_output, text_overflow=text_overflow,
        output_schema=(result_schema if result_schema is not None else
                       review_evidence.schema if investigation and review_evidence is not None
                       else output_schema), validate_structured_output=validate_structured_output,
        require_full_schema=require_full_schema, staged=investigation,
    )
    return output, result_continuation, aborted_reason
