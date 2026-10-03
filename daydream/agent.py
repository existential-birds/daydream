"""Agent interaction and backend management."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import anyio
from jsonschema import Draft202012Validator
from rich.console import Console

if TYPE_CHECKING:
    from claude_agent_sdk.types import AgentDefinition
    from rich.text import Text

from daydream import clock
from daydream.agent_retry import _plan_retry_delay, _resolve_retry_settings, _retry_hint, _RetryTelemetry
from daydream.artifact_visibility import ArtifactVisibilityError, artifact_session_active, assert_model_cwd_clean
from daydream.backends import (
    AgentEventStream,
    Backend,
    ContinuationToken,
    DiagnosticEvent,
    ResultEvent,
    TextEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.backends.codex import supervisor_shell_command
from daydream.config import BUDGET_CLEANUP_GRACE_S
from daydream.diagnostics import exception_text, sanitize_verbose_message
from daydream.extensions import get_registry
from daydream.json_utils import extract_json
from daydream.observability.spans import agent_scope, attempt_scope
from daydream.outage_circuit import CIRCUIT_CLOSED, CIRCUIT_HALF_OPEN
from daydream.prompt_budget import PreparedSanctionedInputs
from daydream.prompts.grounding import REVIEW_STOPPING_GUIDANCE
from daydream.retry_policy import (
    FailureClass,
    RetryRecoveryBudget,
    classify_failure,
)
from daydream.review_budget import ReviewLimits, review_deadline, review_limits_for_scope
from daydream.review_evidence import FinalizationContext, ReviewEvidence
from daydream.run_context import (
    RunContext,
    bind_run_context,
    current_run_context,
    resolve_gate as resolve_gate,
    resolve_run_context,
)
from daydream.trajectory import DaydreamPhase, get_current_recorder, redact_text
from daydream.ui import NEON_THEME, print_error
from daydream.ui.agent_stream import AgentDisplay

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
    """Preserve type name/scrubbed message when custom str/repr or OSError fields cannot be safely rebuilt."""

    def __init__(self, original_type_name: str, message: str) -> None:
        self.original_type_name = original_type_name
        self.retryable: bool = False
        super().__init__(message)


def _scrubbed_supervisor_error(original: BaseException) -> BaseException:
    """Rebuild scrubbed exception args, preserving retryable; unsafe str/repr or OSError fields use
    a credential-free fallback suitable for outer handlers that print without further redaction.
    """
    scrubbed_args = tuple(redact_text(a) if isinstance(a, str) else a for a in original.args)
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

    def __init__(self, event_iter: AgentEventStream) -> None:
        self.event_iter = event_iter
        self._closed = False

    async def __aenter__(self) -> AgentEventStream:
        return self.event_iter

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the invocation once without masking its outcome."""
        if self._closed:
            return
        self._closed = True
        try:
            await self.event_iter.aclose()
        except Exception:  # noqa: BLE001 - cleanup must not mask the invocation outcome
            pass


class _LogRedactingConsole(Console):
    """Redact string console payloads in verbose mode, including UI calls outside agent-event emission."""

    def print(self, *objects: Any, **kwargs: Any) -> None:
        context = current_run_context()
        if context is not None and context.policy.log_mode:
            objects = tuple(redact_text(obj) if isinstance(obj, str) else obj for obj in objects)
        super().print(*objects, **kwargs)


console = _LogRedactingConsole(theme=NEON_THEME)


def detect_test_success(output: str) -> bool:
    """Parse pass/fail counts (including N tests failed), then tooling/agent success sentinels;
    positive failures and hard error signals always win.
    """
    if not output:
        return False

    output_lower = output.lower()

    # Scan every failure count, including collection errors; a later failure beats an earlier zero.
    failed_counts = [
        int(match.group(1).replace(",", ""))
        for match in re.finditer(r"(\d[\d,]*)\s+(?:tests?\s+)?(?:fail(?:ed|ures?)|errors?)\b", output_lower)
    ]
    passed_counts = [
        int(match.group(1).replace(",", ""))
        for match in re.finditer(r"(\d[\d,]*)\s+(?:tests?\s+)?passed\b", output_lower)
    ]

    if any(count > 0 for count in failed_counts):
        return False

    # Hard negative signals win over success sentinels — a late traceback must not be
    # masked by an earlier "all tests pass" phrase.
    error_patterns = [
        r"tests? failing",
        r"test failure",
        r"assertion error",
        r"traceback",
    ]
    for pattern in error_patterns:
        if re.search(pattern, output_lower):
            return False

    # Explicit sentinels emitted by tooling / the test agent.
    success_sentinels = [
        r"test result:\s*ok",  # cargo / rust native
        r"tests?\s+pass(?:ed)?\s*[✅✓]",  # agent emoji summary ("Tests PASS ✅")
        r"all \d+ tests? passed",
        r"tests? passed successfully",
        r"test suite passed",
        r"all tests pass",
        r"no (?:test )?failures?",
        r"\b0\s+failures?\b",
        r"\d+\s+passed(?:,\s*\d+\s+(?:deselected|skipped|xfailed))*(?:,\s*\d+\s+warnings?)?",
    ]
    for pattern in success_sentinels:
        if re.search(pattern, output_lower):
            return True

    # Positive failure counts were excluded above. A positive passed count is
    # enough here; bare "passed" without a count remains a conservative False.
    return any(count > 0 for count in passed_counts)


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


class StructuredOutputFailure(str):
    """Text fallback carrying the host's rejected structured-output witness."""

    reason: str

    def __new__(cls, text: str, reason: str) -> "StructuredOutputFailure":
        value = super().__new__(cls, text)
        value.reason = reason
        return value


def _validates_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Return whether ``value`` validates against ``schema`` (shape + required)."""
    return not any(Draft202012Validator(schema).iter_errors(value))


def _salvageable(value: Any, schema: dict[str, Any]) -> bool:
    """Accept full schema validity or required keys with lists in required array slots;
    callers validate salvageable nested records downstream.
    """
    if _validates_schema(value, schema):
        return True
    if not isinstance(value, dict):
        return False
    required = schema.get("required")
    if not isinstance(required, list):
        return False
    properties = schema.get("properties", {})
    for key in required:
        if key not in value:
            return False
        prop = properties.get(key)
        if isinstance(prop, dict) and prop.get("type") == "array":
            if not isinstance(value[key], list):
                return False
    return True


async def run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    *,
    phase: DaydreamPhase,
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[Text], Any] | None = None,
    continuation: ContinuationToken | None = None,
    agents: dict[str, AgentDefinition] | None = None,
    max_turns: int | None = None,
    read_only: bool = False,
    persist_session: bool = True,
    wall_budget_s: float | None = None,
    deadline: float | None = None,
    tool_call_budget: int | None = None,
    retry_recovery_allowance_s: float | None = None,
    validate_structured_output: bool = True,
    require_full_schema: bool = False,
    sanctioned_inputs: PreparedSanctionedInputs | None = None,
    run_context: RunContext | None = None,
    review_limits: ReviewLimits | None = None,
    finalization_context: FinalizationContext | None = None,
    tools_disabled: bool = False,
    review_system_instructions: str | None = None,
) -> tuple[str | Any, ContinuationToken | None, str | None]:
    """Trace the logical result returned to a phase; the invocation executor owns retries, budgets, and recording."""
    if tools_disabled and not getattr(backend, "supports_tools_disabled", False):
        raise NotImplementedError(f"{type(backend).__name__} does not support tools_disabled")
    if review_system_instructions is not None:
        if not tools_disabled:
            raise ValueError("review_system_instructions requires tools_disabled=True")
        if not getattr(backend, "supports_review_instructions", False):
            raise NotImplementedError(f"{type(backend).__name__} does not support review_instructions")
    if sanctioned_inputs is not None:
        # Callers may already have rendered this suffix. Move it after the
        # host's budget/finalization instructions without duplicating inputs.
        rendered_suffix = sanctioned_inputs.render_prompt("")
        if rendered_suffix and prompt.endswith(rendered_suffix):
            prompt = prompt.removesuffix(rendered_suffix)
    context = resolve_run_context(run_context)
    evidence = ReviewEvidence(output_schema) if review_limits is not None else None
    review_instructions = review_system_instructions
    hard_deadline = deadline
    shared: float | None = None
    if review_limits is not None:
        review_limits = review_limits_for_scope(review_limits)
        started = clock.monotonic()
        bounds = [started + review_limits.investigation_s + review_limits.finalization_s]
        if deadline is not None:
            bounds.append(deadline)
        if wall_budget_s is not None:
            bounds.append(started + wall_budget_s)
        shared = review_deadline(discovery=review_limits.discovery)
        if shared is not None:
            bounds.append(shared)
        hard_deadline = min(bounds)
        deadline = max(
            started, min(started + review_limits.investigation_s, hard_deadline - review_limits.finalization_s),
        )
        investigation_allowance = max(0.0, deadline - started)
        tool_call_budget = min(tool_call_budget, review_limits.tool_calls) if tool_call_budget is not None else (
            review_limits.tool_calls
        )
        budget_instructions = (
            f"Investigation allowance: at most {investigation_allowance:g} seconds and "
            f"{tool_call_budget} tool calls. " + REVIEW_STOPPING_GUIDANCE
        )
        prompt += "\n\n" + budget_instructions
        review_instructions = "\n\n".join(
            item for item in (review_system_instructions, budget_instructions) if item
        )
    if sanctioned_inputs is not None:
        prompt = sanctioned_inputs.render_prompt(prompt)
    backend_name = type(backend).__name__.removesuffix("Backend").lower()
    with bind_run_context(context), agent_scope(
        phase.value, backend=backend_name, model=backend.model
    ) as observed:
        observed.content("traceloop.entity.input", {"prompt": prompt, "output_schema": output_schema})
        try:
            result = await _run_agent(
                backend, cwd, prompt, phase=phase, output_schema=output_schema, progress_callback=progress_callback,
                continuation=continuation, agents=agents, max_turns=max_turns, read_only=read_only,
                persist_session=persist_session, wall_budget_s=wall_budget_s, deadline=deadline,
                tool_call_budget=tool_call_budget,
                retry_recovery_allowance_s=retry_recovery_allowance_s,
                validate_structured_output=validate_structured_output,
                require_full_schema=require_full_schema,
                sanctioned_inputs=sanctioned_inputs,
                run_context=context,
                review_evidence=evidence,
                review_instructions=review_instructions,
                tools_disabled=tools_disabled,
            )
        except Exception as exc:
            from daydream.review_result import ReasonCode, reason_for_exception
            if (evidence is None or not evidence.valid(evidence.checkpoint)
                    or reason_for_exception(exc) != ReasonCode.MODEL_BUDGET_EXHAUSTION):
                raise
            # Provider turn exhaustion retains this invocation's strictly validated
            # checkpoint; it cannot establish completed investigation coverage.
            result = (evidence.checkpoint, None, "error_max_turns")
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
                    finalized, _, final_reason = await _run_agent(
                        backend, cwd, final_prompt, phase=phase,
                        output_schema=output_schema, progress_callback=progress_callback,
                        require_full_schema=require_full_schema,
                        read_only=read_only, persist_session=persist_session, finalization=True,
                        deadline=min(hard_deadline, clock.monotonic() + review_limits.finalization_s),
                        tool_call_budget=0,
                        retry_recovery_allowance_s=retry_recovery_allowance_s,
                        sanctioned_inputs=sanctioned_inputs, run_context=context,
                    )
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
        observed.output(result[0])
        observed.finish(1 if result[2] else 0, reason=result[2])
        return result


async def _run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    *,
    phase: DaydreamPhase,
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[Text], Any] | None = None,
    continuation: ContinuationToken | None = None,
    agents: dict[str, AgentDefinition] | None = None,
    max_turns: int | None = None,
    read_only: bool = False,
    persist_session: bool = True,
    wall_budget_s: float | None = None,
    deadline: float | None = None,
    tool_call_budget: int | None = None,
    retry_recovery_allowance_s: float | None = None,
    validate_structured_output: bool = True,
    require_full_schema: bool = False,
    sanctioned_inputs: PreparedSanctionedInputs | None = None,
    run_context: RunContext,
    review_evidence: ReviewEvidence | None = None,
    review_instructions: str | None = None,
    finalization: bool = False,
    tools_disabled: bool = False,
) -> tuple[str | Any, ContinuationToken | None, str | None]:
    """Execute attempts under one deadline, retry allowance, and tool budget. Observe recorder
    events before display/supervision; backend failures may retry, supervisor failures never do.
    Each attempt closes its own stream. In-flight deadline/tool/veto stops retain partials with
    bounded cleanup; pre-dispatch/backoff stops discard failed partials. Structured/extracted
    results share the salvage gate unless disabled; run_agent owns review finalization.
    """
    output_parts: list[str] = []
    structured_result: Any = None
    result_continuation: ContinuationToken | None = None
    aborted_reason: str | None = None
    evidence_incomplete = False
    tool_supervisor = get_registry().tool_supervisor_if_registered()

    with run_context.backend_registration(backend):
        try:
            # Recorder invocation scopes only observe events; nullcontext preserves the shape when recording is
            # disabled.
            recorder = get_current_recorder()
            settings = _resolve_retry_settings(backend, retry_recovery_allowance_s)
            max_attempts = settings.max_attempts
            base_delay = settings.base_delay_s
            max_delay = settings.max_delay_s
            resolved_allowance = settings.allowance_s

            # One absolute bound covers attempts and backoff; caller deadline wins ties.
            invocation_start = clock.monotonic()
            effective_deadline = deadline
            limit_expired = "caller_deadline" if deadline is not None else None
            if wall_budget_s is not None:
                wall_deadline = invocation_start + wall_budget_s
                if deadline is None or not deadline <= wall_deadline:
                    effective_deadline = wall_deadline
                    limit_expired = "invocation_wall_budget"
            recovery = RetryRecoveryBudget(resolved_allowance)
            # Charge dispatched work separately from recovery allowance; never add recovery charges to
            # backend_s/backoff_s.
            telemetry = _RetryTelemetry()
            cleanup_elapsed_s = 0.0
            # In-flight deadline stops keep partials; pre-dispatch/backoff stops discard failed partials.
            partials_discarded_by_deadline = False
            # Set when the invocation already emitted its one stop record, so the
            # post-loop deadline emitter never writes a second, contradicting one.
            stop_recorded = False

            def _emit_budget_stop(stop_reason: str | None = None, limit_expired: str = "retry_ladder") -> None:
                """Record either retry overhead or a deadline stop without affecting execution."""
                nonlocal stop_recorded
                stop_recorded = True
                if recorder is None:
                    return
                now = clock.monotonic()
                retry_stop = stop_reason is not None
                pending = telemetry.pending_retry_s(now) if retry_stop else 0.0
                try:
                    recorder.emit_agent_budget_stop(
                        phase,
                        limit_expired=limit_expired,
                        elapsed_s=now - invocation_start - (0.0 if retry_stop else cleanup_elapsed_s),
                        backend_s=telemetry.retry_backend_s + pending if retry_stop else telemetry.backend_s,
                        backoff_s=telemetry.backoff_s,
                        attempts=telemetry.retry_attempts if retry_stop else telemetry.attempts_dispatched,
                        cleanup_elapsed_s=None if retry_stop else cleanup_elapsed_s,
                        retry_stop_reason=stop_reason,
                        circuit_state=run_context.outage_circuit.state(),
                        retry_recovery_spent_s=recovery.spent_s + pending,
                        partial_edit_handling=("discarded" if retry_stop or partials_discarded_by_deadline else "kept"),
                    )
                except Exception:  # noqa: BLE001 - telemetry must never break the run
                    _logger.exception("failed to record agent budget stop")

            _logger.debug(
                "invocation deadline: effective=%s limit=%s caller=%s wall_budget_s=%s",
                effective_deadline,
                limit_expired,
                deadline,
                wall_budget_s,
            )

            for attempt in range(max_attempts + 1):
                # Reset failed partials before checking deadlines so pre-dispatch stops cannot return earlier attempt
                # output.
                output_parts = []
                structured_result = None
                result_continuation = None
                evidence_incomplete = False
                tool_calls = 0
                if review_evidence is not None:
                    review_evidence.reset()
                budget_reason: str | None = None
                # Never dispatch an attempt once the deadline is spent: the
                # ladder stops here with the reset state.
                if effective_deadline is not None and clock.monotonic() >= effective_deadline:
                    aborted_reason = "wall_budget_exceeded"
                    partials_discarded_by_deadline = True
                    # When retries or backoff spent time, preserve that overhead in a ladder-stop record.
                    if telemetry.spent_retry_overhead:
                        _emit_budget_stop(
                            "retry_deadline_exhausted",
                            limit_expired=limit_expired or "invocation_wall_budget",
                        )
                    break
                # The opening attempt is useful work; later attempts charge recovery after its first activation.
                telemetry.start_attempt(clock.monotonic(), retry=recovery.active)
                display = AgentDisplay(
                    console,
                    run_context.policy,
                    progress_callback,
                    structured=output_schema is not None,
                )

                try:
                    if artifact_session_active():
                        assert_model_cwd_clean(cwd)
                    if sanctioned_inputs is not None:
                        sanctioned_inputs.revalidate(backend, cwd, read_only)
                    execute_kwargs: dict[str, Any] = {
                        "agents": agents,
                        "max_turns": max_turns,
                        "read_only": read_only,
                    }
                    if finalization and getattr(backend, "supports_finalization", False):
                        execute_kwargs["finalization"] = True
                    if tools_disabled and getattr(backend, "supports_tools_disabled", False):
                        execute_kwargs["tools_disabled"] = True
                    if review_instructions and getattr(backend, "supports_review_instructions", False):
                        execute_kwargs["review_instructions"] = review_instructions
                    if not persist_session:
                        execute_kwargs["persist_session"] = False
                    event_iter = backend.execute(
                        cwd,
                        prompt,
                        output_schema,
                        continuation,
                        **execute_kwargs,
                    )
                    invocation_cm: Any = recorder.invocation(phase=phase) if recorder is not None else nullcontext(None)
                    event_stream_scope = _EventStreamScope(event_iter)

                    async with (
                        attempt_scope(attempt + 1) as observed,
                        invocation_cm as inv,
                        event_stream_scope,
                    ):
                        observed.invocation = inv
                        if inv is not None:
                            inv.observe_user_step(prompt=prompt)

                        # Enforce tool/veto limits in-stream and deadlines through both the injected clock and real-time
                        # cancellation.
                        remaining_s = (
                            max(effective_deadline - clock.monotonic(), 0.0) if effective_deadline is not None else None
                        )
                        wall_scope: Any = anyio.move_on_after(remaining_s) if remaining_s is not None else nullcontext()

                        with wall_scope:
                            async for event in event_iter:
                                # Check the injected clock per event; move_on_after only measures real elapsed time.
                                if effective_deadline is not None and clock.monotonic() >= effective_deadline:
                                    budget_reason = "wall_budget_exceeded"
                                    break
                                # The sole telemetry observer runs before UI callbacks,
                                # supervision and budgets can interrupt event handling.
                                observed.observe(event)
                                if review_evidence is not None:
                                    review_evidence.observe(event)
                                # Forward parser evidence and the invocation ledger before any branch can interrupt
                                # handling.
                                if inv is not None:
                                    inv.observe(event)
                                if (
                                    isinstance(event, DiagnosticEvent)
                                    and event.code == "codex_transport_coverage"
                                    and event.metadata.get("coverage") == "incomplete"
                                ):
                                    evidence_incomplete = True
                                if isinstance(event, TextEvent):
                                    output_parts.append(event.text)
                                elif isinstance(event, ResultEvent):
                                    structured_result = event.structured_output
                                    result_continuation = event.continuation
                                if not (
                                    require_full_schema
                                    and output_schema is not None
                                    and isinstance(event, ResultEvent)
                                    and not _validates_schema(event.structured_output, output_schema)
                                ):
                                    await display.observe(event)
                                if isinstance(event, ToolStartEvent):
                                    if tool_supervisor is not None:
                                        try:
                                            # Use the strip-only entry point for start-anchored deny patterns.
                                            supervisor_input = event.input
                                            if event.name == "shell" and isinstance(event.input, dict):
                                                command = event.input.get("command")
                                                if isinstance(command, str):
                                                    supervisor_input = dict(event.input)
                                                    supervisor_input["command"] = supervisor_shell_command(command)
                                            decision = tool_supervisor(event.name, supervisor_input, phase=phase)
                                        except Exception as exc:  # noqa: BLE001 - policy failures must propagate
                                            raise _ToolSupervisorFailure(exc) from exc
                                        if decision.veto:
                                            if recorder is not None:
                                                recorder.emit_tool_veto(event.name, decision.reason, phase=phase)
                                            budget_reason = f"tool_vetoed:{event.name}"
                                            break

                                    tool_calls += 1
                                    if tool_call_budget is not None and tool_calls > tool_call_budget:
                                        budget_reason = "tool_call_budget_exceeded"
                                        break

                            await display.flush()

                        # Mark the turn aborted and close its invocation-owned stream before returning partial output.
                        wall_cancelled = bool(getattr(wall_scope, "cancelled_caught", False))
                        if budget_reason is None and wall_cancelled:
                            budget_reason = "wall_budget_exceeded"
                        aborted_reason = budget_reason
                        if budget_reason is not None:
                            observed.abort(budget_reason)
                            # Bound and shield cleanup so hung subprocesses cannot extend or cancel the captured partial
                            # result.
                            # Measure its grace separately from invocation elapsed time.
                            cleanup_started_at = clock.monotonic()
                            with anyio.move_on_after(BUDGET_CLEANUP_GRACE_S, shield=True) as cleanup_scope:
                                # Record the abort before aclose, which may hang until the cleanup grace expires.
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
                    # Classify before applying retry caps; permanent conditions override transient tokens and larger
                    # advertised caps.
                    classification = classify_failure(exc)
                    exception_max_retries = min(max_attempts, getattr(exc, "max_retries", max_attempts))
                    if attempt < exception_max_retries and classification.retries_allowed:
                        # Activate recovery once, clamped to the remaining deadline; later failures never reset the
                        # allowance.
                        recovery.activate(clock.monotonic(), effective_deadline)
                        # A spent deadline permits no further sleep or dispatch. Discard failed partials
                        # and record any retry overhead already spent.
                        if effective_deadline is not None and clock.monotonic() >= effective_deadline:
                            output_parts = []
                            structured_result = None
                            result_continuation = None
                            aborted_reason = "wall_budget_exceeded"
                            partials_discarded_by_deadline = True
                            if telemetry.spent_retry_overhead:
                                _emit_budget_stop(
                                    "retry_deadline_exhausted",
                                    limit_expired=limit_expired or "invocation_wall_budget",
                                )
                            break
                        # An exhausted allowance permits no sleep/dispatch and preserves the current failure.
                        if recovery.remaining() <= 0.0:
                            _emit_budget_stop("retry_recovery_allowance_exhausted")
                            raise
                        # The circuit gates retries only; stale open state cannot block a first attempt.
                        # Denial propagates the current failure with its retryable attribute intact.
                        circuit_now = clock.monotonic()
                        admission = run_context.outage_circuit.admit_retry(circuit_now)
                        opened_here = (
                            False
                            if admission.allowed and admission.state == CIRCUIT_HALF_OPEN
                            else run_context.outage_circuit.record_failure(circuit_now)
                        )
                        if not admission.allowed:
                            _emit_budget_stop("circuit_open")
                            raise
                        # Honor server waits within recovery/deadline bounds; oversized hints stop recovery.
                        hint = _retry_hint(exc)
                        delay, delay_stop = _plan_retry_delay(
                            attempt=attempt,
                            base_delay_s=base_delay,
                            max_delay_s=max_delay,
                            allowance_remaining_s=(recovery.remaining() if recovery.active else None),
                            deadline_remaining_s=(
                                None if effective_deadline is None else max(effective_deadline - clock.monotonic(), 0.0)
                            ),
                            hint=hint,
                        )
                        if delay_stop is not None:
                            _emit_budget_stop(delay_stop)
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
                        # Close attempt timing before backoff so sleep is never counted twice in backend_s/backoff_s.
                        charged = telemetry.charge_attempt(clock.monotonic())
                        if telemetry.attempt_is_retry:
                            recovery.charge(charged)
                        # Charge backoff before sleeping so later failures see the unre-based recovery remainder.
                        recovery.charge(delay)
                        await anyio.sleep(delay)
                        telemetry.backoff_s += delay
                        # Recheck after backoff because a sibling may have opened the circuit.
                        # Do not consume a granted half-open probe twice; the tripper retains its retry.
                        if (
                            not opened_here
                            and admission.state == CIRCUIT_CLOSED
                            and not run_context.outage_circuit.admit_retry(clock.monotonic()).allowed
                        ):
                            _emit_budget_stop("circuit_open")
                            raise
                        continue
                    if classification.retries_allowed and exception_max_retries > 0:
                        _emit_budget_stop("retry_attempts_exhausted")
                    raise
                finally:
                    if telemetry.attempt_started_at is not None:
                        # Exclude bounded post-stop cleanup from attempt timing and elapsed invocation time.
                        charged = telemetry.charge_attempt(clock.monotonic(), cleanup_elapsed_s=cleanup_elapsed_s)
                        if telemetry.attempt_is_retry:
                            recovery.charge(charged)

            # Emit at most one stop record. In-flight deadline stops retain partials;
            # pre-dispatch stops discard them. Recorder failure cannot change the return value.
            if aborted_reason == "wall_budget_exceeded" and not stop_recorded:
                _emit_budget_stop(limit_expired=limit_expired or "invocation_wall_budget")

        except _ToolSupervisorFailure as exc:
            original = exc.original
            detail = exception_text(original) or ""
            diagnostic = f"{type(original).__name__}: {detail}" if detail else type(original).__name__
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
            # Redact secrets and neutralize terminal controls before surfacing provider errors.
            print_error(console, "Backend Execution Error", sanitize_verbose_message(diagnostic))
            raise
        except BaseException:
            # SIGINT and task cancellation bypass Exception handlers; reap all tracked invocations before unwinding.
            try:
                await backend.cancel()
            except Exception:  # cancel() must not mask the original signal
                _logger.exception("backend.cancel() failed during shutdown")
            raise

    if evidence_incomplete and aborted_reason is None and require_full_schema:
        aborted_reason = "evidence_incomplete"

    def _usable(value: Any) -> bool:
        """Accept explicit validation opt-out or a downstream-salvageable value."""
        return not validate_structured_output or (
            output_schema is not None
            and (_validates_schema(value, output_schema) if require_full_schema else _salvageable(value, output_schema))
        )

    if output_schema is not None and structured_result is not None and _usable(structured_result):
        return structured_result, result_continuation, aborted_reason
    raw = "".join(output_parts)
    if output_schema is not None and raw.strip():
        parsed = extract_json(raw)
        if parsed is not None and _usable(parsed):
            return parsed, result_continuation, aborted_reason
    if output_schema is not None and require_full_schema:
        reason = "malformed_output" if structured_result is not None or raw.strip() else "missing_output"
        return StructuredOutputFailure(raw, reason), result_continuation, aborted_reason
    return raw, result_continuation, aborted_reason
