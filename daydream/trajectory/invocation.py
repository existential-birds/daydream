"""Turn buffers and typed backend-event handlers for one agent invocation.

Steps are plain transcript dictionaries (the evidence ``daydream.transcript.step``
body). The invocation retains only its own steps; the recorder appends them to the
document's evidence journal once the invocation finishes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

from daydream import timeutil, ui
from daydream.backends import (
    AgentEvent,
    ClaudeRequestConfig,
    CodexRequestConfig,
    CostEvent,
    DiagnosticEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    OspreyRequestConfig,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    request_config_controls,
)
from daydream.backends.lifecycle import (
    BackendLifecycleCollector,
    BackendLifecycleSnapshot,
    current_backend_lifecycle,
)
from daydream.redaction import redact_value as redact_value
from daydream.trajectory.billing import GENERIC_MODEL_LABELS, UsageLedger
from daydream.trajectory.generation import _GenerationLedger
from daydream.trajectory.redactor import redact_result_content, redact_step
from daydream.trajectory.types import DaydreamPhase as DaydreamPhase

if TYPE_CHECKING:
    from daydream.trajectory.recorder import EvidenceRecorder
_console = ui.create_console()

Step = dict[str, Any]


def _reasoning_extra(reasoning_tokens: int | None) -> dict[str, Any] | None:
    """Metrics ``extra`` carrier for reasoning_tokens (#192), or None when absent."""
    return {"reasoning_tokens": reasoning_tokens} if reasoning_tokens is not None else None


def _metrics(*, prompt_tokens: int | None, completion_tokens: int | None, cached_tokens: int | None,
             cost_usd: float | None, extra: dict[str, Any] | None) -> dict[str, Any]:
    """Build a step metrics object; absent dimensions are omitted, never zeroed."""
    return {key: value for key, value in (
        ("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens),
        ("cached_tokens", cached_tokens), ("cost_usd", cost_usd), ("extra", extra)) if value is not None}


def _add(left: Any, right: Any) -> Any:
    """Sum two optional numbers, treating None as absent (not zero)."""
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _merge_metrics(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Sum billed turn metrics and reasoning tokens; carry other extra fields forward."""
    merged_extra: dict[str, Any] | None = None
    if existing.get("extra") is not None or incoming.get("extra") is not None:
        merged_extra = {**(existing.get("extra") or {}), **(incoming.get("extra") or {})}
        reasoning = _add((existing.get("extra") or {}).get("reasoning_tokens"),
                         (incoming.get("extra") or {}).get("reasoning_tokens"))
        if reasoning is not None:
            merged_extra["reasoning_tokens"] = reasoning
    return _metrics(
        prompt_tokens=_add(existing.get("prompt_tokens"), incoming.get("prompt_tokens")),
        completion_tokens=_add(existing.get("completion_tokens"), incoming.get("completion_tokens")),
        cached_tokens=_add(existing.get("cached_tokens"), incoming.get("cached_tokens")),
        cost_usd=_add(existing.get("cost_usd"), incoming.get("cost_usd")),
        extra=merged_extra,
    )


@dataclass(frozen=True)
class _CostDelta:
    """Nonnegative residual max(0, CostEvent total - per-event sum), per dimension.

    Repairs missing usage without double-counting backends that restate totals."""

    prompt: int
    completion: int
    cached: int
    cost: float
    reasoning: int | None

    @property
    def nonzero(self) -> bool:
        """True when any residual dimension exists (delta-0 restatement folds nothing)."""
        return self.prompt != 0 or self.completion != 0 or self.cached != 0 or self.cost != 0.0

    def residual_metrics(self, *, cost_usd: float | None, include_reasoning: bool) -> dict[str, Any]:
        """Project residual usage; reasoning is included only within completion tokens."""
        reasoning = None
        if include_reasoning and self.reasoning is not None and self.reasoning <= self.completion:
            reasoning = _reasoning_extra(self.reasoning)
        return _metrics(prompt_tokens=self.prompt, completion_tokens=self.completion, cached_tokens=self.cached,
                        cost_usd=None if cost_usd is None else self.cost, extra=reasoning)


# Fixed-ASCII synthetic content for the incomplete-call marker emitted by
# ``Invocation.finish()`` for tool calls still in flight. Never formatted with
# tool data, so it is redaction-stable.
INCOMPLETE_CALL_CONTENT = "[interrupted: call did not complete before invocation ended]"


_UNSUPPORTED_DIAGNOSTIC_KEY = "[UNSUPPORTED_DIAGNOSTIC_KEY]"


_UNSUPPORTED_DIAGNOSTIC_VALUE = "[UNSUPPORTED_DIAGNOSTIC_VALUE]"


_DIAGNOSTIC_REDACTION_FAILED = {
    "code": "diagnostic_redaction_failed",
    "message": "[DIAGNOSTIC_REDACTION_FAILED]",
    "metadata": {},
}


def _diagnostic_json_value(value: Any) -> Any:
    """Return a JSON-shaped diagnostic value without stringifying objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNSUPPORTED_DIAGNOSTIC_VALUE
    if isinstance(value, list):
        return [_diagnostic_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = key if isinstance(key, str) else _UNSUPPORTED_DIAGNOSTIC_KEY
            normalized[normalized_key] = _diagnostic_json_value(item)
        return normalized
    return _UNSUPPORTED_DIAGNOSTIC_VALUE


def _normalize_diagnostic_record(event: "DiagnosticEvent") -> dict[str, Any]:
    """Redact JSON-safe step diagnostics; never fall back to raw event data.

    Normalization or redaction failures produce a fixed failure marker."""
    try:
        normalized = _diagnostic_json_value({"code": event.code, "message": event.message, "metadata": event.metadata})
        redacted = redact_value(normalized)
        if not isinstance(redacted, dict):
            raise TypeError("diagnostic redactor returned a non-object")
        json.dumps(redacted, allow_nan=False)
        return redacted
    except Exception:  # noqa: BLE001 - fixed fail-closed boundary, never raw fallback
        return dict(_DIAGNOSTIC_REDACTION_FAILED, metadata={})


def _result_extra(event: ToolResultEvent) -> dict[str, Any]:
    """Project supplied scalar outcome metadata; never copy tool text or arguments."""
    extra: dict[str, Any] = {"is_error": event.is_error}
    if event.exit_code is not None:
        extra["exit_code"] = event.exit_code
    if event.status:
        extra["status"] = event.status
    if event.duration_ms is not None:
        extra["duration_ms"] = event.duration_ms
    if event.cancelled:
        extra["cancelled"] = True
    if event.truncated:
        extra["truncated"] = True
    return extra


def _result(*, source_call_id: str | None, content: Any, extra: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (
        ("source_call_id", source_call_id), ("content", content), ("extra", extra)) if value is not None}


def _with_observation_result(step: Step, result: dict[str, Any]) -> Step:
    """Copy a step with one more observation result."""
    observation = step.get("observation")
    results = [*(observation or {}).get("results", []), result]
    return {**step, "observation": {"results": results}}


@dataclass
class Invocation:
    """Buffer one agent conversation and reconcile its billed usage.

    TurnEnd and Result close the active step; finish also closes partial turns.
    Metrics accumulate across turns. Each tool result remains attached to its
    originating step, even if TurnEnd closed that step before the result arrived.
    The in-flight map points to an open buffer or a closed step index. Recorder
    owns step IDs and parent linkage. Each closed step records the boundary that
    closed it (``turn_end``, ``result`` or ``invocation_end``) as
    ``extra.turn_boundary``."""

    recorder: "EvidenceRecorder"
    phase: DaydreamPhase
    invocation_id: str = ""
    steps: list[Step] = field(default_factory=list)
    # Per-invocation timing boundaries set by the invocation scope;
    # surfaced in the document's subtrajectory summaries.
    started_at: str = ""
    ended_at: str = ""
    _open_step_dict: dict[str, Any] | None = None
    _in_flight_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    _stop_reason: str | None = None
    _error_subtype: str | None = None
    # Per-invocation sum of the MetricsEvent usage observed so far (issue #747).
    # CostEvent totals reconcile against it (per-dimension take-max delta) instead
    # of trusting collapsed per-message digits or re-accumulating restated totals.
    _metered: UsageLedger = field(default_factory=UsageLedger)
    # Drafts open on GenerationStartEvent, seal at GenerationEndEvent and stay
    # UNENDED until finish() resolves the billing owner; terminal/cancel/error
    # paths drain every draft exactly once at its sealed historical end.
    _generation_ledger: _GenerationLedger = field(default_factory=_GenerationLedger)
    _backend_lifecycle_snapshot: BackendLifecycleSnapshot | None = None
    _backend_lifecycle_collector: BackendLifecycleCollector | None = field(default=None, repr=False)
    _host_dispatch: dict[str, int | str] | None = None
    # The recorded FailureClass of the failed attempt this attempt retries.
    _retry_failure_class: str | None = None
    _requests: list[dict[str, Any]] = field(default_factory=list)
    _metered_message_ids: set[str] = field(default_factory=set)

    def summary(self, *, partial: bool = False) -> dict[str, Any]:
        """Project invocation identity/timing; generation billing appears only after completion."""
        summary: dict[str, Any] = {
            "trajectory_id": self.recorder.trajectory_id,
            "invocation_id": self.invocation_id,
            "phase": self.phase.value,
            "started_at": self.started_at,
            "ended_at": None if partial else self.ended_at,
            "step_ids": [step["step_id"] for step in self.steps],
        }
        if not partial and self._generation_ledger.has_drafts():
            summary["generation_lifecycle"] = self._generation_ledger.to_dict()
        lifecycle = (
            self._backend_lifecycle_collector.snapshot()
            if partial and self._backend_lifecycle_collector is not None
            else self._backend_lifecycle_snapshot
        )
        if lifecycle is not None:
            summary["backend_lifecycle"] = lifecycle.to_dict()
        if self._host_dispatch is not None:
            summary["host_dispatch"] = dict(self._host_dispatch)
        if self._retry_failure_class is not None:
            summary["retry_failure_class"] = self._retry_failure_class
        if self._requests:
            summary["requests"] = redact_value(self._requests)
        return summary

    def record_host_dispatch(self, dispatch: dict[str, int | str]) -> None:
        """Retain the host's dispatch measurement for this attempt."""
        self._host_dispatch = dispatch

    def record_retry(self, failure_class: str) -> None:
        """Retain the bounded failure class that admitted this retried attempt."""
        self._retry_failure_class = failure_class

    def observe_backend_lifecycle(self, collector: BackendLifecycleCollector) -> None:
        """Bind only this host-owned attempt for immutable in-flight lifecycle observations."""
        self._backend_lifecycle_collector = collector

    def record_backend_lifecycle(self, snapshot: BackendLifecycleSnapshot) -> None:
        """Retain finalized attempt observations without completing or billing generations."""
        self._backend_lifecycle_snapshot = snapshot
        self._backend_lifecycle_collector = None

    def observe_user_step(self, prompt: str) -> None:
        """Append an initial user step without agent-only fields."""
        try:
            self._close_open_step("invocation_end")
            user_step: Step = {
                "step_id": self.recorder._next_step_id(),
                "timestamp": timeutil.now_iso(),
                "source": "user",
                "message": prompt,
                "extra": {
                    "daydream_phase": self.phase.value,
                    "daydream_run_flow": self.recorder.run_flow.value,
                },
            }
            self.steps.append(redact_step(user_step))
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            ui.print_warning(_console, f"Evidence recording: {type(exc).__name__}: {exc}")

    def mark_aborted(self, reason: str) -> None:
        """Stamp stop_reason on the closing step and mark the document ancestry partial.

        Open a step even before the first event so finish() can persist the reason."""
        self._stop_reason = reason
        self._ensure_open_step()
        recorder: EvidenceRecorder | None = self.recorder
        while recorder is not None:
            recorder._aborted = True
            recorder = recorder.parent

    def mark_errored(self, subtype: str) -> None:
        """Stamp error/error_subtype in the closing step's extra fields.

        Open a step even before the first event so finish() can persist the failure."""
        self._error_subtype = subtype
        self._ensure_open_step()

    def observe(self, event: "AgentEvent") -> None:
        """Record one backend event; recording failures must not interrupt agent work.

        The local catch does not suppress failures in the caller's event loop."""
        try:
            self._dispatch(event)
            lifecycle = current_backend_lifecycle()
            if lifecycle is not None:
                lifecycle.recorder_event()
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            ui.print_warning(_console, f"Evidence recording: {type(exc).__name__}: {exc}")

    def _reconcile_cost_delta(self, event: CostEvent) -> _CostDelta:
        """Compute the per-dimension residual against the invocation's summed metrics."""
        metered = self._metered
        return _CostDelta(
            prompt=max(0, (event.input_tokens or 0) - (metered.prompt or 0)),
            completion=max(0, (event.output_tokens or 0) - (metered.completion or 0)),
            cached=max(0, (event.cached_tokens or 0) - (metered.cached or 0)),
            cost=max(0.0, (event.cost_usd or 0.0) - (metered.cost or 0.0)),
            reasoning=event.reasoning_tokens,
        )

    def _fold_cost_event(self, event: CostEvent) -> None:
        """Add only positive CostEvent residuals to step metrics and the document ledger."""
        delta = self._reconcile_cost_delta(event)
        existing = self._open_step_dict["_metrics"] if self._open_step_dict is not None else None
        if existing is not None:
            # A MetricsEvent already populated this step: fold the residual onto
            # it, then backfill cost_usd / reasoning the per-message path did not
            # surface, so summed step usage equals the final total (#747).
            if delta.nonzero:
                existing = _merge_metrics(existing, delta.residual_metrics(cost_usd=event.cost_usd,
                                                                           include_reasoning=False))
            updates: dict[str, Any] = {}
            if existing.get("cost_usd") is None and event.cost_usd is not None:
                updates["cost_usd"] = event.cost_usd
            # #192: backfill reasoning_tokens when the MetricsEvent path did not carry it.
            if event.reasoning_tokens is not None and "reasoning_tokens" not in (existing.get("extra") or {}):
                updates["extra"] = {**(existing.get("extra") or {}), "reasoning_tokens": event.reasoning_tokens}
            existing = {**existing, **updates}
            assert self._open_step_dict is not None
            self._open_step_dict["_metrics"] = existing
        elif delta.nonzero:
            # No metrics-bearing step is open: mint a residual step holding exactly
            # the delta this CostEvent adds beyond the per-message sum (#747).
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_metrics"] = delta.residual_metrics(cost_usd=event.cost_usd, include_reasoning=True)
        # A delta-0 restatement with no metrics-bearing step mints no phantom step.
        if event.model_name:
            if self._open_step_dict is not None:
                self._open_step_dict["_model_name"] = event.model_name
            self.recorder._upgrade_model_name(event.model_name)
        # Per-dimension take-max within the invocation, summed across invocations.
        self.recorder.usage.charge(
            phase=self.phase.value,
            prompt_tokens=None if event.input_tokens is None else delta.prompt,
            completion_tokens=None if event.output_tokens is None else delta.completion,
            cached_tokens=None if event.cached_tokens is None else delta.cached,
            cost_usd=None if event.cost_usd is None else delta.cost,
        )

    @singledispatchmethod
    def _dispatch(self, event: object) -> None:
        """Ignore metadata events without a recording action."""

    @_dispatch.register
    def _observe_diagnostic(self, event: DiagnosticEvent) -> None:
        self._ensure_open_step()["_backend_diagnostics"].append(_normalize_diagnostic_record(event))

    @_dispatch.register
    def _observe_text(self, event: TextEvent) -> None:
        self._ensure_open_step()["_text_chunks"].append(event.text)

    @_dispatch.register
    def _observe_thinking(self, event: ThinkingEvent) -> None:
        self._ensure_open_step()["_thinking_chunks"].append(event.text)

    @_dispatch.register
    def _observe_tool_start(self, event: ToolStartEvent) -> None:
        step = self._ensure_open_step()
        step["_tool_calls"].append({"tool_call_id": event.id, "function_name": event.name,
                                    "arguments": event.input or {}})
        # Results stay attached to their originating step, even after it closes.
        self._in_flight_tools[event.id] = {"open_dict": step, "closed_index": None}

    @_dispatch.register
    def _observe_tool_result(self, event: ToolResultEvent) -> None:
        host = self._in_flight_tools.pop(event.id, None)
        if host is None:
            # Preserve unmatched evidence without emitting an invalid source_call_id.
            self._ensure_open_step()["_unmatched_tool_results"].append(event.id)
            return
        result = _result(source_call_id=event.id, content=event.output, extra=_result_extra(event))
        if host["open_dict"] is not None:
            host["open_dict"]["_observation_results"].append(result)
        else:
            index = host["closed_index"]
            self.steps[index] = _with_observation_result(self.steps[index], redact_result_content(result))

    @_dispatch.register
    def _observe_request(self, event: RequestEvent) -> None:
        # No prompt, environment, credential or native settings file contents.
        backend = {PiRequestConfig: 'pi', CodexRequestConfig: 'codex', ClaudeRequestConfig: 'claude',
                   OspreyRequestConfig: 'osprey'}.get(type(event.config))
        self._requests.append({"backend": backend,
            "model": event.model_name, "provider": event.provider_name, "reasoning_effort": event.reasoning_effort,
            "controls": request_config_controls(event.config)})

    @_dispatch.register
    def _observe_metrics(self, event: MetricsEvent) -> None:
        # Claude repeats one provider message's usage on each of its content-block messages.
        if event.message_id:
            if event.message_id in self._metered_message_ids:
                if event.model_name and self._open_step_dict is not None:
                    self._open_step_dict["_model_name"] = event.model_name
                return
            self._metered_message_ids.add(event.message_id)
        target = self._open_step_dict
        if target is None and not self.steps:
            target = self._ensure_open_step()
        # Cache and reasoning tokens are subsets of input/output, never additive.
        incoming = _metrics(prompt_tokens=event.prompt_tokens, completion_tokens=event.completion_tokens,
                            cached_tokens=event.cached_tokens, cost_usd=event.cost_usd,
                            extra=_reasoning_extra(event.reasoning_tokens))
        if target is not None:
            prior = target["_metrics"]
            target["_metrics"] = incoming if prior is None else _merge_metrics(prior, incoming)
            if event.model_name:
                target["_model_name"] = event.model_name
        else:
            # Codex usage may follow TurnEnd: amend that step, without inventing one.
            self._fold_metrics_into_closed_last_step(event, incoming)
        if event.model_name:
            self.recorder._upgrade_model_name(event.model_name)
        # Invocation totals reconcile terminal CostEvents; the document ledger bills every turn.
        for ledger in (self._metered, self.recorder.usage):
            ledger.charge(phase=self.phase.value, prompt_tokens=event.prompt_tokens,
                          completion_tokens=event.completion_tokens, cached_tokens=event.cached_tokens,
                          cost_usd=event.cost_usd)
        if event.generation_id:
            self._generation_ledger.record_usage(event)

    @_dispatch.register
    def _observe_cost(self, event: CostEvent) -> None:
        self._fold_cost_event(event)
        # Only terminal/session aggregates close the generation bill.
        if event.measurement_source in ("terminal", "session"):
            self._generation_ledger.record_authoritative_total(event)

    @_dispatch.register
    def _observe_generation_start(self, event: GenerationStartEvent) -> None:
        self._generation_ledger.open(event)

    @_dispatch.register
    def _observe_generation_end(self, event: GenerationEndEvent) -> None:
        self._generation_ledger.seal(event)

    @_dispatch.register
    def _observe_result(self, event: ResultEvent) -> None:
        if event.model_name:
            if self._open_step_dict is not None:
                self._open_step_dict["_model_name"] = event.model_name
            else:
                for index, step in enumerate(self.steps):
                    if step["source"] == "agent" and (step.get("model_name") or "") in GENERIC_MODEL_LABELS:
                        self.steps[index] = {**step, "model_name": event.model_name}
            self.recorder._upgrade_model_name(event.model_name)
        self._close_open_step("result")

    @_dispatch.register
    def _observe_turn_end(self, event: TurnEndEvent) -> None:
        self._close_open_step("turn_end")

    def _ensure_open_step(self) -> dict[str, Any]:
        """Return the active turn buffer, opening it on the first event."""
        if self._open_step_dict is None:
            self._open_step_dict = {
                "_text_chunks": [],
                "_thinking_chunks": [],
                "_tool_calls": [],
                "_observation_results": [],
                "_metrics": None,
                "_model_name": self.recorder.agent_model_name,
                "_unmatched_tool_results": [],
                "_backend_diagnostics": [],
            }
        return self._open_step_dict

    def _materialize_agent_step(self, d: dict[str, Any], *, step_id: int, extra_overrides: dict[str, Any]) -> Step:
        """Materialize the open-step dict *d* into a redacted agent step."""
        extra: dict[str, Any] = {
            "daydream_phase": self.phase.value,
            "daydream_run_flow": self.recorder.run_flow.value,
            **extra_overrides,
        }
        if d["_backend_diagnostics"]:
            extra["backend_diagnostics"] = list(d["_backend_diagnostics"])
        step: Step = {
            "step_id": step_id,
            "timestamp": timeutil.now_iso(),
            "source": "agent",
            "model_name": d["_model_name"],
            "message": "".join(d["_text_chunks"]),
            "reasoning_content": "\n".join(d["_thinking_chunks"]) if d["_thinking_chunks"] else None,
            "tool_calls": [dict(call) for call in d["_tool_calls"]] or None,
            "observation": {"results": list(d["_observation_results"])} if d["_observation_results"] else None,
            "metrics": d["_metrics"],
            "llm_call_count": 1,
            "extra": extra,
        }
        return redact_step({key: value for key, value in step.items() if value is not None})

    def _close_open_step(self, boundary: str) -> None:
        """Redact and append the open step; repeated flushes are harmless.

        Retarget in-flight tools to the closed step index so late results keep their
        original turn. Called at TurnEnd, Result, and final invocation flush."""
        if self._open_step_dict is None:
            return
        d = self._open_step_dict
        self._open_step_dict = None

        extra_overrides: dict[str, Any] = {"turn_boundary": boundary}
        if d["_unmatched_tool_results"]:
            extra_overrides["unmatched_tool_results"] = list(d["_unmatched_tool_results"])
        if self._stop_reason is not None:
            extra_overrides["stop_reason"] = self._stop_reason
        if self._error_subtype is not None:
            extra_overrides["error"] = True
            extra_overrides["error_subtype"] = self._error_subtype

        self.steps.append(
            self._materialize_agent_step(d, step_id=self.recorder._next_step_id(), extra_overrides=extra_overrides)
        )
        closed_index = len(self.steps) - 1
        # A delayed ToolResultEvent finds its closed host through closed_index.
        for entry in self._in_flight_tools.values():
            if entry["open_dict"] is d:
                entry["open_dict"] = None
                entry["closed_index"] = closed_index

    def _fold_metrics_into_closed_last_step(self, event: Any, incoming: dict[str, Any]) -> None:
        """Add late terminal usage to the most recent closed agent step.

        Codex usage arrives after TurnEnd. Folding preserves backend step-shape parity
        and the equality between summed step usage and final usage."""
        for idx in range(len(self.steps) - 1, -1, -1):
            step = self.steps[idx]
            if step["source"] != "agent":
                continue
            metrics = incoming if step.get("metrics") is None else _merge_metrics(step["metrics"], incoming)
            updated = {**step, "metrics": metrics}
            if event.model_name:
                updated["model_name"] = event.model_name
            self.steps[idx] = updated
            return
        # Every recorded step is non-agent: mint a metrics-bearing agent step so summed
        # step usage still equals the final total instead of dropping it (#747).
        self._ensure_open_step()
        assert self._open_step_dict is not None
        self._open_step_dict["_metrics"] = incoming
        if event.model_name:
            self._open_step_dict["_model_name"] = event.model_name

    def snapshot_steps(self, *, snapshot_step_id: int | None = None) -> list[Step]:
        """Materialize flushed and open steps without changing either buffer.

        The caller supplies a unique snapshot_step_id. In-flight calls get the same
        interrupted markers as finish(), while remaining available for later events."""
        in_flight = list(self._in_flight_tools.values())
        if self._open_step_dict is None:
            steps = list(self.steps)
        else:
            d = self._open_step_dict
            extra_overrides: dict[str, Any] = {"partial_step": True}
            if d["_unmatched_tool_results"]:
                extra_overrides["unmatched_tool_results"] = list(d["_unmatched_tool_results"])
            step_id = snapshot_step_id if snapshot_step_id is not None else self.recorder._step_id_counter + 1
            steps = [*self.steps, self._materialize_agent_step(d, step_id=step_id, extra_overrides=extra_overrides)]
        # Markers land per host step in the same LIFO order finish()'s popitem loop uses.
        for host in reversed(in_flight):
            if host["closed_index"] is not None:
                steps[host["closed_index"]] = _with_observation_result(
                    steps[host["closed_index"]], self._interrupted_marker()
                )
            elif self._open_step_dict is not None and host["open_dict"] is self._open_step_dict:
                steps[-1] = _with_observation_result(steps[-1], self._interrupted_marker())
        return steps

    @staticmethod
    def _interrupted_marker() -> dict[str, Any]:
        """Fixed-ASCII interruption marker; a null source_call_id is not a completed result."""
        return _result(source_call_id=None, content=INCOMPLETE_CALL_CONTENT,
                       extra={"is_error": True, "status": "interrupted"})

    def _emit_incomplete_call_markers(self) -> None:
        """Mark remaining calls interrupted on their closed host steps (idempotent)."""
        while self._in_flight_tools:
            _, host = self._in_flight_tools.popitem()
            index = host["closed_index"]
            self.steps[index] = _with_observation_result(self.steps[index], self._interrupted_marker())

    def finish(self) -> None:
        """Close partial steps, mark unfinished calls interrupted, and flush.

        On every terminal path, drain generation drafts exactly once at their sealed
        historical ends and close billing ownership."""
        self._close_open_step("invocation_end")
        self._emit_incomplete_call_markers()
        self._generation_ledger.finalize()
        self.recorder._extend_steps(self.steps)
