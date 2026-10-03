"""Turn buffers and typed backend-event handlers for one agent invocation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, TypedDict

from daydream import timeutil, ui
from daydream.atif import Metrics, Observation, ObservationResult, Step, ToolCall
from daydream.backends import (
    AgentEvent,
    CostEvent,
    DiagnosticEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.redaction import redact_value as redact_value
from daydream.trajectory.generation import _GenerationLedger
from daydream.trajectory.types import DaydreamPhase as DaydreamPhase

if TYPE_CHECKING:
    from daydream.trajectory.recorder import TrajectoryRecorder
_console = ui.create_console()


# Upgrade generic initial backend labels as soon as native model metadata arrives.
_GENERIC_MODEL_LABELS: frozenset[str] = frozenset({"claude", "codex", "osprey", "unknown", ""})


def _reasoning_extra(reasoning_tokens: int | None) -> dict[str, Any] | None:
    """Metrics ``extra`` carrier for reasoning_tokens (#192), or None when absent."""
    return {"reasoning_tokens": reasoning_tokens} if reasoning_tokens is not None else None


def _add(left: Any, right: Any) -> Any:
    """Sum two optional numbers, treating None as absent (not zero)."""
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _merge_metrics(existing: "Metrics", incoming: "Metrics") -> "Metrics":
    """Sum billed turn metrics and reasoning tokens; carry other extra fields forward."""
    merged_extra: dict[str, Any] | None = None
    if existing.extra is not None or incoming.extra is not None:
        merged_extra = {**(existing.extra or {}), **(incoming.extra or {})}
        reasoning = _add(
            (existing.extra or {}).get("reasoning_tokens"),
            (incoming.extra or {}).get("reasoning_tokens"),
        )
        if reasoning is not None:
            merged_extra["reasoning_tokens"] = reasoning
    return existing.model_copy(
        update={
            "prompt_tokens": _add(existing.prompt_tokens, incoming.prompt_tokens),
            "completion_tokens": _add(existing.completion_tokens, incoming.completion_tokens),
            "cached_tokens": _add(existing.cached_tokens, incoming.cached_tokens),
            "cost_usd": _add(existing.cost_usd, incoming.cost_usd),
            "extra": merged_extra,
        }
    )


class _InvMetricsSum(TypedDict):
    """Invocation totals retain integer token counts and fractional cost."""

    prompt: int
    completion: int
    cached: int
    cost: float


# Fixed ASCII interruption text never includes tool data, keeping it redaction-stable.
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
    """Redact JSON-safe Step.extra diagnostics; never fall back to raw event data.

    Normalization or redaction failures produce a fixed failure marker."""
    try:
        normalized = _diagnostic_json_value({"code": event.code, "message": event.message, "metadata": event.metadata})
        redacted = redact_value(normalized)
        if not isinstance(redacted, dict):
            raise TypeError("diagnostic redactor returned a non-object")
        json.dumps(redacted, allow_nan=False)
        return redacted
    except Exception:  # noqa: BLE001 - fixed fail-closed boundary, never raw fallback
        return {
            "code": _DIAGNOSTIC_REDACTION_FAILED["code"],
            "message": _DIAGNOSTIC_REDACTION_FAILED["message"],
            "metadata": {},
        }


def _result_extra(event: ToolResultEvent) -> dict[str, Any]:
    """Project supplied scalar outcome metadata; never copy tool text or arguments."""
    extra: dict[str, Any] = {"is_error": event.is_error}
    for name in ("exit_code", "status", "duration_ms", "cancelled", "truncated"):
        value = getattr(event, name)
        if value is not None and (value or name in ("exit_code", "duration_ms")):
            extra[name] = True if name in ("cancelled", "truncated") else value
    return extra


@dataclass
class Invocation:
    """Buffer one conversation and billed usage. TurnEnd/Result close steps; finish closes partial turns.

    Late tool results retain their originating step through append-only host indices.
    Recorder owns step IDs and ancestry.
    """

    recorder: "TrajectoryRecorder"
    phase: DaydreamPhase
    invocation_id: str = ""
    steps: list[Step] = field(default_factory=list)
    # Per-invocation timing boundaries set by the invocation scope;
    # surfaced via Trajectory.extra["subtrajectories"].
    started_at: str = ""
    ended_at: str = ""
    _open_step_dict: dict[str, Any] | None = None
    # A pending turn occupies the next append-only step index; closing it never retargets tool ownership.
    _in_flight_tools: dict[str, int] = field(default_factory=dict)
    _stop_reason: str | None = None
    _error_subtype: str | None = None
    # Per-dimension invocation sums reconcile restated CostEvent totals without double counting.
    _inv_metrics_sum: _InvMetricsSum = field(
        default_factory=lambda: _InvMetricsSum(prompt=0, completion=0, cached=0, cost=0.0)
    )
    # Generation drafts stay unended until finish resolves billing; terminal paths drain once at sealed ends.
    _generation_ledger: _GenerationLedger = field(default_factory=_GenerationLedger)

    def summary(self, *, partial: bool = False) -> dict[str, Any]:
        """Project invocation identity/timing; generation billing appears only after completion."""
        summary: dict[str, Any] = {
            "trajectory_id": self.recorder.trajectory_id,
            "invocation_id": self.invocation_id,
            "phase": self.phase.value,
            "started_at": self.started_at,
            "ended_at": None if partial else self.ended_at,
            "step_ids": [step.step_id for step in self.steps],
        }
        if not partial and self._generation_ledger.has_drafts():
            summary["generation_lifecycle"] = self._generation_ledger.to_dict()
        return summary

    def observe_user_step(self, prompt: str) -> None:
        """Append an initial user Step without ATIF agent-only fields."""
        try:
            self._close_open_step()
            user_step = Step(
                step_id=self.recorder._next_step_id(),
                timestamp=timeutil.now_iso(),
                source="user",
                message=prompt,
                extra={
                    "daydream_phase": self.phase.value,
                    "daydream_run_flow": self.recorder.run_flow.value,
                },
            )
            self.steps.append(self.recorder.redactor.redact_step(user_step))
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            ui.print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")

    def mark_aborted(self, reason: str) -> None:
        """Stamp stop_reason on the closing step and mark the trajectory ancestry partial.

        Open a step even before the first event so finish() can persist the reason."""
        self._stop_reason = reason
        self._ensure_open_step()
        recorder: TrajectoryRecorder | None = self.recorder
        while recorder is not None:
            recorder._aborted = True
            recorder = recorder.parent

    def mark_errored(self, subtype: str) -> None:
        """Stamp error/error_subtype in the closing step’s ATIF extra fields.

        Open a step even before the first event so finish() can persist the failure."""
        self._error_subtype = subtype
        self._ensure_open_step()

    def observe(self, event: "AgentEvent") -> None:
        """Record one backend event; recording failures must not interrupt agent work.

        The local catch does not suppress failures in the caller’s event loop."""
        try:
            self._dispatch(event)
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            ui.print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")

    def _reconcile_cost_delta(self, event: CostEvent) -> Metrics:
        """Repair only positive residuals; cache/reasoning remain usage subsets."""
        completion = max(0, (event.output_tokens or 0) - self._inv_metrics_sum["completion"])
        return Metrics(
            prompt_tokens=max(0, (event.input_tokens or 0) - self._inv_metrics_sum["prompt"]),
            completion_tokens=completion,
            cached_tokens=max(0, (event.cached_tokens or 0) - self._inv_metrics_sum["cached"]),
            cost_usd=None if event.cost_usd is None else max(0.0, event.cost_usd - self._inv_metrics_sum["cost"]),
            extra=(
                _reasoning_extra(event.reasoning_tokens)
                if event.reasoning_tokens is not None and event.reasoning_tokens <= completion else None
            ),
        )

    def _fold_cost_event(self, event: CostEvent) -> None:
        """Add only positive CostEvent residuals to step metrics and aggregate totals."""
        delta = self._reconcile_cost_delta(event)
        nonzero = any((delta.prompt_tokens, delta.completion_tokens, delta.cached_tokens, delta.cost_usd))
        existing = self._open_step_dict["_metrics"] if self._open_step_dict is not None else None
        if existing is not None:
            # Fold residual usage into this metrics-bearing step so step sums match final totals;
            # backfill missing cost/reasoning.
            if nonzero:
                existing = _merge_metrics(
                    existing,
                    delta.model_copy(update={"extra": None}),
                )
            updates: dict[str, Any] = {}
            if existing.cost_usd is None and event.cost_usd is not None:
                updates["cost_usd"] = event.cost_usd
            # #192: backfill reasoning_tokens via Metrics.extra when the
            # MetricsEvent path didn't carry it (mirrors cost_usd backfill).
            if event.reasoning_tokens is not None and (
                existing.extra is None or "reasoning_tokens" not in existing.extra
            ):
                merged_extra = dict(existing.extra or {})
                merged_extra["reasoning_tokens"] = event.reasoning_tokens
                updates["extra"] = merged_extra
            if updates:
                existing = existing.model_copy(update=updates)
            assert self._open_step_dict is not None
            self._open_step_dict["_metrics"] = existing
        elif nonzero:
            # Create a residual step only for positive usage. Reasoning remains a completion subset in Metrics.extra.
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_metrics"] = delta
        # Zero-residual restatements create no phantom step and do not inflate total_steps.
        if event.model_name:
            if self._open_step_dict is not None:
                self._open_step_dict["_model_name"] = event.model_name
            self.recorder._upgrade_model_name(event.model_name)
        # Sum per-invocation residuals across phases, including CostEvent-only backends.
        self.recorder._accumulate_metrics(
            prompt_tokens=delta.prompt_tokens,
            completion_tokens=delta.completion_tokens,
            cached_tokens=delta.cached_tokens,
            cost_usd=delta.cost_usd,
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
        step["_tool_calls"].append(
            ToolCall(
                tool_call_id=event.id,
                function_name=event.name,
                arguments=event.input or {},
            )
        )
        # Results stay attached to their originating step, even after it closes.
        self._in_flight_tools[event.id] = len(self.steps)

    @_dispatch.register
    def _observe_tool_result(self, event: ToolResultEvent) -> None:
        host = self._in_flight_tools.pop(event.id, None)
        if host is None:
            # Preserve unmatched evidence without emitting an invalid source_call_id.
            self._ensure_open_step()["_unmatched_tool_results"].append(event.id)
            return
        result = ObservationResult(source_call_id=event.id, content=event.output, extra=_result_extra(event))
        if host == len(self.steps):
            self._ensure_open_step()["_observation_results"].append(result)
        else:
            self._amend_closed_step_observation(closed_index=host, result=result)

    @_dispatch.register
    def _observe_metrics(self, event: MetricsEvent) -> None:
        target = self._open_step_dict
        if target is None and not self.steps:
            target = self._ensure_open_step()
        # Cache and reasoning tokens are subsets of input/output, never additive.
        incoming = Metrics(
            prompt_tokens=event.prompt_tokens,
            completion_tokens=event.completion_tokens,
            cached_tokens=event.cached_tokens,
            cost_usd=event.cost_usd,
            extra=_reasoning_extra(event.reasoning_tokens),
        )
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
        # Keep invocation totals for terminal reconciliation and sum every billed turn.
        if event.prompt_tokens is not None:
            self._inv_metrics_sum["prompt"] += event.prompt_tokens
        if event.completion_tokens is not None:
            self._inv_metrics_sum["completion"] += event.completion_tokens
        if event.cached_tokens is not None:
            self._inv_metrics_sum["cached"] += event.cached_tokens
        if event.cost_usd is not None:
            self._inv_metrics_sum["cost"] += event.cost_usd
        self.recorder._accumulate_metrics(
            prompt_tokens=event.prompt_tokens,
            completion_tokens=event.completion_tokens,
            cached_tokens=event.cached_tokens,
            cost_usd=event.cost_usd,
        )
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
                    if step.source == "agent" and (step.model_name or "") in _GENERIC_MODEL_LABELS:
                        self.steps[index] = self.recorder.redactor.redact_step(
                            step.model_copy(update={"model_name": event.model_name})
                        )
            self.recorder._upgrade_model_name(event.model_name)
        self._close_open_step()

    @_dispatch.register
    def _observe_turn_end(self, event: TurnEndEvent) -> None:
        self._close_open_step()

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
        """Materialize the open-step dict *d* into a redacted agent Step."""
        message_text = "".join(d["_text_chunks"])
        reasoning = "\n".join(d["_thinking_chunks"]) if d["_thinking_chunks"] else None
        tool_calls = list(d["_tool_calls"]) or None
        observation = Observation(results=list(d["_observation_results"])) if d["_observation_results"] else None
        extra: dict[str, Any] = {
            "daydream_phase": self.phase.value,
            "daydream_run_flow": self.recorder.run_flow.value,
            **extra_overrides,
        }
        if d["_backend_diagnostics"]:
            extra["backend_diagnostics"] = list(d["_backend_diagnostics"])
        agent_step = Step(
            step_id=step_id,
            timestamp=timeutil.now_iso(),
            source="agent",
            message=message_text,
            model_name=d["_model_name"],
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            observation=observation,
            metrics=d["_metrics"],
            llm_call_count=1,
            extra=extra,
        )
        return self.recorder.redactor.redact_step(agent_step)

    def _close_open_step(self) -> None:
        """Redact and append once; pending tools retain the same host index."""
        if self._open_step_dict is None:
            return
        d = self._open_step_dict
        self._open_step_dict = None

        extra_overrides: dict[str, Any] = {}
        if d["_unmatched_tool_results"]:
            extra_overrides["unmatched_tool_results"] = list(d["_unmatched_tool_results"])
        if self._stop_reason is not None:
            extra_overrides["stop_reason"] = self._stop_reason
        if self._error_subtype is not None:
            extra_overrides["error"] = True
            extra_overrides["error_subtype"] = self._error_subtype

        self.steps.append(
            self._materialize_agent_step(
                d,
                step_id=self.recorder._next_step_id(),
                extra_overrides=extra_overrides,
            )
        )

    def _amend_closed_step_observation(self, *, closed_index: int, result: ObservationResult) -> None:
        """Append a late result to its closed host step, then redact the replacement."""
        updated = self._with_observation_result(self.steps[closed_index], result)
        self.steps[closed_index] = self.recorder.redactor.redact_step(updated)

    def _fold_metrics_into_closed_last_step(self, event: Any, incoming: Metrics) -> None:
        """Fold late terminal usage into the last agent step, preserving step/final usage equality."""
        for idx in range(len(self.steps) - 1, -1, -1):
            step = self.steps[idx]
            if step.source != "agent":
                continue
            metrics = incoming if step.metrics is None else _merge_metrics(step.metrics, incoming)
            updates: dict[str, Any] = {"metrics": metrics}
            if event.model_name:
                updates["model_name"] = event.model_name
            self.steps[idx] = self.recorder.redactor.redact_step(step.model_copy(update=updates))
            return
        # Without a closed agent step, create one so late usage is present in both steps and final totals.
        self._ensure_open_step()
        assert self._open_step_dict is not None
        self._open_step_dict["_metrics"] = incoming
        if event.model_name:
            self._open_step_dict["_model_name"] = event.model_name

    def snapshot_steps(self, *, snapshot_step_id: int | None = None) -> list[Step]:
        """Copy flushed/open steps with caller-supplied IDs and interruption markers; retain live tools."""
        in_flight = list(self._in_flight_tools.values())
        if self._open_step_dict is None:
            steps = list(self.steps)
        else:
            d = self._open_step_dict
            extra_overrides: dict[str, Any] = {"partial_step": True}
            if d["_unmatched_tool_results"]:
                extra_overrides["unmatched_tool_results"] = list(d["_unmatched_tool_results"])
            step_id = snapshot_step_id if snapshot_step_id is not None else self.recorder._step_id_counter + 1
            steps = [
                *self.steps,
                self._materialize_agent_step(d, step_id=step_id, extra_overrides=extra_overrides),
            ]
        # Markers land per host Step in the same LIFO order finish()'s popitem loop uses.
        for host in reversed(in_flight):
            steps[host] = self._with_observation_result(steps[host], self._interrupted_marker())
        return steps

    @staticmethod
    def _interrupted_marker() -> ObservationResult:
        """Build a fixed-ASCII interruption marker with scalar metadata.

        A null source_call_id distinguishes interruption from a completed tool result."""
        return ObservationResult(
            source_call_id=None,
            content=INCOMPLETE_CALL_CONTENT,
            extra={"is_error": True, "status": "interrupted"},
        )

    @staticmethod
    def _with_observation_result(step: Step, result: ObservationResult) -> Step:
        """Copy a step with an appended result for amendment or snapshotting.

        External results need redaction; fixed-ASCII snapshot markers do not."""
        if step.observation is None:
            observation = Observation(results=[result])
        else:
            observation = step.observation.model_copy(update={"results": [*step.observation.results, result]})
        return step.model_copy(update={"observation": observation})

    def _emit_incomplete_call_markers(self) -> None:
        """Pop unfinished calls and mark closed hosts interrupted; null IDs never claim completion."""
        while self._in_flight_tools:
            _, host = self._in_flight_tools.popitem()
            self._amend_closed_step_observation(
                closed_index=host,
                result=self._interrupted_marker(),
            )

    def finish(self) -> None:
        """Flush partial steps and unfinished tools, then drain sealed generations once and resolve billing."""
        self._close_open_step()
        self._emit_incomplete_call_markers()
        self._generation_ledger.finalize()
        self.recorder._extend_steps(self.steps)
