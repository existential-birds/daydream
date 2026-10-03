"""Normalized backend events. Timestamps record host receipt unless declared native."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from daydream.backends._evidence import (
    CostSource,
    EffectiveRequestConfig,
    EvidenceSource,
    JsonValue,
    MeasurementSource,
    _admit_json_value,
    _admit_runtime_tool_name,
)
from daydream.timeutil import now_iso


@dataclass
class RequestEvent:
    """Effective request facts after adapter transformations; system_prompt is only text Daydream sent.
    Internal prompts, environments, and inferred configuration never enter this event.
    """

    prompt: str
    system_prompt: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    session_id: str | None = None
    reasoning_effort: str | None = None
    output_schema: dict[str, Any] | None = None
    timestamp: str = field(default_factory=now_iso)
    # Appended fields preserve positional construction through timestamp.
    config: EffectiveRequestConfig = field(default_factory=EffectiveRequestConfig)
    model_source: EvidenceSource | None = None
    provider_source: EvidenceSource | None = None
    session_source: EvidenceSource | None = None
    timestamp_source: Literal["host_observed", "native"] = "host_observed"


@dataclass
class TextEvent:
    """Agent text, timestamped at backend yield."""

    text: str
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ThinkingEvent:
    """Reasoning text, timestamped at backend yield."""

    text: str
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ToolStartEvent:
    """Tool call with native or synthesized id and non-null arguments."""

    id: str
    name: str
    input: dict[str, Any]
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ToolResultEvent:
    """Completion correlated by ToolStartEvent.id; unavailable native metadata remains None."""

    id: str
    output: str
    is_error: bool
    timestamp: str = field(default_factory=now_iso)
    # Optional metadata follows the original positional fields.
    exit_code: int | None = None
    status: str | None = None
    duration_ms: float | None = None
    cancelled: bool = False
    truncated: bool = False


@dataclass
class DiagnosticEvent:
    """Recorder-only parser/transport evidence, normalized and redacted before persistence."""

    code: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=now_iso)


@dataclass(frozen=True)
class ModelUsageTotals:
    """Selected native per-model billing, with cache subsets of total input."""

    model_name: str
    provider_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class CostEvent:
    """Invocation billing totals; None means unavailable and missing provenance makes no claim.
    Cache counts are input subsets; reasoning is an output subset. model_usage attributes the
    same billing by model; cost_source distinguishes native totals from host estimates.
    """

    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    model_name: str | None = None
    timestamp: str = field(default_factory=now_iso)
    provider_name: str | None = None
    cache_creation_tokens: int | None = None
    model_usage: dict[str, ModelUsageTotals] | None = None
    # Missing provenance means unassessed, never an inferred source.
    measurement_source: MeasurementSource | None = None
    generation_id: str | None = None
    cost_source: CostSource | None = None


@dataclass
class MetricsEvent:
    """Turn usage by message_id; Codex uses an empty id with invocation scope.
    Cache and reasoning counts are subsets of prompt and completion counts respectively.
    Native timing/identity and measurement provenance remain optional.
    """

    message_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None
    cost_usd: float | None
    reasoning_tokens: int | None = None
    model_name: str | None = None
    timestamp: str = field(default_factory=now_iso)
    usage_scope: Literal["message", "invocation"] = "message"
    provider_name: str | None = None
    cache_creation_tokens: int | None = None
    duration_ms: float | None = None
    started_at: str | None = None
    # Missing provenance means unassessed.
    measurement_source: MeasurementSource | None = None
    generation_id: str | None = None


@dataclass
class TurnEndEvent:
    """Close one assistant step; message_id is empty when the backend exposes none."""

    message_id: str = ""
    timestamp: str = field(default_factory=now_iso)
    # Preserve positional construction of message_id and timestamp.
    finish_reason: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    message_id_source: EvidenceSource | None = None
    model_source: EvidenceSource | None = None
    provider_source: EvidenceSource | None = None
    timestamp_source: Literal["host_observed", "native"] = "host_observed"


#
# Provider-generation evidence, frozen per the plan's Task 1 shape. Only Pi
# currently exposes a real generation boundary (assistant message_start /
# message_end); the other three backends stay structural and emit none of
# these events. Correlation is host invocation-local: a generation ID never
# claims provider identity, and a late native response ID attaches to the
# matching end without ever replacing the correlator.


def _new_generation_id() -> str:
    """Mint one bounded host invocation-local generation correlation ID."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class TextChoicePart:
    """Ordered provider-choice text part (sealed at generation end)."""

    kind: Literal["text"] = "text"
    text: str = ""


@dataclass(frozen=True)
class ReasoningChoicePart:
    """Ordered provider-choice reasoning part (sealed at generation end)."""

    kind: Literal["reasoning"] = "reasoning"
    text: str = ""


@dataclass(frozen=True)
class ToolCallChoicePart:
    """Sealed provider tool choice with admitted name and JSON arguments; never coerced."""

    call_id: str
    name: str
    arguments: JsonValue
    kind: Literal["tool_call"] = "tool_call"

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise ValueError("tool_call choice part requires a non-empty call_id")
        name_admitted, diagnostic = _admit_runtime_tool_name(self.name)
        if name_admitted is None or diagnostic is not None:
            raise ValueError(f"tool_call choice part name rejected: {diagnostic.code if diagnostic else 'unsafe'}")
        arguments_admitted, arguments_diagnostic = _admit_json_value(self.arguments)
        if arguments_admitted is None and arguments_diagnostic is not None:
            raise ValueError(f"tool_call choice part arguments rejected: {arguments_diagnostic.code}")


AssistantChoicePart = TextChoicePart | ReasoningChoicePart | ToolCallChoicePart


@dataclass
class GenerationStartEvent:
    """Host receipt of a generation boundary, correlated by a host-local UUID.
    This is not provider start time; incomplete boundaries cannot establish a complete pair.
    """

    generation_id: str
    observed_at_unix_ns: int
    boundary_complete: bool = True
    timestamp: str = field(default_factory=now_iso)


@dataclass
class GenerationEndEvent:
    """Seal ordered provider choices; native start uses validated Unix milliseconds.
    End time is host receipt unless end_source specifies otherwise. Later tools link by call id;
    response_id never replaces the host generation_id or authors duplicate choices.
    """

    generation_id: str
    native_started_at_unix_ms: int | None
    ended_at_unix_ns: int
    end_source: Literal["host_observed_message_end", "native", "fallback"]
    choice_parts: tuple[AssistantChoicePart, ...] = ()
    response_id: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    finish_reason: str | None = None
    boundary_complete: bool = True
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ContinuationToken:
    """Opaque token for multi-turn interactions."""

    backend: str
    data: dict[str, Any]


@dataclass
class ResultEvent:
    """Terminal metadata may precede a backend exception. run_agent validates/salvages output
    unless the caller opts out; session_id exists independently of continuation requests.
    """

    structured_output: Any | None
    continuation: ContinuationToken | None
    timestamp: str = field(default_factory=now_iso)
    # Preserve timestamp as the third positional argument.
    model_name: str | None = None
    provider_name: str | None = None
    session_id: str | None = None
    finish_reason: str | None = None
    duration_ms: float | None = None
    duration_api_ms: float | None = None


AgentEvent = (
    RequestEvent
    | TextEvent
    | ThinkingEvent
    | ToolStartEvent
    | ToolResultEvent
    | DiagnosticEvent
    | CostEvent
    | MetricsEvent
    | TurnEndEvent
    | GenerationStartEvent
    | GenerationEndEvent
    | ResultEvent
)
