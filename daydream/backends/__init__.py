"""Backend protocol, normalized events, and construction.

Events carry host UTC timestamps at yield time. RequestEvent records only exposed
request facts; MetricsEvent owns turn usage, CostEvent owns invocation totals,
and TurnEndEvent closes a recorder step. Generation events describe provider
boundaries using host-local correlators. A terminal ResultEvent may precede an
exception, so its presence does not imply successful execution.
"""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Union, get_args, get_origin, get_type_hints

from daydream.redaction import redact_structured_text
from daydream.retry_policy import coerce_declared_retry_allowance
from daydream.timeutil import now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Complete retry settings owned by one constructed backend."""

    attempts: int
    base_delay_s: float
    max_delay_s: float
    #: Invocation retry overhead in seconds; zero disables recovery, None defers.
    #: A declared RetryPolicy suppresses ambient PI retry settings. Embedded runs
    #: materialize environment overrides into this policy during construction.
    retry_recovery_allowance_s: float | None = None


def _parsed_nonnegative_int(
    environment: Mapping[str, str], name: str, default: int
) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not a valid integer; using default %d", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%r is negative; using default %d", name, raw, default)
        return default
    return value


def _parsed_nonnegative_float(
    environment: Mapping[str, str], name: str, default: float
) -> float:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a valid float; using default %g", name, raw, default)
        return default
    if not math.isfinite(value):
        logger.warning("%s=%r is not finite; using default %g", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%r is negative; using default %g", name, raw, default)
        return default
    return value


def _parsed_positive_int(
    environment: Mapping[str, str], name: str, default: int
) -> int:
    value = _parsed_nonnegative_int(environment, name, default)
    if value == 0:
        logger.warning("%s must be positive; using default %d", name, default)
        return default
    return value


@dataclass(frozen=True)
class BackendExecutionInput:
    """Run-owned native process settings for one backend kind.

    The complete environment is copied into an immutable private mapping.
    Native transports receive a fresh mutable copy for each invocation.
    """

    _environment: Mapping[str, str] = field(repr=False, compare=False)
    retry_policy: RetryPolicy
    fanout_concurrency: int
    pi_provider: str | None
    pi_thinking: str | None
    pi_agent_dir: Path | None
    stream_idle_timeout_s: float | None
    pi_response_idle_timeout_s: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_environment", MappingProxyType(dict(self._environment))
        )

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str], *, backend: str
    ) -> BackendExecutionInput:
        """Parse one complete environment without consulting process globals."""
        from daydream.backends._subprocess import (
            DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S,
            DEFAULT_STREAM_IDLE_TIMEOUT_S,
        )

        if backend not in {"pi", "codex", "claude"}:
            if backend == "osprey":
                raise ValueError("explicit BackendExecutionInput is not supported for osprey")
            raise ValueError(f"unsupported backend for execution input: {backend!r}")

        copied = dict(environment)
        base_delay_default = 10.0 if backend == "pi" else 2.0
        fanout_name = (
            "DAYDREAM_PI_FANOUT_CONCURRENCY"
            if backend == "pi"
            else "DAYDREAM_FANOUT_CONCURRENCY"
        )
        fanout_default = 10 if backend == "pi" else 8
        idle = _parsed_nonnegative_float(
            copied, "DAYDREAM_STREAM_IDLE_TIMEOUT_S", DEFAULT_STREAM_IDLE_TIMEOUT_S
        )
        response_idle = _parsed_nonnegative_float(
            copied,
            "DAYDREAM_STREAM_IDLE_TIMEOUT_S",
            DEFAULT_PI_RESPONSE_IDLE_TIMEOUT_S,
        )
        home = copied.get("HOME")
        agent_dir_raw = copied.get("PI_CODING_AGENT_DIR")
        pi_agent_dir = (
            Path(agent_dir_raw)
            if agent_dir_raw
            else Path(home) / ".pi" / "agent"
            if home
            else None
        )

        return cls(
            copied,
            RetryPolicy(
                attempts=_parsed_nonnegative_int(
                    copied, "DAYDREAM_PI_RETRY_ATTEMPTS", 20
                ),
                base_delay_s=_parsed_nonnegative_float(
                    copied, "DAYDREAM_PI_RETRY_BASE_DELAY_S", base_delay_default
                ),
                max_delay_s=_parsed_nonnegative_float(
                    copied, "DAYDREAM_PI_RETRY_MAX_DELAY_S", 120.0
                ),
                # Parsed here, not in run_agent: this is the construction path
                # every embedded caller uses, and ``run_agent`` resolves the
                # documented top precedence tier (RetryPolicy) before the env, so
                # the operator knob must be materialised into the policy or it
                # would be silently dropped on this path only.
                retry_recovery_allowance_s=coerce_declared_retry_allowance(
                    copied.get("DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S"),
                    "DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S",
                ),
            ),
            _parsed_positive_int(copied, fanout_name, fanout_default),
            copied.get("PI_PROVIDER") or None,
            copied.get("PI_THINKING") or None,
            pi_agent_dir,
            None if idle == 0 else idle,
            None if response_idle == 0 else response_idle,
        )

    def child_environment(self) -> dict[str, str]:
        """Return a fresh complete environment for one native transport."""
        return dict(self._environment)

# host-side capability sentinel, not an external contract: Anthropic's hook
# token is spelled "PreToolUse" (see claude.py); do not version this literal.
AUDIT_ROOT_ISOLATION = "claude-pretooluse"
AuditIsolationReason = Literal[
    "unsupported_backend",
    "missing_capability",
    "wrong_capability",
    "wrong_root",
]


class AuditIsolationError(RuntimeError):
    """A backend cannot satisfy the requested improve audit boundary."""

    def __init__(
        self,
        backend_name: str,
        reason: AuditIsolationReason,
        *,
        phase: str | None = None,
    ) -> None:
        super().__init__(f"{backend_name}: {reason}")
        self.backend_name = backend_name
        self.reason = reason
        self.phase = phase


if TYPE_CHECKING:
    from claude_agent_sdk.types import AgentDefinition


#
# The helpers below admit only the exact typed representations the P18 plan's
# Effective Configuration Admission Contract allows: closed bool/int64/finite
# float admission with bool-not-int strictness, bounded identity labels that
# reject Unicode controls/bidi characters, redaction-changed values and known
# private paths while allowing ordinary model namespace slashes, and ordered
# identity lists capped at 16 entries with whole-list overflow omission.
#
# Every unsafe value is *omitted* (falls back to None) with a fixed diagnostic
# code — never truncated into a false identity and never echoed back.

EvidenceSource = Literal["configured", "host_generated", "native"]

#: Closed measurement provenance for usage records (P18 Task 1): which
#: protocol boundary supplied the numbers. ``None`` means a legacy record
#: emitted before this contract (source not assessed, never a claim).
MeasurementSource = Literal["message_end", "turn_end", "terminal", "session", "synthesized"]

#: Closed cost provenance: native reported cost versus host-synthesized
#: estimate from a price table (plan: distinguish ``reported`` vs ``estimated``).
CostSource = Literal["reported", "estimated"]

#: Closed recursive JSON value type for tool-call arguments (plan Task 1).
JsonValue = Union[None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]]

_MAX_MODEL_NAME_CHARS = 256
_MAX_PROVIDER_NAME_CHARS = 128
_MAX_TOOL_NAME_CHARS = 128
_MAX_ORDERED_IDENTITY_ENTRIES = 16
_INT64_MAX = 2**63 - 1
#: Largest native Unix-millisecond timestamp that survives exact ns conversion.
_MAX_NATIVE_UNIX_MS = _INT64_MAX // 1_000_000

# Fixed diagnostic codes (bounded, low-cardinality — no free-form backend text).
_DIAG_BOOL_TYPE = "config_bool_type_rejected"
_DIAG_INT_TYPE = "config_int_type_rejected"
_DIAG_INT_RANGE = "config_int_range_rejected"
_DIAG_FLOAT_TYPE = "config_float_type_rejected"
_DIAG_FLOAT_NOT_FINITE = "config_float_not_finite"
_DIAG_LITERAL = "config_mode_not_admitted"
_DIAG_IDENTITY_UNSAFE_CHARS = "config_identity_unsafe_characters"
_DIAG_IDENTITY_REDACTED = "config_identity_redaction_changed"
_DIAG_IDENTITY_PRIVATE_PATH = "config_identity_private_path"
_DIAG_IDENTITY_TOO_LONG = "config_identity_too_long"
_DIAG_LIST_MEMBER_UNSAFE = "config_list_member_unsafe"
_DIAG_LIST_OVERFLOW = "config_list_overflow"
_DIAG_JSON_VALUE = "config_json_value_rejected"
_DIAG_TIMESTAMP_TYPE = "config_timestamp_type_rejected"
_DIAG_TIMESTAMP_RANGE = "config_timestamp_range_rejected"

# Unicode format/bidi-control categories (Cf includes LRM/RLM and the bidi
# isolates/overrides) plus the non-ASCII line/paragraph separators. Controls
# (Cc) are rejected separately; ordinary printable text passes.
_BIDI_FORMAT_CATEGORIES = frozenset({"Cf"})
_LINE_SEPARATOR_CATEGORIES = frozenset({"Zl", "Zp"})
_CONTROL_CATEGORIES = frozenset({"Cc"})

# POSIX + Windows absolute-path spellings, and the drive/UNC prefixes that
# reveal a host filesystem location. Model namespace slashes (``org/model``,
# ``provider//model``) never match because they are not absolute.
_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class EvidenceDiagnostic:
    """One fixed-shape admission diagnostic (code + bounded detail)."""

    code: str
    detail: str = ""


def _has_unicode_controls(value: str) -> bool:
    """Reject C0/C1 controls and bidi/line-separator format characters."""
    return any(
        unicodedata.category(ch) in _CONTROL_CATEGORIES | _BIDI_FORMAT_CATEGORIES | _LINE_SEPARATOR_CATEGORIES
        for ch in value
    )


def _changed_by_redaction(value: str) -> bool:
    """Reject telemetry identity labels that contain credential-shaped text."""
    return redact_structured_text(value) != value


def _is_private_path(value: str) -> bool:
    """Identify absolute or home-anchored paths; relative model namespaces remain valid."""
    if not value:
        return False
    if value.startswith(("/", "\\\\")) or _WINDOWS_DRIVE_PREFIX.match(value):
        return True
    if value.startswith("~"):
        return True
    return value.startswith(("home/", "Users/"))


def _admit_identity_label(
    value: Any,
    *,
    max_chars: int,
    context: str,
) -> tuple[str | None, EvidenceDiagnostic | None]:
    """Admit a nonempty bounded label without controls, credentials, or private paths.

    Returns the unchanged value or a fixed diagnostic; never truncates an identity.
    """
    if not isinstance(value, str) or not value:
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_UNSAFE_CHARS, context)
    if len(value) > max_chars:
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_TOO_LONG, context)
    if _has_unicode_controls(value):
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_UNSAFE_CHARS, context)
    if _changed_by_redaction(value):
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_REDACTED, context)
    if _is_private_path(value):
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_PRIVATE_PATH, context)
    return value, None


def _admit_runtime_tool_name(value: Any) -> tuple[str | None, EvidenceDiagnostic | None]:
    """Admit a runtime tool name: ASCII ``[A-Za-z][A-Za-z0-9_.:-]{0,127}``."""
    admitted, diagnostic = _admit_identity_label(value, max_chars=_MAX_TOOL_NAME_CHARS, context="tool_name")
    if admitted is None:
        return None, diagnostic
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", admitted):
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_UNSAFE_CHARS, "tool_name")
    return admitted, None


def _admit_native_unix_ms(value: Any) -> tuple[int | None, EvidenceDiagnostic | None]:
    """Admit an exact nonnegative int whose nanosecond conversion fits int64.

    None is absence; malformed values yield diagnostics without coercion or clamping.
    """
    if value is None:
        return None, None
    if type(value) is not int:
        return None, EvidenceDiagnostic(_DIAG_TIMESTAMP_TYPE, "native_unix_ms")
    if not 0 <= value <= _MAX_NATIVE_UNIX_MS:
        return None, EvidenceDiagnostic(_DIAG_TIMESTAMP_RANGE, "native_unix_ms")
    return value, None


def unix_ms_to_ns(ms: int) -> int:
    """Convert native Unix milliseconds to nanoseconds by exact multiplication."""
    return ms * 1_000_000


def _admit_json_value(value: Any, depth: int = 0) -> tuple[Any, EvidenceDiagnostic | None]:
    """Copy JSON-compatible values, rejecting nonfinite floats and unsafe dictionary keys.

    Nesting beyond 64 levels and non-JSON types produce a fixed diagnostic.
    """
    if depth > 64:
        return None, EvidenceDiagnostic(_DIAG_JSON_VALUE, "json_depth")
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return value, None
    if isinstance(value, float):
        if math.isfinite(value):
            return value, None
        return None, EvidenceDiagnostic(_DIAG_JSON_VALUE, "json_float_not_finite")
    if isinstance(value, int):
        return value, None
    if isinstance(value, list):
        admitted: list[Any] = []
        for item in value:
            item_value, item_diagnostic = _admit_json_value(item, depth + 1)
            if item_diagnostic is not None:
                return None, item_diagnostic
            admitted.append(item_value)
        return admitted, None
    if isinstance(value, dict):
        admitted_map: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or _has_unicode_controls(key):
                return None, EvidenceDiagnostic(_DIAG_JSON_VALUE, "json_key")
            item_value, item_diagnostic = _admit_json_value(item, depth + 1)
            if item_diagnostic is not None:
                return None, item_diagnostic
            admitted_map[key] = item_value
        return admitted_map, None
    return None, EvidenceDiagnostic(_DIAG_JSON_VALUE, "json_type")


class _AdmissionBase:
    """Reject malformed controls at construction using the built-in field declarations.

    Adapters instead omit unsafe runtime evidence and emit fixed diagnostics.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        checks = next(_CONFIG_CHECKS[base] for base in type(self).__mro__ if base in _CONFIG_CHECKS)
        for name, check in checks:
            check(getattr(self, name), name)


def _require_bool(value: Any, field: str) -> None:
    """Reject a non-``bool`` value at construction."""
    if value is None or type(value) is bool:
        return
    raise ValueError(f"{field}: {_DIAG_BOOL_TYPE}")


def _require_nonnegative_int(value: Any, field: str) -> None:
    """Reject a non-``int`` or out-of-range value at construction."""
    if value is None:
        return
    if type(value) is not int:
        raise ValueError(f"{field}: {_DIAG_INT_TYPE}")
    if not 0 <= value <= _INT64_MAX:
        raise ValueError(f"{field}: {_DIAG_INT_RANGE}")


def _require_finite_float(value: Any, field: str) -> None:
    """Reject a non-finite or non-numeric value at construction."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: {_DIAG_FLOAT_TYPE}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field}: {_DIAG_FLOAT_NOT_FINITE}")


def _require_literal(value: Any, field: str, *, allowed: tuple[str, ...]) -> None:
    """Reject a value outside ``allowed`` at construction."""
    if value is None or value in allowed:
        return
    raise ValueError(f"{field}: {_DIAG_LITERAL}")


@dataclass(frozen=True)
class EffectiveRequestConfig(_AdmissionBase):
    """Effective request controls: None means absent, never an inferred default.

    Construction rejects invalid types/ranges/modes; adapters omit unsafe evidence.
    """

    finalization: bool | None = field(default=None, kw_only=True)
    temperature: float | None = None
    max_turns: int | None = None
    read_only: bool | None = None
    persist_session: bool | None = None
    continuation_mode: Literal["fresh", "resume", "fork"] | None = None
    model_mode: Literal["single", "multi_or_dynamic"] | None = None



@dataclass(frozen=True)
class ClaudeRequestConfig(EffectiveRequestConfig):
    """Claude effective controls; paths, environment, prompts, and credentials stay absent."""

    permission_mode: Literal["bypassPermissions"] | None = None
    tools_count: int | None = field(default=None, kw_only=True)
    allowed_tools_count: int | None = None
    allowed_tools_present: bool | None = None
    audit_tools_count: int | None = None
    audit_tools_present: bool | None = None
    setting_sources_present: bool | None = None
    native_output_format: bool | None = None
    buffer_limit_bytes: int | None = None
    hooks_enabled: bool | None = None



@dataclass(frozen=True)
class CodexRequestConfig(EffectiveRequestConfig):
    """Codex-only effective controls (bounded values; paths/stdin/env absent)."""

    sandbox_mode: Literal["read-only", "danger-full-access"] | None = None
    experimental_json: bool | None = None
    native_output_schema: bool | None = None
    read_only_isolation: bool | None = None



@dataclass(frozen=True)
class PiRequestConfig(EffectiveRequestConfig):
    """Pi-only effective controls (bounded values; system/prompt content absent)."""

    selected_tools_count: int | None = None
    selected_tools_present: bool | None = None
    no_tools: bool | None = field(default=None, kw_only=True)
    no_skills: bool | None = None
    schema_emulated: bool | None = None
    no_extensions: bool | None = field(default=None, kw_only=True)



@dataclass(frozen=True)
class OspreyRequestConfig(EffectiveRequestConfig):
    """Osprey argv facts with no arbitrary labels or inferred configuration.

    Temperature is present only when the adapter explicitly passes --temperature.
    """

    persona_present: bool | None = None
    toolset_present: bool | None = None
    approval_mode: Literal["deny-untrusted"] | None = None
    sandbox: bool | None = None
    immutable_surface: bool | None = None
    compress_context: bool | None = None
    ultracode: bool | None = None
    turn_timeout: int | None = None
    stream_idle_timeout_secs: int | None = None
    streaming_timeout_secs: int | None = None
    empty_completion_threshold: int | None = None
    driver_max_retries: int | None = None
    compress_min_bytes: int | None = None
    tool_result_cap: int | None = None
    tool_result_head: int | None = None
    tool_result_tail: int | None = None
    tool_result_max_lines: int | None = None
    retry_failure_threshold: int | None = None
    no_progress_family_threshold: int | None = None
    no_progress_family_window: int | None = None
    no_progress_artifact_threshold: int | None = None
    no_progress_suppression_window: int | None = None
    max_subagents: int | None = None
    llm_rpm: int | None = None
    observation_update_bytes: int | None = None
    observation_inline_bytes: int | None = None
    observation_admission_bytes: int | None = None
    vars_count: int | None = None



_ConfigCheck = Callable[[Any, str], None]


def _config_checks(schema: type[EffectiveRequestConfig]) -> tuple[tuple[str, _ConfigCheck], ...]:
    """Compile closed-field validators once from their declared types.

    Custom subclasses keep their own fields; admission checks only the nearest
    built-in config. Telemetry separately rejects unknown concrete config types.
    """
    hints = get_type_hints(schema)
    scalar_checks: dict[Any, _ConfigCheck] = {
        bool: _require_bool, int: _require_nonnegative_int, float: _require_finite_float,
    }
    checks: list[tuple[str, _ConfigCheck]] = []
    for config_field in fields(schema):
        (value_type,) = (part for part in get_args(hints[config_field.name]) if part is not type(None))
        check = (
            partial(_require_literal, allowed=get_args(value_type))
            if get_origin(value_type) is Literal else scalar_checks[value_type]
        )
        checks.append((config_field.name, check))
    return tuple(checks)


_CONFIG_CHECKS = {
    schema: _config_checks(schema)
    for schema in (
        EffectiveRequestConfig, ClaudeRequestConfig, CodexRequestConfig, PiRequestConfig, OspreyRequestConfig,
    )
}


@dataclass
class RequestEvent:
    """Effective Daydream request, after adapter transformations.

    Only exposed request data belongs here; backend-internal prompts and
    environment/configuration dictionaries are never inferred or copied.
    ``system_prompt`` contains only the system text explicitly sent by Daydream.
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
    input_incomplete: bool = False


@dataclass
class ToolResultEvent:
    """Tool completion correlated with ToolStartEvent.id.

    Native exit/status/duration metadata stays None when unavailable; cancellation
    and truncation flags default to False.
    """

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
    """Invocation billing totals; None means unavailable.

    Cache reads/writes are subsets of total input, and reasoning tokens are a subset
    of output. model_usage contains per-model attribution, not extra billing.
    Native model identity upgrades generic recorder labels. cost_source distinguishes
    provider totals from host price-table estimates; absent provenance makes no claim.
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
    """Turn usage correlated by message_id; Codex uses an empty id and invocation scope.

    Cache reads/writes are subsets of prompt tokens, and reasoning is a subset of
    completion tokens. Native duration/start and model/provider identities are optional.
    Measurement/cost provenance distinguishes native observations from estimates.
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
    """Host receipt of a provider generation boundary, correlated by a host-local UUID.

    The timestamp is not a claimed provider start. boundary_complete=False marks
    synthetic/missing boundaries so consumers cannot infer a complete start/end pair.
    """

    generation_id: str
    observed_at_unix_ns: int
    boundary_complete: bool = True
    timestamp: str = field(default_factory=now_iso)


@dataclass
class GenerationEndEvent:
    """Seal ordered provider choice parts and generation lifecycle evidence.

    The optional native start is validated Unix milliseconds; end time is host receipt
    unless end_source says otherwise. Later tool execution links by call id and never
    authors or duplicates choice parts. Provider response_id does not replace generation_id.
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
    """Terminal metadata; a failed backend may emit it before raising.

    run_agent validates or salvage-checks structured_output when enabled; callers
    opting out validate downstream. session_id is independent of continuation requests.
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
    # Emulated schemas may select text fragments; staged callers validate the original final turn.
    structured_output_origin: Literal["native", "text"] = "native"


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


class AgentEventStream(AsyncIterator[AgentEvent], Protocol):
    """Closable event stream owned by one backend invocation.

    Closing the stream must release only the resources created by the matching
    :meth:`Backend.execute` call. It must not interrupt other streams returned
    by the same backend instance.
    """

    async def aclose(self) -> None:
        """Close this invocation and release its resources."""
        ...


class Backend(Protocol):
    """Yield normalized events from execute; keep invocation resources independent.

    Optional capabilities (read via getattr):
    - fanout_concurrency: scheduling hint, default 4; capped by the workflow.
    - concise_fix_prompts: suppress verbose fix reasoning, default False.
    - read_only_disposable_clone: inline bounded diffs and exploration context for
      disposable checkouts; correction prompts retain the untrusted-content boundary.
    - audit_root_isolation/audit_root: tool-layer confinement to an exact Improve
      snapshot. Callers require AUDIT_ROOT_ISOLATION and canonical root equality;
      this does not claim an OS sandbox.
    - supports_finalization: permits invocation-local finalization=True with reduced
      reasoning and native tool controls. Codex still requires a host zero-tool guard.
    - supports_tools_disabled: removes tools without lowering reasoning or changing
      the task. Distinct from finalization, read-only mode, and tool-call budgets.
    - supports_budget_preamble: accepts wall_budget_s/tool_call_budget so the
      backend's own system prompt can state this turn's real allowances rather
      than its module defaults. The host enforces the bound either way.
    - supports_complete_output: accepts optional require_complete_root for strict
      staged JSON syntax; malformed roots remain terminal.
    - reasoning_effort: native level fixed at construction; None defers to the driver.
      Backend instances are cached by kind, model, effort, and audit root.
    """

    model: str

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, AgentDefinition] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AgentEventStream:
        """Yield events while owning this invocation's stream resources.

        read_only enables a non-mutating tool profile. Codex additionally clones Git
        worktree roots, including staged/tracked/nonignored files but no source remote,
        so Git metadata changes affect the disposable clone. Its filesystem sandbox alone
        does not protect Git refs. The source path is excluded from argv/stdin/env/cwd;
        an independently discovered source path remains a limitation. Non-root cwd values
        use the sandbox in place. Callers explicitly choose read-only diagnostic/recon
        work; mutating phases retain the default.

        persist_session=False requests no resumable session; stateless backends ignore it.
        """
        ...

    async def cancel(self) -> None:
        """Cancel every active invocation on this backend.

        This backend-wide operation is reserved for process shutdown. Callers
        ending one invocation must close the corresponding AgentEventStream.
        """
        ...


def resolve_fanout_concurrency(env_var: str, default: int) -> int:
    """Read a positive endpoint concurrency hint; warn and use the default if malformed."""
    return _parsed_positive_int(os.environ, env_var, default)


def effective_fanout_concurrency(workflow_ceiling: int, backend: object) -> int:
    """Combine a positive workflow ceiling with a backend scheduling hint."""
    hint = getattr(backend, "fanout_concurrency", 4)
    if not isinstance(hint, int) or isinstance(hint, bool) or hint <= 0:
        hint = 4
    return min(workflow_ceiling, hint)


def create_backend(
    name: str,
    model: str | None = None,
    *,
    cwd: Path | None = None,
    reasoning_effort: str | None = None,
    osprey_binary: str | None = None,
    audit_root: Path | None = None,
    audit_outward_symlinks: frozenset[Path] = frozenset(),
    execution_input: BackendExecutionInput | None = None,
) -> Backend:
    """Construct a backend with explicit model, effort, and execution settings.

    Claude/Codex apply built-in models; Pi resolves its configured model before its
    fallback. Effort uses each driver's native control. Only Claude supports an exact
    audit_root with lexical outward-symlink restrictions. Unknown names raise ValueError.
    """
    from daydream.config import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL

    if name == "claude":
        return ClaudeBackend(
            model=model or DEFAULT_CLAUDE_MODEL,
            reasoning_effort=reasoning_effort,
            audit_root=audit_root,
            audit_outward_symlinks=audit_outward_symlinks,
            execution_input=execution_input,
        )
    if audit_root is not None and name in {"codex", "pi", "osprey"}:
        raise AuditIsolationError(name, "unsupported_backend")
    if name == "codex":
        from daydream.backends.codex import CodexBackend

        return CodexBackend(
            model=model or DEFAULT_CODEX_MODEL,
            reasoning_effort=reasoning_effort,
            execution_input=execution_input,
        )
    if name == "pi":
        return PiBackend(
            model=model,
            cwd=cwd,
            reasoning_effort=reasoning_effort,
            execution_input=execution_input,
        )
    if name == "osprey":
        if execution_input is not None:
            raise ValueError("explicit BackendExecutionInput is not supported for osprey")
        return OspreyBackend(
            model=model,
            cwd=cwd,
            reasoning_effort=reasoning_effort,
            osprey_binary=osprey_binary,
        )
    raise ValueError(f"Unknown backend: {name!r}. Expected 'claude', 'codex', 'pi', or 'osprey'.")


from daydream.backends.claude import ClaudeBackend, MaxTurnsError  # noqa: E402
from daydream.backends.osprey import OspreyBackend  # noqa: E402
from daydream.backends.pi import PiBackend  # noqa: E402

__all__ = [
    "AUDIT_ROOT_ISOLATION",
    "AgentEvent",
    "AgentEventStream",
    "AuditIsolationError",
    "Backend",
    "BackendExecutionInput",
    "ClaudeBackend",
    "ClaudeRequestConfig",
    "CodexRequestConfig",
    "ContinuationToken",
    "CostEvent",
    "DiagnosticEvent",
    "GenerationEndEvent",
    "GenerationStartEvent",
    "JsonValue",
    "MaxTurnsError",
    "MetricsEvent",
    "ModelUsageTotals",
    "OspreyBackend",
    "OspreyRequestConfig",
    "PiBackend",
    "PiRequestConfig",
    "ReasoningChoicePart",
    "RequestEvent",
    "RetryPolicy",
    "ResultEvent",
    "TextChoicePart",
    "TextEvent",
    "ThinkingEvent",
    "ToolCallChoicePart",
    "ToolResultEvent",
    "ToolStartEvent",
    "TurnEndEvent",
    "create_backend",
    "effective_fanout_concurrency",
    "unix_ms_to_ns",
]
