"""Backend abstraction layer for daydream.

Defines the unified event stream, Backend protocol, and factory function.
Backends yield AgentEvent instances that the UI layer consumes without
knowing which backend produced them.

Event vocabulary (members of the ``AgentEvent`` TypeAlias union):

- ``RequestEvent`` — actual request after backend transformations.
- ``TextEvent`` — agent text output.
- ``ThinkingEvent`` — extended reasoning / thinking content.
- ``ToolStartEvent`` — tool invocation started.
- ``ToolResultEvent`` — tool invocation completed.
- ``DiagnosticEvent`` — backend parser/transport coverage evidence.
- ``CostEvent`` — end-of-call cost/usage signal.
- ``MetricsEvent`` — per-turn LLM token/cost usage.
- ``TurnEndEvent`` — assistant-turn boundary; closes the recorder's open
  Step so multi-turn invocations are not collapsed into one Step.
- ``GenerationStartEvent`` / ``GenerationEndEvent`` — provider-generation
  lifecycle with a sealed typed choice (currently Pi only); correlation is
  a host invocation-local generation ID, never a provider identity.
- ``ResultEvent`` — terminal metadata, structured output and continuation;
  a failed backend can subsequently raise after exposing its billed usage.
"""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Union

from daydream.trajectory import now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Complete retry settings owned by one constructed backend."""

    attempts: int
    base_delay_s: float
    max_delay_s: float


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

# external contract: "claude-pretooluse" names Anthropic's PreToolUse hook
# capability, not a project-owned generation; do not version it.
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


# --- Effective Configuration Admission Contract (P18 Task 1) ---------------
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
_MAX_PHASE_NAME_CHARS = 128
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

# Characters that can smuggle a directional override through a name.
_BIDI_CODEPOINTS = (
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\u2066",  # LRI
    "\u2067",  # RLI
    "\u2068",  # FSI
    "\u2069",  # PDI
)

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
    return bool(_BIDI_CODEPOINTS_RE.search(value)) or any(
        unicodedata.category(ch) in _CONTROL_CATEGORIES | _BIDI_FORMAT_CATEGORIES | _LINE_SEPARATOR_CATEGORIES
        for ch in value
    )


_BIDI_CODEPOINTS_RE = re.compile("|".join(re.escape(cp) for cp in _BIDI_CODEPOINTS))


def _changed_by_redaction(value: str) -> bool:
    """True when the shared credential redactor would rewrite *value*.

    A configured label that only survives redaction because it *is* shaped
    like a credential must never be emitted verbatim into telemetry; the
    admission boundary drops it with a fixed diagnostic instead.
    """
    from daydream.trajectory import redact_structured_text

    return redact_structured_text(value) != value


def _is_private_path(value: str) -> bool:
    """True for absolute POSIX/Windows paths and home-anchored locations.

    Model namespace slashes (``nous/deepseek-v4``) are ordinary identity
    spellings and never match: a namespace segment is not rooted, has no
    drive, and contains no ``~`` expansion.
    """
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
    """Admit one model/provider/phase identity label per the contract.

    Requires an exact ``str``, length 1..max_chars (counted in Unicode scalar
    values), no C0/C1 controls, no bidi/format characters, no line/paragraph
    separators, unchanged by credential redaction, and never equal to or
    containing a known private absolute path. Returns ``(value, None)`` on
    success (never truncated) or ``(None, diagnostic)``.
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


def _admit_phase_name(value: Any) -> tuple[str | None, EvidenceDiagnostic | None]:
    """Admit an extension phase key: ASCII ``[A-Za-z][A-Za-z0-9_.-]{0,127}``."""
    admitted, diagnostic = _admit_identity_label(value, max_chars=_MAX_PHASE_NAME_CHARS, context="phase_name")
    if admitted is None:
        return None, diagnostic
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", admitted):
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_UNSAFE_CHARS, "phase_name")
    return admitted, None


def _admit_bool(value: Any, context: str) -> tuple[bool | None, EvidenceDiagnostic | None]:
    """Admit an optional real ``bool`` (``type(value) is bool``)."""
    if value is None:
        return None, None
    if type(value) is not bool:
        return None, EvidenceDiagnostic(_DIAG_BOOL_TYPE, context)
    return value, None


def _admit_nonnegative_int(value: Any, context: str) -> tuple[int | None, EvidenceDiagnostic | None]:
    """Admit an optional exact ``int`` in ``[0, 2**63-1]`` (bool rejected)."""
    if value is None:
        return None, None
    if isinstance(value, bool) or type(value) is not int:
        return None, EvidenceDiagnostic(_DIAG_INT_TYPE, context)
    if not 0 <= value <= _INT64_MAX:
        return None, EvidenceDiagnostic(_DIAG_INT_RANGE, context)
    return value, None


def _admit_finite_float(value: Any, context: str) -> tuple[float | None, EvidenceDiagnostic | None]:
    """Admit an optional finite ``float`` (bool rejected; NaN/inf rejected)."""
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, EvidenceDiagnostic(_DIAG_FLOAT_TYPE, context)
    result = float(value)
    if not math.isfinite(result):
        return None, EvidenceDiagnostic(_DIAG_FLOAT_NOT_FINITE, context)
    return result, None


def _admit_literal(value: Any, allowed: tuple[str, ...], context: str) -> tuple[str | None, EvidenceDiagnostic | None]:
    """Admit one exact case-sensitive closed-mode member."""
    if value is None:
        return None, None
    if value in allowed:
        return value, None
    return None, EvidenceDiagnostic(_DIAG_LITERAL, context)


def _admit_native_unix_ms(value: Any) -> tuple[int | None, EvidenceDiagnostic | None]:
    """Admit a strict native Unix-millisecond timestamp.

    Requires ``type(value) is int`` (bool rejected), ``0 <= value`` and a
    value that survives exact multiplication to nanoseconds without int64
    overflow: ``value <= (2**63-1) // 1_000_000``. Returns ``(None, None)``
    for ``None`` (absence is not malformation); malformed values yield
    ``(None, diagnostic)``. Never clamps or coerces.
    """
    if value is None:
        return None, None
    if isinstance(value, bool) or type(value) is not int:
        return None, EvidenceDiagnostic(_DIAG_TIMESTAMP_TYPE, "native_unix_ms")
    if not 0 <= value <= _MAX_NATIVE_UNIX_MS:
        return None, EvidenceDiagnostic(_DIAG_TIMESTAMP_RANGE, "native_unix_ms")
    return value, None


def unix_ms_to_ns(ms: int) -> int:
    """Convert native Unix milliseconds to nanoseconds by exact multiplication."""
    return ms * 1_000_000


def _admit_json_value(value: Any, depth: int = 0) -> tuple[Any, EvidenceDiagnostic | None]:
    """Validate one closed recursive JSON value (tool-call arguments).

    Accepts exactly ``None | bool | int | float | str | list | dict[str, ...]``
    with finite floats and JSON-compatible dict keys. Non-JSON types (bytes,
    sets, objects, non-finite floats) fail closed with a fixed diagnostic.
    Depth is bounded defensively against pathological nesting.
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
    """Construction-time rejection for the frozen admission dataclasses.

    Subclasses implement :meth:`_validate` by calling the ``_require_*``
    helpers on every field; any non-admitted value raises ``ValueError`` so a
    wrong-typed or out-of-range control can never exist inside the closed
    type (plan Step 1: "dataclass construction rejects"). The runtime
    admission helpers above are the complementary layer adapters use to
    *omit* an unsafe backend value with a fixed diagnostic instead of ever
    constructing with one.
    """

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        raise NotImplementedError


def _require_bool(value: Any, field: str) -> None:
    admitted, diagnostic = _admit_bool(value, field)
    if diagnostic is not None:
        raise ValueError(f"{field}: {diagnostic.code}")


def _require_nonnegative_int(value: Any, field: str) -> None:
    admitted, diagnostic = _admit_nonnegative_int(value, field)
    if diagnostic is not None:
        raise ValueError(f"{field}: {diagnostic.code}")


def _require_finite_float(value: Any, field: str) -> None:
    admitted, diagnostic = _admit_finite_float(value, field)
    if diagnostic is not None:
        raise ValueError(f"{field}: {diagnostic.code}")


def _require_literal(value: Any, allowed: tuple[str, ...], field: str) -> None:
    admitted, diagnostic = _admit_literal(value, allowed, field)
    if diagnostic is not None:
        raise ValueError(f"{field}: {diagnostic.code}")


@dataclass(frozen=True)
class EffectiveRequestConfig(_AdmissionBase):
    """Closed common-subset request configuration, per the admission contract.

    Every field is ``None`` when the control was not actually passed/effective
    — absence never claims a default. Construction rejects wrong-typed,
    out-of-range and non-admitted values (``ValueError``); adapters omit an
    unsafe backend value before ever constructing.
    """

    temperature: float | None = None
    max_turns: int | None = None
    read_only: bool | None = None
    persist_session: bool | None = None
    continuation_mode: Literal["fresh", "resume", "fork"] | None = None
    model_mode: Literal["single", "multi_or_dynamic"] | None = None

    def _validate(self) -> None:
        _require_finite_float(self.temperature, "temperature")
        _require_nonnegative_int(self.max_turns, "max_turns")
        _require_bool(self.read_only, "read_only")
        _require_bool(self.persist_session, "persist_session")
        _require_literal(self.continuation_mode, ("fresh", "resume", "fork"), "continuation_mode")
        _require_literal(self.model_mode, ("single", "multi_or_dynamic"), "model_mode")


@dataclass(frozen=True)
class ClaudeRequestConfig(EffectiveRequestConfig):
    """Claude-only effective controls layered on the closed common subset.

    Cwd, environment, agent definitions, prompts, persona labels and
    credentials stay absent by construction: only the admitted shapes below
    exist as fields, so nothing else can be smuggled into telemetry.
    """

    permission_mode: Literal["bypassPermissions"] | None = None
    allowed_tools_count: int | None = None
    allowed_tools_present: bool | None = None
    audit_tools_count: int | None = None
    audit_tools_present: bool | None = None
    setting_sources_present: bool | None = None
    native_output_format: bool | None = None
    buffer_limit_bytes: int | None = None
    hooks_enabled: bool | None = None

    def _validate(self) -> None:
        super()._validate()
        _require_literal(self.permission_mode, ("bypassPermissions",), "permission_mode")
        _require_nonnegative_int(self.allowed_tools_count, "allowed_tools_count")
        _require_bool(self.allowed_tools_present, "allowed_tools_present")
        _require_nonnegative_int(self.audit_tools_count, "audit_tools_count")
        _require_bool(self.audit_tools_present, "audit_tools_present")
        _require_bool(self.setting_sources_present, "setting_sources_present")
        _require_bool(self.native_output_format, "native_output_format")
        _require_nonnegative_int(self.buffer_limit_bytes, "buffer_limit_bytes")
        _require_bool(self.hooks_enabled, "hooks_enabled")


@dataclass(frozen=True)
class CodexRequestConfig(EffectiveRequestConfig):
    """Codex-only effective controls (bounded values; paths/stdin/env absent)."""

    sandbox_mode: Literal["read-only", "danger-full-access"] | None = None
    experimental_json: bool | None = None
    native_output_schema: bool | None = None
    read_only_isolation: bool | None = None

    def _validate(self) -> None:
        super()._validate()
        _require_literal(self.sandbox_mode, ("read-only", "danger-full-access"), "sandbox_mode")
        _require_bool(self.experimental_json, "experimental_json")
        _require_bool(self.native_output_schema, "native_output_schema")
        _require_bool(self.read_only_isolation, "read_only_isolation")


@dataclass(frozen=True)
class PiRequestConfig(EffectiveRequestConfig):
    """Pi-only effective controls (bounded values; system/prompt content absent)."""

    selected_tools_count: int | None = None
    selected_tools_present: bool | None = None
    no_skills: bool | None = None
    schema_emulated: bool | None = None

    def _validate(self) -> None:
        super()._validate()
        _require_nonnegative_int(self.selected_tools_count, "selected_tools_count")
        _require_bool(self.selected_tools_present, "selected_tools_present")
        _require_bool(self.no_skills, "no_skills")
        _require_bool(self.schema_emulated, "schema_emulated")


@dataclass(frozen=True)
class OspreyRequestConfig(EffectiveRequestConfig):
    """Osprey-only effective controls (exact argv facts; labels stay absent).

    Only explicit temperature=0.0 is eligible for the inherited common
    ``temperature`` field (the Osprey adapter passes it only when it actually
    emitted ``--temperature``); hidden/config-resolved temperatures stay
    absent.
    """

    persona_present: bool | None = None
    toolset_present: bool | None = None
    approval_mode: Literal["deny-untrusted"] | None = None
    sandbox: bool | None = None
    immutable_surface: bool | None = None
    compress_context: bool | None = None
    ultracode: bool | None = None
    max_turns: int | None = None
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

    def _validate(self) -> None:
        super()._validate()
        _require_bool(self.persona_present, "persona_present")
        _require_bool(self.toolset_present, "toolset_present")
        _require_literal(self.approval_mode, ("deny-untrusted",), "approval_mode")
        _require_bool(self.sandbox, "sandbox")
        _require_bool(self.immutable_surface, "immutable_surface")
        _require_bool(self.compress_context, "compress_context")
        _require_bool(self.ultracode, "ultracode")
        for int_field in (
            "max_turns",
            "turn_timeout",
            "stream_idle_timeout_secs",
            "streaming_timeout_secs",
            "empty_completion_threshold",
            "driver_max_retries",
            "compress_min_bytes",
            "tool_result_cap",
            "tool_result_head",
            "tool_result_tail",
            "tool_result_max_lines",
            "retry_failure_threshold",
            "no_progress_family_threshold",
            "no_progress_family_window",
            "no_progress_artifact_threshold",
            "no_progress_suppression_window",
            "max_subagents",
            "llm_rpm",
            "observation_update_bytes",
            "observation_inline_bytes",
            "observation_admission_bytes",
            "vars_count",
        ):
            _require_nonnegative_int(getattr(self, int_field), int_field)


@dataclass
class RequestEvent:
    """Effective Daydream request, after adapter transformations.

    Only exposed request data belongs here; backend-internal prompts and
    environment/configuration dictionaries are never inferred or copied.
    ``system_prompt`` contains only the system text explicitly sent by Daydream.

    P18 Task 1 additions (all additive; older call sites stay valid):

    ``config`` carries the closed typed Effective Configuration Admission
    Contract dataclass (a frozen subclass of :class:`_AdmissionBase`), never a
    free-form bag. ``*_source`` fields distinguish configured, host-generated
    and native provenance for identity fields; ``None`` means the field's
    provenance was not separately established. ``timestamp_source`` is
    ``"native"`` only when the backend supplied the request timestamp from its
    own protocol handshake (Osprey ``session_start``); every other backend is
    ``"host_observed"``.
    """

    prompt: str
    system_prompt: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    session_id: str | None = None
    reasoning_effort: str | None = None
    output_schema: dict[str, Any] | None = None
    timestamp: str = field(default_factory=now_iso)
    # Keep P18 fields after timestamp so existing positional callers keep
    # interpreting their arguments exactly as before. ``config`` holds one
    # frozen closed dataclass — the common subset (``EffectiveRequestConfig``)
    # or a backend-specific subclass of it (a discriminated union via
    # inheritance per the plan; never ``dict[str, Any]``).
    config: EffectiveRequestConfig = field(default_factory=EffectiveRequestConfig)
    model_source: EvidenceSource | None = None
    provider_source: EvidenceSource | None = None
    session_source: EvidenceSource | None = None
    timestamp_source: Literal["host_observed", "native"] = "host_observed"


@dataclass
class TextEvent:
    """Agent text output.

    Attributes:
        text: The text emitted by the agent.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time
            via ``now_iso()`` (Pitfall 2 single-source-of-truth).
    """

    text: str
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ThinkingEvent:
    """Extended thinking / reasoning.

    Attributes:
        text: Reasoning content emitted by the agent.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    text: str
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ToolStartEvent:
    """Tool invocation started.

    Attributes:
        id: Tool call identifier (Claude block.id or Codex item.id /
            synthesized UUID).
        name: Tool function name.
        input: Tool arguments dict; may be empty but is never None.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    id: str
    name: str
    input: dict[str, Any]
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ToolResultEvent:
    """Tool invocation completed.

    Attributes:
        id: Tool call identifier matching the prior ToolStartEvent.id.
        output: Tool output as a string.
        is_error: True if the tool reported an error.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
        exit_code: Exit code as reported by the backend; None when the
            backend has no structured exit code (Claude/Pi).
        status: Backend-native status string (e.g. Codex's
            "completed"/"declined"); None when unavailable.
        duration_ms: Wall-clock duration in milliseconds; None when
            unavailable.
        cancelled: True if the backend reported the tool call as cancelled.
        truncated: True if the backend marked the output as truncated.
        All five are optional; they default to None/False for backends
        without structured metadata (Claude/Pi).
    """

    id: str
    output: str
    is_error: bool
    timestamp: str = field(default_factory=now_iso)
    # Keep these after timestamp so existing positional ToolResultEvent
    # callers continue to interpret their arguments up to is_error/timestamp
    # as-is; the metadata fields are optional and defaulted.
    exit_code: int | None = None
    status: str | None = None
    duration_ms: float | None = None
    cancelled: bool = False
    truncated: bool = False


@dataclass
class DiagnosticEvent:
    """Backend parser or transport coverage evidence for the active invocation.

    Diagnostics are recorder-only signals. The trajectory recorder applies the
    backend-neutral JSON normalization and redaction boundary before persisting
    any of these fields.
    """

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
    """Cost and usage information (end-of-call signal feeding FinalMetrics).

    ``provider_name`` is the exposed native provider identity;
    ``cache_creation_tokens`` is the cache-write subset of total input.
    ``model_usage`` contains selected native per-model totals, not extra billing.

    Attributes:
        cost_usd: Total cost in USD; None when unavailable. Codex synthesizes
            via the #61 price table (#194 reverses D-16); None only when the
            model is unknown to the table.
        input_tokens: Prompt tokens (None when unavailable).
        output_tokens: Completion tokens (None when unavailable).
        cached_tokens: Cache-read hit subset of input_tokens. input_tokens
            is the total input (backends fold cache read+creation into it);
            cached_tokens is the read subset, NOT added to input_tokens.
            None when unavailable. Default ``None`` keeps existing
            3-positional-arg call sites in ``backends/claude.py`` and
            ``backends/codex.py`` valid until Plans 03/04 update them.
        reasoning_tokens: Reasoning portion of output_tokens (subset, NOT
            additive — Codex's ``accounting.rs`` already counts these
            inside ``output_tokens``). Surfaces Codex's
            ``reasoning_output_tokens`` for cost attribution / perf
            observability (#192; openai/codex#26428 — count-only, no
            reasoning *content* is emitted). ``None`` on Claude (reasoning
            arrives via ThinkingEvent, a separate path) and when Codex
            omits the field.
        model_name: Real SDK model id observed during this call (e.g.
            ``claude-opus-4-5-20250901``). ``None`` when unavailable; the
            recorder uses it to upgrade a generic backend label
            (``"claude"``, ``"codex"``, ``"osprey"``) to the actual model id.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
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
    # P18 Task 1 (additive): closed measurement provenance, generation
    # correlation and cost provenance. ``None`` keeps every existing call
    # site valid and means "not assessed", never a claim.
    measurement_source: MeasurementSource | None = None
    generation_id: str | None = None
    cost_source: CostSource | None = None


@dataclass
class MetricsEvent:
    """Per-step LLM token/cost usage.

    ``usage_scope`` distinguishes per-message usage from an invocation
    aggregate (Codex); cache creation is a subset of total prompt tokens.
    ``duration_ms`` and ``started_at`` are native backend timing, when exposed.

    Emitted once per AssistantMessage by the Claude backend (keyed via
    ``AssistantMessage.message_id``), and once per ``turn.completed`` by
    the Codex backend (with empty ``message_id`` since Codex has no
    per-message id). The recorder uses ``message_id`` to attach Metrics
    to the correct agent Step (D-04, MAP-06).

    Attributes:
        message_id: Identifier matching the AssistantMessage that owns
            this metric. Empty string for Codex (D-16).
        prompt_tokens: Prompt tokens for this turn. REQUIRED per EVNT-02
            (int, not Optional) — every AssistantMessage / turn.completed
            carries it. Backends read the SDK key (Claude
            ``usage["input_tokens"]``, Codex ``usage["input_tokens"]``)
            and rename at the boundary.
        completion_tokens: Completion tokens for this turn. REQUIRED per
            EVNT-02 (int, not Optional). Backends read the SDK key
            (Claude ``usage["output_tokens"]``, Codex
            ``usage["output_tokens"]``) and rename at the boundary.
        cached_tokens: Cache-read hit subset of ``prompt_tokens``
            (None when unavailable). ``prompt_tokens`` is the total input
            (backends fold cache read+creation into it); cached_tokens is
            the read subset, NOT additive to ``prompt_tokens``.
        cost_usd: Per-turn cost in USD (None when unavailable). Codex
            synthesizes via the #61 price table (#194 reverses D-16); None
            only when the model is unknown to the table.
        reasoning_tokens: Reasoning portion of ``completion_tokens``
            (subset, NOT additive — Codex's ``accounting.rs`` already
            counts these inside ``output_tokens``). Surfaces Codex's
            ``reasoning_output_tokens`` for cost attribution / perf
            observability (#192; openai/codex#26428 — count-only, no
            reasoning *content* is emitted). ``None`` on Claude (reasoning
            arrives via ThinkingEvent, a separate path) and when Codex
            omits the field.
        model_name: Real SDK model id observed for this turn (e.g.
            ``claude-opus-4-5-20250901``). ``None`` when unavailable;
            recorder uses it to upgrade a generic backend label
            (``"claude"``, ``"codex"``, ``"osprey"``) to the actual model id.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
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
    # P18 Task 1 (additive): closed measurement provenance and generation
    # correlation. ``None`` keeps every existing call site valid and means
    # "not assessed", never a claim.
    measurement_source: MeasurementSource | None = None
    generation_id: str | None = None


@dataclass
class TurnEndEvent:
    """Assistant-turn boundary signal.

    Emitted by each backend at the end of an assistant "turn" — for
    Claude, once per ``AssistantMessage``; for Codex, once per
    ``item.completed`` of type ``agent_message``. The trajectory
    recorder uses this to close its open Step so multi-turn invocations
    are recorded as one Step per turn (instead of collapsing into a
    single Step at invocation finish).

    Attributes:
        message_id: Correlator matching the message that ended this turn
            (e.g. Claude's ``AssistantMessage.message_id``). Empty string
            when the backend cannot supply one (Codex has no per-message
            id surface — D-04 correlator unused for Codex).
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
        P18 Task 1 additions: where a backend exposes a per-turn finish
        reason/model/provider on the boundary itself, it is carried here
        with provenance; backends without such an exposure leave them
        ``None`` (Pi fills ``model_name``/``provider_name`` from its
        ``turn_end.message``; Claude leaves them unset and keeps its
        existing ``ResultEvent``-level metadata). ``timestamp_source``
        distinguishes a native protocol timestamp (``"native"``) from the
        host yield time (``"host_observed"``).
    """

    message_id: str = ""
    timestamp: str = field(default_factory=now_iso)
    # P18 additions sit after the existing fields; every existing
    # positional caller (message_id, timestamp) is unaffected.
    finish_reason: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    message_id_source: EvidenceSource | None = None
    model_source: EvidenceSource | None = None
    provider_source: EvidenceSource | None = None
    timestamp_source: Literal["host_observed", "native"] = "host_observed"


# --- Generation lifecycle (P18 Task 1) --------------------------------------
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
    """Ordered provider-choice tool-call part with exact call identity.

    ``arguments`` is validated as a closed recursive JSON value at
    construction; a non-JSON payload is rejected (never coerced).
    """

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
    """Generation started: host receipt of a provider generation boundary.

    ``generation_id`` is a host invocation-local correlation UUID — never a
    provider identity. ``observed_at_unix_ns`` is the host receipt time only;
    it is never relabeled as the provider request start. ``boundary_complete``
    is False for a partial/missing boundary (e.g. an end without a start or a
    synthetic error start) so downstream consumers can distinguish a complete
    start/end pair from an explicit incomplete one.
    """

    generation_id: str
    observed_at_unix_ns: int
    boundary_complete: bool = True
    timestamp: str = field(default_factory=now_iso)


@dataclass
class GenerationEndEvent:
    """Generation ended: sealed provider choice plus lifecycle evidence.

    ``native_started_at_unix_ms`` is the strict validated native Unix-ms
    start from the completed provider message (``None`` when absent or
    malformed — never clamped or guessed). ``ended_at_unix_ns`` is the host
    receipt of the end boundary. ``end_source`` records which boundary
    supplied the end: the host-observed ``message_end`` receipt, a native
    end timestamp, or an explicit fallback. ``choice_parts`` is the ordered
    provider choice sealed at this boundary; later tool execution links by
    call ID but never authors or duplicates these parts.
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
    """Terminal data, including native identity, finish reason and duration.

    A failed backend exposes known terminal data before raising; receiving
    this event is not evidence that the enclosing invocation succeeded.
    ``session_id`` is independent of whether a continuation token is requested.

    Attributes:
        structured_output: Structured result as emitted by the backend,
            schema-validated (or salvage-checked) at the run_agent return
            path when the caller opts in (``validate_structured_output``
            True), or None. Callers that pass
            ``validate_structured_output=False`` re-validate downstream.
        continuation: Optional continuation token for multi-turn flows.
        model_name: Real SDK model id observed for this invocation. Backends
            should populate this when the model is only available from a
            session-level terminal event rather than per-turn usage.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    structured_output: Any | None
    continuation: ContinuationToken | None
    timestamp: str = field(default_factory=now_iso)
    # Keep this after timestamp so existing three-positional-argument
    # ResultEvent callers continue to interpret their third argument as the
    # timestamp.
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
    """Protocol for agent backends.

    Each backend yields a stream of AgentEvent instances from execute().

    Optional extension: backends may expose ``fanout_concurrency: int`` as a
    scheduling hint for orchestrator-managed parallel calls. Callers combine
    the hint with their workflow ceiling via
    :func:`effective_fanout_concurrency`; absent hints fall back to four.

    Optional extension: backends may expose ``concise_fix_prompts: bool`` to
    request verbosity-suppressing fix-phase prompts (set True for pi/GLM, which
    produces verbose reasoning). When absent, the caller falls back to False via
    ``getattr(backend, "concise_fix_prompts", False)``.

    Optional extension: backends may expose ``read_only_disposable_clone: bool``
    to indicate the backend runs against a disposable read-only checkout (Codex).
    Such backends get over-budget diffs inlined truncated to the inline budget
    and exploration summaries inlined instead of file pointers, and their
    correction-loop rebuilds are framed with the untrusted-content boundary.
    When absent, the caller falls back to False via
    ``getattr(backend, "read_only_disposable_clone", False)``.

    Optional extension: backends may expose ``audit_root_isolation`` and
    ``audit_root`` when they mediate every filesystem-capable tool against one
    exact improve audit snapshot. This tool-layer capability is separate from
    ``read_only_disposable_clone`` and does not claim an OS/container sandbox.
    Callers must compare the capability to :data:`AUDIT_ROOT_ISOLATION` and
    the bound root by canonical identity; missing or different values fail
    closed.

    Optional extension: backends may expose ``reasoning_effort``, the per-phase
    reasoning level resolved by ``daydream.runner._resolved_reasoning_effort``
    and applied through the driver's native knob (Claude
    ``ClaudeAgentOptions.effort``, Codex ``-c model_reasoning_effort=``, Pi
    ``--thinking``). All three shipped backends set it; it stays off the
    protocol because each narrows it to its own driver's literal vocabulary.
    Read it via ``getattr(backend, "reasoning_effort", None)``. It is set at
    construction rather than per ``execute`` call because a backend instance is
    already cached per resolved ``(kind, model, reasoning_effort, audit_root)``
    tuple, so one instance serves exactly one effort level and audit boundary.
    ``None`` means no source supplied one and the driver applies its own ambient
    default.
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
        """Yield AgentEvents for *prompt*.

        Args:
            read_only: When True, the backend enforces a non-mutating tool
                profile at the tool layer (Claude via a PreToolUse guard hook)
                so the agent can inspect history but cannot write/edit/delete
                or mutate the working tree. The Codex backend combines its
                ``--sandbox read-only`` with a disposable standalone clone
                whenever *cwd* is a Git worktree root: the subprocess runs in
                a clone that mirrors HEAD, the staged index, and tracked /
                nonignored untracked files with no source remote, and the
                clone is deleted after the subprocess exits. Codex's sandbox
                restricts filesystem writes but not git index/object-store
                operations, so the clone is what makes a commit update only
                the disposable clone's refs and index — never the caller's
                HEAD, staged index, refs, or remotes, which are unreachable
                via any path the subprocess is given (its argv, stdin, env,
                and cwd); a model that independently discovers the source
                path could still write to its refs. Any other *cwd* — one
                outside a Git worktree, or inside a worktree but not at its
                root — uses the read-only sandbox in place. Callers select
                this flag explicitly per call site: the diagnostic subagents
                (setup-investigator, recommendation-verifier), the failure
                summarizer, and the exploration and repository
                reconnaissance specialists (pre_scan, repo_scan, improve
                recon) pass True, while mutating phases keep the False
                default.
            persist_session: When False, request an invocation that leaves no
                resumable backend session. Backends without persisted sessions
                accept and ignore this option.
        """
        ...

    async def cancel(self) -> None:
        """Cancel every active invocation on this backend.

        This backend-wide operation is reserved for process shutdown. Callers
        ending one invocation must close the corresponding AgentEventStream.
        """
        ...


def resolve_fanout_concurrency(env_var: str, default: int) -> int:
    """Read a backend's fan-out hint from *env_var*, falling back to *default*.

    The right value is a property of the endpoint serving the turns, not of the
    backend, which is why it is an environment override rather than a constant.
    A non-integer or non-positive value warns and falls back rather than failing
    the run: a malformed knob should not cost a review.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s is not a valid integer; using default %d", env_var, default)
        return default
    if value <= 0:
        logger.warning("%s must be positive; using default %d", env_var, default)
        return default
    return value


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
    """Create a backend by name.

    Args:
        name: Backend name ("claude", "codex", "pi", or "osprey").
        model: Optional model override. Claude and Codex apply their built-in
            defaults here. Pi receives ``None`` unchanged so its own configured
            default can win before Pi's GLM fallback is selected.
        cwd: Target workspace used to resolve Pi's configured default model.
        reasoning_effort: Optional reasoning-effort override (one of
            ``daydream.config.REASONING_EFFORT_LEVELS``). Every backend applies
            it through its own native knob: Claude via
            ``ClaudeAgentOptions.effort``, Codex via
            ``-c model_reasoning_effort=...``, Pi via ``--thinking``.
        audit_root: Exact standalone improve snapshot to which every exposed
            filesystem-capable tool must be confined. Currently supported only
            by Claude.
        audit_outward_symlinks: Lexical paths of snapshot symlinks whose
            targets resolve outside ``audit_root``.

    Returns:
        A Backend instance whose ``.model`` attribute is a non-empty string.

    Raises:
        ValueError: If the backend name is unknown.
    """
    from daydream.config import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL

    if name == "claude":
        from daydream.backends.claude import ClaudeBackend

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
        from daydream.backends.pi import PiBackend

        return PiBackend(
            model=model,
            cwd=cwd,
            reasoning_effort=reasoning_effort,
            execution_input=execution_input,
        )
    if name == "osprey":
        if execution_input is not None:
            raise ValueError("explicit BackendExecutionInput is not supported for osprey")
        from daydream.backends.osprey import OspreyBackend

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
    "AssistantChoicePart",
    "AuditIsolationError",
    "AuditIsolationReason",
    "Backend",
    "BackendExecutionInput",
    "ClaudeBackend",
    "ClaudeRequestConfig",
    "CodexRequestConfig",
    "ContinuationToken",
    "CostEvent",
    "DiagnosticEvent",
    "EvidenceDiagnostic",
    "EvidenceSource",
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
