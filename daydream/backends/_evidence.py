"""Typed request controls and bounded runtime evidence admission."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from daydream.redaction import redact_structured_text

# Unsafe runtime evidence is omitted with fixed diagnostics, never truncated or echoed.
EvidenceSource = Literal["configured", "host_generated", "native"]

# Absent provenance means unassessed, never an inferred source.
MeasurementSource = Literal["message_end", "turn_end", "terminal", "session", "synthesized"]

CostSource = Literal["reported", "estimated"]

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

# Controls, bidi formatting, and Unicode line/paragraph separators are unsafe identities.
_UNSAFE_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
# Absolute drive paths are private; relative model namespaces remain valid.
_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class EvidenceDiagnostic:
    """One fixed-shape admission diagnostic (code + bounded detail)."""

    code: str
    detail: str = ""


def _has_unicode_controls(value: str) -> bool:
    """Reject C0/C1 controls and bidi/line-separator format characters."""
    return any(
        unicodedata.category(ch) in _UNSAFE_UNICODE_CATEGORIES
        for ch in value
    )


def _is_private_path(value: str) -> bool:
    """Identify absolute or home-anchored paths; relative model namespaces remain valid."""
    return bool(value) and (
        value.startswith(("/", "\\\\", "~", "home/", "Users/")) or bool(_WINDOWS_DRIVE_PREFIX.match(value))
    )


def _admit_identity_label(
    value: Any,
    *,
    max_chars: int,
    context: str,
) -> tuple[str | None, EvidenceDiagnostic | None]:
    """Admit an unchanged, bounded nonempty identity without controls, credentials, or private paths.
    Rejected values return fixed diagnostics; identities are never truncated.
    """
    if not isinstance(value, str) or not value:
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_UNSAFE_CHARS, context)
    if len(value) > max_chars:
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_TOO_LONG, context)
    if _has_unicode_controls(value):
        return None, EvidenceDiagnostic(_DIAG_IDENTITY_UNSAFE_CHARS, context)
    if redact_structured_text(value) != value:
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
    """Admit exact nonnegative integers whose nanosecond conversion fits int64.
    None means absence; invalid values yield diagnostics without coercion or clamping.
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
    """Copy JSON values; reject nonfinite floats, unsafe keys, non-JSON types, and nesting beyond 64."""
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
    """Reject malformed built-in controls at construction; adapters omit unsafe runtime evidence."""

    def __post_init__(self) -> None:
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
    """Effective controls: None means absent. Construction rejects invalid types, ranges, and modes."""

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


@dataclass(frozen=True)
class OspreyRequestConfig(EffectiveRequestConfig):
    """Explicit Osprey argv facts; labels and inferred configuration stay absent."""

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
    """Compile validators from built-in field types; custom subclasses use their nearest built-in config.
    Telemetry separately rejects unknown concrete config types.
    """
    hints = get_type_hints(schema)
    scalar_checks: dict[Any, _ConfigCheck] = {
        bool: _require_bool,
        int: _require_nonnegative_int,
        float: _require_finite_float,
    }
    checks: list[tuple[str, _ConfigCheck]] = []
    for config_field in fields(schema):
        (value_type,) = (part for part in get_args(hints[config_field.name]) if part is not type(None))
        check = (
            partial(_require_literal, allowed=get_args(value_type))
            if get_origin(value_type) is Literal
            else scalar_checks[value_type]
        )
        checks.append((config_field.name, check))
    return tuple(checks)


_CONFIG_CHECKS = {
    schema: _config_checks(schema)
    for schema in (
        EffectiveRequestConfig,
        ClaudeRequestConfig,
        CodexRequestConfig,
        PiRequestConfig,
        OspreyRequestConfig,
    )
}
