"""Request configuration, native identity and generation admission boundaries."""
from __future__ import annotations

from typing import Any

import pytest

from daydream.backends import (
    ClaudeRequestConfig,
    CodexRequestConfig,
    EffectiveRequestConfig,
    JsonValue,
    OspreyRequestConfig,
    ToolCallChoicePart,
    _admit_identity_label,
    _admit_native_unix_ms,
    unix_ms_to_ns,
)
from daydream.observability.spans import _admit_observed_identity_list

# --- P18 Task 1: closed typed Effective Configuration Admission Contract ----

def test_admission_contract_construction_rejects_wrong_types() -> None:
    rejected: list[tuple[type[Any], dict[str, Any], str]] = [
        (EffectiveRequestConfig, {"temperature": True}, "temperature"),
        (EffectiveRequestConfig, {"temperature": float("nan")}, "temperature"),
        (EffectiveRequestConfig, {"max_turns": True}, "max_turns"),
        (EffectiveRequestConfig, {"max_turns": -1}, "max_turns"),
        (EffectiveRequestConfig, {"max_turns": 2**63}, "max_turns"),
        (EffectiveRequestConfig, {"read_only": 1}, "read_only"),
        (EffectiveRequestConfig, {"continuation_mode": "restart"}, "continuation_mode"),
        (ClaudeRequestConfig, {"permission_mode": "default"}, "permission_mode"),
        (CodexRequestConfig, {"sandbox_mode": "workspace-write"}, "sandbox_mode"),
        (OspreyRequestConfig, {"approval_mode": "on-request"}, "approval_mode"),
    ]
    for config_cls, kwargs, field_name in rejected:
        try:
            with pytest.raises(ValueError, match=field_name):
                config_cls(**kwargs)
        except BaseException as exc:
            exc.add_note(f"case: {config_cls.__name__}({kwargs!r})")
            raise

def test_identity_label_admission_rejects_unsafe_and_allows_namespace() -> None:
    for safe in ("opus", "claude-opus-4-5-20250901", "nous/deepseek-v4", "openai/gpt-5.2",
                 "provider//model", "glm-4.6"):
        admitted, diagnostic = _admit_identity_label(safe, max_chars=256, context="model")
        assert admitted == safe and diagnostic is None, safe
    for unsafe in (
        "/Users/ka/private/model",
        "C:\\repo\\model",
        "~/secret", "home/ka/model", "Users/ka/model",
        "a\x00b",
        "a\nb",
        "a\tb",
        "abc\u202ered",
        "abc\u200blad",
        "line\u2028sep",
        "AKIAIOSFODNN7EXAMPLE", "password=hunter2", "x-API-key: v", "m" * 257, "", None, 5, 1.5, b"model",
    ):
        admitted, diagnostic = _admit_identity_label(unsafe, max_chars=256, context="model")
        assert admitted is None and diagnostic is not None, repr(unsafe)
        assert diagnostic.code.startswith("config_identity_") or diagnostic.code in (
            "config_bool_type_rejected", "config_float_type_rejected", "config_int_type_rejected",
        )

def test_native_unix_ms_validation_bounds_and_conversion() -> None:
    ms, diagnostic = _admit_native_unix_ms(1788690314289)
    assert ms == 1788690314289 and diagnostic is None
    assert unix_ms_to_ns(1788690314289) == 1788690314289000000
    ms, diagnostic = _admit_native_unix_ms(0)
    assert ms == 0 and diagnostic is None
    bound = (2**63 - 1) // 1_000_000
    ms, diagnostic = _admit_native_unix_ms(bound)
    assert ms == bound and diagnostic is None
    ms, diagnostic = _admit_native_unix_ms(bound + 1)
    assert ms is None and diagnostic is not None
    for bad in (True, False, 1.5, "123", -1, -(10**6)):
        ms, diagnostic = _admit_native_unix_ms(bad)
        assert ms is None and diagnostic is not None, repr(bad)
    # Absence is not malformation.
    ms, diagnostic = _admit_native_unix_ms(None)
    assert ms is None and diagnostic is None

def test_observed_identity_lists_whole_overflow_omission() -> None:
    admitted, diagnostic = _admit_observed_identity_list(
        [f"model-{i}" for i in range(16)], max_chars=256, context="models",
    )
    assert admitted is not None and len(admitted) == 16
    assert diagnostic is None
    admitted, diagnostic = _admit_observed_identity_list(
        [f"model-{i}" for i in range(17)], max_chars=256, context="models",
    )
    assert admitted is None
    assert diagnostic is not None
    code, detail = diagnostic.split(":", 1)
    assert code == "config_list_overflow"
    assert detail == "17"  # total distinct count only
    # One unsafe member poisons the whole list (no partial lists).
    admitted, diagnostic = _admit_observed_identity_list(
        ["ok/model", "/Users/ka/secret"], max_chars=256, context="models",
    )
    assert admitted is None and diagnostic is not None
    assert diagnostic.split(":", 1)[0] == "config_list_member_unsafe"

def test_tool_call_choice_part_json_arguments_are_schema_admitted() -> None:
    part = ToolCallChoicePart(call_id="t1", name="read", arguments={"path": "/x", "n": 3, "ok": True, "f": 0.5})
    assert part.arguments == {"path": "/x", "n": 3, "ok": True, "f": 0.5}
    assert part.kind == "tool_call"
    nested = ToolCallChoicePart(call_id="t2", name="edit", arguments={"a": [1, {"b": None}]})
    nested_arguments: JsonValue = nested.arguments
    assert nested_arguments == {"a": [1, {"b": None}]}
    with pytest.raises(ValueError, match="call_id"):
        ToolCallChoicePart(call_id="", name="read", arguments={})
    with pytest.raises(ValueError, match="name"):
        ToolCallChoicePart(call_id="t3", name="not a tool name!", arguments={})
    with pytest.raises(ValueError, match="arguments"):
        bad_object_arguments: dict[str, JsonValue] = {"x": object()}  # type: ignore[dict-item]
        ToolCallChoicePart(call_id="t4", name="read", arguments=bad_object_arguments)
    with pytest.raises(ValueError, match="arguments"):
        ToolCallChoicePart(call_id="t5", name="read", arguments=float("nan"))
    with pytest.raises(ValueError, match="name") as private_name:
        ToolCallChoicePart(call_id="t6", name="/Users/ka/bin/tool", arguments={})
    assert "/Users/ka" not in str(private_name.value)
