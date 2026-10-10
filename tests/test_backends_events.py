"""Request configuration, native identity and generation admission boundaries."""
from __future__ import annotations

from typing import Any

import pytest

from daydream import backends
from daydream.backends import (
    AgentEvent,
    ClaudeRequestConfig,
    CodexRequestConfig,
    DiagnosticEvent,
    EffectiveRequestConfig,
    JsonValue,
    OspreyRequestConfig,
    PiRequestConfig,
    ToolCallChoicePart,
    _admit_identity_label,
    _admit_native_unix_ms,
    unix_ms_to_ns,
)
from daydream.observability.spans import _admit_observed_identity_list


def test_diagnostic_event_has_fresh_metadata_and_is_exported() -> None:
    first = DiagnosticEvent(code="parser_gap", message="first")
    second = DiagnosticEvent(code="parser_gap", message="second")
    first.metadata["count"] = 1
    assert second.metadata == {}
    assert isinstance(first, AgentEvent)
    assert "DiagnosticEvent" in backends.__all__


# --- P18 Task 1: closed typed Effective Configuration Admission Contract ----

def _rejected_configs() -> list[tuple[type[Any], dict[str, Any], str]]:
    """(config class, kwargs, expected diagnostic substring) tuples."""
    return [
        (EffectiveRequestConfig, {"temperature": True}, "temperature"),
        (EffectiveRequestConfig, {"temperature": float("nan")}, "temperature"),
        (EffectiveRequestConfig, {"temperature": float("inf")}, "temperature"),
        (EffectiveRequestConfig, {"temperature": "0.7"}, "temperature"),
        (EffectiveRequestConfig, {"max_turns": True}, "max_turns"),
        (EffectiveRequestConfig, {"max_turns": -1}, "max_turns"),
        (EffectiveRequestConfig, {"max_turns": 2.0}, "max_turns"),
        (EffectiveRequestConfig, {"max_turns": 2**63}, "max_turns"),
        (EffectiveRequestConfig, {"read_only": 1}, "read_only"),
        (EffectiveRequestConfig, {"persist_session": "yes"}, "persist_session"),
        (EffectiveRequestConfig, {"continuation_mode": "restart"}, "continuation_mode"),
        (EffectiveRequestConfig, {"model_mode": "Multi"}, "model_mode"),
        (EffectiveRequestConfig, {"model_mode": "multi"}, "model_mode"),
        (ClaudeRequestConfig, {"permission_mode": "default"}, "permission_mode"),
        (ClaudeRequestConfig, {"allowed_tools_count": -3}, "allowed_tools_count"),
        (ClaudeRequestConfig, {"hooks_enabled": 0}, "hooks_enabled"),
        (CodexRequestConfig, {"sandbox_mode": "workspace-write"}, "sandbox_mode"),
        (CodexRequestConfig, {"experimental_json": 1}, "experimental_json"),
        (PiRequestConfig, {"selected_tools_count": -1}, "selected_tools_count"),
        (PiRequestConfig, {"schema_emulated": 0}, "schema_emulated"),
        (OspreyRequestConfig, {"approval_mode": "on-request"}, "approval_mode"),
        (OspreyRequestConfig, {"approval_mode": "unless-trusted"}, "approval_mode"),
        (OspreyRequestConfig, {"max_turns": True}, "max_turns"), (OspreyRequestConfig, {"llm_rpm": -5}, "llm_rpm"),
        (OspreyRequestConfig, {"compress_min_bytes": 2.5}, "compress_min_bytes"),
        (OspreyRequestConfig, {"vars_count": 2**63}, "vars_count"),
    ]

@pytest.mark.parametrize(("config_cls", "kwargs", "field_name"), _rejected_configs())
def test_admission_contract_construction_rejects_wrong_types(
    config_cls: type[Any], kwargs: dict[str, Any], field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        config_cls(**kwargs)

@pytest.mark.parametrize(
    ("config_cls", "kwargs"),
    [
        pytest.param(
            EffectiveRequestConfig,
            {
                "temperature": 0.0, "max_turns": 2**63 - 1, "read_only": False, "persist_session": True,
                "continuation_mode": "resume", "model_mode": "multi_or_dynamic",
            }, id="effective-boundaries-and-zero",
        ), pytest.param(EffectiveRequestConfig, {"temperature": -0.5}, id="negative-temperature-is-a-finite-float"),
        pytest.param(
            ClaudeRequestConfig,
            {
                "permission_mode": "bypassPermissions", "allowed_tools_count": 0, "allowed_tools_present": False,
                "buffer_limit_bytes": 0,
            }, id="claude-zeros-retained",
        ),
        pytest.param(
            CodexRequestConfig,
            {"sandbox_mode": "read-only", "native_output_schema": True, "read_only_isolation": True},
            id="codex-closed-modes",
        ),
        pytest.param(
            PiRequestConfig,
            {"selected_tools_count": 4, "selected_tools_present": True, "no_skills": True, "schema_emulated": True},
            id="pi-effective-controls",
        ),
        pytest.param(
            OspreyRequestConfig,
            {
                "approval_mode": "deny-untrusted", "sandbox": True, "max_turns": 0, "empty_completion_threshold": 0,
                "observation_admission_bytes": 2 * 1024 * 1024,
            }, id="osprey-zeros-and-closed-mode",
        ),
    ],
)
def test_admission_contract_accepts_exact_zero_and_boundaries(config_cls: type[Any], kwargs: dict[str, Any]) -> None:
    config = config_cls(**kwargs)
    for name, value in kwargs.items():
        assert getattr(config, name) == value

def test_admission_contract_defaults_mean_absent_not_effective() -> None:
    config = EffectiveRequestConfig()
    assert config.temperature is None
    assert config.max_turns is None
    assert config.read_only is None
    assert config.persist_session is None
    assert config.continuation_mode is None
    assert config.model_mode is None


def test_arbitrary_persona_and_toolset_labels_have_no_field() -> None:
    config = OspreyRequestConfig(persona_present=True, toolset_present=False)
    assert config.persona_present is True
    assert config.toolset_present is False
    assert not hasattr(config, "persona")
    assert not hasattr(config, "toolset")

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

def test_provider_label_bound_is_128() -> None:
    admitted, diagnostic = _admit_identity_label("p" * 128, max_chars=128, context="provider")
    assert admitted == "p" * 128 and diagnostic is None
    admitted, diagnostic = _admit_identity_label("p" * 129, max_chars=128, context="provider")
    assert admitted is None
    assert diagnostic is not None and diagnostic.code == "config_identity_too_long"

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

def test_tool_call_choice_part_never_echoes_private_tool_names() -> None:
    with pytest.raises(ValueError, match="name"):
        ToolCallChoicePart(call_id="t6", name="/Users/ka/bin/tool", arguments={})
