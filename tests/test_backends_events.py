"""Tests for the ``AgentEvent`` dataclasses in ``daydream/backends/__init__.py``.

Field values, nullable defaults, and the union/export surface all live here;
``tests/test_backends_init.py`` covers the ``Backend`` protocol and the
``create_backend`` factory. The two files previously asserted the same event
field defaults twice.

Covers Plan 02-02 of phase 02-recorder-core-event-enrichment-mapping:

- Every event dataclass carries a ``timestamp: str`` field defaulted via
  ``now_iso()`` (Pitfall 2 single source of truth).
- The ``MetricsEvent`` dataclass exists and uses the EVNT-02 verbatim
  field names (``prompt_tokens``, ``completion_tokens``, NOT
  ``input_tokens`` / ``output_tokens`` — those are the SDK boundary keys
  that backends rename when emitting MetricsEvent).
- ``CostEvent`` carries the ``cached_tokens`` field (default ``None``
  for backward compatibility with the existing 3-positional-arg call sites
  in ``backends/claude.py:124`` and ``backends/codex.py:310``).
- ``MetricsEvent`` is part of the ``AgentEvent`` TypeAlias union and is
  exported in ``__all__``.
"""

from __future__ import annotations

from typing import Any

import pytest

from daydream.backends import (
    AgentEvent,
    ClaudeRequestConfig,
    CodexRequestConfig,
    ContinuationToken,
    CostEvent,
    DiagnosticEvent,
    EffectiveRequestConfig,
    GenerationEndEvent,
    GenerationStartEvent,
    JsonValue,
    MetricsEvent,
    OspreyRequestConfig,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolCallChoicePart,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)


def _assert_fields(event: Any, expected: dict[str, Any]) -> None:
    """Assert each expected field, by identity for ``None`` / bools."""
    for name, want in expected.items():
        got = getattr(event, name)
        if want is None or isinstance(want, bool):
            assert got is want, name
        else:
            assert got == want, name


@pytest.mark.parametrize(
    ("cls", "kwargs", "expected"),
    [
        pytest.param(TextEvent, {"text": "hello"}, {"text": "hello"}, id="text-event"),
        pytest.param(
            DiagnosticEvent,
            {"code": "parser_gap", "message": "unknown item", "metadata": {"count": 2}},
            {"code": "parser_gap", "message": "unknown item", "metadata": {"count": 2}},
            id="diagnostic-event",
        ),
        pytest.param(ThinkingEvent, {"text": "reasoning..."}, {"text": "reasoning..."}, id="thinking-event"),
        pytest.param(
            ToolStartEvent,
            {"id": "t1", "name": "Bash", "input": {"command": "ls"}},
            {"id": "t1", "name": "Bash", "input": {"command": "ls"}},
            id="tool-start-event",
        ),
        pytest.param(
            ToolResultEvent,
            {"id": "t1", "output": "file.py", "is_error": False},
            {"id": "t1", "output": "file.py", "is_error": False},
            id="tool-result-event",
        ),
        pytest.param(
            CostEvent,
            {"cost_usd": 0.01, "input_tokens": 100, "output_tokens": 50},
            {"cost_usd": 0.01, "input_tokens": 100, "output_tokens": 50, "cached_tokens": None},
            id="cost-event",
        ),
        pytest.param(
            CostEvent,
            {"cost_usd": None, "input_tokens": None, "output_tokens": None},
            {"cost_usd": None, "input_tokens": None, "output_tokens": None},
            id="cost-event-nullable",
        ),
        pytest.param(
            CostEvent,
            {"cost_usd": 0.5, "input_tokens": 10, "output_tokens": 20, "cached_tokens": 3},
            {"cached_tokens": 3},
            id="cost-event-cached-tokens",
        ),
        # Backward compat: the existing 3-arg call sites still work; cached_tokens defaults to None.
        pytest.param(
            CostEvent,
            {"cost_usd": 0.5, "input_tokens": 10, "output_tokens": 20},
            {"cached_tokens": None},
            id="cost-event-cached-tokens-default-none",
        ),
        pytest.param(
            ResultEvent,
            {"structured_output": None, "continuation": None},
            {"structured_output": None, "continuation": None},
            id="result-event-nullable",
        ),
        pytest.param(
            MetricsEvent,
            {
                "message_id": "msg_01",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 5,
                "cost_usd": 0.001,
            },
            {
                "message_id": "msg_01",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 5,
                "cost_usd": 0.001,
            },
            id="metrics-event",
        ),
    ],
)
def test_event_field_values(cls: type, kwargs: dict[str, Any], expected: dict[str, Any]) -> None:
    """Each event dataclass exposes the constructed values on the documented field names."""
    _assert_fields(cls(**kwargs), expected)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        pytest.param(TextEvent, {"text": "hi"}, id="text-event"),
        pytest.param(
            DiagnosticEvent,
            {"code": "parser_gap", "message": "unknown item"},
            id="diagnostic-event",
        ),
        pytest.param(ThinkingEvent, {"text": "reasoning"}, id="thinking-event"),
        pytest.param(
            ToolStartEvent, {"id": "abc", "name": "Read", "input": {"file_path": "/tmp/a"}}, id="tool-start-event"
        ),
        pytest.param(ToolResultEvent, {"id": "abc", "output": "ok", "is_error": False}, id="tool-result-event"),
        pytest.param(
            CostEvent,
            {"cost_usd": 0.5, "input_tokens": 10, "output_tokens": 20, "cached_tokens": 3},
            id="cost-event",
        ),
        pytest.param(
            MetricsEvent,
            {
                "message_id": "msg_01",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "cached_tokens": 5,
                "cost_usd": 0.001,
            },
            id="metrics-event",
        ),
        pytest.param(ResultEvent, {"structured_output": None, "continuation": None}, id="result-event"),
        pytest.param(TurnEndEvent, {}, id="turn-end-event"),
    ],
)
def test_event_has_default_z_timestamp(cls: type, kwargs: dict[str, Any]) -> None:
    """Every member of the AgentEvent union defaults ``timestamp`` to a Z-suffixed stamp."""
    event = cls(**kwargs)
    assert isinstance(event.timestamp, str)
    assert event.timestamp.endswith("Z"), f"timestamp must end with Z: {event.timestamp!r}"


def test_result_event_carries_the_continuation_token() -> None:
    """ResultEvent holds the exact ContinuationToken instance it was given."""
    token = ContinuationToken(backend="codex", data={})
    event = ResultEvent(structured_output={"key": "val"}, continuation=token)
    assert event.structured_output == {"key": "val"}
    assert event.continuation is token


def test_metrics_event_is_accepted_by_agent_event_union() -> None:
    event = MetricsEvent(
        message_id="msg_01",
        prompt_tokens=10,
        completion_tokens=20,
        cached_tokens=5,
        cost_usd=0.001,
    )
    assert isinstance(event, AgentEvent)


def test_metrics_event_in_all_export() -> None:
    from daydream import backends

    assert "MetricsEvent" in backends.__all__


def test_turn_end_event_is_in_agent_event_union() -> None:
    """TurnEndEvent is a recognized AgentEvent so trajectory.py can dispatch."""
    ev = TurnEndEvent()
    ev2 = TurnEndEvent(message_id="msg_abc123")
    assert ev2.message_id == "msg_abc123"
    assert isinstance(ev.timestamp, str) and ev.timestamp.endswith("Z")
    # Runtime confirmation that TurnEndEvent is part of the AgentEvent union.
    assert isinstance(ev, AgentEvent)


def test_diagnostic_event_has_fresh_metadata_and_is_exported() -> None:
    first = DiagnosticEvent(code="parser_gap", message="first")
    second = DiagnosticEvent(code="parser_gap", message="second")
    first.metadata["count"] = 1

    assert second.metadata == {}
    assert isinstance(first, AgentEvent)

    from daydream import backends

    assert "DiagnosticEvent" in backends.__all__


def test_tool_result_event_status_fields_default_to_none() -> None:
    ev = ToolResultEvent(id="t1", output="ok", is_error=False)
    assert ev.exit_code is None
    assert ev.status is None
    assert ev.duration_ms is None
    assert ev.cancelled is False
    assert ev.truncated is False


def test_tool_result_event_accepts_status_metadata() -> None:
    ev = ToolResultEvent(id="t2", output="boom", is_error=True, exit_code=128, status="completed")
    assert ev.exit_code == 128
    assert ev.status == "completed"


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
        (OspreyRequestConfig, {"max_turns": True}, "max_turns"),
        (OspreyRequestConfig, {"llm_rpm": -5}, "llm_rpm"),
        (OspreyRequestConfig, {"compress_min_bytes": 2.5}, "compress_min_bytes"),
        (OspreyRequestConfig, {"vars_count": 2**63}, "vars_count"),
    ]


@pytest.mark.parametrize(("config_cls", "kwargs", "field_name"), _rejected_configs())
def test_admission_contract_construction_rejects_wrong_types(
    config_cls: type[Any],
    kwargs: dict[str, Any],
    field_name: str,
) -> None:
    """Wrong booleans-as-ints, NaN, negatives, and invalid enums fail construction."""
    with pytest.raises(ValueError, match=field_name):
        config_cls(**kwargs)


@pytest.mark.parametrize(
    ("config_cls", "kwargs"),
    [
        pytest.param(
            EffectiveRequestConfig,
            {
                "temperature": 0.0,
                "max_turns": 2**63 - 1,
                "read_only": False,
                "persist_session": True,
                "continuation_mode": "resume",
                "model_mode": "multi_or_dynamic",
            },
            id="effective-boundaries-and-zero",
        ),
        pytest.param(
            EffectiveRequestConfig,
            {"temperature": -0.5},
            id="negative-temperature-is-a-finite-float",
        ),
        pytest.param(
            ClaudeRequestConfig,
            {
                "permission_mode": "bypassPermissions",
                "allowed_tools_count": 0,
                "allowed_tools_present": False,
                "buffer_limit_bytes": 0,
            },
            id="claude-zeros-retained",
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
                "approval_mode": "deny-untrusted",
                "sandbox": True,
                "max_turns": 0,
                "empty_completion_threshold": 0,
                "observation_admission_bytes": 2 * 1024 * 1024,
            },
            id="osprey-zeros-and-closed-mode",
        ),
    ],
)
def test_admission_contract_accepts_exact_zero_and_boundaries(
    config_cls: type[Any],
    kwargs: dict[str, Any],
) -> None:
    """Zero, int64 boundary and closed-mode values construct and are preserved."""
    config = config_cls(**kwargs)
    for name, value in kwargs.items():
        assert getattr(config, name) == value


def test_admission_contract_defaults_mean_absent_not_effective() -> None:
    """Every defaulted field is None — absence never claims an effective value."""
    config = EffectiveRequestConfig()
    assert config.temperature is None
    assert config.max_turns is None
    assert config.read_only is None
    assert config.persist_session is None
    assert config.continuation_mode is None
    assert config.model_mode is None


def test_osprey_hidden_temperature_stays_absent() -> None:
    """Osprey's config-resolved temperature is hidden: only explicit admission."""
    # A config that does not pass temperature (the adapter omits it when the
    # flag was not emitted) must carry temperature=None — never a default.
    config = OspreyRequestConfig(sandbox=True)
    assert config.temperature is None


def test_arbitrary_persona_and_toolset_labels_have_no_field() -> None:
    """Persona/toolset labels never cross the config boundary (presence only)."""
    config = OspreyRequestConfig(persona_present=True, toolset_present=False)
    assert config.persona_present is True
    assert config.toolset_present is False
    # No label-carrying field exists to accept the arbitrary value.
    assert not hasattr(config, "persona")
    assert not hasattr(config, "toolset")


def test_identity_label_admission_rejects_unsafe_and_allows_namespace() -> None:
    """Model labels admit namespace slashes; reject controls/bidi/paths/secrets."""
    from daydream.backends import _admit_identity_label

    # Ordinary namespace spellings pass unchanged.
    for safe in (
        "opus",
        "claude-opus-4-5-20250901",
        "nous/deepseek-v4",
        "openai/gpt-5.2",
        "provider//model",
        "glm-4.6",
    ):
        admitted, diagnostic = _admit_identity_label(safe, max_chars=256, context="model")
        assert admitted == safe and diagnostic is None, safe

    # Controls, bidi, private paths, redaction-changed values, wrong types.
    for unsafe in (
        "/Users/ka/private/model",
        "C:\\repo\\model",
        "~/secret",
        "home/ka/model",
        "Users/ka/model",
        "a\x00b",
        "a\nb",
        "a\tb",
        "abc\u202ered",
        "abc\u200blad",
        "line\u2028sep",
        "AKIAIOSFODNN7EXAMPLE",
        "password=hunter2",
        "x-API-key: v",
        "m" * 257,
        "",
        None,
        5,
        1.5,
        b"model",
    ):
        admitted, diagnostic = _admit_identity_label(unsafe, max_chars=256, context="model")
        assert admitted is None and diagnostic is not None, repr(unsafe)
        assert diagnostic.code.startswith("config_identity_") or diagnostic.code in (
            "config_bool_type_rejected",
            "config_float_type_rejected",
            "config_int_type_rejected",
        )


def test_provider_label_bound_is_128() -> None:
    """Provider labels admit up to 128 scalar values and reject beyond."""
    from daydream.backends import _admit_identity_label

    admitted, diagnostic = _admit_identity_label("p" * 128, max_chars=128, context="provider")
    assert admitted == "p" * 128 and diagnostic is None
    admitted, diagnostic = _admit_identity_label("p" * 129, max_chars=128, context="provider")
    assert admitted is None
    assert diagnostic is not None and diagnostic.code == "config_identity_too_long"


def test_native_unix_ms_validation_bounds_and_conversion() -> None:
    """Native ms: bool/float/string rejected, bounded, exact ns multiplication."""
    from daydream.backends import _admit_native_unix_ms, unix_ms_to_ns

    # Exact producer shape converts by multiplication only.
    ms, diagnostic = _admit_native_unix_ms(1788690314289)
    assert ms == 1788690314289 and diagnostic is None
    assert unix_ms_to_ns(1788690314289) == 1788690314289000000

    # Zero is a valid instant; the int64//1e6 bound is exact.
    ms, diagnostic = _admit_native_unix_ms(0)
    assert ms == 0 and diagnostic is None
    bound = (2**63 - 1) // 1_000_000
    ms, diagnostic = _admit_native_unix_ms(bound)
    assert ms == bound and diagnostic is None
    ms, diagnostic = _admit_native_unix_ms(bound + 1)
    assert ms is None and diagnostic is not None

    # Bool (True == 1) is never a timestamp; nor floats/strings/negatives.
    for bad in (True, False, 1.5, "123", -1, -(10**6)):
        ms, diagnostic = _admit_native_unix_ms(bad)
        assert ms is None and diagnostic is not None, repr(bad)

    # Absence is not malformation.
    ms, diagnostic = _admit_native_unix_ms(None)
    assert ms is None and diagnostic is None


def test_observed_identity_lists_whole_overflow_omission() -> None:
    """A 17-entry model list omits the entire list with a fixed overflow code."""
    from daydream.observability.spans import _admit_observed_identity_list

    admitted, diagnostic = _admit_observed_identity_list(
        [f"model-{i}" for i in range(16)],
        max_chars=256,
        context="models",
    )
    assert admitted is not None and len(admitted) == 16
    assert diagnostic is None

    admitted, diagnostic = _admit_observed_identity_list(
        [f"model-{i}" for i in range(17)],
        max_chars=256,
        context="models",
    )
    assert admitted is None
    assert diagnostic is not None
    code, detail = diagnostic.split(":", 1)
    assert code == "config_list_overflow"
    assert detail == "17"  # total distinct count only

    # One unsafe member poisons the whole list (no partial lists).
    admitted, diagnostic = _admit_observed_identity_list(
        ["ok/model", "/Users/ka/secret"],
        max_chars=256,
        context="models",
    )
    assert admitted is None and diagnostic is not None
    assert diagnostic.split(":", 1)[0] == "config_list_member_unsafe"


def test_pi_selected_tools_count_derives_from_read_only_tool_constant() -> None:
    """The Pi read-only tool count derives from the argv tool list, not a copy."""
    from daydream.backends.pi import _PI_READ_ONLY_TOOLS

    assert len(_PI_READ_ONLY_TOOLS.split(",")) == 4
    assert _PI_READ_ONLY_TOOLS == "read,find,ls,grep"


def test_tool_call_choice_part_json_arguments_are_schema_admitted() -> None:
    """Choice-part arguments accept closed JSON and reject arbitrary objects."""
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
    """A path-shaped tool name is rejected, not truncated into an alias."""
    with pytest.raises(ValueError, match="name"):
        ToolCallChoicePart(call_id="t6", name="/Users/ka/bin/tool", arguments={})


def test_generation_lifecycle_events_are_in_the_agent_event_union() -> None:
    """GenerationStart/End participate in AgentEvent; timestamps default to Z."""
    start = GenerationStartEvent(generation_id="g-1", observed_at_unix_ns=1)
    end = GenerationEndEvent(
        generation_id="g-1",
        native_started_at_unix_ms=None,
        ended_at_unix_ns=2,
        end_source="fallback",
    )
    assert isinstance(start, AgentEvent)
    assert isinstance(end, AgentEvent)
    assert start.timestamp.endswith("Z") and end.timestamp.endswith("Z")


def test_generation_ids_are_host_invocation_local_uuids() -> None:
    """Minted IDs are UUID-shaped and distinct per call (never provider IDs)."""
    from daydream.backends import _new_generation_id

    first = _new_generation_id()
    second = _new_generation_id()
    assert first != second
    assert len(first) == 36 and first.count("-") == 4


def test_request_event_defaults_preserve_backward_compatibility() -> None:
    """Positional construction keeps its meaning; P18 fields default to absent."""
    legacy = RequestEvent("prompt only")
    assert legacy.prompt == "prompt only"
    assert legacy.config.model_mode is None
    assert legacy.model_source is None
    assert legacy.session_source is None
    assert legacy.timestamp_source == "host_observed"

    provenanced = RequestEvent(
        "p",
        model_name="m",
        model_source="configured",
        timestamp_source="native",
    )
    assert provenanced.model_source == "configured"
    assert provenanced.timestamp_source == "native"


def test_turn_end_event_defaults_preserve_backward_compatibility() -> None:
    """TurnEndEvent's new identity fields default to None/host_observed."""
    ev = TurnEndEvent()
    assert ev.message_id == ""
    assert ev.finish_reason is None
    assert ev.model_name is None
    assert ev.provider_name is None
    assert ev.message_id_source is None
    assert ev.model_source is None
    assert ev.timestamp_source == "host_observed"

    ev2 = TurnEndEvent(
        message_id="m1",
        finish_reason="stop",
        model_name="glm-4.6",
        provider_name="nous",
        model_source="native",
        provider_source="native",
    )
    assert ev2.finish_reason == "stop"
    assert ev2.model_source == "native"


def test_measurement_events_carry_closed_provenance() -> None:
    """CostEvent/MetricsEvent provenance fields default None and accept closed values."""
    cost = CostEvent(cost_usd=0.01, input_tokens=10, output_tokens=5, measurement_source="terminal")
    assert cost.measurement_source == "terminal"
    assert cost.generation_id is None
    assert cost.cost_source is None

    legacy = CostEvent(cost_usd=None, input_tokens=None, output_tokens=None)
    assert legacy.measurement_source is None  # legacy record, not an assessed claim

    metrics = MetricsEvent(
        message_id="m1",
        prompt_tokens=1,
        completion_tokens=2,
        cached_tokens=None,
        cost_usd=None,
        measurement_source="message_end",
        generation_id="g-1",
    )
    assert metrics.measurement_source == "message_end"
    assert metrics.generation_id == "g-1"
