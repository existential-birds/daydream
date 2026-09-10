"""HoneyHive compatibility through runner.run and parallel real OTLP exports."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.backends import (
    CostEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextChoicePart,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.observability.config import ObservabilityConfig
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.otlp import attributes, otlp_collector
from tests.test_observability_integration import _FLOW, _PROMPT, _REPLY, _SECRET, _configure


@pytest.mark.parametrize("capture_content", [True, False])
@pytest.mark.parametrize("tool_failed", [False, True])
async def test_honeyhive_native_mapping_is_isolated_from_generic_export(
    capture_content: bool,
    tool_failed: bool,
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext_dir.write_module(_FLOW)
    install_backend(
        ScriptedBackend(
            events=[
                RequestEvent(_PROMPT, model_name="effective-model", provider_name="observed-provider"),
                ToolStartEvent("call-one", "read_file", {"path": "sample.py", "api_key": _SECRET}),
                ToolResultEvent("call-one", f"Tool output includes credential {_SECRET}", tool_failed),
                TextEvent(_REPLY),
                MetricsEvent(
                    message_id="message-one",
                    prompt_tokens=100,
                    completion_tokens=12,
                    cached_tokens=20,
                    cache_creation_tokens=5,
                    reasoning_tokens=6,
                    cost_usd=0.004,
                ),
                CostEvent(
                    cost_usd=0.004,
                    input_tokens=100,
                    output_tokens=12,
                    cached_tokens=20,
                    cache_creation_tokens=5,
                    reasoning_tokens=6,
                    measurement_source="terminal",
                ),
                ResultEvent({"answer": "safe"}, None, finish_reason="stop"),
            ]
        )
    )
    with otlp_collector() as honeyhive, otlp_collector() as generic:
        _configure(monkeypatch, generic.base_url, "honeyhive,otlp")
        monkeypatch.setenv("HH_API_URL", honeyhive.base_url)
        config = make_config(
            feature_branch_repo,
            flow_name="trace-probe",
            observability=ObservabilityConfig(destinations=("honeyhive", "otlp"), capture_content=capture_content),
        )
        assert await runner.run(config) == 0

    assert json.loads((feature_branch_repo / ".daydream/observability-result.json").read_text()) == {"answer": "safe"}
    generic_spans = {span["spanId"]: span for span in generic.spans}
    assert len(honeyhive.spans) == len(generic_spans) == 5
    type_mapping = {"run": "chain", "step": "chain", "agent": "chain", "attempt": "chain", "tool": "tool"}
    trajectory_paths = list((feature_branch_repo / ".daydream/runs").glob("*/trajectory.json"))
    assert len(trajectory_paths) == 1
    session_id = json.loads(trajectory_paths[0].read_text())["session_id"]
    for mapped in honeyhive.spans:
        native = attributes(mapped)
        original = generic_spans[mapped["spanId"]]
        portable = attributes(original)
        kind = native["daydream.span.kind"]
        assert native["honeyhive_event_type"] == type_mapping[kind]
        assert native["honeyhive.session_id"] == session_id
        assert native["honeyhive.session_auto_create"] is True
        assert native["honeyhive.session_name"] == "daydream.trace-probe"
        assert {key: value for key, value in native.items() if not key.startswith("honeyhive")} == portable
        assert not any(key.startswith("honeyhive") for key in portable)
        assert {key: value for key, value in mapped.items() if key != "attributes"} == {
            key: value for key, value in original.items() if key != "attributes"
        }
        if kind == "attempt":
            assert native["honeyhive_metadata.prompt_tokens"] == native["gen_ai.usage.input_tokens"] == 100
            assert native["honeyhive_metadata.completion_tokens"] == native["gen_ai.usage.output_tokens"] == 12
            assert native["honeyhive_metadata.cost"] == native["gen_ai.usage.cost"] == 0.004
            assert native["honeyhive_metadata.cache_read_input_tokens"] == 20
            assert native["honeyhive_metadata.cache_write_input_tokens"] == 5
            assert native["honeyhive_metadata.reasoning_tokens"] == 6
            assert native["gen_ai.usage.input_tokens"] == 100
            assert native["gen_ai.usage.output_tokens"] == 12
        else:
            assert not any(key.startswith("honeyhive_metadata.") for key in native)
            assert "gen_ai.usage.cost" not in native
        if kind == "tool":
            assert (mapped["status"]["code"] == "STATUS_CODE_ERROR") == tool_failed
    assert sum(attributes(span).get("honeyhive_metadata.cost", 0) for span in honeyhive.spans) == 0.004
    payload = json.dumps([request["body"] for request in honeyhive.requests])
    assert _SECRET not in payload
    for text in (_PROMPT, _REPLY, "sample.py", "Tool output includes credential"):
        assert (text in payload) == capture_content
    assert all(request["path"] == "/opentelemetry/v1/traces" for request in honeyhive.requests)
    assert all(request["headers"]["authorization"] == f"Bearer {_SECRET}" for request in honeyhive.requests)


async def test_honeyhive_early_workspace_failure_materializes_session_from_run_identity(
    tmp_path: Path,
    make_config: Callable[..., RunConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with otlp_collector() as honeyhive, otlp_collector() as generic:
        _configure(monkeypatch, generic.base_url, "honeyhive,otlp")
        monkeypatch.setenv("HH_API_URL", honeyhive.base_url)
        assert await runner.run(make_config(tmp_path, output_mode="review")) == 1
    assert len(honeyhive.spans) == len(generic.spans) == 1
    root = honeyhive.spans[0]
    native = attributes(root)
    portable = attributes(generic.spans[0])
    assert root["status"]["code"] == "STATUS_CODE_ERROR"
    assert root["spanId"] == generic.spans[0]["spanId"]
    assert root["traceId"] == generic.spans[0]["traceId"]
    assert native["honeyhive_event_type"] == "chain"
    assert native["honeyhive.session_id"] == portable["daydream.run.id"]
    assert native["honeyhive.session_auto_create"] is True
    assert native["honeyhive.session_name"] == f"daydream.{portable['daydream.flow']}"
    assert "traceloop.association.properties.session_id" not in portable
    assert {key: value for key, value in native.items() if not key.startswith("honeyhive")} == portable
    assert not any(key.startswith(("honeyhive", "gen_ai.usage.")) for key in portable)
    assert not any(key.startswith("honeyhive_metadata.") for key in native)
    assert not (tmp_path / ".daydream/runs").exists()


async def test_honeyhive_outage_preserves_result_and_generic_export(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ext_dir.write_module(_FLOW)
    install_backend(ScriptedBackend(events=[RequestEvent(_PROMPT), ResultEvent({"answer": "safe"}, None)]))
    with otlp_collector(status=400, reason=_SECRET) as honeyhive, otlp_collector() as generic:
        _configure(monkeypatch, generic.base_url, "honeyhive,otlp")
        monkeypatch.setenv("HH_API_URL", honeyhive.base_url)
        assert await runner.run(make_config(feature_branch_repo, flow_name="trace-probe")) == 0
    assert json.loads((feature_branch_repo / ".daydream/observability-result.json").read_text()) == {"answer": "safe"}
    assert len(honeyhive.spans) == len(generic.spans) == 4
    assert all(attributes(span)["honeyhive_event_type"] for span in honeyhive.spans)
    assert "Failed to export" in caplog.text
    assert _SECRET not in caplog.text


async def test_honeyhive_generation_child_is_model_and_attempt_stays_chain(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real Pi-style stream yields one model child per approved generation.

    The structural attempt and the logical agent remain chain; only the sealed
    generation becomes a HoneyHive model event, and the exactly-once historical
    end survives to the wire (issue #1156 AC-10/AC-18).
    """

    ext_dir.write_module(_FLOW)
    install_backend(
        ScriptedBackend(
            events=[
                RequestEvent(_PROMPT, model_name="pi-model", provider_name="pi"),
                GenerationStartEvent(generation_id="gen-one", observed_at_unix_ns=1788690314289000000),
                GenerationEndEvent(
                    generation_id="gen-one",
                    native_started_at_unix_ms=1788690314289,
                    ended_at_unix_ns=1788690709621000000,
                    end_source="host_observed_message_end",
                    choice_parts=(TextChoicePart(text=_REPLY),),
                    response_id="late-response",
                    model_name="pi-model",
                    provider_name="pi",
                    finish_reason="stop",
                ),
                TextEvent(_REPLY),
                MetricsEvent(
                    message_id="",
                    prompt_tokens=10,
                    completion_tokens=2,
                    cached_tokens=None,
                    cost_usd=0.001,
                    generation_id="gen-one",
                ),
                CostEvent(
                    cost_usd=0.001,
                    input_tokens=10,
                    output_tokens=2,
                    measurement_source="terminal",
                ),
                ResultEvent({"answer": "safe"}, None, finish_reason="stop"),
            ]
        )
    )
    with otlp_collector() as honeyhive:
        _configure(monkeypatch, honeyhive.base_url, "honeyhive")
        monkeypatch.setenv("HH_API_URL", honeyhive.base_url)
        assert await runner.run(make_config(feature_branch_repo, flow_name="trace-probe")) == 0

    by_kind: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for span in honeyhive.spans:
        native = attributes(span)
        by_kind.setdefault(native["daydream.span.kind"], []).append((span, native))
    assert set(by_kind) == {"run", "step", "agent", "attempt", "generation"}
    generation_raw, generation = by_kind["generation"][0]
    attempt_raw, attempt = by_kind["attempt"][0]
    assert attempt["honeyhive_event_type"] == "chain"
    assert attempt_raw["parentSpanId"] == by_kind["agent"][0][0]["spanId"]
    assert generation["honeyhive_event_type"] == "model"
    # The generation child hangs under the attempt and ends at the sealed
    # historical host message_end (1788690709621000000 ns).
    assert generation_raw["parentSpanId"] == attempt_raw["spanId"]
    assert generation_raw["startTimeUnixNano"] == "1788690314289000000"
    assert generation_raw["endTimeUnixNano"] == "1788690709621000000"
    assert generation["daydream.generation.billed"] is True
    # One billable owner: the model child carries the native usage/cost and
    # the structural chain does not double-bill it.
    assert generation["honeyhive_metadata.prompt_tokens"] == 10
    assert generation["honeyhive_metadata.cost"] == 0.001
    assert not any(key.startswith("honeyhive_metadata.") for key in attempt)
    # Standard agent identity on the logical agent scope.
    assert by_kind["agent"][0][1]["gen_ai.agent.name"] == "review"
