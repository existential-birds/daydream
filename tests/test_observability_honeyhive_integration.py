"""HoneyHive compatibility through runner.run and parallel real OTLP exports."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream import runner
from daydream.backends import (
    CostEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
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
    type_mapping = {"run": "chain", "step": "chain", "agent": "chain", "attempt": "model", "tool": "tool"}
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
