"""Tracing through the production runner, real git/files, and real OTLP HTTP."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream import cli, runner
from daydream.backends import (
    CostEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.observability.config import ObservabilityConfig
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.otlp import attributes, otlp_collector

_SECRET = "opaque-observability-credential-73951"
_PROMPT = "Review the observability sample and return its answer."
_SYSTEM = "Use the declared answer schema."
_REPLY = "The sample was checked successfully."
_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

_FLOW = '''
import json
from daydream.agent import run_agent
from daydream.extensions import FlowStep
from daydream.trajectory import DaydreamPhase

async def trace_probe(ctx):
    output, _, _ = await run_agent(
        ctx.backend_for("review"), ctx.work.repo,
        "Review the observability sample and return its answer.",
        phase=DaydreamPhase.REVIEW,
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    )
    (ctx.work.repo / ".daydream" / "observability-result.json").write_text(json.dumps(output))

def register(r):
    r.register_phase(FlowStep(name="trace-probe", run=trace_probe))
    r.set_flow("trace-probe", ["trace-probe"])
'''


def _backend() -> ScriptedBackend:
    return ScriptedBackend(events=[
        RequestEvent(prompt=_PROMPT, system_prompt=_SYSTEM, output_schema=_SCHEMA,
                     model_name="observed-model", provider_name="observed-provider"),
        ThinkingEvent(text="Inspect the sample's return value."),
        ToolStartEvent(id="call-one", name="read_file", input={"path": "sample.py", "api_key": _SECRET}),
        ToolResultEvent(id="call-one", output=f"return 'safe'; credential={_SECRET}", is_error=False),
        TextEvent(text=_REPLY),
        MetricsEvent(message_id="message-one", prompt_tokens=100, completion_tokens=12,
                     cached_tokens=20, cache_creation_tokens=5, cost_usd=0.004,
                     model_name="observed-model", provider_name="observed-provider"),
        CostEvent(cost_usd=0.004, input_tokens=100, output_tokens=12, cached_tokens=20,
                  cache_creation_tokens=5, model_name="observed-model", provider_name="observed-provider"),
        ResultEvent(structured_output={"answer": "safe"}, continuation=None, model_name="observed-model",
                    provider_name="observed-provider", session_id="native-session-one", finish_reason="stop"),
    ])


def _configure(monkeypatch: pytest.MonkeyPatch, base_url: str, destination: str) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET)
    monkeypatch.setenv("LANGSMITH_PROJECT", "daydream-wire-test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", base_url)
    monkeypatch.setenv("HH_API_KEY", _SECRET)
    monkeypatch.setenv("HH_API_URL", base_url)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", base_url)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("DAYDREAM_TRACE_TO", destination)


@pytest.mark.parametrize("destination", ["otlp", "langsmith", "honeyhive"])
async def test_runner_exports_complete_portable_trace(
    destination: str,
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext_dir.write_module(_FLOW)
    install_backend(_backend())
    with otlp_collector() as receiver:
        _configure(monkeypatch, receiver.base_url, destination)
        rc = await runner.run(make_config(feature_branch_repo, flow_name="trace-probe"))

    assert rc == 0
    assert json.loads((feature_branch_repo / ".daydream/observability-result.json").read_text()) == {"answer": "safe"}
    spans = receiver.spans
    roots = [span for span in spans if not span.get("parentSpanId")]
    assert len(roots) == 1
    assert len({span["traceId"] for span in spans}) == 1
    by_id = {span["spanId"]: span for span in spans}
    tool = next(span for span in spans if attributes(span).get("gen_ai.tool.name") == "read_file")
    attempt = by_id[tool["parentSpanId"]]
    agent = by_id[attempt["parentSpanId"]]
    phase = by_id[agent["parentSpanId"]]
    assert phase["parentSpanId"] == roots[0]["spanId"]
    billed = attributes(attempt)
    assert billed["gen_ai.usage.input_tokens"] == 100
    assert billed["gen_ai.usage.output_tokens"] == 12
    assert billed["gen_ai.response.model"] == "observed-model"
    assert billed["gen_ai.provider.name"] == "observed-provider"
    assert _PROMPT in billed["gen_ai.input.messages"]
    assert _SYSTEM in billed["gen_ai.input.messages"]
    assert _REPLY in billed["gen_ai.output.messages"]
    assert "call-one" in billed["gen_ai.output.messages"]
    assert "sample.py" in attributes(tool)["traceloop.entity.input"]
    assert "return 'safe'" in attributes(tool)["traceloop.entity.output"]
    assert all("gen_ai.usage.input_tokens" not in attributes(span) for span in (agent, phase, roots[0]))
    payload = json.dumps([request["body"] for request in receiver.requests])
    assert _SECRET not in payload
    assert "native-session-one" in payload
    paths = {request["path"] for request in receiver.requests}
    assert paths == {dict(otlp="/v1/traces", langsmith="/otel/v1/traces",
                          honeyhive="/opentelemetry/v1/traces")[destination]}
    if destination == "langsmith":
        assert billed["langsmith.span.kind"] == "llm"
        usage = json.loads(billed["langsmith.usage_metadata"])
        assert usage["input_tokens"] == 100
        assert usage["input_token_details"]["cache_read"] == 20
        assert usage["input_token_details"]["cache_creation"] == 5
        assert usage["total_cost"] == 0.004


async def test_runner_metadata_policy_preserves_structure_and_omits_content(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext_dir.write_module(_FLOW)
    install_backend(_backend())
    with otlp_collector() as receiver:
        _configure(monkeypatch, receiver.base_url, "otlp")
        config = make_config(feature_branch_repo, flow_name="trace-probe",
                             observability=ObservabilityConfig(destinations=("otlp",), capture_content=False))
        assert await runner.run(config) == 0

    assert len(receiver.spans) >= 5
    payload = json.dumps([request["body"] for request in receiver.requests])
    for content in (_PROMPT, _SYSTEM, _REPLY, _SECRET, "sample.py", "return 'safe'", "Inspect the sample"):
        assert content not in payload
    assert any(attributes(span).get("gen_ai.usage.input_tokens") == 100 for span in receiver.spans)
    for span in receiver.spans:
        attrs = attributes(span)
        assert not any(key in attrs for key in (
            "gen_ai.input.messages", "gen_ai.output.messages", "traceloop.entity.input", "traceloop.entity.output",
        ))


async def test_explicit_off_overrides_environment_and_repository_cannot_enable_tracing(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext_dir.write_module(_FLOW)
    install_backend(_backend())
    with otlp_collector() as receiver:
        _configure(monkeypatch, receiver.base_url, "otlp")
        config = make_config(feature_branch_repo, flow_name="trace-probe", observability=ObservabilityConfig())
        assert await runner.run(config) == 0
        monkeypatch.delenv("DAYDREAM_TRACE_TO")
        (feature_branch_repo / ".daydream.toml").write_text(
            'model = "file-config-was-loaded"\n'
            '[observability]\ndestinations = ["otlp"]\nendpoint = "' + receiver.base_url + '"\n'
        )
        config = cli._parse_args([
            str(feature_branch_repo), "--flow", "trace-probe", "--non-interactive", "--no-archive",
        ])
        assert config.file_config is not None
        assert config.file_config.model == "file-config-was-loaded"
        assert await runner.run(config) == 0
    assert receiver.requests == []


async def test_multiple_destinations_receive_same_trace(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext_dir.write_module(_FLOW)
    install_backend(_backend())
    with otlp_collector() as generic, otlp_collector() as langsmith:
        _configure(monkeypatch, generic.base_url, "otlp,langsmith")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", langsmith.base_url)
        assert await runner.run(make_config(feature_branch_repo, flow_name="trace-probe")) == 0
    assert generic.spans
    assert {span["spanId"] for span in generic.spans} == {span["spanId"] for span in langsmith.spans}
    assert {span["traceId"] for span in generic.spans} == {span["traceId"] for span in langsmith.spans}


async def test_extension_factory_exports_real_runner_spans(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_flow = _FLOW.replace(
        'def register(r):',
        'def custom_exporter(config):\n'
        '    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter\n'
        '    return OTLPSpanExporter()\n\n'
        'def register(r):\n'
        '    r.register_trace_exporter("custom", custom_exporter)',
    )
    ext_dir.write_module(custom_flow)
    install_backend(_backend())
    with otlp_collector() as receiver:
        _configure(monkeypatch, receiver.base_url, "custom")
        assert await runner.run(make_config(feature_branch_repo, flow_name="trace-probe")) == 0
    assert any(attributes(span).get("gen_ai.tool.name") == "read_file" for span in receiver.spans)


@pytest.mark.parametrize("destination", ["unknown-exporter", "langsmith", "honeyhive"])
async def test_invalid_destination_setup_fails_before_agent_work(
    destination: str,
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ext_dir.write_module(_FLOW)
    backend = _backend()
    install_backend(backend)
    monkeypatch.setenv("DAYDREAM_TRACE_TO", destination)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("HH_API_KEY", raising=False)
    assert await runner.run(make_config(feature_branch_repo, flow_name="trace-probe")) == 1
    assert backend.calls == []
    assert destination in capsys.readouterr().out.lower()
