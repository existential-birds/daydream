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
from daydream.artifact_visibility import artifact_dir_for
from daydream.extensions import FlowStep
from daydream.trajectory import DaydreamPhase

async def trace_probe(ctx):
    output, _, _ = await run_agent(
        ctx.backend_for("review"), ctx.work.repo,
        "Review the observability sample and return its answer.",
        phase=DaydreamPhase.REVIEW,
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    )
    (artifact_dir_for(ctx.work.repo) / "observability-result.json").write_text(json.dumps(output))

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
    trajectory_paths = list((feature_branch_repo / ".daydream/runs").glob("*/trajectory.json"))
    assert len(trajectory_paths) == 1
    trajectory = json.loads(trajectory_paths[0].read_text())
    for span in (roots[0], phase, agent, attempt, tool):
        identity = attributes(span)
        assert identity["daydream.session.id"] == trajectory["session_id"]
        assert identity["daydream.trajectory.id"] == trajectory["trajectory_id"]
        assert identity["traceloop.association.properties.session_id"] == trajectory["session_id"]
    billed = attributes(attempt)
    assert billed["gen_ai.usage.input_tokens"] == 100
    assert billed["gen_ai.usage.output_tokens"] == 12
    assert billed["gen_ai.request.model"] == "observed-model"
    assert billed["gen_ai.response.model"] == "observed-model"
    assert billed["gen_ai.operation.name"] == "chat"
    assert billed["daydream.invocation.aggregate"] is True
    assert billed["gen_ai.response.finish_reasons"] == ["stop"]
    assert billed["gen_ai.provider.name"] == "observed-provider"
    assert _PROMPT in billed["gen_ai.input.messages"]
    assert _SYSTEM in billed["gen_ai.input.messages"]
    assert _REPLY in billed["gen_ai.output.messages"]
    assert "call-one" in billed["gen_ai.output.messages"]
    assert "sample.py" in attributes(tool)["traceloop.entity.input"]
    assert "return 'safe'" in attributes(tool)["traceloop.entity.output"]
    for span in (agent, phase, roots[0], tool):
        local = attributes(span)
        assert "gen_ai.request.model" not in local
        assert "daydream.invocation.aggregate" not in local
        assert not any(key.startswith("gen_ai.usage.") for key in local)
    assert attributes(agent)["daydream.configured.model"] == "test-model"
    assert attributes(tool)["gen_ai.operation.name"] == "execute_tool"
    assert attributes(tool)["daydream.attempt"] == 1
    assert attributes(tool)["daydream.phase"] == "review"
    assert attributes(tool)["daydream.step"] == "trace-probe"
    payload = json.dumps([request["body"] for request in receiver.requests])
    assert _SECRET not in payload
    assert "native-session-one" in payload
    paths = {request["path"] for request in receiver.requests}
    assert paths == {dict(otlp="/v1/traces", langsmith="/otel/v1/traces",
                          honeyhive="/opentelemetry/v1/traces")[destination]}
    if destination == "langsmith":
        assert billed["langsmith.span.kind"] == "llm"
        assert attributes(tool)["langsmith.span.kind"] == "tool"
        assert all(attributes(span)["langsmith.span.kind"] == "chain" for span in (roots[0], phase, agent))
        usage = json.loads(billed["langsmith.usage_metadata"])
        assert usage["input_tokens"] == 100
        assert usage["input_token_details"]["cache_read"] == 20
        assert usage["input_token_details"]["cache_creation"] == 5
        assert usage["total_cost"] == 0.004


async def test_runner_preserves_redacted_scalar_arrays_and_encodes_nested_attributes(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _FLOW.replace(
        "async def trace_probe(ctx):",
        "async def trace_probe(ctx):\n"
        "    from daydream.observability.runtime import current_session\n"
        "    from daydream.observability.spans import SpanScope\n"
        "    with SpanScope(current_session(), 'attribute-probe', 'task') as scope:\n"
        "        scope.attrs({\n"
        f"            'probe.strings': ['safe', '{_SECRET}'],\n"
        "            'probe.integers': [1, 2],\n"
        "            'probe.doubles': [1.5, 2.5],\n"
        "            'probe.booleans': [True, False],\n"
        "            'probe.empty': [],\n"
        "            'probe.mixed': [1, True],\n"
        f"            'probe.nested': {{'items': ['safe', '{_SECRET}']}},\n"
        "        })",
    )
    ext_dir.write_module(probe)
    install_backend(ScriptedBackend(events=[
        RequestEvent(_PROMPT), TextEvent(_REPLY),
        ResultEvent({"answer": "safe"}, None, finish_reason=f"stop {_SECRET}"),
    ]))
    with otlp_collector() as receiver:
        _configure(monkeypatch, receiver.base_url, "otlp")
        assert await runner.run(make_config(feature_branch_repo, flow_name="trace-probe")) == 0
    attempt = next(span for span in receiver.spans if attributes(span).get("daydream.span.kind") == "attempt")
    reasons = attributes(attempt)["gen_ai.response.finish_reasons"]
    assert isinstance(reasons, list) and len(reasons) == 1
    assert reasons[0].startswith("stop ") and _SECRET not in reasons[0]
    attrs = attributes(next(span for span in receiver.spans if span["name"] == "attribute-probe"))
    assert attrs["probe.strings"] == ["safe", "[REDACTED_CREDENTIAL]"]
    assert attrs["probe.integers"] == [1, 2]
    assert attrs["probe.doubles"] == [1.5, 2.5]
    assert attrs["probe.booleans"] == [True, False]
    assert attrs["probe.empty"] == []
    assert json.loads(attrs["probe.mixed"]) == [1, True]
    assert json.loads(attrs["probe.nested"]) == {"items": ["safe", "[REDACTED_CREDENTIAL]"]}
    assert _SECRET not in json.dumps([request["body"] for request in receiver.requests])


@pytest.mark.parametrize("endpoint_setting", ["UPSTREAM_API_URL", "UPSTREAM_ENDPOINT"])
async def test_runner_redacts_encoded_and_decoded_url_credentials_despite_malformed_unrelated_url(
    endpoint_setting: str,
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = ("operator%2Fname", "operator/name", "opaque%2Fsecret", "opaque/secret")
    exposed_text = "Observed values: " + " then ".join(credentials)
    monkeypatch.setenv(endpoint_setting, "https://operator%2Fname:opaque%2Fsecret@service.test")
    monkeypatch.setenv("BROKEN_API_URL", "https://[malformed-ipv6")
    ext_dir.write_module(_FLOW)
    install_backend(ScriptedBackend(events=[
        RequestEvent(exposed_text),
        ToolStartEvent("credential-probe", "Read", {"note": exposed_text}),
        ToolResultEvent("credential-probe", exposed_text, False),
        TextEvent(exposed_text), CostEvent(0.005, 20, 4),
        ResultEvent({"answer": "safe"}, None),
    ]))
    with otlp_collector() as receiver:
        _configure(monkeypatch, receiver.base_url, "otlp")
        assert await runner.run(make_config(feature_branch_repo, flow_name="trace-probe")) == 0
    assert json.loads((feature_branch_repo / ".daydream/observability-result.json").read_text()) == {"answer": "safe"}
    payload = json.dumps([request["body"] for request in receiver.requests])
    for credential in credentials:
        assert credential not in payload
    spans = receiver.spans
    assert len(spans) == 5
    by_id = {span["spanId"]: span for span in spans}
    tool = next(span for span in spans if attributes(span).get("daydream.span.kind") == "tool")
    attempt = by_id[tool["parentSpanId"]]
    agent = by_id[attempt["parentSpanId"]]
    phase = by_id[agent["parentSpanId"]]
    root = by_id[phase["parentSpanId"]]
    assert not root.get("parentSpanId")
    assert len({span["traceId"] for span in spans}) == 1
    assert attributes(attempt)["gen_ai.usage.input_tokens"] == 20
    assert attributes(attempt)["gen_ai.usage.output_tokens"] == 4
    assert attributes(attempt)["gen_ai.usage.cost"] == 0.005
    for text in (
        attributes(attempt)["gen_ai.input.messages"], attributes(attempt)["gen_ai.output.messages"],
        attributes(tool)["traceloop.entity.input"], attributes(tool)["traceloop.entity.output"],
    ):
        assert "Observed values:" in text and "[REDACTED_CREDENTIAL]" in text


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


async def test_workspace_failure_exports_root_before_a_trajectory_exists(
    tmp_path: Path,
    make_config: Callable[..., RunConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with otlp_collector() as receiver:
        _configure(monkeypatch, receiver.base_url, "otlp")
        assert await runner.run(make_config(tmp_path, output_mode="review")) == 1
    assert len(receiver.spans) == 1
    root = receiver.spans[0]
    identity = attributes(root)
    assert identity["daydream.span.kind"] == "run"
    assert root["status"]["code"] == "STATUS_CODE_ERROR"
    assert "daydream.session.id" not in identity
    assert "daydream.trajectory.id" not in identity
    assert "traceloop.association.properties.session_id" not in identity
    assert not (tmp_path / ".daydream/runs").exists()
