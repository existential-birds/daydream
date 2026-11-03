"""Tracing through the production runner, real git/files, and real OTLP HTTP."""

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
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.run_config import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import Responder, ScriptedBackend
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
    result_path = artifact_dir_for(
        ctx.work.repo, session=ctx.artifacts, allow_standalone=False
    ) / "observability-result.json"
    result_path.write_text(json.dumps(output))

def register(r):
    r.register_phase(FlowStep(name="trace-probe", run=trace_probe))
    r.set_flow("trace-probe", ["trace-probe"])
'''


def _backend(*, responder: Responder | None = None) -> ScriptedBackend:
    return ScriptedBackend(events=[
        RequestEvent(prompt=_PROMPT, system_prompt=_SYSTEM, output_schema=_SCHEMA,
                     model_name="observed-model", provider_name="observed-provider"),
        ThinkingEvent(text="Inspect the sample's return value."),
        ToolStartEvent(id="call-one", name="read_file", input={"path": "sample.py", "api_key": _SECRET}),
        ToolResultEvent(id="call-one", output=f"return 'safe'; credential={_SECRET}", is_error=False),
        TextEvent(text=_REPLY), MetricsEvent(message_id="message-one", prompt_tokens=100, completion_tokens=12,
                     cached_tokens=20, cache_creation_tokens=5, cost_usd=0.004,
                     model_name="observed-model", provider_name="observed-provider"),
        CostEvent(cost_usd=0.004, input_tokens=100, output_tokens=12, cached_tokens=20,
                  cache_creation_tokens=5, model_name="observed-model", provider_name="observed-provider",
                  measurement_source="terminal"),
        ResultEvent(structured_output={"answer": "safe"}, continuation=None, model_name="observed-model",
                    provider_name="observed-provider", session_id="native-session-one", finish_reason="stop"),
    ], responder=responder)


def _configure(monkeypatch: pytest.MonkeyPatch, base_url: str, destination: str) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET)
    monkeypatch.setenv("LANGSMITH_PROJECT", "daydream-wire-test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", base_url)
    monkeypatch.setenv("HH_API_KEY", _SECRET)
    monkeypatch.setenv("HH_API_URL", base_url)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", base_url)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("DAYDREAM_TRACE_TO", destination)

async def test_runner_redacts_encoded_and_decoded_url_credentials_despite_malformed_unrelated_url(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_setting = "UPSTREAM_ENDPOINT"
    credentials = ("operator%2Fname", "operator/name", "opaque%2Fsecret", "opaque/secret")
    exposed_text = "Observed values: " + " then ".join(credentials)
    monkeypatch.setenv(endpoint_setting, "https://operator%2Fname:opaque%2Fsecret@service.test")
    monkeypatch.setenv("BROKEN_API_URL", "https://[malformed-ipv6")
    ext_dir.write_module(_FLOW)
    install_backend(ScriptedBackend(events=[
        RequestEvent(exposed_text), ToolStartEvent("credential-probe", "Read", {"note": exposed_text}),
        ToolResultEvent("credential-probe", exposed_text, False), TextEvent(exposed_text), CostEvent(0.005, 20, 4),
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
    by_id = {span["spanId"]: span for span in spans}
    tool = next(span for span in spans if attributes(span).get("daydream.span.kind") == "tool")
    attempt = by_id[tool["parentSpanId"]]
    agent = by_id[attempt["parentSpanId"]]
    phase = by_id[agent["parentSpanId"]]
    root = by_id[phase["parentSpanId"]]
    assert not root.get("parentSpanId")
    finalization, = [span for span in spans if attributes(span).get("daydream.step") == "artifact-finalization"]
    assert finalization["parentSpanId"] == root["spanId"]
    assert len({span["traceId"] for span in spans}) == 1
    assert attributes(attempt)["gen_ai.usage.input_tokens"] == 20
    assert attributes(attempt)["gen_ai.usage.output_tokens"] == 4
    assert attributes(attempt)["gen_ai.usage.cost"] == 0.005
    for text in (
        attributes(attempt)["gen_ai.input.messages"], attributes(attempt)["gen_ai.output.messages"],
        attributes(tool)["traceloop.entity.input"], attributes(tool)["traceloop.entity.output"],
    ):
        assert "Observed values:" in text and "[REDACTED_CREDENTIAL]" in text

async def test_invalid_destination_setup_fails_before_agent_work(
    ext_dir: ExtDir, feature_branch_repo: Path, make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    ext_dir.write_module(_FLOW)
    backend = _backend()
    install_backend(backend)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("HH_API_KEY", raising=False)
    for destination in ("unknown-exporter", "langsmith"):
        monkeypatch.setenv("DAYDREAM_TRACE_TO", destination)
        assert await runner.run(make_config(feature_branch_repo, flow_name="trace-probe")) == 1
        assert backend.calls == []
        assert destination in capsys.readouterr().out.lower()

# --- P18 Task 3: sanitized resource reaches every destination identically ------
