"""Preserve per-exec usage observed on native Codex CLI 0.153.4 resume."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from daydream import runner
from daydream.backends.codex import CodexBackend
from daydream.observability.config import ObservabilityConfig
from daydream.pricing import compute_cost_from_totals, load_user_prices, resolve_prices
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.fake_cli_process import FakeCliProcess
from tests.harness.otlp import attributes, otlp_collector


@pytest.mark.asyncio
async def test_runner_resumed_codex_usage_is_per_exec_not_prior_session_delta(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext_dir.write_module('''
import json
from daydream.agent import run_agent
from daydream.extensions import FlowStep
from daydream.trajectory import DaydreamPhase

async def resume_probe(ctx):
    continuation = None
    outputs = []
    for _ in range(2):
        output, continuation, aborted = await run_agent(
            ctx.backend_for("review"), ctx.work.repo, "Continue",
            phase=DaydreamPhase.REVIEW, continuation=continuation,
        )
        if aborted or continuation is None:
            raise RuntimeError("Codex continuation was not completed")
        outputs.append(output)
    # The public .daydream tree is detached for the run's duration; write
    # through the session-routed live dir, which is republished publicly at
    # run end (keeps the post-run public-path assertion below unchanged).
    (ctx.data["daydream_dir"] / "resume-outputs.json").write_text(json.dumps(outputs))

def register(r):
    r.register_phase(FlowStep(name="resume-probe", run=resume_probe))
    r.set_flow("resume-probe", ["resume-probe"])
''')
    for name in os.environ:
        if name.startswith(("OTEL_", "_OTEL_", "DAYDREAM_TRACE_")):
            monkeypatch.delenv(name)
    backend = CodexBackend(model="gpt-5.3-codex")
    install_backend(backend)
    trajectory_path = feature_branch_repo / ".daydream/resume-trajectory.json"
    # Observed in one native thread across two exec processes: the resumed
    # process totals 28,674 + 28,849 input and 104 + 60 output, without adding
    # the first process's 44,472 input / 167 output to its usage counters.
    native_totals = [
        {"input_tokens": 44472, "output_tokens": 167, "cached_input_tokens": 22016},
        {"input_tokens": 57523, "output_tokens": 164, "cached_input_tokens": 50560},
    ]
    processes = [
        FakeCliProcess([
            json.dumps({"type": "thread.started", "thread_id": "native-thread"}),
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "item": {
                "type": "agent_message", "id": "reply", "text": "Done",
            }}),
            json.dumps({"type": "turn.completed", "usage": usage}),
        ])
        for usage in native_totals
    ]
    with otlp_collector() as collector, patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=processes,
    ) as launch:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", collector.base_url + "/v1/traces")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
        result = await runner.run(make_config(
            feature_branch_repo, flow_name="resume-probe", backend="codex", model=backend.model,
            observability=ObservabilityConfig(destinations=("otlp",)), trajectory_path=trajectory_path,
        ))
    assert result == 0
    assert launch.call_count == 2
    assert "resume" in launch.call_args_list[1].args
    assert "native-thread" in launch.call_args_list[1].args
    assert json.loads((feature_branch_repo / ".daydream/resume-outputs.json").read_text()) == ["Done", "Done"]
    attempts = sorted(
        (span for span in collector.spans if attributes(span).get("daydream.span.kind") == "attempt"),
        key=lambda span: int(span["startTimeUnixNano"]),
    )
    assert [attributes(span)["gen_ai.usage.input_tokens"] for span in attempts] == [44472, 57523]
    assert [attributes(span)["gen_ai.usage.output_tokens"] for span in attempts] == [167, 164]
    assert [attributes(span)["gen_ai.usage.cache_read.input_tokens"] for span in attempts] == [22016, 50560]
    assert all(span["status"]["code"] == "STATUS_CODE_OK" for span in attempts)
    expected_costs = [
        compute_cost_from_totals(
            backend.model, total_input_tokens=usage["input_tokens"],
            cached_input_tokens=usage["cached_input_tokens"], output_tokens=usage["output_tokens"],
            prices=resolve_prices(load_user_prices()),
        )
        for usage in native_totals
    ]
    assert [attributes(span)["gen_ai.usage.cost"] for span in attempts] == pytest.approx(expected_costs)
    trajectory = json.loads(trajectory_path.read_text())
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 101995
    assert trajectory["final_metrics"]["total_completion_tokens"] == 331
    assert sum(step.get("metrics", {}).get("prompt_tokens", 0) for step in trajectory["steps"]) == 101995
    assert trajectory["final_metrics"]["total_cost_usd"] == pytest.approx(
        sum(cost for cost in expected_costs if cost is not None)
    )
