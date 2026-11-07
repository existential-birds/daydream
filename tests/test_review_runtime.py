"""Bounded review investigation retains evidence without claiming completion."""

from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from daydream.agent import run_agent
from daydream.backends import AgentEvent, ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent, TurnEndEvent
from daydream.extensions.registry import Registry
from daydream.observability.config import ObservabilityConfig
from daydream.observability.runtime import trace_run
from daydream.prompt_budget import SanctionedInputUnavailable, inline_context_file, prepare_sanctioned_inputs
from daydream.review_budget import (
    BudgetLedger,
    ReviewLimits,
    review_deadline_scope,
    review_limits_for_scope,
    review_scale_for_scope,
)
from daydream.review_evidence import FinalizationContext, ReviewEvidence
from daydream.run_context import InteractionPolicy, RunContext
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_clock import FakeClock

SCHEMA = {"type": "object", "properties": {"findings": {"type": "array", "items": {"type": "string"}}},
    "required": ["findings"], "additionalProperties": False,
}
FINDINGS = {"findings": ["src.py:2 divides by zero for an empty batch"]}

def test_large_diff_expands_review_allowances_without_changing_small_reviews() -> None:
    small = "diff --git a/a.py b/a.py\n" + "+line\n" * 100
    large = "diff --git a/a.py b/a.py\n" + "+line\n" * 6_500
    with review_deadline_scope(2700, diff=large, scale_deadline=True):
        assert review_scale_for_scope() > 1
        limits = review_limits_for_scope(ReviewLimits())
        assert limits.investigation_s > 480
        assert limits.tool_calls > 120
        # Batch scaling takes the larger multiplier, never the product: tier 6 already covers 12 batches.
        assert review_limits_for_scope(ReviewLimits(), stages=12) == limits
        assert review_limits_for_scope(ReviewLimits(), stages=40) == ReviewLimits(
            investigation_s=4800, finalization_s=1200, tool_calls=1200)
    with review_deadline_scope(2700, diff=small, scale_deadline=True):
        assert review_limits_for_scope(ReviewLimits()) == ReviewLimits()
        assert review_limits_for_scope(ReviewLimits(), stages=4) == ReviewLimits()
        # Partial allowances round up: five batches need a second one.
        assert review_limits_for_scope(ReviewLimits(), stages=5) == ReviewLimits(
            investigation_s=960, finalization_s=240, tool_calls=240)

async def test_checkpoint_survives_hung_stream_without_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock().install(monkeypatch)
    async def responder(*args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        yield TextEvent(text='{"findings": ["confirmed defect"]}')
        yield TurnEndEvent()
        yield TextEvent(text="unfinished speculation")
        # Spend the host deadline only after the checkpoint is admitted; the
        # real AnyIO watchdog must still cancel this eventless hung stream.
        clock.advance(0.02)
        await anyio.sleep_forever()
    backend = ScriptedBackend(responder=responder)
    with anyio.fail_after(2):
        result, _, reason = await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
            run_context=RunContext(InteractionPolicy(quiet=True)),
        review_limits=ReviewLimits(investigation_s=0.02, finalization_s=0.02, tool_calls=5),
        )
    assert result == {"findings": ["confirmed defect"]}
    assert reason == "wall_budget_exceeded"
    assert backend.call_count == 1
    assert backend.cancel_calls == 0

async def _hung_finalizer_is_bounded_without_canceling_sibling(tmp_path: Path) -> None:
    finalizer_started = anyio.Event()
    sibling_finished = anyio.Event()
    results = {}

    async def responder(*args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if args[1] == "sibling":
            await finalizer_started.wait()
            yield TextEvent(text="sibling survived")
            sibling_finished.set()
        elif "INVESTIGATION HAS ENDED" in args[1]:
            finalizer_started.set()
            await anyio.sleep_forever()
        else:
            yield ToolStartEvent(id="extra", name="read", input={"path": "src.py"})

    backend = ScriptedBackend(responder=responder)
    exporter = InMemorySpanExporter()
    registry = Registry()
    registry.register_trace_exporter("memory", lambda _: exporter)

    async def sibling() -> None:
        results["sibling"] = await run_agent(backend, tmp_path, "sibling", phase=DaydreamPhase.DEEP)

    async with trace_run(ObservabilityConfig(destinations=("memory",)), registry, flow="review") as run:
        with anyio.fail_after(2):
            async with anyio.create_task_group() as group:
                group.start_soon(sibling)
                results["bounded"] = await run_agent(
                    backend, tmp_path, "review", phase=DaydreamPhase.DEEP, review_limits=ReviewLimits(10, 0.02, 0),
                )
        run.finish(0)
    assert sibling_finished.is_set()
    assert results["bounded"][2] == "tool_call_budget_exceeded"
    # Both stops reach the trace: the investigation's as the agent span's record, the
    # timed-out finalization turn's as an event on the same span.
    (bounded,) = [span for span in exporter.get_finished_spans()
                  if "daydream.budget.dimension" in (span.attributes or {})]
    assert (bounded.attributes or {})["daydream.budget.dimension"] == "tool_calls"
    assert [(event.attributes or {})["daydream.budget.dimension"]
            for event in bounded.events if event.name == "daydream.budget.stop"] == ["time"]
    assert results["sibling"][0] == "sibling survived"
    assert backend.cancel_calls == 0

async def test_retry_discards_failed_attempt_checkpoint_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock().install(monkeypatch)
    class RetryableError(RuntimeError):
        retryable = True
    backend = ScriptedBackend(script=[[ToolStartEvent(id="old", name="read", input={"path": "old.py"}),
        ToolResultEvent(id="old", output="FAILED_ATTEMPT_EVIDENCE", is_error=False),
        TextEvent(text='{"findings": ["FAILED_ATTEMPT_FINDING"]}'), TurnEndEvent(), RetryableError("retry"),
    ], [ToolStartEvent(id="new", name="read", input={"path": "src.py"}),
        ToolResultEvent(id="new", output="CURRENT_EVIDENCE", is_error=False),
        ToolStartEvent(id="extra", name="read", input={"path": "other.py"}),
    ], [ResultEvent(structured_output=FINDINGS, continuation=None)]],
        retry_attempts=1, retry_base_delay_s=0, retry_max_delay_s=0,
    )
    result = await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
        budget=BudgetLedger("deep", deadline=clock.monotonic() + 5), review_limits=ReviewLimits(10, 2, 1),
        progress_callback=lambda _: None,
    )
    assert result[0] == FINDINGS
    assert "at most 3 seconds and 1 tool calls" in backend.prompts[0]
    assert "FAILED_ATTEMPT" not in backend.last_prompt
    assert "CURRENT_EVIDENCE" in backend.last_prompt

async def test_finalization_revalidates_inputs_and_propagates_capture_failure(tmp_path: Path) -> None:

    artifact = tmp_path / "intent.md"
    artifact.write_text("captured")
    async def responder(*args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        artifact.write_text("changed")
        yield ToolStartEvent(id="extra", name="read", input={"path": "src.py"})
    backend = ScriptedBackend(responder=responder)
    prepared = prepare_sanctioned_inputs(backend, tmp_path, {"intent": artifact}, read_only=False)
    with pytest.raises(SanctionedInputUnavailable, match="changed"):
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.DEEP, sanctioned_inputs=prepared,
            review_limits=ReviewLimits(10, 2, 0), progress_callback=lambda _: None,
        )
    assert backend.call_count == 1

def test_evidence_retention_is_bounded_deduplicated_and_preserves_associations() -> None:

    evidence = ReviewEvidence(SCHEMA)
    for n in range(500):
        evidence.observe(ToolStartEvent(id=str(n), name="read", input={"path": "src.py"}))
        evidence.observe(ToolResultEvent(id=str(n), output="FOUNDATIONAL_SOURCE", is_error=False,
            exit_code=137, status="cancelled", cancelled=True, truncated=True))
    assert len(evidence.blocks) == 1
    assert evidence.clipped is True
    for n in range(1000):
        evidence.observe(ToolStartEvent(id=str(n), name="search", input={"query": str(n)}))
        evidence.observe(ToolResultEvent(id=str(n), output="x" * 15000, is_error=True))
    evidence.observe(ToolStartEvent(id="pending", name="read", input={"path": "unread.py"}))
    prompt = evidence.finalization_prompt(FinalizationContext(task="Review", supplied_context=(("diff", "DIFF"),)))
    assert "FOUNDATIONAL_SOURCE" in prompt
    assert '"path": "src.py"' in prompt
    assert "error=False" in prompt and "error=True" in prompt
    assert '"exit_code": 137' in prompt
    assert '"status": "cancelled"' in prompt
    assert '"cancelled": true' in prompt
    assert '"truncated": true' in prompt
    assert "clipped=True" in prompt
    assert "[tool output truncated]" in prompt
    assert "DIFF" in prompt
    assert "unread.py" not in prompt
    assert "unmatched tool starts=1" in prompt
    assert evidence.omitted > 0
    assert evidence.retained_bytes <= 48000
    assert len(prompt.encode()) < 51000

def test_exact_path_finalization_capture_is_bounded_and_rejects_changed_identity(tmp_path: Path) -> None:

    diff = tmp_path / "diff.patch"
    diff.write_text("DIFF_BYTES\n" + "é" * 16000)
    backend = ScriptedBackend()
    prepared = prepare_sanctioned_inputs(backend, tmp_path, {"diff": diff}, read_only=True)
    assert prepared.inputs[0].text is None
    text = prepared.finalization_text(backend, tmp_path, True)
    assert "DIFF_BYTES" in text
    assert "input truncated" in text
    assert len(text.encode()) <= 24000
    with pytest.raises(SanctionedInputUnavailable, match="backend changed"):
        prepared.finalization_text(ScriptedBackend(), tmp_path, True)
    with pytest.raises(SanctionedInputUnavailable, match="call mode changed"):
        prepared.finalization_text(backend, tmp_path, False)
    diff.write_text("replacement")
    with pytest.raises(SanctionedInputUnavailable, match="changed"):
        prepared.finalization_text(backend, tmp_path, True)
    # Inline context falls back without truncating large or invalid artifacts.
    artifact = tmp_path / "context.md"
    artifact.write_text("é" * 2049)
    assert inline_context_file(artifact) is None
    artifact.write_bytes(b"\xff")
    assert inline_context_file(artifact) is None
    artifact.unlink()
    assert inline_context_file(artifact) is None

async def test_finalization_is_bounded_and_invocation_local_with_shared_backend(tmp_path: Path) -> None:
    await _hung_finalizer_is_bounded_without_canceling_sibling(tmp_path)

    class ControlledBackend(ScriptedBackend):
        supports_finalization = True
        reasoning_effort = "high"

        async def execute(self, *args: Any, finalization: bool = False, **kwargs: Any) -> AsyncGenerator[AgentEvent]:
            flags.append((args[1], finalization))
            if finalization:
                finalizer_started.set()
                await sibling_finished.wait()
                yield ResultEvent(structured_output={"findings": []}, continuation=None)
            elif args[1] == "sibling":
                await finalizer_started.wait()
                sibling_finished.set()
                yield TextEvent(text="ok")
            else:
                yield ToolStartEvent(id="extra", name="read", input={})

    flags: list[tuple[str, bool]] = []
    finalizer_started = anyio.Event()
    sibling_finished = anyio.Event()
    backend = ControlledBackend(reasoning_effort="high")

    async def sibling() -> None:
        await run_agent(backend, tmp_path, "sibling", phase=DaydreamPhase.DEEP, progress_callback=lambda _: None)

    with anyio.fail_after(2):
        async with anyio.create_task_group() as group:
            group.start_soon(sibling)
            result = await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
                review_limits=ReviewLimits(10, 1, 0), progress_callback=lambda _: None,
            )
    assert result == ({"findings": []}, None, "tool_call_budget_exceeded")
    assert backend.reasoning_effort == "high"
    assert sum(finalization for _, finalization in flags) == 1
    assert ("sibling", False) in flags
