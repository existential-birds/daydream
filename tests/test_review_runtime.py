"""Bounded review investigation retains evidence without claiming completion."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import AgentEvent, ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent, TurnEndEvent
from daydream.review_budget import ReviewLimits, review_deadline_scope
from daydream.run_context import InteractionPolicy, RunContext
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_clock import FakeClock

SCHEMA = {
    "type": "object", "properties": {"findings": {"type": "array", "items": {"type": "string"}}},
    "required": ["findings"], "additionalProperties": False,
}
FINDINGS = {"findings": ["src.py:2 divides by zero for an empty batch"]}


def test_review_prompts_reuse_small_context_and_request_only_json(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_per_stack_prompt

    intent = tmp_path / "intent.md"
    intent.write_text("Preserve empty-batch semantics.")
    exploration = tmp_path / "exploration"
    exploration.mkdir()
    (exploration / "summary.md").write_text("Use the canonical batch validator.")
    (exploration / "affected_files.md").write_text("src.py -> batch.py")
    prompt = build_per_stack_prompt(
        strategy="Review correctness.", stack_name="python", files=["src.py"],
        diff_path=tmp_path / "diff.patch", intent_path=intent, alternatives_path=tmp_path / "alternatives.json",
        output_path=tmp_path / "review.md", cwd=tmp_path, exploration_dir=exploration,
        include_alternatives=False,
    )
    assert "Preserve empty-batch semantics." in prompt
    assert "Use the canonical batch validator." in prompt
    assert "src.py -> batch.py" in prompt
    assert "Write your full review" not in prompt
    assert "verdict line" not in prompt
    assert "JSON" in prompt


async def test_tool_budget_finalizes_from_completed_evidence(tmp_path: Path) -> None:
    backend = ScriptedBackend(script=[[
        ToolStartEvent(id="read", name="read", input={"path": "src.py"}),
        ToolResultEvent(id="read", output="1 def mean(xs):\n2     return sum(xs) / len(xs)", is_error=False),
        ToolStartEvent(id="extra", name="bash", input={"command": "unneeded investigation"}),
    ], [ResultEvent(structured_output=FINDINGS, continuation=None)]])
    result, _, reason = await run_agent(
        backend, tmp_path, "review src.py", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
        run_context=RunContext(InteractionPolicy(quiet=True)),
        review_limits=ReviewLimits(investigation_s=10, finalization_s=2, tool_calls=1),
    )
    assert result == FINDINGS
    assert reason == "tool_call_budget_exceeded"
    assert "return sum(xs) / len(xs)" in backend.last_prompt
    assert "src.py" in backend.last_prompt
    assert backend.continuations == [None, None]
    assert backend.cancel_calls == 0


async def test_checkpoint_survives_hung_stream_without_finalizer(tmp_path: Path) -> None:
    async def responder(*args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        yield TextEvent(text='{"findings": ["confirmed defect"]}')
        yield TurnEndEvent()
        yield TextEvent(text="unfinished speculation")
        await anyio.sleep_forever()

    backend = ScriptedBackend(responder=responder)
    with anyio.fail_after(2):
        result, _, reason = await run_agent(
            backend, tmp_path, "review", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
            run_context=RunContext(InteractionPolicy(quiet=True)),
        review_limits=ReviewLimits(investigation_s=0.02, finalization_s=0.02, tool_calls=5),
        )
    assert result == {"findings": ["confirmed defect"]}
    assert reason == "wall_budget_exceeded"
    assert backend.call_count == 1
    assert backend.cancel_calls == 0


async def test_invalid_finalization_is_not_published_as_findings(tmp_path: Path) -> None:
    backend = ScriptedBackend(script=[
        [ToolStartEvent(id="extra", name="read", input={"path": "src.py"})],
        [ResultEvent(structured_output={"findings": [42]}, continuation=None)],
    ])
    result, _, reason = await run_agent(
        backend, tmp_path, "review", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
        run_context=RunContext(InteractionPolicy(quiet=True)),
        review_limits=ReviewLimits(investigation_s=10, finalization_s=2, tool_calls=0),
    )
    assert result == ""
    assert reason == "tool_call_budget_exceeded"


async def test_spent_shared_deadline_skips_queued_review_but_not_fix(tmp_path: Path) -> None:
    backend = ScriptedBackend(events=[TextEvent(text="fixed")])
    with review_deadline_scope(0):
        result = await run_agent(
            backend, tmp_path, "review", phase=DaydreamPhase.DEEP,
            run_context=RunContext(InteractionPolicy(quiet=True)),
        review_limits=ReviewLimits(),
        )
        assert result == ("", None, "wall_budget_exceeded")
        assert backend.call_count == 0
        assert (await run_agent(backend, tmp_path, "fix", phase=DaydreamPhase.FIX))[0] == "fixed"
    assert (await run_agent(backend, tmp_path, "new review", phase=DaydreamPhase.DEEP))[2] is None


async def test_comparable_runaway_workload_retains_findings_with_less_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same 200-read workload, ten virtual seconds per round trip; no real model claims."""
    measured = []
    for limits in (None, ReviewLimits()):
        clock = FakeClock().install(monkeypatch)
        calls = 0

        async def responder(*args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            nonlocal calls
            if "INVESTIGATION HAS ENDED" in args[1]:
                clock.advance(10)
                yield ResultEvent(structured_output=FINDINGS, continuation=None)
                return
            for n in range(200):
                calls += 1
                clock.advance(10)
                yield ToolStartEvent(id=str(n), name="read", input={"path": "src.py"})
                yield ToolResultEvent(id=str(n), output="2 return sum(xs)/len(xs)", is_error=False)
            yield ResultEvent(structured_output=FINDINGS, continuation=None)

        backend = ScriptedBackend(responder=responder)
        result, _, reason = await run_agent(
            backend, tmp_path, "review", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
            wall_budget_s=3600, review_limits=limits, progress_callback=lambda _: None,
        )
        assert result == FINDINGS
        measured.append((clock.monotonic(), calls, reason))
    assert measured == [(2000, 200, None), (490, 48, "wall_budget_exceeded")]


async def test_hung_finalizer_is_bounded_without_canceling_sibling(tmp_path: Path) -> None:
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

    async def sibling() -> None:
        results["sibling"] = await run_agent(backend, tmp_path, "sibling", phase=DaydreamPhase.DEEP)

    with anyio.fail_after(2):
        async with anyio.create_task_group() as group:
            group.start_soon(sibling)
            results["bounded"] = await run_agent(
                backend, tmp_path, "review", phase=DaydreamPhase.DEEP,
                review_limits=ReviewLimits(10, 0.02, 0),
            )
    assert sibling_finished.is_set()
    assert results["bounded"][2] == "tool_call_budget_exceeded"
    assert results["sibling"][0] == "sibling survived"
    assert backend.cancel_calls == 0


def test_inline_context_falls_back_without_truncating_large_or_invalid_artifacts(tmp_path: Path) -> None:
    from daydream.prompt_budget import inline_context_file

    artifact = tmp_path / "context.md"
    artifact.write_text("é" * 2049)
    assert inline_context_file(artifact) is None
    artifact.write_bytes(b"\xff")
    assert inline_context_file(artifact) is None
    artifact.unlink()
    assert inline_context_file(artifact) is None


async def test_synthesis_uses_reserved_time_after_discovery_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock().install(monkeypatch)
    backend = ScriptedBackend(events=[ResultEvent(structured_output=FINDINGS, continuation=None)])
    with review_deadline_scope(600):
        clock.advance(490)
        result = await run_agent(
            backend, tmp_path, "queued discovery", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
            review_limits=ReviewLimits(), progress_callback=lambda _: None,
        )
        assert result[2] == "wall_budget_exceeded"
        assert backend.call_count == 0
        result = await run_agent(
            backend, tmp_path, "adjudicate", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
            review_limits=ReviewLimits(30, 10, 4, discovery=False), progress_callback=lambda _: None,
        )
        assert result == (FINDINGS, None, None)


async def test_retry_discards_failed_attempt_checkpoint_and_evidence(tmp_path: Path) -> None:
    class RetryableError(RuntimeError):
        retryable = True

    backend = ScriptedBackend(script=[[
        ToolStartEvent(id="old", name="read", input={"path": "old.py"}),
        ToolResultEvent(id="old", output="FAILED_ATTEMPT_EVIDENCE", is_error=False),
        TextEvent(text='{"findings": ["FAILED_ATTEMPT_FINDING"]}'), TurnEndEvent(), RetryableError("retry"),
    ], [
        ToolStartEvent(id="new", name="read", input={"path": "src.py"}),
        ToolResultEvent(id="new", output="CURRENT_EVIDENCE", is_error=False),
        ToolStartEvent(id="extra", name="read", input={"path": "other.py"}),
    ], [ResultEvent(structured_output=FINDINGS, continuation=None)]],
        retry_attempts=1, retry_base_delay_s=0, retry_max_delay_s=0,
    )
    result = await run_agent(
        backend, tmp_path, "review", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
        review_limits=ReviewLimits(10, 2, 1), progress_callback=lambda _: None,
    )
    assert result[0] == FINDINGS
    assert "FAILED_ATTEMPT" not in backend.last_prompt
    assert "CURRENT_EVIDENCE" in backend.last_prompt


async def test_finalization_revalidates_inputs_and_propagates_capture_failure(tmp_path: Path) -> None:
    from daydream.prompt_budget import SanctionedInputUnavailable, prepare_sanctioned_inputs

    artifact = tmp_path / "intent.md"
    artifact.write_text("captured")

    async def responder(*args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        artifact.write_text("changed")
        yield ToolStartEvent(id="extra", name="read", input={"path": "src.py"})

    backend = ScriptedBackend(responder=responder)
    prepared = prepare_sanctioned_inputs(backend, tmp_path, {"intent": artifact}, read_only=False)
    with pytest.raises(SanctionedInputUnavailable, match="changed"):
        await run_agent(
            backend, tmp_path, "review", phase=DaydreamPhase.DEEP, sanctioned_inputs=prepared,
            review_limits=ReviewLimits(10, 2, 0), progress_callback=lambda _: None,
        )
    assert backend.call_count == 1
