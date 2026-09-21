"""Bounded review investigation retains evidence without claiming completion."""

from collections.abc import AsyncGenerator, AsyncIterator
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


def test_merge_prompt_retains_validated_records_from_incomplete_stacks(tmp_path: Path) -> None:
    from daydream.deep.prompts import build_merge_prompt

    prompt = build_merge_prompt(
        strategy="Merge provided findings.", per_stack_records_paths=[tmp_path / "stack-python-records.json"],
        intent_path=tmp_path / "intent.md", alternatives_path=tmp_path / "alternatives.json",
        dedup_candidates_path=tmp_path / "dedup.json", failed_stacks={"python": "budget exhausted"},
    )
    assert "no records available" not in prompt
    assert "validated partial records" in prompt


async def test_pre_rendered_inputs_become_captured_bytes_during_finalization(tmp_path: Path) -> None:
    from daydream.prompt_budget import prepare_sanctioned_inputs

    artifact = tmp_path / "intent.md"
    artifact.write_text("captured")
    backend = ScriptedBackend(script=[
        [ToolStartEvent(id="extra", name="read", input={"path": "src.py"})],
        [ResultEvent(structured_output=FINDINGS, continuation=None)],
    ])
    prepared = prepare_sanctioned_inputs(backend, tmp_path, {"intent": artifact}, read_only=False)
    result = await run_agent(
        backend, tmp_path, prepared.render_prompt("review"), phase=DaydreamPhase.DEEP,
        output_schema=SCHEMA, sanctioned_inputs=prepared, review_limits=ReviewLimits(10, 2, 0),
        progress_callback=lambda _: None,
    )
    assert result[0] == FINDINGS
    assert backend.prompts[0].count("Sanctioned phase inputs") == 1
    assert backend.prompts[0].endswith(f"- intent: {artifact}")
    assert "captured" in backend.prompts[1]
    assert "read only these exact files" not in backend.prompts[1]
    assert str(artifact) not in backend.prompts[1]


@pytest.mark.parametrize("schema", [None, SCHEMA])
async def test_finalization_contract_excludes_discovery_and_supports_output_kinds(
    tmp_path: Path, schema: dict[str, Any] | None,
) -> None:
    from daydream.review_evidence import FinalizationContext

    backend = ScriptedBackend(script=[
        [TextEvent(text="SPECULATIVE_NOTE"), ToolStartEvent(id="extra", name="read", input={})],
        [ResultEvent(structured_output={"findings": []}, continuation=None)] if schema else
        [TextEvent(text="Incomplete exploration: supplied map covers src.py only.")],
    ])
    result = await run_agent(
        backend, tmp_path, "DISCOVERY_STRATEGY: run tests, search upstream, read every file",
        phase=DaydreamPhase.DEEP, output_schema=schema,
        finalization_context=FinalizationContext(
            task="Produce the requested map" if schema is None else "Review assigned code",
            assigned_files=("src.py",), output_semantics="Only the assigned deliverable",
            supplied_context=(("diff", "+ return 1"),),
        ),
        review_limits=ReviewLimits(10, 2, 0), progress_callback=lambda _: None,
    )
    assert result[2] == "tool_call_budget_exceeded"
    prompt = backend.last_prompt
    assert "DISCOVERY_STRATEGY" not in prompt
    assert "Investigation allowance" not in prompt
    assert "SPECULATIVE_NOTE" not in prompt
    assert "+ return 1" in prompt
    assert "src.py" in prompt
    assert ("plain-text deliverable" if schema is None else '"additionalProperties": false') in prompt
    assert backend.max_turns[-1] is None


async def test_time_shorter_than_reserve_only_dispatches_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock().install(monkeypatch)
    backend = ScriptedBackend(events=[ResultEvent(structured_output={"findings": []}, continuation=None)])
    result = await run_agent(
        backend, tmp_path, "FRESH_INVESTIGATION", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
        deadline=clock.monotonic() + 1, review_limits=ReviewLimits(10, 2, 5),
        progress_callback=lambda _: None,
    )
    assert result == ({"findings": []}, None, "wall_budget_exceeded")
    assert backend.call_count == 1
    assert "FRESH_INVESTIGATION" not in backend.last_prompt
    assert "INVESTIGATION HAS ENDED" in backend.last_prompt


async def test_displayed_allowance_is_clamped_to_absolute_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock().install(monkeypatch)
    backend = ScriptedBackend(events=[ResultEvent(structured_output={"findings": []}, continuation=None)])
    await run_agent(
        backend, tmp_path, "Review", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
        deadline=clock.monotonic() + 5, review_limits=ReviewLimits(10, 2, 5),
        progress_callback=lambda _: None,
    )
    assert "at most 3 seconds and 5 tool calls" in backend.last_prompt


def test_evidence_retention_is_bounded_deduplicated_and_preserves_associations() -> None:
    from daydream.review_evidence import FinalizationContext, ReviewEvidence

    evidence = ReviewEvidence(SCHEMA)
    for n in range(500):
        evidence.observe(ToolStartEvent(id=str(n), name="read", input={"path": "src.py"}))
        evidence.observe(ToolResultEvent(id=str(n), output="FOUNDATIONAL_SOURCE", is_error=False))
    assert len(evidence.blocks) == 1
    for n in range(1000):
        evidence.observe(ToolStartEvent(id=str(n), name="search", input={"query": str(n)}))
        evidence.observe(ToolResultEvent(id=str(n), output="x" * 15000, is_error=True))
    evidence.observe(ToolStartEvent(id="pending", name="read", input={"path": "unread.py"}))
    prompt = evidence.finalization_prompt(FinalizationContext(task="Review", supplied_context=(("diff", "DIFF"),)))
    assert "FOUNDATIONAL_SOURCE" in prompt
    assert '"path": "src.py"' in prompt
    assert "error=False" in prompt and "error=True" in prompt
    assert "[tool output truncated]" in prompt
    assert "DIFF" in prompt
    assert "unread.py" not in prompt
    assert "unmatched tool starts=1" in prompt
    assert evidence.omitted > 0
    assert evidence.retained_bytes <= 48000
    assert len(prompt.encode()) < 51000


def test_exact_path_finalization_capture_is_bounded_and_rejects_changed_identity(tmp_path: Path) -> None:
    from daydream.prompt_budget import SanctionedInputUnavailable, prepare_sanctioned_inputs

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


async def test_finalization_control_is_invocation_local_with_shared_backend(tmp_path: Path) -> None:
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
            result = await run_agent(
                backend, tmp_path, "review", phase=DaydreamPhase.DEEP, output_schema=SCHEMA,
                review_limits=ReviewLimits(10, 1, 0), progress_callback=lambda _: None,
            )
    assert result == ({"findings": []}, None, "tool_call_budget_exceeded")
    assert backend.reasoning_effort == "high"
    assert sum(finalization for _, finalization in flags) == 1
    assert ("sibling", False) in flags


def test_finalization_preserves_native_tool_failure_and_truncation_metadata() -> None:
    from daydream.review_evidence import FinalizationContext, ReviewEvidence

    evidence = ReviewEvidence(SCHEMA)
    evidence.observe(ToolStartEvent(id="read", name="read", input={"path": "src.py"}))
    evidence.observe(ToolResultEvent(
        id="read", output="short partial output", is_error=False,
        exit_code=137, status="cancelled", cancelled=True, truncated=True,
    ))
    prompt = evidence.finalization_prompt(FinalizationContext(task="Review"))
    assert '"exit_code": 137' in prompt
    assert '"status": "cancelled"' in prompt
    assert '"cancelled": true' in prompt
    assert '"truncated": true' in prompt
    assert "clipped=True" in prompt


def test_explicit_capture_priority_retains_task_inputs_before_large_advisories(tmp_path: Path) -> None:
    from daydream.prompt_budget import prepare_sanctioned_inputs

    paths = {}
    for label, text in {
        "alternatives": "A" * 12000,
        "dedup-candidates": "B" * 12000,
        "diff": "REQUIRED_DIFF",
        "stack-records-000": "REQUIRED_MERGE_RECORDS",
    }.items():
        paths[label] = tmp_path / label
        paths[label].write_text(text)
    backend = ScriptedBackend()
    prepared = prepare_sanctioned_inputs(backend, tmp_path, paths, read_only=False)
    text = prepared.finalization_text(
        backend, tmp_path, False, input_priority=("diff", "stack-records-000"),
    )
    assert "REQUIRED_DIFF" in text
    assert "REQUIRED_MERGE_RECORDS" in text
    assert text.index("REQUIRED_MERGE_RECORDS") < text.index("Input 'alternatives'")
    assert len(text.encode()) <= 24000
    assert "truncated" in text
