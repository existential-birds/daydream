"""Bounded review policy reaches Pi's effective system prompt per invocation."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from daydream.agent import run_agent
from daydream.backends.pi import PiBackend
from daydream.prompts.grounding import REVIEW_STOPPING_GUIDANCE
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.trajectory import DaydreamPhase
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture


async def test_bounded_review_system_contract_is_exact_and_does_not_leak_to_fix(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []
    system_prompts: list[str] = []
    async def spawn(*args: str, **kwargs: Any) -> Any:
        commands.append(args)
        value = args[args.index("--append-system-prompt") + 1]
        system_prompts.append(Path(value).read_text() if value.startswith("/") else value)
        return make_mock_process_from_fixture("simple_text.jsonl")
    backend = PiBackend(model="fixture-model", reasoning_effort="high")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=spawn):
        await run_agent(backend, tmp_path, "Review the assigned files", phase=DaydreamPhase.DEEP,
            review_limits=ReviewLimits(17, 3, 3), progress_callback=lambda _: None,
        )
        await run_agent(backend, tmp_path, "Implement the requested fix", phase=DaydreamPhase.FIX,
                        progress_callback=lambda _: None)
    review_system, fix_system = system_prompts
    assert not Path(commands[0][commands[0].index("--append-system-prompt") + 1]).exists()
    assert "at most 17 seconds and 3 tool calls" in review_system
    assert REVIEW_STOPPING_GUIDANCE in review_system
    assert "Closed candidate decisions stay closed" not in review_system
    assert "repository-scoped" in review_system
    assert "Search before you read" not in review_system
    assert "typically 50" not in review_system
    assert REVIEW_STOPPING_GUIDANCE not in fix_system
    assert "Investigation allowance" not in fix_system
    assert all(args[args.index("--thinking") + 1] == "high" for args in commands)


async def test_staged_pi_retry_and_fresh_stage_keep_exact_host_allowance(tmp_path: Path) -> None:
    """Real Pi request construction carries charged retry spend into fresh calls."""
    system_prompts: list[str] = []

    async def spawn(*args: str, **kwargs: Any) -> Any:
        value = args[args.index("--append-system-prompt") + 1]
        system_prompts.append(Path(value).read_text())
        if len(system_prompts) == 1:
            return make_mock_process([json.dumps(event) for event in [
                {"type": "session", "sessionId": "failed-review"},
                {"type": "tool_execution_start", "toolCallId": "read-one", "toolName": "read",
                 "args": {"path": "api.py"}},
                {"type": "tool_execution_start", "toolCallId": "read-two", "toolName": "read",
                 "args": {"path": "App.tsx"}},
                {"type": "turn_end", "message": {"role": "assistant", "content": [],
                 "stopReason": "error", "errorMessage": "503 Service Unavailable"}},
            ]])
        return make_mock_process_from_fixture("simple_text.jsonl")

    backend = PiBackend(model="fixture-model")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 0
    budget = ReviewInvestigationBudget.from_limits(ReviewLimits(17, 3, 4))
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=spawn):
        first = await run_agent(backend, tmp_path, "First pass", phase=DaydreamPhase.DEEP,
                                investigation_budget=budget, tool_call_budget=4, progress_callback=lambda _: None)
        second = await run_agent(backend, tmp_path, "Structural integration", phase=DaydreamPhase.DEEP,
                                 investigation_budget=budget, tool_call_budget=4, progress_callback=lambda _: None)
    assert first[2] is None and second[2] is None
    assert budget.observed_tool_starts == 2 and budget.remaining_tool_calls == 2
    initial, retry, integration = system_prompts
    assert "4 tool calls this stage" in initial
    assert "At most 2 tool calls remain this stage" in retry
    assert "2 remain for this reviewer after 2 observed starts" in retry
    assert "2 tool calls this stage" in integration
    assert "remaining reviewer allowance: 2" in integration
    assert all(REVIEW_STOPPING_GUIDANCE in instructions for instructions in system_prompts)
    for instructions in system_prompts:
        assert instructions.index("Staged review contract:") > instructions.index(REVIEW_STOPPING_GUIDANCE)
        assert "Closed candidate decisions stay closed" in instructions
        assert "Report contradictions by their existing candidate IDs" in instructions
        assert "Return exactly the assigned target or triage candidate IDs" in instructions
        assert "Do not produce a terminal findings serializer" in instructions
