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


async def test_pi_review_system_is_scoped_and_preserves_retry_and_stage_spend(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []
    system_prompts: list[str] = []
    async def spawn(*args: str, **kwargs: Any) -> Any:
        commands.append(args)
        value = args[args.index("--append-system-prompt") + 1]
        system_prompts.append(Path(value).read_text() if value.startswith("/") else value)
        if len(commands) == 3:
            return make_mock_process([json.dumps(event) for event in [
                {"type": "session", "sessionId": "failed-review"},
                *[{"type": "tool_execution_start", "toolCallId": path, "toolName": "read", "args": {"path": path}}
                  for path in ("api.py", "App.tsx")],
                {"type": "turn_end", "message": {"role": "assistant", "content": [],
                 "stopReason": "error", "errorMessage": "503 Service Unavailable"}},
            ]])
        return make_mock_process_from_fixture("simple_text.jsonl")
    backend = PiBackend(model="fixture-model", reasoning_effort="high")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 0
    budget = ReviewInvestigationBudget.from_limits(ReviewLimits(17, 3, 4))
    invocations: list[tuple[DaydreamPhase, dict[str, Any]]] = [
        (DaydreamPhase.DEEP, {"review_limits": ReviewLimits(17, 3, 3)}),
        (DaydreamPhase.FIX, {}),
        *[(DaydreamPhase.DEEP, {"investigation_budget": budget, "tool_call_budget": 4}) for _ in range(2)],
    ]
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=spawn):
        for phase, options in invocations:
            assert (await run_agent(backend, tmp_path, "Assigned work", phase=phase,
                                    progress_callback=lambda _: None, **options))[2] is None
    review_system, fix_system, initial, retry, integration = system_prompts
    paths = [args[args.index("--append-system-prompt") + 1] for args in commands]
    assert all(not Path(path).exists() for path in paths if path.startswith("/"))
    assert all(args[args.index("--thinking") + 1] == "high" for args in commands)
    assert "at most 17 seconds and 3 tool calls" in review_system
    assert REVIEW_STOPPING_GUIDANCE in review_system
    assert "Closed candidate decisions stay closed" not in review_system
    assert "repository-scoped" in review_system
    assert "Search before you read" not in review_system
    assert "typically 50" not in review_system
    assert REVIEW_STOPPING_GUIDANCE not in fix_system
    assert "Investigation allowance" not in fix_system
    assert budget.observed_tool_starts == 2 and budget.remaining_tool_calls == 2
    assert "4 remaining cumulative tool starts" in initial
    assert "Hard remaining cumulative reviewer allowance: 2 tool calls" in retry
    assert "2 remain for this reviewer after 2 observed starts" in retry
    assert "2 remaining cumulative tool starts" in integration
    for instructions in (initial, retry, integration):
        assert REVIEW_STOPPING_GUIDANCE not in instructions
        assert "Closed decisions stay closed" in instructions
        assert "Return exactly assigned target or triage candidate IDs" in instructions
        assert "host publishes terminal findings" in instructions
