"""Bounded review policy reaches Pi's effective system prompt per invocation."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from daydream.agent import run_agent
from daydream.backends import ContinuationToken
from daydream.backends.pi import PiBackend
from daydream.phases.review_prompts import build_review_stage_system_instruction
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
    access_guide = json.dumps({'supporting_catalog': {'path': str(tmp_path / 'catalog.json'),
                                                   'status': 'partial', 'omitted_part_count': 1}})
    invocations: list[tuple[DaydreamPhase, dict[str, Any]]] = [
        (DaydreamPhase.DEEP, {"review_limits": ReviewLimits(17, 3, 3)}),
        (DaydreamPhase.FIX, {}),
        *[(DaydreamPhase.DEEP, {
            "investigation_budget": budget,
            "tool_call_budget": 4,
            "review_system_instructions": build_review_stage_system_instruction({
                "stage": stage,
                "assigned_target_ids": [target],
                "access_guide": access_guide,
            }),
            **({'continuation': ContinuationToken('pi', {'session_id': 'retained-navigation'})}
               if stage == 'integration' else {}),
        }) for stage, target in (("first_pass", "target-1"), ("integration", "whole-change"))],
    ]
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=spawn):
        for phase, options in invocations:
            assert (await run_agent(backend, tmp_path, "Assigned work", phase=phase,
                                    progress_callback=lambda _: None, **options))[2] is None
    review_system, fix_system, initial, retry, integration = system_prompts
    paths = [args[args.index("--append-system-prompt") + 1] for args in commands]
    assert all(not Path(path).exists() for path in paths if path.startswith("/"))
    assert all(args[args.index("--thinking") + 1] == "high" for args in commands)
    assert commands[-1][commands[-1].index('--session-id') + 1] == 'retained-navigation'
    assert "at most 17 seconds and 3 tool calls" in review_system
    assert REVIEW_STOPPING_GUIDANCE in review_system
    assert "planted" not in review_system
    assert "hidden evaluation" not in review_system
    assert "Reopen a rejected candidate only when new evidence" in review_system
    assert "an empty findings result is valid" in review_system
    assert "Closed candidate decisions stay closed" not in review_system
    assert "repository-scoped" in review_system
    assert "Search before you read" not in review_system
    assert "typically 50" not in review_system
    assert REVIEW_STOPPING_GUIDANCE not in fix_system
    assert "Investigation allowance" not in fix_system
    assert budget.observed_tool_starts == 2 and budget.remaining_tool_calls == 2
    assert "Reviewer dispatch-time snapshot:" in initial
    assert "current invocation allocation: 4 tool calls" in initial
    assert "Retry dispatch-time snapshot (supersedes the earlier invocation allocation):" in retry
    assert "current invocation allocation: 2 tool calls" in retry
    assert "Failed-attempt calls remain charged; future-stage reserves remain excluded" in retry
    assert "current invocation allocation: 2 tool calls" in integration
    for instructions, objective, target in (
        (initial, "Review assigned changed behavior", "target-1"),
        (retry, "Review assigned changed behavior", "target-1"),
        (integration, "Review whole-change interactions and boundaries", "whole-change"),
    ):
        assignment_line = next(line for line in instructions.splitlines()
                               if line.startswith("Daydream final-submission assignment: "))
        assert json.loads(assignment_line.removeprefix("Daydream final-submission assignment: ")) == {
            "objective": objective, "target_ids": [target], "candidate_ids": [],
        }
        assert 'Persistent exact access guide (supporting metadata): ' + access_guide in instructions
        assert f'"targets": ["{target}"]' in instructions
    for instructions in (initial, retry, integration):
        assert "Cumulative observed starts:" not in instructions
        assert "remaining cumulative tool starts" not in instructions
        assert "at most one final-submission warning per native attempt" in instructions
        assert "not a recurring counter" in instructions
        assert "refreshed" not in instructions
        assert "Unused capacity is expected" in instructions
        assert "submit when those" in instructions
        assert "planted" not in instructions
        assert "hidden evaluation" not in instructions
        assert REVIEW_STOPPING_GUIDANCE not in instructions
        assert "Closed decisions stay closed" in instructions
        assert "Return exactly assigned target or triage candidate IDs" in instructions
        assert "host publishes terminal findings" in instructions
