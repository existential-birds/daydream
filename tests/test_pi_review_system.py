"""Bounded review policy reaches Pi's effective system prompt per invocation."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from daydream.agent import run_agent
from daydream.backends.pi import PiBackend
from daydream.prompts.grounding import REVIEW_STOPPING_GUIDANCE
from daydream.review_budget import ReviewLimits
from daydream.trajectory import DaydreamPhase
from tests.harness.pi_replay import make_mock_process_from_fixture


async def test_bounded_review_system_contract_is_exact_and_does_not_leak_to_fix(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []

    async def spawn(*args: str, **kwargs: Any) -> Any:
        commands.append(args)
        return make_mock_process_from_fixture("simple_text.jsonl")

    backend = PiBackend(model="fixture-model", reasoning_effort="high")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=spawn):
        await run_agent(
            backend, tmp_path, "Review the assigned files", phase=DaydreamPhase.DEEP,
            review_limits=ReviewLimits(17, 3, 3), progress_callback=lambda _: None,
        )
        await run_agent(backend, tmp_path, "Implement the requested fix", phase=DaydreamPhase.FIX,
                        progress_callback=lambda _: None)

    review_system, fix_system = [args[args.index("--append-system-prompt") + 1] for args in commands]
    assert "at most 17 seconds and 3 tool calls" in review_system
    assert REVIEW_STOPPING_GUIDANCE in review_system
    assert "repository-scoped" in review_system
    assert "Search before you read" not in review_system
    assert "typically 50" not in review_system
    assert REVIEW_STOPPING_GUIDANCE not in fix_system
    assert "Investigation allowance" not in fix_system
    assert all(args[args.index("--thinking") + 1] == "high" for args in commands)
