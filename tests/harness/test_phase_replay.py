"""Real Codex parsing across successive agent phases under one recorder.

Only the external subprocess boundary is stubbed. The parse result must pass
through the real backend parser and ``run_agent`` structured-output handling.
"""
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from daydream.agent import run_agent
from daydream.backends.codex import CodexBackend
from daydream.phases import FEEDBACK_SCHEMA
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder
from tests.harness.codex_replay import make_mock_process
from tests.harness.scripts import build_codex_jsonl_for_phase


@pytest.fixture
def recorder(tmp_path: Path) -> Any:
    """A real recorder shared by the review and parse invocations."""
    return TrajectoryRecorder(
        path=tmp_path / "t.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="m",
        session_id="00000000-0000-0000-0000-0000000000ff",
    )


async def test_successive_agent_phases_preserve_structured_parse_output(tmp_path: Path, recorder: Any) -> None:
    rev_script = {"turns": [{"message_id": "r1", "text": "reviewed"}]}
    parse_script = {
        "turns": [{"message_id": "p1", "text": ""}],
        "structured_output": {
            "issues": [
                {
                    "id": 1,
                    "description": "x",
                    "file": "a.py",
                    "line": 1,
                    "confidence": "HIGH",
                    "rationale": "r",
                }
            ]
        },
    }
    processes = [make_mock_process(build_codex_jsonl_for_phase(script)) for script in (rev_script, parse_script)]

    async with recorder:
        with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=processes):
            await run_agent(
                CodexBackend("m"), tmp_path, "go", phase=DaydreamPhase.REVIEW
            )
            par = cast(dict[str, Any], (await run_agent(
                CodexBackend("m"),
                tmp_path,
                "go",
                output_schema=FEEDBACK_SCHEMA,
                phase=DaydreamPhase.PARSE,
            ))[0])

    # The second invocation retains its parsed result after a text-only review.
    assert par["issues"][0]["id"] == 1
