"""Tests for the phase-keyed replay context managers (the harness core).

The unit-level proof here is that the Codex subprocess ``side_effect`` keys on
the firing phase read from ``TrajectoryRecorder.current_phase()``: two phases
fire under one open recorder, and each ``run_agent`` call observes ITS phase's
fixture. The PARSE phase's structured-output ``id == 1`` (produced by the REAL
backend parser) proves the keying — a mis-keyed factory would serve REVIEW's
lines and the structured output would be absent.
"""

import pytest

from daydream.agent import run_agent
from daydream.backends.codex import CodexBackend
from daydream.phases import FEEDBACK_SCHEMA
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder
from tests.harness.phase_replay import codex_subprocess_for_phases


@pytest.fixture
def recorder(tmp_path):
    """A real, un-entered ``TrajectoryRecorder`` (mirrors the accessor test).

    The test enters it with ``async with`` so ``get_current_recorder()`` —
    which the replay factory reads — returns this instance while the phase
    invocations fire.
    """
    return TrajectoryRecorder(
        path=tmp_path / "t.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="m",
        session_id="00000000-0000-0000-0000-0000000000ff",
    )


async def test_side_effect_serves_per_phase(tmp_path, recorder):
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
    phase_scripts = {DaydreamPhase.REVIEW: rev_script, DaydreamPhase.PARSE: parse_script}

    async with recorder:
        with codex_subprocess_for_phases(phase_scripts):
            await run_agent(
                CodexBackend("m"), tmp_path, "go", phase=DaydreamPhase.REVIEW
            )
            par, _, _ = await run_agent(
                CodexBackend("m"),
                tmp_path,
                "go",
                output_schema=FEEDBACK_SCHEMA,
                phase=DaydreamPhase.PARSE,
            )

    # PARSE fixture's structured output — proves the factory keyed on the
    # firing phase, not REVIEW's (text-only) lines.
    assert par["issues"][0]["id"] == 1
