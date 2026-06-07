"""Real-path Codex test — closes #151.

Drives the genuine ``CodexBackend`` parser end-to-end through ``runner.run`` on
a single-pass shallow run. The Codex subprocess boundary is replayed per phase
(REVIEW/PARSE/FIX/TEST) via the harness, but the *real* backend, the *real*
JSONL parser, and the *real* shallow loop all execute — only the external
``create_subprocess_exec`` boundary is stubbed.

Non-trivial shape (the point of #151): four distinct phases fire in sequence; the
REVIEW phase carries a tool-call span; the PARSE phase returns one real
``FEEDBACK_SCHEMA`` issue extracted by the genuine parser, which drives the FIX
phase. Assertions pin OBSERVABLE outcomes only — exit code and the on-disk ATIF
trajectory (the REVIEW tool-call span and a fix-phase step) — never the
agent-written ``.review-output.md`` (under replay the parse phase consumes
structured output, so that file's presence is irrelevant).
"""

import json
from pathlib import Path

import pytest

from daydream.runner import RunConfig, run
from daydream.trajectory import DaydreamPhase
from tests.harness.phase_replay import replay_through_runner

# Per-phase canonical scripts (synthesized via the Task 4 builders through
# render_codex inside the harness). One response per firing phase:
#   REVIEW  — text + one mcp_tool_call span (the non-trivial shape).
#   PARSE   — structured_output satisfying FEEDBACK_SCHEMA (one real issue),
#             so the real parser extracts it and the fix phase fires.
#   FIX     — text "fixed".
#   TEST    — "All 1 tests passed. 0 failed." so detect_test_success() -> True.
PHASE_SCRIPTS: dict[DaydreamPhase, dict] = {
    DaydreamPhase.REVIEW: {
        "turns": [
            {
                "message_id": "rev1",
                "text": "Reviewed main.py; found one issue.",
                "tool_calls": [
                    {"id": "call_read_main", "name": "Read", "input": {"path": "main.py"}},
                ],
            },
        ],
        "tool_results": [
            {"id": "call_read_main", "output": "def hello():\n    return 'universe'\n"},
        ],
        "final_usage": {"input_tokens": 100, "output_tokens": 20},
    },
    DaydreamPhase.PARSE: {
        "turns": [{"message_id": "parse1", "text": ""}],
        "structured_output": {
            "issues": [
                {
                    "id": 1,
                    "description": "Add a docstring to hello().",
                    "file": "main.py",
                    "line": 1,
                    "confidence": "HIGH",
                    "rationale": "Public function lacks documentation.",
                },
            ],
        },
        "final_usage": {"input_tokens": 80, "output_tokens": 30},
    },
    DaydreamPhase.FIX: {
        "turns": [{"message_id": "fix1", "text": "fixed"}],
        "final_usage": {"input_tokens": 90, "output_tokens": 10},
    },
    DaydreamPhase.TEST: {
        "turns": [{"message_id": "test1", "text": "All 1 tests passed. 0 failed."}],
        "final_usage": {"input_tokens": 70, "output_tokens": 12},
    },
}


@pytest.mark.asyncio
async def test_codex_realpath_shallow_run(feature_branch_repo: Path, tmp_path: Path) -> None:
    """Single-pass shallow run, real CodexBackend, subprocess replayed per phase.

    Asserts OBSERVABLE outcomes only: exit code 0 and an on-disk ATIF trajectory
    whose steps include the REVIEW tool-call span and a ``daydream_phase == "fix"``
    step (proof the real parser ran and the extracted issue drove the fix). NOT
    ``.review-output.md``.

    ``assume="no"`` declines the post-pass commit gate so the run finishes on the
    four named phases without a real ``git push`` (the commit path is out of
    scope for #151).
    """
    traj = tmp_path / "trajectory.json"
    config = RunConfig(
        target=str(feature_branch_repo),
        skill="python",
        quiet=True,
        cleanup=False,
        shallow=True,
        loop=False,
        backend="codex",
        assume="no",
        trajectory_path=traj,
    )

    with replay_through_runner("codex", PHASE_SCRIPTS):
        exit_code = await run(config)

    assert exit_code == 0

    steps = json.loads(traj.read_text(encoding="utf-8"))["steps"]
    # REVIEW span: the genuine parser emitted a step carrying tool_calls.
    assert any(s.get("tool_calls") for s in steps), "real parser must record the REVIEW tool-call span"
    # The extracted issue drove the FIX phase — daydream_phase lives in step.extra.
    assert any(
        (s.get("extra") or {}).get("daydream_phase") == "fix" for s in steps
    ), "extracted PARSE issue must drive a fix-phase step"
