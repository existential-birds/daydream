"""Drive the real Claude backend, agent and recorder through the SDK boundary.

Rendered comments must preserve SDK model identity and per-phase usage/cost.
Per-step metrics matter: correct final totals alone cannot verify the renderer."""
from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends.claude import ClaudeBackend
from daydream.pr_comment_renderer import render_run_info_block
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    patch_claude_sdk,
    scripted_client,
)
from tests.harness.trajectory import make_recorder

# Mirror runner.py: ``agent_model_name`` is stamped per-step, not at recorder init.
_make_recorder = partial(make_recorder, run_flow=DaydreamRunFlow.TTT, agent_model_name="claude")


@pytest.fixture
def patch_sdk(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch the SDK boundary; each call re-routes the backend to a fresh stream."""

    def _patch(messages: list[Any]) -> None:
        patch_claude_sdk(monkeypatch, scripted_client(messages))

    return _patch


def _line(markdown: str, prefix: str) -> str:
    for line in markdown.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"No {prefix!r} line in markdown:\n{markdown}")


def _phase_row(markdown: str, label: str) -> str:
    needle = f"| {label} |"
    for line in markdown.splitlines():
        if line.startswith(needle):
            return line
    raise AssertionError(f"No phase row {label!r} in markdown:\n{markdown}")


FIXTURE_MODEL_ID = "fixture-sdk-model-id"


def _stream(text: str, cost: float, *, input_tokens: int, output_tokens: int, cache_read_input_tokens: int,
) -> list[Any]:
    """One assistant turn plus its result usage; the shape every test streams."""
    return [MockAssistantMessage(content=[MockTextBlock(text=text)], model=FIXTURE_MODEL_ID,),
        MockResultMessage(total_cost_usd=cost,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            },
        ),
    ]

async def test_render_uses_real_sdk_model_id_not_backend_alias(tmp_path: Path, patch_sdk: Any) -> None:
    """Carry AssistantMessage.model through to the rendered model label."""
    patch_sdk(_stream("reviewing the code", 0.42, input_tokens=1000, output_tokens=200, cache_read_input_tokens=800))

    recorder = _make_recorder(tmp_path)
    target_path = recorder.path
    async with recorder:
        backend = ClaudeBackend(model="opus")
        await run_agent(backend, tmp_path, "review please", phase=DaydreamPhase.REVIEW)

    assert target_path.exists(), "Trajectory file should have been written"
    markdown = render_run_info_block([target_path])

    model_line = _line(markdown, "- **Model:**")
    assert FIXTURE_MODEL_ID in model_line, (f"Bug A: expected real SDK model id {FIXTURE_MODEL_ID!r} in Model line, "
        f"got: {model_line!r}\n\nFull markdown:\n{markdown}"
    )
    # Exact-line check: real id begins with "claude-", so a substring match on
    # "claude" alone would be unsafe.
    assert model_line.strip() != "- **Model:** claude", (
        f"Bug A: Model line is the backend alias instead of the SDK model id: {model_line!r}"
    )

async def test_render_shows_real_cost_and_tokens_from_sdk_usage(tmp_path: Path, patch_sdk: Any) -> None:
    """Per-step metrics must reach the renderer when usage arrives on ResultMessage."""
    review_messages = _stream("reviewing", 0.30, input_tokens=5000, output_tokens=600, cache_read_input_tokens=2000)
    fix_messages = _stream("fixing", 0.15, input_tokens=2500, output_tokens=400, cache_read_input_tokens=1000)

    recorder = _make_recorder(tmp_path)
    target_path = recorder.path
    async with recorder:
        # Reapplying patch_sdk between phases re-routes ClaudeSDKClient so the
        # second run_agent() picks up the new stream.
        patch_sdk(review_messages)
        backend = ClaudeBackend(model="opus")
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

        patch_sdk(fix_messages)
        backend = ClaudeBackend(model="opus")
        await run_agent(backend, tmp_path, "fix", phase=DaydreamPhase.FIX)

    assert target_path.exists()
    traj_data = json.loads(target_path.read_text())
    # Surface per-step metrics for failure diagnostics.
    agent_steps = [s for s in traj_data["steps"] if s["source"] == "agent"]
    per_step_metrics = [s.get("metrics") for s in agent_steps]

    markdown = render_run_info_block([target_path])

    cost_line = _line(markdown, "- **Cost:**")
    tokens_line = _line(markdown, "- **Tokens:**")

    assert "$0.00" not in cost_line, (
        f"Bug B/C: rollup cost is $0.00 — per-step metrics never landed.\n"
        f"  per-step metrics: {per_step_metrics}\n"
        f"  cost line: {cost_line!r}\n"
        f"  full markdown:\n{markdown}"
    )
    # Tokens must be non-zero in/out. With Bug B/C totals collapse to "0 in → 0 out";
    # word-boundary regex avoids matching e.g. "500 in".
    assert not re.search(r"(?<!\d)0 in\b", tokens_line), (
        f"Bug B/C: rollup input tokens are zero — per-step metrics never landed.\n"
        f"  per-step metrics: {per_step_metrics}\n"
        f"  tokens line: {tokens_line!r}\n"
        f"  full markdown:\n{markdown}"
    )
    assert not re.search(r"(?<!\d)0 out\b", tokens_line), (
        f"Bug B/C: rollup output tokens are zero — per-step metrics never landed.\n"
        f"  per-step metrics: {per_step_metrics}\n"
        f"  tokens line: {tokens_line!r}\n"
        f"  full markdown:\n{markdown}"
    )

    # Per-phase rows must carry non-zero cost + tokens for both Review and Fix.
    review_row = _phase_row(markdown, "Review")
    fix_row = _phase_row(markdown, "Fix")
    assert "$0.00" not in review_row, (
        f"Bug B/C: Review row shows $0.00 cost.\n  row: {review_row!r}\n"
        f"  per-step metrics: {per_step_metrics}"
    )
    assert "$0.00" not in fix_row, (
        f"Bug B/C: Fix row shows $0.00 cost.\n  row: {fix_row!r}\n"
        f"  per-step metrics: {per_step_metrics}"
    )
    # 6-column layout: Phase | Model | Tools | Input (cached) | Output | Cost.
    # Spot-check Output != 0 as a canary (fixture counts are >= 100).
    for label, row in (("Review", review_row), ("Fix", fix_row)):
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[4] != "0", (
            f"Bug B/C: {label} row has Output=0.\n  row: {row!r}\n"
            f"  per-step metrics: {per_step_metrics}"
        )
        # Cache-percentage parenthetical (e.g. "5,000 (40%)") appears when cached > 0.
        assert re.search(r"\(\d+%\)", cells[3]), (
            f"Bug B/C: {label} row missing cache-percentage parenthetical.\n"
            f"  input cell: {cells[3]!r}\n  row: {row!r}\n"
            f"  per-step metrics: {per_step_metrics}"
        )

async def test_per_phase_rollup_distinguishes_phases(tmp_path: Path, patch_sdk: Any) -> None:
    """Separate calls must render distinct phase rows with their own usage and cost."""
    review_messages = _stream("reviewing", 0.20, input_tokens=4000, output_tokens=500, cache_read_input_tokens=1500)
    parse_messages = _stream("parsed", 0.05, input_tokens=1000, output_tokens=100, cache_read_input_tokens=500)

    recorder = _make_recorder(tmp_path)
    target_path = recorder.path
    async with recorder:
        patch_sdk(review_messages)
        backend = ClaudeBackend(model="opus")
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)

        patch_sdk(parse_messages)
        backend = ClaudeBackend(model="opus")
        await run_agent(backend, tmp_path, "parse", phase=DaydreamPhase.PARSE)

    assert target_path.exists()
    traj_data = json.loads(target_path.read_text())
    agent_steps = [s for s in traj_data["steps"] if s["source"] == "agent"]
    per_step_metrics = [s.get("metrics") for s in agent_steps]

    markdown = render_run_info_block([target_path])

    review_row = _phase_row(markdown, "Review")
    parse_row = _phase_row(markdown, "Parse Feedback")

    # prompt_tokens is the total input (uncached remainder + cache read folded in).
    # Review: 4,000 + 1,500 read = 5,500 → row should display "5,500".
    # Parse Feedback: 1,000 + 500 read = 1,500 → row should display "1,500".
    assert "5,500" in review_row, (
        f"Bug B/C: Review row missing real input tokens (expected 5,500 total).\n"
        f"  row: {review_row!r}\n  per-step metrics: {per_step_metrics}"
    )
    assert "1,500" in parse_row, (
        f"Bug B/C: Parse Feedback row missing real input tokens (expected 1,500 total).\n"
        f"  row: {parse_row!r}\n  per-step metrics: {per_step_metrics}"
    )
    # And costs should differ — Review $0.20 vs Parse $0.05.
    assert "$0.20" in review_row, (
        f"Bug B/C: Review row missing real cost ($0.20).\n"
        f"  row: {review_row!r}\n  per-step metrics: {per_step_metrics}"
    )
    assert "$0.05" in parse_row, (
        f"Bug B/C: Parse Feedback row missing real cost ($0.05).\n"
        f"  row: {parse_row!r}\n  per-step metrics: {per_step_metrics}"
    )
