"""End-to-end PR-comment renderer integration test (Bugs A/B/C).

This test is intentionally red: it drives the FULL daydream code path —
real ``TrajectoryRecorder``, real ``daydream.agent.run_agent``, real
``daydream.backends.claude.ClaudeBackend.execute`` — mocking only at the
``ClaudeSDKClient`` boundary so we exercise the production isinstance
dispatch in ``ClaudeBackend.execute`` and the production event-fold in
``daydream.trajectory.Invocation._dispatch``.

The three bugs the test is designed to expose:

A. ``ClaudeBackend`` never reads ``AssistantMessage.model``. The trajectory's
   per-step ``model_name`` ends up as the backend NAME (``"claude"``) the
   runner stamped at recorder init, never the real SDK model id.

B. ``ClaudeBackend`` looks for ``msg.usage`` on ``AssistantMessage``, but
   the real SDK only carries ``usage`` on ``ResultMessage``. ``MetricsEvent``
   is therefore never emitted per step. ``Step.metrics`` stays ``None``.

C. ``Invocation._dispatch(CostEvent)`` only updates ``_final_totals`` and
   never sets the open step's ``_metrics``. Even when the SDK sends a
   ``ResultMessage`` with usage + cost, every per-step ``Step.metrics`` is
   ``None`` and the renderer's per-phase rollup shows ``$0.00 / 0 tokens``.

The renderer aggregates **per-step** metrics, so even though
``FinalMetrics.total_cost_usd`` ends up correct, the user-facing comment
silently understates cost and tokens. We assert against the rendered
markdown to make the regression visible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends.claude import ClaudeBackend
from daydream.pr_comment_renderer import render_run_info_block
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    TrajectoryRecorder,
)

# Fake SDK message types matching real claude_agent_sdk shapes. We patch the
# symbols imported into daydream.backends.claude so its isinstance checks pass.


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeThinkingBlock:
    thinking: str


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] | None = None


@dataclass
class FakeToolResultBlock:
    tool_use_id: str
    content: str | None = None
    is_error: bool = False


@dataclass
class FakeAssistantMessage:
    """Real-shape AssistantMessage: has ``content`` + ``model``, NO ``usage``."""

    content: list[Any]
    model: str
    parent_tool_use_id: str | None = None
    error: object | None = None


@dataclass
class FakeUserMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class FakeResultMessage:
    """Real-shape ResultMessage: ``usage`` + ``total_cost_usd`` live here."""

    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    structured_output: Any = None
    subtype: str = "success"
    duration_ms: int = 0
    duration_api_ms: int = 0
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "fake-session"
    result: str | None = None


# Per-test FakeClient factory: each test queues a message sequence, then patches
# ClaudeSDKClient to a class whose instances replay it.


class _FakeClientBase:
    """Subclass per test and override ``MESSAGES`` to control the stream."""

    MESSAGES: list[Any] = []

    def __init__(self, options: Any = None) -> None:
        self.options = options
        self._prompt: str = ""

    async def __aenter__(self) -> _FakeClientBase:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self._prompt = prompt

    async def receive_response(self):  # noqa: ANN201 - matches SDK shape
        for msg in self.MESSAGES:
            yield msg


def _make_fake_client(messages: list[Any]) -> type[_FakeClientBase]:
    """Build a fresh FakeClient class whose instances replay *messages*."""

    class _ScopedFakeClient(_FakeClientBase):
        MESSAGES = messages

    return _ScopedFakeClient


@pytest.fixture
def patch_sdk(monkeypatch: pytest.MonkeyPatch):
    """Patch every SDK symbol that ``ClaudeBackend.execute`` does isinstance on."""

    def _patch(messages: list[Any]) -> None:
        client_cls = _make_fake_client(messages)
        monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", client_cls)
        monkeypatch.setattr("daydream.backends.claude.AssistantMessage", FakeAssistantMessage)
        monkeypatch.setattr("daydream.backends.claude.UserMessage", FakeUserMessage)
        monkeypatch.setattr("daydream.backends.claude.ResultMessage", FakeResultMessage)
        monkeypatch.setattr("daydream.backends.claude.TextBlock", FakeTextBlock)
        monkeypatch.setattr("daydream.backends.claude.ThinkingBlock", FakeThinkingBlock)
        monkeypatch.setattr("daydream.backends.claude.ToolUseBlock", FakeToolUseBlock)
        monkeypatch.setattr("daydream.backends.claude.ToolResultBlock", FakeToolResultBlock)

    return _patch


def _make_recorder(tmp_path: Path) -> TrajectoryRecorder:
    """Mirror runner.py: ``agent_model_name`` is stamped per-step, not at recorder init."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.TTT,
        target_dir=tmp_path,
        # Backend alias is only a legacy fallback; daydream stamps the resolved
        # model on each Step explicitly (config.model was replaced by per-phase fields).
        agent_model_name="claude",
        session_id="test",
    )


def _model_line(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("- **Model:**"):
            return line
    raise AssertionError(f"No Model line in markdown:\n{markdown}")


def _cost_line(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("- **Cost:**"):
            return line
    raise AssertionError(f"No Cost line in markdown:\n{markdown}")


def _tokens_line(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("- **Tokens:**"):
            return line
    raise AssertionError(f"No Tokens line in markdown:\n{markdown}")


def _phase_row(markdown: str, label: str) -> str:
    needle = f"| {label} |"
    for line in markdown.splitlines():
        if line.startswith(needle):
            return line
    raise AssertionError(f"No phase row {label!r} in markdown:\n{markdown}")


FIXTURE_MODEL_ID = "fixture-sdk-model-id"


async def test_render_uses_real_sdk_model_id_not_backend_alias(
    tmp_path: Path, patch_sdk: Any
) -> None:
    """Bug A: rendered model line must surface the SDK model id.

    The real ``AssistantMessage`` has a ``model`` field carrying the actual
    SDK model id (e.g. ``claude-opus-4-5-20250901``). ``ClaudeBackend`` is
    expected to thread that into the trajectory so the renderer's rollup
    line reads ``- **Model:** claude-opus-4-5-20250901``. Today the
    backend ignores ``msg.model`` entirely, so the renderer sees only the
    recorder-stamped backend alias ``"claude"``.
    """
    patch_sdk([
        FakeAssistantMessage(
            content=[FakeTextBlock(text="reviewing the code")],
            model=FIXTURE_MODEL_ID,
        ),
        FakeResultMessage(
            total_cost_usd=0.42,
            usage={
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 800,
            },
        ),
    ])

    recorder = _make_recorder(tmp_path)
    target_path = recorder.path
    async with recorder:
        backend = ClaudeBackend(model="opus")
        await run_agent(backend, tmp_path, "review please", phase=DaydreamPhase.REVIEW)

    assert target_path.exists(), "Trajectory file should have been written"
    markdown = render_run_info_block([target_path])

    model_line = _model_line(markdown)
    assert FIXTURE_MODEL_ID in model_line, (
        f"Bug A: expected real SDK model id {FIXTURE_MODEL_ID!r} in Model line, "
        f"got: {model_line!r}\n\nFull markdown:\n{markdown}"
    )
    # Exact-line check: real id begins with "claude-", so a substring match on
    # "claude" alone would be unsafe.
    assert model_line.strip() != "- **Model:** claude", (
        f"Bug A: Model line is the backend alias instead of the SDK model id: {model_line!r}"
    )


async def test_render_shows_real_cost_and_tokens_from_sdk_usage(
    tmp_path: Path, patch_sdk: Any
) -> None:
    """Bugs B + C: rollup cost / tokens must reflect SDK usage data.

    The renderer aggregates per-step ``Step.metrics``. With Bug B (no
    MetricsEvent emitted from AssistantMessage) and Bug C (CostEvent never
    populates the open step's _metrics), every Step.metrics is ``None`` →
    aggregator skips → rollup reads ``$0.00`` / ``0 in / 0 out``.
    """
    review_messages = [
        FakeAssistantMessage(
            content=[FakeTextBlock(text="reviewing")],
            model=FIXTURE_MODEL_ID,
        ),
        FakeResultMessage(
            total_cost_usd=0.30,
            usage={
                "input_tokens": 5000,
                "output_tokens": 600,
                "cache_read_input_tokens": 2000,
            },
        ),
    ]
    fix_messages = [
        FakeAssistantMessage(
            content=[FakeTextBlock(text="fixing")],
            model=FIXTURE_MODEL_ID,
        ),
        FakeResultMessage(
            total_cost_usd=0.15,
            usage={
                "input_tokens": 2500,
                "output_tokens": 400,
                "cache_read_input_tokens": 1000,
            },
        ),
    ]

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

    cost_line = _cost_line(markdown)
    tokens_line = _tokens_line(markdown)

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


async def test_per_phase_rollup_distinguishes_phases(
    tmp_path: Path, patch_sdk: Any
) -> None:
    """Bug B/C end-to-end: the per-phase breakdown must show one row per phase.

    Two separate ``run_agent()`` calls under different phases must produce
    distinct rows in the per-phase breakdown table, each carrying the
    metrics from their own ResultMessage usage. The rows existing isn't
    enough — they must reflect the right Steps / Tools / Cost values.
    """
    review_messages = [
        FakeAssistantMessage(
            content=[FakeTextBlock(text="reviewing")],
            model=FIXTURE_MODEL_ID,
        ),
        FakeResultMessage(
            total_cost_usd=0.20,
            usage={
                "input_tokens": 4000,
                "output_tokens": 500,
                "cache_read_input_tokens": 1500,
            },
        ),
    ]
    parse_messages = [
        FakeAssistantMessage(
            content=[FakeTextBlock(text="parsed")],
            model=FIXTURE_MODEL_ID,
        ),
        FakeResultMessage(
            total_cost_usd=0.05,
            usage={
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_input_tokens": 500,
            },
        ),
    ]

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
