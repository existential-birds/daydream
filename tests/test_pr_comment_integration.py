"""Drive the real Claude backend, agent and recorder through the SDK boundary.

Rendered comments must preserve SDK model identity and per-phase usage/cost.
Per-step metrics matter: correct final totals alone cannot verify the renderer."""

from __future__ import annotations

import json
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


async def test_sdk_review_parse_and_fix_render_exact_phase_and_run_usage(tmp_path: Path, patch_sdk: Any) -> None:
    """One real multi-phase recording exposes model attribution and per-phase billing independently."""
    recorder = _make_recorder(tmp_path)
    backend = ClaudeBackend(model="opus")
    async with recorder:
        patch_sdk(_stream("Reviewing", 0.30, input_tokens=5000, output_tokens=600, cache_read_input_tokens=2000))
        await run_agent(backend, tmp_path, "review", phase=DaydreamPhase.REVIEW)
        patch_sdk(_stream("Parsed", 0.05, input_tokens=1000, output_tokens=100, cache_read_input_tokens=500))
        await run_agent(backend, tmp_path, "parse", phase=DaydreamPhase.PARSE)
        patch_sdk(_stream("Fixed", 0.15, input_tokens=2500, output_tokens=400, cache_read_input_tokens=1000))
        await run_agent(backend, tmp_path, "fix", phase=DaydreamPhase.FIX)
    document = json.loads(recorder.path.read_text())
    steps = [step for step in document["steps"] if step["source"] == "agent" and step["message"]]
    assert [step["message"] for step in steps] == ["Reviewing", "Parsed", "Fixed"]
    # Content without message usage has no native model witness until terminal billing arrives.
    assert [step["model_name"] for step in steps] == ["claude", FIXTURE_MODEL_ID, FIXTURE_MODEL_ID]
    assert all(step.get("metrics") is None for step in steps)
    billed = [step for step in document["steps"] if step["source"] == "agent" and step.get("metrics")]
    assert [(step["extra"]["daydream_phase"], step["model_name"], step["message"],
             step["metrics"]["prompt_tokens"], step["metrics"]["cached_tokens"],
             step["metrics"]["completion_tokens"], step["metrics"]["cost_usd"]) for step in billed] == [
        ("review", FIXTURE_MODEL_ID, "", 7000, 2000, 600, 0.30),
        ("parse", FIXTURE_MODEL_ID, "", 1500, 500, 100, 0.05),
        ("fix", FIXTURE_MODEL_ID, "", 3500, 1000, 400, 0.15),
    ]
    report = render_run_info_block([recorder.path])
    assert _line(report, "- **Model:**") == f"- **Model:** {FIXTURE_MODEL_ID}"
    assert _line(report, "- **Cost:**") == "- **Cost:** $0.50"
    assert _line(report, "- **Tokens:**") == "- **Tokens:** 12,000 in (3,500 cached, 29% hit) → 1,100 out"
    assert [line.split("|")[1].strip() for line in report.splitlines()
            if line.startswith("| ") and not line.startswith("| Phase |")] == ["Review", "Parse Feedback", "Fix"]
    rows = {
        phase: [cell.strip() for cell in _phase_row(report, phase).strip("|").split("|")][1:6]
        for phase in ("Review", "Parse Feedback", "Fix")
    }
    assert rows == {
        "Review": [FIXTURE_MODEL_ID, "0", "7,000 (29%)", "600", "$0.30"],
        "Parse Feedback": [FIXTURE_MODEL_ID, "0", "1,500 (33%)", "100", "$0.05"],
        "Fix": [FIXTURE_MODEL_ID, "0", "3,500 (29%)", "400", "$0.15"],
    }
