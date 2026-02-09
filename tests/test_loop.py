"""Tests for continuous loop mode."""

import pytest

from daydream.runner import RunConfig


def test_runconfig_loop_defaults():
    config = RunConfig()
    assert config.loop is False
    assert config.max_iterations == 5


import re
from io import StringIO

from rich.console import Console

from daydream.ui import NEON_THEME, print_iteration_divider

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def test_print_iteration_divider():
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80, theme=NEON_THEME)
    print_iteration_divider(test_console, 2, 5)
    plain = strip_ansi(output.getvalue())
    assert "Iteration 2 of 5" in plain
    assert "━" in plain


from daydream.ui import SummaryData, print_summary


def test_summary_data_loop_fields_default():
    data = SummaryData(
        skill="python", target="/tmp", feedback_count=3,
        fixes_applied=3, test_retries=0, tests_passed=True,
    )
    assert data.loop_mode is False
    assert data.iterations_used == 1


def test_summary_loop_mode_shows_iterations():
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80, theme=NEON_THEME)
    data = SummaryData(
        skill="python", target="/tmp", feedback_count=8,
        fixes_applied=8, test_retries=1, tests_passed=True,
        loop_mode=True, iterations_used=3,
    )
    print_summary(test_console, data)
    plain = strip_ansi(output.getvalue())
    assert "Iterations" in plain
    assert "3" in plain


from pathlib import Path
from typing import Any

from daydream.backends import ResultEvent, TextEvent
from daydream.runner import RunConfig, run


class LoopMockBackend:
    """Mock backend that returns different results on successive calls.

    Tracks call count and uses prompt content to determine responses.
    The `review_results` list controls what phase_parse_feedback returns
    on each iteration (list of issue lists, one per iteration).
    """

    def __init__(self, review_results: list[list[dict[str, Any]]], tests_pass: bool = True):
        self._review_results = review_results
        self._tests_pass = tests_pass
        self._parse_call = 0
        self.call_log: list[str] = []

    async def execute(self, cwd, prompt, output_schema=None, continuation=None):
        prompt_lower = prompt.lower()
        self.call_log.append(prompt_lower[:80])

        if "beagle-" in prompt_lower and "review" in prompt_lower:
            yield TextEvent(text="Review complete.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "extract" in prompt_lower and "json" in prompt_lower:
            issues = (
                self._review_results[self._parse_call]
                if self._parse_call < len(self._review_results)
                else []
            )
            self._parse_call += 1
            yield TextEvent(text="Parsed.")
            yield ResultEvent(structured_output={"issues": issues}, continuation=None)
        elif "fix this issue" in prompt_lower:
            yield TextEvent(text="Fixed.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "test suite" in prompt_lower or "run the project" in prompt_lower:
            if self._tests_pass:
                yield TextEvent(text="All 1 tests passed. 0 failed.")
            else:
                yield TextEvent(text="1 test failed.")
            yield ResultEvent(structured_output=None, continuation=None)
        elif "commit-push" in prompt_lower:
            yield TextEvent(text="Committed.")
            yield ResultEvent(structured_output=None, continuation=None)
        else:
            yield TextEvent(text="OK")
            yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self):
        pass

    def format_skill_invocation(self, skill_key, args=""):
        return f"/{skill_key}" + (f" {args}" if args else "")


@pytest.fixture
def loop_target(tmp_path: Path) -> Path:
    project = tmp_path / "loop_project"
    project.mkdir()
    (project / "main.py").write_text("def hello():\n    return 'world'\n")
    (project / ".review-output.md").write_text("# Review\n\n1. Issue in main.py:1\n")
    return project


@pytest.fixture
def mock_ui_loop(monkeypatch):
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *a, **kw: "n")


@pytest.mark.asyncio
async def test_loop_exits_on_zero_issues(loop_target, mock_ui_loop, monkeypatch):
    """Issues on iteration 1, zero on iteration 2 -> exits 0."""
    issue = {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}
    backend = LoopMockBackend(review_results=[[issue], []])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert backend._parse_call == 2  # called parse twice


@pytest.mark.asyncio
async def test_loop_respects_max_iterations(loop_target, mock_ui_loop, monkeypatch):
    """Always returns issues -> stops at max_iterations, exits 1."""
    issue = {"id": 1, "description": "Persistent issue", "file": "main.py", "line": 1}
    # 3 iterations, all return the same issue
    backend = LoopMockBackend(review_results=[[issue], [issue], [issue]])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=3,
    )
    exit_code = await run(config)

    assert exit_code == 1
    assert backend._parse_call == 3


@pytest.mark.asyncio
async def test_loop_stops_on_test_failure(loop_target, mock_ui_loop, monkeypatch):
    """Tests fail mid-loop -> stops immediately, exits 1."""
    issue = {"id": 1, "description": "Issue", "file": "main.py", "line": 1}
    backend = LoopMockBackend(review_results=[[issue], [issue]], tests_pass=False)

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
    )
    exit_code = await run(config)

    assert exit_code == 1
    assert backend._parse_call == 1  # stopped after first iteration


@pytest.mark.asyncio
async def test_loop_accumulates_stats(loop_target, mock_ui_loop, monkeypatch):
    """Stats accumulate across iterations."""
    issue1 = {"id": 1, "description": "Issue A", "file": "main.py", "line": 1}
    issue2 = {"id": 2, "description": "Issue B", "file": "main.py", "line": 2}
    # Iteration 1: 2 issues, Iteration 2: 1 issue, Iteration 3: 0 issues
    backend = LoopMockBackend(review_results=[[issue1, issue2], [issue1], []])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    # Capture summary data
    captured_summary = {}

    original_print_summary = None
    import daydream.runner as runner_mod
    original_print_summary = runner_mod.print_summary

    def capture_summary(console, data):
        captured_summary["feedback_count"] = data.feedback_count
        captured_summary["fixes_applied"] = data.fixes_applied
        captured_summary["iterations_used"] = data.iterations_used
        captured_summary["loop_mode"] = data.loop_mode
        original_print_summary(console, data)

    monkeypatch.setattr("daydream.runner.print_summary", capture_summary)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert captured_summary["feedback_count"] == 3  # 2 + 1
    assert captured_summary["fixes_applied"] == 3  # 2 + 1
    assert captured_summary["iterations_used"] == 3
    assert captured_summary["loop_mode"] is True


@pytest.mark.asyncio
async def test_loop_false_single_pass(loop_target, mock_ui_loop, monkeypatch):
    """loop=False behaves identically to existing single-pass flow."""
    issue = {"id": 1, "description": "Issue", "file": "main.py", "line": 1}
    backend = LoopMockBackend(review_results=[[issue]])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=False,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert backend._parse_call == 1  # single pass only
