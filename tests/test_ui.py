"""Tests for daydream.ui helpers."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_plan_renderer_dims_ungrounded_steps():
    from rich.console import Console

    from daydream.ui import render_ttt_plan  # type: ignore[attr-defined]

    plan = {
        "changes": [
            {
                "file": "x.py",
                "description": "grounded change",
                "references": [{"file": "x.py", "symbol": "f"}],
            },
            {
                "file": "y.py",
                "description": "ungrounded change",
                "references": [],
            },
        ]
    }

    console = Console(record=True)
    render_ttt_plan(console, plan)
    output = console.export_text()
    assert "(ungrounded)" in output


def _run_renderer_and_count_panels(width: int, height: int, text_lines: list[str]) -> tuple[int, object]:
    from rich.console import Console
    from rich.panel import Panel

    from daydream.ui import AgentTextRenderer

    console = Console(width=width, height=height, force_terminal=True)
    renderer = AgentTextRenderer(console)

    panel_prints: list[Panel] = []
    original_print = console.print

    def spy_print(*args, **kwargs):
        for arg in args:
            if isinstance(arg, Panel):
                panel_prints.append(arg)
        return original_print(*args, **kwargs)

    console.print = spy_print  # type: ignore[method-assign]

    renderer.start()
    for line in text_lines:
        renderer.append(line)
    renderer.finish()

    console.print = original_print  # type: ignore[method-assign]
    return len(panel_prints), renderer


def test_agent_text_renderer_overflow_single_panel():
    lines = [f"line {i} with some content to fill horizontally\n" for i in range(200)]
    panel_count, renderer = _run_renderer_and_count_panels(80, 20, lines)

    # finish() must NOT print an extra Panel via console.print after stopping Live
    assert panel_count == 0, f"finish() printed {panel_count} extra panel(s) via console.print"
    assert renderer._live is None  # type: ignore[attr-defined]
    assert renderer._buffer == []  # type: ignore[attr-defined]


def test_agent_text_renderer_small_content_single_panel():
    lines = ["hello\n", "world\n", "short content\n"]
    panel_count, renderer = _run_renderer_and_count_panels(80, 40, lines)

    assert panel_count == 0, f"finish() printed {panel_count} extra panel(s) via console.print"
    assert renderer._live is None  # type: ignore[attr-defined]
    assert renderer._buffer == []  # type: ignore[attr-defined]


def test_prompt_user_returns_default_on_eof(monkeypatch):
    from unittest.mock import Mock

    from rich.console import Console

    from daydream.agent import reset_state
    from daydream.ui import prompt_user

    reset_state()
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError("EOF when reading a line")))
    # Issue #126 exact repro expectation:
    console = Console(record=True)
    assert prompt_user(console, "Apply fixes now?", default="n") == "n"
    # Operator must receive a visible signal that EOF caused the decline.
    output = console.export_text()
    assert "EOF" in output, f"expected EOF warning in output, got: {output!r}"


def test_prompt_user_non_interactive_skips_stdin(monkeypatch):
    from unittest.mock import Mock

    from rich.console import Console

    from daydream.agent import reset_state, set_non_interactive
    from daydream.ui import prompt_user

    set_non_interactive(True)
    sentinel = Mock(side_effect=AssertionError("input() must not be called"))
    monkeypatch.setattr("builtins.input", sentinel)
    assert prompt_user(Console(), "Apply fixes now?", default="n") == "n"
    sentinel.assert_not_called()
    reset_state()


def test_prompt_user_returns_typed_value_interactively(monkeypatch):
    from rich.console import Console

    from daydream.agent import reset_state
    from daydream.ui import prompt_user

    reset_state()
    monkeypatch.setattr("builtins.input", lambda: "y")
    assert prompt_user(Console(), "Confirm?", default="n") == "y"


# Polarity contract: each destructive prompt must decline by default. Pairs
# mirror the source defaults; an "n" -> "y" flip in source (or here) breaks this.
@pytest.mark.parametrize(
    "message",
    [
        "Cleanup review output after completion? [y/N]",  # runner.py:814 (file deletion)
        "Post these as a PR review? [y/N]",  # pr_review.py:854 (external mutation)
        "Use suggested command instead?",  # phases.py:1729
        "Commit and push changes? [y/N]",  # phases.py:1788 (git push)
    ],
)
def test_prompt_user_destructive_defaults_decline_on_eof(monkeypatch, message):
    from unittest.mock import Mock

    from rich.console import Console

    from daydream.agent import reset_state
    from daydream.ui import prompt_user

    reset_state()
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError("EOF when reading a line")))
    assert prompt_user(Console(record=True), message, default="n") == "n"


def test_parse_background_task_id_from_launch_string():
    from daydream.ui import _parse_assigned_task_id

    launch = (Path(__file__).parent / "fixtures/task_tools/bash_bg_launch.txt").read_text()
    assert _parse_assigned_task_id("Bash", launch) == "b0nsmwb99"
    create = (Path(__file__).parent / "fixtures/task_tools/taskcreate_result.txt").read_text()
    assert _parse_assigned_task_id("TaskCreate", create) == "1"
    assert _parse_assigned_task_id("Bash", "no id here") is None
