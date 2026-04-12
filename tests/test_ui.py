"""Tests for daydream.ui helpers."""

from __future__ import annotations


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
