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


# =============================================================================
# Safe-default contract for all prompt_user callsites
# =============================================================================
# Each entry documents one prompt_user() call in the codebase:
#   (module_path, prompt_fragment, expected_default, rationale)
#
# Polarity is intentionally mixed:
#   "n" (decline) — destructive / side-effectful actions the user must opt in to.
#   "y" (affirm)  — continuations where the happy path IS the expected answer.
#   other         — non-boolean menus / free-text inputs (path, number, enum).
#
# To add a new callsite: add a row here AND the corresponding call in source.
# To change a default: update source AND the row here — the test will catch drift.
_PROMPT_DEFAULT_INVENTORY = [
    # (module, prompt_fragment, default, rationale)
    ("daydream.runner",           "Enter target directory",            ".",     "free-text path input"),
    ("daydream.runner",           "Choice",                            "1",     "menu pick — first option is safest default"),
    ("daydream.runner",           "Cleanup review output",             "n",     "side-effect: file deletion — must opt in"),
    ("daydream.pr_review",        "Post these as a PR review",         "n",     "external mutation — must opt in"),
    ("daydream.deep.orchestrator","Apply fixes now",                   "n",     "potentially destructive write — must opt in"),
    ("daydream.phases",           "Copy handoff to clipboard",         "y",     "read-only convenience — safe to default on"),
    ("daydream.phases",           "Choice",                            "2",     "menu pick — documented safe selection"),
    ("daydream.phases",           "Use suggested command instead",     "n",     "replaces user input — must opt in"),
    ("daydream.phases",           "Commit and push changes",           "n",     "external git push — must opt in"),
    ("daydream.phases",           "Is this understanding correct",     "y",     "loop continuation — affirm advances, any text corrects"),
    ("daydream.phases",           "Create an implementation plan",     "all",   "non-boolean enum; 'all' is the expected happy path"),
]


import pytest  # noqa: E402  (import after module-level table to keep table readable)


@pytest.mark.parametrize(
    "module_path,prompt_fragment,expected_default,rationale",
    _PROMPT_DEFAULT_INVENTORY,
    ids=[row[1][:40] for row in _PROMPT_DEFAULT_INVENTORY],
)
def test_prompt_default_inventory(module_path, prompt_fragment, expected_default, rationale, monkeypatch):
    """Assert that the documented callsite-default table matches source.

    This test does NOT call the live prompt_user() — it introspects the source
    so that any accidental default change in source will fail here, making the
    safe-default contract explicit and reviewable in code review.

    The rationale column is not asserted; it exists purely as human documentation.
    """
    import ast
    import importlib
    import importlib.util
    import inspect

    mod = importlib.import_module(module_path)
    source = inspect.getsource(mod)
    tree = ast.parse(source)

    # Collect every prompt_user() call that contains the prompt_fragment
    matched_defaults: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute) else
                func.id if isinstance(func, ast.Name) else None)
        if name != "prompt_user":
            continue

        # Resolve the prompt argument: positional index 1 (0=console, 1=message, 2=default)
        # or keyword "message".
        message_val: str | None = None
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            message_val = node.args[1].value
        else:
            for kw in node.keywords:
                if kw.arg == "message" and isinstance(kw.value, ast.Constant):
                    message_val = kw.value.value

        if message_val is None or prompt_fragment not in message_val:
            continue

        # Resolve the default argument: positional index 2 or keyword "default".
        default_val: str | None = None
        if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
            default_val = node.args[2].value
        else:
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    default_val = kw.value.value

        if default_val is not None:
            matched_defaults.append(default_val)

    assert matched_defaults, (
        f"No prompt_user() call with message containing {prompt_fragment!r} "
        f"found in {module_path}. Update _PROMPT_DEFAULT_INVENTORY."
    )
    for actual in matched_defaults:
        assert actual == expected_default, (
            f"{module_path}: prompt containing {prompt_fragment!r} has default "
            f"{actual!r}, expected {expected_default!r}. "
            f"Update source or _PROMPT_DEFAULT_INVENTORY. Rationale: {rationale}"
        )
