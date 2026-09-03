"""Tests for daydream.ui helpers."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from daydream.ui.panels import LiveToolPanelRegistry


def test_format_verdict_join_renders_table_counts() -> None:
    from rich.console import Console

    from daydream.ui import format_verdict_join

    table = format_verdict_join(matched=[1, 2], unmatched=[3], structural=[4, 5], other=[], total=5)
    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    console.print(table)
    out = console.export_text()
    assert "2" in out and "matched" in out.lower()
    assert "structural" in out.lower()
    assert "{" not in out


def _run_renderer_and_count_panels(width: int, height: int, text_lines: list[str]) -> tuple[int, object]:
    from rich.console import Console
    from rich.panel import Panel

    from daydream.ui import AgentTextRenderer

    console = Console(width=width, height=height, force_terminal=True)
    renderer = AgentTextRenderer(console)

    panel_prints: list[Panel] = []
    original_print = console.print

    def spy_print(*args: Any, **kwargs: Any) -> Any:
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


def test_agent_text_renderer_overflow_single_panel() -> None:
    lines = [f"line {i} with some content to fill horizontally\n" for i in range(200)]
    panel_count, renderer = _run_renderer_and_count_panels(80, 20, lines)

    # finish() must NOT print an extra Panel via console.print after stopping Live
    assert panel_count == 0, f"finish() printed {panel_count} extra panel(s) via console.print"
    assert renderer._live is None  # type: ignore[attr-defined]
    assert renderer._buffer == []  # type: ignore[attr-defined]


def test_render_exploration_summary_shows_content_not_json() -> None:
    from rich.console import Console

    from daydream.exploration import Convention, Dependency, ExplorationContext, FileInfo
    from daydream.ui import render_exploration_summary

    ctx = ExplorationContext(
        affected_files=[FileInfo(path="services/library/openapi.yaml", role="modified")],
        conventions=[
            Convention(name="OpenAPI First", description="openapi.yaml is the HTTP contract", source="CLAUDE.md")
        ],
        dependencies=[Dependency(source="router.go", target="gen/server.go", relationship="imports")],
    )
    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    console.print(render_exploration_summary(ctx))
    out = console.export_text()
    assert "OpenAPI First" in out
    assert "1 convention" in out  # count line
    assert "{" not in out  # no raw JSON


def test_render_exploration_summary_empty_is_quiet() -> None:
    from rich.console import Console

    from daydream.exploration import ExplorationContext
    from daydream.ui import render_exploration_summary

    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    console.print(render_exploration_summary(ExplorationContext()))
    out = console.export_text()
    assert "{" not in out and "[" not in out  # never dumps a structure; one dim line at most


def test_prompt_user_returns_default_on_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from rich.console import Console

    from daydream.agent import reset_state
    from daydream.ui import prompt_user

    reset_state()
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError("EOF when reading a line")))
    # Issue #126 exact repro expectation:
    console = Console(file=StringIO(), record=True)
    assert prompt_user(console, "Apply fixes now?", default="n") == "n"
    # Operator must receive a visible signal that EOF caused the decline.
    output = console.export_text()
    assert "EOF" in output, f"expected EOF warning in output, got: {output!r}"


def test_prompt_user_non_interactive_skips_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_prompt_user_returns_typed_value_interactively(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console

    from daydream.agent import reset_state
    from daydream.ui import prompt_user

    reset_state()
    monkeypatch.setattr("builtins.input", lambda: "y")
    assert prompt_user(Console(), "Confirm?", default="n") == "y"


def test_parse_background_task_id_from_launch_string() -> None:
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "Bash", {"command": "pytest", "run_in_background": True, "description": "Run tests"})
    launch = (Path(__file__).parent / "fixtures/task_tools/bash_bg_launch.txt").read_text()
    reg.observe_result("c1", launch)
    assert reg.resolve_label("Bash", "b0nsmwb99") == "Run tests"
    assert reg.resolve_label("Bash", "unknown") is None

    reg.create("c2", "TaskCreate", {"subject": "Find tool-call render code", "description": "d"})
    create = (Path(__file__).parent / "fixtures/task_tools/taskcreate_result.txt").read_text()
    reg.observe_result("c2", create)
    assert reg.resolve_label("TaskCreate", "1") == "Find tool-call render code"


def test_bash_panel_shows_command_drops_mechanical_keys() -> None:
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=False)
    reg.create("c1", "Bash", {"command": "pytest", "block": True, "timeout": 120000})
    out = _render_panel_text(reg, "c1")
    assert "pytest" in out
    assert "block" not in out and "timeout" not in out


def _render_panel_text(reg: LiveToolPanelRegistry, tool_use_id: str) -> str:
    from rich.console import Console

    c = Console(file=StringIO(), record=True)
    panel = reg.get(tool_use_id)
    assert panel is not None
    c.print(panel._render_panel())
    return c.export_text()


def test_taskoutput_header_leads_with_label_demotes_id() -> None:
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "Bash", {"command": "x", "run_in_background": True, "description": "Run tests"})
    reg.observe_result("c1", "Command running in background with ID: a066168. ...")
    reg.create("c2", "TaskOutput", {"task_id": "a066168", "block": True, "timeout": 120000})
    out = _render_panel_text(reg, "c2")
    assert "Run tests" in out and "a066168" in out
    assert "block" not in out and "timeout" not in out


def test_taskoutput_header_unknown_id_falls_back_to_bare_id() -> None:
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c2", "TaskOutput", {"task_id": "zzz999", "block": True, "timeout": 1})
    out = _render_panel_text(reg, "c2")
    assert "zzz999" in out and "block" not in out


def test_taskcreate_header_shows_subject_and_body() -> None:
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "TaskCreate", {"subject": "Fix auth bug", "description": "details here"})
    out = _render_panel_text(reg, "c1")
    assert "Fix auth bug" in out and "details here" in out


def test_taskupdate_resolves_subject_and_shows_status() -> None:
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "TaskCreate", {"subject": "Fix auth bug", "description": "d"})
    reg.observe_result("c1", "Task #1 created successfully: Fix auth bug")
    reg.create("c2", "TaskUpdate", {"taskId": "1", "status": "completed"})
    out = _render_panel_text(reg, "c2")
    assert "Fix auth bug" in out and "completed" in out


def test_tasklist_header_omits_empty_id_suffix() -> None:
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "TaskList", {})
    out = _render_panel_text(reg, "c1")
    assert "TaskList" in out
    assert "(#)" not in out and "()" not in out


def test_taskoutput_result_shows_output_snippet() -> None:
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    # quiet_mode=False so the result body renders (quiet mode suppresses result
    # output entirely); R8 is about the rendered TaskOutput result snippet.
    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=False)
    reg.create("c2", "TaskOutput", {"task_id": "a066168", "block": True, "timeout": 1})
    result = (Path(__file__).parent / "fixtures/task_tools/taskoutput_result.txt").read_text()
    c2_panel = reg.get("c2")
    assert c2_panel is not None
    c2_panel.set_result(result, is_error=False)
    out = _render_panel_text(reg, "c2")
    assert "done-with-bg-work" in out  # the <output> snippet surfaces
    assert "<retrieval_status>" not in out  # tag plumbing is stripped


def test_task_prompt_truncation_uses_named_limit() -> None:
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry
    from daydream.ui.theme import _TASK_PROMPT_MAX_LINES

    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "Task", {"description": "d", "prompt": "\n".join(f"l{i}" for i in range(40))})
    out = _render_panel_text(reg, "c1")
    assert f"({40 - _TASK_PROMPT_MAX_LINES} more lines)" in out
    assert "l0" in out
    assert "l39" not in out


def _taskoutput_backend() -> Any:
    """Build a backend stream containing a background task and its final output."""
    from daydream.backends import ResultEvent, ToolResultEvent, ToolStartEvent
    from tests.test_agent_recorder_integration import MockBackend

    return MockBackend(
        [
            ToolStartEvent(
                id="c1",
                name="Bash",
                input={"command": "pytest", "run_in_background": True, "description": "Run tests"},
            ),
            ToolResultEvent(
                id="c1",
                output="Command running in background with ID: a066168. ...",
                is_error=False,
            ),
            ToolStartEvent(
                id="c2",
                name="TaskOutput",
                input={"task_id": "a066168", "block": True, "timeout": 120000},
            ),
            ToolResultEvent(
                id="c2",
                output="<task_id>a066168</task_id>\n<output>\ndone-with-bg-work\n</output>",
                is_error=False,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )


async def test_run_agent_renders_taskoutput_with_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Render TaskOutput with its task label while hiding mechanical arguments."""
    from rich.console import Console

    import daydream.agent as agent_mod
    from daydream.agent import run_agent
    from daydream.trajectory import DaydreamPhase

    rec = Console(file=StringIO(), record=True, width=120)
    monkeypatch.setattr(agent_mod, "console", rec)
    backend = _taskoutput_backend()
    await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW)
    out = rec.export_text()
    assert "Run tests" in out and "a066168" in out
    assert "done-with-bg-work" in out
    assert "block=True" not in out and "timeout=120000" not in out


async def test_run_agent_callback_path_labels_taskoutput(tmp_path: Path) -> None:
    from rich.text import Text

    from daydream.agent import run_agent
    from daydream.trajectory import DaydreamPhase

    backend = _taskoutput_backend()
    lines: list[Text] = []
    await run_agent(
        backend,
        tmp_path,
        "go",
        phase=DaydreamPhase.REVIEW,
        progress_callback=lines.append,
    )
    joined = "\n".join(line.plain for line in lines)
    assert "Run tests" in joined  # resolved label surfaces in callback mode
    assert "block" not in joined and "timeout" not in joined
    assert "TaskOutput a066168" not in joined  # opaque bare-id dump form is gone


async def test_run_agent_callback_coalesces_streaming_text_deltas(tmp_path: Path) -> None:
    """Token-sized text deltas render as one parallel-fix narration line."""
    from rich.text import Text

    from daydream.agent import run_agent
    from daydream.backends import ResultEvent, TextEvent
    from daydream.trajectory import DaydreamPhase
    from tests.test_agent_recorder_integration import MockBackend

    backend = MockBackend(
        [
            TextEvent("B"),
            TextEvent("ash"),
            TextEvent(" is"),
            TextEvent(" blocked."),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    lines: list[Text] = []

    result, _, _ = await run_agent(
        backend,
        tmp_path,
        "go",
        phase=DaydreamPhase.FIX,
        progress_callback=lines.append,
    )

    assert result == "Bash is blocked."
    assert [line.plain for line in lines] == ["    Bash is blocked."]


async def test_run_agent_callback_path_edit_shows_file_not_bool(tmp_path: Path) -> None:
    """The parallel-fix callback line names the edited file, never a stray flag.

    Regression: the old blind ``next(iter(args.values()))`` surfaced a leading
    ``replace_all`` flag as ``"Edit False"`` instead of the file being edited.
    """
    from rich.text import Text

    from daydream.agent import run_agent
    from daydream.backends import ResultEvent, ToolStartEvent
    from daydream.trajectory import DaydreamPhase
    from tests.test_agent_recorder_integration import MockBackend

    backend = MockBackend(
        [
            ToolStartEvent(
                id="e1",
                name="Edit",
                input={
                    "replace_all": False,
                    "file_path": "/repo/daydream/git_ops.py",
                    "old_string": "a",
                    "new_string": "b",
                },
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    lines: list[Text] = []
    await run_agent(
        backend,
        tmp_path,
        "go",
        phase=DaydreamPhase.FIX,
        progress_callback=lines.append,
    )
    joined = "\n".join(line.plain for line in lines)
    assert "/repo/daydream/git_ops.py" in joined  # the meaningful primary arg
    assert "Edit False" not in joined  # the stray-boolean dump is gone


def test_primary_tool_value_bash_prefers_command() -> None:
    """Bash primary arg is `command` (required, always present) over `description`."""
    from daydream.ui.tools import _primary_tool_value

    value, key = _primary_tool_value("Bash", {"command": "git diff --stat", "description": "Show changes"})
    assert (value, key) == ("git diff --stat", "command")

    # description-less call still resolves via the table, not the mechanical fallback.
    value, key = _primary_tool_value("Bash", {"command": "ls -la /tmp"})
    assert (value, key) == ("ls -la /tmp", "command")


def test_format_callback_progress_bash_shows_command() -> None:
    """Callback single-line path renders the command, not the paraphrase (issue #1108)."""
    from io import StringIO

    from rich.console import Console

    from daydream.ui.tools import format_callback_progress

    line = format_callback_progress("Bash", {"command": "git diff --stat", "description": "Show changes"}, None)
    c = Console(file=StringIO(), force_terminal=True, width=120, record=True)
    c.print(line)
    text = c.export_text()
    assert "git diff --stat" in text
    assert "Show changes" not in text
