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


def test_format_verdict_join_renders_table_counts():
    from rich.console import Console

    from daydream.ui import format_verdict_join  # type: ignore[attr-defined]

    table = format_verdict_join(matched=[1, 2], unmatched=[3], structural=[4, 5], other=[], total=5)
    console = Console(record=True, force_terminal=True, width=100)
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


def test_render_exploration_summary_shows_content_not_json():
    from rich.console import Console

    from daydream.exploration import Convention, Dependency, ExplorationContext, FileInfo
    from daydream.ui import render_exploration_summary  # type: ignore[attr-defined]

    ctx = ExplorationContext(
        affected_files=[FileInfo(path="services/library/openapi.yaml", role="modified")],
        conventions=[
            Convention(name="OpenAPI First", description="openapi.yaml is the HTTP contract", source="CLAUDE.md")
        ],
        dependencies=[Dependency(source="router.go", target="gen/server.go", relationship="imports")],
    )
    console = Console(record=True, force_terminal=True, width=100)
    console.print(render_exploration_summary(ctx))
    out = console.export_text()
    assert "OpenAPI First" in out
    assert "1 convention" in out  # count line
    assert "{" not in out  # no raw JSON


def test_render_exploration_summary_empty_is_quiet():
    from rich.console import Console

    from daydream.exploration import ExplorationContext
    from daydream.ui import render_exploration_summary  # type: ignore[attr-defined]

    console = Console(record=True, force_terminal=True, width=100)
    console.print(render_exploration_summary(ExplorationContext()))
    out = console.export_text()
    assert "{" not in out and "[" not in out  # never dumps a structure; one dim line at most


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
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=True)
    reg.create("c1", "Bash", {"command": "pytest", "run_in_background": True, "description": "Run tests"})
    launch = (Path(__file__).parent / "fixtures/task_tools/bash_bg_launch.txt").read_text()
    reg.observe_result("c1", launch)
    assert reg.resolve_label("Bash", "b0nsmwb99") == "Run tests"
    assert reg.resolve_label("Bash", "unknown") is None

    reg.create("c2", "TaskCreate", {"subject": "Find tool-call render code", "description": "d"})
    create = (Path(__file__).parent / "fixtures/task_tools/taskcreate_result.txt").read_text()
    reg.observe_result("c2", create)
    assert reg.resolve_label("TaskCreate", "1") == "Find tool-call render code"


def test_bash_panel_shows_command_drops_mechanical_keys():
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=False)
    reg.create("c1", "Bash", {"command": "pytest", "block": True, "timeout": 120000})
    out = _render_panel_text(reg, "c1")
    assert "pytest" in out
    assert "block" not in out and "timeout" not in out


def test_registry_harvests_task_label_from_originating_result():
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=True)
    reg.create("c1", "Bash", {"command": "pytest", "run_in_background": True, "description": "Run tests"})
    reg.observe_result("c1", "Command running in background with ID: a066168. Output ...")
    assert reg.resolve_label("Bash", "a066168") == "Run tests"
    assert reg.resolve_label("Bash", "unknown") is None


def _render_panel_text(reg, tool_use_id):
    from rich.console import Console

    c = Console(record=True)
    c.print(reg.get(tool_use_id)._render_panel())
    return c.export_text()


def test_taskoutput_header_leads_with_label_demotes_id():
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=True)
    reg.create("c1", "Bash", {"command": "x", "run_in_background": True, "description": "Run tests"})
    reg.observe_result("c1", "Command running in background with ID: a066168. ...")
    reg.create("c2", "TaskOutput", {"task_id": "a066168", "block": True, "timeout": 120000})
    out = _render_panel_text(reg, "c2")
    assert "Run tests" in out and "a066168" in out
    assert "block" not in out and "timeout" not in out


def test_taskoutput_header_unknown_id_falls_back_to_bare_id():
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=True)
    reg.create("c2", "TaskOutput", {"task_id": "zzz999", "block": True, "timeout": 1})
    out = _render_panel_text(reg, "c2")
    assert "zzz999" in out and "block" not in out


def test_taskcreate_header_shows_subject_and_body():
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=True)
    reg.create("c1", "TaskCreate", {"subject": "Fix auth bug", "description": "details here"})
    out = _render_panel_text(reg, "c1")
    assert "Fix auth bug" in out and "details here" in out


def test_taskupdate_resolves_subject_and_shows_status():
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=True)
    reg.create("c1", "TaskCreate", {"subject": "Fix auth bug", "description": "d"})
    reg.observe_result("c1", "Task #1 created successfully: Fix auth bug")
    reg.create("c2", "TaskUpdate", {"taskId": "1", "status": "completed"})
    c = Console(record=True)
    c.print(reg.get("c2")._render_panel())
    out = c.export_text()
    assert "Fix auth bug" in out and "completed" in out


def test_tasklist_header_omits_empty_id_suffix():
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=True)
    reg.create("c1", "TaskList", {})
    out = _render_panel_text(reg, "c1")
    assert "TaskList" in out
    assert "(#)" not in out and "()" not in out


def test_taskoutput_result_shows_output_snippet():
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry

    # quiet_mode=False so the result body renders (quiet mode suppresses result
    # output entirely); R8 is about the rendered TaskOutput result snippet.
    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=False)
    reg.create("c2", "TaskOutput", {"task_id": "a066168", "block": True, "timeout": 1})
    result = (Path(__file__).parent / "fixtures/task_tools/taskoutput_result.txt").read_text()
    reg.get("c2").set_result(result, is_error=False)
    c = Console(record=True)
    c.print(reg.get("c2")._render_panel())
    out = c.export_text()
    assert "done-with-bg-work" in out  # the <output> snippet surfaces
    assert "<retrieval_status>" not in out  # tag plumbing is stripped


def test_task_prompt_truncation_uses_named_limit():
    from rich.console import Console

    from daydream.ui import LiveToolPanelRegistry
    from daydream.ui.theme import _TASK_PROMPT_MAX_LINES

    reg = LiveToolPanelRegistry(Console(record=True), quiet_mode=True)
    reg.create("c1", "Task", {"description": "d", "prompt": "\n".join(f"l{i}" for i in range(40))})
    out = _render_panel_text(reg, "c1")
    assert f"({40 - _TASK_PROMPT_MAX_LINES} more lines)" in out
    assert "l0" in out
    assert "l39" not in out


async def test_run_agent_renders_taskoutput_with_label(tmp_path, monkeypatch):
    from rich.console import Console

    import daydream.agent as agent_mod
    from daydream.agent import run_agent
    from daydream.backends import ResultEvent, ToolResultEvent, ToolStartEvent
    from daydream.trajectory import DaydreamPhase
    from tests.test_agent_recorder_integration import MockBackend  # existing event-replay mock

    rec = Console(record=True, width=120)
    monkeypatch.setattr(agent_mod, "console", rec)
    backend = MockBackend(
        [
            ToolStartEvent(
                id="c1",
                name="Bash",
                input={"command": "pytest", "run_in_background": True, "description": "Run tests"},
            ),
            ToolResultEvent(id="c1", output="Command running in background with ID: a066168. ...", is_error=False),
            ToolStartEvent(id="c2", name="TaskOutput", input={"task_id": "a066168", "block": True, "timeout": 120000}),
            ToolResultEvent(
                id="c2",
                output="<task_id>a066168</task_id>\n<output>\ndone-with-bg-work\n</output>",
                is_error=False,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW)
    out = rec.export_text()
    assert "Run tests" in out and "a066168" in out
    assert "done-with-bg-work" in out
    assert "block=True" not in out and "timeout=120000" not in out


async def test_run_agent_callback_path_labels_taskoutput(tmp_path):
    from daydream.agent import run_agent
    from daydream.backends import ResultEvent, ToolResultEvent, ToolStartEvent
    from daydream.trajectory import DaydreamPhase
    from tests.test_agent_recorder_integration import MockBackend  # existing event-replay mock

    backend = MockBackend(
        [
            ToolStartEvent(
                id="c1",
                name="Bash",
                input={"command": "pytest", "run_in_background": True, "description": "Run tests"},
            ),
            ToolResultEvent(id="c1", output="Command running in background with ID: a066168. ...", is_error=False),
            ToolStartEvent(id="c2", name="TaskOutput", input={"task_id": "a066168", "block": True, "timeout": 120000}),
            ToolResultEvent(
                id="c2",
                output="<task_id>a066168</task_id>\n<output>\ndone-with-bg-work\n</output>",
                is_error=False,
            ),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )
    from rich.text import Text

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


async def test_run_agent_callback_path_edit_shows_file_not_bool(tmp_path):
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
