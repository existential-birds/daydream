"""Tests for daydream.ui helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

import daydream.agent as agent_mod
import daydream.ui.tools as ui_tools
from daydream.agent import run_agent
from daydream.backends import ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.exploration import Convention, Dependency, ExplorationContext, FileInfo
from daydream.run_context import InteractionPolicy, RunContext, resolve_run_context
from daydream.trajectory import DaydreamPhase
from daydream.ui import (
    AgentTextRenderer,
    format_verdict_join,
    print_verification_summary,
    render_exploration_summary,
)
from daydream.ui.colorize import render_segments
from daydream.ui.panels import LiveToolPanelRegistry
from daydream.ui.theme import _TASK_PROMPT_MAX_LINES
from daydream.ui.tools import (
    _BASH_COMMAND_MAX_CHARS,
    format_callback_progress,
)
from tests.harness.backend import ScriptedBackend


def _render(renderable: object) -> str:
    """Render one rich renderable to plain text for assertions."""
    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    console.print(renderable)
    return console.export_text()


def _capture_console(fn: Callable[[Console], None]) -> str:
    """Run *fn* against a recording console and return the captured text."""
    console = Console(file=StringIO(), record=True, force_terminal=True, width=200)
    fn(console)
    return console.export_text()


def _ask(console: Console, message: str, default: str = "n") -> str:
    """Ask through the run's interaction gateway, the sole owner of prompt policy."""
    return resolve_run_context().choice(message, default=default, safe_default=default, console=console)


def _write_verdicts_artifact(directory: Path, *, verdicts: list[dict[str, Any]], selected: int, skipped: int) -> Path:
    """Write a ``recommendation-verdicts.json`` carrying the sibling selection block."""
    path = directory / "recommendation-verdicts.json"
    path.write_text(json.dumps({"verdicts": verdicts,
                "selection": {"rule_version": 1, "mode": "selective", "extra_categories": [], "decisions": [],
                    "selected": selected, "skipped": skipped,
                },
            }
        )
    )
    return path

def test_format_verdict_join_renders_table_counts() -> None:


    table = format_verdict_join(matched=[1, 2], unmatched=[3], skipped=[], structural=[4, 5], other=[], total=5)
    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    console.print(table)
    out = console.export_text()
    assert "2" in out and "matched" in out.lower()
    assert "structural" in out.lower()
    assert "{" not in out

def test_verdict_join_reports_selection_skips_as_their_own_bucket() -> None:
    table = format_verdict_join(matched=[1], unmatched=[], skipped=[2, 3], structural=[4], other=[], total=4)
    rendered = _render(table)
    assert "2, 3" in rendered
    assert "Skipped" in rendered
    assert "Unmatched" not in rendered  # a selection skip is never an unmatched verdict

def test_verification_summary_line_names_selected_and_skipped(tmp_path: Path) -> None:
    _write_verdicts_artifact(tmp_path, verdicts=[], selected=2, skipped=5)
    out = _capture_console(lambda c: print_verification_summary(c, tmp_path / "recommendation-verdicts.json"))
    assert "2 selected" in out and "5 skipped" in out

def test_verification_summary_omits_selection_when_block_absent(tmp_path: Path) -> None:
    path = tmp_path / "recommendation-verdicts.json"
    path.write_text(json.dumps({"verdicts": []}))
    out = _capture_console(lambda c: print_verification_summary(c, path))
    assert "selected" not in out
    assert "Recommendation verification: 0 findings" in out


def _run_renderer_and_count_panels(width: int, height: int, text_lines: list[str]) -> tuple[int, object]:


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


    ctx = ExplorationContext(affected_files=[FileInfo(path="services/library/openapi.yaml", role="modified")],
        conventions=[
            Convention(name="OpenAPI First", description="openapi.yaml is the HTTP contract", source="CLAUDE.md")
        ], dependencies=[Dependency(source="router.go", target="gen/server.go", relationship="imports")],
    )
    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    console.print(render_exploration_summary(ctx))
    out = console.export_text()
    assert "OpenAPI First" in out
    assert "1 convention" in out  # count line
    assert "{" not in out  # no raw JSON

def test_render_exploration_summary_empty_is_quiet() -> None:


    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    console.print(render_exploration_summary(ExplorationContext()))
    out = console.export_text()
    assert "{" not in out and "[" not in out  # never dumps a structure; one dim line at most


def test_choice_non_interactive_skips_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unattended policy resolves the safe default without touching stdin."""
    sentinel = Mock(side_effect=AssertionError("input() must not be called"))
    monkeypatch.setattr("builtins.input", sentinel)
    context = RunContext(InteractionPolicy(interactive=False))
    assert context.choice("Apply fixes now?", default="n", safe_default="n", console=Console()) == "n"
    sentinel.assert_not_called()

def test_choice_returns_typed_value_interactively(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda: "y")
    assert _ask(Console(), "Confirm?", default="n") == "y"

def test_panel_refresh_survives_a_tool_finishing_mid_render() -> None:
    registry = LiveToolPanelRegistry(Console(file=StringIO()))
    first = registry.create("first", "Read", {"file_path": "one.py"})
    second = registry.create("second", "Read", {"file_path": "two.py"})
    rendering = registry.iter_active_panels()
    assert next(rendering) is first
    registry.remove("first")
    assert list(rendering) == [second]
    registry.finish_all()


def test_panel_discard_allows_tool_id_reuse_without_a_live_leak() -> None:
    console = Console(file=StringIO())
    registry = LiveToolPanelRegistry(console)
    registry.create("reused", "Task", {"description": "discarded"})
    registry.discard_all()

    panel = registry.create("reused", "Read", {"file_path": "next.py"})
    assert list(registry.iter_active_panels()) == [panel]
    registry.remove("reused")
    assert list(registry.iter_active_panels()) == []
    assert not console._live_stack


def test_parse_background_task_id_from_launch_string() -> None:


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


def _render_panel_text(reg: LiveToolPanelRegistry, tool_use_id: str) -> str:

    c = Console(file=StringIO(), record=True)
    panel = reg.get(tool_use_id)
    assert panel is not None
    c.print(panel._render_panel())
    return c.export_text()


def test_taskoutput_header_unknown_id_falls_back_to_bare_id() -> None:


    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c2", "TaskOutput", {"task_id": "zzz999", "block": True, "timeout": 1})
    out = _render_panel_text(reg, "c2")
    assert "zzz999" in out and "block" not in out

def test_taskcreate_header_shows_subject_and_body() -> None:


    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "TaskCreate", {"subject": "Fix auth bug", "description": "details here"})
    out = _render_panel_text(reg, "c1")
    assert "Fix auth bug" in out and "details here" in out

def test_taskupdate_resolves_subject_and_shows_status() -> None:


    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "TaskCreate", {"subject": "Fix auth bug", "description": "d"})
    reg.observe_result("c1", "Task #1 created successfully: Fix auth bug")
    reg.create("c2", "TaskUpdate", {"taskId": "1", "status": "completed"})
    out = _render_panel_text(reg, "c2")
    assert "Fix auth bug" in out and "completed" in out

def test_tasklist_header_omits_empty_id_suffix() -> None:


    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "TaskList", {})
    out = _render_panel_text(reg, "c1")
    assert "TaskList" in out
    assert "(#)" not in out and "()" not in out

def test_taskoutput_result_shows_output_snippet() -> None:


    # Use normal mode to render the TaskOutput result body.
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


    reg = LiveToolPanelRegistry(Console(file=StringIO(), record=True), quiet_mode=True)
    reg.create("c1", "Task", {"description": "d", "prompt": "\n".join(f"l{i}" for i in range(40))})
    out = _render_panel_text(reg, "c1")
    assert f"({40 - _TASK_PROMPT_MAX_LINES} more lines)" in out
    assert "l0" in out
    assert "l39" not in out


def _taskoutput_backend() -> Any:
    """Build a backend stream containing a background task and its final output."""

    return ScriptedBackend(events=[ToolStartEvent(id="c1", name="Bash",
                input={"command": "pytest", "run_in_background": True, "description": "Run tests"},
            ), ToolResultEvent(id="c1", output="Command running in background with ID: a066168. ...", is_error=False,),
            ToolStartEvent(id="c2", name="TaskOutput", input={"task_id": "a066168", "block": True, "timeout": 120000},
            ), ToolResultEvent(id="c2",
                output="<task_id>a066168</task_id>\n<output>\ndone-with-bg-work\n</output>",
                is_error=False,
            ), ResultEvent(structured_output=None, continuation=None),
        ], model="mock-model",
    )

async def test_run_agent_renders_taskoutput_with_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Console(file=StringIO(), record=True, width=120)
    monkeypatch.setattr(agent_mod, "console", rec)
    backend = _taskoutput_backend()
    await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW)
    out = rec.export_text()
    assert "Run tests" in out and "a066168" in out
    assert "done-with-bg-work" in out
    assert "block=True" not in out and "timeout=120000" not in out

async def test_run_agent_callback_path_labels_taskoutput(tmp_path: Path) -> None:


    backend = _taskoutput_backend()
    lines: list[Text] = []
    await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW, progress_callback=lines.append,)
    joined = "\n".join(line.plain for line in lines)
    assert "Run tests" in joined  # resolved label surfaces in callback mode
    assert "block" not in joined and "timeout" not in joined
    assert "TaskOutput a066168" not in joined  # opaque bare-id dump form is gone

async def test_run_agent_callback_coalesces_streaming_text_deltas(tmp_path: Path) -> None:
    backend = ScriptedBackend(events=[TextEvent("B"), TextEvent("ash"), TextEvent(" is"), TextEvent(" blocked."),
            ResultEvent(structured_output=None, continuation=None),
        ], model="mock-model",
    )
    lines: list[Text] = []

    result, _, _ = await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX, progress_callback=lines.append,)

    assert result == "Bash is blocked."
    assert [line.plain for line in lines] == ["    Bash is blocked."]

async def test_run_agent_callback_path_edit_shows_file_not_bool(tmp_path: Path) -> None:
    """The parallel-fix callback line names the edited file, never a stray flag.

    Regression: the old blind ``next(iter(args.values()))`` surfaced a leading ``replace_all`` flag as ``"Edit
    False"`` instead of the file being edited."""


    backend = ScriptedBackend(events=[ToolStartEvent(id="e1", name="Edit",
                input={"replace_all": False, "file_path": "/repo/daydream/git_ops.py", "old_string": "a",
                    "new_string": "b",
                },
            ), ResultEvent(structured_output=None, continuation=None),
        ], model="mock-model",
    )
    lines: list[Text] = []
    await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.FIX, progress_callback=lines.append,)
    joined = "\n".join(line.plain for line in lines)
    assert "/repo/daydream/git_ops.py" in joined  # the meaningful primary arg
    assert "Edit False" not in joined  # the stray-boolean dump is gone


def test_callback_command_relies_on_the_owner_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The owner is the only capper of a command value (issue #1227).

    A stub owner returning more than max_len proves the callback passes the command through whole; a second
    `value[:max_len]` at the call site would truncate it. Non-command values keep the generic single-line cap."""
    long_value = "z" * (_BASH_COMMAND_MAX_CHARS + 50)
    monkeypatch.setattr(ui_tools, "_redacted_bash_command", lambda *a, **k: long_value)

    command_line = format_callback_progress("Bash", {"command": "anything"}, None)
    assert long_value in command_line.plain

    pattern = "p" * (_BASH_COMMAND_MAX_CHARS + 50)
    generic_line = format_callback_progress("Grep", {"pattern": pattern}, None)
    assert pattern[: _BASH_COMMAND_MAX_CHARS] in generic_line.plain
    assert pattern not in generic_line.plain

def test_format_callback_progress_redacts_only_bash_commands() -> None:
    """Callback redaction is scoped to the Bash command, like the panel header.

    Paths and grep patterns render raw on the panel header; the callback line must not rewrite the operator's own
    /home/<user>/ paths into [REDACTED_USER] markers or chew grep patterns into [REDACTED_CREDENTIAL]."""

    edit_line = format_callback_progress(
        "Edit", {"file_path": "/home/user/work/daydream/git_ops.py", "old_string": "a", "new_string": "b"}, None,
    )
    assert "/home/user/work/daydream/git_ops.py" in edit_line.plain
    assert "[REDACTED" not in edit_line.plain

    grep_line = format_callback_progress(
        "Grep", {"pattern": "the config: token=opaque-test-12345", "path": "/home/user/work"}, None
    )
    assert "opaque-test-12345" in grep_line.plain
    assert "[REDACTED" not in grep_line.plain

    bash_line = format_callback_progress("Bash", {"command": "the config: token=opaque-test-12345"}, None)
    assert "opaque-test-12345" not in bash_line.plain
    assert "[REDACTED" in bash_line.plain


_BOUNDARY_PAD = 185  # 200-char cap: a token starting here straddles it (15 of 20 chars inside)
_AKIA_TOKEN = "AKIA" + "Q7" * 8  # AKIA + 16 [A-Z0-9] — the pattern needs all 16


def _straddling_command(token: str, *, prefix: str = "") -> str:
    """A >cap command whose credential token starts at _BOUNDARY_PAD."""
    before_token = prefix + "echo " + "a" * (_BOUNDARY_PAD - len(prefix) - 12) + " --key "
    assert len(before_token) == _BOUNDARY_PAD
    return before_token + token + " tail"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["panel", "callback"])
async def test_run_agent_command_display_preserves_replayable_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    command = "cd /srv/app && " + _straddling_command(_AKIA_TOKEN)
    tool_event = ToolStartEvent(id="command-1", name="shell", input={"command": command})
    backend = ScriptedBackend(
        events=[tool_event, ResultEvent(structured_output=None, continuation=None)], model="mock-model",
    )
    if mode == "panel":
        rec = Console(file=StringIO(), record=True, width=500)
        monkeypatch.setattr(agent_mod, "console", rec)
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW)
        displayed = rec.export_text()
    else:
        lines: list[Text] = []
        await run_agent(backend, tmp_path, "go", phase=DaydreamPhase.REVIEW, progress_callback=lines.append)
        displayed = "\n".join(line.plain for line in lines)

    assert _AKIA_TOKEN[:8] not in displayed
    assert "[REDACTED" in displayed
    assert "cd /srv/app" not in displayed
    assert tool_event.input == {"command": command}

def test_render_segments_skips_span_contained_in_an_earlier_one() -> None:
    """Longest-first ordering means a contained span is skipped, not double-styled."""
    source = "abcdefgh"
    wide = Style(bold=True)
    narrow = Style(italic=True)
    rendered = render_segments(
        source, [(0, 4, source[0:4], wide), (1, 3, source[1:3], narrow), (4, 8, source[4:8], narrow)], Style(),
    )
    assert rendered.plain == source
    assert [(span.start, span.end, str(span.style)) for span in rendered.spans] == [(0, 4, "bold"), (4, 8, "italic")]
