"""Tests for daydream.ui helpers."""

from __future__ import annotations

import json
import re
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
from daydream.backends import AgentEvent, ResultEvent, TextEvent, ThinkingEvent, ToolResultEvent, ToolStartEvent
from daydream.exploration import Convention, Dependency, ExplorationContext, FileInfo
from daydream.run_context import InteractionPolicy, RunContext, resolve_run_context
from daydream.trajectory import DaydreamPhase
from daydream.ui import (
    NEON_THEME,
    AgentTextRenderer,
    format_verdict_join,
    print_verification_summary,
    render_exploration_summary,
)
from daydream.ui.agent_stream import AgentDisplay
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
            Convention(name="OpenAPI First", description="openapi.yaml is the HTTP contract", source="AGENTS.md")
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


@pytest.mark.parametrize('layout', ['wrapped', 'line_cut'])
async def test_tool_result_redacts_native_auth_before_rich_wrap_or_line_cut(
    layout: str,
) -> None:
    """Redact the complete tool result before Rich wrapping or line cuts lose its credential tag."""
    secret = 'opaque-ui-native-auth-canary'
    credentials = {'key': secret}
    payload: dict[str, object] = {**credentials, 'public': {'key': 'ordinary-public-key'}}
    if layout == 'line_cut':
        payload.update({f'public_{index}': index for index in range(30)})
    payload['type'] = 'api_key'
    content = json.dumps(payload, indent=2 if layout == 'line_cut' else None)
    event = ToolResultEvent('credential', content, is_error=False)
    console = Console(file=StringIO(), record=True, width=42, force_terminal=False)
    display = AgentDisplay(console, InteractionPolicy(), None, structured=False)
    await display.observe(ToolStartEvent('credential', 'Read', {'file_path': 'fixture-input.json'}))
    await display.observe(event)
    display.tools.finish_all()
    rendered = ''.join(console.export_text().replace('│', '').split())
    assert secret not in rendered
    assert 'ordinary-public-key' in rendered and '[REDACTED' in rendered
    assert event.output == content


@pytest.mark.parametrize('route', ['panel', 'callback', 'thinking', 'log'])
async def test_text_displays_redact_complete_native_auth_before_selecting_or_formatting(
    route: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """Narration, callbacks, thinking caps, and logs redact complete buffered context before formatting."""
    secret = 'opaque-ui-text-native-auth-canary'
    fields = {'key': secret}
    public = {'public': {'key': 'ordinary-public-key'}}
    payload = ({**fields, **public, 'padding': 'x' * 400, 'type': 'api_key'} if route == 'thinking' else
               {'type': 'api_key', **fields, **public})
    content = json.dumps(payload).replace(', "', ',\n"', 1)
    console = Console(file=StringIO(), record=True, width=40, force_terminal=False)
    callback: list[Text] = []
    display = AgentDisplay(console, InteractionPolicy(log_mode=route == 'log'),
        callback.append if route == 'callback' else None, structured=False)
    if route == 'thinking':
        thinking = ThinkingEvent(content)
        await display.observe(thinking)
        assert thinking.text == content
    else:
        split = content.index(secret)
        events = [TextEvent(content[:split]), TextEvent(content[split:])]
        for event in events:
            await display.observe(event)
        await display.flush()
        display.text.finish()
        assert ''.join(event.text for event in events) == content
    display.tools.finish_all()
    rendered = console.export_text() + '\n'.join(value.plain for value in callback) + capsys.readouterr().out
    visible = ''.join(rendered.replace('│', '').split())
    assert secret not in visible
    assert 'ordinary-public-key' in visible and '[REDACTED' in visible


@pytest.mark.parametrize('route', ['direct', 'callback', 'panel', 'log'])
async def test_callback_tool_input_redacts_native_auth_before_primary_argument_selection(
    route: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """Each public tool-input presentation route protects opaque native fields before selecting a value."""
    secret = 'opaque-ui-callback-native-auth-canary'
    values: dict[str, object] = {'key': secret}
    values['type'] = 'api_key'
    before = dict(values)
    public: dict[str, object] = {'key': 'ordinary-public-key', 'type': 'record'}
    console = Console(file=StringIO(), record=True, width=40, force_terminal=False)
    lines: list[Text] = []
    if route == 'direct':
        lines = [format_callback_progress('Inspect', values, None), format_callback_progress('Inspect', public, None)]
    else:
        display = AgentDisplay(console, InteractionPolicy(log_mode=route == 'log'),
            lines.append if route == 'callback' else None, structured=False)
        await display.observe(ToolStartEvent('credential', 'Inspect', values))
        await display.observe(ToolStartEvent('public', 'Inspect', public))
        display.tools.finish_all()
    rendered = ''.join((console.export_text() + capsys.readouterr().out +
                        '\n'.join(line.plain for line in lines)).replace('│', '').split())
    assert secret not in rendered and '[REDACTED' in rendered
    assert 'ordinary-public-key' in rendered
    assert values == before


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


def _task_lifecycle_events() -> list[AgentEvent]:
    """One background job and one tracked task move through their actual event lifecycle."""
    return [
        ToolStartEvent(id="launch", name="Bash", input={"command": "pytest", "run_in_background": True,
                       "description": "Run tests"}),
        ToolResultEvent(id="launch",
                        output=(Path(__file__).parent / "fixtures/task_tools/bash_bg_launch.txt").read_text(),
                        is_error=False),
        ToolStartEvent(id="create", name="TaskCreate",
                       input={"subject": "Fix auth bug", "description": "details here"}),
        ToolResultEvent(id="create", output="Task #1 created successfully: Fix auth bug", is_error=False),
        ToolStartEvent(id="update", name="TaskUpdate", input={"taskId": "1", "status": "completed"}),
        ToolResultEvent(id="update", output="Task updated", is_error=False),
        ToolStartEvent(id="list", name="TaskList", input={}),
        ToolResultEvent(id="list", output="Tracked task completed", is_error=False),
        ToolStartEvent(id="known-output", name="TaskOutput", input={"task_id": "b0nsmwb99", "block": True,
                       "timeout": 120000}),
        ToolResultEvent(id="known-output", output=(Path(__file__).parent / "fixtures/task_tools/taskoutput_result.txt")
                        .read_text(), is_error=False),
        ToolStartEvent(id="unknown-output", name="TaskOutput", input={"task_id": "zzz999", "block": True,
                       "timeout": 1}),
        ToolResultEvent(id="unknown-output", output="Untracked task returned", is_error=False),
        ToolStartEvent(id="audit", name="Task", input={"description": "bounded audit",
                       "prompt": "\n".join(f"prompt-line-{i}" for i in range(40))}),
        ToolResultEvent(id="audit", output="Audit finished", is_error=False),
        ToolStartEvent(id="edit", name="Edit", input={"replace_all": False,
                       "file_path": "/home/user/work/daydream/git_ops.py", "old_string": "a", "new_string": "b"}),
        ToolResultEvent(id="edit", output="", is_error=False),
        ToolStartEvent(id="grep", name="Grep", input={"pattern": "the config: token=literal-grep-pattern",
                       "path": "/home/user/work"}),
        ToolResultEvent(id="grep", output="", is_error=False),
    ]


async def test_normal_agent_transcript_preserves_overlapping_tools_tasks_and_thought_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native sibling tool events keep their results and labels through a single panel lifetime."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, no_color=False, width=120, theme=NEON_THEME)
    monkeypatch.setattr(agent_mod, "console", console)
    events = [
        *[ThinkingEvent(text=f"thought {i} of the stream") for i in range(5)],
        TextEvent(text="I'll search for Python files in the project."),
        ToolStartEvent(id="cmd-1", name="shell", input={"command": "git diff -- file1.py"}),
        ToolStartEvent(id="cmd-2", name="shell", input={"command": "git diff -- file2.py"}),
        ToolStartEvent(id="cmd-3", name="shell", input={"command": "git diff -- file3.py"}),
        ToolResultEvent(id="cmd-3", output="+added line in file3", is_error=False),
        ToolResultEvent(id="cmd-1", output="+added line in file1", is_error=False),
        ToolResultEvent(id="cmd-2", output="+added line in file2", is_error=False),
        ToolStartEvent(id="glob-three", name="Glob", input={"pattern": "**/*.py", "path": "/project"}),
        ToolResultEvent(id="glob-three", output="/project/src/main.py\n/project/src/utils/helper.py\n"
                        "/project/tests/test_main.py", is_error=False),
        ToolStartEvent(id="glob-one", name="Glob", input={"pattern": "*.py"}),
        ToolResultEvent(id="glob-one", output="/project/main.py", is_error=False),
        ToolStartEvent(id="glob-wide", name="Glob", input={"pattern": "**/*.py"}),
        ToolResultEvent(id="glob-wide", output="\n".join(f"/project/src/module{i}.py" for i in range(25)),
                        is_error=False),
        *_task_lifecycle_events(), ResultEvent(structured_output=None, continuation=None),
    ]
    await run_agent(ScriptedBackend(events=events, model="mock-model"), tmp_path, "inspect and repair",
                    phase=DaydreamPhase.REVIEW)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output.getvalue())
    assert plain.count("Thinking") == 5
    positions = [plain.index(f"thought {i} of the stream") for i in range(5)]
    assert positions == sorted(positions)
    assert "I'll search for Python files" in plain and "**/*.py" in plain
    assert all(f"git diff -- file{i}.py" in plain and f"+added line in file{i}" in plain for i in (1, 2, 3))
    assert all(value in plain for value in ("Found 3 files", "main.py", "helper.py", "test_main.py"))
    assert "Found 1 file" in plain and "Found 1 files" not in plain
    assert "Found 25 files" in plain and "and 5 more" in plain
    assert "module19.py" in plain and "module20.py" not in plain
    assert all(value in plain for value in
               ("Run tests", "b0nsmwb99", "Fix auth bug", "completed", "details here", "zzz999"))
    assert "done-with-bg-work" in plain and "<retrieval_status>" not in plain
    assert "block=True" not in plain and "timeout=120000" not in plain and "TaskList (#)" not in plain
    assert f"({40 - _TASK_PROMPT_MAX_LINES} more lines)" in plain
    assert "prompt-line-0" in plain and "prompt-line-39" not in plain
    assert not console._live_stack


async def test_quiet_agent_transcript_preserves_headers_redaction_and_error_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quiet delivery hides result bodies while its headers retain task identity and safe commands."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, no_color=False, width=120, color_system="truecolor",
                      theme=NEON_THEME)
    monkeypatch.setattr(agent_mod, "console", console)
    events = [
        ToolStartEvent(id="read", name="Read", input={"file_path": "/project/main.py"}),
        ToolResultEvent(id="read", output="def hello():\n    return 'world'", is_error=False),
        ToolStartEvent(id="stat", name="Bash", input={"command": "git diff --stat"}),
        ToolResultEvent(id="stat", output="", is_error=False),
        ToolStartEvent(id="list-shell", name="Bash", input={"command": "ls -la /tmp"}),
        ToolResultEvent(id="list-shell", output="", is_error=False),
        ToolStartEvent(id="secret", name="Bash", input={"command": "DB_PASSWORD=hunter2 make db-up"}),
        ToolResultEvent(id="secret", output="", is_error=False),
        ToolStartEvent(id="cap", name="Bash", input={"command": "x" * 180 + " token=opaque-test-12345"}),
        ToolResultEvent(id="cap", output="", is_error=False),
        ToolStartEvent(id="empty", name="Bash", input={"command": "true"}),
        ToolResultEvent(id="empty", output="", is_error=False),
        ToolStartEvent(id="error", name="Bash", input={"command": "false"}),
        ToolResultEvent(id="error", output="Command failed with exit code 1", is_error=True),
        ToolStartEvent(id="skill", name="Skill", input={"skill": "review-python"}),
        ToolResultEvent(id="skill", output="Launching skill: review-python", is_error=False),
        *_task_lifecycle_events(), ResultEvent(structured_output=None, continuation=None),
    ]
    await run_agent(ScriptedBackend(events=events, model="mock-model"), tmp_path, "quiet inspection",
                    phase=DaydreamPhase.REVIEW, run_context=RunContext(InteractionPolicy(quiet=True)))
    raw = output.getvalue()
    plain = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    assert all(value in plain for value in ("Read", "/project/main.py", "$ git diff --stat", "$ ls -la /tmp", "$ true",
                                            "$ false", "Skill", "review-python"))
    assert all(value not in plain for value in
               ("hello", "world", "Command failed", "Launching skill:", "done-with-bg-work"))
    assert not any(line.strip(" │") == "Output" for line in plain.splitlines())
    assert "╭" in raw or "│" in raw
    assert re.search(r"\x1b\[[0-9;]*38;2;255;85;85(?:;[0-9;]+)?m[╭│╰─]", raw)
    assert "hunter2" not in plain and "$ DB_PASSWORD=[REDACTED_ENV_VAR] make db-up" in plain
    assert "opaque" not in plain and 'token="[REDA' in plain and "[REDACTED_CREDENTIAL]" not in plain
    assert all(value in plain for value in
               ("Run tests", "b0nsmwb99", "Fix auth bug", "completed", "details here", "zzz999"))
    assert "block=True" not in plain and "timeout=120000" not in plain
    assert f"({40 - _TASK_PROMPT_MAX_LINES} more lines)" in plain
    assert "prompt-line-0" in plain and "prompt-line-39" not in plain
    assert not console._live_stack


async def test_callback_agent_transcript_keeps_task_labels_edit_identity_and_coalesced_text(tmp_path: Path) -> None:
    """The callback gets semantic task/file labels while only command values are redacted."""
    events = [*_task_lifecycle_events(),
        ToolStartEvent(id="command", name="Bash", input={"command": "the config: token=opaque-test-12345"}),
        ToolResultEvent(id="command", output="", is_error=False),
        TextEvent("B"), TextEvent("ash"), TextEvent(" is"), TextEvent(" blocked."),
        ResultEvent(structured_output=None, continuation=None),
    ]
    lines: list[Text] = []
    result, _, _ = await run_agent(ScriptedBackend(events=events, model="mock-model"), tmp_path, "fix the tracked task",
                                    phase=DaydreamPhase.FIX, progress_callback=lines.append)
    assert result == "Bash is blocked."
    plain = [line.plain for line in lines]
    assert [line for line in plain if "Bash is blocked." in line] == ["    Bash is blocked."]
    assert "    B" not in plain and "    ash" not in plain
    tools = "\n".join(line for line in plain if "Bash is blocked." not in line)
    assert "    🎠 TaskUpdate Fix auth bug (1)" in plain
    assert "    🎠 TaskOutput Run tests (b0nsmwb99)" in plain
    assert "    🎠 TaskOutput zzz999" in plain
    assert "TaskOutput b0nsmwb99" not in tools and "block" not in tools and "timeout" not in tools
    assert "/home/user/work/daydream/git_ops.py" in tools and "Edit False" not in tools
    assert "the config: token=literal-grep-pattern" in tools and "[REDACTED_USER]" not in tools
    assert "opaque-test-12345" not in tools and "[REDACTED" in tools
