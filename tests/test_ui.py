"""Tests for daydream.ui helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

import daydream.agent as agent_mod
from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.exploration import ExplorationContext
from daydream.run_context import InteractionPolicy, RunContext, resolve_run_context
from daydream.trajectory import DaydreamPhase
from daydream.ui import (
    NEON_THEME,
    ShutdownPanel,
    print_verification_summary,
    render_exploration_summary,
)
from daydream.ui.agent_stream import AgentDisplay
from daydream.ui.panels import LiveToolPanelRegistry
from daydream.ui.theme import _TASK_PROMPT_MAX_LINES
from daydream.ui.tools import format_callback_progress
from tests.harness.backend import ScriptedBackend


def _capture_console(fn: Callable[[Console], None]) -> str:
    """Run *fn* against a recording console and return the captured text."""
    console = Console(file=StringIO(), record=True, force_terminal=True, width=200)
    fn(console)
    return console.export_text()


def _ask(console: Console, message: str, default: str = "n") -> str:
    """Ask through the run's interaction gateway, the sole owner of prompt policy."""
    return resolve_run_context().choice(message, default=default, safe_default=default, console=console)


def test_summaries_stay_quiet_and_choice_returns_the_typed_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "recommendation-verdicts.json"
    path.write_text(json.dumps({"verdicts": []}))
    out = _capture_console(lambda c: print_verification_summary(c, path))
    assert "selected" not in out
    assert "Recommendation verification: 0 findings" in out

    console = Console(file=StringIO(), record=True, force_terminal=True, width=100)
    console.print(render_exploration_summary(ExplorationContext()))
    out = console.export_text()
    assert "{" not in out and "[" not in out  # never dumps a structure; one dim line at most

    monkeypatch.setattr("builtins.input", lambda: "y")
    assert _ask(Console(), "Confirm?", default="n") == "y"

async def test_log_lines_live_headers_and_shutdown_steps_render_every_input_shape(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Log mode summarizes path/generic inputs and usage; live headers render Write, None and blank-line text."""
    log = AgentDisplay(Console(file=StringIO()), InteractionPolicy(log_mode=True), None, structured=False)
    for event in (ToolStartEvent("mv", "Move", {"path": "a.py", "new_path": "b.py"}),
                  ToolStartEvent("odd", "Custom", {"dry_run": True, "note": "generic value"}),
                  MetricsEvent("m", 12, 3, None, None), CostEvent(None, None, None), CostEvent(0.5, 1, 1)):
        await log.observe(event)
    out = capsys.readouterr().out
    assert "a.py -> b.py" in out and "generic value" in out
    assert "[metrics] prompt=12 completion=3" in out
    assert "[cost] unknown" in out and "[cost] $0.5000" in out

    console = Console(file=StringIO(), record=True, width=100, force_terminal=False)
    live = AgentDisplay(console, InteractionPolicy(), None, structured=False)
    await live.observe(ToolStartEvent("w", "Write", {"file_path": "out.py", "content": "x"}))
    await live.observe(ToolStartEvent("u", "Custom", {"dry_run": True, "limit": None, "note": "generic value"}))
    await live.observe(TextEvent("first\n\nsecond"))
    await live.flush()
    live.text.finish()
    live.tools.finish_all()
    panel = ShutdownPanel(console)
    panel.start("stopped")
    panel.complete_step(panel.add_step("cleanup"))
    panel.finish()
    text = console.export_text()
    assert all(value in text for value in ("out.py", "limit=None", "first", "second", "stopped", "cleanup"))
    # A boolean, mechanical or non-string argument is never chosen as an unknown tool's primary value.
    progress = format_callback_progress("Custom", {"dry_run": True, "count": 3, "note": "generic value"}, None)
    assert "generic value" in progress.plain
    assert not console._live_stack


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


def _visible(console: Console, capsys: pytest.CaptureFixture[str], lines: list[Text]) -> str:
    rendered = console.export_text() + '\n'.join(value.plain for value in lines) + capsys.readouterr().out
    return ''.join(rendered.replace('│', '').split())


async def test_displays_redact_complete_native_auth_before_selecting_or_formatting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Native auth is redacted as a whole document before text selection, tool-argument choice, wrap or line cut.

    Each route owns a different cut: streamed text split mid-secret, a wrapped or
    line-capped tool result, and the primary tool argument chosen for a header.
    """
    secret = 'opaque-ui-native-auth-canary'
    public = {'public': {'key': 'ordinary-public-key'}}
    for route in ('thinking', 'log'):
        payload = ({'key': secret, **public, 'padding': 'x' * 400, 'type': 'api_key'} if route == 'thinking' else
                   {'type': 'api_key', 'key': secret, **public})
        content = json.dumps(payload).replace(', "', ',\n"', 1)
        console = Console(file=StringIO(), record=True, width=40, force_terminal=False)
        display = AgentDisplay(console, InteractionPolicy(log_mode=route == 'log'), None, structured=False)
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
        visible = _visible(console, capsys, [])
        assert secret not in visible, route
        assert 'ordinary-public-key' in visible and '[REDACTED' in visible

    for layout in ('wrapped', 'line_cut'):
        result: dict[str, object] = {'key': secret, **public}
        if layout == 'line_cut':
            result.update({f'public_{index}': index for index in range(30)})
        result['type'] = 'api_key'
        content = json.dumps(result, indent=2 if layout == 'line_cut' else None)
        output = ToolResultEvent('credential', content, is_error=False)
        console = Console(file=StringIO(), record=True, width=42, force_terminal=False)
        display = AgentDisplay(console, InteractionPolicy(), None, structured=False)
        await display.observe(ToolStartEvent('credential', 'Read', {'file_path': 'fixture-input.json'}))
        await display.observe(output)
        display.tools.finish_all()
        visible = _visible(console, capsys, [])
        assert secret not in visible, layout
        assert 'ordinary-public-key' in visible and '[REDACTED' in visible
        assert output.output == content

    values: dict[str, object] = {'key': secret, 'type': 'api_key'}
    before = dict(values)
    record: dict[str, object] = {'key': 'ordinary-public-key', 'type': 'record'}
    for route in ('direct', 'callback', 'panel', 'log'):
        console = Console(file=StringIO(), record=True, width=40, force_terminal=False)
        lines: list[Text] = []
        if route == 'direct':
            lines = [format_callback_progress('Inspect', args, None) for args in (values, record)]
        else:
            display = AgentDisplay(console, InteractionPolicy(log_mode=route == 'log'),
                lines.append if route == 'callback' else None, structured=False)
            await display.observe(ToolStartEvent('credential', 'Inspect', values))
            await display.observe(ToolStartEvent('public', 'Inspect', record))
            display.tools.finish_all()
        visible = _visible(console, capsys, lines)
        assert secret not in visible and '[REDACTED' in visible, route
        assert 'ordinary-public-key' in visible
        assert values == before


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
    console = Console(file=output, force_terminal=True, no_color=False, width=120, color_system="truecolor",
                      theme=NEON_THEME)
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
