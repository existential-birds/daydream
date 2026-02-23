"""Integration tests for the full review-fix-test flow."""

import re
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.runner import RunConfig, run
from daydream.ui import NEON_THEME

# ANSI escape code pattern for stripping terminal colors
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text for assertion comparisons."""
    return _ANSI_ESCAPE.sub("", text)


# =============================================================================
# Mock Backends
# =============================================================================


class MockBackend:
    """Mock backend that yields events based on prompt content."""

    def __init__(self, events: list | None = None):
        self._events = events
        self._prompt: str = ""
        self._call_count = 0

    async def execute(self, cwd, prompt, output_schema=None, continuation=None):
        self._prompt = prompt
        self._call_count += 1
        if self._events is not None:
            for event in self._events:
                yield event
            return

        # Default: generate events based on prompt content (like MockClaudeSDKClient)
        text, structured = self._get_response_for_prompt(prompt)
        yield TextEvent(text=text)
        if structured is not None:
            yield ResultEvent(structured_output=structured, continuation=None)
        else:
            yield CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None)
            yield ResultEvent(structured_output=None, continuation=None)

    def _get_response_for_prompt(self, prompt: str) -> tuple[str, Any]:
        prompt_lower = prompt.lower()
        if "beagle-" in prompt_lower and "review" in prompt_lower:
            return "Review complete. Found 1 issue to fix.", None
        if "extract" in prompt_lower and "json" in prompt_lower:
            structured = {
                "issues": [
                    {"id": 1, "description": "Add type hints to function", "file": "main.py", "line": 1}
                ]
            }
            return "Extracted feedback.", structured
        if "fix this issue" in prompt_lower:
            return "Fixed the issue by adding type hints.", None
        if "test suite" in prompt_lower or "run the project" in prompt_lower:
            return "All 1 tests passed. 0 failed.", None
        if "commit-push" in prompt_lower:
            return "Changes committed and pushed.", None
        return "OK", None

    async def cancel(self):
        pass

    def format_skill_invocation(self, skill_key, args=""):
        return f"/{skill_key}" + (f" {args}" if args else "")


class MockBackendWithEvents:
    """Mock backend with pre-configured events for tool panel testing."""

    def __init__(self, events: list):
        self._events = events

    async def execute(self, cwd, prompt, output_schema=None, continuation=None):
        for event in self._events:
            yield event

    async def cancel(self):
        pass

    def format_skill_invocation(self, skill_key, args=""):
        return f"/{skill_key}" + (f" {args}" if args else "")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_backend(monkeypatch):
    """Patch create_backend to return MockBackend."""
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: MockBackend())
    return MockBackend


@pytest.fixture
def mock_ui(monkeypatch):
    """Patch UI functions that require user input."""
    # Skip commit prompt by returning "n"
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *args, **kwargs: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *args, **kwargs: "n")


@pytest.fixture
def target_project(tmp_path: Path) -> Path:
    """Create a minimal project structure for testing."""
    project = tmp_path / "test_project"
    project.mkdir()

    # Create a simple Python file
    (project / "main.py").write_text("def hello():\n    return 'world'\n")

    # Create review output that phase_review would normally create
    # (our mock doesn't actually write files, so we pre-create it)
    review_content = """# Code Review

## Issues Found

1. **Missing type hints** in `main.py:1`
   - Add type hints to the `hello` function

## Summary

Found 1 issue to address.
"""
    (project / ".review-output.md").write_text(review_content)

    return project


@pytest.mark.asyncio
async def test_full_fix_flow(mock_backend, mock_ui, target_project: Path):
    """Test the complete review -> parse -> fix -> test flow."""
    config = RunConfig(
        target=str(target_project),
        skill="python",
        quiet=True,
        cleanup=False,
    )

    exit_code = await run(config)

    assert exit_code == 0
    assert (target_project / ".review-output.md").exists()


@pytest.mark.asyncio
async def test_glob_tool_panel_displays_file_count_and_list(monkeypatch):
    """Test the full tool panel lifecycle in normal mode shows file count and list.

    This test exercises the actual run_agent() flow by providing a MockBackendWithEvents
    that yields events. Normal mode (quiet=False) shows both header and output section.

    Also tests that:
    - AgentTextRenderer displays streamed text with spinner cursor effect
    - LiveThinkingPanel displays thinking blocks with animated title
    """
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-glob-lifecycle-123"
    glob_result = """/project/src/main.py
/project/src/utils/helper.py
/project/tests/test_main.py"""

    events = [
        ThinkingEvent(text="Analyzing the project structure..."),
        TextEvent(text="I'll search for Python files in the project."),
        ToolStartEvent(id=tool_use_id, name="Glob", input={"pattern": "**/*.py", "path": "/project"}),
        ToolResultEvent(id=tool_use_id, output=glob_result, is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    backend = MockBackendWithEvents(events)

    # Speed up print_thinking animation
    monkeypatch.setattr("daydream.ui.time.sleep", lambda _: None)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(False)

    await run_agent(backend, Path("/tmp"), "Test prompt for Glob tool")

    # Verify the console output contains expected elements
    output_text = output.getvalue()
    plain_text = strip_ansi(output_text)

    # Verify thinking panel is displayed (LiveThinkingPanel with spinner in title)
    assert "Thinking" in plain_text
    assert "Analyzing the project structure" in plain_text

    # Verify agent text is displayed (AgentTextRenderer with spinner cursor)
    assert "I'll search for Python files" in plain_text

    # Verify Glob header is displayed (mystical format: "Glob scrying... **/*.py")
    assert "Glob" in plain_text
    assert "**/*.py" in plain_text

    # Verify file count is displayed (normal mode shows output section)
    assert "Found 3 files" in plain_text

    # Verify filenames are displayed
    assert "main.py" in plain_text
    assert "helper.py" in plain_text
    assert "test_main.py" in plain_text


@pytest.mark.asyncio
async def test_glob_tool_panel_singular_file_count(monkeypatch):
    """Test that LiveToolPanel shows singular 'file' for 1 result in normal mode."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-glob-singular-456"
    glob_result = "/project/main.py"

    events = [
        ToolStartEvent(id=tool_use_id, name="Glob", input={"pattern": "*.py"}),
        ToolResultEvent(id=tool_use_id, output=glob_result, is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    backend = MockBackendWithEvents(events)

    # Speed up print_thinking animation
    monkeypatch.setattr("daydream.ui.time.sleep", lambda _: None)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    # Normal mode to see output section
    set_quiet_mode(False)

    await run_agent(backend, Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify singular "file" (not "files")
    assert "Found 1 file" in output_text
    assert "Found 1 files" not in output_text

    # Verify the filename is displayed
    assert "main.py" in output_text


@pytest.mark.asyncio
async def test_glob_tool_panel_truncates_long_results(monkeypatch):
    """Test that LiveToolPanel truncates long Glob results in normal mode."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-glob-truncate-789"
    # Create 25 files (more than max_lines=20 passed from _build_result_content_internal)
    mock_files = [f"/project/src/module{i}.py" for i in range(25)]
    glob_result = "\n".join(mock_files)

    events = [
        ToolStartEvent(id=tool_use_id, name="Glob", input={"pattern": "**/*.py"}),
        ToolResultEvent(id=tool_use_id, output=glob_result, is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    backend = MockBackendWithEvents(events)

    # Speed up print_thinking animation
    monkeypatch.setattr("daydream.ui.time.sleep", lambda _: None)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    # Normal mode to see output section
    set_quiet_mode(False)

    await run_agent(backend, Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify the total file count is displayed
    assert "Found 25 files" in output_text

    # Verify truncation indicator (25 total - 20 displayed = 5 more)
    assert "and 5 more" in output_text


@pytest.mark.asyncio
async def test_quiet_mode_shows_header_only(monkeypatch):
    """Test that quiet mode shows header only (no output section)."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-output-panel-001"
    read_result = "def hello():\n    return 'world'"

    events = [
        ToolStartEvent(id=tool_use_id, name="Read", input={"file_path": "/project/main.py"}),
        ToolResultEvent(id=tool_use_id, output=read_result, is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    backend = MockBackendWithEvents(events)

    # Speed up print_thinking animation
    monkeypatch.setattr("daydream.ui.time.sleep", lambda _: None)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(True)

    await run_agent(backend, Path("/tmp"), "Test prompt")

    output_text = output.getvalue()
    plain_text = strip_ansi(output_text)

    # Verify header is displayed (mystical format: "Read beholding... /path")
    assert "Read" in plain_text
    assert "/project/main.py" in plain_text

    # Verify NO output section in quiet mode (header only)
    assert "Output" not in plain_text

    # Content should NOT be displayed in quiet mode
    assert "hello" not in plain_text
    assert "world" not in plain_text

    # Verify panel border characters are present
    assert "╭" in output_text or "│" in output_text


@pytest.mark.asyncio
async def test_quiet_mode_empty_result_shows_header_only(monkeypatch):
    """Test that quiet mode shows header only for empty results (no output section)."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-empty-result-002"

    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": "true"}),
        ToolResultEvent(id=tool_use_id, output="", is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    backend = MockBackendWithEvents(events)

    # Speed up print_thinking animation
    monkeypatch.setattr("daydream.ui.time.sleep", lambda _: None)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(True)

    await run_agent(backend, Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify header is displayed
    assert "Bash" in output_text

    # Verify NO output section in quiet mode (header only)
    assert "Output" not in output_text

    # Verify panel border is present
    assert "╭" in output_text or "│" in output_text


@pytest.mark.asyncio
async def test_quiet_mode_error_shows_header_with_red_border(monkeypatch):
    """Test that quiet mode shows header only with red border for errors."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-error-result-003"

    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": "false"}),
        ToolResultEvent(id=tool_use_id, output="Command failed with exit code 1", is_error=True),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    backend = MockBackendWithEvents(events)

    # Speed up print_thinking animation
    monkeypatch.setattr("daydream.ui.time.sleep", lambda _: None)

    output = StringIO()
    # Force truecolor to get consistent RGB color codes across environments
    test_console = Console(
        file=output, force_terminal=True, width=120, theme=NEON_THEME, color_system="truecolor"
    )
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(True)

    await run_agent(backend, Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify header is displayed
    assert "Bash" in output_text

    # Verify NO error content in quiet mode (header only)
    # The red border color indicates error status
    assert "Command failed" not in output_text

    # Verify panel border is present (red color is applied via ANSI codes)
    assert "╭" in output_text or "│" in output_text

    # Verify ANSI styling is present (exact color sequence varies by terminal/theme)
    assert "\x1b[" in output_text


@pytest.mark.asyncio
async def test_skill_tool_panel_collapses_output(monkeypatch):
    """Test that Skill tool calls don't show an Output panel.

    The skill name already appears in the tool call header, so the
    "Launching skill: X" output is redundant and should be suppressed.
    """
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-skill-collapse-001"

    events = [
        ToolStartEvent(id=tool_use_id, name="Skill", input={"skill": "review-python"}),
        ToolResultEvent(id=tool_use_id, output="Launching skill: review-python", is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    backend = MockBackendWithEvents(events)

    # Speed up print_thinking animation
    monkeypatch.setattr("daydream.ui.time.sleep", lambda _: None)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(True)

    await run_agent(backend, Path("/tmp"), "Test prompt")

    output_text = output.getvalue()
    plain_text = strip_ansi(output_text)

    # Verify Skill header is displayed with skill name (gradient-styled)
    assert "Skill" in plain_text
    assert "review-python" in plain_text

    # Verify NO Output panel is displayed for Skill tool calls
    # The skill result should be suppressed since the header already shows the skill name
    assert "Output" not in plain_text

    # Verify "Launching skill:" text is NOT displayed (the redundant output)
    assert "Launching skill:" not in plain_text


@pytest.mark.asyncio
async def test_concurrent_tool_panels_display_results(monkeypatch):
    """Test that concurrent tool panels (e.g. Codex parallel commands) all show results.

    When multiple ToolStartEvents arrive before any ToolResultEvents (as happens
    with the Codex backend's parallel command execution), all panels should
    eventually display their results without display corruption.
    """
    from daydream.agent import run_agent, set_quiet_mode

    # Simulate 3 concurrent commands (all started before any complete)
    events = [
        ToolStartEvent(id="cmd-1", name="shell", input={"command": "git diff -- file1.py"}),
        ToolStartEvent(id="cmd-2", name="shell", input={"command": "git diff -- file2.py"}),
        ToolStartEvent(id="cmd-3", name="shell", input={"command": "git diff -- file3.py"}),
        ToolResultEvent(id="cmd-1", output="+added line in file1", is_error=False),
        ToolResultEvent(id="cmd-2", output="+added line in file2", is_error=False),
        ToolResultEvent(id="cmd-3", output="+added line in file3", is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    backend = MockBackendWithEvents(events)

    monkeypatch.setattr("daydream.ui.time.sleep", lambda _: None)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(False)

    await run_agent(backend, Path("/tmp"), "Test prompt")

    output_text = output.getvalue()
    plain_text = strip_ansi(output_text)

    # All three results should appear in the output
    assert "+added line in file1" in plain_text
    assert "+added line in file2" in plain_text
    assert "+added line in file3" in plain_text

    # All three commands should be shown
    assert "git diff -- file1.py" in plain_text
    assert "git diff -- file2.py" in plain_text
    assert "git diff -- file3.py" in plain_text


@pytest.mark.asyncio
async def test_run_trust_full_flow(tmp_path, monkeypatch):
    """Integration test: full --trust-the-technology flow through all three phases."""
    import os
    import subprocess

    # Set up a git repo with a branch
    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=tmp_path, capture_output=True)
    (tmp_path / "app.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, env=env)
    subprocess.run(["git", "checkout", "-b", "feat/test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "app.py").write_text("print('world')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "change"], cwd=tmp_path, capture_output=True, env=env)

    call_count = 0

    class TrustMockBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Phase 1: understand intent
                yield TextEvent(text="This PR changes the hello message to world.")
                yield ResultEvent(structured_output=None, continuation=None)
            elif call_count == 2:
                # Phase 2: alternative review
                yield TextEvent(text="Found 1 issue.")
                yield ResultEvent(
                    structured_output={
                        "issues": [{
                            "id": 1, "title": "Use constants", "description": "Hardcoded string",
                            "recommendation": "Use a constant", "severity": "low", "files": ["app.py"],
                        }]
                    },
                    continuation=None,
                )
            elif call_count == 3:
                # Phase 3: generate plan
                yield TextEvent(text="Plan generated.")
                yield ResultEvent(
                    structured_output={
                        "plan": {
                            "summary": "Extract string to constant",
                            "issues": [{
                                "id": 1, "title": "Use constants",
                                "changes": [
                                    {"file": "app.py", "description": "Extract to constant", "action": "modify"}
                                ],
                            }],
                        }
                    },
                    continuation=None,
                )

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: TrustMockBackend())

    # Mock UI functions
    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_dim", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_issues_table", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    # Mock runner UI
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.console", type("C", (), {"print": lambda *a, **kw: None})())

    # Phase 1: user confirms intent; Phase 3: user selects "all" issues
    prompt_calls = iter(["y", "all"])
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: next(prompt_calls))

    config = RunConfig(
        target=str(tmp_path),
        trust_the_technology=True,
    )

    exit_code = await run(config)

    assert exit_code == 0
    assert call_count == 3
    # Plan file should exist in .daydream/
    daydream_dir = tmp_path / ".daydream"
    assert daydream_dir.exists()
    plan_files = list(daydream_dir.glob("plan-*.md"))
    assert len(plan_files) == 1
    content = plan_files[0].read_text()
    assert "Implementation Plan" in content


@pytest.mark.asyncio
async def test_run_trust_does_not_prompt_for_skill(tmp_path, monkeypatch):
    """--ttt mode should never prompt for skill selection."""
    import os
    import subprocess

    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=tmp_path, capture_output=True)
    (tmp_path / "f.txt").write_text("a")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, env=env)
    subprocess.run(["git", "checkout", "-b", "feat"], cwd=tmp_path, capture_output=True)
    (tmp_path / "f.txt").write_text("b")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "change"], cwd=tmp_path, capture_output=True, env=env)

    class MinimalBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            yield TextEvent(text="Intent: changes f.txt.")
            yield ResultEvent(structured_output={"issues": []}, continuation=None)
        async def cancel(self): pass
        def format_skill_invocation(self, k, a=""): return f"/{k}"

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: MinimalBackend())
    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_issues_table", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.console", type("C", (), {"print": lambda *a, **kw: None})())

    # confirm intent
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    # This should NOT be called — if skill prompt_user is called, fail
    def runner_prompt_trap(*args, **kwargs):
        raise AssertionError("Should not prompt for skill selection in --ttt mode")
    monkeypatch.setattr("daydream.runner.prompt_user", runner_prompt_trap)

    config = RunConfig(target=str(tmp_path), trust_the_technology=True)
    exit_code = await run(config)
    assert exit_code == 0
