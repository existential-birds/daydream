"""Integration tests for the full review-fix-test flow."""

import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream.runner import RunConfig, run
from daydream.ui import NEON_THEME

# ANSI escape code pattern for stripping terminal colors
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text for assertion comparisons."""
    return _ANSI_ESCAPE.sub("", text)

# =============================================================================
# Mock SDK Types
# =============================================================================


@dataclass
class MockTextBlock:
    """Mock TextBlock from claude_agent_sdk.types."""

    text: str


@dataclass
class MockToolUseBlock:
    """Mock ToolUseBlock from claude_agent_sdk.types."""

    id: str
    name: str
    input: dict[str, Any] | None = None


@dataclass
class MockToolResultBlock:
    """Mock ToolResultBlock from claude_agent_sdk.types."""

    tool_use_id: str
    content: str | None = None
    is_error: bool = False


@dataclass
class MockThinkingBlock:
    """Mock ThinkingBlock from claude_agent_sdk.types."""

    thinking: str


@dataclass
class MockAssistantMessage:
    """Mock AssistantMessage from claude_agent_sdk.types."""

    content: list[Any] = field(default_factory=list)


@dataclass
class MockUserMessage:
    """Mock UserMessage from claude_agent_sdk.types."""

    content: list[Any] = field(default_factory=list)


@dataclass
class MockResultMessage:
    """Mock ResultMessage from claude_agent_sdk.types."""

    total_cost_usd: float | None = 0.001
    structured_output: Any = None


# =============================================================================
# Mock SDK Clients
# =============================================================================


class MockClaudeSDKClient:
    """Mock ClaudeSDKClient that returns canned responses based on prompt."""

    def __init__(self, options: Any = None):
        self.options = options
        self._prompt: str = ""
        self._call_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt
        self._call_count += 1

    async def receive_response(self):
        """Yield mock responses based on prompt content."""
        response_text, structured_output = self._get_response_for_prompt()
        # First yield AssistantMessage with text content
        yield MockAssistantMessage(content=[MockTextBlock(text=response_text)])
        # Then yield ResultMessage with cost and optional structured output
        yield MockResultMessage(total_cost_usd=0.001, structured_output=structured_output)

    def _get_response_for_prompt(self) -> tuple[str, Any]:
        """Return (text_response, structured_output) for the current prompt."""
        prompt_lower = self._prompt.lower()

        # Phase 1: Review skill invocation
        if "beagle-" in prompt_lower and "review" in prompt_lower:
            return "Review complete. Found 1 issue to fix.", None

        # Phase 2: Parse feedback (looking for JSON extraction)
        # Now returns structured output directly instead of text
        if "extract" in prompt_lower and "json" in prompt_lower:
            structured = {
                "issues": [
                    {"id": 1, "description": "Add type hints to function", "file": "main.py", "line": 1}
                ]
            }
            return "Extracted feedback.", structured

        # Phase 3: Fix (contains file path and line)
        if "fix this issue" in prompt_lower:
            return "Fixed the issue by adding type hints.", None

        # Phase 4: Test (run test suite)
        if "test suite" in prompt_lower or "run the project" in prompt_lower:
            return "All 1 tests passed. 0 failed.", None

        # Commit push skill
        if "commit-push" in prompt_lower:
            return "Changes committed and pushed.", None

        return "OK", None


class MockClaudeSDKClientWithToolUse:
    """Mock ClaudeSDKClient that yields tool use/result sequences.

    This mock is used for integration tests that need to exercise the
    full tool panel lifecycle: create() -> start() -> set_result() -> finish().
    """

    def __init__(self, options: Any = None, messages: list[Any] | None = None):
        """Initialize with optional pre-configured message sequence.

        Args:
            options: SDK options (ignored).
            messages: List of messages to yield from receive_response().

        """
        self.options = options
        self._messages = messages or []
        self._prompt: str = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        """Yield the pre-configured message sequence."""
        for msg in self._messages:
            yield msg


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_sdk_client(monkeypatch):
    """Patch ClaudeSDKClient and message types with our mocks."""
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", MockClaudeSDKClient)
    monkeypatch.setattr("daydream.agent.AssistantMessage", MockAssistantMessage)
    monkeypatch.setattr("daydream.agent.ResultMessage", MockResultMessage)
    monkeypatch.setattr("daydream.agent.TextBlock", MockTextBlock)
    return MockClaudeSDKClient


@pytest.fixture
def mock_sdk_with_tool_use(monkeypatch):
    """Patch SDK types to allow tool use/result testing.

    Returns a factory function to create clients with custom message sequences.
    """
    # Patch all the type checks used by run_agent()
    monkeypatch.setattr("daydream.agent.AssistantMessage", MockAssistantMessage)
    monkeypatch.setattr("daydream.agent.UserMessage", MockUserMessage)
    monkeypatch.setattr("daydream.agent.ResultMessage", MockResultMessage)
    monkeypatch.setattr("daydream.agent.TextBlock", MockTextBlock)
    monkeypatch.setattr("daydream.agent.ToolUseBlock", MockToolUseBlock)
    monkeypatch.setattr("daydream.agent.ToolResultBlock", MockToolResultBlock)
    monkeypatch.setattr("daydream.agent.ThinkingBlock", MockThinkingBlock)
    # Speed up print_thinking animation
    monkeypatch.setattr("daydream.ui.time.sleep", lambda _: None)

    def factory(messages: list[Any]) -> type:
        """Create a mock client class with the given message sequence."""

        class ConfiguredClient(MockClaudeSDKClientWithToolUse):
            def __init__(self, options: Any = None):
                super().__init__(options=options, messages=messages)

        return ConfiguredClient

    return factory


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
async def test_full_fix_flow(mock_sdk_client, mock_ui, target_project: Path):
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
async def test_glob_tool_panel_displays_file_count_and_list(mock_sdk_with_tool_use, monkeypatch):
    """Test the full tool panel lifecycle in normal mode shows file count and list.

    This test exercises the actual run_agent() flow by mocking the Claude SDK
    to return a Glob ToolUseBlock followed by a ToolResultBlock with file paths.
    Normal mode (quiet=False) shows both header and output section.

    Also tests that:
    - AgentTextRenderer displays streamed text with spinner cursor effect
    - LiveThinkingPanel displays thinking blocks with animated title
    """
    from daydream.agent import run_agent, set_quiet_mode

    # Build the message sequence that run_agent() will receive
    tool_use_id = "test-glob-lifecycle-123"
    glob_result = """/project/src/main.py
/project/src/utils/helper.py
/project/tests/test_main.py"""

    messages = [
        # 1. AssistantMessage with ThinkingBlock tests LiveThinkingPanel spinner
        MockAssistantMessage(content=[
            MockThinkingBlock(thinking="Analyzing the project structure..."),
        ]),
        # 2. AssistantMessage with TextBlock tests AgentTextRenderer spinner cursor
        MockAssistantMessage(content=[
            MockTextBlock(text="I'll search for Python files in the project."),
        ]),
        # 3. AssistantMessage with ToolUseBlock triggers panel creation
        MockAssistantMessage(content=[
            MockToolUseBlock(
                id=tool_use_id,
                name="Glob",
                input={"pattern": "**/*.py", "path": "/project"},
            ),
        ]),
        # 4. UserMessage with ToolResultBlock delivers the result
        MockUserMessage(content=[
            MockToolResultBlock(
                tool_use_id=tool_use_id,
                content=glob_result,
                is_error=False,
            ),
        ]),
        # 5. ResultMessage ends the session
        MockResultMessage(total_cost_usd=0.001),
    ]

    # Create mock client class with our message sequence
    mock_client_class = mock_sdk_with_tool_use(messages)
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", mock_client_class)

    # Capture console output using a custom console
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    # Normal mode (not quiet) to see full output with file counts
    set_quiet_mode(False)

    # Run the agent - this exercises the full lifecycle
    await run_agent(Path("/tmp"), "Test prompt for Glob tool")

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
async def test_glob_tool_panel_singular_file_count(mock_sdk_with_tool_use, monkeypatch):
    """Test that LiveToolPanel shows singular 'file' for 1 result in normal mode."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-glob-singular-456"
    glob_result = "/project/main.py"

    messages = [
        MockAssistantMessage(content=[
            MockToolUseBlock(
                id=tool_use_id,
                name="Glob",
                input={"pattern": "*.py"},
            ),
        ]),
        MockUserMessage(content=[
            MockToolResultBlock(
                tool_use_id=tool_use_id,
                content=glob_result,
                is_error=False,
            ),
        ]),
        MockResultMessage(total_cost_usd=0.001),
    ]

    mock_client_class = mock_sdk_with_tool_use(messages)
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", mock_client_class)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    # Normal mode to see output section
    set_quiet_mode(False)

    await run_agent(Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify singular "file" (not "files")
    assert "Found 1 file" in output_text
    assert "Found 1 files" not in output_text

    # Verify the filename is displayed
    assert "main.py" in output_text


@pytest.mark.asyncio
async def test_glob_tool_panel_truncates_long_results(mock_sdk_with_tool_use, monkeypatch):
    """Test that LiveToolPanel truncates long Glob results in normal mode."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-glob-truncate-789"
    # Create 25 files (more than max_lines=20 passed from _build_result_content_internal)
    mock_files = [f"/project/src/module{i}.py" for i in range(25)]
    glob_result = "\n".join(mock_files)

    messages = [
        MockAssistantMessage(content=[
            MockToolUseBlock(
                id=tool_use_id,
                name="Glob",
                input={"pattern": "**/*.py"},
            ),
        ]),
        MockUserMessage(content=[
            MockToolResultBlock(
                tool_use_id=tool_use_id,
                content=glob_result,
                is_error=False,
            ),
        ]),
        MockResultMessage(total_cost_usd=0.001),
    ]

    mock_client_class = mock_sdk_with_tool_use(messages)
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", mock_client_class)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    # Normal mode to see output section
    set_quiet_mode(False)

    await run_agent(Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify the total file count is displayed
    assert "Found 25 files" in output_text

    # Verify truncation indicator (25 total - 20 displayed = 5 more)
    assert "and 5 more" in output_text


@pytest.mark.asyncio
async def test_quiet_mode_shows_header_only(mock_sdk_with_tool_use, monkeypatch):
    """Test that quiet mode shows header only (no output section)."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-output-panel-001"
    read_result = "def hello():\n    return 'world'"

    messages = [
        MockAssistantMessage(content=[
            MockToolUseBlock(
                id=tool_use_id,
                name="Read",
                input={"file_path": "/project/main.py"},
            ),
        ]),
        MockUserMessage(content=[
            MockToolResultBlock(
                tool_use_id=tool_use_id,
                content=read_result,
                is_error=False,
            ),
        ]),
        MockResultMessage(total_cost_usd=0.001),
    ]

    mock_client_class = mock_sdk_with_tool_use(messages)
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", mock_client_class)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(True)

    await run_agent(Path("/tmp"), "Test prompt")

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
async def test_quiet_mode_empty_result_shows_header_only(mock_sdk_with_tool_use, monkeypatch):
    """Test that quiet mode shows header only for empty results (no output section)."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-empty-result-002"

    messages = [
        MockAssistantMessage(content=[
            MockToolUseBlock(
                id=tool_use_id,
                name="Bash",
                input={"command": "true"},  # Command that produces no output
            ),
        ]),
        MockUserMessage(content=[
            MockToolResultBlock(
                tool_use_id=tool_use_id,
                content="",  # Empty result
                is_error=False,
            ),
        ]),
        MockResultMessage(total_cost_usd=0.001),
    ]

    mock_client_class = mock_sdk_with_tool_use(messages)
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", mock_client_class)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(True)

    await run_agent(Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify header is displayed
    assert "Bash" in output_text

    # Verify NO output section in quiet mode (header only)
    assert "Output" not in output_text

    # Verify panel border is present
    assert "╭" in output_text or "│" in output_text


@pytest.mark.asyncio
async def test_quiet_mode_error_shows_header_with_red_border(mock_sdk_with_tool_use, monkeypatch):
    """Test that quiet mode shows header only with red border for errors."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-error-result-003"

    messages = [
        MockAssistantMessage(content=[
            MockToolUseBlock(
                id=tool_use_id,
                name="Bash",
                input={"command": "false"},
            ),
        ]),
        MockUserMessage(content=[
            MockToolResultBlock(
                tool_use_id=tool_use_id,
                content="Command failed with exit code 1",
                is_error=True,
            ),
        ]),
        MockResultMessage(total_cost_usd=0.001),
    ]

    mock_client_class = mock_sdk_with_tool_use(messages)
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", mock_client_class)

    output = StringIO()
    # Force truecolor to get consistent RGB color codes across environments
    test_console = Console(
        file=output, force_terminal=True, width=120, theme=NEON_THEME, color_system="truecolor"
    )
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(True)

    await run_agent(Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify header is displayed
    assert "Bash" in output_text

    # Verify NO error content in quiet mode (header only)
    # The red border color indicates error status
    assert "Command failed" not in output_text

    # Verify panel border is present (red color is applied via ANSI codes)
    assert "╭" in output_text or "│" in output_text

    # Verify red color is in the output (ANSI escape for red: 255;85;85)
    assert "255;85;85" in output_text


@pytest.mark.asyncio
async def test_skill_tool_panel_collapses_output(mock_sdk_with_tool_use, monkeypatch):
    """Test that Skill tool calls don't show an Output panel.

    The skill name already appears in the tool call header, so the
    "Launching skill: X" output is redundant and should be suppressed.
    """
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-skill-collapse-001"

    messages = [
        MockAssistantMessage(content=[
            MockToolUseBlock(
                id=tool_use_id,
                name="Skill",
                input={"skill": "review-python"},
            ),
        ]),
        MockUserMessage(content=[
            MockToolResultBlock(
                tool_use_id=tool_use_id,
                content="Launching skill: review-python",
                is_error=False,
            ),
        ]),
        MockResultMessage(total_cost_usd=0.001),
    ]

    mock_client_class = mock_sdk_with_tool_use(messages)
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", mock_client_class)

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(True)

    await run_agent(Path("/tmp"), "Test prompt")

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
