"""Integration tests for the full review-fix-test flow."""

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream.runner import RunConfig, run
from daydream.ui import NEON_THEME

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
        response_text = self._get_response_for_prompt()
        # First yield AssistantMessage with text content
        yield MockAssistantMessage(content=[MockTextBlock(text=response_text)])
        # Then yield ResultMessage with cost
        yield MockResultMessage(total_cost_usd=0.001)

    def _get_response_for_prompt(self) -> str:
        prompt_lower = self._prompt.lower()

        # Phase 1: Review skill invocation
        if "beagle:review-" in prompt_lower or "/beagle:review" in self._prompt:
            return "Review complete. Found 1 issue to fix."

        # Phase 2: Parse feedback (looking for JSON extraction)
        if "extract" in prompt_lower and "json" in prompt_lower:
            return """```json
[
  {"id": 1, "description": "Add type hints to function", "file": "main.py", "line": 1}
]
```"""

        # Phase 3: Fix (contains file path and line)
        if "fix this issue" in prompt_lower:
            return "Fixed the issue by adding type hints."

        # Phase 4: Test (run test suite)
        if "test suite" in prompt_lower or "run the project" in prompt_lower:
            return "All 1 tests passed. 0 failed."

        # Commit push skill
        if "commit-push" in prompt_lower:
            return "Changes committed and pushed."

        return "OK"


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
    """Test the full tool panel lifecycle: create -> start -> set_result -> finish.

    This test exercises the actual run_agent() flow by mocking the Claude SDK
    to return a Glob ToolUseBlock followed by a ToolResultBlock with file paths.
    """
    from daydream.agent import run_agent, set_quiet_mode

    # Build the message sequence that run_agent() will receive
    tool_use_id = "test-glob-lifecycle-123"
    glob_result = """/project/src/main.py
/project/src/utils/helper.py
/project/tests/test_main.py"""

    messages = [
        # 1. AssistantMessage with ToolUseBlock triggers panel creation
        MockAssistantMessage(content=[
            MockToolUseBlock(
                id=tool_use_id,
                name="Glob",
                input={"pattern": "**/*.py", "path": "/project"},
            ),
        ]),
        # 2. UserMessage with ToolResultBlock delivers the result
        MockUserMessage(content=[
            MockToolResultBlock(
                tool_use_id=tool_use_id,
                content=glob_result,
                is_error=False,
            ),
        ]),
        # 3. ResultMessage ends the session
        MockResultMessage(total_cost_usd=0.001),
    ]

    # Create mock client class with our message sequence
    mock_client_class = mock_sdk_with_tool_use(messages)
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", mock_client_class)

    # Capture console output using a custom console
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    # Enable quiet mode (the default for daydream)
    set_quiet_mode(True)

    # Run the agent - this exercises the full lifecycle
    await run_agent(Path("/tmp"), "Test prompt for Glob tool")

    # Verify the console output contains expected elements
    output_text = output.getvalue()

    # Verify Glob header is displayed
    assert "Glob" in output_text
    assert "pattern=" in output_text
    assert "**/*.py" in output_text

    # Verify file count is displayed
    assert "Found 3 files" in output_text

    # Verify filenames are displayed
    assert "main.py" in output_text
    assert "helper.py" in output_text
    assert "test_main.py" in output_text


@pytest.mark.asyncio
async def test_glob_tool_panel_singular_file_count(mock_sdk_with_tool_use, monkeypatch):
    """Test that LiveToolPanel shows singular 'file' for 1 result through full lifecycle."""
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

    set_quiet_mode(True)

    await run_agent(Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify singular "file" (not "files")
    assert "Found 1 file" in output_text
    assert "Found 1 files" not in output_text

    # Verify the filename is displayed
    assert "main.py" in output_text


@pytest.mark.asyncio
async def test_glob_tool_panel_truncates_long_results(mock_sdk_with_tool_use, monkeypatch):
    """Test that LiveToolPanel truncates long Glob results through full lifecycle."""
    from daydream.agent import run_agent, set_quiet_mode

    tool_use_id = "test-glob-truncate-789"
    # Create 25 files (more than max_lines=20 default in _build_result_content_internal)
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

    set_quiet_mode(True)

    await run_agent(Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify the total file count is displayed
    assert "Found 25 files" in output_text

    # Verify truncation indicator (25 total - 20 displayed = 5 more)
    assert "and 5 more" in output_text


@pytest.mark.asyncio
async def test_quiet_mode_output_in_panel_without_emoji(mock_sdk_with_tool_use, monkeypatch):
    """Test that quiet mode shows output in a panel without emoji prefix."""
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

    # Verify "Output" label appears without emoji (no 📤 or \U0001f4e4)
    assert "Output" in output_text
    assert "\U0001f4e4" not in output_text  # No outbox tray emoji

    # Verify content is displayed
    assert "hello" in output_text
    assert "world" in output_text

    # Verify panel border characters are present (rounded box uses these)
    # Rich's ROUNDED box uses Unicode box-drawing characters
    assert "╭" in output_text or "│" in output_text  # Panel border indicators


@pytest.mark.asyncio
async def test_quiet_mode_empty_result_shows_panel(mock_sdk_with_tool_use, monkeypatch):
    """Test that quiet mode shows a panel even for empty/whitespace-only results."""
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

    # Verify "Output" label still appears for empty results
    assert "Output" in output_text

    # Verify panel border is present
    assert "╭" in output_text or "│" in output_text


@pytest.mark.asyncio
async def test_quiet_mode_error_result_in_panel(mock_sdk_with_tool_use, monkeypatch):
    """Test that quiet mode shows error results in a panel with Error label."""
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
    test_console = Console(file=output, force_terminal=True, width=120, theme=NEON_THEME)
    monkeypatch.setattr("daydream.agent.console", test_console)

    set_quiet_mode(True)

    await run_agent(Path("/tmp"), "Test prompt")

    output_text = output.getvalue()

    # Verify "Error" label appears (not "Output") for error results
    assert "Error" in output_text

    # Verify error content is displayed
    assert "Command failed" in output_text or "exit code" in output_text

    # Verify panel border is present
    assert "╭" in output_text or "│" in output_text
