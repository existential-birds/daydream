"""Integration tests for the full review-fix-test flow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daydream.runner import RunConfig, run


@dataclass
class MockTextBlock:
    """Mock TextBlock from claude_agent_sdk.types."""

    text: str


@dataclass
class MockAssistantMessage:
    """Mock AssistantMessage from claude_agent_sdk.types."""

    content: list[Any]


@dataclass
class MockResultMessage:
    """Mock ResultMessage from claude_agent_sdk.types."""

    total_cost_usd: float | None = 0.001


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


@pytest.fixture
def mock_sdk_client(monkeypatch):
    """Patch ClaudeSDKClient and message types with our mocks."""
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", MockClaudeSDKClient)
    monkeypatch.setattr("daydream.agent.AssistantMessage", MockAssistantMessage)
    monkeypatch.setattr("daydream.agent.ResultMessage", MockResultMessage)
    monkeypatch.setattr("daydream.agent.TextBlock", MockTextBlock)
    return MockClaudeSDKClient


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
