"""Tests for exploration context data structures and prompt rendering."""
from __future__ import annotations

from pathlib import Path

from daydream.exploration import (
    Convention,
    Dependency,
    ExplorationContext,
    FileInfo,
    read_cache_key,
    safe_explore,
)
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY


async def test_populated_context_produces_markdown_and_failed_exploration_is_empty(tmp_path: Path) -> None:
    ctx = ExplorationContext(affected_files=[FileInfo("src/app.py", "modified", "Main entry point")],
        conventions=[Convention("snake_case", "All functions use snake_case", "AGENTS.md")],
        dependencies=[Dependency("app.py", "utils.py", "imports")], guidelines=["Use type annotations everywhere"],
        raw_notes="Found interesting patterns in the codebase.",
    )
    output = ctx.to_prompt_section()
    assert "# Exploration Context" in output
    assert "## Affected Files" in output
    assert "## Codebase Conventions" in output
    assert "## Dependencies" in output
    assert "## Project Guidelines" in output
    assert "src/app.py" in output
    assert "snake_case" in output
    assert "imports" in output
    assert "Use type annotations everywhere" in output
    assert "Found interesting patterns in the codebase." in output
    output = ctx.to_prompt_section()
    assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in output
    assert output.index(UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY) < output.index("## Affected Files")

    async def failing_explore() -> ExplorationContext:
        raise RuntimeError("SDK timeout")
    result = await safe_explore(failing_explore)
    assert result.completed is False
    assert result.affected_files == []
    assert result.conventions == []
    assert result.dependencies == []
    assert result.guidelines == []
    assert result.raw_notes == ""
    assert read_cache_key(tmp_path / "never-explored") is None, "an unreadable cache key is a miss"
