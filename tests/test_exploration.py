"""Tests for exploration context data structures and prompt rendering."""
from __future__ import annotations

from daydream.exploration import Convention, Dependency, ExplorationContext, FileInfo, merge_contexts, safe_explore
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY


def test_populated_context_produces_markdown() -> None:
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

async def test_safe_explore_returns_empty_on_failure() -> None:
    async def failing_explore() -> ExplorationContext:
        raise RuntimeError("SDK timeout")
    result = await safe_explore(failing_explore)
    assert result.completed is False
    assert result.affected_files == []
    assert result.conventions == []
    assert result.dependencies == []
    assert result.guidelines == []
    assert result.raw_notes == ""

def test_merge_contexts_single_returns_fresh_lists() -> None:
    original = ExplorationContext(guidelines=["a", "b"])
    merged = merge_contexts(original)
    assert merged.guidelines == ["a", "b"]
    assert merged.guidelines is not original.guidelines


def test_merge_contexts_restores_source_file_on_static_tie() -> None:
    """A winning static row must not net out an empty source_file when a duplicate (the deterministic row carries
    none, but the LLM test-mapper duplicate does) has one recorded, or the test-map filter drops the mapping."""
    static = ExplorationContext(affected_files=[FileInfo("tests/test_a.py", "test", "static note", provenance="static")]
    )
    llm = ExplorationContext(affected_files=[FileInfo(
                "tests/test_a.py", "test", "a much longer LLM test summary", provenance="llm",
                source_file="daydream/a.py",
            )
        ]
    )
    merged = merge_contexts(static, llm)
    assert len(merged.affected_files) == 1
    row = merged.affected_files[0]
    assert row.provenance == "static"
    assert row.summary == "a much longer LLM test summary"
    assert row.source_file == "daydream/a.py"
