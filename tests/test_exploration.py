"""Tests for exploration context data structures and prompt rendering."""
from __future__ import annotations

from pathlib import Path

from daydream.exploration import Convention, Dependency, ExplorationContext, FileInfo, merge_contexts, safe_explore
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY


def test_populated_context_produces_markdown() -> None:
    ctx = ExplorationContext(affected_files=[FileInfo("src/app.py", "modified", "Main entry point")],
        conventions=[Convention("snake_case", "All functions use snake_case", "CLAUDE.md")],
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

def test_merge_contexts_prefers_static_provenance_on_tie() -> None:
    static = ExplorationContext(affected_files=[FileInfo("a.py", "modified", "", provenance="static")])
    llm = ExplorationContext(
        affected_files=[FileInfo("a.py", "modified", "a much longer LLM summary", provenance="llm")]
    )
    merged = merge_contexts(static, llm)
    assert len(merged.affected_files) == 1
    assert merged.affected_files[0].summary == "a much longer LLM summary"
    assert merged.affected_files[0].provenance == "static"

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

def test_write_to_dir_creates_all_files(tmp_path: Path) -> None:
    ctx = ExplorationContext(affected_files=[FileInfo("src/app.py", "modified", "Main entry point")],
        conventions=[Convention("snake_case", "All functions use snake_case", "CLAUDE.md")],
        dependencies=[Dependency("app.py", "utils.py", "imports")], guidelines=["Use type annotations everywhere"],
        raw_notes="Found interesting patterns.",
    )
    exploration_dir = tmp_path / "exploration"
    ctx.write_to_dir(exploration_dir)

    assert (exploration_dir / "summary.md").exists()
    assert (exploration_dir / "affected_files.md").exists()
    assert (exploration_dir / "conventions.md").exists()
    assert (exploration_dir / "dependencies.md").exists()

    affected = (exploration_dir / "affected_files.md").read_text()
    assert "src/app.py" in affected
    assert "modified" in affected

    conventions = (exploration_dir / "conventions.md").read_text()
    assert "snake_case" in conventions
    assert "Use type annotations everywhere" in conventions

    deps = (exploration_dir / "dependencies.md").read_text()
    assert "app.py" in deps
    assert "utils.py" in deps

    summary = (exploration_dir / "summary.md").read_text()
    assert "affected_files.md" in summary
    assert "Additional Notes" in summary
    assert "Found interesting patterns." in summary

    for name in ("summary.md", "affected_files.md", "conventions.md", "dependencies.md"):
        content = (exploration_dir / name).read_text()
        assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in content
    # Boundary sits directly below the top-level heading, before the table/data.
    affected = (exploration_dir / "affected_files.md").read_text()
    assert affected.index(UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY) < affected.index("src/app.py")
