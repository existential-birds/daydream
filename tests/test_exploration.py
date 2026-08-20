# tests/test_exploration.py
"""Tests for exploration context data structures and prompt rendering."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from daydream.exploration import Convention, Dependency, ExplorationContext, FileInfo, merge_contexts, safe_explore
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY


def test_file_info_creates_valid_instance():
    info = FileInfo("src/app.py", "modified", "Main entry point")
    assert info.path == "src/app.py"
    assert info.role == "modified"
    assert info.summary == "Main entry point"


def test_convention_creates_valid_instance():
    conv = Convention("snake_case", "All functions use snake_case", "CLAUDE.md")
    assert conv.name == "snake_case"
    assert conv.description == "All functions use snake_case"
    assert conv.source == "CLAUDE.md"


def test_dependency_creates_valid_instance():
    dep = Dependency("app.py", "utils.py", "imports")
    assert dep.source == "app.py"
    assert dep.target == "utils.py"
    assert dep.relationship == "imports"


def test_empty_exploration_context():
    ctx = ExplorationContext()
    assert ctx.affected_files == []
    assert ctx.conventions == []
    assert ctx.dependencies == []
    assert ctx.guidelines == []
    assert ctx.raw_notes == ""


def test_empty_context_produces_empty_string():
    ctx = ExplorationContext()
    assert ctx.to_prompt_section() == ""


def test_populated_context_produces_markdown():
    ctx = ExplorationContext(
        affected_files=[FileInfo("src/app.py", "modified", "Main entry point")],
        conventions=[Convention("snake_case", "All functions use snake_case", "CLAUDE.md")],
        dependencies=[Dependency("app.py", "utils.py", "imports")],
        guidelines=["Use type annotations everywhere"],
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


@pytest.mark.parametrize(
    ("context", "expected_content"),
    [
        pytest.param(
            ExplorationContext(affected_files=[FileInfo("a.py", "modified")]),
            "## Affected Files",
            id="affected-files",
        ),
        pytest.param(
            ExplorationContext(conventions=[Convention("test", "desc")]),
            "## Codebase Conventions",
            id="codebase-conventions",
        ),
        pytest.param(
            ExplorationContext(dependencies=[Dependency("a.py", "b.py", "imports")]),
            "## Dependencies",
            id="dependencies",
        ),
        pytest.param(
            ExplorationContext(guidelines=["Always lint"]),
            "## Project Guidelines",
            id="project-guidelines",
        ),
        pytest.param(
            ExplorationContext(raw_notes="Some notes here"),
            "Some notes here",
            id="raw-notes",
        ),
    ],
)
def test_to_prompt_section_includes_populated_content(context, expected_content):
    output = context.to_prompt_section()
    assert expected_content in output


def test_partial_context_only_includes_populated_sections():
    ctx = ExplorationContext(
        affected_files=[FileInfo("a.py", "modified", "Entry point")],
    )
    output = ctx.to_prompt_section()
    assert "## Affected Files" in output
    assert "## Codebase Conventions" not in output
    assert "## Dependencies" not in output
    assert "## Project Guidelines" not in output


async def test_safe_explore_returns_result_on_success():
    expected = ExplorationContext(
        affected_files=[FileInfo("main.py", "modified", "App entry")],
        guidelines=["Use type hints"],
    )

    async def fake_explore() -> ExplorationContext:
        return expected

    result = await safe_explore(fake_explore)
    assert result is expected
    assert result.completed is True
    assert result.affected_files == expected.affected_files


async def test_safe_explore_returns_empty_on_failure():
    async def failing_explore() -> ExplorationContext:
        raise RuntimeError("SDK timeout")

    result = await safe_explore(failing_explore)
    assert result.completed is False
    assert result.affected_files == []
    assert result.conventions == []
    assert result.dependencies == []
    assert result.guidelines == []
    assert result.raw_notes == ""


@patch("daydream.ui.print_warning")
@patch("daydream.ui.create_console")
async def test_safe_explore_shows_warning_on_failure(mock_create_console, mock_print_warning):
    mock_console = object()
    mock_create_console.return_value = mock_console

    async def failing_explore() -> ExplorationContext:
        raise RuntimeError("SDK timeout")

    await safe_explore(failing_explore)
    mock_print_warning.assert_called_once_with(mock_console, "Exploration failed -- proceeding with review only")


def test_merge_pattern_scanner_result():
    partial = ExplorationContext(
        conventions=[Convention(name="snake_case", description="use snake_case for functions", source="inferred")]
    )
    merged = merge_contexts(ExplorationContext(), partial)
    assert len(merged.conventions) == 1
    assert merged.conventions[0].name == "snake_case"


def test_merge_contexts_empty():
    merged = merge_contexts()
    assert merged.affected_files == []
    assert merged.conventions == []
    assert merged.dependencies == []
    assert merged.guidelines == []
    assert merged.raw_notes == ""


def test_merge_contexts_single_returns_fresh_lists():
    original = ExplorationContext(guidelines=["a", "b"])
    merged = merge_contexts(original)
    assert merged.guidelines == ["a", "b"]
    assert merged.guidelines is not original.guidelines


def test_merge_contexts_dedups_file_info():
    a = ExplorationContext(affected_files=[FileInfo("a.py", "modified", "short")])
    b = ExplorationContext(affected_files=[FileInfo("a.py", "modified", "this is a much longer summary")])
    merged = merge_contexts(a, b)
    assert len(merged.affected_files) == 1
    assert merged.affected_files[0].summary == "this is a much longer summary"


def test_merge_contexts_prefers_static_provenance_on_tie():
    static = ExplorationContext(
        affected_files=[FileInfo("a.py", "modified", "", provenance="static")]
    )
    llm = ExplorationContext(
        affected_files=[FileInfo("a.py", "modified", "a much longer LLM summary", provenance="llm")]
    )
    merged = merge_contexts(static, llm)
    assert len(merged.affected_files) == 1
    assert merged.affected_files[0].summary == "a much longer LLM summary"
    assert merged.affected_files[0].provenance == "static"


def test_merge_contexts_restores_source_file_on_static_tie():
    """A winning static row must not net out an empty source_file when a
    duplicate (the deterministic row carries none, but the LLM test-mapper
    duplicate does) has one recorded, or the test-map filter drops the mapping."""
    static = ExplorationContext(
        affected_files=[FileInfo("tests/test_a.py", "test", "static note", provenance="static")]
    )
    llm = ExplorationContext(
        affected_files=[
            FileInfo(
                "tests/test_a.py",
                "test",
                "a much longer LLM test summary",
                provenance="llm",
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


def test_merge_contexts_dedups_dependencies():
    dep = Dependency("a.py", "b.py", "imports")
    a = ExplorationContext(dependencies=[dep])
    b = ExplorationContext(dependencies=[Dependency("a.py", "b.py", "imports")])
    merged = merge_contexts(a, b)
    assert len(merged.dependencies) == 1


def test_merge_contexts_dedups_conventions_and_guidelines():
    a = ExplorationContext(
        conventions=[Convention("snake", "desc1", "CLAUDE.md")],
        guidelines=["use type hints"],
    )
    b = ExplorationContext(
        conventions=[Convention("snake", "desc2", "inferred")],
        guidelines=["use type hints", "no print statements"],
    )
    merged = merge_contexts(a, b)
    assert len(merged.conventions) == 1
    assert len(merged.guidelines) == 2


def test_merge_contexts_joins_raw_notes():
    a = ExplorationContext(raw_notes="first")
    b = ExplorationContext(raw_notes="")
    c = ExplorationContext(raw_notes="second")
    merged = merge_contexts(a, b, c)
    assert merged.raw_notes == "first\n\nsecond"


def test_write_to_dir_creates_all_files(tmp_path):
    ctx = ExplorationContext(
        affected_files=[FileInfo("src/app.py", "modified", "Main entry point")],
        conventions=[Convention("snake_case", "All functions use snake_case", "CLAUDE.md")],
        dependencies=[Dependency("app.py", "utils.py", "imports")],
        guidelines=["Use type annotations everywhere"],
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


def test_write_to_dir_emits_exploration_json_with_provenance(tmp_path):
    ctx = ExplorationContext(
        affected_files=[
            FileInfo("src/app.py", "modified", "Main entry point", provenance="static"),
            FileInfo("tests/test_app.py", "test", "covers app", provenance="llm"),
        ],
        conventions=[Convention("snake_case", "all funcs snake_case", "CLAUDE.md")],
        dependencies=[Dependency("app.py", "utils.py", "imports")],
    )
    exploration_dir = tmp_path / "exploration"
    ctx.write_to_dir(exploration_dir)
    data = json.loads((exploration_dir / "exploration.json").read_text())
    rows = {row["path"]: row for row in data["affected_files"]}
    assert rows["src/app.py"]["provenance"] == "static"
    assert rows["tests/test_app.py"]["provenance"] == "llm"
    assert data["conventions"][0]["name"] == "snake_case"
    assert data["dependencies"][0]["target"] == "utils.py"


def test_write_to_dir_empty_context(tmp_path):
    ctx = ExplorationContext()
    exploration_dir = tmp_path / "exploration"
    ctx.write_to_dir(exploration_dir)

    for name in ("summary.md", "affected_files.md", "conventions.md", "dependencies.md"):
        path = exploration_dir / name
        assert path.exists()

    assert "No data collected" in (exploration_dir / "affected_files.md").read_text()
    assert "No data collected" in (exploration_dir / "conventions.md").read_text()
    assert "No data collected" in (exploration_dir / "dependencies.md").read_text()

    for name in ("summary.md", "affected_files.md", "conventions.md", "dependencies.md"):
        assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in (exploration_dir / name).read_text()


def test_write_to_dir_creates_directory(tmp_path):
    ctx = ExplorationContext()
    nested = tmp_path / "a" / "b" / "exploration"
    ctx.write_to_dir(nested)
    assert nested.is_dir()
    assert (nested / "summary.md").exists()


def test_write_to_dir_returns_path(tmp_path):
    ctx = ExplorationContext()
    exploration_dir = tmp_path / "exploration"
    result = ctx.write_to_dir(exploration_dir)
    assert result == exploration_dir


def test_write_to_dir_partial_context(tmp_path):
    ctx = ExplorationContext(
        affected_files=[FileInfo("a.py", "modified", "Entry point")],
    )
    exploration_dir = tmp_path / "exploration"
    ctx.write_to_dir(exploration_dir)

    affected = (exploration_dir / "affected_files.md").read_text()
    assert "a.py" in affected

    assert "No data collected" in (exploration_dir / "conventions.md").read_text()
    assert "No data collected" in (exploration_dir / "dependencies.md").read_text()
