# tests/test_exploration.py
"""Tests for exploration context data structures and prompt rendering."""

from unittest.mock import AsyncMock, patch

from daydream.exploration import Convention, Dependency, ExplorationContext, FileInfo, safe_explore


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


def test_affected_files_section_header():
    ctx = ExplorationContext(affected_files=[FileInfo("a.py", "modified")])
    output = ctx.to_prompt_section()
    assert "## Affected Files" in output


def test_conventions_section_header():
    ctx = ExplorationContext(conventions=[Convention("test", "desc")])
    output = ctx.to_prompt_section()
    assert "## Codebase Conventions" in output


def test_dependencies_section_header():
    ctx = ExplorationContext(dependencies=[Dependency("a.py", "b.py", "imports")])
    output = ctx.to_prompt_section()
    assert "## Dependencies" in output


def test_guidelines_section_header():
    ctx = ExplorationContext(guidelines=["Always lint"])
    output = ctx.to_prompt_section()
    assert "## Project Guidelines" in output


def test_raw_notes_included_when_nonempty():
    ctx = ExplorationContext(raw_notes="Some notes here")
    output = ctx.to_prompt_section()
    assert "Some notes here" in output


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
    assert result.affected_files == expected.affected_files


async def test_safe_explore_returns_empty_on_failure():
    async def failing_explore() -> ExplorationContext:
        raise RuntimeError("SDK timeout")

    result = await safe_explore(failing_explore)
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
