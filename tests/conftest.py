"""Shared pytest fixtures for the daydream test suite."""

import subprocess
from pathlib import Path

import pytest

from daydream.exploration import (
    Convention,
    Dependency,
    ExplorationContext,
    FileInfo,
)


@pytest.fixture
def exploration_context_fixture() -> ExplorationContext:
    """Populated ExplorationContext used by Phase 03 review-integration tests.

    Provides a minimal but realistic context: one modified file, one
    convention, and one dependency edge. Wave 1 implementation work
    will consume this fixture to verify prompt-injection behavior.
    """
    return ExplorationContext(
        affected_files=[
            FileInfo(
                path="daydream/runner.py",
                role="modified",
                summary="Run orchestrator",
            ),
        ],
        conventions=[
            Convention(
                name="snake_case_modules",
                description="All modules use snake_case filenames",
                source="inferred from code",
            ),
        ],
        dependencies=[
            Dependency(
                source="daydream/runner.py",
                target="daydream/phases.py",
                relationship="imports",
            ),
        ],
    )


@pytest.fixture
def exploration_dir_fixture(tmp_path: Path, exploration_context_fixture: ExplorationContext) -> Path:
    """Write exploration context to a temp dir for phase prompt tests."""
    exploration_dir = tmp_path / "exploration"
    exploration_context_fixture.write_to_dir(exploration_dir)
    return exploration_dir


@pytest.fixture
def multi_stack_target(tmp_path: Path) -> Path:
    """Git repo with a Python + React + Markdown diff on a feature branch.

    Used by deep-mode orchestrator and integration tests (plan 05-09 / 05-10)
    to exercise the multi-stack routing path. The repo has an ``init`` commit
    on ``main`` and a ``change`` commit on a ``feature`` branch that modifies
    one file per stack (``api.py``, ``App.tsx``, ``README.md``).
    """
    project = tmp_path / "multi_stack"
    project.mkdir()
    (project / "api.py").write_text("def hello():\n    return 'world'\n")
    (project / "App.tsx").write_text("export const App = () => <div>hello</div>;\n")
    (project / "README.md").write_text("# Project\n")
    subprocess.run(  # noqa: S603
        ["git", "init", "-b", "main"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "config", "user.email", "test@test.com"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "config", "user.name", "Test"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "add", "."],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "commit", "-m", "init"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "checkout", "-b", "feature"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    (project / "api.py").write_text("def hello():\n    return 'universe'\n")
    (project / "App.tsx").write_text("export const App = () => <div>universe</div>;\n")
    (project / "README.md").write_text("# Project\n\nUpdated.\n")
    subprocess.run(  # noqa: S603
        ["git", "add", "."],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "commit", "-m", "change"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    return project


@pytest.fixture(autouse=True)
def _reset_trajectory_recorder():
    """Clear the trajectory ContextVar before AND after every test.

    Mirrors ``daydream.agent.reset_state()`` for ``AgentState`` (CORE-10 / D-17).
    Prevents cross-test bleed when a test forgets to wrap recorder usage in
    ``async with TrajectoryRecorder(...)``.

    Lazy-imports ``_reset_recorder_for_tests`` to avoid eagerly loading
    ``daydream.trajectory`` (and its Pydantic-heavy ATIF imports) at
    pytest-collect time.
    """
    from daydream.trajectory import _reset_recorder_for_tests

    _reset_recorder_for_tests()
    yield
    _reset_recorder_for_tests()
