"""Shared pytest fixtures for the daydream test suite."""

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
