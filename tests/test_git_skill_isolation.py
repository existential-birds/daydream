"""Git-backed regressions for project-local skill isolation."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_tracked_project_local_skills_absent_and_ignored(repo_root: Path) -> None:
    """M20: the two deleted skill paths are untracked; future skill paths are ignored."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for path in (
        ".claude/skills/rerun-review-bot-bench/SKILL.md",
        ".claude/skills/resume-run/SKILL.md",
        ".claude/skills/some-future/SKILL.md",
        ".agents/skills/some-future/SKILL.md",
    ):
        assert path not in tracked, f"{path} is still tracked (M18/M19)"

    for path in (
        ".claude/skills/some-future/SKILL.md",
        ".agents/skills/some-future/SKILL.md",
    ):
        status = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=repo_root,
            capture_output=True,
        )
        assert status.returncode == 0, f"{path} is not ignored (M19)"
