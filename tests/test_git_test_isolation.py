"""Regression tests for Git configuration isolation in the test suite."""

import subprocess


def test_git_subprocesses_disable_commit_signing() -> None:
    """Test-created repositories must never require an interactive signing key."""
    result = subprocess.run(
        ["git", "config", "--bool", "commit.gpgsign"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "false"
