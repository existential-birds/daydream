"""Shared real-git subprocess helpers for the test suite.

Consolidates the ``_git`` helper (and its repo-building family) that was
previously duplicated across tests/conftest.py and several test modules.
tests/test_workspace.py intentionally keeps its own copies (it needs extra
plumbing for bare-origin push semantics).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def git(repo: Path, *args: str, check: bool = True) -> str:
    """Run a git command in *repo* and return stripped stdout (test helper)."""
    proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", *args],  # noqa: S607 - git is a trusted command
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    )
    return proc.stdout.strip()


def configure_identity(repo: Path) -> None:
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Tester")


def commit(repo: Path, message: str) -> str:
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-b", "main")
    configure_identity(repo)


def bare_remote(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "--bare", "-b", "main")
    return path
