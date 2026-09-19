"""Shared real-git subprocess helpers for the test suite.

Consolidates the ``_git`` helper (and its repo-building family) that was
previously duplicated across tests/conftest.py and several test modules.
tests/test_workspace.py builds on these too, keeping only the extra plumbing
its bare-origin push semantics genuinely need.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def git(repo: Path, *args: str, check: bool = True, env: dict[str, str] | None = None) -> str:
    """Run a git command in *repo* and return stripped stdout (test helper)."""
    proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", *args],  # noqa: S607 - git is a trusted command
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, **env} if env else None,
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


def tracked_source_state(repo: Path) -> dict[str, Any]:
    """Everything a review run must never mutate in the source checkout."""
    tracked = git(repo, "ls-files").splitlines()
    return {
        "head": git(repo, "rev-parse", "HEAD"),
        "refs": git(repo, "show-ref"),
        "index": git(repo, "ls-files", "--stage"),
        "diff": git(repo, "diff", "--binary", "HEAD"),
        "bytes": {name: (repo / name).read_bytes() for name in tracked},
    }
