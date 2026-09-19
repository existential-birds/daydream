"""Shared real-git subprocess helpers for the test suite.

Consolidates the ``_git`` helper (and its repo-building family) that was
previously duplicated across tests/conftest.py and several test modules.
tests/test_workspace.py builds on these too, keeping only the extra plumbing
its bare-origin push semantics genuinely need.

The seed family (identity + write + commit) is the single source of truth for
deterministic test seed repositories. Deliberate exceptions are migrate's
dateless identity, ``harbor_build``'s bundle identity, and the identities in
production ``daydream/benchmark/snapshot.py`` and RL ``fixture.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

SEED_ENV: dict[str, str] = {
    "GIT_AUTHOR_NAME": "Tester",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Tester",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


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


def commit(repo: Path, message: str, *, env: dict[str, str] | None = None) -> str:
    git(repo, "commit", "-m", message, env=env)
    return git(repo, "rev-parse", "HEAD")


def seed_write(repo: Path, name: str, content: str | bytes) -> None:
    """Write ``content`` to ``repo/name`` (creating parents) and stage it.

    Half of the deterministic seed family — see the module docstring.
    """
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    git(repo, "add", name)


def seed_commit(repo: Path, message: str) -> str:
    """Commit the staged tree under :data:`SEED_ENV`; return the HEAD SHA."""
    return commit(repo, message, env=SEED_ENV)


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
