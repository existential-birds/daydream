"""Shared real-git subprocess helpers for the test suite.

Consolidates the ``_git`` helper (and its repo-building family) that was
previously duplicated across tests/conftest.py and several test modules.
tests/test_workspace.py builds on these too, keeping only the extra plumbing
its bare-origin push semantics genuinely need.

The seed family (``SEED_ENV`` + :func:`write_and_stage` +
:func:`seeded_commit`) is the single source of truth for deterministic
test seed repositories. Deliberate exceptions are migrate's dateless
identity, ``harbor_build``'s bundle identity, and the identities in
production ``daydream/benchmark/snapshot.py`` and RL ``fixture.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from daydream.workspace import WorkContext

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


def work_context(repo: Path, *, run_id: str) -> WorkContext:
    """A real-HEAD, non-ephemeral ``WorkContext`` for a source checkout."""
    head = git(repo, "rev-parse", "HEAD")
    return WorkContext(
        repo=repo,
        source=repo,
        base_branch="main",
        base_sha=head,
        head_branch="main",
        head_sha=head,
        is_ephemeral=False,
        run_id=run_id,
    )


def write_and_stage(repo: Path, name: str, content: str | bytes) -> None:
    """Write *name* under *repo* and stage it (``str`` or ``bytes``).

    Half of the deterministic seed family — see the module docstring.
    """
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    git(repo, "add", name)


def seeded_commit(repo: Path, message: str) -> str:
    """Commit with the deterministic ``SEED_ENV`` identity and return the new SHA."""
    git(repo, "commit", "-m", message, env=SEED_ENV)
    return git(repo, "rev-parse", "HEAD")


def seed_pr_origin(
    tmp_path: Path,
    *,
    repo_name: str = "local_wt",
    bare_name: str = "origin_local.git",
    feature_body: str = "FEATURE = 1\n",
    feature_message: str = "feature",
    number: int = 101,
) -> tuple[str, str, str]:
    """Build a real local bare origin whose base/head are the PR's SHAs."""
    import shutil as _sh

    repo = tmp_path / repo_name
    if repo.exists():
        _sh.rmtree(repo)
    repo.mkdir()
    git(repo, "init", "-b", "main")
    write_and_stage(repo, "readme.txt", "base1\n")
    seeded_commit(repo, "base1")
    write_and_stage(repo, "base.py", "BASE = 2\n")
    base_sha = seeded_commit(repo, "base2")
    write_and_stage(repo, "beyond.py", "BEYOND = 3\n")
    seeded_commit(repo, "base3")
    git(repo, "checkout", "--detach", base_sha)
    (repo / "base.py").write_text("BASE = 20\n")
    git(repo, "add", "base.py")
    write_and_stage(repo, "feature.py", feature_body)
    head_sha = seeded_commit(repo, feature_message)
    bare = tmp_path / bare_name
    if bare.exists():
        _sh.rmtree(bare)
    bare.mkdir()
    git(bare, "init", "--bare")
    git(repo, "remote", "add", "origin", str(bare))
    git(repo, "push", "origin", "main:main")
    git(repo, "push", "origin", f"{head_sha}:refs/pull/{number}/head", check=False)
    return str(bare), base_sha, head_sha


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
