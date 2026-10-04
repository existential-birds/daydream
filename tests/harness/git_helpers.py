"""Real Git helpers and deterministic seed repositories.

SEED_ENV fixes seed identity and timestamps. Migrate's dateless identity,
Harbor bundle identity, and production snapshot/RL fixtures remain distinct.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.workspace import WorkContext

SEED_ENV: dict[str, str] = {
    "GIT_AUTHOR_NAME": "Tester", "GIT_AUTHOR_EMAIL": "test@example.com", "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Tester", "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def git(repo: Path, *args: str, check: bool = True, env: dict[str, str] | None = None) -> str:
    """Run a git command in *repo* and return stripped stdout (test helper)."""
    proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", *args],  # noqa: S607 - git is a trusted command
        cwd=repo, capture_output=True, text=True, env={**os.environ, **env} if env else None, check=check,
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
    return WorkContext(repo=repo, source=repo, base_branch="main", base_sha=head, head_branch="main", head_sha=head,
        is_ephemeral=False, run_id=run_id,
    )


def write_and_stage(repo: Path, name: str, content: str | bytes) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    git(repo, "add", name)


def seeded_commit(repo: Path, message: str) -> str:
    """Commit with the deterministic ``SEED_ENV`` identity and return the new SHA."""
    return commit(repo, message, env=SEED_ENV)


def seed_pr_origin(tmp_path: Path, *, repo_name: str = "local_wt", bare_name: str = "origin_local.git",
    feature_body: str = "FEATURE = 1\n",
    feature_message: str = "feature", number: int = 101,
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


def seed_feature_branch(repo: Path, *, base: dict[str, str], feature: dict[str, str], base_message: str = "base",
    feature_message: str = "feature", base_ref: str | None = None,
) -> str:
    """Create main and feature commits from path/content maps; return feature HEAD.

    ``base_ref`` additionally branches the base commit under that name, for callers whose
    review config resolves ``--base`` against a literal ref rather than a SHA.
    """
    init_repo(repo)
    for name, content in base.items():
        write_and_stage(repo, name, content)
    commit(repo, base_message)
    if base_ref is not None:
        git(repo, "branch", base_ref)
    git(repo, "checkout", "-b", "feature")
    for name, content in feature.items():
        write_and_stage(repo, name, content)
    return commit(repo, feature_message)


def refreshing_session(name: str, refresh_calls: dict[str, int]) -> git_ops.RefreshingGitHubAuth:
    """Create an expired session whose refresh increments its counter and returns a fresh token."""

    def refresh() -> tuple[git_ops.StaticGitHubAuth, float]:
        refresh_calls[name] += 1
        return (git_ops.StaticGitHubAuth({"PATH": f"/{name}/tools", "GH_TOKEN": f"ghs_{name}_fresh_token_1234567890"}),
            float("inf"),
        )

    return git_ops.RefreshingGitHubAuth(
        git_ops.StaticGitHubAuth({"PATH": f"/{name}/tools", "GH_TOKEN": f"ghs_{name}_expired_token_1234567890"}),
        expires_at=0, refresh=refresh,
    )


def bare_remote(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "--bare", "-b", "main")
    return path


def tracked_source_state(repo: Path) -> dict[str, Any]:
    """Everything a review run must never mutate in the source checkout."""
    tracked = git(repo, "ls-files").splitlines()
    return {"head": git(repo, "rev-parse", "HEAD"), "refs": git(repo, "show-ref"),
        "index": git(repo, "ls-files", "--stage"), "diff": git(repo, "diff", "--binary", "HEAD"),
        "bytes": {name: (repo / name).read_bytes() for name in tracked},
    }
