"""Real Git helpers and deterministic seed repositories.

SEED_ENV fixes seed identity and timestamps. Migrate's dateless identity,
Harbor bundle identity, and production snapshot/RL fixtures remain distinct.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

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


@contextlib.contextmanager
def install_diff_base_git_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mode: str, head: str,
) -> Iterator[Path]:
    """Put an external ``git`` shim on ``PATH`` for the duration of the block.

    The shim answers the branch-focus resolver's probes from ``mode`` (one of
    ``malformed-head``, ``uppercase-head``, ``abbreviated-head``, ``sha256-head``,
    ``invalid-utf8-head``, ``malformed-merge``, ``remove-after-preference``, which delegates then
    deletes itself, and ``timeout-preference``) and delegates everything else to the real git, so
    resolver failure paths and their redaction run against a real transport. ``head`` is the
    recorded SHA whose ``rev-parse --verify <head>^{commit}`` probe is answered. Yields the shim
    directory so callers can assert its name never reaches a message.
    """
    real_git = shutil.which("git")
    assert real_git is not None
    shim_dir = tmp_path / "diff base git shim"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "git"
    # Adjacent string literals, not an f-string: the invalid-utf8 mode must emit exact bytes.
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys, time\n"
        "args = sys.argv[1:]\n"
        "mode = os.environ['DAYDREAM_TEST_GIT_SHIM_MODE']\n"
        "real = os.environ['DAYDREAM_TEST_REAL_GIT']\n"
        "head = os.environ['DAYDREAM_TEST_HEAD']\n"
        "is_preference = args[:2] == ['rev-parse', '--verify'] and "
        "len(args) == 3 and args[2].startswith('refs/remotes/origin/')\n"
        "is_head = args[:2] == ['rev-parse', '--verify'] and "
        "len(args) == 3 and args[2] == head + '^{commit}'\n"
        "if mode == 'malformed-head' and is_head:\n"
        "    print('PRIVATE_STDOUT_SENTINEL')\n"
        "    raise SystemExit(0)\n"
        "if mode == 'uppercase-head' and is_head:\n"
        "    print(head.upper())\n"
        "    raise SystemExit(0)\n"
        "if mode == 'abbreviated-head' and is_head:\n"
        "    print(head[:12])\n"
        "    raise SystemExit(0)\n"
        "if mode == 'sha256-head' and is_head:\n"
        "    print(head + head[:24])\n"
        "    raise SystemExit(0)\n"
        "if mode == 'invalid-utf8-head' and is_head:\n"
        "    os.write(1, b'\\xff\\xfe\\n')\n"
        "    raise SystemExit(0)\n"
        "if mode == 'malformed-merge' and args[:1] == ['merge-base']:\n"
        "    print('PRIVATE_MERGE_SENTINEL')\n"
        "    raise SystemExit(0)\n"
        "if mode == 'timeout-preference' and is_preference:\n"
        "    time.sleep(30)\n"
        "if mode == 'remove-after-preference' and is_preference:\n"
        "    result = subprocess.run([real, *args])\n"
        "    os.unlink(sys.argv[0])\n"
        "    raise SystemExit(result.returncode)\n"
        "raise SystemExit(subprocess.run([real, *args]).returncode)\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    with monkeypatch.context() as shim_env:
        shim_env.setenv("DAYDREAM_TEST_GIT_SHIM_MODE", mode)
        shim_env.setenv("DAYDREAM_TEST_REAL_GIT", real_git)
        shim_env.setenv("DAYDREAM_TEST_HEAD", head)
        shim_env.setenv("PATH", str(shim_dir))
        yield shim_dir


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
