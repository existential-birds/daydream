"""Git metadata capture for archived runs.

Captures branch, commit SHA, remote URL, and repo slug from the target
directory at archive time. Each subprocess call is independent with a
5-second timeout so a single git failure doesn't block the others.

Exports:
    GitContext: Dataclass holding captured git metadata.
    capture_git_context: Capture current git state from a directory.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitContext:
    """Git metadata for an archived run.

    Attributes:
        remote_url: Origin remote URL (HTTPS or SSH).
        repo_slug: ``owner/repo`` extracted from remote_url.
        branch: Current branch name.
        base_branch: Default branch (main/master).
        head_sha: Full commit SHA of HEAD.
    """

    remote_url: str | None = None
    repo_slug: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    head_sha: str | None = None


_SSH_REMOTE_RE = re.compile(r"[^@]+@[^:]+:(.+?)(?:\.git)?$")
_HTTPS_REMOTE_RE = re.compile(r"https?://[^/]+/(.+?)(?:\.git)?$")


def _parse_repo_slug(remote_url: str) -> str | None:
    """Extract ``owner/repo`` from a git remote URL.

    Handles both SSH (``git@github.com:owner/repo.git``) and HTTPS
    (``https://github.com/owner/repo.git``) formats.
    """
    m = _SSH_REMOTE_RE.match(remote_url)
    if m:
        return m.group(1)
    m = _HTTPS_REMOTE_RE.match(remote_url)
    if m:
        return m.group(1)
    return None


def _run_git(cwd: Path, *args: str) -> str | None:
    """Run a git command and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(  # noqa: S603 - args are not user-controlled
            ["git", *args],  # noqa: S607 - git is a trusted command
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def capture_git_context(target_dir: Path) -> GitContext:
    """Capture current git state from *target_dir*.

    Each field is captured independently — a failure in one does not
    prevent the others from being populated.
    """
    ctx = GitContext()

    ctx.remote_url = _run_git(target_dir, "config", "--get", "remote.origin.url")
    if ctx.remote_url:
        ctx.repo_slug = _parse_repo_slug(ctx.remote_url)

    ctx.branch = _run_git(target_dir, "branch", "--show-current")
    ctx.head_sha = _run_git(target_dir, "rev-parse", "HEAD")

    # Detect default branch
    symbolic = _run_git(target_dir, "symbolic-ref", "refs/remotes/origin/HEAD")
    if symbolic:
        ctx.base_branch = symbolic.rsplit("/", 1)[-1]
    else:
        # Fallback: check for main or master
        for candidate in ("main", "master"):
            check = _run_git(target_dir, "rev-parse", "--verify", f"refs/heads/{candidate}")
            if check:
                ctx.base_branch = candidate
                break

    return ctx
