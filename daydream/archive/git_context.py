"""Git metadata capture for archived runs.

Captures branch, commit SHA, remote URL, and repo slug from the target
directory at archive time. Each ``git_ops`` call is independent with a
5-second timeout so a single git failure doesn't block the others.

Exports:
    GitContext: Dataclass holding captured git metadata.
    capture_git_context: Capture current git state from a directory.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from daydream import git_ops
from daydream.git_ops import BranchNotFoundError, GitError


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


def capture_git_context(target_dir: Path) -> GitContext:
    """Capture current git state from *target_dir*.

    Each field is captured independently — a failure in one does not
    prevent the others from being populated.
    """
    ctx = GitContext()

    ctx.remote_url = git_ops.remote_url(target_dir)
    if ctx.remote_url:
        ctx.repo_slug = _parse_repo_slug(ctx.remote_url)

    try:
        ctx.branch = git_ops.current_branch(target_dir)
    except GitError:
        ctx.branch = None

    try:
        ctx.head_sha = git_ops.head_sha(target_dir)
    except GitError:
        ctx.head_sha = None

    try:
        ctx.base_branch = git_ops.default_branch(target_dir)
    except (BranchNotFoundError, GitError):
        ctx.base_branch = None

    return ctx
