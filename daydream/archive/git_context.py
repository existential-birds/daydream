"""Git metadata capture for archived runs.

Captures branch, commit SHA, remote URL, and repo slug from the target
directory at archive time. Each ``git_ops`` call is independent with a
5-second timeout so a single git failure doesn't block the others.

Exports:
    GitContext: Dataclass holding captured git metadata.
    capture_git_context: Capture current git state from a directory.
"""

from dataclasses import dataclass, field
from pathlib import Path

from daydream import git_ops
from daydream.archive.git_safe import normalize_remote_url
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
        base_sha: Merge-base SHA between ``base_branch`` and HEAD; ``None``
            when either side cannot be resolved.
        changed_files: Repo-relative paths changed between ``base_sha`` and
            ``head_sha``. Empty list when ``base_sha`` is ``None`` or the
            diff cannot be computed.
    """

    remote_url: str | None = None
    repo_slug: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    head_sha: str | None = None
    base_sha: str | None = None
    changed_files: list[str] = field(default_factory=list)


def capture_git_context(target_dir: Path) -> GitContext:
    """Capture current git state from *target_dir*.

    Each field is captured independently — a failure in one does not
    prevent the others from being populated.
    """
    ctx = GitContext()

    raw_remote = git_ops.remote_url(target_dir)
    if raw_remote:
        repo_slug, canonical_url = normalize_remote_url(raw_remote)
        # Fail closed: an unparseable raw URL is never stored, not even
        # credential-stripped. Identity alone is None for hosts outside the
        # allowlist, but the credential-stripped canonical URL is still kept.
        if canonical_url is not None:
            ctx.remote_url = canonical_url
            if repo_slug is not None:
                ctx.repo_slug = repo_slug

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

    if ctx.base_branch and ctx.head_sha:
        try:
            ctx.base_sha = git_ops.merge_base(target_dir, ctx.base_branch, ctx.head_sha)
        except GitError:
            ctx.base_sha = None

    if ctx.base_sha and ctx.head_sha:
        try:
            ctx.changed_files = git_ops.diff_name_only(target_dir, ctx.base_sha, ctx.head_sha)
        except GitError:
            ctx.changed_files = []

    return ctx
