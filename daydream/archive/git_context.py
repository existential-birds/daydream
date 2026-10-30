"""Capture target-repository identity with independent, five-second Git queries."""

from dataclasses import dataclass, replace
from pathlib import Path

from daydream import git_ops
from daydream.archive.git_safe import normalize_remote_url
from daydream.git_ops import BranchNotFoundError, GitError


@dataclass(frozen=True)
class GitContext:
    """Credential-free repository identity retained for archiving.

    Unresolved merge bases are None; changed_files is empty when the merge base
    or diff is unavailable.
    """

    remote_url: str | None = None
    repo_slug: str | None = None
    branch: str | None = None
    base_branch: str | None = None
    head_sha: str | None = None
    base_sha: str | None = None
    changed_files: tuple[str, ...] = ()


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
            ctx = replace(ctx, remote_url=canonical_url, repo_slug=repo_slug)

    try:
        ctx = replace(ctx, branch=git_ops.current_branch(target_dir))
    except GitError:
        pass

    try:
        ctx = replace(ctx, head_sha=git_ops.head_sha(target_dir))
    except GitError:
        pass

    try:
        ctx = replace(ctx, base_branch=git_ops.default_branch(target_dir))
    except (BranchNotFoundError, GitError):
        pass

    if ctx.base_branch and ctx.head_sha:
        try:
            ctx = replace(ctx, base_sha=git_ops.merge_base(target_dir, ctx.base_branch, ctx.head_sha))
        except GitError:
            pass

    if ctx.base_sha and ctx.head_sha:
        try:
            ctx = replace(ctx, changed_files=tuple(git_ops.diff_name_only(target_dir, ctx.base_sha, ctx.head_sha)))
        except GitError:
            pass

    return ctx
