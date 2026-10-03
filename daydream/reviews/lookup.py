"""Resolve fork-aware pull request identity and immutable base evidence."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.git_ops import INHERIT_GITHUB_AUTH, GitError, GitHubAuth
from daydream.reviews.models import PRInfo


def _head_repo_slug_from_row(row: dict[str, Any]) -> str | None:
    """Resolve the head repository, rejecting malformed or contradictory identities.

    A deleted fork (null repository) allows base-repository link fallback. Older gh
    versions omit nameWithOwner or emit an empty string; validated owner/name may
    reconstruct it. Populated fields must agree case-insensitively."""
    if "headRepository" not in row or "headRepositoryOwner" not in row:
        raise GitError("invalid PR row: missing requested head repository metadata")
    head_owner = row["headRepositoryOwner"]
    owner_login: str | None = None
    if head_owner is not None:
        if not isinstance(head_owner, dict) or not isinstance(head_owner.get("login"), str):
            raise GitError("invalid PR row: malformed head repository owner")
        owner_login = head_owner["login"]
        if git_ops.split_owner_repo(f"{owner_login}/repository") is None:
            raise GitError("invalid PR row: malformed head repository owner")
    head_repo = row["headRepository"]
    if head_repo is None:
        return None
    if not isinstance(head_repo, dict):
        raise GitError("invalid PR row: headRepository must be an object or null")
    repo_name = head_repo.get("name")
    if "name" in head_repo and (
        not isinstance(repo_name, str)
        or git_ops.split_owner_repo(f"owner/{repo_name}") is None
    ):
        raise GitError("invalid PR row: malformed head repository name")
    if "nameWithOwner" in head_repo:
        name_with_owner = head_repo["nameWithOwner"]
        if not isinstance(name_with_owner, str):
            raise GitError("invalid PR row: head repository nameWithOwner must be a string")
        if name_with_owner != "":
            slug = git_ops.split_owner_repo(name_with_owner)
            if slug is None:
                raise GitError("invalid PR row: malformed head repository slug")
            if (
                (owner_login is not None and slug[0].casefold() != owner_login.casefold())
                or (isinstance(repo_name, str) and slug[1].casefold() != repo_name.casefold())
            ):
                raise GitError("invalid PR row: contradictory head repository identity")
            return name_with_owner
    if owner_login is not None and isinstance(repo_name, str):
        return f"{owner_login}/{repo_name}"
    raise GitError("invalid PR row: unavailable head repository slug requires owner and name")



def _pr_info_from_row(
    target_dir: Path, row: dict[str, Any], *, auth: GitHubAuth = INHERIT_GITHUB_AUTH
) -> PRInfo:
    """Resolve the base posting repository and fork-aware head link.

    Raise GitError for invalid PR metadata, repository context, or local merge base."""
    from daydream.archive.git_safe import normalize_remote_url

    number = row.get("number")
    head_sha = row.get("headRefOid")
    head_ref = row.get("headRefName")
    base_ref = row.get("baseRefName")
    url = row.get("url")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise GitError("invalid PR row: number must be a positive integer")
    if not isinstance(head_sha, str):
        raise GitError("invalid PR row: headRefOid must be a string")
    if not isinstance(head_ref, str) or not head_ref:
        raise GitError("invalid PR row: headRefName must be a non-empty string")
    if not isinstance(base_ref, str) or not base_ref:
        raise GitError("invalid PR row: baseRefName must be a non-empty string")
    if not isinstance(url, str) or not url:
        raise GitError("invalid PR row: url must be a non-empty string")
    head_repo = _head_repo_slug_from_row(row)
    git_ops.validate_branch_name(target_dir, head_ref)
    git_ops.validate_branch_name(target_dir, base_ref)

    owner, repo = git_ops.gh_repo_view_required(target_dir, auth=auth)
    base_slug = f"{owner}/{repo}"
    matching_remotes: list[str] = []
    for remote, raw_url in git_ops.remote_urls(target_dir).items():
        identity, _safe_url = normalize_remote_url(raw_url)
        if identity is not None and identity.lower() == base_slug.lower():
            matching_remotes.append(f"refs/remotes/{remote}/{base_ref}")
    base_sha = git_ops.resolve_pr_merge_base(
        target_dir,
        matching_remotes,
        f"refs/heads/{base_ref}",
        head_sha,
    )
    return PRInfo(
        number=number,
        head_sha=head_sha,
        base_sha=base_sha,
        base_ref=base_ref,
        head_ref=head_ref,
        owner=owner,
        repo=repo,
        url=url,
        head_repo=head_repo,
        pr_base_sha=row.get("baseRefOid") if isinstance(row.get("baseRefOid"), str) else None,
    )



def capture_pr_base_tip(
    target_dir: Path, pr: PRInfo, *, auth: GitHubAuth = INHERIT_GITHUB_AUTH,
) -> str | None:
    """Read optional base-tip evidence without requiring newer gh JSON fields.

    A PR that advanced between reads cannot supply evidence for this snapshot.
    The required identity and merge base already come from the initial lookup.
    """
    if pr.pr_base_sha is not None:
        return pr.pr_base_sha
    try:
        data = git_ops.gh_api(target_dir, f"repos/{pr.owner}/{pr.repo}/pulls/{pr.number}", auth=auth)
    except GitError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("head"), dict):
        return None
    if data["head"].get("sha") != pr.head_sha or not isinstance(data.get("base"), dict):
        return None
    sha = data["base"].get("sha")
    return sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha) else None



def find_open_pr(
    target_dir: Path, *, auth: GitHubAuth = INHERIT_GITHUB_AUTH
) -> PRInfo | None:
    """Find the branch's open PR; return None for absence and raise GitError on failure."""
    branch = git_ops.current_branch(target_dir)
    if not branch:
        return None
    rows = git_ops.gh_pr_list_for_branch(target_dir, branch, auth=auth)
    if not rows:
        return None
    return _pr_info_from_row(target_dir, rows[0], auth=auth)



def find_pr_by_number(
    target_dir: Path, pr_number: int, *, auth: GitHubAuth = INHERIT_GITHUB_AUTH
) -> PRInfo | None:
    """Find an explicit PR without branch fallback; raise GitError on lookup failure."""
    data = git_ops.gh_pr_view(target_dir, pr_number, auth=auth)
    if data is None:
        return None
    return _pr_info_from_row(target_dir, data, auth=auth)
