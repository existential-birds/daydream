"""Blobless PR acquisition for the benchmark harness.

Acquires a checkout of a pull request's head commit while keeping the base
commit resolvable (so it can be passed to ``daydream --base``). Uses a
blobless partial clone cached per source repo, then fetches the PR head via
GitHub's ``refs/pull/<N>/head`` layout and detaches HEAD onto the head SHA.

All git failures raise :class:`daydream.git_ops.GitError`; nothing is
swallowed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from daydream import git_ops


def _cache_subdir_name(clone_url: str) -> str:
    """Derive a deterministic per-repo subdirectory name from *clone_url*.

    The same URL always maps to the same directory, so a second acquisition
    of the same repo reuses the existing clone instead of re-cloning.
    """
    digest = hashlib.sha256(clone_url.encode("utf-8")).hexdigest()[:16]
    return f"repo-{digest}"


def acquire_checkout(
    clone_url: str,
    pr_number: int,
    base_sha: str,
    head_sha: str,
    *,
    cache_dir: Path,
) -> Path:
    """Acquire a checkout of a PR head with the base commit resolvable.

    One checkout directory is created per *clone_url* under *cache_dir* and
    reused on subsequent calls (no re-clone). The repo is cloned blobless,
    the PR head is fetched via ``refs/pull/<pr_number>/head``, the base SHA
    is fetched if still unresolvable, and HEAD is detached onto *head_sha*.

    Args:
        clone_url: Remote URL or local path to clone from.
        pr_number: Pull request number whose head to acquire.
        base_sha: Base commit SHA; must end up resolvable in the checkout.
        head_sha: Head commit SHA to detach HEAD onto.
        cache_dir: Directory under which the per-repo checkout lives.

    Returns:
        Path to the checkout directory (HEAD detached at *head_sha*).

    Raises:
        GitError: If any git step fails, or HEAD does not land on *head_sha*.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkout = cache_dir / _cache_subdir_name(clone_url)

    if not (checkout / ".git").exists():
        git_ops.clone(clone_url, checkout, blobless=True)

    git_ops.fetch_ref(checkout, f"pull/{pr_number}/head")

    if not git_ops.ref_exists(checkout, base_sha):
        git_ops.fetch_ref(checkout, base_sha)

    git_ops.checkout_paths(checkout, [Path(".")])
    git_ops.clean_untracked(checkout)
    git_ops.checkout_detach(checkout, head_sha)

    resolved = git_ops.head_sha(checkout)
    if resolved != head_sha:
        raise git_ops.GitError(f"checkout HEAD is {resolved!r}, expected {head_sha!r} in {checkout}")

    return checkout
