"""Blobless PR acquisition for the benchmark harness.

Acquires a checkout of a pull request's head commit while keeping the base
commit resolvable (so it can be passed to ``daydream --base``). Uses a
blobless partial clone cached per source repo, then fetches the PR head via
GitHub's ``refs/pull/<N>/head`` layout and detaches HEAD onto the head SHA.

Two corpora supply the base differently, so the base is either **pinned** or
**derived**:

- withmartian: ``base_sha`` is pinned in :mod:`daydream.benchmark.prs`.
- harvested: ``base_sha`` is the PR's recorded base commit, or — for corpora
  harvested before that was captured — only the base branch is known and the
  base is the 3-dot merge-base of ``origin/<base_ref>`` and the bot's review
  snapshot. That derivation is history-dependent: once the snapshot is an
  ancestor of the base branch tip (a merge-commit merge), the merge-base is the
  snapshot itself and the diff is empty, so that case fails loudly. There is
  likewise no first-parent fallback: a two-dot ``head^`` base reviews a
  *different* diff than the bot did, silently corrupting the comparison this
  harness exists to make, so a missing merge-base fails loudly too.

All git failures raise :class:`daydream.git_ops.GitError`; nothing is
swallowed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from daydream import git_ops


@dataclass(frozen=True)
class AcquiredCheckout:
    """A ready-to-review checkout and the base commit its diff is taken against.

    Attributes:
        path: The checkout directory, with HEAD detached onto the head commit.
        base_sha: The resolved base — the pinned SHA, or the derived merge-base.
    """

    path: Path
    base_sha: str


def _cache_subdir_name(clone_url: str, pr_number: int) -> str:
    """Derive a deterministic per-PR subdirectory name from *clone_url* and *pr_number*.

    Keying on both the URL and the PR number gives each PR its own isolated
    working tree, so concurrent sweeps over PRs from the same repo do not
    race on ``git checkout`` / ``git clean`` steps.
    """
    key = f"{clone_url}#{pr_number}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"repo-{digest}"


def acquire_checkout(
    clone_url: str,
    pr_number: int,
    head_sha: str,
    *,
    base_sha: str | None = None,
    base_ref: str | None = None,
    cache_dir: Path,
) -> AcquiredCheckout:
    """Acquire a checkout of *head_sha* with its review base resolved.

    One checkout directory is created per *(clone_url, pr_number)* pair under
    *cache_dir* and reused on subsequent calls (no re-clone). The repo is
    cloned blobless, the PR head ref is fetched via
    ``refs/pull/<pr_number>/head``, and HEAD is detached onto *head_sha*.

    Args:
        head_sha: The commit to review — the PR head, or a bot's review snapshot.
        base_sha: A pinned base commit. Mutually exclusive with *base_ref*.
        base_ref: The PR's base branch name; the base is derived as the
            merge-base of ``origin/<base_ref>`` and *head_sha*. Mutually
            exclusive with *base_sha*.

    Raises:
        ValueError: If both or neither of *base_sha* / *base_ref* is given.
        GitError: If any git step fails, HEAD does not land on *head_sha*, or
            the merge-base of ``origin/<base_ref>`` and *head_sha* is missing or
            is *head_sha* itself (which would yield an empty diff).
    """
    if (base_sha is None) == (base_ref is None):
        raise ValueError("acquire_checkout needs exactly one of base_sha / base_ref")

    cache_dir.mkdir(parents=True, exist_ok=True)
    checkout = cache_dir / _cache_subdir_name(clone_url, pr_number)

    if not (checkout / ".git").exists():
        git_ops.clone(clone_url, checkout, blobless=True)

    git_ops.fetch_ref(checkout, f"pull/{pr_number}/head")

    # Fetching a raw SHA via `git fetch origin <sha>` is rejected by GitHub's
    # upload-pack for non-advertised (non-tip) commits.  A plain `git fetch
    # origin` fetches all advertised refs and their reachable history, which is
    # sufficient to make a pinned base — or an older review snapshot — resolvable.
    probe = base_sha if base_sha is not None else head_sha
    if not git_ops.ref_exists(checkout, probe):
        git_ops.fetch(checkout)

    git_ops.checkout_paths(checkout, [Path(".")])
    git_ops.clean_untracked(checkout)
    git_ops.checkout_detach(checkout, head_sha)

    resolved = git_ops.head_sha(checkout)
    if resolved != head_sha:
        raise git_ops.GitError(f"checkout HEAD is {resolved!r}, expected {head_sha!r} in {checkout}")

    if base_sha is not None:
        return AcquiredCheckout(path=checkout, base_sha=base_sha)

    merge_base = git_ops.merge_base(checkout, f"origin/{base_ref}", head_sha)
    if merge_base is None:
        raise git_ops.GitError(f"no merge-base between origin/{base_ref} and {head_sha} in {checkout}")
    if merge_base == head_sha:
        raise git_ops.GitError(
            f"derived base equals head {head_sha} in {checkout}: {head_sha} is already an ancestor of "
            f"origin/{base_ref}, so the diff would be empty; re-harvest to pin the PR's base_sha"
        )
    return AcquiredCheckout(path=checkout, base_sha=merge_base)
