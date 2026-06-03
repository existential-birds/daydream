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
import subprocess
from pathlib import Path

from daydream import git_ops


def _cache_subdir_name(clone_url: str) -> str:
    """Derive a deterministic per-repo subdirectory name from *clone_url*.

    The same URL always maps to the same directory, so a second acquisition
    of the same repo reuses the existing clone instead of re-cloning.
    """
    digest = hashlib.sha256(clone_url.encode("utf-8")).hexdigest()[:16]
    return f"repo-{digest}"


def _fetch_ref(repo: Path, refspec: str) -> None:
    """Fetch a single *refspec* from ``origin`` into *repo*.

    No ``git_ops`` helper exists for ``refs/pull/*`` fetches, so this issues
    the guarded subprocess call directly.

    Raises:
        GitError: If the fetch fails for any reason.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
            ["git", "-C", str(repo), "fetch", "origin", refspec],  # noqa: S607 - git is a trusted command
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise git_ops.GitError(f"git fetch origin {refspec} failed in {repo}: {type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        raise git_ops.GitError(f"git fetch origin {refspec} failed in {repo}: {proc.stderr.strip()}")


def _checkout_detach(repo: Path, sha: str) -> None:
    """Detach HEAD onto *sha* in *repo*.

    Raises:
        GitError: If the checkout fails or HEAD does not land on *sha*.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
            ["git", "-C", str(repo), "checkout", "--detach", sha],  # noqa: S607 - git is a trusted command
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise git_ops.GitError(f"git checkout --detach {sha} failed in {repo}: {type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        raise git_ops.GitError(f"git checkout --detach {sha} failed in {repo}: {proc.stderr.strip()}")


def _rev_parse_head(repo: Path) -> str:
    """Return the SHA that HEAD resolves to in *repo*.

    Raises:
        GitError: If the rev-parse fails.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
            ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S607 - git is a trusted command
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise git_ops.GitError(f"git rev-parse HEAD failed in {repo}: {type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        raise git_ops.GitError(f"git rev-parse HEAD failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip()


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

    _fetch_ref(checkout, f"pull/{pr_number}/head")

    if not git_ops.ref_exists(checkout, base_sha):
        _fetch_ref(checkout, base_sha)

    _checkout_detach(checkout, head_sha)

    resolved = _rev_parse_head(checkout)
    if resolved != head_sha:
        raise git_ops.GitError(f"checkout HEAD is {resolved!r}, expected {head_sha!r} in {checkout}")

    return checkout
