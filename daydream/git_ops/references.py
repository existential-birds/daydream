"""Branch/reference resolution and immutable diff/PR merge-base selection."""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from daydream.git_ops import process
from daydream.git_ops.models import BranchNotFoundError, GitError

_FULL_OBJECT_ID_RE = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")


def default_branch(repo: Path) -> str:
    """Resolve ``origin/HEAD``, then local ``main``, then local ``master``.

    Raise ``BranchNotFoundError`` if none exists.
    """
    sym = process._run_git(repo, ["symbolic-ref", "refs/remotes/origin/HEAD"], timeout=5)
    if sym.returncode == 0 and sym.stdout.strip():
        return sym.stdout.strip().rsplit("/", 1)[-1]

    for candidate in ("main", "master"):
        check = process._run_git(repo, ["rev-parse", "--verify", f"refs/heads/{candidate}"], timeout=5)
        if check.returncode == 0:
            return candidate

    raise BranchNotFoundError(f"no default branch (origin/HEAD, main, master) found in {repo}")


def branch_exists(repo: Path, ref: str) -> bool:
    """Check whether *ref* exists locally or as ``origin/<ref>``."""
    local = process._run_git(repo, ["rev-parse", "--verify", f"refs/heads/{ref}"], timeout=5)
    if local.returncode == 0:
        return True
    remote = process._run_git(repo, ["rev-parse", "--verify", f"refs/remotes/origin/{ref}"], timeout=5)
    return remote.returncode == 0


def _has_leading_dash(*refs: str) -> bool:
    """Reject commit-ish inputs that Git would interpret as option flags."""
    return any(ref.startswith("-") for ref in refs)


def ref_exists(repo: Path, ref: str) -> bool:
    """Probe a local commit-ish, with an ``origin/<ref>`` fallback for branch names.

    Missing refs and leading-dash inputs return ``False``; subprocess failures raise.
    """
    if branch_exists(repo, ref):
        return True
    return commit_exists(repo, ref)


def commit_exists(repo: Path, revision: str) -> bool:
    """Probe ``revision^{commit}`` using Git's normal local revision resolution.

    No implicit ``origin/`` fallback. Missing refs, non-commit objects, and leading-dash
    inputs return ``False``; subprocess failures raise.
    """
    # Same guard as ref_exists: no valid commit-ish starts with '-'.
    if _has_leading_dash(revision):
        return False
    commit = process._run_git(repo, ["rev-parse", "--verify", f"{revision}^{{commit}}"], timeout=5)
    return commit.returncode == 0


def is_ancestor(repo: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    """Probe ancestry; missing, unrelated, or leading-dash refs return ``False``.

    Subprocess failures still raise ``GitError``.
    """
    # Same guard as ref_exists and merge_base: a leading '-' would be
    # mis-parsed by git as an option flag. No valid branch name or commit-ish
    # starts with '-'.
    if _has_leading_dash(ancestor, descendant):
        return False

    proc = process._run_git(repo, ["merge-base", "--is-ancestor", ancestor, descendant], timeout=5)
    return proc.returncode == 0


def merge_base(repo: Path, base: str, head: str = "HEAD") -> str | None:
    """Find the merge-base, preferring an upstream with commits ahead of local base.

    Follows the algorithm in ``codex-rs/git-utils/src/branch.rs`` to avoid stale local
    bases. Missing refs or empty repositories return ``None``; subprocess failures raise.
    """
    # Same guard as ref_exists: a leading '-' would be mis-parsed as a git
    # option flag.  No valid branch name or commit-ish starts with '-'.
    if _has_leading_dash(head, base):
        return None

    head_proc = process._run_git(repo, ["rev-parse", "--verify", head], timeout=5)
    if head_proc.returncode != 0:
        return None

    base_proc = process._run_git(repo, ["rev-parse", "--verify", base], timeout=5)
    if base_proc.returncode != 0:
        return None
    preferred_ref = base

    upstream, ahead = _upstream_and_ahead(repo, base)
    if upstream is not None and ahead > 0:
        preferred_ref = upstream

    mb = process._run_git(repo, ["merge-base", head, preferred_ref], timeout=5)
    if mb.returncode != 0:
        return None
    out = mb.stdout.strip()
    return out or None


def _upstream_and_ahead(repo: Path, branch: str) -> tuple[str | None, int]:
    """Return the symbolic upstream and its right-side ahead count; unreadable or absent
    upstream/count yields (None, 0).
    """
    upstream_name_proc = process._run_git(
        repo,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", f"{branch}@{{upstream}}"],
        timeout=5,
    )
    if upstream_name_proc.returncode != 0:
        return None, 0
    upstream = upstream_name_proc.stdout.strip()
    if not upstream:
        return None, 0

    counts_proc = process._run_git(
        repo,
        ["rev-list", "--left-right", "--count", f"{branch}...{upstream}"],
        timeout=5,
    )
    if counts_proc.returncode != 0:
        return None, 0
    parts = counts_proc.stdout.strip().split()
    try:
        right = int(parts[1]) if len(parts) >= 2 else 0
    except ValueError:
        right = 0
    return upstream, right


def _prefer_remote_base(repo: Path, base: str) -> str:
    """Return ``origin/<base>`` when it exists, otherwise *base* unchanged."""
    remote_ref = f"origin/{base}"
    check = process._run_git(repo, ["rev-parse", "--verify", f"refs/remotes/{remote_ref}"], timeout=5)
    if check.returncode == 0:
        return remote_ref
    return base


def _validated_diff_object_id(stdout: str) -> str | None:
    """Return one canonical SHA-1 line, rejecting every other representation."""
    if re.fullmatch(r"[0-9a-f]{40}\n?", stdout) is None:
        return None
    return stdout.removesuffix("\n")


def resolve_diff_merge_base(repo: Path, base: str, head_sha: str) -> str:
    """Pin both input commits before computing the same merge-base that ``diff`` uses.

    Prefer a present ``origin/<base>`` even without an upstream relationship. Resolving
    immutable OIDs first prevents concurrent ref moves from changing the second probe.
    Unresolvable commits or merge-base raise ``GitError``.
    """
    if _has_leading_dash(base, head_sha):
        raise GitError("branch-focus diff base or head is invalid")

    try:
        preferred_base = _prefer_remote_base(repo, base)
    except (GitError, UnicodeError) as exc:
        raise GitError("branch-focus preferred base probe failed") from exc

    def _commit_oid(ref: str, label: str) -> str:
        try:
            proc = process._run_git(
                repo,
                ["rev-parse", "--verify", f"{ref}^{{commit}}"],
                timeout=5,
            )
        except (GitError, UnicodeError) as exc:
            raise GitError(f"branch-focus {label} commit probe failed") from exc
        oid = _validated_diff_object_id(proc.stdout) if proc.returncode == 0 else None
        if oid is None:
            raise GitError(f"branch-focus {label} commit cannot be resolved")
        return oid

    resolved_head = _commit_oid(head_sha, "recorded head")
    resolved_base = _commit_oid(preferred_base, "preferred base")
    try:
        proc = process._run_git(repo, ["merge-base", resolved_head, resolved_base], timeout=5)
    except (GitError, UnicodeError) as exc:
        raise GitError("branch-focus diff merge-base probe failed") from exc
    merge_base_sha = _validated_diff_object_id(proc.stdout) if proc.returncode == 0 else None
    if merge_base_sha is None:
        raise GitError("branch-focus diff merge-base cannot be resolved")
    return merge_base_sha


def _validate_pr_base_ref(repo: Path, ref: str, *, prefix: str) -> None:
    if not ref.startswith(prefix):
        raise GitError(f"invalid PR base ref {ref[:120]!r} in {repo}")
    proc = process._run_git(repo, ["check-ref-format", ref], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"invalid PR base ref {ref[:120]!r} in {repo}")


def _merge_base_strict(repo: Path, base_ref: str, head_sha: str) -> str:
    proc = process._run_git(repo, ["merge-base", base_ref, head_sha], timeout=10)
    merge_sha = proc.stdout.strip()
    if proc.returncode != 0 or _FULL_OBJECT_ID_RE.fullmatch(merge_sha) is None:
        raise GitError(f"no merge-base for PR head {head_sha} and base {base_ref} in {repo}")
    return merge_sha.lower()


def resolve_pr_merge_base(
    repo: Path,
    remote_base_refs: Sequence[str],
    local_base_ref: str,
    head_sha: str,
) -> str:
    """Resolve the exact PR head's merge-base against authoritative local refs."""
    if _FULL_OBJECT_ID_RE.fullmatch(head_sha) is None:
        raise GitError(f"exact PR head {head_sha[:120]!r} is not a full object ID in {repo}")
    head_proc = process._run_git(repo, ["rev-parse", "--verify", f"{head_sha}^{{commit}}"], timeout=5)
    resolved_head = head_proc.stdout.strip()
    if (
        head_proc.returncode != 0
        or _FULL_OBJECT_ID_RE.fullmatch(resolved_head) is None
        or resolved_head.lower() != head_sha.lower()
    ):
        raise GitError(f"exact PR head {head_sha} is not a local commit in {repo}")

    _validate_pr_base_ref(repo, local_base_ref, prefix="refs/heads/")
    unique_remote_refs = sorted(set(remote_base_refs))
    for ref in unique_remote_refs:
        _validate_pr_base_ref(repo, ref, prefix="refs/remotes/")

    present_remote_refs: list[str] = []
    for ref in unique_remote_refs:
        check = process._run_git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"], timeout=5)
        if check.returncode == 0:
            present_remote_refs.append(ref)

    if present_remote_refs:
        bases = {ref: _merge_base_strict(repo, ref, head_sha) for ref in present_remote_refs}
        distinct = set(bases.values())
        if len(distinct) != 1:
            refs = ", ".join(present_remote_refs)
            raise GitError(f"matching PR base remotes disagree ({refs}) in {repo}; fetch/align the base remote refs")
        return next(iter(distinct))

    local_check = process._run_git(repo, ["rev-parse", "--verify", f"{local_base_ref}^{{commit}}"], timeout=5)
    if local_check.returncode != 0:
        raise GitError(f"PR base {local_base_ref} is unavailable for head {head_sha} in {repo}")
    return _merge_base_strict(repo, local_base_ref, head_sha)


def upstream_ahead_count(repo: Path, branch: str) -> int:
    """Return the upstream right-side ahead count, or zero when unavailable."""
    return _upstream_and_ahead(repo, branch)[1]
