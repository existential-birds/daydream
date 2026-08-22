"""Freeze reproducible, self-contained ``base -> head`` PR snapshot bundles.

This module owns the shared bare mirror (``cache/repository.git``), the base tip
+ ``refs/pull/N/head`` + explicit-head ref fetch into it, ancestor-of-PR-head
enforcement, merge-base + tree resolution, synthetic-commit + minimal-bundle
construction, offline-clone validation, and the ``freeze_one`` orchestrator
that yields exactly one ``ready`` or ``unreplayable`` snapshot outcome.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from daydream import git_ops
from daydream.benchmark import schema, storage

# Pinned synthetic-commit identity/timestamp so a bundle is byte-identical
# across repeated builds (Spike Finding 1).
_SYNTH_AUTHOR = {
    "GIT_AUTHOR_NAME": "Daydream Snapshot",
    "GIT_AUTHOR_EMAIL": "benchmark@daydream",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Daydream Snapshot",
    "GIT_COMMITTER_EMAIL": "benchmark@daydream",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def mirror(root: Path) -> Path:
    """The shared bare mirror path for a workspace root."""
    return Path(root) / "cache" / "repository.git"


def rev_parse(repo: Path, ref: str) -> str:
    """Resolve *ref* to a 40-hex SHA in *repo*, raising GitError on absence."""
    proc = git_ops._run_git(repo, ["rev-parse", "--verify", ref], retries=0)
    if proc.returncode != 0:
        raise git_ops.GitError(
            f"git rev-parse --verify {ref} failed in {repo}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _fetch_env() -> dict[str, str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    return env


def ensure_mirror(root: Path, repo_slug: str, origin_url: str | None = None) -> Path:
    """Return ``root/cache/repository.git``, creating the bare mirror if absent.

    Idempotent: a present bare mirror is returned untouched.
    """
    root = Path(root)
    m = mirror(root)
    if not m.exists():
        m.parent.mkdir(parents=True, exist_ok=True)
        proc = git_ops._run_git(
            root, ["init", "--bare", str(m)], env_cmd=_fetch_env(), retries=0, timeout=30,
        )
        if proc.returncode != 0:
            raise git_ops.GitError(
                f"git init --bare {m} failed: {proc.stderr.strip()}"
            )
    return m


def _git_fetch(mirror_repo: Path, url: str, refspecs: list[str]) -> None:
    """Fetch refspecs into the mirror. An unresolved refspec raises ``GitError``."""
    url = str(url)
    args = ["fetch", url, *refspecs]
    if url.startswith("https://"):
        args = ["-c", "credential.helper=!gh auth git-credential", *args]
    proc = git_ops._run_git(mirror_repo, args, env_cmd=_fetch_env(), retries=0, timeout=300)
    if proc.returncode != 0:
        raise git_ops.GitError(
            f"git fetch {url} {' '.join(refspecs)} failed: {proc.stderr.strip()}"
        )


def fetch_pr_refs(
    root: Path,
    repo_slug: str,
    pr_number: int,
    base_tip: str,
    explicit_shas: list[str] | tuple[str, ...] = (),
    origin_url: str | None = None,
) -> Path:
    """Fetch base tip + ``refs/pull/N/head`` + explicit heads into the mirror.

    The base tip and PR-head ref are fetched by name; the PR-head SHA is then
    resolvable from ``refs/pull/N/head``. Returns the mirror path. Idempotent.
    """
    root = Path(root)
    origin_url = origin_url or f"https://github.com/{repo_slug}.git"
    m = ensure_mirror(root, repo_slug, origin_url)
    refspecs = [f"{base_tip}:refs/heads/base_tip", f"refs/pull/{pr_number}/head:refs/pull/{pr_number}/head"]
    for sha in explicit_shas:
        refspecs.append(f"{sha}:refs/heads/explicit-{sha[:12]}")
    _git_fetch(m, origin_url, refspecs)
    return m


def head_reachability(mirror_repo: Path, sha: str, pr_head_sha: str) -> str:
    """Classify how an explicit *sha* relates to the PR-head ancestry.

    Returns ``"ok"`` when ``sha`` equals the PR head or is an ancestor of it,
    ``"head_not_on_pr"`` when ``sha`` is in the mirror but on a different
    ancestry, and ``"head_unreachable"`` when ``sha`` is absent entirely.
    Never raises for a merely-missing or merely-unrelated SHA.
    """
    if sha == pr_head_sha:
        return "ok"
    verify = git_ops._run_git(mirror_repo, ["rev-parse", "--verify", f"{sha}^{{commit}}"], retries=0)
    if verify.returncode != 0:
        return "head_unreachable"
    anc = git_ops._run_git(mirror_repo, ["merge-base", "--is-ancestor", sha, pr_head_sha], retries=0)
    anc = git_ops._run_git(mirror_repo, ["merge-base", "--is-ancestor", sha, pr_head_sha], retries=0)
    return "ok" if anc.returncode == 0 else "head_not_on_pr"


def resolve_original_base(mirror_repo: Path, base_tip_ref: str, head_sha: str) -> str | None:
    """The merge-base of the base tip and head, or None when none exists.

    Soft-failure: returns ``None`` for a documented no-merge-base case (a real
    broken git invocation propagates as ``GitError``).
    """
    proc = git_ops._run_git(mirror_repo, ["merge-base", base_tip_ref, head_sha], retries=0)
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def resolve_trees(mirror_repo: Path, base_sha: str, head_sha: str):
    """Peel ``^{tree}`` for both commits, or return ``"missing_object"``."""
    def _tree(sha: str) -> str | None:
        proc = git_ops._run_git(mirror_repo, ["rev-parse", "--verify", f"{sha}^{{tree}}"], retries=0)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    bt = _tree(base_sha)
    if bt is None:
        return "missing_object"
    ht = _tree(head_sha)
    if ht is None:
        return "missing_object"
    return (bt, ht)
