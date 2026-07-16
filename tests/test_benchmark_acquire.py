"""Tests for blobless PR acquisition (daydream/benchmark/acquire.py).

Uses REAL git against a local bare "upstream" repo with a base commit on
main and a head commit published under ``refs/pull/7/head`` — a local path
is a valid git remote, so there is NO network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from daydream import git_ops
from daydream.benchmark.acquire import acquire_checkout
from tests.harness.git_helpers import git as _git


@dataclass
class _Upstream:
    """A local bare upstream repo carrying a base commit and a PR head ref."""

    url: str
    base_sha: str
    head_sha: str


def _make_upstream_with_pr(tmp_path: Path) -> _Upstream:
    """Build a real local bare repo whose ``clone_url`` is a filesystem path.

    The repo has a base commit on ``main`` and a separate head commit
    published under ``refs/pull/7/head`` (mirroring GitHub's PR ref layout).
    """
    work = tmp_path / "upstream-work"
    work.mkdir(parents=True, exist_ok=True)
    _git(work, "init", "-b", "main")
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Tester")

    (work / "base.txt").write_text("base\n")
    _git(work, "add", "base.txt")
    _git(work, "commit", "-m", "base on main")
    base_sha = _git(work, "rev-parse", "HEAD")

    # A head commit that is NOT on main — only reachable via the PR ref.
    (work / "feature.txt").write_text("feature\n")
    _git(work, "add", "feature.txt")
    _git(work, "commit", "-m", "pr head")
    head_sha = _git(work, "rev-parse", "HEAD")

    # Reset main back to base so head lives solely under refs/pull/7/head.
    _git(work, "update-ref", "refs/pull/7/head", head_sha)
    _git(work, "update-ref", "refs/heads/main", base_sha)
    _git(work, "checkout", "main")

    bare = tmp_path / "upstream.git"
    _git(work, "clone", "--bare", str(work), str(bare))
    # The plain bare clone does not copy refs/pull/*; publish it explicitly.
    _git(bare, "update-ref", "refs/pull/7/head", head_sha)

    return _Upstream(url=str(bare), base_sha=base_sha, head_sha=head_sha)


def test_acquire_checks_out_pr_head_and_keeps_base_resolvable(tmp_path):
    up = _make_upstream_with_pr(tmp_path)
    checkout = acquire_checkout(up.url, 7, up.base_sha, up.head_sha, cache_dir=tmp_path / "cache")
    assert _git(checkout, "rev-parse", "HEAD") == up.head_sha
    assert git_ops.ref_exists(checkout, up.base_sha)  # usable as daydream --base


def test_second_acquire_reuses_clone_cache(tmp_path):
    up = _make_upstream_with_pr(tmp_path)
    a = acquire_checkout(up.url, 7, up.base_sha, up.head_sha, cache_dir=tmp_path / "c")
    b = acquire_checkout(up.url, 7, up.base_sha, up.head_sha, cache_dir=tmp_path / "c")
    assert a == b
