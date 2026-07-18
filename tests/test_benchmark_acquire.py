"""Tests for blobless PR acquisition (daydream/benchmark/acquire.py).

Uses REAL git against a local bare "upstream" repo with a base commit on
main and a head commit published under ``refs/pull/7/head`` — a local path
is a valid git remote, so there is NO network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

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


@dataclass
class _BranchedUpstream:
    """A bare upstream whose PR branch diverged before the base branch moved on.

    ``main`` is ``m1 -> m2``; the PR branch forks at ``m1`` and runs
    ``c0 -> c1 -> c2``, with ``refs/pull/7/head`` at ``c2``. ``c1`` stands in for
    a bot review snapshot taken mid-PR: it is neither the PR head nor a child of
    the merge-base, so a first-parent base (``c1^`` == ``c0``) and the tip of the
    base branch (``m2``) are both wrong; only the 3-dot merge-base ``m1`` is right.
    """

    url: str
    m1: str
    m2: str
    c0: str
    c1: str
    c2: str


def _make_branched_upstream(tmp_path: Path) -> _BranchedUpstream:
    work = tmp_path / "branched-work"
    work.mkdir(parents=True, exist_ok=True)
    _git(work, "init", "-b", "main")
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Tester")

    def _commit(name: str, content: str) -> str:
        (work / name).write_text(content)
        _git(work, "add", name)
        _git(work, "commit", "-m", f"add {name}")
        return _git(work, "rev-parse", "HEAD")

    m1 = _commit("m1.txt", "m1\n")
    _git(work, "checkout", "-b", "pr")
    c0 = _commit("c0.txt", "c0\n")
    c1 = _commit("c1.txt", "c1\n")
    c2 = _commit("c2.txt", "c2\n")
    _git(work, "checkout", "main")
    m2 = _commit("m2.txt", "m2\n")

    _git(work, "update-ref", "refs/pull/7/head", c2)
    # An orphan branch with no common ancestor, for the no-merge-base case.
    _git(work, "checkout", "--orphan", "unrelated")
    _git(work, "rm", "-rf", ".")
    _commit("u.txt", "unrelated\n")
    _git(work, "checkout", "main")

    bare = tmp_path / "branched.git"
    _git(work, "clone", "--bare", str(work), str(bare))
    _git(bare, "update-ref", "refs/pull/7/head", c2)

    return _BranchedUpstream(url=str(bare), m1=m1, m2=m2, c0=c0, c1=c1, c2=c2)


def test_acquire_checks_out_pr_head_and_keeps_base_resolvable(tmp_path):
    up = _make_upstream_with_pr(tmp_path)
    acquired = acquire_checkout(up.url, 7, up.head_sha, base_sha=up.base_sha, cache_dir=tmp_path / "cache")
    assert _git(acquired.path, "rev-parse", "HEAD") == up.head_sha
    assert acquired.base_sha == up.base_sha
    assert git_ops.ref_exists(acquired.path, up.base_sha)  # usable as daydream --base


def test_second_acquire_reuses_clone_cache(tmp_path):
    up = _make_upstream_with_pr(tmp_path)
    a = acquire_checkout(up.url, 7, up.head_sha, base_sha=up.base_sha, cache_dir=tmp_path / "c")
    b = acquire_checkout(up.url, 7, up.head_sha, base_sha=up.base_sha, cache_dir=tmp_path / "c")
    assert a.path == b.path


def test_acquire_derives_merge_base_for_bot_commit_not_pr_head(tmp_path):
    up = _make_branched_upstream(tmp_path)
    acquired = acquire_checkout(up.url, 7, up.c1, base_ref="main", cache_dir=tmp_path / "cache")

    # The 3-dot base GitHub's compare view (and the bot) saw — not the base
    # branch tip, and not the snapshot's first parent.
    assert acquired.base_sha == up.m1
    assert acquired.base_sha != up.m2
    assert acquired.base_sha != up.c0
    assert _git(acquired.path, "rev-parse", f"{up.c1}^") == up.c0  # first-parent really differs
    # HEAD sits on the bot's snapshot, not the PR head.
    assert _git(acquired.path, "rev-parse", "HEAD") == up.c1
    assert _git(acquired.path, "rev-parse", "HEAD") != up.c2


def test_acquire_raises_giterror_when_no_merge_base(tmp_path):
    up = _make_branched_upstream(tmp_path)
    with pytest.raises(git_ops.GitError, match="no merge-base"):
        acquire_checkout(up.url, 7, up.c1, base_ref="unrelated", cache_dir=tmp_path / "cache")


def test_acquire_rejects_both_or_neither_base_arguments(tmp_path):
    up = _make_branched_upstream(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        acquire_checkout(up.url, 7, up.c1, base_sha=up.m1, base_ref="main", cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="exactly one"):
        acquire_checkout(up.url, 7, up.c1, cache_dir=tmp_path / "cache")
