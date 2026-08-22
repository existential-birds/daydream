"""Tests for the frozen benchmark snapshot bundles (issue 3).

All snapshot tests drive real git objects (no fake git): the seed helpers build
a deterministic minimal origin whose base/head SHAs are module literals matching
exactly what the deterministic commits produce, so every freeze path runs
against the real object store. Only the import-gating tests (in
``test_benchmark_import_prs``) go through the ``fake_gh`` network boundary.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from daydream import git_ops


# ---------------------------------------------------------------------------
# real-git seed helpers (deterministic commit SHAs)
# ---------------------------------------------------------------------------

_SEED_ENV = {
    "GIT_AUTHOR_NAME": "Tester",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Tester",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def _git(repo: Path, *args: str, env: dict[str, str] | None = None, check: bool = True) -> str:
    proc_env = {**os.environ, **env} if env is not None else os.environ.copy()
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, env=proc_env, check=check
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "commit", "-m", message, env=_SEED_ENV)
    return _git(repo, "rev-parse", "HEAD")


def _write(repo: Path, name: str, content: str | bytes) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    _git(repo, "add", name)


def _seed_origin(tmp_path: Path) -> Path:
    """Bare origin: main (base1->base2->base3) + "refs/pull/1/head" off base2."""
    repo = tmp_path / "seed_wt"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _write(repo, "readme.txt", "base1\n")
    _commit(repo, "base1")
    _write(repo, "base.py", "BASE = 2\n")
    base2_sha = _commit(repo, "base2")
    _write(repo, "beyond.py", "BEYOND = 3\n")
    _commit(repo, "base3")
    _git(repo, "checkout", "--detach", base2_sha)
    repo.joinpath("base.py").write_text("BASE = 20\n")
    _git(repo, "add", "base.py")
    _write(repo, "feature.py", "FEATURE = 1\n")
    head_sha = _commit(repo, "feature")

    bare = tmp_path / "origin.git"
    bare.mkdir(parents=True, exist_ok=True)
    _git(bare, "init", "--bare")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "origin", "main:main")
    _git(repo, "push", "origin", f"{head_sha}:refs/pull/1/head", check=False)
    return bare


# Deterministic SHAs/trees produced by ``_seed_origin`` (verified at seed build).
_SHA_BASE1 = 'cae67fc3eb4c5d3dd3353ca7fb41f909837bf0a2'
_SHA_BASE1_TREE = '2cd99bd20f7b3bac54014e20db1831d64b2c4fc9'
_SHA_BASE2 = 'd35f2cbffc81b6292f67cf891ac1c4256fe948a4'
_SHA_BASE2_TREE = 'a54e8fefe4dd3ffe592efe5fc64eb32f9eb7dbd4'
_SHA_BASE3 = '7a447892308f03f6861099ad03b5895397591f02'
_SHA_BASE3_TREE = '7bf0703afd952cd48a9a9e231cd6fee8e09cc5d0'
_SHA_HEAD = 'd9a75fd29107db73ef6cb08f877e644381c31f25'
_SHA_HEAD_TREE = '100c61d903cabfd705776af46193bc55d494940d'


def _seed_base_tree() -> str:
    return 'a54e8fefe4dd3ffe592efe5fc64eb32f9eb7dbd4'


def _seed_head_tree() -> str:
    return '100c61d903cabfd705776af46193bc55d494940d'


# ---------------------------------------------------------------------------
# Task 1: shared bare-mirror establishment + PR ref fetch
# ---------------------------------------------------------------------------


def test_ensure_mirror_and_fetch_pr_head(tmp_path):
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    mirror = tmp_path / "cache" / "repository.git"
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    assert mirror.is_dir()
    sn.fetch_pr_refs(tmp_path, "o/r", pr_number=1, base_tip=_SHA_BASE2,
                     explicit_shas=[], origin_url=origin)
    assert sn.rev_parse(mirror, "refs/pull/1/head") == _SHA_HEAD
    assert sn.rev_parse(mirror, "refs/heads/base_tip") == _SHA_BASE2
    # second call is idempotent
    sn.fetch_pr_refs(tmp_path, "o/r", pr_number=1, base_tip=_SHA_BASE2,
                     explicit_shas=[], origin_url=origin)
    assert sn.rev_parse(mirror, "refs/pull/1/head") == _SHA_HEAD


# ---------------------------------------------------------------------------
# Task 2: ancestor-of-PR-head enforcement
# ---------------------------------------------------------------------------


def test_ancestor_of_pr_head_enforced(tmp_path):
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)   # base3 reachable via main, NOT an ancestor of the PR head
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE3,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    pr_head = sn.rev_parse(m, "refs/pull/1/head")
    assert sn.head_reachability(m, _SHA_BASE2, pr_head) == "ok"     # ancestor
    assert sn.head_reachability(m, _SHA_HEAD, pr_head) == "ok"      # equal
    assert sn.head_reachability(m, _SHA_BASE3, pr_head) == "head_not_on_pr"
    assert sn.head_reachability(m, "0" * 40, pr_head) == "head_unreachable"
