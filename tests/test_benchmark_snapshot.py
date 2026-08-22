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


# ---------------------------------------------------------------------------
# Task 3: merge-base resolution + tree reachability
# ---------------------------------------------------------------------------


def test_resolve_base_and_trees(tmp_path):
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    base = sn.resolve_original_base(m, "refs/heads/base_tip", _SHA_HEAD)
    assert base == _SHA_BASE2
    bt, ht = sn.resolve_trees(m, base, _SHA_HEAD)
    assert bt == _seed_base_tree() and ht == _seed_head_tree()
    assert sn.resolve_trees(m, base, "0" * 40) == "missing_object"


# ---------------------------------------------------------------------------
# Task 4: degenerate-case detection + canonical diff sha
# ---------------------------------------------------------------------------


def test_degenerate_equal_trees_and_canonical_diff(tmp_path):
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    assert sn.degenerate(m, _seed_base_tree(), _seed_base_tree()) == "equal_trees"
    assert sn.degenerate(m, _seed_base_tree(), _seed_head_tree()) is None   # real change
    d = sn.canonical_diff_sha256(m, _SHA_BASE2, _SHA_HEAD)
    assert re.fullmatch(r"[0-9a-f]{64}", d)


# ---------------------------------------------------------------------------
# Task 5: synthetic commits + deterministic minimal bundle
# ---------------------------------------------------------------------------


def test_bundle_two_refs_deterministic(tmp_path):
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    bundle = tmp_path / "snapshots" / "pr-000001-aaaaaaaaaaaa.bundle"
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, bundle, case_id="pr-000001-aaaaaaaaaaaa")
    assert sn.bundle_heads(bundle) == {"refs/heads/base", "refs/heads/head"}
    base_commit = sn.rev_parse(m, "refs/heads/base")
    head_commit = sn.rev_parse(m, "refs/heads/head")
    assert sn.rev_parse(m, f"{base_commit}^{{tree}}") == _seed_base_tree()
    assert sn.rev_parse(m, f"{head_commit}^{{tree}}") == _seed_head_tree()
    assert sn.rev_parse(m, f"{head_commit}^") == base_commit           # single parent
    # determinism: rebuild and compare bytes
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, bundle, case_id="pr-000001-aaaaaaaaaaaa")
    assert sn.sha256_of(bundle) == sn.sha256_of(bundle)


# ---------------------------------------------------------------------------
# Task 6: offline-clone validation
# ---------------------------------------------------------------------------


def test_offline_clone_validates(tmp_path):
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    bundle = tmp_path / "snapshots" / "pr-000001-aaaaaaaaaaaa.bundle"
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, bundle, case_id="pr-000001-aaaaaaaaaaaa")
    diff_sha = sn.canonical_diff_sha256(m, _SHA_BASE2, _SHA_HEAD)
    sn.validate_offline_clone(bundle, _seed_base_tree(), _seed_head_tree(), diff_sha,
                              workdir=tmp_path)
    bad = tmp_path / "snapshots" / "bad.bundle"
    bad.write_bytes(bundle.read_bytes()[: len(bundle.read_bytes()) // 2])
    with pytest.raises(git_ops.GitError):
        sn.validate_offline_clone(bad, _seed_base_tree(), _seed_head_tree(), diff_sha,
                                  workdir=tmp_path)


# ---------------------------------------------------------------------------
# Task 7: freeze_one ready / unreplayable reason matrix
# ---------------------------------------------------------------------------


def test_freeze_one_ready_and_reasons(tmp_path):
    from daydream.benchmark import snapshot as sn
    from daydream.benchmark.schema import case_id_for

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    ready = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE2, head_sha=_SHA_HEAD,
                          policy="final_pr_head", requested_head="final", origin_url=origin)
    assert ready["status"] == "ready"
    assert ready["original_base_sha"] == _SHA_BASE2 and ready["original_head_sha"] == _SHA_HEAD
    assert ready["base_tree_sha"] == _seed_base_tree() and ready["head_tree_sha"] == _seed_head_tree()
    assert re.fullmatch(r"[0-9a-f]{64}", ready["diff_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", ready["bundle_sha256"])
    expect_rel = f"snapshots/{case_id_for(1, _SHA_HEAD)}.bundle"
    assert ready["bundle_file"] == expect_rel
    assert (tmp_path / expect_rel).exists()
    # head_not_on_pr: a base3 head reachable elsewhere is rejected
    ur = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE3, head_sha=_SHA_BASE3,
                       policy="explicit_head", requested_head=_SHA_BASE3, origin_url=origin)
    assert ur["status"] == "unreplayable" and ur["error"]["reason"] == "head_not_on_pr"
    assert ur["bundle_file"] is None and ur["base_tree_sha"] is None
    # head_unreachable: a sha absent from the mirror
    ur2 = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE2, head_sha="0" * 40,
                        policy="explicit_head", requested_head="0" * 40, origin_url=origin)
    assert ur2["status"] == "unreplayable" and ur2["error"]["reason"] == "head_unreachable"
    assert ur2["bundle_file"] is None


# ---------------------------------------------------------------------------
# Task 10: crash injection at case/bundle/manifest transaction boundaries
# ---------------------------------------------------------------------------


def test_freeze_crash_recovers_whole_before_or_after(tmp_path):
    """A crash at any freeze {bundle, case, manifest} boundary heals whole.

    Mirrors the import-crash transaction test, substituting the snapshot bundle
    for one staged file: ``journal``/``data`` restore the whole before-state,
    ``manifest`` keeps the complete after-state, and ``transactions/`` is left
    empty after recovery.
    """
    from daydream.benchmark import storage
    from daydream.benchmark.storage import recover_startup

    for boundary in ("journal", "data", "manifest"):
        case_dir = tmp_path / "cases"
        snap_dir = tmp_path / "snapshots"
        for d in (case_dir, snap_dir):
            d.mkdir(parents=True, exist_ok=True)
        (case_dir / "pr-000001-aaaaaaaaaaaa.yaml").write_text("case-before")
        (snap_dir / "pr-000001-aaaaaaaaaaaa.bundle").write_bytes(b"bundle-before")
        (tmp_path / "benchmark.yaml").write_text("ledger-before")
        with storage.Transaction(tmp_path, op_id=f"freeze-{boundary}", kind="freeze") as tx:
            tx.stage("snapshots/pr-000001-aaaaaaaaaaaa.bundle", b"bundle-after")
            tx.stage("cases/pr-000001-aaaaaaaaaaaa.yaml", b"case-after")
            tx.stage("benchmark.yaml", b"ledger-after")
            tx.inject_crash(boundary)
        recover_startup(tmp_path)
        if boundary in ("journal", "data"):
            assert (snap_dir / "pr-000001-aaaaaaaaaaaa.bundle").read_bytes() == b"bundle-before"
            assert (case_dir / "pr-000001-aaaaaaaaaaaa.yaml").read_text() == "case-before"
            assert (tmp_path / "benchmark.yaml").read_text() == "ledger-before"
        else:  # manifest (complete journal kept)
            assert (snap_dir / "pr-000001-aaaaaaaaaaaa.bundle").read_bytes() == b"bundle-after"
            assert (case_dir / "pr-000001-aaaaaaaaaaaa.yaml").read_text() == "case-after"
            assert (tmp_path / "benchmark.yaml").read_text() == "ledger-after"
        assert not (tmp_path / "transactions").exists() or not list(
            (tmp_path / "transactions").iterdir()
        )
