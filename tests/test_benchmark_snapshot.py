"""Tests for the frozen benchmark snapshot bundles (issue 3).

All snapshot tests drive real git objects (no fake git): the seed helpers build
a deterministic minimal origin whose base/head SHAs are module literals matching
exactly what the deterministic commits produce, so every freeze path runs
against the real object store. Only the import-gating tests (in
``test_benchmark_import_prs``) go through the ``fake_gh`` network boundary.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.benchmark import snapshot, snapshot as sn
from daydream.benchmark.schema import case_id_for
from daydream.git_ops import GitError, process as git_process
from tests.harness.git_helpers import git as _git, seed_pr_origin, seeded_commit, write_and_stage


def _publish_origin(repo: Path, bare: Path, *refspecs: str) -> None:
    """Init *bare*, add it as *repo*'s origin, and push each *refspec*.

    A refspec prefixed with ``!`` is pushed with ``check=False`` -- the
    non-fast-forward PR-head refs a plain push rejects; every other refspec is
    a hard-failing push.
    """
    bare.mkdir(parents=True, exist_ok=True)
    _git(bare, "init", "--bare")
    _git(repo, "remote", "add", "origin", str(bare))
    for refspec in refspecs:
        _git(repo, "push", "origin", refspec.removeprefix("!"), check=not refspec.startswith("!"))


def _primed_mirror(
    tmp_path: Path, origin: str, *, base_tip: str, explicit_shas: list[str] | tuple[str, ...] = (), pr_number: int = 1,
) -> Path:
    """Bring up the shared mirror and prime it with the base tip and PR head(s)."""

    sn.ensure_mirror(tmp_path)
    sn.fetch_base_tip(tmp_path, "o/r", base_tip, origin)
    sn.fetch_head_refs(tmp_path, "o/r", pr_number, explicit_shas, origin)
    return sn.mirror(tmp_path)


def _seed_origin(tmp_path: Path) -> str:
    """Bare origin: main (base1->base2->base3) + "refs/pull/1/head" off base2."""
    bare, _, _ = seed_pr_origin(tmp_path, repo_name="seed_wt", bare_name="origin.git", number=1)
    return bare


# Deterministic SHAs/trees produced by ``_seed_origin`` (verified at seed build).
_SHA_BASE2 = 'd35f2cbffc81b6292f67cf891ac1c4256fe948a4'
_SHA_BASE2_TREE = 'a54e8fefe4dd3ffe592efe5fc64eb32f9eb7dbd4'
_SHA_BASE3 = '7a447892308f03f6861099ad03b5895397591f02'
_SHA_HEAD = 'd9a75fd29107db73ef6cb08f877e644381c31f25'
_SHA_HEAD_TREE = '100c61d903cabfd705776af46193bc55d494940d'


@pytest.fixture
def origin(tmp_path: Path) -> str:
    """A seeded origin repository under tmp_path."""
    return _seed_origin(tmp_path)


def _seed_anchor_origin(tmp_path: Path, *, pr: int = 1) -> tuple[str, str, str]:
    """Bare origin with no renames: main (authoring->edit) + ``refs/pull/N/head``.

    Commit 1 authors ``a.py``; commit 2 edits it in place (a plain ``M`` diff,
    no rename records). Returns ``(bare, authoring_sha, head_sha)``.
    """
    repo = tmp_path / "anchor_wt"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    write_and_stage(repo, "a.py", "def a() -> int:\n    return 1\n")
    authoring_sha = seeded_commit(repo, "author a.py")
    repo.joinpath("a.py").write_text("def a() -> int:\n    return 2\n")
    _git(repo, "add", "a.py")
    head_sha = seeded_commit(repo, "edit a.py")

    bare = tmp_path / "anchor_origin.git"
    _publish_origin(repo, bare, "main:main", f"!{head_sha}:refs/pull/{pr}/head")
    return str(bare), authoring_sha, head_sha


def _seed_double_rename_origin(tmp_path: Path, *, pr: int = 1) -> tuple[str, str, str]:
    """Bare origin with two renames in one diff: main (authoring->rename) + PR ref.

    Commit 1 authors ``old1.py`` and ``old2.py``; commit 2 renames both (via
    ``git mv``, no content change) so ``git diff -M`` emits two ``R100`` rows.
    Returns ``(bare, authoring_sha, head_sha)``.
    """
    repo = tmp_path / "dbl_wt"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    write_and_stage(repo, "old1.py", "def old1() -> int:\n    return 1\n")
    write_and_stage(repo, "old2.py", "def old2() -> int:\n    return 2\n")
    authoring_sha = seeded_commit(repo, "author old1 old2")
    _git(repo, "mv", "old1.py", "new1.py")
    _git(repo, "mv", "old2.py", "new2.py")
    head_sha = seeded_commit(repo, "rename both")

    bare = tmp_path / "dbl_origin.git"
    _publish_origin(repo, bare, "main:main", f"!{head_sha}:refs/pull/{pr}/head")
    return str(bare), authoring_sha, head_sha


def test_derive_authoring_path_fails_closed(tmp_path: Path) -> None:
    """Derivation never guesses: a missing authoring commit is
    ``history-unavailable``; a path that is neither in the authoring tree nor an
    exact rename dest fails ``path-unavailable`` -- even when the diff does
    contain rename candidates (the two-R-row diff must not be resolved by
    picking among the candidates)."""

    anchor_origin, anchor_auth, anchor_head = _seed_anchor_origin(tmp_path, pr=1)
    snapshot.fetch_head_refs(tmp_path, "o/r", 1, explicit_shas=[anchor_head], origin_url=anchor_origin)
    dbl_origin, dbl_auth, dbl_head = _seed_double_rename_origin(tmp_path, pr=3)
    snapshot.fetch_head_refs(tmp_path, "o/r", 3, explicit_shas=[dbl_head], origin_url=dbl_origin)
    m = snapshot.mirror(tmp_path)

    sha_absent = ("0" * 40, "a.py", anchor_head)  # (a) commit absent
    sha_present = (anchor_auth, "missing.py", anchor_head)  # (b) no rename, no path
    sha_ambiguous = (dbl_auth, "missing.py", dbl_head)  # (c) two R-candidates
    for args, expected_reason in [
        (sha_absent, "history-unavailable"), (sha_present, "path-unavailable"), (sha_ambiguous, "path-unavailable"),
    ]:
        with pytest.raises(snapshot.AnchorDerivationError) as exc:
            snapshot.derive_authoring_path(m, *args)
        assert expected_reason in str(exc.value)

def test_offline_clone_fidelity_rejects_tampering(tmp_path: Path, origin: str) -> None:
    """Acceptance (b/c) at unit level: the offline-clone fidelity contract
    rejects every structurally-distinct tampered bundle shape (extra ref,
    extra reachable commit, wrong parent, wrong tree) while a valid bundle
    passes all probes."""

    m = _primed_mirror(tmp_path, origin, base_tip=_SHA_BASE2, explicit_shas=[_SHA_HEAD])
    base_tree, head_tree = _SHA_BASE2_TREE, _SHA_HEAD_TREE

    valid = tmp_path / "snapshots" / "pr-000001-aaaaaaaaaaaa.bundle"
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, valid)
    sn.validate_offline_clone(valid, base_tree, head_tree, workdir=tmp_path)

    # Synthetic commits with the same pinned identity as build_bundle's
    # (deterministic: identical tree+message+env => identical sha).
    synth_env = sn._synthetic_env()
    base_commit = _git(m, "commit-tree", base_tree, "-m", "snapshot base", env=synth_env)
    head_commit = _git(m, "commit-tree", head_tree, "-p", base_commit, "-m", "snapshot head", env=synth_env)

    # Advertised tags/custom refs may disappear from cloned remote refs. Point
    # them at the existing base so extra commit history cannot mask this guard.
    extra_ref_bundle = tmp_path / "snapshots" / "extra-ref.bundle"
    for extra_ref in ("refs/heads/extra", "refs/tags/extra", "refs/custom/extra"):
        _git(m, "update-ref", extra_ref, base_commit)
        _git(m, "bundle", "create", str(extra_ref_bundle), "refs/heads/base", "refs/heads/head", extra_ref)
        with pytest.raises(git_ops.GitError, match="refs"):
            sn.validate_offline_clone(extra_ref_bundle, base_tree, head_tree, workdir=tmp_path)
        _git(m, "update-ref", "-d", extra_ref)

    # (c) extra reachable commit (no extra ref): head parented on an extra
    # commit parented on base, violating the exact root/child topology.
    extra_commit_bundle = tmp_path / "snapshots" / "extra-commit.bundle"
    extra_commit = _git(m, "commit-tree", base_tree, "-p", base_commit, "-m", "extra", env=synth_env)
    head_tampered = _git(m, "commit-tree", head_tree, "-p", extra_commit, "-m", "snapshot head", env=synth_env)
    _git(m, "update-ref", "refs/heads/head", head_tampered)
    _git(m, "bundle", "create", str(extra_commit_bundle), "refs/heads/base", "refs/heads/head")
    with pytest.raises(git_ops.GitError, match="exactly two commits"):
        sn.validate_offline_clone(extra_commit_bundle, base_tree, head_tree, workdir=tmp_path)

    # (c) wrong parent: head with NO parent violates the single-child topology.
    wrong_parent_bundle = tmp_path / "snapshots" / "wrong-parent.bundle"
    parentless_head = _git(m, "commit-tree", head_tree, "-m", "snapshot head", env=synth_env)
    _git(m, "update-ref", "refs/heads/head", parentless_head)
    _git(m, "bundle", "create", str(wrong_parent_bundle), "refs/heads/base", "refs/heads/head")
    with pytest.raises(git_ops.GitError):
        sn.validate_offline_clone(wrong_parent_bundle, base_tree, head_tree, workdir=tmp_path)

    # (c) wrong tree: head's tree is the base tree -> tree mismatch.
    wrong_tree_bundle = tmp_path / "snapshots" / "wrong-tree.bundle"
    wrong_tree_head = _git(m, "commit-tree", base_tree, "-p", base_commit, "-m", "snapshot head", env=synth_env)
    _git(m, "update-ref", "refs/heads/head", wrong_tree_head)
    _git(m, "bundle", "create", str(wrong_tree_bundle), "refs/heads/base", "refs/heads/head")
    with pytest.raises(git_ops.GitError, match="tree mismatch"):
        sn.validate_offline_clone(wrong_tree_bundle, base_tree, head_tree, workdir=tmp_path)

    # Restore the mirror refs so the mirror is reusable for later freezes.
    _git(m, "update-ref", "refs/heads/base", base_commit)
    _git(m, "update-ref", "refs/heads/head", head_commit)

def test_changed_paths_rejects_malformed_nul_records_and_degenerate_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert sn.degenerate(_SHA_BASE2_TREE, _SHA_BASE2_TREE) == "equal_trees"
    assert sn.degenerate(_SHA_BASE2_TREE, _SHA_HEAD_TREE) is None   # real change
    for stdout in (b"M\x00a.py", b"M\x00\x00", b"R100\x00old.py\x00", b"R101\x00old.py\x00new.py\x00",
                   b"R100\x00old.py\x00new.py\x00junk\x00"):
        monkeypatch.setattr(git_process, "_run_git",
                            lambda *args, out=stdout, **kwargs: subprocess.CompletedProcess(args, 0, out, b""))
        with pytest.raises(GitError, match="name-status"):
            sn.changed_paths(tmp_path, "a" * 40, "b" * 40)


def test_freeze_one_ready_and_reasons(tmp_path: Path, origin: str) -> None:
    _primed_mirror(tmp_path, origin, base_tip=_SHA_BASE2, explicit_shas=[_SHA_HEAD])
    ready, bundle = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE2, head_sha=_SHA_HEAD,
                           policy="final_pr_head", requested_head="final", pr_changed_files=set(), origin_url=origin)
    assert ready["status"] == "ready"
    assert ready["original_base_sha"] == _SHA_BASE2 and ready["original_head_sha"] == _SHA_HEAD
    assert ready["requested_base_sha"] == _SHA_BASE2
    assert ready["base_tree_sha"] == _SHA_BASE2_TREE and ready["head_tree_sha"] == _SHA_HEAD_TREE
    assert "diff_sha256" not in ready
    assert re.fullmatch(r"[0-9a-f]{64}", ready["bundle_sha256"])
    expect_rel = f"snapshots/{case_id_for(1, _SHA_HEAD)}.bundle"
    assert ready["bundle_file"] == expect_rel
    # freeze returns the bundle bytes to stage through the crash-consistent
    # transaction; it never writes the final snapshots/<case>.bundle itself.
    assert isinstance(bundle, bytes) and bundle.startswith(b"# v2 git bundle")
    assert not (tmp_path / expect_rel).exists()
    # head_not_on_pr: a base3 head reachable elsewhere is rejected
    ur, bundle = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE3, head_sha=_SHA_BASE3,
                               policy="explicit_head", requested_head=_SHA_BASE3,
                               pr_changed_files=set(), origin_url=origin)
    assert ur["status"] == "unreplayable" and ur["error"]["reason"] == "head_not_on_pr"
    assert bundle is None
    assert ur["bundle_file"] is None and ur["base_tree_sha"] is None
    m = sn.mirror(tmp_path)
    pr_head = sn.rev_parse(m, "refs/pull/1/head")
    assert sn.head_reachability(m, _SHA_BASE2, pr_head) == "ok"     # ancestor
    assert sn.head_reachability(m, _SHA_HEAD, pr_head) == "ok"      # equal
    assert sn.head_reachability(m, _SHA_BASE3, pr_head) == "head_not_on_pr"
    assert sn.head_reachability(m, "0" * 40, pr_head) == "head_unreachable"
    # head_unreachable: a sha absent from the mirror
    ur2, bundle2 = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE2, head_sha="0" * 40,
                                 policy="explicit_head", requested_head="0" * 40,
                                 pr_changed_files=set(), origin_url=origin)
    assert ur2["status"] == "unreplayable" and ur2["error"]["reason"] == "head_unreachable"
    assert bundle2 is None
    assert ur2["bundle_file"] is None

# ---------------------------------------------------------------------------


def _seed_delta_origin(tmp_path: Path) -> tuple[str, str, str, dict[str, str]]:
    """Branch edit, rename, delete, binary, and unrelated heads from one shared base.

    Return the origin, base SHA, unrelated sibling SHA, and four scenario heads.
    """
    repo = tmp_path / "delta_wt"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    write_and_stage(repo, "readme.txt", "base1\n")
    seeded_commit(repo, "base1")
    write_and_stage(repo, "base.py", "BASE = 2\n")
    write_and_stage(repo, "wide.py", "".join(f"line{i}\n" for i in range(1, 11)))
    write_and_stage(repo, "renamed.txt", "r\n")
    write_and_stage(repo, "deleted.txt", "d\n")
    base2_sha = seeded_commit(repo, "base2")

    _git(repo, "checkout", "--detach", base2_sha)
    content = (repo / "wide.py").read_text().replace("line10\n", "line10 edited\n")
    repo.joinpath("wide.py").write_text(content)
    _git(repo, "add", "wide.py")
    edit_head = seeded_commit(repo, "edit wide.py line 10")

    _git(repo, "checkout", "--detach", base2_sha)
    _git(repo, "mv", "renamed.txt", "renamed2.txt")
    rename_head = seeded_commit(repo, "rename renamed.txt")

    _git(repo, "checkout", "--detach", base2_sha)
    _git(repo, "rm", "deleted.txt")
    delete_head = seeded_commit(repo, "delete deleted.txt")

    _git(repo, "checkout", "--detach", base2_sha)
    (repo / "bin.dat").write_bytes(b"\x00\x01\x02binary")
    _git(repo, "add", "bin.dat")
    binary_head = seeded_commit(repo, "add bin.dat")

    _git(repo, "checkout", "--detach", base2_sha)
    repo.joinpath("orphan.txt").write_text("o\n")
    _git(repo, "add", "orphan.txt")
    orphan_sha = seeded_commit(repo, "orphan")

    bare = tmp_path / "delta_origin.git"
    _publish_origin(repo, bare, "main:main",
        *(f"{sha}:refs/heads/{name}" for name, sha in (
            ("edit", edit_head), ("rename", rename_head), ("delete", delete_head), ("binary", binary_head),
            ("orphan", orphan_sha),
        )),
    )
    heads = {"edit": edit_head, "rename": rename_head, "delete": delete_head, "binary": binary_head}
    return str(bare), base2_sha, orphan_sha, heads


def _anchor(path: str | None, start: int | None, end: int | None,
            commit: str, status: str = "derived") -> dict[str, Any]:
    return {"version": 1, "status": status, "commit_id": commit, "path": path, "start_line": start, "end_line": end}


def _delta_mirror(tmp_path: Path) -> tuple[Path, str, str, dict[str, str]]:
    origin, base, orphan, heads = _seed_delta_origin(tmp_path)
    m = sn.ensure_mirror(tmp_path)
    sn._git_fetch(m, origin, [f"{sha}:refs/heads/fact-{i}" for i, sha in enumerate((base, orphan, *heads.values()))])
    return sn.mirror(tmp_path), base, orphan, heads


def test_anchor_delta_intersecting_vs_elsewhere_rename_delete_binary_locationless(tmp_path: Path,) -> None:
    m, base, orphan, heads = _delta_mirror(tmp_path)
    cache: dict[Any, Any] = {}
    assert sn.anchor_delta(m, base, heads["edit"], _anchor("wide.py", 10, 10, base), diff_cache=cache) == "changed"
    # The cached whole-tree classification still re-checks each anchor's own range.
    assert sn.anchor_delta(m, base, heads["edit"], _anchor("wide.py", 1, 5, base), diff_cache=cache) == "unchanged"
    assert sn.anchor_delta(m, base, "0" * 40, _anchor("wide.py", 1, 5, base)) == "unavailable"
    assert sn.commit_relation(m, heads["edit"], "0" * 40) == "unavailable"
    assert sn.anchor_delta(m, base, heads["rename"], _anchor("renamed.txt", 1, 1, base)) == "renamed"
    assert sn.anchor_delta(m, base, heads["delete"], _anchor("deleted.txt", 1, 1, base)) == "deleted"
    assert sn.anchor_delta(m, base, heads["binary"], _anchor("bin.dat", 1, 1, base)) == "binary"
    assert sn.anchor_delta(m, base, heads["edit"], _anchor(None, None, None, base)) == "locationless"
    assert sn.anchor_delta(m, base, heads["edit"],
                           _anchor("wide.py", 1, 1, base, status="path-unavailable")) == "locationless"
