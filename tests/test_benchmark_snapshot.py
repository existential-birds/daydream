"""Tests for the frozen benchmark snapshot bundles (issue 3).

All snapshot tests drive real git objects (no fake git): the seed helpers build
a deterministic minimal origin whose base/head SHAs are module literals matching
exactly what the deterministic commits produce, so every freeze path runs
against the real object store. Only the import-gating tests (in
``test_benchmark_import_prs``) go through the ``fake_gh`` network boundary.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

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


@contextlib.contextmanager
def monkeypatch_relative_cwd(cwd: Path) -> Any:
    old = os.getcwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(old)


def _seed_origin(tmp_path: Path) -> str:
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
    return str(bare)


def _seed_two_pr_origin(tmp_path: Path) -> tuple[str, str, str]:
    """Bare origin with two PRs on unrelated ancestries: main(base1->base2->base3)
    with refs/pull/1/head off base2, plus a diverged `dev` branch (off base1) whose
    first commit is PR2's base tip and second commit is PR2's head. Returns
    (bare, dev_base_tip_sha, pr2_head_sha)."""
    repo = tmp_path / "seed_wt"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _write(repo, "readme.txt", "base1\n")
    _commit(repo, "base1")
    _write(repo, "base.py", "BASE = 2\n")
    _commit(repo, "base2")
    _write(repo, "beyond.py", "BEYOND = 3\n")
    _commit(repo, "base3")
    # PR 1 head off base2 (identical seed to _seed_origin => same _SHA_HEAD)
    _git(repo, "checkout", "--detach", "HEAD~1")            # base2
    repo.joinpath("base.py").write_text("BASE = 20\n")
    _git(repo, "add", "base.py")
    _write(repo, "feature.py", "FEATURE = 1\n")
    _commit(repo, "feature")
    pr1_head = _git(repo, "rev-parse", "HEAD")
    # PR 2: unrelated `dev` branch diverged from base1; base tip = dev1, head = dev2
    _git(repo, "checkout", "-b", "dev", "HEAD~2")           # base1
    _write(repo, "dev.py", "DEV = 1\n")
    dev_tip = _commit(repo, "dev1")
    repo.joinpath("dev.py").write_text("DEV = 2\n")
    _git(repo, "add", "dev.py")
    pr2_head = _commit(repo, "dev2")
    bare = tmp_path / "origin.git"
    bare.mkdir(parents=True, exist_ok=True)
    _git(bare, "init", "--bare")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "origin", "main:main", "dev:dev")
    _git(repo, "push", "origin", f"{pr1_head}:refs/pull/1/head", check=False)
    _git(repo, "push", "origin", f"{pr2_head}:refs/pull/2/head", check=False)
    return str(bare), dev_tip, pr2_head


def _seed_rename_origin(tmp_path: Path) -> tuple[str, str, str]:
    """Bare origin with a pure rename: main (authoring->rename) + refs/pull/1/head.

    Commit 1 authors ``old.py``; commit 2 renames it to ``new.py`` via
    ``git mv`` with no content change (a 100% rename). Returns
    ``(bare, authoring_sha, head_sha)``.
    """
    repo = tmp_path / "rename_wt"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _write(repo, "old.py", "def old() -> int:\n    return 1\n")
    authoring_sha = _commit(repo, "author old.py")
    _git(repo, "mv", "old.py", "new.py")
    head_sha = _commit(repo, "rename old.py to new.py")

    bare = tmp_path / "origin_rename.git"
    bare.mkdir(parents=True, exist_ok=True)
    _git(bare, "init", "--bare")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "origin", "main:main")
    _git(repo, "push", "origin", f"{head_sha}:refs/pull/1/head", check=False)
    return str(bare), authoring_sha, head_sha


# Deterministic SHAs/trees produced by ``_seed_origin`` (verified at seed build).
_SHA_BASE1 = 'cae67fc3eb4c5d3dd3353ca7fb41f909837bf0a2'
_SHA_BASE1_TREE = '2cd99bd20f7b3bac54014e20db1831d64b2c4fc9'
_SHA_BASE2 = 'd35f2cbffc81b6292f67cf891ac1c4256fe948a4'
_SHA_BASE2_TREE = 'a54e8fefe4dd3ffe592efe5fc64eb32f9eb7dbd4'
_SHA_BASE3 = '7a447892308f03f6861099ad03b5895397591f02'
_SHA_BASE3_TREE = '7bf0703afd952cd48a9a9e231cd6fee8e09cc5d0'
_SHA_HEAD = 'd9a75fd29107db73ef6cb08f877e644381c31f25'
_SHA_HEAD_TREE = '100c61d903cabfd705776af46193bc55d494940d'


# ---------------------------------------------------------------------------
# Task 0: spike -- the bare mirror retains full history for rename tracing
# ---------------------------------------------------------------------------


def test_mirror_supports_rename_tracing_for_anchor_derivation(tmp_path: Path) -> None:
    """The shared bare mirror must support both rename-tracing primitives the
    authoring-anchor derivation depends on: ``git log --follow`` (path history
    across renames) and ``git diff -M`` (rename detection). This pins the spec's
    Key Decision -- that a plain bare mirror (no shallow flags, plain fetch)
    retains full history -- so the rename-derivation strategy is sound.

    Spike finding (recorded task notes): in the bare mirror, ``git log`` with
    no start commit resolves to the symbolic ``HEAD`` -> ``refs/heads/main``,
    which is never fetched (the mirror carries only ``base_tip`` + pull/explicit
    refs), so the dotless form fails with "current branch 'main' does not have
    any commits yet". The decision survives: full history is retained, and
    ``--follow`` works as long as the caller names a start commit -- which the
    anchor helper always will (it traces from a concrete head/authoring SHA).
    If either probe fails, mirror handling must be revised (or the decision
    re-routed to the spec) before any anchor task runs.
    """
    from daydream.benchmark import snapshot as sn

    origin, authoring_sha, head_sha = _seed_rename_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_head_refs(tmp_path, "o/r", 1, explicit_shas=[head_sha], origin_url=origin)
    m = sn.mirror(tmp_path)
    assert sn.rev_parse(m, "refs/pull/1/head") == head_sha

    # (a) --follow (with an explicit start commit) must surface BOTH commits
    # when tracing the pre-rename path -- the bare mirror retains full history.
    traced = _git(m, "log", "--follow", "--format=%H", head_sha, "--", "old.py").splitlines()
    assert authoring_sha in traced, f"authoring commit missing from log --follow: {traced}"
    assert head_sha in traced, f"rename commit missing from log --follow: {traced}"
    assert traced[0] == head_sha  # most recent commit first

    # (b) diff -M must emit a pure rename record R100 old.py -> new.py.
    statuses = _git(m, "diff", "--name-status", authoring_sha, head_sha, "-M")
    assert "R100\told.py\tnew.py" in statuses, f"no R100 rename record in: {statuses!r}"


def _seed_anchor_origin(tmp_path: Path, *, pr: int = 1) -> tuple[str, str, str]:
    """Bare origin with no renames: main (authoring->edit) + ``refs/pull/N/head``.

    Commit 1 authors ``a.py``; commit 2 edits it in place (a plain ``M`` diff,
    no rename records). Returns ``(bare, authoring_sha, head_sha)``.
    """
    repo = tmp_path / "anchor_wt"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _write(repo, "a.py", "def a() -> int:\n    return 1\n")
    authoring_sha = _commit(repo, "author a.py")
    repo.joinpath("a.py").write_text("def a() -> int:\n    return 2\n")
    _git(repo, "add", "a.py")
    head_sha = _commit(repo, "edit a.py")

    bare = tmp_path / "anchor_origin.git"
    bare.mkdir(parents=True, exist_ok=True)
    _git(bare, "init", "--bare")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "origin", "main:main")
    _git(repo, "push", "origin", f"{head_sha}:refs/pull/{pr}/head", check=False)
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
    _write(repo, "old1.py", "def old1() -> int:\n    return 1\n")
    _write(repo, "old2.py", "def old2() -> int:\n    return 2\n")
    authoring_sha = _commit(repo, "author old1 old2")
    _git(repo, "mv", "old1.py", "new1.py")
    _git(repo, "mv", "old2.py", "new2.py")
    head_sha = _commit(repo, "rename both")

    bare = tmp_path / "dbl_origin.git"
    bare.mkdir(parents=True, exist_ok=True)
    _git(bare, "init", "--bare")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "origin", "main:main")
    _git(repo, "push", "origin", f"{head_sha}:refs/pull/{pr}/head", check=False)
    return str(bare), authoring_sha, head_sha


# ---------------------------------------------------------------------------
# Task 3 (plan): fail-closed authoring-path derivation over the pinned mirror
# ---------------------------------------------------------------------------


def test_derive_authoring_path_direct_hit(tmp_path: Path) -> None:
    """A path that exists in the authoring tree derives to itself -- the
    authoring commit is present and ``cat-file`` succeeds, so no rename trace
    (and no ``mapped_sha`` consultation) is needed."""
    from daydream.benchmark import snapshot

    origin, authoring_sha, head_sha = _seed_anchor_origin(tmp_path)
    snapshot.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    snapshot.fetch_head_refs(tmp_path, "o/r", 1, explicit_shas=[head_sha], origin_url=origin)
    m = snapshot.mirror(tmp_path)
    assert snapshot.derive_authoring_path(m, authoring_sha, "a.py", head_sha) == "a.py"


def test_derive_authoring_path_rename_traced(tmp_path: Path) -> None:
    """A path absent from the authoring tree whose head name is the ``R`` dest
    of a mirror rename trace resolves to the authoring-time (old) name."""
    from daydream.benchmark import snapshot

    origin, authoring_sha, head_sha = _seed_rename_origin(tmp_path)
    snapshot.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    snapshot.fetch_head_refs(tmp_path, "o/r", 1, explicit_shas=[head_sha], origin_url=origin)
    m = snapshot.mirror(tmp_path)
    assert snapshot.derive_authoring_path(m, authoring_sha, "new.py", head_sha) == "old.py"


def test_derive_authoring_path_fails_closed(tmp_path: Path) -> None:
    """Derivation never guesses: a missing authoring commit is
    ``history-unavailable``; a path that is neither in the authoring tree nor an
    exact rename dest fails ``path-unavailable`` -- even when the diff does
    contain rename candidates (the two-R-row diff must not be resolved by
    picking among the candidates)."""
    from daydream.benchmark import snapshot

    anchor_origin, anchor_auth, anchor_head = _seed_anchor_origin(tmp_path, pr=1)
    snapshot.fetch_head_refs(tmp_path, "o/r", 1, explicit_shas=[anchor_head], origin_url=anchor_origin)
    dbl_origin, dbl_auth, dbl_head = _seed_double_rename_origin(tmp_path, pr=3)
    snapshot.fetch_head_refs(tmp_path, "o/r", 3, explicit_shas=[dbl_head], origin_url=dbl_origin)
    m = snapshot.mirror(tmp_path)

    sha_absent = ("0" * 40, "a.py", anchor_head)  # (a) commit absent
    sha_present = (anchor_auth, "missing.py", anchor_head)  # (b) no rename, no path
    sha_ambiguous = (dbl_auth, "missing.py", dbl_head)  # (c) two R-candidates
    for args, expected_reason in [
        (sha_absent, "history-unavailable"),
        (sha_present, "path-unavailable"),
        (sha_ambiguous, "path-unavailable"),
    ]:
        with pytest.raises(snapshot.AnchorDerivationError) as exc:
            snapshot.derive_authoring_path(m, *args)
        assert expected_reason in str(exc.value)


# ---------------------------------------------------------------------------
# Task 1: shared bare-mirror establishment + PR ref fetch
# ---------------------------------------------------------------------------


def test_ensure_mirror_and_fetch_pr_head(tmp_path: Path) -> None:
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


def test_ancestor_of_pr_head_enforced(tmp_path: Path) -> None:
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


def test_resolve_base_and_trees(tmp_path: Path) -> None:
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    base = sn.resolve_original_base(m, "refs/heads/base_tip", _SHA_HEAD)
    assert base == _SHA_BASE2
    trees = sn.resolve_trees(m, base, _SHA_HEAD)
    assert isinstance(trees, tuple)
    bt, ht = trees
    assert bt == _SHA_BASE2_TREE and ht == _SHA_HEAD_TREE
    assert sn.resolve_trees(m, base, "0" * 40) == "missing_object"


# ---------------------------------------------------------------------------
# Task 4: degenerate-case detection + canonical diff sha
# ---------------------------------------------------------------------------


def test_degenerate_equal_trees_and_canonical_diff(tmp_path: Path) -> None:
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    assert sn.degenerate(m, _SHA_BASE2_TREE, _SHA_BASE2_TREE) == "equal_trees"
    assert sn.degenerate(m, _SHA_BASE2_TREE, _SHA_HEAD_TREE) is None   # real change
    d = sn.canonical_diff_sha256(m, _SHA_BASE2, _SHA_HEAD)
    assert re.fullmatch(r"[0-9a-f]{64}", d)


# ---------------------------------------------------------------------------
# Task 5: synthetic commits + deterministic minimal bundle
# ---------------------------------------------------------------------------


def test_bundle_two_refs_deterministic(tmp_path: Path) -> None:
    from daydream.benchmark import snapshot as sn
    from daydream.benchmark import storage

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    bundle = tmp_path / "snapshots" / "pr-000001-aaaaaaaaaaaa.bundle"
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, bundle)
    assert sn.bundle_heads(bundle) == {"refs/heads/base", "refs/heads/head"}
    base_commit = sn.rev_parse(m, "refs/heads/base")
    head_commit = sn.rev_parse(m, "refs/heads/head")
    assert sn.rev_parse(m, f"{base_commit}^{{tree}}") == _SHA_BASE2_TREE
    assert sn.rev_parse(m, f"{head_commit}^{{tree}}") == _SHA_HEAD_TREE
    assert sn.rev_parse(m, f"{head_commit}^") == base_commit           # single parent
    # determinism: rebuild and compare bytes against the first build's hash
    first = storage.sha256_file(bundle)
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, bundle)
    assert storage.sha256_file(bundle) == first


def test_bundle_heads_accepts_relative_path_from_any_cwd(tmp_path: Path) -> None:
    import os

    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    bundle = tmp_path / "snapshots" / "pr-000001-aaaaaaaaaaaa.bundle"
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, bundle)
    rel_bundle = Path(os.path.relpath(bundle, tmp_path))
    with monkeypatch_relative_cwd(tmp_path):
        heads_rel = sn.bundle_heads(rel_bundle)
        heads_abs = sn.bundle_heads(bundle)
    assert heads_rel == {"refs/heads/base", "refs/heads/head"}
    assert heads_abs == {"refs/heads/base", "refs/heads/head"}


# ---------------------------------------------------------------------------
# Task 6: offline-clone validation
# ---------------------------------------------------------------------------


def test_canonical_diff_digest_is_abbreviation_stable(tmp_path: Path) -> None:
    """The same base/head pair hashes identically whether the diff runs in a
    mirror whose effective core.abbrev is widened past the clone's. Failing-by-
    construction: pre-fix the mirror's 12-hex index lines mismatch the clone's
    default, so validate_offline_clone raises a digest mismatch."""
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[], origin_url=origin)
    m = sn.mirror(tmp_path)
    # widen the mirror's effective abbrev past the fresh 2-commit clone's default
    _git(m, "config", "core.abbrev", "12")
    bundle = tmp_path / "snapshots" / "pr-000001-aaaaaaaaaaaa.bundle"
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, bundle)
    diff_sha = sn.canonical_diff_sha256(m, _SHA_BASE2, _SHA_HEAD)
    # must not raise: the clone's diff digest must equal the mirror's
    sn.validate_offline_clone(bundle, _SHA_BASE2_TREE, _SHA_HEAD_TREE, diff_sha,
                              workdir=tmp_path)


def test_git_fetch_wires_command_scoped_credential_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    mirror = tmp_path / "mirror.git"
    git_ops._run_git(tmp_path, ["init", "--bare", str(mirror)], retries=0)
    captured: dict[str, list[str]] = {}

    real_run = subprocess.run

    def spy(args: Any, *pargs: Any, **kwargs: Any) -> Any:
        grg = list(args)
        if grg[:1] == ["git"] and "fetch" in grg:
            captured["argv"] = grg
        return real_run(args, *pargs, **kwargs)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", spy)
    sn._git_fetch(mirror, origin, ["refs/heads/main"])
    argv = captured["argv"]
    assert "-c" in argv and any(a.startswith("credential.helper=!gh auth git-credential") for a in argv)


def test_offline_clone_validates(tmp_path: Path) -> None:
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    bundle = tmp_path / "snapshots" / "pr-000001-aaaaaaaaaaaa.bundle"
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, bundle)
    diff_sha = sn.canonical_diff_sha256(m, _SHA_BASE2, _SHA_HEAD)
    sn.validate_offline_clone(bundle, _SHA_BASE2_TREE, _SHA_HEAD_TREE, diff_sha,
                              workdir=tmp_path)
    bad = tmp_path / "snapshots" / "bad.bundle"
    bad.write_bytes(bundle.read_bytes()[: len(bundle.read_bytes()) // 2])
    with pytest.raises(git_ops.GitError):
        sn.validate_offline_clone(bad, _SHA_BASE2_TREE, _SHA_HEAD_TREE, diff_sha,
                                  workdir=tmp_path)


def test_offline_clone_fidelity_rejects_tampering(tmp_path: Path) -> None:
    """Acceptance (b/c) at unit level: the offline-clone fidelity contract
    rejects every structurally-distinct tampered bundle shape (extra ref,
    extra reachable commit, wrong parent, wrong tree) while a valid bundle
    passes all probes."""
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    m = sn.mirror(tmp_path)
    base_tree, head_tree = _SHA_BASE2_TREE, _SHA_HEAD_TREE
    diff_sha = sn.canonical_diff_sha256(m, _SHA_BASE2, _SHA_HEAD)

    valid = tmp_path / "snapshots" / "pr-000001-aaaaaaaaaaaa.bundle"
    sn.build_bundle(m, _SHA_BASE2, _SHA_HEAD, valid)
    sn.validate_offline_clone(valid, base_tree, head_tree, diff_sha, workdir=tmp_path)

    # Synthetic commits with the same pinned identity as build_bundle's
    # (deterministic: identical tree+message+env => identical sha).
    synth_env = sn._synthetic_env()
    base_commit = _git(m, "commit-tree", base_tree, "-m", "snapshot base", env=synth_env)
    head_commit = _git(m, "commit-tree", head_tree, "-p", base_commit,
                       "-m", "snapshot head", env=synth_env)

    # (c) extra ref: a third ref in the bundle -> for-each-ref sees 3.
    extra_ref_bundle = tmp_path / "snapshots" / "extra-ref.bundle"
    extra = _git(m, "commit-tree", head_tree, "-p", base_commit, "-m", "extra", env=synth_env)
    _git(m, "update-ref", "refs/heads/extra", extra)
    _git(m, "bundle", "create", str(extra_ref_bundle),
         "refs/heads/base", "refs/heads/head", "refs/heads/extra")
    with pytest.raises(git_ops.GitError):
        sn.validate_offline_clone(extra_ref_bundle, base_tree, head_tree, diff_sha,
                                  workdir=tmp_path)

    # (c) extra reachable commit (no extra ref): head parented on an extra
    # commit parented on base -> rev-list --count origin/head == 3.
    extra_commit_bundle = tmp_path / "snapshots" / "extra-commit.bundle"
    extra_commit = _git(m, "commit-tree", base_tree, "-p", base_commit,
                        "-m", "extra", env=synth_env)
    head_tampered = _git(m, "commit-tree", head_tree, "-p", extra_commit,
                         "-m", "snapshot head", env=synth_env)
    _git(m, "update-ref", "refs/heads/head", head_tampered)
    _git(m, "bundle", "create", str(extra_commit_bundle),
         "refs/heads/base", "refs/heads/head")
    with pytest.raises(git_ops.GitError, match="exactly two reachable commits"):
        sn.validate_offline_clone(extra_commit_bundle, base_tree, head_tree, diff_sha,
                                  workdir=tmp_path)

    # (c) wrong parent: head with NO parent -> GitError (head^ fails, count 1).
    wrong_parent_bundle = tmp_path / "snapshots" / "wrong-parent.bundle"
    parentless_head = _git(m, "commit-tree", head_tree, "-m", "snapshot head", env=synth_env)
    _git(m, "update-ref", "refs/heads/head", parentless_head)
    _git(m, "bundle", "create", str(wrong_parent_bundle),
         "refs/heads/base", "refs/heads/head")
    with pytest.raises(git_ops.GitError):
        sn.validate_offline_clone(wrong_parent_bundle, base_tree, head_tree, diff_sha,
                                  workdir=tmp_path)

    # (c) wrong tree: head's tree is the base tree -> tree mismatch.
    wrong_tree_bundle = tmp_path / "snapshots" / "wrong-tree.bundle"
    wrong_tree_head = _git(m, "commit-tree", base_tree, "-p", base_commit,
                           "-m", "snapshot head", env=synth_env)
    _git(m, "update-ref", "refs/heads/head", wrong_tree_head)
    _git(m, "bundle", "create", str(wrong_tree_bundle),
         "refs/heads/base", "refs/heads/head")
    with pytest.raises(git_ops.GitError, match="tree mismatch"):
        sn.validate_offline_clone(wrong_tree_bundle, base_tree, head_tree, diff_sha,
                                  workdir=tmp_path)

    # Restore the mirror refs so the mirror is reusable for later freezes.
    _git(m, "update-ref", "refs/heads/base", base_commit)
    _git(m, "update-ref", "refs/heads/head", head_commit)



# ---------------------------------------------------------------------------
# Task 7: freeze_one ready / unreplayable reason matrix
# ---------------------------------------------------------------------------


def test_freeze_one_ready_and_reasons(tmp_path: Path) -> None:
    from daydream.benchmark import snapshot as sn
    from daydream.benchmark.schema import case_id_for

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    ready, bundle = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE2, head_sha=_SHA_HEAD,
                           policy="final_pr_head", requested_head="final", origin_url=origin)
    assert ready["status"] == "ready"
    assert ready["original_base_sha"] == _SHA_BASE2 and ready["original_head_sha"] == _SHA_HEAD
    assert ready["requested_base_sha"] == _SHA_BASE2
    assert ready["base_tree_sha"] == _SHA_BASE2_TREE and ready["head_tree_sha"] == _SHA_HEAD_TREE
    assert re.fullmatch(r"[0-9a-f]{64}", ready["diff_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", ready["bundle_sha256"])
    expect_rel = f"snapshots/{case_id_for(1, _SHA_HEAD)}.bundle"
    assert ready["bundle_file"] == expect_rel
    # freeze returns the bundle bytes to stage through the crash-consistent
    # transaction; it never writes the final snapshots/<case>.bundle itself.
    assert isinstance(bundle, bytes) and bundle.startswith(b"# v2 git bundle")
    assert not (tmp_path / expect_rel).exists()
    # head_not_on_pr: a base3 head reachable elsewhere is rejected
    ur, bundle = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE3, head_sha=_SHA_BASE3,
                               policy="explicit_head", requested_head=_SHA_BASE3, origin_url=origin)
    assert ur["status"] == "unreplayable" and ur["error"]["reason"] == "head_not_on_pr"
    assert bundle is None
    assert ur["bundle_file"] is None and ur["base_tree_sha"] is None
    # head_unreachable: a sha absent from the mirror
    ur2, bundle2 = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE2, head_sha="0" * 40,
                                 policy="explicit_head", requested_head="0" * 40, origin_url=origin)
    assert ur2["status"] == "unreplayable" and ur2["error"]["reason"] == "head_unreachable"
    assert bundle2 is None
    assert ur2["bundle_file"] is None


def test_freeze_two_prs_unrelated_base_tips_both_ready(tmp_path: Path) -> None:
    """The forced +{base_tip} refspec lets two PRs with unrelated, non-fast-forward
    base tips both freeze ready in one shared mirror (regression for defect 3)."""
    from daydream.benchmark import snapshot as sn

    origin, dev_tip, pr2_head = _seed_two_pr_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    m = sn.mirror(tmp_path)
    ready1, b1 = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE2, head_sha=_SHA_HEAD,
                               policy="final_pr_head", requested_head="final", origin_url=origin)
    ready2, b2 = sn.freeze_one(tmp_path, "o/r", 2, base_tip=dev_tip, head_sha=pr2_head,
                               policy="final_pr_head", requested_head="final", origin_url=origin)
    assert ready1["status"] == "ready" and isinstance(b1, bytes)
    assert ready2["status"] == "ready" and isinstance(b2, bytes)
    # the shared base_tip ref was force-re-pointed from base2 to the unrelated dev tip
    assert sn.rev_parse(m, "refs/heads/base_tip") == dev_tip


def test_freeze_one_base_advanced_two_sha(tmp_path: Path) -> None:
    """Acceptance (a): a base branch advanced past the PR fork records the true
    merge base as original_base_sha and the selected base tip as
    requested_base_sha — two distinct SHAs."""
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=_SHA_BASE2,
                     explicit_shas=[_SHA_HEAD], origin_url=origin)
    # PR head is forked from base2; main has advanced to base3.
    ready, bundle = sn.freeze_one(tmp_path, "o/r", 1, base_tip=_SHA_BASE3, head_sha=_SHA_HEAD,
                           policy="final_pr_head", requested_head="final", origin_url=origin)
    assert ready["status"] == "ready"
    assert ready["original_base_sha"] == _SHA_BASE2      # the true merge base
    assert ready["requested_base_sha"] == _SHA_BASE3      # the selected base-branch tip
    assert ready["original_base_sha"] != ready["requested_base_sha"]
    assert isinstance(bundle, bytes) and bundle.startswith(b"# v2 git bundle")


def test_freeze_distinct_base_vs_head_unreachable(tmp_path: Path) -> None:
    """A base-tip fetch failure classifies ``base_unreachable``; a PR-head fetch
    failure classifies ``head_unreachable`` — never collapsed to one reason."""
    from daydream.benchmark import snapshot as sn

    origin = _seed_origin(tmp_path)
    # base-tip ref absent on the origin (only base1..3 + refs/pull/1/head exist)
    ur, b = sn.freeze_one(tmp_path, "o/r", 1, base_tip="0" * 40, head_sha=_SHA_HEAD,
                          policy="final_pr_head", requested_head="final", origin_url=origin)
    assert ur["status"] == "unreplayable" and ur["error"]["reason"] == "base_unreachable"
    assert b is None
    assert ur["requested_base_sha"] == "0" * 40
    # PR-head ref absent (origin has only refs/pull/1/head, not 999)
    ur2, b2 = sn.freeze_one(tmp_path, "o/r", 999, base_tip=_SHA_BASE2, head_sha="0" * 40,
                            policy="final_pr_head", requested_head="final", origin_url=origin)
    assert ur2["error"]["reason"] == "head_unreachable" and b2 is None
    assert ur2["requested_base_sha"] == _SHA_BASE2


# ---------------------------------------------------------------------------
# Task 10: crash injection at case/bundle/manifest transaction boundaries
# ---------------------------------------------------------------------------


def test_freeze_crash_recovers_whole_before_or_after(tmp_path: Path) -> None:
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


# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Task 13: rich-origin fidelity seed + end-to-end matrix
# ---------------------------------------------------------------------------


def _seed_rich_origin(tmp_path: Path) -> tuple[str, str, str, str, str]:
    """A rich bare origin; returns ``(origin, base, head, base_tree, head_tree)``.

    base: text file + 100755 executable + 120000 symlink + binary blob.
    head: renames the text file, deletes the symlink, edits the binary.
    """
    repo = tmp_path / "rich_wt"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _write(repo, "readme.txt", "hello rich\n")
    (repo / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (repo / "run.sh").chmod(0o755)
    _git(repo, "add", "run.sh")
    (repo / "payload.bin").write_bytes(b"\x00\x01\x02\xfe\xff")
    _git(repo, "add", "payload.bin")
    if hasattr(os, "symlink"):
        os.symlink("readme.txt", repo / "alias.txt")
        _git(repo, "add", "alias.txt")
    base_sha = _commit(repo, "rich base")
    base_tree = _git(repo, "rev-parse", f"{base_sha}^{{tree}}")

    _git(repo, "mv", "readme.txt", "renamed.txt")
    if (repo / "alias.txt").exists():
        _git(repo, "rm", "alias.txt")
    (repo / "payload.bin").write_bytes(b"\x00\x01\x02\x03\x04\x05")
    _git(repo, "add", "payload.bin")
    head_sha = _commit(repo, "rich head")
    head_tree = _git(repo, "rev-parse", f"{head_sha}^{{tree}}")

    bare = tmp_path / "rich_origin.git"
    if bare.exists():
        shutil.rmtree(bare)
    bare.mkdir()
    _git(bare, "init", "--bare")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "origin", "main:main")
    _git(repo, "push", "origin", f"{head_sha}:refs/pull/1/head", check=False)
    return str(bare), base_sha, head_sha, base_tree, head_tree


def test_e2e_fidelity_trees_modes_symlinks_renames_deletions_binaries(tmp_path: Path) -> None:
    from daydream.benchmark import snapshot as sn
    from daydream.benchmark import storage

    origin, base, head, base_tree, head_tree = _seed_rich_origin(tmp_path)
    sn.ensure_mirror(tmp_path, "o/r", origin_url=origin)
    sn.fetch_pr_refs(tmp_path, "o/r", 1, base_tip=base, explicit_shas=[head],
                     origin_url=origin)
    m = sn.mirror(tmp_path)
    bundle = tmp_path / "snapshots" / "pr-000001-000000000000.bundle"
    sn.build_bundle(m, base, head, bundle)
    assert sn.bundle_heads(bundle) == {"refs/heads/base", "refs/heads/head"}
    # trees match the origin exactly (modes, symlink targets, binary preserved)
    assert sn.rev_parse(m, "refs/heads/base^{tree}") == base_tree
    assert sn.rev_parse(m, "refs/heads/head^{tree}") == head_tree
    # offline clone recomputes matching trees + diff
    diff = sn.canonical_diff_sha256(m, base, head)
    sn.validate_offline_clone(bundle, base_tree, head_tree, diff, workdir=tmp_path)
    # repeatable bytes
    first = storage.sha256_file(bundle)
    sn.build_bundle(m, base, head, bundle)
    assert storage.sha256_file(bundle) == first
    # no origin remote and only the two refs in an offline clone
    clone = _clone_offline(bundle, tmp_path)
    try:
        url = _git(Path(clone), "remote", "get-url", "origin")
        # the clone's only remote points at the local bundle artifact, so it is
        # fully offline-replayable with no GitHub/HTTPS network dependency
        assert url and not url.startswith(("https://", "http://", "git://", "ssh://"))
        refs = _git(Path(clone), "for-each-ref", "--format=%(refname)", "refs/remotes")
        assert {line for line in refs.splitlines() if line} == {
            "refs/remotes/origin/base",
            "refs/remotes/origin/head",
        }
    finally:
        shutil.rmtree(clone, ignore_errors=True)


def _clone_offline(bundle: Path, workdir: Path) -> Path:
    """Clone *bundle* into a fresh temp dir (network-disabled local source)."""
    import tempfile

    workdir.mkdir(parents=True, exist_ok=True)
    clone = tempfile.mkdtemp(prefix="e2e-", dir=str(workdir))
    _git(bundle.parent, "clone", "--no-local", "--no-checkout", str(bundle), clone)
    return Path(clone)
