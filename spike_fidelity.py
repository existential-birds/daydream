#!/usr/bin/env python3
"""Spike: offline-clone fidelity probe commands + tampered-bundle construction.

Standalone (no daydream import) — mirrors the freeze topology with raw git:
an origin whose main is ``base1 -> base2 -> base3`` and a PR head off ``base2``
(exactly ``tests/test_benchmark_snapshot.py::_seed_origin``), a deterministic
two-ref bundle built with the same pinned commit-tree identity, an offline
``git clone --no-local --no-checkout <bundle>``, and the fidelity probes that
``validate_offline_clone`` will run plus one probe per tampered bundle shape.

Prints PASS for the valid bundle and one FAIL line per tampered shape —
proving each probe command rejects the shape it must reject.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SYNTH_ENV = {
    "GIT_AUTHOR_NAME": "Daydream Snapshot",
    "GIT_AUTHOR_EMAIL": "benchmark@daydream",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Daydream Snapshot",
    "GIT_COMMITTER_EMAIL": "benchmark@daydream",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}
_SEED_ENV = {
    "GIT_AUTHOR_NAME": "Tester",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Tester",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def git(repo: Path, *args: str, env: dict[str, str] | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        env={**os.environ, **env} if env else os.environ.copy(), check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def commit(repo: Path, message: str) -> str:
    git(repo, "commit", "-m", message, env=_SEED_ENV)
    return git(repo, "rev-parse", "HEAD")


def write(repo: Path, name: str, content: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    git(repo, "add", name)


def seed_origin(tmp: Path):
    """Mirror ``_seed_origin``: main base1->base2->base3 + PR head off base2."""
    repo = tmp / "seed_wt"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    write(repo, "readme.txt", "base1\n")
    commit(repo, "base1")
    write(repo, "base.py", "BASE = 2\n")
    base2 = commit(repo, "base2")
    write(repo, "beyond.py", "BEYOND = 3\n")
    commit(repo, "base3")
    git(repo, "checkout", "--detach", base2)
    (repo / "base.py").write_text("BASE = 20\n")
    git(repo, "add", "base.py")
    write(repo, "feature.py", "FEATURE = 1\n")
    head = commit(repo, "feature")
    bare = tmp / "origin.git"
    bare.mkdir()
    git(bare, "init", "--bare")
    git(repo, "remote", "add", "origin", str(bare))
    git(repo, "push", "origin", "main:main")
    git(repo, "push", "origin", f"{head}:refs/pull/1/head")
    return bare, base2, head


def build_bundle(repo: Path, base: str, head: str, bundle: Path, extra_refs: tuple = ()) -> tuple[str, str]:
    """Deterministic minimal bundle: synthetic base/head commits, two refs (+ extras)."""
    base_tree = git(repo, "rev-parse", f"{base}^{{tree}}")
    head_tree = git(repo, "rev-parse", f"{head}^{{tree}}")
    base_commit = git(repo, "commit-tree", base_tree, "-m", "snapshot base", env=_SYNTH_ENV)
    head_commit = git(repo, "commit-tree", head_tree, "-p", base_commit, "-m", "snapshot head",
                      env=_SYNTH_ENV)
    git(repo, "update-ref", "refs/heads/base", base_commit)
    git(repo, "update-ref", "refs/heads/head", head_commit)
    bundle = Path(bundle)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    refs = ["refs/heads/base", "refs/heads/head", *extra_refs]
    git(repo, "bundle", "create", str(bundle), *refs)
    return base_commit, head_commit


def build_tampered_bundle(repo: Path, base: str, head: str, bundle: Path, shape: str) -> None:
    """Build one tampered bundle shape; the probes must each reject it."""
    base_tree = git(repo, "rev-parse", f"{base}^{{tree}}")
    head_tree = git(repo, "rev-parse", f"{head}^{{tree}}")
    base_commit = git(repo, "commit-tree", base_tree, "-m", "snapshot base", env=_SYNTH_ENV)
    if shape == "extra_ref":
        extra = git(repo, "commit-tree", head_tree, "-p", base_commit, "-m", "extra", env=_SYNTH_ENV)
        git(repo, "update-ref", "refs/heads/extra", extra)
        build_bundle(repo, base, head, bundle, extra_refs=("refs/heads/extra",))
    elif shape == "extra_commit":
        extra = git(repo, "commit-tree", base_tree, "-p", base_commit, "-m", "extra", env=_SYNTH_ENV)
        head_tampered = git(repo, "commit-tree", head_tree, "-p", extra, "-m", "snapshot head",
                            env=_SYNTH_ENV)
        build_bundle(repo, base, head, bundle)
        # rewrite the head ref inside the bundle to the tampered commit
        git(repo, "update-ref", "refs/heads/head", head_tampered)
        git(repo, "bundle", "create", str(bundle), "refs/heads/base", "refs/heads/head")
    elif shape == "wrong_parent":
        head_tampered = git(repo, "commit-tree", head_tree, "-m", "snapshot head", env=_SYNTH_ENV)
        git(repo, "update-ref", "refs/heads/head", head_tampered)
        git(repo, "bundle", "create", str(bundle), "refs/heads/base", "refs/heads/head")
    elif shape == "wrong_tree":
        head_tampered = git(repo, "commit-tree", base_tree, "-p", base_commit, "-m", "snapshot head",
                            env=_SYNTH_ENV)
        git(repo, "update-ref", "refs/heads/head", head_tampered)
        git(repo, "bundle", "create", str(bundle), "refs/heads/base", "refs/heads/head")
    else:
        raise AssertionError(f"unknown shape {shape}")


def clone_offline(bundle: Path, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    clone = Path(tempfile.mkdtemp(prefix="spike-clone-", dir=str(workdir)))
    git(workdir, "clone", "--no-local", "--no-checkout", str(bundle), str(clone))
    return clone


def probe_valid(clone: Path, base_tree: str, head_tree: str, diff_sha256: str) -> None:
    """The fidelity probe set a valid offline clone must pass."""
    refs = git(clone, "for-each-ref", "--format=%(refname)", "refs/remotes")
    assert set(refs.splitlines()) == {"refs/remotes/origin/base", "refs/remotes/origin/head"}, refs
    assert git(clone, "rev-list", "--count", "refs/remotes/origin/head") == "2"
    assert git(clone, "rev-parse", "refs/remotes/origin/head^") == \
        git(clone, "rev-parse", "refs/remotes/origin/base")
    assert len(git(clone, "rev-list", "--parents", "refs/remotes/origin/base").splitlines()) == 1
    assert git(clone, "rev-parse", "--verify", "refs/remotes/origin/base^{tree}") == base_tree
    assert git(clone, "rev-parse", "--verify", "refs/remotes/origin/head^{tree}") == head_tree
    diff = git(clone, "diff", "--binary", "refs/remotes/origin/base", "refs/remotes/origin/head")
    import hashlib
    assert hashlib.sha256(diff.encode()).hexdigest() == diff_sha256


def probe_rejects(clone: Path, base_tree: str, head_tree: str, diff_sha256: str, shape: str) -> list[str]:
    """Run each probe against a tampered clone; collect the probes that reject it."""
    import hashlib

    rejected: list[str] = []
    try:
        refs = git(clone, "for-each-ref", "--format=%(refname)", "refs/remotes")
        if set(refs.splitlines()) != {"refs/remotes/origin/base", "refs/remotes/origin/head"}:
            rejected.append("for-each-ref (ref set)")
    except AssertionError:
        rejected.append("for-each-ref (git failure)")
    try:
        count = git(clone, "rev-list", "--count", "refs/remotes/origin/head")
        if count != "2":
            rejected.append(f"rev-list --count (got {count})")
    except AssertionError:
        rejected.append("rev-list --count (git failure)")
    try:
        if git(clone, "rev-parse", "refs/remotes/origin/head^") != \
                git(clone, "rev-parse", "refs/remotes/origin/base"):
            rejected.append("rev-parse head^ (parent mismatch)")
    except AssertionError:
        rejected.append("rev-parse head^ (git failure)")
    try:
        if len(git(clone, "rev-list", "--parents", "refs/remotes/origin/base").splitlines()) != 1:
            rejected.append("rev-list --parents base (root probe)")
    except AssertionError:
        rejected.append("rev-list --parents base (git failure)")
    try:
        if git(clone, "rev-parse", "--verify", "refs/remotes/origin/head^{tree}") != head_tree:
            rejected.append("tree mismatch (head)")
    except AssertionError:
        rejected.append("tree probe (git failure)")
    try:
        diff = git(clone, "diff", "--binary", "refs/remotes/origin/base", "refs/remotes/origin/head")
        if hashlib.sha256(diff.encode()).hexdigest() != diff_sha256:
            rejected.append("diff digest mismatch")
    except AssertionError:
        rejected.append("diff probe (git failure)")
    return rejected


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="spike-fidelity-"))
    try:
        origin, base2, head = seed_origin(tmp)
        base_tree = git(origin, "rev-parse", f"{base2}^{{tree}}")
        head_tree = git(origin, "rev-parse", f"{head}^{{tree}}")
        import hashlib
        diff_sha256 = hashlib.sha256(
            git(origin, "diff", "--binary", base2, head).encode()
        ).hexdigest()

        # --- valid bundle passes every probe ---
        valid_bundle = tmp / "snapshots" / "valid.bundle"
        build_bundle(origin, base2, head, valid_bundle)
        clone = clone_offline(valid_bundle, tmp / "clones")
        try:
            probe_valid(clone, base_tree, head_tree, diff_sha256)
        finally:
            shutil.rmtree(clone, ignore_errors=True)
        print("PASS valid bundle passes all fidelity probes")

        # --- tampered shapes, each rejected by >= 1 probe ---
        shapes = ["extra_ref", "extra_commit", "wrong_parent", "wrong_tree"]
        for shape in shapes:
            bundle = tmp / "snapshots" / f"{shape}.bundle"
            build_tampered_bundle(origin, base2, head, bundle, shape)
            clone = clone_offline(bundle, tmp / "clones")
            try:
                rejected = probe_rejects(clone, base_tree, head_tree, diff_sha256, shape)
            finally:
                shutil.rmtree(clone, ignore_errors=True)
            if rejected:
                print(f"FAIL {shape}: rejected by {'; '.join(rejected)}")
            else:
                print(f"FAIL {shape}: NO PROBE REJECTED IT")

        # --- (b) content-restamped: bytes changed + checksum restamped ---
        restamped = tmp / "snapshots" / "restamped.bundle"
        raw = valid_bundle.read_bytes()
        for label, tampered in (
            ("append", raw + b"INJECTED"),
            ("byte-flip", raw[: len(raw) // 2] + b"\x00" + raw[len(raw) // 2 + 1:]),
        ):
            restamped.write_bytes(tampered)
            restamp_sha = hashlib.sha256(tampered).hexdigest()
            try:
                clone = clone_offline(restamped, tmp / "clones")
            except AssertionError as exc:
                print(f"FAIL content-restamped ({label}): rejected at clone ({exc})")
                continue
            try:
                rejected = probe_rejects(clone, base_tree, head_tree, diff_sha256, "restamped")
            finally:
                shutil.rmtree(clone, ignore_errors=True)
            if rejected:
                print(f"FAIL content-restamped ({label}): rejected by {'; '.join(rejected)} "
                      f"(restamped sha {restamp_sha[:12]}...)")
            else:
                print(f"FAIL content-restamped ({label}): NO PROBE REJECTED IT "
                      f"(restamped sha {restamp_sha[:12]}...)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
