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


def degenerate(mirror_repo: Path, base_tree: str, head_tree: str) -> str | None:
    """Classify a degenerate (empty) base/head change, or None when real.

    Equal trees return ``"equal_trees"`` (the more specific diagnosis); a
    non-empty diff returns ``None``. A git diff failure raises ``GitError``.
    """
    if base_tree == head_tree:
        return "equal_trees"
    proc = git_ops._run_git(mirror_repo, ["diff", "--binary", "--quiet", base_tree, head_tree], retries=0)
    if proc.returncode != 0:
        # non-zero means the trees differ (git diff --quiet: 0 = no changes).
        return None
    return "empty_diff"


def canonical_diff_sha256(mirror_repo: Path, base_sha: str, head_sha: str) -> str:
    """sha256 of the canonical binary-safe diff between two commits."""
    proc = git_ops._run_git(
        mirror_repo, ["diff", "--binary", base_sha, head_sha], retries=0, capture_bytes=True
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise git_ops.GitError(f"git diff --binary {base_sha} {head_sha} failed: {stderr.strip()}")
    return hashlib.sha256(proc.stdout).hexdigest()


def sha256_of(path: Path) -> str:
    """sha256 file digest of *path*."""
    return storage.sha256_file(path)


def _synthetic_env() -> dict[str, str]:
    return {**os.environ, **_SYNTH_AUTHOR, "GIT_TERMINAL_PROMPT": "0"}


def build_bundle(
    mirror_repo: Path, base_sha: str, head_sha: str, bundle_path: Path, case_id: str | None = None
) -> None:
    """Write a deterministic minimal ``refs/heads/base`` + ``refs/heads/head`` bundle.

    Builds two synthetic commits directly from the original base/head tree
    objects with pinned identity/timestamp, then exposes only the two refs. Any
    non-zero step raises :class:`GitError` so the caller maps it to
    ``bundle_failure``.
    """
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    env = _synthetic_env()
    base_tree = rev_parse(mirror_repo, f"{base_sha}^{{tree}}")
    head_tree = rev_parse(mirror_repo, f"{head_sha}^{{tree}}")

    base_proc = git_ops._run_git(
        mirror_repo, ["commit-tree", base_tree, "-m", "snapshot base"], env_cmd=env, retries=0, timeout=30
    )
    if base_proc.returncode != 0:
        raise git_ops.GitError(f"snapshot base commit-tree failed: {base_proc.stderr.strip()}")
    base_commit = base_proc.stdout.strip()
    head_proc = git_ops._run_git(
        mirror_repo, ["commit-tree", head_tree, "-p", base_commit, "-m", "snapshot head"],
        env_cmd=env, retries=0, timeout=30,
    )
    if head_proc.returncode != 0:
        raise git_ops.GitError(f"snapshot head commit-tree failed: {head_proc.stderr.strip()}")
    head_commit = head_proc.stdout.strip()
    for ref, sha in (("refs/heads/base", base_commit), ("refs/heads/head", head_commit)):
        up = git_ops._run_git(mirror_repo, ["update-ref", ref, sha], env_cmd=env, retries=0, timeout=30)
        if up.returncode != 0:
            raise git_ops.GitError(f"git update-ref {ref} failed: {up.stderr.strip()}")
    bundle = git_ops._run_git(
        mirror_repo, ["bundle", "create", str(bundle_path), "refs/heads/base", "refs/heads/head"],
        env_cmd=env, retries=0, timeout=120,
    )
    if bundle.returncode != 0:
        raise git_ops.GitError(f"git bundle create failed: {bundle.stderr.strip()}")


def bundle_heads(bundle_path: Path) -> set[str]:
    """The list of refs a bundle exposes."""
    bundle_path = Path(bundle_path)
    proc = git_ops._run_git(
        bundle_path.parent, ["bundle", "list-heads", str(bundle_path)], retries=0, timeout=30
    )
    if proc.returncode != 0:
        raise git_ops.GitError(f"git bundle list-heads failed: {proc.stderr.strip()}")
    heads: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            heads.add(parts[1])
    return heads


def validate_offline_clone(
    bundle_path: Path, base_tree: str, head_tree: str, diff_sha256: str, workdir: Path
) -> None:
    """Clone the bundle with network disabled and verify refs/trees/diff.

    Raises :class:`GitError` naming the failing check on any clone error,
    missing ref, tree mismatch, or diff-digest mismatch. Returns ``None``.
    """
    bundle_path = Path(bundle_path)
    import tempfile

    clone_dir = tempfile.mkdtemp(prefix="clone-", dir=str(workdir))
    try:
        proc = git_ops._run_git(
            Path(workdir), ["clone", "--no-local", "--no-checkout", str(bundle_path), clone_dir],
            retries=0, timeout=120,
        )
        if proc.returncode != 0:
            raise git_ops.GitError(f"offline clone of {bundle_path} failed: {proc.stderr.strip()}")
        for ref, expected in (
            ("refs/remotes/origin/base", base_tree),
            ("refs/remotes/origin/head", head_tree),
        ):
            got = _run_git_cwd(clone_dir, ["rev-parse", "--verify", f"{ref}^{{tree}}"])
            if got != expected:
                raise git_ops.GitError(f"offline clone tree mismatch for {ref} (expected {expected}, got {got})")
        diff = _run_git_cwd(clone_dir, ["diff", "--binary", "refs/remotes/origin/base", "refs/remotes/origin/head"], capture_bytes=True)
        if hashlib.sha256(diff).hexdigest() != diff_sha256:
            raise git_ops.GitError(f"offline clone diff digest mismatch (case {bundle_path})")
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)
    return None


def _run_git_cwd(repo: Path, args: list[str], *, capture_bytes: bool = False) -> str | bytes:
    """Run git and raise GitError on non-zero exit (offline-clone helper)."""
    proc = git_ops._run_git(repo, args, retries=0, capture_bytes=capture_bytes, timeout=30)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
        raise git_ops.GitError(f"git {' '.join(args)} failed: {stderr.strip()}")
    if capture_bytes:
        return proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    return proc.stdout.strip()


def freeze_one(
    root: Path,
    repo_slug: str,
    pr_number: int,
    *,
    base_tip: str,
    head_sha: str,
    policy: str,
    requested_head: str,
    origin_url: str | None = None,
) -> dict:
    """Freeze one requested head into a ``ready`` or ``unreplayable`` snapshot.

    Runs the full pipeline (mirror ensure -> fetch -> ancestry -> merge-base -> trees
    -> degenerate -> bundle -> offline validate). Classified git failures return an
    ``unreplayable`` dict with an exact reason; only unexpected errors propagate.
    """
    root = Path(root)
    origin_url = origin_url or f"https://github.com/{repo_slug}.git"
    case_id = schema.case_id_for(pr_number, head_sha)
    bundle_rel = f"snapshots/{case_id}.bundle"
    bundle_path = root / bundle_rel

    def unreplayable(reason: str, detail: str) -> dict:
        return {
            "status": "unreplayable",
            "policy": policy,
            "requested_head": requested_head,
            "original_base_sha": base_tip,
            "original_head_sha": head_sha,
            "base_tree_sha": None,
            "head_tree_sha": None,
            "diff_sha256": None,
            "bundle_file": None,
            "bundle_sha256": None,
            "error": {"reason": reason, "detail": detail},
        }

    # 1) mirror ensure + fetch base tip, PR head ref, and the requested head.
    try:
        m = ensure_mirror(root, repo_slug, origin_url)
        fetch_pr_refs(root, repo_slug, pr_number, base_tip, [head_sha], origin_url)
        pr_head = rev_parse(m, f"refs/pull/{pr_number}/head")
    except git_ops.GitError as exc:
        return unreplayable("head_unreachable", f"could not fetch/resolve the PR head: {exc}")

    # 2) the requested head must be the PR head or an ancestor on its ancestry.
    reach = head_reachability(m, head_sha, pr_head)
    if reach == "head_unreachable":
        return unreplayable("head_unreachable", f"requested head {head_sha[:12]} is not in the mirror")
    if reach != "ok":
        return unreplayable("head_not_on_pr", f"requested head {head_sha[:12]} is reachable elsewhere but not on the PR head")

    # 3) resolve the merge base and both trees.
    base = resolve_original_base(m, "refs/heads/base_tip", head_sha)
    if base is None:
        return unreplayable("base_unreachable", "no merge-base could be resolved for the sourced base tip and head")
    trees = resolve_trees(m, base, head_sha)
    if trees == "missing_object":
        return unreplayable("missing_object", "a source tree object is absent from the mirror")
    base_tree, head_tree = trees

    # 4) a clean review still requires a real code change.
    degen = degenerate(m, base_tree, head_tree)
    if degen is not None:
        return unreplayable(degen, f"no real code change between base and head ({degen})")

    # 5) canonical diff + deterministic bundle + offline validation.
    try:
        diff_sha = canonical_diff_sha256(m, base, head_sha)
        storage.ensure_private_dir(root / "snapshots")
        build_bundle(m, base, head_sha, bundle_path, case_id)
        bundle_sha = storage.sha256_file(bundle_path)
        validate_offline_clone(bundle_path, base_tree, head_tree, diff_sha, workdir=root / "cache")
    except git_ops.GitError as exc:
        return unreplayable("bundle_failure", f"bundle build/validate failed: {exc}")

    return {
        "status": "ready",
        "policy": policy,
        "requested_head": requested_head,
        "original_base_sha": base_tip,
        "original_head_sha": head_sha,
        "base_tree_sha": base_tree,
        "head_tree_sha": head_tree,
        "diff_sha256": diff_sha,
        "bundle_file": bundle_rel,
        "bundle_sha256": bundle_sha,
        "error": None,
    }
