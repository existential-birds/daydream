"""Freeze PR heads into deterministic, self-contained base/head bundles. A shared mirror
supplies ancestry and tree proofs; freezing returns ready or a classified unreplayable
outcome after offline fidelity validation.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import AbstractSet, Any, Literal, NamedTuple, cast, overload

from daydream import git_ops
from daydream.benchmark import schema, storage
from daydream.git_ops import process as git_process

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
    proc = git_process._run_git(repo, ["rev-parse", "--verify", ref], retries=0)
    git_process._require_ok(proc, f"git rev-parse --verify {ref} failed in {repo}")
    return proc.stdout.strip()


def _fetch_env() -> dict[str, str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    return env


def ensure_mirror(root: Path) -> Path:
    """Return ``root/cache/repository.git``, creating the bare mirror if absent."""
    root = Path(root)
    m = mirror(root)
    if not m.exists():
        m.parent.mkdir(parents=True, exist_ok=True)
        proc = git_process._run_git(
            root, ["init", "--bare", str(m)], env_cmd=_fetch_env(), retries=0, timeout=30,
        )
        git_process._require_ok(proc, f"git init --bare {m} failed")
    return m


def _git_fetch(mirror_repo: Path, url: str, refspecs: list[str]) -> None:
    """Fetch with command-scoped gh credential-helper authentication and terminal prompts
    disabled. Local/file origins skip the helper; unresolved refspecs raise GitError.
    """
    url = str(url)
    args = [*git_process._credential_helper_args(), "fetch", url, *refspecs]
    proc = git_process._run_git(mirror_repo, args, env_cmd=_fetch_env(), retries=0, timeout=300)
    git_process._require_ok(proc, f"git fetch {url} {' '.join(refspecs)} failed")


def fetch_base_tip(
    root: Path,
    repo_slug: str,
    base_tip: str,
    origin_url: str | None = None,
) -> Path:
    """Force-update the mirror base_tip ref, including moves to ancestor commits. Return
    the mirror path; fetch failure raises GitError for base_unreachable classification.
    """
    root = Path(root)
    origin_url = origin_url or f"https://github.com/{repo_slug}.git"
    m = ensure_mirror(root)
    _git_fetch(m, origin_url, [f"+{base_tip}:refs/heads/base_tip"])
    return m


def fetch_head_refs(
    root: Path,
    repo_slug: str,
    pr_number: int,
    explicit_shas: list[str] | tuple[str, ...] = (),
    origin_url: str | None = None,
) -> Path:
    """Fetch ``refs/pull/N/head`` + explicit heads into the mirror."""
    root = Path(root)
    origin_url = origin_url or f"https://github.com/{repo_slug}.git"
    m = ensure_mirror(root)
    refspecs = [f"refs/pull/{pr_number}/head:refs/pull/{pr_number}/head"]
    for sha in explicit_shas:
        refspecs.append(f"{sha}:refs/heads/explicit-{sha[:12]}")
    _git_fetch(m, origin_url, refspecs)
    return m


def head_reachability(mirror_repo: Path, sha: str, pr_head_sha: str) -> str:
    """Classify a SHA as ok, head_not_on_pr, or head_unreachable without raising for
    missing objects.
    """
    if sha == pr_head_sha:
        return "ok"
    verify = git_process._run_git(mirror_repo, ["rev-parse", "--verify", f"{sha}^{{commit}}"], retries=0)
    if verify.returncode != 0:
        return "head_unreachable"
    anc = git_process._run_git(mirror_repo, ["merge-base", "--is-ancestor", sha, pr_head_sha], retries=0)
    return "ok" if anc.returncode == 0 else "head_not_on_pr"


def commit_relation(mirror_repo: Path, head: str, commit: str) -> str:
    """Classify at_head/ancestor/non_ancestor; absent objects or probe failures return
    unavailable.
    """
    if commit == head:
        return "at_head"
    try:
        verify = git_process._run_git(
            mirror_repo, ["rev-parse", "--verify", f"{commit}^{{commit}}"], retries=0,
        )
        if verify.returncode != 0:
            return "unavailable"
        anc = git_process._run_git(
            mirror_repo, ["merge-base", "--is-ancestor", commit, head], retries=0,
        )
    except git_ops.GitError:
        return "unavailable"
    return "ancestor" if anc.returncode == 0 else "non_ancestor"


class AnchorDiff(NamedTuple):
    """Disjoint whole-tree change buckets reusable for all anchors on one content-addressed
    commit pair.
    """

    renames: frozenset[str]
    binary: frozenset[str]
    modified: frozenset[str]
    deleted: frozenset[str]


def _nul_fields(stdout: str | bytes) -> list[str]:
    """Split raw git -z path bytes on NUL and decode with surrogateescape."""
    stdout = stdout if isinstance(stdout, bytes) else stdout.encode()
    return [f.decode("utf-8", errors="surrogateescape") for f in stdout.split(b"\0") if f]


def changed_paths(mirror_repo: Path, base_sha: str, head_sha: str) -> frozenset[str]:
    """Return every old/new path in a strict NUL-framed Git tree diff."""
    proc = git_process._run_git(
        mirror_repo,
        ["diff", "--name-status", "-z", "-M", base_sha, head_sha],
        retries=0,
        capture_bytes=True,
        timeout=30,
    )
    git_process._require_ok(proc, "git diff --name-status failed")
    raw = proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    if not raw:
        return frozenset()
    if not raw.endswith(b"\0"):
        raise git_ops.GitError("git diff --name-status returned a truncated NUL record")
    fields = raw[:-1].split(b"\0")
    if any(field == b"" for field in fields):
        raise git_ops.GitError("git diff --name-status returned an empty field")

    paths: set[str] = set()
    index = 0
    while index < len(fields):
        status = fields[index]
        if status[:1] in (b"R", b"C"):
            if (
                len(status) < 2
                or not status[1:].isdigit()
                or int(status[1:]) > 100
                or index + 2 >= len(fields)
            ):
                raise git_ops.GitError("git diff --name-status returned a malformed rename/copy record")
            record_paths = fields[index + 1:index + 3]
            index += 3
        elif status in (b"A", b"D", b"M", b"T", b"U", b"X", b"B"):
            if index + 1 >= len(fields):
                raise git_ops.GitError("git diff --name-status returned a malformed one-path record")
            record_paths = fields[index + 1:index + 2]
            index += 2
        else:
            raise git_ops.GitError("git diff --name-status returned an unsupported status record")
        paths.update(path.decode("utf-8", errors="surrogateescape") for path in record_paths)
    return frozenset(paths)


def _classify_diff(name_status: str | bytes, numstat: str | bytes) -> AnchorDiff:
    """Classify captured name-status and numstat bytes without probing Git. Renames/copies
    include both paths; deletions, modifications, and binary numstat markers populate
    their respective buckets.
    """
    fields = _nul_fields(name_status)
    # numstat -z records are NUL-terminated with tab-separated fields:
    # add, del, path (add/del are "-" for binary content).
    nfields = _nul_fields(numstat)
    binary: set[str] = set()
    for record in nfields:
        parts = record.split("\t", 2)
        if len(parts) == 3 and parts[0] == "-" and parts[1] == "-":
            binary.add(parts[2])
    renames: set[str] = set()
    modified: set[str] = set()
    deleted: set[str] = set()
    j = 0
    while j < len(fields):
        status = fields[j]
        if status.startswith(("R", "C")):
            if j + 2 >= len(fields):
                break
            renames.add(fields[j + 1])
            renames.add(fields[j + 2])
            j += 3
        else:
            if j + 1 >= len(fields):
                break
            path = fields[j + 1]
            if status == "D":
                deleted.add(path)
            else:
                modified.add(path)
            j += 2
    return AnchorDiff(
        renames=frozenset(renames),
        binary=frozenset(binary),
        modified=frozenset(modified),
        deleted=frozenset(deleted),
    )


def anchor_delta(
    mirror_repo: Path,
    base_tip: str,
    head: str,
    anchor: Any,
    diff_cache: dict[tuple[str, str], AnchorDiff] | None = None,
) -> str:
    """Classify local base..head changes against an anchor in base-commit coordinates.
    Undeclared locations return locationless. Check rename, deletion, binary, then
    base-side hunk intersection: only edits overlapping the anchor range are changed.
    Equal commits are unchanged without probes; Git failures are unavailable. Cache
    whole-tree classification per commit pair, while path-specific hunk checks retain
    each anchor range.
    """
    if not isinstance(anchor, dict) or anchor.get("status") != "derived":
        return "locationless"
    path = anchor.get("path")
    start = anchor.get("start_line")
    end = anchor.get("end_line")
    if not path or start is None or end is None:
        return "locationless"
    if base_tip == head:
        # An at-head anchor is by definition unmodified by the (empty)
        # base..head change — and the most common extraction case, so it must
        # never spawn git probes.
        return "unchanged"
    try:
        diff = diff_cache.get((base_tip, head)) if diff_cache is not None else None
        if diff is None:
            st = git_process._run_git(
                mirror_repo,
                ["diff", "--name-status", "-z", "-M", base_tip, head],
                retries=0,
                capture_bytes=True,
            )
            if st.returncode != 0:
                return "unavailable"
            numstat = git_process._run_git(
                mirror_repo,
                ["diff", "--numstat", "-z", base_tip, head],
                retries=0,
                capture_bytes=True,
            )
            if numstat.returncode != 0:
                return "unavailable"
            # Both streams are pure functions of (base_tip, head): compute the
            # classification once per distinct pair and share it across every
            # anchor on that pair (the -U0 probe below still runs per record).
            diff = _classify_diff(st.stdout, numstat.stdout)
            if diff_cache is not None:
                diff_cache[(base_tip, head)] = diff
        if path in diff.deleted:
            return "deleted"
        if path in diff.renames:
            return "renamed"
        if path in diff.binary:
            return "binary"
        if path not in diff.modified:
            return "unchanged"
        # Authoring coordinates are on the diff base: intersect -U0 base-side hunk
        # ranges, never new-side ranges that shift when preceding lines change.
        hunks = git_process._run_git(
            mirror_repo, ["diff", "-U0", base_tip, head, "--", path], retries=0,
        )
        if hunks.returncode != 0:
            return "unavailable"
        for line in hunks.stdout.splitlines():
            if not line.startswith("@@"):
                continue
            minus = line.split("+", 1)[0].strip()
            old = minus.rpartition(" ")[2]
            if not old.startswith("-"):
                continue
            c, _, d = old[1:].partition(",")
            cstart = int(c)
            clen = int(d) if d else 1
            if clen == 0:
                # A zero-length base-side range (a pure insertion) cannot
                # touch any existing authoring line, so the span is unaffected.
                continue
            cend = cstart + max(clen - 1, 0)
            if start <= cend and end >= cstart:
                return "changed"
        return "unchanged"
    except git_ops.GitError:
        return "unavailable"


def resolve_original_base(mirror_repo: Path, base_tip_ref: str, head_sha: str) -> str | None:
    """The merge-base of the base tip and head, or None when none exists."""
    proc = git_process._run_git(mirror_repo, ["merge-base", base_tip_ref, head_sha], retries=0)
    if proc.returncode == 1:
        # exit code 1 is git's documented signal for "no common ancestor" --
        # the soft-failure sentinel. Any other non-zero code is a real failure.
        return None
    git_process._require_ok(
        proc, f"git merge-base {base_tip_ref} {head_sha} failed in {mirror_repo}"
    )
    out = proc.stdout.strip()
    return out or None


class AnchorDerivationError(git_ops.GitError):
    """Closed history-unavailable/path-unavailable failure; callers must not guess an
    anchor.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def derive_authoring_path(mirror_repo: Path, authoring_sha: str, path: str, mapped_sha: str) -> str:
    """Resolve the path at the original authoring commit using the pinned mirror. Use the
    path directly if it exists there; otherwise require exactly one rename to the mapped
    path. Missing objects/diff failures are history-unavailable; ambiguous or absent
    renames are path-unavailable. NUL-delimited Git output preserves arbitrary path
    bytes.
    """
    verify = git_process._run_git(mirror_repo, ["rev-parse", "--verify", f"{authoring_sha}^{{commit}}"], retries=0)
    if verify.returncode != 0:
        raise AnchorDerivationError(
            "history-unavailable",
            f"authoring commit {authoring_sha[:12]} is absent from the mirror {mirror_repo}",
        )
    exists = git_process._run_git(mirror_repo, ["cat-file", "-e", f"{authoring_sha}:{path}"], retries=0)
    if exists.returncode == 0:
        return path
    trace = git_process._run_git(
        mirror_repo,
        ["diff", "--name-status", "-z", "-M", authoring_sha, mapped_sha],
        retries=0,
        capture_bytes=True,
    )
    if trace.returncode != 0:
        stderr = trace.stderr.decode("utf-8", errors="replace") if isinstance(trace.stderr, bytes) else trace.stderr
        raise AnchorDerivationError(
            "history-unavailable",
            f"git diff --name-status -M {authoring_sha} {mapped_sha} failed in {mirror_repo}: {stderr.strip()}",
        )
    # Capture NUL-framed bytes and surrogateescape-decode non-UTF-8 paths.
    fields = _nul_fields(trace.stdout)
    matches: list[str] = []
    i = 0
    while i < len(fields):
        status = fields[i]
        if status.startswith("R"):
            if i + 2 >= len(fields):
                break
            old, new = fields[i + 1], fields[i + 2]
            if new == path:
                matches.append(old)
            i += 3
        else:
            i += 2
    if len(matches) == 1:
        return matches[0]
    raise AnchorDerivationError(
        "path-unavailable",
        f"path {path!r} has no unique authoring-time name between "
        f"{authoring_sha[:12]} and {mapped_sha[:12]} (matches={len(matches)})",
    )


def resolve_trees(mirror_repo: Path, base_sha: str, head_sha: str) -> str | tuple[str, str]:
    """Peel ``^{tree}`` for both commits, or return ``"missing_object"``."""
    try:
        return (
            rev_parse(mirror_repo, f"{base_sha}^{{tree}}"),
            rev_parse(mirror_repo, f"{head_sha}^{{tree}}"),
        )
    except git_ops.GitError:
        return "missing_object"


def degenerate(base_tree: str, head_tree: str) -> str | None:
    """Classify a degenerate (empty) base/head change, or None when real."""
    if base_tree == head_tree:
        return "equal_trees"
    return None


def _canonical_diff_args(base: str, head: str) -> list[str]:
    """The canonical diff argv: digest-pinned refs with full (non-abbreviated) SHAs."""
    return ["-c", "core.abbrev=40", "diff", "--binary", base, head]


def canonical_diff_sha256(mirror_repo: Path, base_sha: str, head_sha: str) -> str:
    """sha256 of the canonical binary-safe diff between two commits."""
    proc = git_process._run_git(
        mirror_repo, _canonical_diff_args(base_sha, head_sha),
        retries=0, capture_bytes=True,
    )
    git_process._require_ok(proc, f"git diff --binary {base_sha} {head_sha} failed")
    return hashlib.sha256(proc.stdout).hexdigest()


def _synthetic_env() -> dict[str, str]:
    return {**os.environ, **_SYNTH_AUTHOR, "GIT_TERMINAL_PROMPT": "0"}


@overload
def _run_git_checked(
    repo: Path | str,
    args: list[str],
    *,
    env_cmd: dict[str, str] | None = None,
    timeout: int = 30,
    capture_bytes: Literal[False] = False,
) -> str: ...


@overload
def _run_git_checked(
    repo: Path | str,
    args: list[str],
    *,
    env_cmd: dict[str, str] | None = None,
    timeout: int = 30,
    capture_bytes: Literal[True],
) -> bytes: ...


def _run_git_checked(
    repo: Path | str,
    args: list[str],
    *,
    env_cmd: dict[str, str] | None = None,
    timeout: int = 30,
    capture_bytes: bool = False,
) -> str | bytes:
    """Run git and raise GitError on a non-zero exit."""
    repo = Path(repo)
    proc: subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]
    if capture_bytes:
        proc = git_process._run_git(
            repo, args, env_cmd=env_cmd, retries=0, timeout=timeout, capture_bytes=True
        )
    else:
        proc = git_process._run_git(
            repo, args, env_cmd=env_cmd, retries=0, timeout=timeout, capture_bytes=False
        )
    git_process._require_ok(proc, f"git {' '.join(args)} failed")
    if capture_bytes:
        return proc.stdout
    return proc.stdout.strip()


def build_bundle(
    mirror_repo: Path, base_sha: str, head_sha: str, bundle_path: Path
) -> None:
    """Build exactly base/head synthetic commits with pinned identity/time; Git failures
    propagate.
    """
    bundle_path = Path(bundle_path).resolve()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    env = _synthetic_env()
    base_tree = rev_parse(mirror_repo, f"{base_sha}^{{tree}}")
    head_tree = rev_parse(mirror_repo, f"{head_sha}^{{tree}}")

    base_commit = _run_git_checked(
        mirror_repo, ["commit-tree", base_tree, "-m", "snapshot base"], env_cmd=env, timeout=30
    )
    head_commit = _run_git_checked(
        mirror_repo, ["commit-tree", head_tree, "-p", base_commit, "-m", "snapshot head"],
        env_cmd=env, timeout=30,
    )
    for ref, sha in (("refs/heads/base", base_commit), ("refs/heads/head", head_commit)):
        _run_git_checked(mirror_repo, ["update-ref", ref, sha], env_cmd=env, timeout=30)
    _run_git_checked(
        mirror_repo,
        ["bundle", "create", str(bundle_path), "refs/heads/base", "refs/heads/head"],
        env_cmd=env, timeout=120,
    )


def bundle_heads(bundle_path: Path) -> set[str]:
    """The list of refs a bundle exposes."""
    bundle_path = Path(bundle_path).resolve()
    proc = git_process._run_git(
        bundle_path.parent, ["bundle", "list-heads", str(bundle_path)], retries=0, timeout=30
    )
    git_process._require_ok(proc, "git bundle list-heads failed")
    heads: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            heads.add(parts[1])
    return heads


def validate_offline_clone(
    bundle_path: Path, base_tree: str, head_tree: str, diff_sha256: str, workdir: Path
) -> None:
    """Clone with network disabled and verify the full portable fidelity contract. Require
    exactly origin/base and origin/head refs, exactly two reachable commits, a root
    base, head parented only on base, matching tree ids, and the canonical diff digest.
    Any mismatch or clone error raises GitError.
    """
    bundle_path = Path(bundle_path).resolve()
    import tempfile

    clone_dir = Path(tempfile.mkdtemp(prefix="clone-", dir=str(workdir)))
    try:
        proc = git_process._run_git(
            Path(workdir), ["clone", "--no-local", "--no-checkout", str(bundle_path), str(clone_dir)],
            retries=0, timeout=120,
        )
        git_process._require_ok(proc, f"offline clone of {bundle_path} failed")
        refs_out = _run_git_checked(clone_dir, ["for-each-ref", "--format=%(refname)", "refs/remotes"])
        refs = set(refs_out.splitlines())
        expected_refs = {"refs/remotes/origin/base", "refs/remotes/origin/head"}
        if refs != expected_refs:
            raise git_ops.GitError(
                f"offline clone exposes unexpected refs (expected {sorted(expected_refs)}, "
                f"got {sorted(refs)})"
            )
        count = _run_git_checked(clone_dir, ["rev-list", "--count", "refs/remotes/origin/head"])
        if count != "2":
            raise git_ops.GitError(
                f"offline clone head ancestry must contain exactly two reachable commits "
                f"(got {count})"
            )
        parents_out = _run_git_checked(
            clone_dir, ["rev-list", "--parents", "refs/remotes/origin/base"]
        )
        if len(parents_out.splitlines()) != 1:
            raise git_ops.GitError(
                f"offline clone base must be a root commit with no parent "
                f"(rev-list --parents base yielded {len(parents_out.splitlines())} commits)"
            )
        head_parent = _run_git_checked(
            clone_dir, ["rev-parse", "--verify", "refs/remotes/origin/head^"]
        )
        base_commit = _run_git_checked(clone_dir, ["rev-parse", "--verify", "refs/remotes/origin/base"])
        if head_parent != base_commit:
            raise git_ops.GitError(
                f"offline clone head's parent must be the base commit "
                f"(expected {base_commit}, got {head_parent})"
            )
        for ref, expected in (
            ("refs/remotes/origin/base", base_tree),
            ("refs/remotes/origin/head", head_tree),
        ):
            got = _run_git_checked(clone_dir, ["rev-parse", "--verify", f"{ref}^{{tree}}"])
            if got != expected:
                raise git_ops.GitError(f"offline clone tree mismatch for {ref} (expected {expected}, got {got})")
        diff = _run_git_checked(
            clone_dir,
            _canonical_diff_args("refs/remotes/origin/base", "refs/remotes/origin/head"),
            capture_bytes=True,
        )
        if hashlib.sha256(diff).hexdigest() != diff_sha256:
            raise git_ops.GitError(f"offline clone diff digest mismatch (case {bundle_path})")
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)
    return None


def freeze_one(
    root: Path,
    repo_slug: str,
    pr_number: int,
    *,
    base_tip: str,
    head_sha: str,
    policy: str,
    requested_head: str,
    pr_changed_files: AbstractSet[str],
    origin_url: str | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    """Return a ready/unreplayable snapshot plus optional bundle bytes. Classified Git
    failures become exact unreplayable reasons; unexpected errors propagate. Bundle
    bytes stay in private scratch until the caller journal commits them, preventing
    untracked final bundles after interruption.
    """
    root = Path(root)
    origin_url = origin_url or f"https://github.com/{repo_slug}.git"
    case_id = schema.case_id_for(pr_number, head_sha)
    bundle_rel = f"snapshots/{case_id}.bundle"

    # The resolved merge base, recorded on any unreplayable dict produced after
    # merge-base resolution (None for the earlier fetch/ancestry failures).
    resolved_base: str | None = None

    def unreplayable(reason: str, detail: str) -> tuple[dict[str, Any], None]:
        record = schema.SnapshotUnreplayable(
            status="unreplayable",
            policy=cast(Literal["final_pr_head", "explicit_head"], policy),
            requested_head=requested_head,
            original_base_sha=resolved_base,
            requested_base_sha=base_tip,
            original_head_sha=head_sha,
            error=schema._SnapshotError.model_validate({"reason": reason, "detail": detail}),
        )
        return record.model_dump(), None

    # 1) establish the shared bare mirror (local-only, no network). A failure
    #    here is a base-side (environment) problem: no ref on either side can be
    #    sourced, so it classifies ``base_unreachable``.
    try:
        m = ensure_mirror(root)
    except git_ops.GitError as exc:
        return unreplayable(
            "base_unreachable", f"could not establish the shared bare mirror: {exc}"
        )
    # 2) fetch the selected base tip (its own failure reason) and then the PR
    #    head + explicit heads (their own failure reason) into the mirror
    #    (credential/network/timeout or a missing refspec on the remote).
    try:
        fetch_base_tip(root, repo_slug, base_tip, origin_url)
    except git_ops.GitError as exc:
        return unreplayable(
            "base_unreachable", f"could not fetch the base tip from the origin: {exc}"
        )
    try:
        fetch_head_refs(root, repo_slug, pr_number, [head_sha], origin_url)
    except git_ops.GitError as exc:
        return unreplayable(
            "head_unreachable", f"could not fetch the PR head refs from the origin: {exc}"
        )
    # 3) the PR head ref must resolve after a successful fetch.
    try:
        pr_head = rev_parse(m, f"refs/pull/{pr_number}/head")
    except git_ops.GitError as exc:
        return unreplayable(
            "head_unreachable", f"could not resolve the PR head ref after fetching: {exc}"
        )

    # 4) the requested head must be the PR head or an ancestor on its ancestry.
    reach = head_reachability(m, head_sha, pr_head)
    if reach == "head_unreachable":
        return unreplayable("head_unreachable", f"requested head {head_sha[:12]} is not in the mirror")
    if reach != "ok":
        return unreplayable(
            "head_not_on_pr",
            f"requested head {head_sha[:12]} is reachable elsewhere but not on the PR head",
        )

    # 5) resolve the merge base and both trees.
    base = resolve_original_base(m, "refs/heads/base_tip", head_sha)
    resolved_base = base
    if base is None:
        return unreplayable("base_unreachable", "no merge-base could be resolved for the sourced base tip and head")
    trees = resolve_trees(m, base, head_sha)
    if not isinstance(trees, tuple):
        return unreplayable(trees, "a source tree object is absent from the mirror")
    base_tree, head_tree = trees

    # 6) a clean review still requires a real code change.
    degen = degenerate(base_tree, head_tree)
    if degen is not None:
        return unreplayable(degen, f"no real code change between base and head ({degen})")

    if policy == "explicit_head":
        try:
            extra_paths = sorted(changed_paths(m, base, head_sha) - pr_changed_files)
        except git_ops.GitError as exc:
            return unreplayable(
                "bundle_failure", f"snapshot path-inventory probe failed: {exc}"
            )
        if extra_paths:
            preview = ", ".join(repr(path) for path in extra_paths[:20])
            suffix = "" if len(extra_paths) <= 20 else f", and {len(extra_paths) - 20} more"
            return unreplayable(
                "base_drift",
                f"snapshot contains {len(extra_paths)} path(s) outside PR inventory: {preview}{suffix}",
            )

    # Build and validate the deterministic bundle in private scratch, then clean it.
    # Only the caller's journaled Transaction writes the final snapshots/ path.
    scratch_bundle = root / "cache" / "freeze-scratch" / f"{case_id}.bundle"
    try:
        diff_sha = canonical_diff_sha256(m, base, head_sha)
        build_bundle(m, base, head_sha, scratch_bundle)
        bundle_sha = storage.sha256_file(scratch_bundle)
        validate_offline_clone(scratch_bundle, base_tree, head_tree, diff_sha, workdir=root / "cache")
        bundle_bytes = scratch_bundle.read_bytes()
    except (git_ops.GitError, OSError) as exc:
        # A raw file-I/O error (storage.sha256_file / read_bytes) is likewise a
        # per-PR bundle failure and must never escape to abort the whole import.
        return unreplayable("bundle_failure", f"bundle build/validate failed: {exc}")
    finally:
        scratch_bundle.unlink(missing_ok=True)

    ready = {
        "status": "ready",
        "base_resolution": "merge_base_v1",
        "policy": policy,
        "requested_head": requested_head,
        # original_base_sha is the true merge base of the selected base tip and
        # the head; requested_base_sha is the selected base-branch tip.
        "original_base_sha": base,
        "requested_base_sha": base_tip,
        "original_head_sha": head_sha,
        "base_tree_sha": base_tree,
        "head_tree_sha": head_tree,
        "diff_sha256": diff_sha,
        "bundle_file": bundle_rel,
        "bundle_sha256": bundle_sha,
        "error": None,
    }
    return ready, bundle_bytes
