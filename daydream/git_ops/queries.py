"""Git repository discovery, reference resolution, and diff queries."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from daydream.git_ops import process
from daydream.git_ops.models import BranchNotFoundError, GitError, GitTimeoutError, NotAWorktreeError, PathAbsentError
from daydream.repository_paths import git_observed_path_is_confined

# --- Pre-flight --------------------------------------------------------------


def assert_is_worktree(repo: Path) -> None:
    """Require the worktree root, rejecting absent paths and nested repository directories."""
    if not repo.exists() or not repo.is_dir():
        raise NotAWorktreeError(f"{repo} is not a directory")

    proc = process._run_git(repo, ["rev-parse", "--is-inside-work-tree"], timeout=5)
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise NotAWorktreeError(f"{repo} is not inside a git worktree")

    top_proc = process._run_git(repo, ["rev-parse", "--show-toplevel"], timeout=5)
    if top_proc.returncode != 0:
        raise NotAWorktreeError(f"{repo} could not resolve its worktree top-level")

    top = Path(top_proc.stdout.strip()).resolve()
    if top != repo.resolve():
        raise NotAWorktreeError(
            f"{repo} is inside a worktree but its top-level is {top}; pass the worktree root instead",
        )


def is_inside_worktree(repo: Path) -> bool:
    """Return True iff :func:`assert_is_worktree` would succeed for *repo*."""
    try:
        assert_is_worktree(repo)
    except NotAWorktreeError:
        return False
    return True


# --- Read-only queries -------------------------------------------------------


def head_sha(repo: Path) -> str:
    """Resolve the full HEAD SHA; empty repositories and query failures raise ``GitError``."""
    proc = process._run_git(repo, ["rev-parse", "HEAD"], timeout=5, error_context=f"cannot resolve HEAD in {repo}")
    return proc.stdout.strip()


def has_executable_pre_push_hook(repo: Path) -> bool:
    """Check the executable pre-push hook, honoring Git's configured hooks path.

    If Git cannot resolve that path, inspect ``<repo>/.git/hooks``. Missing files or
    non-executable hooks return ``False``.
    """
    proc = process._run_git(repo, ["rev-parse", "--git-path", "hooks"], timeout=5)
    if proc.returncode != 0:
        # Not a resolvable git repository: the canonical default location is
        # still checked so "no executable hook" stays a plain False answer.
        hooks_dir = repo / ".git" / "hooks"
    else:
        hooks_dir = Path(proc.stdout.strip())
        if not hooks_dir.is_absolute():
            hooks_dir = repo / hooks_dir
    hooks_path = hooks_dir / "pre-push"
    return hooks_path.is_file() and os.access(hooks_path, os.X_OK)


def list_local_branches(repo: Path) -> dict[str, str]:
    """Map local branch names to full OIDs; only a successful empty query returns no branches."""
    proc = process._run_git(
        repo,
        ["for-each-ref", "refs/heads", "--format=%(refname) %(objectname)"],
        timeout=10,
        error_context=f"cannot list local branches in {repo}",
    )
    branches: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        name, _, oid = line.strip().partition(" ")
        if not name.startswith("refs/heads/"):
            raise GitError(f"invalid local branch reference in {repo}")
        branches[name.removeprefix("refs/heads/")] = oid
    return branches


def remote_url(repo: Path, remote: str = "origin") -> str | None:
    """Read a configured fetch URL; missing values and Git failures return ``None``."""
    try:
        proc = process._run_git(repo, ["config", "--get", f"remote.{remote}.url"], timeout=5)
    except GitError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def remote_urls(repo: Path) -> dict[str, str]:
    """Return every configured remote's nonempty fetch URL.

    An empty repository remote set is valid. Any enumeration or per-remote URL
    failure is hard because PR base selection must not silently ignore a
    partially readable remote configuration.
    """
    proc = process._run_git(
        repo,
        ["config", "--null", "--name-only", "--get-regexp", r"^remote\..*\."],
        timeout=5,
    )
    if proc.returncode not in (0, 1):
        raise GitError(f"cannot enumerate remotes in {repo}: {proc.stderr.strip()}")
    names: set[str] = set()
    for key in proc.stdout.split("\0"):
        match = re.fullmatch(r"remote\.(.+)\.[^.]+", key)
        if match is not None:
            names.add(match.group(1))
    result: dict[str, str] = {}
    for name in sorted(names):
        url_proc = process._run_git(repo, ["config", "--get", f"remote.{name}.url"], timeout=5)
        url = url_proc.stdout.strip() if url_proc.returncode == 0 else ""
        if not url:
            raise GitError(f"remote {name!r} has no readable fetch URL in {repo}")
        result[name] = url
    return result


def validate_branch_name(repo: Path, name: str) -> None:
    """Raise when *name* is not a literal Git branch name."""
    proc = process._run_git(repo, ["check-ref-format", "--branch", name], timeout=5)
    if proc.returncode != 0:
        display = name[:120] + ("..." if len(name) > 120 else "")
        raise GitError(f"invalid PR base branch name {display!r} in {repo}")


def current_branch(repo: Path) -> str | None:
    """Read the current branch, returning ``None`` for detached HEAD and raising on failure."""
    proc = process._run_git(
        repo, ["branch", "--show-current"], timeout=5, error_context=f"cannot read current branch in {repo}"
    )
    name = proc.stdout.strip()
    return name or None


def default_branch(repo: Path) -> str:
    """Resolve ``origin/HEAD``, then local ``main``, then local ``master``.

    Raise ``BranchNotFoundError`` if none exists.
    """
    sym = process._run_git(repo, ["symbolic-ref", "refs/remotes/origin/HEAD"], timeout=5)
    if sym.returncode == 0 and sym.stdout.strip():
        return sym.stdout.strip().rsplit("/", 1)[-1]

    for candidate in ("main", "master"):
        check = process._run_git(repo, ["rev-parse", "--verify", f"refs/heads/{candidate}"], timeout=5)
        if check.returncode == 0:
            return candidate

    raise BranchNotFoundError(f"no default branch (origin/HEAD, main, master) found in {repo}")


def branch_exists(repo: Path, ref: str) -> bool:
    """Check whether *ref* exists locally or as ``origin/<ref>``."""
    local = process._run_git(repo, ["rev-parse", "--verify", f"refs/heads/{ref}"], timeout=5)
    if local.returncode == 0:
        return True
    remote = process._run_git(repo, ["rev-parse", "--verify", f"refs/remotes/origin/{ref}"], timeout=5)
    return remote.returncode == 0


def _has_leading_dash(*refs: str) -> bool:
    """Reject commit-ish inputs that Git would interpret as option flags."""
    return any(ref.startswith("-") for ref in refs)


def ref_exists(repo: Path, ref: str) -> bool:
    """Probe a local commit-ish, with an ``origin/<ref>`` fallback for branch names.

    Missing refs and leading-dash inputs return ``False``; subprocess failures raise.
    """
    if branch_exists(repo, ref):
        return True
    return commit_exists(repo, ref)


def commit_exists(repo: Path, revision: str) -> bool:
    """Probe ``revision^{commit}`` using Git's normal local revision resolution.

    No implicit ``origin/`` fallback. Missing refs, non-commit objects, and leading-dash
    inputs return ``False``; subprocess failures raise.
    """
    # Same guard as ref_exists: no valid commit-ish starts with '-'.
    if _has_leading_dash(revision):
        return False
    commit = process._run_git(repo, ["rev-parse", "--verify", f"{revision}^{{commit}}"], timeout=5)
    return commit.returncode == 0


def is_ancestor(repo: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    """Probe ancestry; missing, unrelated, or leading-dash refs return ``False``.

    Subprocess failures still raise ``GitError``.
    """
    # Same guard as ref_exists and merge_base: a leading '-' would be
    # mis-parsed by git as an option flag. No valid branch name or commit-ish
    # starts with '-'.
    if _has_leading_dash(ancestor, descendant):
        return False

    proc = process._run_git(repo, ["merge-base", "--is-ancestor", ancestor, descendant], timeout=5)
    return proc.returncode == 0


def merge_base(repo: Path, base: str, head: str = "HEAD") -> str | None:
    """Find the merge-base, preferring an upstream with commits ahead of local base.

    Follows the algorithm in ``codex-rs/git-utils/src/branch.rs`` to avoid stale local
    bases. Missing refs or empty repositories return ``None``; subprocess failures raise.
    """
    # Same guard as ref_exists: a leading '-' would be mis-parsed as a git
    # option flag.  No valid branch name or commit-ish starts with '-'.
    if _has_leading_dash(head, base):
        return None

    head_proc = process._run_git(repo, ["rev-parse", "--verify", head], timeout=5)
    if head_proc.returncode != 0:
        return None

    base_proc = process._run_git(repo, ["rev-parse", "--verify", base], timeout=5)
    if base_proc.returncode != 0:
        return None
    preferred_ref = base

    upstream, ahead = _upstream_and_ahead(repo, base)
    if upstream is not None and ahead > 0:
        preferred_ref = upstream

    mb = process._run_git(repo, ["merge-base", head, preferred_ref], timeout=5)
    if mb.returncode != 0:
        return None
    out = mb.stdout.strip()
    return out or None


def _upstream_and_ahead(repo: Path, branch: str) -> tuple[str | None, int]:
    """Return the symbolic upstream and its right-side ahead count; unreadable or absent
    upstream/count yields (None, 0).
    """
    upstream_name_proc = process._run_git(
        repo,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", f"{branch}@{{upstream}}"],
        timeout=5,
    )
    if upstream_name_proc.returncode != 0:
        return None, 0
    upstream = upstream_name_proc.stdout.strip()
    if not upstream:
        return None, 0

    counts_proc = process._run_git(
        repo,
        ["rev-list", "--left-right", "--count", f"{branch}...{upstream}"],
        timeout=5,
    )
    if counts_proc.returncode != 0:
        return None, 0
    parts = counts_proc.stdout.strip().split()
    try:
        right = int(parts[1]) if len(parts) >= 2 else 0
    except ValueError:
        right = 0
    return upstream, right


def _prefer_remote_base(repo: Path, base: str) -> str:
    """Return ``origin/<base>`` when it exists, otherwise *base* unchanged."""
    remote_ref = f"origin/{base}"
    check = process._run_git(repo, ["rev-parse", "--verify", f"refs/remotes/{remote_ref}"], timeout=5)
    if check.returncode == 0:
        return remote_ref
    return base


def _validated_diff_object_id(stdout: str) -> str | None:
    """Return one canonical SHA-1 line, rejecting every other representation."""
    if re.fullmatch(r"[0-9a-f]{40}\n?", stdout) is None:
        return None
    return stdout.removesuffix("\n")


def resolve_diff_merge_base(repo: Path, base: str, head_sha: str) -> str:
    """Pin both input commits before computing the same merge-base that ``diff`` uses.

    Prefer a present ``origin/<base>`` even without an upstream relationship. Resolving
    immutable OIDs first prevents concurrent ref moves from changing the second probe.
    Unresolvable commits or merge-base raise ``GitError``.
    """
    if _has_leading_dash(base, head_sha):
        raise GitError("branch-focus diff base or head is invalid")

    try:
        preferred_base = _prefer_remote_base(repo, base)
    except (GitError, UnicodeError) as exc:
        raise GitError("branch-focus preferred base probe failed") from exc

    def _commit_oid(ref: str, label: str) -> str:
        try:
            proc = process._run_git(
                repo,
                ["rev-parse", "--verify", f"{ref}^{{commit}}"],
                timeout=5,
            )
        except (GitError, UnicodeError) as exc:
            raise GitError(f"branch-focus {label} commit probe failed") from exc
        oid = _validated_diff_object_id(proc.stdout) if proc.returncode == 0 else None
        if oid is None:
            raise GitError(f"branch-focus {label} commit cannot be resolved")
        return oid

    resolved_head = _commit_oid(head_sha, "recorded head")
    resolved_base = _commit_oid(preferred_base, "preferred base")
    try:
        proc = process._run_git(repo, ["merge-base", resolved_head, resolved_base], timeout=5)
    except (GitError, UnicodeError) as exc:
        raise GitError("branch-focus diff merge-base probe failed") from exc
    merge_base_sha = _validated_diff_object_id(proc.stdout) if proc.returncode == 0 else None
    if merge_base_sha is None:
        raise GitError("branch-focus diff merge-base cannot be resolved")
    return merge_base_sha


def diff(repo: Path, base: str, head: str = "HEAD", *, exclude: list[str] | None = None) -> str:
    """Return the three-dot commit diff plus tracked working-tree changes.

    Prefer a present ``origin/<base>`` over local base. ``exclude`` supplies pathspec
    exclusions; failed queries raise ``GitError``.
    """
    preferred_base = _prefer_remote_base(repo, base)
    paths = ["--", ".", *(f":(exclude){p.rstrip('/')}" for p in exclude)] if exclude else []
    # Retain committed and tracked working-tree changes so resume fingerprints see both.
    result = []
    for revision, label in ((f"{preferred_base}...{head}", f"{base}...{head}"), (head, head)):
        proc = process._run_git(
            repo, ["diff", revision, *paths], timeout=30, error_context=f"git diff {label} failed"
        )
        result.append(proc.stdout)
    return "".join(result)


def diff_name_only(repo: Path, base: str, head: str = "HEAD") -> list[str]:
    """List paths differing directly between the two refs, in Git output order.

    The archive uses this two-dot query best-effort: missing refs and subprocess
    failures return ``[]``.
    """
    try:
        proc = process._run_git(repo, ["diff", "--name-only", f"{base}..{head}"], timeout=10)
    except GitError:
        return []
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line]


def diff_paths(
    repo: Path,
    base: str,
    head: str,
    paths: list[str],
    *,
    unified: int = 3,
    merge_base_diff: bool = False,
) -> str:
    """Diff only named paths with explicit context, raising on failure.

    Use a direct two-dot range by default (needed for PR comment line resolution);
    ``merge_base_diff`` selects a three-dot range instead.
    """
    sep = "..." if merge_base_diff else ".."
    range_arg = f"{base}{sep}{head}"
    args = ["diff", f"--unified={unified}", range_arg, "--", *paths]
    proc = process._run_git(repo, args, timeout=30, error_context=f"git diff {range_arg} failed")
    return proc.stdout


def diff_worktree_against(repo: Path, ref: str, paths: list[str]) -> str:
    """Capture named working-tree paths against a ref before restoring partial edits.

    Git omits paths untracked at the ref. Query failures raise ``GitError``.
    """
    if not paths:
        return ""
    args = ["diff", ref, "--", *paths]
    proc = process._run_git(
        repo, args, timeout=30, retries=0, error_context=f"git diff {ref} -- {paths} failed in {repo}"
    )
    return proc.stdout


def log(repo: Path, base: str, head: str = "HEAD") -> str:
    """Return stripped one-line commit history for ``base..head``, raising on query failure."""
    proc = process._run_git(
        repo, ["log", f"{base}..{head}", "--oneline"], timeout=30, error_context=f"git log {base}..{head} failed"
    )
    return proc.stdout.strip()


def _log_shas_range(
    repo: Path,
    range_from: str,
    range_to: str,
    *,
    label: str,
    failure_value: list[str] | None,
    failure_reason: str,
) -> list[str] | None:
    """Read a commit window with the caller-defined failure sentinel and a contextual
    warning. An unavailable query must remain distinguishable from a successful empty
    walk.
    """
    rev_range = f"{range_from}..{range_to}"
    try:
        proc = process._run_git(repo, ["log", "--pretty=%H", rev_range], timeout=30)
    except GitTimeoutError:
        failure = "timed out after retries"
    except GitError as exc:
        failure = f"failed: {exc}"
    else:
        if proc.returncode == 0:
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        failure = f"exited non-zero ({proc.returncode})"
    process._logger.warning("%s: git log %s %s; %s", label, rev_range, failure, failure_reason)
    return failure_value


def log_shas(repo: Path, ref: str, *, since: str) -> list[str] | None:
    """List full SHAs in ``since..ref``, newest first.

    A successful empty walk returns ``[]``; an unanswerable query returns ``None`` so
    callers cannot mistake a deleted ref for evidence that no follow-up commits exist.
    """
    return _log_shas_range(
        repo,
        since,
        ref,
        label="log_shas",
        failure_value=None,
        failure_reason="returning None (commit window unavailable)",
    )


def log_shas_since(repo: Path, head: str, base: str) -> list[str]:
    """List full SHAs in ``head..base``, newest first; warn and return ``[]`` on failure.

    The revision range bounds the walk without a redundant date filter.
    """
    shas = _log_shas_range(
        repo,
        head,
        base,
        label="log_shas_since",
        failure_value=[],
        failure_reason="returning empty window (fix-applied verdict may degrade to unknown)",
    )
    return shas if shas is not None else []


def daydream_commits(repo: Path, base: str, head: str = "HEAD") -> str | None:
    """Return the stripped oneline log of daydream commits in ``base..head``.

    Return None for no matches or a Git failure (logged as a warning).
    """
    proc = process._run_git(
        repo,
        ["log", f"{base}..{head}", "--oneline", "--grep=Daydream-Run:"],
        timeout=30,
    )
    if proc.returncode != 0:
        process._logger.warning(
            "git log %s..%s --grep=Daydream-Run: failed (rc=%d): %s",
            base,
            head,
            proc.returncode,
            (proc.stderr or "").strip(),
        )
        return None
    output = proc.stdout.strip()
    return output or None


_GIT_PATH_ABSENT_RE = re.compile(r"does not exist in|exists on disk, but not in")


def show(repo: Path, ref: str, path: str) -> bytes:
    """Read raw path bytes at a ref.

    Raise ``PathAbsentError`` only when Git's diagnostic proves absence. Timeouts,
    damaged objects, and unrecognized failures remain ``GitError``.
    """
    proc = process._run_git(repo, ["show", f"{ref}:{path}"], timeout=30, capture_bytes=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
        message = f"git show {ref}:{path} failed: {stderr.strip()}"
        if _GIT_PATH_ABSENT_RE.search(stderr):
            raise PathAbsentError(message)
        raise GitError(message)
    return proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()


def grep_fixed_matches(
    repo: Path,
    patterns: Sequence[str],
    *,
    word: bool = False,
    pathspecs: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Return one ``(path, matched_pattern)`` per occurrence from batched ``git grep``.

    Deduplicate patterns in input order, skipping empty or NUL/CR/LF-containing values.
    ``word`` requires word boundaries; absent ``pathspecs`` searches all tracked files.
    Exit 1 returns no matches; malformed records or other failures raise ``GitError``.
    """
    seen: set[str] = set()
    normalized: list[str] = []
    for pattern in patterns:
        if not pattern or pattern in seen:
            continue
        if "\x00" in pattern or "\r" in pattern or "\n" in pattern:
            continue
        seen.add(pattern)
        normalized.append(pattern)
    if not normalized:
        return []

    tmp = tempfile.NamedTemporaryFile(delete=False)
    proc: subprocess.CompletedProcess[Any] | None = None
    try:
        with tmp:
            tmp.write(b"\n".join(os.fsencode(p) for p in normalized))
        args = ["grep", "--no-color", "-F", "-o", "-z", "-f", tmp.name]
        if word:
            args.append("-w")
        if pathspecs:
            args.append("--")
            args.extend(pathspecs)
        proc = process._run_git(repo, args, timeout=30, capture_bytes=True)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if proc is None:  # pragma: no cover - _run_git returns or raises
        raise GitError("git grep -F -o -z -f failed")
    if proc.returncode == 1:
        return []
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
        raise GitError(f"git grep -F -o -z -f failed: {stderr.strip()}")

    stdout = proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    matches: list[tuple[str, str]] = []
    for line in stdout.split(b"\n"):
        if not line:
            continue
        parts = line.split(b"\x00", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise GitError("git grep -F -o -z -f returned a malformed record")
        matches.append(
            (
                parts[0].decode("utf-8", errors="surrogateescape"),
                parts[1].decode("utf-8", errors="surrogateescape"),
            )
        )
    return matches


def status_porcelain(repo: Path) -> str:
    """Read porcelain status, raising on failure; a clean tree returns empty text."""
    proc = process._run_git(repo, ["status", "--porcelain"], timeout=10, error_context=f"git status failed in {repo}")
    return proc.stdout


def staged_patch(repo: Path) -> bytes:
    """Read the staged index as a binary patch, preserving exact bytes.

    Failures raise ``GitError`` with stderr; patch content never enters the error text.
    """
    proc = process._run_git(repo, ["diff", "--cached", "--binary"], timeout=30, capture_bytes=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
        raise GitError(f"git diff --cached --binary failed in {repo}: {stderr.strip()}")
    return proc.stdout


def _ordered_unique_names(lines: list[str]) -> list[str]:
    """Return *lines* stripped, with blanks and later duplicates removed."""
    return list(dict.fromkeys(name for line in lines if (name := line.strip())))


def changed_files(repo: Path, *, preexisting_untracked: set[str] | None = None) -> list[str]:
    """Combine tracked changes against HEAD with new untracked paths, in first-seen order.

    Exclude the pre-fix untracked snapshot so existing user files are not attributed to
    Daydream. Each failed subquery contributes no paths while the other still runs.
    """
    try:
        proc = process._run_git(repo, ["diff", "--name-only", "HEAD"], timeout=10)
        tracked = proc.stdout.splitlines() if proc.returncode == 0 else []
    except GitError:
        tracked = []
    untracked = _filter_preexisting_untracked(list_untracked(repo), preexisting_untracked)
    return _ordered_unique_names([*tracked, *untracked])


def changed_files_against(
    repo: Path,
    ref: str,
    *,
    preexisting_untracked: set[str] | None = None,
) -> list[str]:
    """Return paths changed from *ref*, raising when they cannot be enumerated.

    This is the strict counterpart to :func:`changed_files` for destructive
    recovery guards, where treating a Git failure as an empty change set would
    make the guard's safety decision unreliable.
    """
    proc = process._run_git(
        repo, ["diff", "--name-only", ref], timeout=10, error_context=f"git diff --name-only {ref} failed in {repo}"
    )
    untracked_proc = process._run_git(
        repo,
        ["ls-files", "--others", "--exclude-standard"],
        timeout=10,
        error_context=f"git ls-files --others failed in {repo}",
    )

    untracked = [line.strip() for line in untracked_proc.stdout.splitlines() if line.strip()]
    untracked = _filter_preexisting_untracked(untracked, preexisting_untracked)
    return _ordered_unique_names([*proc.stdout.splitlines(), *untracked])


def diff_name_only_strict(repo: Path, from_ref: str, to_ref: str) -> list[str]:
    """Read exact NUL-delimited paths differing between two commit trees.

    Failure raises so destructive guards cannot mistake an unknown result for no
    changes. Filesystem decoding preserves names without trimming or Git quoting.
    """
    proc = process._run_git(
        repo,
        ["diff", "--name-only", "-z", from_ref, to_ref],
        timeout=10,
        capture_bytes=True,
        error_context=f"git diff --name-only -z {from_ref} {to_ref} failed in {repo}",
    )
    return _decode_nul_paths(proc.stdout)


def list_untracked(repo: Path, *, strict: bool = False) -> list[str]:
    """List untracked, non-ignored paths; Git errors return ``[]`` unless ``strict``.

    Strict enumeration protects snapshots from silently omitting unknown content.
    """
    try:
        proc = process._run_git(repo, ["ls-files", "--others", "--exclude-standard"], timeout=10)
    except GitError:
        if strict:
            raise
        return []
    if proc.returncode != 0:
        if strict:
            raise GitError(f"git ls-files --others --exclude-standard failed in {repo}: {proc.stderr.strip()}")
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _filter_preexisting_untracked(untracked: list[str], preexisting_untracked: set[str] | None) -> list[str]:
    """Exclude pre-existing user files without reordering; ``None`` leaves the list unchanged."""
    if preexisting_untracked is None:
        return untracked
    return [path for path in untracked if path not in preexisting_untracked]


def _decode_nul_paths(stdout: bytes) -> list[str]:
    """Decode exact NUL-delimited Git paths with filesystem round-tripping."""
    return [os.fsdecode(raw) for raw in stdout.split(b"\0") if raw]


def _path_sort_key(path: str) -> bytes:
    return os.fsencode(path)


def _literal_pathspec(path: str) -> str:
    return f":(literal){path}"


def _require_git_path_confined(repo: Path, path: str, *, allow_leaf_symlink: bool = False) -> None:
    if not git_observed_path_is_confined(repo, path, allow_leaf_symlink=allow_leaf_symlink):
        raise GitError("Git-observed path is not confined to the repository")


def _is_untracked_runtime_artifact(path: str) -> bool:
    from daydream.config import REVIEW_OUTPUT_FILE

    return path.startswith(".daydream/") or path == REVIEW_OUTPUT_FILE


def _list_untracked_z(repo: Path) -> list[str]:
    """Return NUL-delimited untracked paths, raising on a git failure."""
    proc = process._run_git(
        repo,
        ["ls-files", "--others", "--exclude-standard", "-z"],
        timeout=10,
        capture_bytes=True,
        error_context=f"git ls-files --others -z failed in {repo}",
    )
    return _decode_nul_paths(proc.stdout)


def changed_paths_z(
    repo: Path,
    ref: str,
    *,
    include_untracked: bool = True,
    include_runtime_artifacts: bool = True,
) -> list[str]:
    """Strictly enumerate paths changed from *ref* using NUL delimiters.

    Fix evidence may exclude untracked runtime output. Tracked changes are
    always included, even inside the runtime namespace.
    """
    proc = process._run_git(repo, ["diff", "--name-only", "-z", ref], timeout=10, capture_bytes=True)
    if proc.returncode != 0:
        stderr = os.fsdecode(proc.stderr)
        raise GitError(f"git diff --name-only -z {ref} failed in {repo}: {stderr.strip()}")
    paths = _decode_nul_paths(proc.stdout)
    if include_untracked:
        paths.extend(
            path
            for path in _list_untracked_z(repo)
            if include_runtime_artifacts or not _is_untracked_runtime_artifact(path)
        )
    unique = dict.fromkeys(paths)
    for path in unique:
        _require_git_path_confined(repo, path, allow_leaf_symlink=True)
    return list(unique)


def ls_files(repo: Path, *, strict: bool = False) -> list[str]:
    """Read tracked paths with NUL separation and filesystem decoding.

    Newlines and invalid UTF-8 survive intact. A nonzero exit returns ``[]`` unless
    ``strict``; subprocess failures always propagate so callers can distinguish an
    empty repository from an unanswerable query.
    """
    proc = process._run_git(repo, ["ls-files", "-z"], capture_bytes=True)
    if proc.returncode != 0:
        if strict:
            stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
            raise GitError(f"git ls-files failed in {repo}: {stderr.strip()}")
        return []
    stdout = proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    return [path.decode("utf-8", errors="surrogateescape") for path in stdout.split(b"\0") if path]


def tracked_path_collisions(repo: Path, *relatives: str) -> tuple[str, ...]:
    """Return tracked entries overlapping any repository-relative output.

    Both a tracked ancestor and any tracked descendant make a requested output
    unsafe. The query is strict so Git failure cannot be mistaken for an
    untracked destination; absence is only ever an empty tuple.
    """
    return tuple(
        sorted(
            {
                path
                for path in ls_files(repo, strict=True)
                for relative in relatives
                if path == relative or path.startswith(f"{relative}/") or relative.startswith(f"{path}/")
            }
        )
    )


def tracked_artifact_collisions(repo: Path) -> tuple[str, ...]:
    """Return tracked paths that collide with the generated compatibility roots."""
    return tracked_path_collisions(repo, ".review-output.md", ".daydream")


def symbolic_head(repo: Path, *, strict: bool = False) -> str | None:
    """Return the exact short symbolic HEAD, or None for a detached HEAD."""
    proc = process._run_git(repo, ["symbolic-ref", "--quiet", "HEAD"])
    if proc.returncode == 0:
        ref = proc.stdout.rstrip("\n")
        if ref.startswith("refs/heads/"):
            return ref.removeprefix("refs/heads/")
        if strict:
            raise GitError(f"HEAD does not name a local branch in {repo}")
        return None
    if strict and proc.returncode != 1:
        raise GitError(f"cannot resolve symbolic HEAD in {repo}")
    return None


def is_unborn_head(repo: Path) -> bool:
    """Distinguish a genuinely missing symbolic HEAD ref from corrupt Git state."""
    branch = symbolic_head(repo, strict=True)
    if branch is None:
        head_sha(repo)  # a broken detached HEAD is an error, not an unborn repo
        return False
    proc = process._run_git(repo, ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"])
    if proc.returncode == 1:
        return True
    if proc.returncode != 0:
        raise GitError(f"cannot validate HEAD reference in {repo}")
    head_sha(repo)
    return False


def init_repository(repo: Path, *, initial_branch: str | None = None) -> None:
    """Initialize a standalone repository, preserving an optional branch name."""
    repo.mkdir(parents=True, exist_ok=True)
    args = ["init"]
    if initial_branch is not None:
        args.extend(["--initial-branch", initial_branch])
    proc = process._run_git(repo, args, retries=0)
    if proc.returncode != 0:
        raise GitError(f"cannot initialize repository in {repo}")


def _snapshot_git_path(repo: Path, name: str) -> Path:
    proc = process._run_git(repo, ["rev-parse", "--path-format=absolute", "--git-path", name])
    if proc.returncode != 0 or not proc.stdout.rstrip("\n"):
        raise GitError(f"cannot resolve repository {name} path in {repo}")
    return Path(proc.stdout.rstrip("\n")).resolve()


def git_common_dir(repo: Path) -> Path:
    """Return the canonical common Git directory, raising on query failure."""
    proc = process._run_git(repo, ["rev-parse", "--path-format=absolute", "--git-common-dir"])
    if proc.returncode != 0 or not proc.stdout.rstrip("\n"):
        raise GitError(f"cannot resolve common Git directory in {repo}")
    return Path(proc.stdout.rstrip("\n")).resolve()


def git_dirs(repo: Path) -> tuple[Path, Path]:
    """Return the canonical ``(git_dir, git_common_dir)`` pair from one query."""
    proc = process._run_git(repo, ["rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir"])
    lines = [line for line in proc.stdout.split("\n") if line]
    if proc.returncode != 0 or len(lines) != 2:
        raise GitError(f"cannot resolve Git directories in {repo}")
    return Path(lines[0]).resolve(strict=True), Path(lines[1]).resolve()


def git_dir(repo: Path) -> Path:
    """Return the canonical worktree-specific Git directory, raising on query failure."""
    return git_dirs(repo)[0]


def list_remotes(repo: Path, *, strict: bool = False) -> list[str]:
    """List remote names; strict queries never confuse failure with absence."""
    proc = process._run_git(repo, ["remote"])
    if proc.returncode != 0:
        if strict:
            raise GitError(f"cannot list remotes in {repo}")
        return []
    return proc.stdout.splitlines()


def object_alternates(repo: Path, *, strict: bool = False) -> tuple[Path, ...]:
    """Read file-backed alternates relative to the standalone objects directory."""
    try:
        objects = _snapshot_git_path(repo, "objects")
        try:
            raw = (objects / "info" / "alternates").read_bytes()
        except FileNotFoundError:
            return ()
        result = []
        for entry in raw.splitlines():
            if not entry or b"\0" in entry or entry.startswith(b'"'):
                raise GitError("malformed or quoted object alternate")
            path = Path(os.fsdecode(entry))
            result.append((objects / path).resolve(strict=True))
        return tuple(result)
    except (OSError, GitError) as exc:
        if strict:
            raise GitError(f"cannot inspect object alternates in {repo}: {type(exc).__name__}") from exc
        return ()


def _snapshot_path_names(repo: Path, args: list[str], *, strict: bool) -> list[str]:
    proc = process._run_git(repo, args, capture_bytes=True)
    if proc.returncode != 0:
        if strict:
            raise GitError(f"cannot enumerate snapshot paths in {repo}")
        return []
    raw = proc.stdout
    if raw and (not raw.endswith(b"\0") or b"\0\0" in raw):
        raise GitError("malformed snapshot path enumeration")
    return [os.fsdecode(entry) for entry in raw.split(b"\0") if entry]


def ls_tree_files(repo: Path, ref: str, *, strict: bool = False) -> list[str]:
    """List tree paths without quoting, trimming, or newline-splitting names."""
    return _snapshot_path_names(repo, ["ls-tree", "-rz", "--name-only", ref], strict=strict)


def _snapshot_remote_refs(repo: Path) -> list[str]:
    proc = process._run_git(repo, ["for-each-ref", "refs/remotes", "--format=%(refname)"])
    if proc.returncode != 0:
        raise GitError("cannot inspect snapshot remote-tracking refs")
    return proc.stdout.splitlines()


def stash_create(repo: Path) -> str | None:
    """Capture tracked worktree and index changes as a dangling stash commit.

    Leaves working files and stash refs untouched; untracked files require a separate
    snapshot. Return ``None`` when tracked state equals HEAD, and raise on failure.
    """
    proc = process._run_git(
        repo, ["stash", "create"], timeout=30, retries=0, error_context=f"git stash create failed in {repo}"
    )
    return proc.stdout.strip() or None


_FULL_OBJECT_ID_RE = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")


def _validate_pr_base_ref(repo: Path, ref: str, *, prefix: str) -> None:
    if not ref.startswith(prefix):
        raise GitError(f"invalid PR base ref {ref[:120]!r} in {repo}")
    proc = process._run_git(repo, ["check-ref-format", ref], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"invalid PR base ref {ref[:120]!r} in {repo}")


def _merge_base_strict(repo: Path, base_ref: str, head_sha: str) -> str:
    proc = process._run_git(repo, ["merge-base", base_ref, head_sha], timeout=10)
    merge_sha = proc.stdout.strip()
    if proc.returncode != 0 or _FULL_OBJECT_ID_RE.fullmatch(merge_sha) is None:
        raise GitError(f"no merge-base for PR head {head_sha} and base {base_ref} in {repo}")
    return merge_sha.lower()


def resolve_pr_merge_base(
    repo: Path,
    remote_base_refs: Sequence[str],
    local_base_ref: str,
    head_sha: str,
) -> str:
    """Resolve the exact PR head's merge-base against authoritative local refs."""
    if _FULL_OBJECT_ID_RE.fullmatch(head_sha) is None:
        raise GitError(f"exact PR head {head_sha[:120]!r} is not a full object ID in {repo}")
    head_proc = process._run_git(repo, ["rev-parse", "--verify", f"{head_sha}^{{commit}}"], timeout=5)
    resolved_head = head_proc.stdout.strip()
    if (
        head_proc.returncode != 0
        or _FULL_OBJECT_ID_RE.fullmatch(resolved_head) is None
        or resolved_head.lower() != head_sha.lower()
    ):
        raise GitError(f"exact PR head {head_sha} is not a local commit in {repo}")

    _validate_pr_base_ref(repo, local_base_ref, prefix="refs/heads/")
    unique_remote_refs = sorted(set(remote_base_refs))
    for ref in unique_remote_refs:
        _validate_pr_base_ref(repo, ref, prefix="refs/remotes/")

    present_remote_refs: list[str] = []
    for ref in unique_remote_refs:
        check = process._run_git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"], timeout=5)
        if check.returncode == 0:
            present_remote_refs.append(ref)

    if present_remote_refs:
        bases = {ref: _merge_base_strict(repo, ref, head_sha) for ref in present_remote_refs}
        distinct = set(bases.values())
        if len(distinct) != 1:
            refs = ", ".join(present_remote_refs)
            raise GitError(f"matching PR base remotes disagree ({refs}) in {repo}; fetch/align the base remote refs")
        return next(iter(distinct))

    local_check = process._run_git(repo, ["rev-parse", "--verify", f"{local_base_ref}^{{commit}}"], timeout=5)
    if local_check.returncode != 0:
        raise GitError(f"PR base {local_base_ref} is unavailable for head {head_sha} in {repo}")
    return _merge_base_strict(repo, local_base_ref, head_sha)


def upstream_ahead_count(repo: Path, branch: str) -> int:
    """Return the upstream right-side ahead count, or zero when unavailable."""
    return _upstream_and_ahead(repo, branch)[1]


def check_ignore(repo: Path, path: str) -> bool:
    """Probe ignored status; subprocess failures return False for best-effort copy
    filtering.
    """
    try:
        proc = process._run_git(repo, ["check-ignore", "--quiet", path], timeout=5)
    except GitError:
        return False
    return proc.returncode == 0


def split_owner_repo(slug: str) -> tuple[str, str] | None:
    """Split exactly one slash with nonempty, whitespace-free owner/repo components;
    otherwise return None.
    """
    if slug.count("/") != 1:
        return None
    owner, _, repo = slug.partition("/")
    if not owner or not repo or any(char.isspace() for char in owner + repo):
        return None
    return owner, repo


def git_ls_remote(repo: Path, url: str) -> str:
    """Read remote refs using command-scoped ``gh`` credentials and no terminal prompts.

    Credentials stay out of URLs, argv, and persistent config. Failures raise.
    """
    args = [*process._credential_helper_args(), "ls-remote", url]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    proc = process._run_git(repo, args, env_cmd=env, error_context=f"git ls-remote {url} failed")
    return proc.stdout


def remote_contains_commit(repo: Path, branch: str, sha: str, *, remote: str = "origin") -> bool:
    """Check the authenticated remote branch ref for an exact SHA; failed reads raise."""
    proc = process._run_git(
        repo,
        ["ls-remote", remote, f"refs/heads/{branch}"],
        env_cmd={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        error_context=f"git ls-remote {remote} refs/heads/{branch} failed",
    )
    return any(line.split()[0] == sha for line in proc.stdout.splitlines() if line.strip())
