"""Single point of contact for all ``git`` and ``gh`` subprocess calls.

This module centralises every shell-out daydream performs against the local
repository or GitHub.  All callers should depend on the public API declared
here rather than spawning ``git`` / ``gh`` directly — that lets the worktree
isolation work evolve invariants (timeouts, working directory contracts,
error semantics) without rewriting the same incantations across the codebase.

Conventions:
    * Every command receives ``cwd=repo`` (no ``git -C`` shenanigans).
    * Read-only queries time out at 5 seconds, IO-bound at 30 seconds, and
      ``gh`` operations at 60 seconds.
    * ``git`` timeouts are retried a bounded number of times before raising
      :class:`GitTimeoutError` — a trivial command exceeding its timeout means
      the host is overloaded, not that the command hung.

Error-handling patterns:
    Functions in this module follow one of two documented patterns:

    **Hard failure (raise GitError)**: Used when the caller cannot proceed
    without the result. Examples: :func:`head_sha`, :func:`diff`, :func:`fetch`.
    These raise :class:`GitError` (or a subclass) on any non-zero exit.

    **Soft failure (return sentinel)**: Used when "data not available" is a
    valid, expected outcome the caller can handle inline. These return
    ``None``, ``False``, ``0``, or ``[]`` on non-zero exit instead of raising.
    Examples: :func:`remote_url`, :func:`merge_base`, :func:`gh_pr_view`.

    Each function's docstring specifies which pattern it follows under its
    **Raises** or **Returns** section.

The module is intentionally dependency-free: stdlib only.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

# Substrings (lowercased) that identify a GitHub API rate-limit response in
# ``gh`` stderr output.  Kept as a module constant so the classifier and any
# future callers share a single source of truth.
#
# NOTE: "429" is intentionally absent here; HTTP 429 is matched separately in
# ``_gh_error_for`` with a word-boundary regex to avoid false positives on
# arbitrary digit sequences (e.g. URLs, SHAs, file sizes) that happen to
# contain those three digits.
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "secondary rate limit",
)

# --- Errors ------------------------------------------------------------------


class GitError(Exception):
    """Base class for all git/gh failures raised by :mod:`daydream.git_ops`."""


class GitTimeoutError(GitError):
    """Raised when a ``git``/``gh`` subprocess exceeds its timeout.

    Distinct from the generic :class:`GitError` so callers can tell a
    *transient* host-load timeout (a trivial command starved of CPU) apart
    from a genuine git failure (bad ref, missing object, not a worktree). A
    timeout signals the machine is overloaded, not that the command is wrong,
    so it is retried a bounded number of times in :func:`_run_git` before it
    surfaces as this exception.
    """


class RateLimitError(GitError):
    """Raised when a ``gh`` call fails due to a GitHub API rate limit.

    Attributes:
        retry_after: Suggested seconds to wait before retrying, parsed from the
            ``gh`` stderr when available; ``None`` when no hint was present.
    """

    def __init__(self, *args: Any, retry_after: float | None = None) -> None:
        super().__init__(*args)
        self.retry_after = retry_after


class NotAWorktreeError(GitError):
    """Raised when an operation requires a worktree but the path is not one."""


class BranchNotFoundError(GitError):
    """Raised when an operation references a branch that does not exist."""


class WrongBranchError(GitError):
    """Raised when a worktree is checked out to an unexpected branch.

    Defined here for callers (notably the worktree isolation logic) to raise
    when invariant checks fail. Not raised by this module today.
    """


# --- Internal subprocess helpers --------------------------------------------


# A trivial git command exceeding its timeout means the host was momentarily
# starved of CPU (heavy concurrent load), not that the command hung — so a
# timeout is retried a bounded number of times before it surfaces as a
# GitTimeoutError. This is what kept the deep-orchestrator full-suite tests
# flaky: under load a 5s `git rev-parse`/`diff` would time out, collapse to a
# generic GitError, and the run would exit 1 (see issue #120).
_GIT_TIMEOUT_RETRIES = 2


def _run_git(
    repo: Path,
    args: list[str],
    *,
    timeout: int = 5,
    capture_bytes: bool = False,
    retries: int = _GIT_TIMEOUT_RETRIES,
) -> subprocess.CompletedProcess[Any]:
    """Run ``git`` in *repo* with hardened defaults.

    Args:
        repo: Repository working directory.
        args: Arguments after ``git``.
        timeout: Subprocess timeout in seconds.
        capture_bytes: When True, capture stdout/stderr as bytes (no decoding).
        retries: How many additional attempts to make after a
            :class:`subprocess.TimeoutExpired` (total attempts = ``retries + 1``).
            Only timeouts are retried; other failures raise immediately.

    Returns:
        The completed process. ``returncode`` is left to the caller to inspect.

    Raises:
        GitTimeoutError: If every attempt times out. Subclass of
            :class:`GitError`, so existing ``except GitError`` handlers still
            catch it.
        GitError: If the underlying subprocess machinery fails for any other
            reason (missing binary, OS-level error).
    """
    last_timeout: subprocess.TimeoutExpired | None = None
    for attempt in range(retries + 1):
        try:
            return subprocess.run(  # noqa: S603 - arguments are not user-controlled
                ["git", *args],  # noqa: S607 - git is a trusted command
                cwd=repo,
                capture_output=True,
                text=not capture_bytes,
                timeout=timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_timeout = exc
            if attempt < retries:
                _logger.warning(
                    "git %s timed out after %ss (attempt %d/%d); retrying",
                    " ".join(args),
                    timeout,
                    attempt + 1,
                    retries + 1,
                )
        except (subprocess.SubprocessError, OSError) as exc:
            raise GitError(f"git {' '.join(args)} failed: {type(exc).__name__}: {exc}") from exc

    raise GitTimeoutError(
        f"git {' '.join(args)} timed out after {timeout}s ({retries + 1} attempts)",
    ) from last_timeout


def _run_gh(
    repo: Path,
    args: list[str],
    *,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Run ``gh`` in *repo* with hardened defaults.

    Args:
        repo: Repository working directory.
        args: Arguments after ``gh``.
        timeout: Subprocess timeout in seconds.

    Returns:
        The completed process with text-decoded stdout/stderr.

    Raises:
        GitTimeoutError: If the call times out. Subclass of :class:`GitError`.
        GitError: If the subprocess machinery fails for any other reason
            (missing ``gh``, OS-level error).
    """
    try:
        return subprocess.run(  # noqa: S603 - arguments are not user-controlled
            ["gh", *args],  # noqa: S607 - gh is a trusted command
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitTimeoutError(f"gh {' '.join(args)} timed out after {timeout}s") from exc
    except (subprocess.SubprocessError, OSError) as exc:
        raise GitError(f"gh {' '.join(args)} failed: {type(exc).__name__}: {exc}") from exc


# --- Pre-flight --------------------------------------------------------------


def assert_is_worktree(repo: Path) -> None:
    """Verify *repo* is the root of a git worktree.

    Args:
        repo: Path that should resolve to a worktree top level.

    Raises:
        NotAWorktreeError: If *repo* is not inside a git worktree, or if it is
            inside one but is not itself the worktree's top-level directory
            (catches the "org dir holding repo subdirs" footgun).
    """
    if not repo.exists() or not repo.is_dir():
        raise NotAWorktreeError(f"{repo} is not a directory")

    proc = _run_git(repo, ["rev-parse", "--is-inside-work-tree"], timeout=5)
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise NotAWorktreeError(f"{repo} is not inside a git worktree")

    top_proc = _run_git(repo, ["rev-parse", "--show-toplevel"], timeout=5)
    if top_proc.returncode != 0:
        raise NotAWorktreeError(f"{repo} could not resolve its worktree top-level")

    top = Path(top_proc.stdout.strip()).resolve()
    if top != repo.resolve():
        raise NotAWorktreeError(
            f"{repo} is inside a worktree but its top-level is {top}; pass the worktree root instead",
        )


def is_inside_worktree(repo: Path) -> bool:
    """Return True iff :func:`assert_is_worktree` would succeed for *repo*.

    Args:
        repo: Candidate worktree path.

    Returns:
        True when *repo* is the top-level of a git worktree, otherwise False.
    """
    try:
        assert_is_worktree(repo)
    except NotAWorktreeError:
        return False
    return True


# --- Read-only queries -------------------------------------------------------


def head_sha(repo: Path) -> str:
    """Return the full SHA of ``HEAD`` in *repo*.

    Args:
        repo: Repository working directory.

    Returns:
        The 40-character SHA of the current ``HEAD`` commit.

    Raises:
        GitError: If ``git rev-parse HEAD`` fails (e.g. empty repository).
    """
    proc = _run_git(repo, ["rev-parse", "HEAD"], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"cannot resolve HEAD in {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def head_commit_message(repo: Path) -> str:
    """Return the full commit message of ``HEAD``.

    Args:
        repo: Repository working directory.

    Returns:
        The commit message body of the current ``HEAD`` commit.

    Raises:
        GitError: If ``git log`` fails (e.g. empty repository).
    """
    proc = _run_git(repo, ["log", "-1", "--format=%B", "HEAD"], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"cannot read HEAD message in {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def amend_trailers(repo: Path, trailers: dict[str, str], *, message: str | None = None) -> None:
    """Amend ``HEAD`` to append missing git trailers.

    Uses ``git interpret-trailers`` to inject trailers, then amends the
    commit with the updated message.  This is a no-op if *trailers* is empty.

    Args:
        repo: Repository working directory.
        trailers: Mapping of trailer keys to values (e.g.
            ``{"Daydream-Run": "abc123"}``).
        message: When provided, use this as the current ``HEAD`` commit message
            instead of reading it via ``git log``.  Avoids a redundant
            subprocess call when the caller has already read the message.

    Raises:
        GitError: If the amend fails.
    """
    if not trailers:
        return

    # Build the amended message via git interpret-trailers.
    trailer_args: list[str] = []
    for key, value in trailers.items():
        trailer_args += ["--trailer", f"{key}: {value}"]

    if message is None:
        msg_proc = _run_git(repo, ["log", "-1", "--format=%B", "HEAD"], timeout=5)
        if msg_proc.returncode != 0:
            raise GitError(f"cannot read HEAD message: {msg_proc.stderr.strip()}")
        raw_message = msg_proc.stdout
    else:
        raw_message = message

    # Pipe message through interpret-trailers (run directly, not via _run_git
    # because we need stdin).
    try:
        interp = subprocess.run(  # noqa: S603 - arguments are not user-controlled
            ["git", "interpret-trailers", *trailer_args],  # noqa: S607 - git is a trusted command
            input=raw_message,
            capture_output=True,
            text=True,
            cwd=repo,
            timeout=5,
            shell=False,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise GitError(f"git interpret-trailers failed: {exc}") from exc

    if interp.returncode != 0:
        raise GitError(f"git interpret-trailers failed: {interp.stderr.strip()}")

    amended_msg = interp.stdout
    amend_proc = _run_git(
        repo, ["commit", "--amend", "-m", amended_msg.strip()], timeout=30,
    )
    if amend_proc.returncode != 0:
        raise GitError(f"git commit --amend failed: {amend_proc.stderr.strip()}")


def remote_url(repo: Path, remote: str = "origin") -> str | None:
    """Return the URL configured for *remote*, or ``None`` when unset/missing.

    Soft-failure semantics: returns ``None`` on any non-zero ``git`` exit
    (mirrors :func:`default_branch` / :func:`merge_base` / :func:`current_branch`
    behavior for "the data isn't there" cases) and on subprocess machinery
    failures (timeout, missing binary).

    Args:
        repo: Repository working directory.
        remote: Remote name. Defaults to ``"origin"``.

    Returns:
        The configured URL, or ``None`` when the remote is not set.
    """
    try:
        proc = _run_git(repo, ["config", "--get", f"remote.{remote}.url"], timeout=5)
    except GitError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def current_branch(repo: Path) -> str | None:
    """Return the current branch name, or ``None`` when ``HEAD`` is detached.

    Args:
        repo: Repository working directory.

    Returns:
        The branch name (e.g. ``"main"``) or ``None`` for detached ``HEAD``.

    Raises:
        GitError: If the underlying subprocess fails to execute.
    """
    proc = _run_git(repo, ["branch", "--show-current"], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"cannot read current branch in {repo}: {proc.stderr.strip()}")
    name = proc.stdout.strip()
    return name or None


def default_branch(repo: Path) -> str:
    """Resolve the repository's default branch name.

    Resolution order: ``origin/HEAD`` symbolic ref → local ``main`` → local
    ``master``. Raises if none of those exist.

    Args:
        repo: Repository working directory.

    Returns:
        The default branch name (e.g. ``"main"``).

    Raises:
        BranchNotFoundError: If no default branch can be detected.
    """
    sym = _run_git(repo, ["symbolic-ref", "refs/remotes/origin/HEAD"], timeout=5)
    if sym.returncode == 0 and sym.stdout.strip():
        return sym.stdout.strip().rsplit("/", 1)[-1]

    for candidate in ("main", "master"):
        check = _run_git(repo, ["rev-parse", "--verify", f"refs/heads/{candidate}"], timeout=5)
        if check.returncode == 0:
            return candidate

    raise BranchNotFoundError(f"no default branch (origin/HEAD, main, master) found in {repo}")


def branch_exists(repo: Path, ref: str) -> bool:
    """Check whether *ref* exists locally or as ``origin/<ref>``.

    Args:
        repo: Repository working directory.
        ref: Branch name to look up.

    Returns:
        True when ``refs/heads/<ref>`` or ``refs/remotes/origin/<ref>`` exists.
    """
    local = _run_git(repo, ["rev-parse", "--verify", f"refs/heads/{ref}"], timeout=5)
    if local.returncode == 0:
        return True
    remote = _run_git(repo, ["rev-parse", "--verify", f"refs/remotes/origin/{ref}"], timeout=5)
    return remote.returncode == 0


def ref_exists(repo: Path, ref: str) -> bool:
    """Check whether *ref* resolves to a commit git can name.

    Accepts a named branch (local or ``origin/<ref>``) plus any commit-ish:
    a full or abbreviated SHA, a tag, or a relative expression such as
    ``HEAD~3``.

    Args:
        repo: Repository working directory.
        ref: Branch name, SHA, tag, or any commit-ish expression.

    Returns:
        True when *ref* names a branch or resolves to a commit object.

    Raises:
        GitError: Only for unexpected subprocess failures (timeout, missing
            git binary). The documented soft-failure modes return ``False``.
    """
    if branch_exists(repo, ref):
        return True
    # No valid commit-ish (SHA, tag, relative expression) starts with '-'.
    # A leading dash would be mis-parsed by git as an option flag; reject it
    # early rather than letting an attacker-controlled string reach the shell.
    if ref.startswith("-"):
        return False
    commit = _run_git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"], timeout=5)
    return commit.returncode == 0


def merge_base(repo: Path, base: str, head: str = "HEAD") -> str | None:
    """Compute the merge-base between *head* and *base*, preferring upstream.

    Ports the algorithm in ``codex-rs/git-utils/src/branch.rs``: when
    ``<base>@{upstream}`` exists and is **ahead** of the local *base*
    (i.e. ``rev-list --left-right --count base...upstream`` reports
    ``right > 0``), the merge-base is computed against the upstream ref instead
    of the local branch.  This avoids stale merge-bases when the local copy of
    the base branch has been rewritten or simply not pulled.

    Returns ``None`` (rather than raising) on the codex-documented "soft"
    failure modes — empty repo, missing ``HEAD``, missing branch — so callers
    can treat them as "no merge-base available" without try/except plumbing.

    Args:
        repo: Repository working directory.
        base: Base branch name (e.g. ``"main"``).
        head: Ref whose merge-base to find. Defaults to ``"HEAD"``.

    Returns:
        The merge-base SHA, or ``None`` when one cannot be resolved.

    Raises:
        GitError: Only for unexpected subprocess failures (timeout, missing
            git binary). The documented soft-failure modes return ``None``.
    """
    # Same guard as ref_exists: a leading '-' would be mis-parsed as a git
    # option flag.  No valid branch name or commit-ish starts with '-'.
    if head.startswith("-") or base.startswith("-"):
        return None

    head_proc = _run_git(repo, ["rev-parse", "--verify", head], timeout=5)
    if head_proc.returncode != 0:
        return None

    base_proc = _run_git(repo, ["rev-parse", "--verify", base], timeout=5)
    if base_proc.returncode != 0:
        return None
    preferred_ref = base

    upstream = _resolve_upstream_if_remote_ahead(repo, base)
    if upstream is not None:
        upstream_check = _run_git(repo, ["rev-parse", "--verify", upstream], timeout=5)
        if upstream_check.returncode == 0:
            preferred_ref = upstream

    mb = _run_git(repo, ["merge-base", head, preferred_ref], timeout=5)
    if mb.returncode != 0:
        return None
    out = mb.stdout.strip()
    return out or None


def _resolve_upstream_if_remote_ahead(repo: Path, branch: str) -> str | None:
    """Return ``<branch>@{upstream}`` iff it has commits the local branch lacks.

    Mirrors codex's ``resolve_upstream_if_remote_ahead``: parses the right-side
    count from ``rev-list --left-right --count <branch>...<upstream>`` and
    returns the upstream symbolic name when ``right > 0``.
    """
    upstream_name_proc = _run_git(
        repo,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", f"{branch}@{{upstream}}"],
        timeout=5,
    )
    if upstream_name_proc.returncode != 0:
        return None
    upstream = upstream_name_proc.stdout.strip()
    if not upstream:
        return None

    counts_proc = _run_git(
        repo,
        ["rev-list", "--left-right", "--count", f"{branch}...{upstream}"],
        timeout=5,
    )
    if counts_proc.returncode != 0:
        return None
    parts = counts_proc.stdout.strip().split()
    try:
        right = int(parts[1]) if len(parts) >= 2 else 0
    except ValueError:
        right = 0
    return upstream if right > 0 else None


def _prefer_remote_base(repo: Path, base: str) -> str:
    """Return ``origin/<base>`` when it exists, otherwise *base* unchanged."""
    remote_ref = f"origin/{base}"
    check = _run_git(repo, ["rev-parse", "--verify", f"refs/remotes/{remote_ref}"], timeout=5)
    if check.returncode == 0:
        return remote_ref
    return base


def diff(repo: Path, base: str, head: str = "HEAD", *, exclude: list[str] | None = None) -> str:
    """Return the unified diff between *base* and *head*.

    Uses three-dot syntax (``base...head``) so the diff reflects changes on
    *head* since it diverged from *base*.  When ``origin/<base>`` exists the
    function prefers it via :func:`_prefer_remote_base` so the diff is
    computed against the remote default branch — not a potentially stale
    (or identical-to-HEAD) local copy.

    Args:
        repo: Repository working directory.
        base: Base ref (e.g. ``"main"``).
        head: Comparison ref. Defaults to ``"HEAD"``.
        exclude: Optional pathspec excludes (each becomes ``:(exclude)<p>``).

    Returns:
        The diff text. Empty string when there are no changes.

    Raises:
        GitError: If ``git diff`` fails.
    """
    preferred_base = _prefer_remote_base(repo, base)
    args = ["diff", f"{preferred_base}...{head}"]
    if exclude:
        args.append("--")
        args.append(".")
        args.extend(f":(exclude){p.rstrip('/')}" for p in exclude)
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git diff {base}...{head} failed: {proc.stderr.strip()}")
    return proc.stdout


def diff_name_only(repo: Path, base: str, head: str = "HEAD") -> list[str]:
    """Return the list of paths changed between *base* and *head*.

    Uses ``git diff --name-only base..head`` (two-dot) so the result is the
    direct set of files differing between the two refs at archive time.

    Soft-failure semantics mirror :func:`merge_base`: returns an empty list
    when either ref cannot be resolved or the subprocess fails. Callers in
    archive paths should not propagate git transients into manifest failure.

    Args:
        repo: Repository working directory.
        base: Base ref or SHA.
        head: Comparison ref or SHA. Defaults to ``"HEAD"``.

    Returns:
        Repo-relative path strings in git output order. Empty list on any
        soft failure.
    """
    try:
        proc = _run_git(repo, ["diff", "--name-only", f"{base}..{head}"], timeout=10)
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
    """Diff a specific set of paths between *base* and *head*.

    Distinct from :func:`diff` because:
      * This uses two-dot range syntax (``base..head``) by default; :func:`diff`
        uses three-dot. Two-dot shows the working diff between the two refs;
        three-dot shows what *head* introduced relative to the merge-base. PR
        comment line resolution needs two-dot.
      * Always passes ``--unified=<unified>`` for explicit context.
      * Restricts output to *paths*.

    Args:
        repo: Repository root.
        base: Base ref.
        head: Head ref.
        paths: Pathspec list (relative to repo root).
        unified: Lines of context.
        merge_base_diff: When False (default), use ``base..head`` (direct diff);
            when True, use ``base...head`` (diff since merge-base).

    Returns:
        The diff text. Empty string when there are no changes for *paths*.

    Raises:
        GitError: If ``git diff`` fails.
    """
    sep = "..." if merge_base_diff else ".."
    range_arg = f"{base}{sep}{head}"
    args = ["diff", f"--unified={unified}", range_arg, "--", *paths]
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git diff {range_arg} failed: {proc.stderr.strip()}")
    return proc.stdout


def log(repo: Path, base: str, head: str = "HEAD") -> str:
    """Return the one-line commit log for ``base..head``.

    Args:
        repo: Repository working directory.
        base: Base ref.
        head: Comparison ref. Defaults to ``"HEAD"``.

    Returns:
        Stripped ``--oneline`` log output. Empty string when no commits.

    Raises:
        GitError: If ``git log`` fails.
    """
    proc = _run_git(repo, ["log", f"{base}..{head}", "--oneline"], timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git log {base}..{head} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def log_shas(repo: Path, ref: str, *, since: str) -> list[str]:
    """Return full SHAs of commits on ``since..ref``.

    Soft-failure semantics: returns ``[]`` on any git error or non-zero exit
    so callers can treat "no commits available" without try/except plumbing.

    Args:
        repo: Repository working directory.
        ref: Head ref or branch name (e.g. ``"main"``).
        since: Base ref or SHA; commits reachable from *ref* but not from
            *since* are returned (``git log since..ref``).

    Returns:
        List of 40-character SHA strings in ``git log`` output order
        (newest first). Empty list on any soft failure.
    """
    try:
        proc = _run_git(repo, ["log", "--pretty=%H", f"{since}..{ref}"], timeout=10)
    except GitError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def log_shas_since(repo: Path, head: str, base: str, *, since_days: int) -> list[str]:
    """Return full SHAs of commits on ``head..base`` within *since_days*.

    Equivalent to ``git log --since=<n> days ago --pretty=%H head..base``.
    Used to bound the upstream review window to commits authored recently
    enough to be plausibly related to a given patch.

    Soft-failure semantics: returns ``[]`` on any git error or non-zero exit.

    Args:
        repo: Repository working directory.
        head: The divergence point; commits reachable from *head* are excluded.
        base: The tip ref; only commits reachable from *base* are included.
        since_days: How many days back to search.

    Returns:
        List of 40-character SHA strings in ``git log`` output order
        (newest first). Empty list on any soft failure.
    """
    try:
        proc = _run_git(
            repo,
            ["log", f"--since={since_days} days ago", "--pretty=%H", f"{head}..{base}"],
            timeout=10,
        )
    except GitError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def daydream_commits(repo: Path, base: str, head: str = "HEAD") -> str | None:
    """Return oneline log of prior daydream commits in ``base..head``.

    Args:
        repo: Repository working directory.
        base: Base ref (e.g. ``"main"``).
        head: Comparison ref. Defaults to ``"HEAD"``.

    Returns:
        Stripped log output, or ``None`` if no daydream commits found.
    """
    proc = _run_git(
        repo,
        ["log", f"{base}..{head}", "--oneline", "--grep=Daydream-Run:"],
        timeout=30,
    )
    if proc.returncode != 0:
        _logger.warning(
            "git log %s..%s --grep=Daydream-Run: failed (rc=%d): %s",
            base,
            head,
            proc.returncode,
            (proc.stderr or "").strip(),
        )
        return None
    output = proc.stdout.strip()
    return output or None


def show(repo: Path, ref: str, path: str) -> bytes:
    """Return the raw bytes of *path* at *ref* via ``git show``.

    Args:
        repo: Repository working directory.
        ref: Commit / branch / tag.
        path: Path within the repository.

    Returns:
        Raw bytes of the file at the given revision.

    Raises:
        GitError: If ``git show`` fails (e.g. path missing at that revision).
    """
    proc = _run_git(repo, ["show", f"{ref}:{path}"], timeout=30, capture_bytes=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
        raise GitError(f"git show {ref}:{path} failed: {stderr.strip()}")
    return proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()


def grep(repo: Path, pattern: str) -> list[str]:
    """Return file paths matching *pattern* via ``git grep -l``.

    Args:
        repo: Repository working directory.
        pattern: Pattern passed to ``git grep -l``.

    Returns:
        List of matching file paths (one per line of stdout).

    Raises:
        GitError: If ``git grep`` exits with an unexpected status.  Exit code
            ``1`` is "no matches" and is treated as success (empty list).
    """
    proc = _run_git(repo, ["grep", "-l", "--", pattern], timeout=30)
    # git grep returns 1 when there are simply no matches.
    if proc.returncode not in (0, 1):
        raise GitError(f"git grep {pattern!r} failed: {proc.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def status_porcelain(repo: Path) -> str:
    """Return ``git status --porcelain`` output.

    Args:
        repo: Repository working directory.

    Returns:
        Porcelain-formatted status text. Empty when the tree is clean.

    Raises:
        GitError: If ``git status`` fails.
    """
    proc = _run_git(repo, ["status", "--porcelain"], timeout=10)
    if proc.returncode != 0:
        raise GitError(f"git status failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout


def changed_files(repo: Path) -> list[str]:
    """Return repo-relative paths of files changed in the working tree.

    Best-effort: combines staged + unstaged changes (``git diff --name-only
    HEAD``) with new untracked files (``git ls-files --others
    --exclude-standard``).  Files created during a daydream fix are still
    untracked at abort time, so the diff alone would omit them and leave the
    handoff missing critical context.

    Soft-failure semantics: returns an empty list when git is unavailable, the
    repo has no commits yet, or either subcommand fails.  Individual
    subcommand failures are logged and skipped — the other subcommand's
    results are still returned.

    Args:
        repo: Repository working directory.

    Returns:
        De-duplicated list of repo-relative path strings.  Empty on error.
    """
    names: list[str] = []
    seen: set[str] = set()
    for args in (
        ["diff", "--name-only", "HEAD"],
        ["ls-files", "--others", "--exclude-standard"],
    ):
        try:
            proc = _run_git(repo, args, timeout=10)
        except GitError:
            continue
        if proc.returncode != 0:
            continue
        for line in proc.stdout.splitlines():
            name = line.strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def upstream_ahead_count(repo: Path, branch: str) -> int:
    """Return the number of commits ``<branch>@{upstream}`` is ahead of *branch*.

    Args:
        repo: Repository working directory.
        branch: Branch name to compare against its tracked upstream.

    Returns:
        The right-side count from ``rev-list --left-right --count``. Returns
        ``0`` when *branch* has no configured upstream.
    """
    upstream_name_proc = _run_git(
        repo,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", f"{branch}@{{upstream}}"],
        timeout=5,
    )
    if upstream_name_proc.returncode != 0:
        return 0
    upstream = upstream_name_proc.stdout.strip()
    if not upstream:
        return 0
    counts_proc = _run_git(
        repo,
        ["rev-list", "--left-right", "--count", f"{branch}...{upstream}"],
        timeout=5,
    )
    if counts_proc.returncode != 0:
        return 0
    parts = counts_proc.stdout.strip().split()
    try:
        return int(parts[1]) if len(parts) >= 2 else 0
    except ValueError:
        return 0


def check_ignore(repo: Path, path: str) -> bool:
    """Return True iff ``git check-ignore`` says *path* is ignored.

    Soft-failure semantics: returns ``False`` on any subprocess error (timeout,
    missing binary, OS-level failure) to avoid blocking callers that use this
    for optional file-copy filtering.

    Args:
        repo: Repository working directory.
        path: Path (relative to *repo*) to check.

    Returns:
        True when *path* is gitignored, False otherwise or on error.
    """
    try:
        proc = _run_git(repo, ["check-ignore", "--quiet", path], timeout=5)
    except GitError:
        return False
    return proc.returncode == 0


# --- Mutating ----------------------------------------------------------------


def fetch(repo: Path, remote: str = "origin") -> None:
    """Run ``git fetch`` against *remote*.

    Args:
        repo: Repository working directory.
        remote: Remote name. Defaults to ``"origin"``.

    Raises:
        GitError: If the fetch fails.
    """
    proc = _run_git(repo, ["fetch", remote], timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git fetch {remote} failed in {repo}: {proc.stderr.strip()}")


def fetch_ref(repo: Path, refspec: str, remote: str = "origin", *, timeout: int = 300) -> None:
    """Fetch a single *refspec* from *remote* into *repo*.

    Useful for fetching refs that are not covered by the default fetch
    configuration, such as ``refs/pull/<N>/head`` on GitHub.

    Args:
        repo: Repository working directory.
        refspec: Refspec to fetch (e.g. ``"pull/42/head"``).
        remote: Remote name. Defaults to ``"origin"``.
        timeout: Subprocess timeout in seconds. Defaults to 300 s to
            accommodate first-run blobless fetches of large repositories.

    Raises:
        GitError: If the fetch fails for any reason.
    """
    proc = _run_git(repo, ["fetch", remote, refspec], timeout=timeout)
    if proc.returncode != 0:
        raise GitError(f"git fetch {remote} {refspec} failed in {repo}: {proc.stderr.strip()}")


def checkout_detach(repo: Path, sha: str, *, timeout: int = 300) -> None:
    """Detach HEAD onto *sha* in *repo*.

    Args:
        repo: Repository working directory.
        sha: Commit SHA or any commit-ish to detach onto.
        timeout: Subprocess timeout in seconds.  Defaults to 300 s because
            detaching HEAD in a blobless clone triggers lazy blob fetches that
            can take several minutes on large repositories.

    Raises:
        GitError: If the checkout fails.
    """
    proc = _run_git(repo, ["checkout", "--detach", sha], timeout=timeout)
    if proc.returncode != 0:
        raise GitError(f"git checkout --detach {sha} failed in {repo}: {proc.stderr.strip()}")


def clone(remote_url: str, target: Path, *, blobless: bool = False, timeout: int = 300) -> None:
    """Run ``git clone <remote_url> <target>``.

    Args:
        remote_url: Remote URL or local path to clone from.
        target: Destination directory for the new working tree.
        blobless: When ``True``, pass ``--filter=blob:none`` to perform a
            partial clone that omits blobs until they are accessed.  Reduces
            initial transfer and storage at the cost of lazy blob fetches on
            first access.  Requires server-side partial-clone support.
        timeout: Subprocess timeout in seconds. Defaults to 300 s to
            accommodate first-run blobless clones of large repositories.

    Raises:
        GitError: If the clone fails.
    """
    cmd = ["git", "clone"]
    if blobless:
        cmd.append("--filter=blob:none")
    cmd += [remote_url, str(target)]
    try:
        proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
            cmd,  # noqa: S607 - git is a trusted command
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise GitError(f"git clone {remote_url} failed: {type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        raise GitError(f"git clone {remote_url} failed: {proc.stderr.strip()}")


def checkout_paths(repo: Path, paths: list[Path]) -> None:
    """Run ``git checkout -- <paths>`` to discard local changes for *paths*.

    Args:
        repo: Repository working directory.
        paths: Paths (relative to *repo*) to restore from the index. Pass
            ``[Path(".")]`` to restore the entire working tree.

    Raises:
        GitError: If the checkout fails.
    """
    if not paths:
        return
    args = ["checkout", "--", *(str(p) for p in paths)]
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git checkout -- {paths} failed in {repo}: {proc.stderr.strip()}")


def clean_untracked(repo: Path) -> None:
    """Run ``git clean -fd`` to remove untracked files and directories.

    Args:
        repo: Repository working directory.

    Raises:
        GitError: If the clean fails.
    """
    proc = _run_git(repo, ["clean", "-fd"], timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git clean -fd failed in {repo}: {proc.stderr.strip()}")


def worktree_add(repo: Path, path: Path, ref: str, *, detach: bool = True) -> None:
    """Create a new worktree at *path* pointing at *ref*.

    Args:
        repo: The source repository.
        path: Filesystem path for the new worktree (must not already exist).
        ref: Commit / branch / tag to check out.
        detach: When True, pass ``--detach`` so the new worktree is detached.

    Raises:
        GitError: If ``git worktree add`` fails.
    """
    args = ["worktree", "add"]
    if detach:
        args.append("--detach")
    args.extend([str(path), ref])
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git worktree add {path} {ref} failed: {proc.stderr.strip()}")


def worktree_remove(repo: Path, path: Path, *, force: bool = True) -> None:
    """Remove the worktree at *path*.

    Args:
        repo: The source repository (or any worktree linked to the same repo).
        path: Path of the worktree to remove.
        force: When True, pass ``--force`` to remove dirty worktrees.

    Raises:
        GitError: If ``git worktree remove`` fails.
    """
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git worktree remove {path} failed: {proc.stderr.strip()}")


# --- gh wrappers -------------------------------------------------------------


def gh_pr_view(repo: Path, pr: int | None = None) -> dict | None:
    """Return ``gh pr view`` output as a dict, or ``None`` on failure.

    When *pr* is ``None``, ``gh pr view`` infers the PR from the currently
    checked-out branch. This mirrors the auto-detection flow used by the CLI
    when the user does not pass an explicit PR number.

    Args:
        repo: Repository working directory.
        pr: Pull request number, or ``None`` to let ``gh`` infer from the
            current branch.

    Returns:
        Parsed JSON dict, or ``None`` when no PR is found / the call fails.
    """
    args = ["pr", "view"]
    if pr is not None:
        args.append(str(pr))
    args.extend(
        [
            "--json",
            "number,title,body,state,headRefName,baseRefName,headRefOid,baseRefOid,url",
        ]
    )
    proc = _run_gh(repo, args, timeout=60)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def gh_pr_list_for_branch(repo: Path, branch: str) -> list[dict]:
    """List open PRs whose head ref is *branch*.

    Args:
        repo: Repository working directory.
        branch: Head branch name.

    Returns:
        List of PR dicts (empty when no PRs match or the call fails).
    """
    proc = _run_gh(
        repo,
        [
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "number,headRefOid,baseRefOid,baseRefName,url,headRepository,headRepositoryOwner",
        ],
        timeout=60,
    )
    if proc.returncode != 0:
        return []
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def gh_pr_diff(repo: Path, pr: int) -> str:
    """Return the unified diff for *pr* as text.

    Args:
        repo: Repository working directory.
        pr: Pull request number.

    Returns:
        The diff text.

    Raises:
        GitError: If ``gh pr diff`` fails.
    """
    proc = _run_gh(repo, ["pr", "diff", str(pr)], timeout=60)
    if proc.returncode != 0:
        raise GitError(f"gh pr diff {pr} failed: {proc.stderr.strip()}")
    return proc.stdout


def gh_repo_view(repo: Path) -> tuple[str, str] | None:
    """Return the ``(owner, name)`` slug for the current repository.

    Args:
        repo: Repository working directory.

    Returns:
        Tuple of ``(owner, name)``, or ``None`` when the call fails or the
        slug cannot be parsed.
    """
    proc = _run_gh(
        repo,
        ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    slug = proc.stdout.strip()
    if "/" not in slug:
        return None
    owner, _, name = slug.partition("/")
    if not owner or not name:
        return None
    return owner, name


def _gh_error_for(message: str, stderr: str) -> GitError:
    """Classify a ``gh`` failure into a rate-limit error or a plain GitError.

    Detection is on the stderr string only: a case-insensitive match against
    ``rate limit``, ``secondary rate limit``, a word-boundary ``429`` (HTTP
    status code), or ``403`` co-occurring with ``rate`` yields a
    :class:`RateLimitError` (with ``retry_after`` parsed from the stderr when
    an integer hint is present). Anything else returns a plain
    :class:`GitError` so non-rate-limit failures are never swallowed.
    """
    lowered = stderr.lower()
    is_rate_limit = (
        any(marker in lowered for marker in _RATE_LIMIT_MARKERS)
        or re.search(r"\b429\b", lowered) is not None
        or ("403" in lowered and "rate" in lowered)
    )
    if not is_rate_limit:
        return GitError(message)
    retry_after: float | None = None
    match = re.search(r"retry[- ]after[:\s]+(\d+)", lowered)
    if match:
        retry_after = float(match.group(1))
    return RateLimitError(message, retry_after=retry_after)


def gh_api(
    repo: Path,
    endpoint: str,
    *,
    method: str = "GET",
    paginate: bool = False,
    input_data: Any | None = None,
) -> Any:
    """Call ``gh api <endpoint>`` and return parsed JSON.

    Args:
        repo: Repository working directory.
        endpoint: API path (e.g. ``"repos/owner/repo/pulls/1/comments"``).
        method: HTTP method. Defaults to ``"GET"``.
        paginate: When True, pass ``--paginate`` to walk all result pages.
        input_data: Optional JSON-serialisable payload. When provided, it is
            written to a temporary file and passed via ``--input <path>`` and
            the call uses ``--method <method>`` (gh's preferred form). On
            success the tempfile is removed; on failure it is preserved and
            its path is included in the raised :class:`GitError` so callers
            can inspect the exact request body that was sent.

    Returns:
        The parsed JSON value (object, list, or scalar).

    Raises:
        RateLimitError: If the call fails due to a GitHub API rate limit
            (detected from the ``gh`` stderr marker-set).
        GitError: If the call fails for any other reason or returns invalid JSON.
    """
    if input_data is None:
        args = ["api"]
        if method.upper() != "GET":
            args.extend(["-X", method.upper()])
        if paginate:
            args.append("--paginate")
        args.append(endpoint)
        proc = _run_gh(repo, args, timeout=60)
        if proc.returncode != 0:
            raise _gh_error_for(f"gh api {endpoint} failed: {proc.stderr.strip()}", proc.stderr)
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise GitError(f"gh api {endpoint} returned invalid JSON: {exc}") from exc

    # input_data path: serialise to a tempfile and shell out via `--input`.
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - lifecycle managed below
        suffix=".json", mode="w", delete=False, encoding="utf-8"
    )
    tmp_path = Path(tmp.name)
    succeeded = False
    try:
        try:
            json.dump(input_data, tmp)
        finally:
            tmp.close()
        args = ["api", endpoint, "--method", method.upper(), "--input", str(tmp_path)]
        if paginate:
            args.append("--paginate")
        proc = _run_gh(repo, args, timeout=60)
        if proc.returncode != 0:
            raise _gh_error_for(
                f"gh api {endpoint} failed: {proc.stderr.strip()} "
                f"(request payload preserved at {tmp_path})",
                proc.stderr,
            )
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise GitError(
                f"gh api {endpoint} returned invalid JSON: {exc} "
                f"(request payload preserved at {tmp_path})"
            ) from exc
        succeeded = True
        return result
    finally:
        if succeeded:
            tmp_path.unlink(missing_ok=True)
