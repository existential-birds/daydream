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
      the host is overloaded, not that the command hung. Only read-only queries
      are retried; mutating operations (fetch/checkout/clean/worktree/amend)
      pass ``retries=0`` because re-running a non-idempotent command after a
      timeout could happen on top of partial repo changes.

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
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

# Lowercased substrings that identify a GitHub API rate-limit response in ``gh``
# stderr. "429" is intentionally absent: HTTP 429 is matched separately in
# ``_gh_error_for`` with a word-boundary regex to avoid false positives on
# arbitrary digit sequences (URLs, SHAs, file sizes) containing those digits.
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "secondary rate limit",
)

# Module-level singleton for the ``gh`` subprocess environment. Set once at run
# entry from GitHub App credentials; read by ``_run_gh`` so every ``gh`` call
# authenticates under the minted token. ``None`` means inherit the parent env.
# Access only through the getter/setter/reset functions below.
_gh_token_env: dict[str, str] | None = None


def set_gh_token_env(env: dict[str, str] | None) -> None:
    """Set the environment overrides passed to ``gh`` subprocesses.

    Args:
        env: Mapping of env-var overrides (e.g. ``{"GH_TOKEN": token}``) merged
            with the live ``os.environ`` at subprocess call time, or ``None`` to
            inherit the parent process environment without any overrides.

    Returns:
        None

    """
    global _gh_token_env
    _gh_token_env = env


def get_gh_token_env() -> dict[str, str] | None:
    """Get the environment currently passed to ``gh`` subprocesses.

    Returns:
        The environment mapping, or ``None`` when ``gh`` inherits the parent
        process environment.

    """
    return _gh_token_env


def reset_gh_token_env() -> None:
    """Reset the ``gh`` subprocess environment to parent-process inheritance.

    Returns:
        None

    """
    global _gh_token_env
    _gh_token_env = None


# Header names whose values are secrets. ``gh api -H "Authorization: Bearer
# <jwt>"`` carries a minted App/installation token as a plain argument, so any
# log line or error message that joins raw args masks these values.
_SENSITIVE_HEADER_PREFIXES = ("authorization:",)


def _redact_args(args: list[str]) -> list[str]:
    """Mask secret-bearing args (e.g. ``Authorization: Bearer <jwt>``).

    The token is passed as a plain ``gh``/``git`` argument, so joining raw args
    into a warning or :class:`GitError` message would leak it into logs. The
    header name is kept for debuggability; only the value is replaced.

    Args:
        args: The argument list passed after ``gh``/``git``.

    Returns:
        A copy with sensitive header values replaced by ``***``.
    """
    redacted: list[str] = []
    for arg in args:
        if any(arg.lower().startswith(prefix) for prefix in _SENSITIVE_HEADER_PREFIXES):
            redacted.append(f"{arg.split(':', 1)[0]}: ***")
        else:
            redacted.append(arg)
    return redacted


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


# A trivial git command timing out means the host was CPU-starved, not that the
# command hung — so retry a bounded number of times. Without this, under load a
# 5s `git rev-parse`/`diff` would time out and exit the run 1 (see #120).
_GIT_TIMEOUT_RETRIES = 2

# `gh` calls time out under the same host CPU starvation as git (see #120), so
# read-only `gh` invocations retry too. Mutations must NOT inherit this: `gh`
# cannot tell an idempotent GraphQL *query* from a *mutation* by HTTP method
# (both are POST), so retry is opt-in per caller — the default is 0 attempts,
# keeping every existing mutating caller unretried unless it explicitly opts in.
#
# Both budgets are read from the environment at call time, not frozen at import,
# so a test harness can shrink them: a fake `gh` shim must never be granted the
# production network-sized 60s budget, which under host CPU starvation turns a
# sub-second call into a 3 x 60s = 180s stall (the fake-gh pre-push flake).
_GH_DEFAULT_TIMEOUT = 60
_GH_DEFAULT_RETRIES = 2


def _gh_timeout() -> int:
    """Default ``gh`` subprocess timeout in seconds (env-overridable)."""
    raw = os.environ.get("DAYDREAM_GH_TIMEOUT_SECONDS")
    return int(raw) if raw else _GH_DEFAULT_TIMEOUT


def _gh_retries() -> int:
    """Read-only ``gh`` timeout-retry budget (env-overridable)."""
    raw = os.environ.get("DAYDREAM_GH_TIMEOUT_RETRIES")
    return int(raw) if raw else _GH_DEFAULT_RETRIES


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
            Mutating wrappers pass ``retries=0`` because re-running a
            non-idempotent git command after a timeout is unsafe; only
            read-only queries inherit the retrying default.

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
    timeout: int | None = None,
    input_text: str | None = None,
    retries: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Run ``gh`` in *repo* with hardened defaults.

    Args:
        repo: Repository working directory.
        args: Arguments after ``gh``.
        timeout: Subprocess timeout in seconds. ``None`` (the default) uses the
            env-overridable :func:`_gh_timeout`.
        input_text: Optional text piped to the subprocess on **stdin** (used to
            pass secret values via ``gh secret set --body-file -`` so the value
            never appears in the process argument vector).
        retries: How many additional attempts to make after a
            :class:`subprocess.TimeoutExpired` (total attempts = ``retries + 1``).
            Only timeouts are retried; other failures raise immediately. Defaults
            to ``0`` so a non-idempotent ``gh`` call (e.g. ``pr create``,
            ``secret set``, a GraphQL mutation) is never re-run after a timeout.
            Read-only callers pass :func:`_gh_retries` to ride out host CPU
            starvation.

    The subprocess environment is sourced from the module ``_gh_token_env``
    singleton when set (via :func:`set_gh_token_env`), so ``gh`` authenticates
    under the minted installation token. When the singleton is ``None``, ``env``
    is ``None`` and ``gh`` inherits the parent process environment.

    Returns:
        The completed process with text-decoded stdout/stderr.

    Raises:
        GitTimeoutError: If every attempt times out. Subclass of
            :class:`GitError`.
        GitError: If the subprocess machinery fails for any other reason
            (missing ``gh``, OS-level error).
    """
    if timeout is None:
        timeout = _gh_timeout()
    token_env = get_gh_token_env()
    env = {**os.environ, **token_env} if token_env is not None else None
    last_timeout: subprocess.TimeoutExpired | None = None
    for attempt in range(retries + 1):
        try:
            return subprocess.run(  # noqa: S603 - arguments are not user-controlled
                ["gh", *args],  # noqa: S607 - gh is a trusted command
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                check=False,
                input=input_text,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            last_timeout = exc
            if attempt < retries:
                _logger.warning(
                    "gh %s timed out after %ss (attempt %d/%d); retrying",
                    " ".join(_redact_args(args)),
                    timeout,
                    attempt + 1,
                    retries + 1,
                )
        except (subprocess.SubprocessError, OSError) as exc:
            raise GitError(f"gh {' '.join(_redact_args(args))} failed: {type(exc).__name__}: {exc}") from exc

    suffix = f" ({retries + 1} attempts)" if retries else ""
    raise GitTimeoutError(
        f"gh {' '.join(_redact_args(args))} timed out after {timeout}s{suffix}"
    ) from last_timeout


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

    # Run directly, not via _run_git, because interpret-trailers needs stdin.
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
        repo, ["commit", "--amend", "-m", amended_msg.strip()], timeout=30, retries=0,
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


def diff_worktree_against(repo: Path, ref: str, paths: list[str]) -> str:
    """Return ``git diff <ref> -- <paths>`` (working tree vs *ref* for *paths*).

    Unlike :func:`diff`/:func:`diff_paths` (which compare two refs), this diffs
    the *current working tree* against *ref*, restricted to *paths*. Used to
    snapshot a path's uncommitted partial-edit content before it is reverted, so
    the patch is recoverable even after the working file is restored.

    Args:
        repo: Repository working directory.
        ref: Ref to diff against (e.g. ``"HEAD"`` or a ``git stash create`` SHA).
        paths: Repo-relative pathspec list.

    Returns:
        The diff text. Empty string when *paths* match *ref* exactly (or when a
        path is untracked at *ref*, which ``git diff <ref> --`` does not show).

    Raises:
        GitError: If ``git diff`` fails.
    """
    if not paths:
        return ""
    args = ["diff", ref, "--", *paths]
    proc = _run_git(repo, args, timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git diff {ref} -- {paths} failed in {repo}: {proc.stderr.strip()}")
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


def log_shas_since(repo: Path, head: str, base: str) -> list[str]:
    """Return full SHAs of commits on ``head..base``.

    The ``head..base`` range already bounds the walk; no ``--since`` date
    filter is needed (it was redundant and caused timeouts on large
    monorepos — see #167).

    Soft-failure semantics: returns ``[]`` on git error, but logs a
    warning so a degraded fix-applied verdict is not silent.

    Args:
        repo: Repository working directory.
        head: The divergence point; commits reachable from *head* are excluded.
        base: The tip ref; only commits reachable from *base* are included.

    Returns:
        List of 40-character SHA strings in ``git log`` output order
        (newest first). Empty list on any soft failure.
    """
    try:
        proc = _run_git(
            repo,
            ["log", "--pretty=%H", f"{head}..{base}"],
            timeout=30,
        )
    except GitError:
        _logger.warning(
            "log_shas_since: git log %s..%s failed after retries; "
            "returning empty window (fix-applied verdict may degrade to unknown)",
            head,
            base,
        )
        return []
    if proc.returncode != 0:
        _logger.warning(
            "log_shas_since: git log %s..%s exited non-zero (%d); "
            "returning empty window (fix-applied verdict may degrade to unknown)",
            head,
            base,
            proc.returncode,
        )
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


def list_untracked(repo: Path) -> list[str]:
    """Return repo-relative paths of untracked, non-ignored files.

    Soft-failure semantics mirror :func:`changed_files`: returns ``[]`` on any
    git error or non-zero exit. Used to snapshot the untracked set before a fix
    pass so newly-orphaned files created by a failed group can be detected.

    Args:
        repo: Repository working directory.

    Returns:
        Untracked file paths (``git ls-files --others --exclude-standard``).
    """
    try:
        proc = _run_git(repo, ["ls-files", "--others", "--exclude-standard"], timeout=10)
    except GitError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def stash_create(repo: Path) -> str | None:
    """Capture tracked working-tree + index changes as a dangling commit.

    Runs ``git stash create``, which builds a stash commit object WITHOUT
    touching the working tree or the stash ref list. The returned SHA names a
    snapshot of the current tracked state; pass it to :func:`diff_worktree_against`
    or :func:`restore_paths_from_ref` to capture or restore a single path's
    pre-mutation content. Untracked files are NOT included (``git stash create``
    ignores them), so callers track those separately via :func:`list_untracked`.

    Args:
        repo: Repository working directory.

    Returns:
        The 40-character snapshot SHA, or ``None`` when the tree has no tracked
        changes (``git stash create`` prints nothing). A ``None`` result means
        the pre-mutation tracked state equals ``HEAD``.

    Raises:
        GitError: If ``git stash create`` fails.
    """
    proc = _run_git(repo, ["stash", "create"], timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git stash create failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip() or None


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
    proc = _run_git(repo, ["fetch", remote], timeout=30, retries=0)
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
    proc = _run_git(repo, ["fetch", remote, refspec], timeout=timeout, retries=0)
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
    proc = _run_git(repo, ["checkout", "--detach", sha], timeout=timeout, retries=0)
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
    proc = _run_git(repo, args, timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git checkout -- {paths} failed in {repo}: {proc.stderr.strip()}")


def restore_paths_from_ref(repo: Path, ref: str, paths: list[str]) -> None:
    """Restore *paths* to their content at *ref* (``git checkout <ref> -- <paths>``).

    Discards working-tree edits for exactly *paths*, replacing them with the
    *ref* version (and staging that version). Distinct from :func:`checkout_paths`,
    which restores from the index (``git checkout -- <paths>``) rather than a ref.
    Used to roll a single path back to its pre-fix content after a fix group
    failed mid-edit, leaving the rest of the tree untouched.

    Args:
        repo: Repository working directory.
        ref: Ref to restore from (e.g. ``"HEAD"`` or a ``git stash create`` SHA).
        paths: Repo-relative paths to restore. No-op when empty.

    Raises:
        GitError: If the checkout fails (e.g. a path absent at *ref*, which is
            the signal that the path was newly created and untracked).
    """
    if not paths:
        return
    args = ["checkout", ref, "--", *(str(p) for p in paths)]
    proc = _run_git(repo, args, timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git checkout {ref} -- {paths} failed in {repo}: {proc.stderr.strip()}")


def clean_untracked(repo: Path) -> None:
    """Run ``git clean -fd`` to remove untracked files and directories.

    Args:
        repo: Repository working directory.

    Raises:
        GitError: If the clean fails.
    """
    proc = _run_git(repo, ["clean", "-fd"], timeout=30, retries=0)
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
    proc = _run_git(repo, args, timeout=30, retries=0)
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
    proc = _run_git(repo, args, timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git worktree remove {path} failed: {proc.stderr.strip()}")


def create_branch(repo: Path, name: str) -> None:
    """Create and check out a new branch *name* via ``git checkout -b``.

    Args:
        repo: Repository working directory.
        name: The branch name to create and switch to.

    Raises:
        GitError: If the branch already exists (``git checkout -b`` refuses to
            overwrite it) or the checkout otherwise fails. The caller decides
            whether to reuse or force the branch.
    """
    proc = _run_git(repo, ["checkout", "-b", name], timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git checkout -b {name} failed in {repo}: {proc.stderr.strip()}")


def checkout_branch(repo: Path, name: str) -> None:
    """Switch to an existing local branch *name*, or track it from origin.

    Uses ``git checkout <name>`` when the branch exists locally (``refs/heads/<name>``),
    or ``git checkout -b <name> origin/<name>`` when it exists only on the remote.

    Args:
        repo: Repository working directory.
        name: An existing branch name (local or ``origin/<name>``).

    Raises:
        GitError: If the checkout fails, or the branch does not exist either
            locally or on the remote.
    """
    local = _run_git(repo, ["rev-parse", "--verify", f"refs/heads/{name}"], timeout=5)
    if local.returncode == 0:
        proc = _run_git(repo, ["checkout", name], timeout=30, retries=0)
    else:
        proc = _run_git(repo, ["checkout", "-b", name, f"origin/{name}"], timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git checkout {name} failed in {repo}: {proc.stderr.strip()}")


def commit_paths(repo: Path, paths: list[Path], message: str) -> None:
    """Stage only *paths* and commit them with *message*.

    Stages exactly the named paths via ``git add <paths…>`` (never ``-A`` /
    ``--all``) so only the intended files are committed, then commits with
    ``git commit -m <message>``.

    Args:
        repo: Repository working directory.
        paths: Repo-relative paths to stage and commit. Must be non-empty.
        message: The commit message.

    Raises:
        GitError: If *paths* is empty, or the ``git add`` / ``git commit`` call
            fails.
    """
    if not paths:
        raise GitError("commit_paths requires at least one path")
    add = _run_git(repo, ["add", "--", *(str(p) for p in paths)], timeout=30, retries=0)
    if add.returncode != 0:
        raise GitError(f"git add {paths} failed in {repo}: {add.stderr.strip()}")
    # git commit fails "Author identity unknown" with no user.email/user.name
    # (common in fresh CI). Inject fallback values via -c only when none is set.
    identity_ok = (
        _run_git(repo, ["config", "user.email"], timeout=5).returncode == 0
        and _run_git(repo, ["config", "user.name"], timeout=5).returncode == 0
    )
    commit_args = ["commit", "-m", message]
    if not identity_ok:
        commit_args = [
            "-c", "user.email=daydream@localhost",
            "-c", "user.name=daydream",
            *commit_args,
        ]
    commit = _run_git(repo, commit_args, timeout=30, retries=0)
    if commit.returncode != 0:
        raise GitError(f"git commit failed in {repo}: {commit.stderr.strip()}")


def push_branch(repo: Path, branch: str, *, remote: str = "origin") -> None:
    """Push *branch* to *remote*, setting upstream tracking.

    Runs ``git push -u <remote> <branch>``.

    Args:
        repo: Repository working directory.
        branch: The branch to push.
        remote: Remote name. Defaults to ``"origin"``.

    Raises:
        GitError: If the push fails (propagates stderr).
    """
    proc = _run_git(repo, ["push", "-u", remote, branch], timeout=60, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git push -u {remote} {branch} failed in {repo}: {proc.stderr.strip()}")


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
    try:
        proc = _run_gh(repo, args, retries=_gh_retries())
    except GitError as exc:
        _logger.warning("gh pr view failed (%s, returning None): %s", type(exc).__name__, exc)
        return None
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
        retries=_gh_retries(),
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
    proc = _run_gh(repo, ["pr", "diff", str(pr)], retries=_gh_retries())
    if proc.returncode != 0:
        raise GitError(f"gh pr diff {pr} failed: {proc.stderr.strip()}")
    return proc.stdout


def split_owner_repo(slug: str) -> tuple[str, str] | None:
    """Split an ``"owner/repo"`` slug into its components.

    Args:
        slug: A GitHub ``"owner/repo"`` string.

    Returns:
        A ``(owner, repo)`` tuple when *slug* contains exactly one ``"/"``
        and both parts are non-empty, or ``None`` otherwise.
    """
    if "/" not in slug:
        return None
    owner, _, repo = slug.partition("/")
    if not owner or not repo:
        return None
    return owner, repo


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
        retries=_gh_retries(),
    )
    if proc.returncode != 0:
        return None
    return split_owner_repo(proc.stdout.strip())


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
    jq: str | None = None,
    headers: dict[str, str] | None = None,
    idempotent: bool = False,
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
        jq: Optional ``gh --jq`` filter. The filtered stdout is parsed as
            NDJSON (one JSON value per line) and returned as a list. With
            ``paginate=True`` gh concatenates each page's raw JSON, which is
            not itself valid JSON for array endpoints — a filter like ``".[]"``
            flattens every page to one value per line instead.
        headers: Optional extra request headers passed via ``gh api -H``. An
            explicit ``Authorization`` header takes precedence over the
            ``token``-scheme header gh derives from ``GH_TOKEN`` — required
            for App JWT calls, which GitHub only accepts as ``Bearer``.
        idempotent: When True, the call is retried on timeout (host CPU
            starvation). Set this only for reads — GET endpoints and GraphQL
            *queries* — never for mutations, since ``method``/``input_data``
            alone cannot distinguish a GraphQL query from a mutation (both POST).

    Returns:
        The parsed JSON value (object, list, or scalar); with *jq*, a list of
        the filtered values.

    Raises:
        RateLimitError: If the call fails due to a GitHub API rate limit
            (detected from the ``gh`` stderr marker-set).
        GitError: If the call fails for any other reason or returns invalid JSON.
    """
    header_args = [arg for name, value in (headers or {}).items() for arg in ("-H", f"{name}: {value}")]
    if input_data is None:
        args = ["api", *header_args]
        if method.upper() != "GET":
            args.extend(["-X", method.upper()])
        if paginate:
            args.append("--paginate")
        if jq is not None:
            args.extend(["--jq", jq])
        args.append(endpoint)
        proc = _run_gh(repo, args, retries=_gh_retries() if idempotent else 0)
        if proc.returncode != 0:
            raise _gh_error_for(f"gh api {endpoint} failed: {proc.stderr.strip()}", proc.stderr)
        try:
            if jq is not None:
                return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
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
        args = ["api", *header_args, endpoint, "--method", method.upper(), "--input", str(tmp_path)]
        if paginate:
            args.append("--paginate")
        if jq is not None:
            args.extend(["--jq", jq])
        proc = _run_gh(repo, args, retries=_gh_retries() if idempotent else 0)
        if proc.returncode != 0:
            raise _gh_error_for(
                f"gh api {endpoint} failed: {proc.stderr.strip()} "
                f"(request payload preserved at {tmp_path})",
                proc.stderr,
            )
        try:
            if jq is not None:
                result = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
            else:
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


# --- gh secret / variable / PR primitives ------------------------------------


def _scope_args(org: str | None, repo_slug: str | None) -> list[str]:
    """Build the ``--org``/``--repo`` scope flags, requiring exactly one.

    Raises:
        GitError: If neither or both of *org* and *repo_slug* are provided.
    """
    if (org is None) == (repo_slug is None):
        raise GitError("exactly one of org or repo_slug must be provided")
    return ["--org", org] if org is not None else ["--repo", repo_slug or ""]


def gh_secret_set(
    repo: Path,
    name: str,
    value: str,
    *,
    org: str | None = None,
    repo_slug: str | None = None,
) -> None:
    """Set an Actions secret via ``gh secret set <name> --body-file -``.

    The *value* is piped on **stdin** (never the argument vector) so secret
    material such as a PEM private key cannot leak into process listings.

    Args:
        repo: Repository working directory (ambient ``gh`` auth context).
        name: The secret name.
        value: The secret value; piped on stdin via ``--body-file -``.
        org: Set at the organization scope (``--org``).
        repo_slug: Set at the repository scope (``--repo <owner/repo>``).

    Raises:
        GitError: If neither/both scopes are given, or the ``gh`` call fails.
    """
    args = ["secret", "set", name, "--body-file", "-", *_scope_args(org, repo_slug)]
    proc = _run_gh(repo, args, input_text=value)
    if proc.returncode != 0:
        raise _gh_error_for(f"gh secret set {name} failed: {proc.stderr.strip()}", proc.stderr)


def gh_variable_set(
    repo: Path,
    name: str,
    value: str,
    *,
    org: str | None = None,
    repo_slug: str | None = None,
) -> None:
    """Set an Actions variable via ``gh variable set <name> --body <value>``.

    Variables are non-secret handles, so the value is passed via ``--body``.

    Args:
        repo: Repository working directory.
        name: The variable name.
        value: The variable value.
        org: Set at the organization scope (``--org``).
        repo_slug: Set at the repository scope (``--repo <owner/repo>``).

    Raises:
        GitError: If neither/both scopes are given, or the ``gh`` call fails.
    """
    args = ["variable", "set", name, "--body", value, *_scope_args(org, repo_slug)]
    proc = _run_gh(repo, args)
    if proc.returncode != 0:
        raise _gh_error_for(f"gh variable set {name} failed: {proc.stderr.strip()}", proc.stderr)


def _gh_name_list(repo: Path, kind: str, org: str | None, repo_slug: str | None) -> list[str]:
    """Run ``gh <kind> list --json name`` and return the names."""
    args = [kind, "list", "--json", "name", *_scope_args(org, repo_slug)]
    proc = _run_gh(repo, args, retries=_gh_retries())
    if proc.returncode != 0:
        raise _gh_error_for(f"gh {kind} list failed: {proc.stderr.strip()}", proc.stderr)
    try:
        entries = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GitError(f"gh {kind} list returned invalid JSON: {exc}") from exc
    return [entry["name"] for entry in entries]


def gh_secret_list(repo: Path, *, org: str | None = None, repo_slug: str | None = None) -> list[str]:
    """Return the names of Actions secrets at the given scope.

    Args:
        repo: Repository working directory.
        org: List at the organization scope (``--org``).
        repo_slug: List at the repository scope (``--repo <owner/repo>``).

    Returns:
        The secret names (values are not exposed by ``gh secret list``).

    Raises:
        GitError: If neither/both scopes are given, or the ``gh`` call fails.
    """
    return _gh_name_list(repo, "secret", org, repo_slug)


def gh_variable_list(repo: Path, *, org: str | None = None, repo_slug: str | None = None) -> list[str]:
    """Return the names of Actions variables at the given scope.

    Args:
        repo: Repository working directory.
        org: List at the organization scope (``--org``).
        repo_slug: List at the repository scope (``--repo <owner/repo>``).

    Returns:
        The variable names.

    Raises:
        GitError: If neither/both scopes are given, or the ``gh`` call fails.
    """
    return _gh_name_list(repo, "variable", org, repo_slug)


def gh_pr_create(
    repo: Path, *, head: str, base: str, title: str, body: str, repo_slug: str | None = None
) -> str:
    """Open a pull request via ``gh pr create`` and return its URL.

    Args:
        repo: Repository working directory.
        head: The head branch to open the PR from.
        base: The base branch to merge into.
        title: The PR title.
        body: The PR body.
        repo_slug: Explicit ``owner/repo`` target (``--repo``).  When *None*
            the ambient ``gh`` context (cwd) is used.

    Returns:
        The PR URL ``gh`` prints on success.

    Raises:
        GitError: If the ``gh pr create`` call fails (stderr included).
    """
    args = ["pr", "create", "--head", head, "--base", base, "--title", title, "--body", body]
    if repo_slug is not None:
        args += ["--repo", repo_slug]
    proc = _run_gh(repo, args)
    if proc.returncode != 0:
        raise _gh_error_for(f"gh pr create failed: {proc.stderr.strip()}", proc.stderr)
    return proc.stdout.strip()
