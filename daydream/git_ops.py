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
    Examples: :func:`remote_url`, :func:`merge_base`, :func:`gh_repo_view`.

    Each function's docstring specifies which pattern it follows under its
    **Raises** or **Returns** section.

The module is intentionally dependency-free: stdlib only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Literal, overload
from urllib.parse import urlparse

_logger = logging.getLogger(__name__)

# Lowercased substrings that identify a GitHub API rate-limit response in ``gh``
# stderr. "429" is intentionally absent: HTTP 429 is matched separately in
# ``_gh_error_for`` with a word-boundary regex to avoid false positives on
# arbitrary digit sequences (URLs, SHAs, file sizes) containing those digits.
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "secondary rate limit",
)

# Module-level state for the ``gh`` subprocess environment. Configured at run
# entry from GitHub App credentials and refreshed before expiry; read by
# ``_run_gh`` so every ``gh`` call authenticates under a current installation
# token. ``None`` means inherit the parent env. Access only through the helpers
# below.
_gh_token_env: dict[str, str] | None = None
_gh_token_expires_at: float | None = None
_gh_token_refresh: Callable[[], tuple[dict[str, str], float]] | None = None

# Refresh before the credential's actual deadline so a request cannot start
# with a token that expires while GitHub is processing it.
_GH_TOKEN_REFRESH_SKEW_SECONDS = 300


def set_gh_token_env(
    env: dict[str, str] | None,
    *,
    expires_at: float | None = None,
    refresh: Callable[[], tuple[dict[str, str], float]] | None = None,
) -> None:
    """Set the environment overrides passed to ``gh`` subprocesses.

    Args:
        env: Mapping of env-var overrides (e.g. ``{"GH_TOKEN": token}``) merged
            with the live ``os.environ`` at subprocess call time, or ``None`` to
            inherit the parent process environment without any overrides.
        expires_at: Unix timestamp at which the injected token expires.
        refresh: Callback that returns a replacement environment and expiry.
    """
    global _gh_token_env, _gh_token_expires_at, _gh_token_refresh
    _gh_token_env = env
    _gh_token_expires_at = expires_at
    _gh_token_refresh = refresh


def get_gh_token_env() -> dict[str, str] | None:
    """Get the environment currently passed to ``gh`` subprocesses.

    Returns:
        The environment mapping, or ``None`` when ``gh`` inherits the parent
        process environment.
    """
    return _gh_token_env


@contextmanager
def scoped_gh_token_env(env: dict[str, str] | None) -> Generator[None, None, None]:
    """Temporarily replace the complete ``gh`` token state."""
    prior = (_gh_token_env, _gh_token_expires_at, _gh_token_refresh)
    set_gh_token_env(env)
    try:
        yield
    finally:
        set_gh_token_env(prior[0], expires_at=prior[1], refresh=prior[2])


def reset_gh_token_env() -> None:
    """Reset the ``gh`` subprocess environment to parent-process inheritance."""
    global _gh_token_env, _gh_token_expires_at, _gh_token_refresh
    _gh_token_env = None
    _gh_token_expires_at = None
    _gh_token_refresh = None


def _gh_token_env_for_request() -> dict[str, str] | None:
    """Return the injected token environment, refreshing it before expiry."""
    global _gh_token_env, _gh_token_expires_at, _gh_token_refresh
    if (
        _gh_token_env is None
        or _gh_token_expires_at is None
        or _gh_token_refresh is None
        or time.time() < _gh_token_expires_at - _GH_TOKEN_REFRESH_SKEW_SECONDS
    ):
        return _gh_token_env

    prior = (_gh_token_env, _gh_token_expires_at, _gh_token_refresh)
    try:
        env, expires_at = _gh_token_refresh()
    except Exception as exc:
        _gh_token_env, _gh_token_expires_at, _gh_token_refresh = prior
        raise GitError(f"failed to refresh GitHub App installation token: {exc}") from exc

    _gh_token_env = env
    _gh_token_expires_at = expires_at
    _gh_token_refresh = prior[2]
    return _gh_token_env


# Header names whose values are secrets. ``gh api -H "Authorization: Bearer
# <jwt>"`` carries a minted App/installation token as a plain argument, so any
# log line or error message that joins raw args masks these values.
_SENSITIVE_HEADER_PREFIXES = ("authorization:",)

# The GitHub App manifest-conversion credential rides in the single path segment
# between the literal prefix and suffix of the ``gh api`` conversion endpoint
# (e.g. ``/app-manifests/<code>/conversions``). Mask that segment in any
# rendered diagnostic so the code never leaks verbatim, while preserving the
# route for debuggability.
_APP_MANIFEST_CONVERSION_CODE_RE = re.compile(r"(/app-manifests/)[^/\s]+(/conversions)")
_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
)
# Start at the fixed authority delimiter. Searching for an arbitrary-length
# scheme at every input character makes long non-URL diagnostics quadratic.
_URL_USERINFO_RE = re.compile(r"://[^/@\s]+@")


def _redact_sensitive_text(text: str) -> str:
    """Mask secrets in *diagnostic* text only; never mutate a live request.

    Replaces the credential segment of any ``/app-manifests/<code>/conversions``
    endpoint with ``***`` (e.g. ``/app-manifests/***/conversions``), preserving
    the surrounding route. This is a pure string transform for diagnostics such
    as error messages and log lines — it must never be applied to the ``args``
    list passed to :func:`subprocess.run`, which keeps the original endpoint so
    GitHub still receives the real credential.
    """
    redacted = _APP_MANIFEST_CONVERSION_CODE_RE.sub(r"\1***\2", text)
    redacted = _GITHUB_TOKEN_RE.sub("***", redacted)
    redacted = _URL_USERINFO_RE.sub("://***@", redacted)
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(key)
        if token and len(token) >= 8:
            redacted = redacted.replace(token, "***")
    return redacted


def _redact_args(args: list[str]) -> list[str]:
    """Mask secret-bearing args for diagnostics (header values + manifest codes).

    Secrets travel as plain ``gh``/``git`` arguments, so joining raw args into a
    warning or :class:`GitError` message would leak them into logs. This masks
    sensitive header values (e.g. ``Authorization: Bearer <jwt>``, keeping the
    header name for debuggability) and manifest-conversion endpoint segments. It
    never mutates the input ``args`` list.

    Returns:
        A copy with sensitive header values and manifest-conversion codes
        replaced by ``***``.
    """
    redacted: list[str] = []
    for arg in args:
        if any(arg.lower().startswith(prefix) for prefix in _SENSITIVE_HEADER_PREFIXES):
            redacted.append(f"{arg.split(':', 1)[0]}: ***")
        else:
            redacted.append(arg)
    return [_redact_sensitive_text(a) for a in redacted]


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


@dataclass(frozen=True)
class GitPathState:
    """Binary-safe identity for one exact repository path."""

    path: str
    state: Literal["missing", "regular", "symlink", "gitlink"]
    mode: int | None
    digest: str | None


@dataclass(frozen=True)
class IndexSnapshot:
    """A complete index tree plus its HEAD-relative changed path set."""

    tree_sha: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class WorktreeRollbackSnapshot:
    """One round's tracked, untracked, and index rollback point."""

    ref: str
    index: IndexSnapshot
    path_states: tuple[GitPathState, ...]
    untracked: dict[str, GitPathState]


# --- Internal subprocess helpers --------------------------------------------


# A trivial git command timing out means the host was CPU-starved, not that the
# command hung — so retry a bounded number of times. Without this, under load a
# 5s `git rev-parse`/`diff` would time out and exit the run 1 (see #120).
_GIT_TIMEOUT_RETRIES = 2

# Command-scoped git credential helper backed by ambient ``gh`` auth. Git calls
# this as a shell helper with the operation (get/store/erase) and
# ``protocol``/``host`` on stdin; ``gh auth git-credential`` resolves stored
# auth internally so no token ever lands on argv, in a URL, in a file, or in
# global/local git config. Read at call time (see :func:`_credential_helper_args`)
# so a test harness can substitute a recording wrapper.
GH_CREDENTIAL_HELPER = "!gh auth git-credential"

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


def _gh_int_env(name: str, default: int, *, is_valid: Callable[[int], bool], invalid_msg: str) -> int:
    """Read *name* from the environment as an int, falling back to *default*.

    A missing, non-integer, or invalid (per :func:`is_valid`) value logs a
    warning and returns *default*, never raising.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _logger.warning(
            "%s=%r is not a valid integer; using default %d",
            name,
            raw,
            default,
        )
        return default
    if not is_valid(value):
        _logger.warning(
            "%s=%r %s; using default %d",
            name,
            raw,
            invalid_msg,
            default,
        )
        return default
    return value


def _gh_timeout() -> int:
    """Default ``gh`` subprocess timeout in seconds (env-overridable)."""
    return _gh_int_env(
        "DAYDREAM_GH_TIMEOUT_SECONDS",
        _GH_DEFAULT_TIMEOUT,
        is_valid=lambda value: value > 0,
        invalid_msg="must be positive",
    )


def _gh_retries() -> int:
    """Read-only ``gh`` timeout-retry budget (env-overridable)."""
    return _gh_int_env(
        "DAYDREAM_GH_TIMEOUT_RETRIES",
        _GH_DEFAULT_RETRIES,
        is_valid=lambda value: value >= 0,
        invalid_msg="is negative",
    )


@overload
def _run_git(
    repo: Path,
    args: list[str],
    *,
    timeout: int = 5,
    capture_bytes: Literal[True],
    retries: int = _GIT_TIMEOUT_RETRIES,
    input_text: str | None = None,
    input_bytes: bytes | None = None,
    env_cmd: Any | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Binary-capture variant: ``capture_bytes=True`` reads bytes stdout."""


@overload
def _run_git(
    repo: Path,
    args: list[str],
    *,
    timeout: int = 5,
    capture_bytes: Literal[False] = False,
    retries: int = _GIT_TIMEOUT_RETRIES,
    input_text: str | None = None,
    input_bytes: bytes | None = None,
    env_cmd: Any | None = None,
) -> subprocess.CompletedProcess[str]:
    """Text-capture variant (the default): decoded ``str`` stdout."""


def _run_git(
    repo: Path,
    args: list[str],
    *,
    timeout: int = 5,
    capture_bytes: bool = False,
    retries: int = _GIT_TIMEOUT_RETRIES,
    input_text: str | None = None,
    input_bytes: bytes | None = None,
    env_cmd: Any | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Run ``git`` in *repo* with hardened defaults.

    Args:
        capture_bytes: When True, capture stdout/stderr as bytes (no decoding).
        input_text: Optional text piped to the subprocess on **stdin** (used
            for ``git update-ref --stdin`` batch transactions whose payload
            cannot express the whole ref set on argv). Encoded to UTF-8 when
            *capture_bytes* is set so the binary variant never feeds ``str``
            to the subprocess.
        input_bytes: Optional raw bytes piped to stdin. Requires
            ``capture_bytes=True`` and is mutually exclusive with
            ``input_text``.
        retries: How many additional attempts to make after a
            :class:`subprocess.TimeoutExpired` (total attempts = ``retries + 1``).
            Only timeouts are retried; other failures raise immediately.
            Mutating wrappers pass ``retries=0`` because re-running a
            non-idempotent git command after a timeout is unsafe; only
            read-only queries inherit the retrying default.
        env_cmd: Optional environment mapping for the subprocess (default None
            inherits the parent environment). Preflight read-only helpers pass
            ``GIT_TERMINAL_PROMPT=0`` here so a credential failure is surfaced
            rather than prompting on stdin.

    Returns:
        The completed process. ``returncode`` is left to the caller to inspect.

    Raises:
        GitTimeoutError: If every attempt times out. Subclass of
            :class:`GitError`, so existing ``except GitError`` handlers still
            catch it.
        GitError: If the underlying subprocess machinery fails for any other
            reason (missing binary, OS-level error).
    """
    if input_text is not None and input_bytes is not None:
        raise GitError("git subprocess input must be text or bytes, not both")
    if input_bytes is not None and not capture_bytes:
        raise GitError("binary git subprocess input requires binary capture")
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
                # capture_bytes=True reads bytes stdout/stderr, so a str
                # input_text must be encoded: subprocess.run raises a raw
                # TypeError for str input with text=False.
                input=(
                    input_bytes
                    if input_bytes is not None
                    else input_text.encode("utf-8")
                    if capture_bytes and input_text is not None
                    else input_text
                ),
                env=env_cmd,
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
        timeout: Subprocess timeout in seconds. ``None`` (the default) uses the
            env-overridable :func:`_gh_timeout`.
        input_text: Optional text piped to the subprocess on **stdin** (used to
            pass secret values to ``gh secret set``, which reads stdin when
            ``--body`` is omitted, so the value never appears in the process
            argument vector).
        retries: How many additional attempts to make after a
            :class:`subprocess.TimeoutExpired` (total attempts = ``retries + 1``).
            Only timeouts are retried; other failures raise immediately. Defaults
            to ``0`` so a non-idempotent ``gh`` call (e.g. ``pr create``,
            ``secret set``, a GraphQL mutation) is never re-run after a timeout.
            Read-only callers pass :func:`_gh_retries` to ride out host CPU
            starvation.

    The subprocess environment is sourced from the module token state when set
    (via :func:`set_gh_token_env`), so ``gh`` authenticates under the minted
    installation token. Expiring tokens are refreshed before the subprocess is
    launched. When no token state is configured, ``env`` is ``None`` and ``gh``
    inherits the parent process environment.

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
    token_env = _gh_token_env_for_request()
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
            raise GitError(
                f"gh {' '.join(_redact_args(args))} failed: {type(exc).__name__}: {_redact_sensitive_text(str(exc))}"
            ) from exc

    suffix = f" ({retries + 1} attempts)" if retries else ""
    raise GitTimeoutError(
        f"gh {' '.join(_redact_args(args))} timed out after {timeout}s{suffix}"
    ) from last_timeout


# --- Pre-flight --------------------------------------------------------------


def assert_is_worktree(repo: Path) -> None:
    """Verify *repo* is the root of a git worktree.

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
    """Return True iff :func:`assert_is_worktree` would succeed for *repo*."""
    try:
        assert_is_worktree(repo)
    except NotAWorktreeError:
        return False
    return True


# --- Read-only queries -------------------------------------------------------


def head_sha(repo: Path) -> str:
    """Return the full SHA of ``HEAD`` in *repo*.

    Raises:
        GitError: If ``git rev-parse HEAD`` fails (e.g. empty repository).
    """
    proc = _run_git(repo, ["rev-parse", "HEAD"], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"cannot resolve HEAD in {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def has_executable_pre_push_hook(repo: Path) -> bool:
    """Return True when *repo* has an executable ``pre-push`` hook.

    Honors ``core.hooksPath`` when configured; otherwise resolves the default
    hooks directory via ``git rev-parse --git-path hooks`` (falling back to
    ``<repo>/.git/hooks`` when *repo* is not a resolvable git repository). The hook counts
    only when the file exists **and** is executable. Absence of the directory
    or file is a normal ``False`` (no exception); a git failure resolving the
    hooks path silently falls back to ``<repo>/.git/hooks`` so "no executable
    hook" stays a plain ``False`` answer.
    """
    proc = _run_git(repo, ["rev-parse", "--git-path", "hooks"], timeout=5)
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
    """Return a snapshot of local branch names mapped to their full OIDs.

    Read-only. Runs ``git for-each-ref refs/heads`` and parses each output
    line into ``short_name -> full OID``. An empty dict is returned only when
    the repository genuinely has zero local branches and git exits 0; any
    non-zero return code raises.

    Raises:
        GitError: If ``git for-each-ref`` fails (e.g. *repo* is not a
            repository).
    """
    proc = _run_git(
        repo,
        ["for-each-ref", "refs/heads", "--format=%(refname:short) %(objectname)"],
        timeout=10,
    )
    if proc.returncode != 0:
        raise GitError(
            f"cannot list local branches in {repo}: {proc.stderr.strip()}",
        )
    branches: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        name, _, oid = line.strip().partition(" ")
        branches[name] = oid
    return branches


def head_commit_message(repo: Path) -> str:
    """Return the full commit message of ``HEAD``.

    Raises:
        GitError: If ``git log`` fails (e.g. empty repository).
    """
    proc = _run_git(repo, ["log", "-1", "--format=%B", "HEAD"], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"cannot read HEAD message in {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def remote_url(repo: Path, remote: str = "origin") -> str | None:
    """Return the URL configured for *remote*, or ``None`` when unset/missing.

    Soft-failure semantics: returns ``None`` on any non-zero ``git`` exit
    (mirrors :func:`default_branch` / :func:`merge_base` / :func:`current_branch`
    behavior for "the data isn't there" cases) and on subprocess machinery
    failures (timeout, missing binary).
    """
    try:
        proc = _run_git(repo, ["config", "--get", f"remote.{remote}.url"], timeout=5)
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
    proc = _run_git(
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
        url_proc = _run_git(repo, ["config", "--get", f"remote.{name}.url"], timeout=5)
        url = url_proc.stdout.strip() if url_proc.returncode == 0 else ""
        if not url:
            raise GitError(f"remote {name!r} has no readable fetch URL in {repo}")
        result[name] = url
    return result


def validate_branch_name(repo: Path, name: str) -> None:
    """Raise when *name* is not a literal Git branch name."""
    proc = _run_git(repo, ["check-ref-format", "--branch", name], timeout=5)
    if proc.returncode != 0:
        display = name[:120] + ("..." if len(name) > 120 else "")
        raise GitError(f"invalid PR base branch name {display!r} in {repo}")


def current_branch(repo: Path) -> str | None:
    """Return the current branch name, or ``None`` when ``HEAD`` is detached.

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
    """Check whether *ref* exists locally or as ``origin/<ref>``."""
    local = _run_git(repo, ["rev-parse", "--verify", f"refs/heads/{ref}"], timeout=5)
    if local.returncode == 0:
        return True
    remote = _run_git(repo, ["rev-parse", "--verify", f"refs/remotes/origin/{ref}"], timeout=5)
    return remote.returncode == 0


def _has_leading_dash(*refs: str) -> bool:
    """True if any *refs* starts with ``-``, which git would mis-parse as an option flag.

    No valid branch name or commit-ish starts with ``-``, so callers reject
    these early rather than letting an attacker-controlled string reach the
    shell.
    """
    return any(ref.startswith("-") for ref in refs)


def ref_exists(repo: Path, ref: str) -> bool:
    """Check whether *ref* resolves to a commit git can name.

    Accepts a named branch (local or ``origin/<ref>``) plus any commit-ish:
    a full or abbreviated SHA, a tag, or a relative expression such as
    ``HEAD~3``.

    Raises:
        GitError: Only for unexpected subprocess failures (timeout, missing
            git binary). The documented soft-failure modes return ``False``.
    """
    if branch_exists(repo, ref):
        return True
    # No valid commit-ish (SHA, tag, relative expression) starts with '-'.
    # A leading dash would be mis-parsed by git as an option flag; reject it
    # early rather than letting an attacker-controlled string reach the shell.
    if _has_leading_dash(ref):
        return False
    commit = _run_git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"], timeout=5)
    return commit.returncode == 0


def commit_exists(repo: Path, revision: str) -> bool:
    """Check whether *revision* resolves locally to a named commit.

    This is the ``git rev-parse --verify <revision>^{commit}`` probe: only commits
    git can name via normal revision resolution are accepted. Unlike
    :func:`ref_exists`, it does NOT treat a bare branch name that exists only
    as a remote-tracking ``origin/<revision>`` as existing -- git's short-name
    resolution does not fall back to ``refs/remotes/origin`` for a name without
    a slash, so such a name fails this probe and callers report it "invalid".

    Returns ``False`` (rather than raising) on the soft-failure modes: missing
    ref, a tree/blob that is not a commit, or a leading-dash ref that would be
    mis-parsed as an option flag.

    Raises:
        GitError: Only for unexpected subprocess failures (timeout, missing
            git binary). The documented soft-failure modes return ``False``.
    """
    # Same guard as ref_exists: no valid commit-ish starts with '-'.
    if _has_leading_dash(revision):
        return False
    commit = _run_git(repo, ["rev-parse", "--verify", f"{revision}^{{commit}}"], timeout=5)
    return commit.returncode == 0


def is_ancestor(repo: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    """Check whether *ancestor* is an ancestor of *descendant*.

    Returns ``False`` (rather than raising) on the soft-failure modes:
    missing ref, unrelated histories, or a leading-dash ref that would be
    mis-parsed by git as an option flag.

    Raises:
        GitError: Only for unexpected subprocess failures (timeout, missing
            git binary). The documented soft-failure modes return ``False``.
    """
    # Same guard as ref_exists and merge_base: a leading '-' would be
    # mis-parsed by git as an option flag. No valid branch name or commit-ish
    # starts with '-'.
    if _has_leading_dash(ancestor, descendant):
        return False

    proc = _run_git(repo, ["merge-base", "--is-ancestor", ancestor, descendant], timeout=5)
    return proc.returncode == 0


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

    Raises:
        GitError: Only for unexpected subprocess failures (timeout, missing
            git binary). The documented soft-failure modes return ``None``.
    """
    # Same guard as ref_exists: a leading '-' would be mis-parsed as a git
    # option flag.  No valid branch name or commit-ish starts with '-'.
    if _has_leading_dash(head, base):
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
        preferred_ref = upstream

    mb = _run_git(repo, ["merge-base", head, preferred_ref], timeout=5)
    if mb.returncode != 0:
        return None
    out = mb.stdout.strip()
    return out or None


def _upstream_and_ahead(repo: Path, branch: str) -> tuple[str | None, int]:
    """Return ``(<branch>@{upstream} symbolic name, right-side ahead count)``.

    Parses the right-side count from ``rev-list --left-right --count
    <branch>...<upstream>``. Soft-failure semantics: returns ``(None, 0)`` when
    *branch* has no configured upstream or the rev-list count cannot be read.
    """
    upstream_name_proc = _run_git(
        repo,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", f"{branch}@{{upstream}}"],
        timeout=5,
    )
    if upstream_name_proc.returncode != 0:
        return None, 0
    upstream = upstream_name_proc.stdout.strip()
    if not upstream:
        return None, 0

    counts_proc = _run_git(
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


def _resolve_upstream_if_remote_ahead(repo: Path, branch: str) -> str | None:
    """Return ``<branch>@{upstream}`` iff it has commits the local branch lacks.

    Mirrors codex's ``resolve_upstream_if_remote_ahead``: returns the upstream
    symbolic name when the right-side count from :func:`_upstream_and_ahead`
    is positive.
    """
    upstream, ahead = _upstream_and_ahead(repo, branch)
    return upstream if ahead > 0 else None


def _prefer_remote_base(repo: Path, base: str) -> str:
    """Return ``origin/<base>`` when it exists, otherwise *base* unchanged."""
    remote_ref = f"origin/{base}"
    check = _run_git(repo, ["rev-parse", "--verify", f"refs/remotes/{remote_ref}"], timeout=5)
    if check.returncode == 0:
        return remote_ref
    return base


def diff(repo: Path, base: str, head: str = "HEAD", *, exclude: list[str] | None = None) -> str:
    """Return the unified diff between *base* and *head*, plus tracked worktree changes.

    Uses three-dot syntax (``base...head``) so the diff reflects changes on
    *head* since it diverged from *base*.  When ``origin/<base>`` exists the
    function prefers it via :func:`_prefer_remote_base` so the diff is
    computed against the remote default branch — not a potentially stale
    (or identical-to-HEAD) local copy.

    Args:
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
    result = proc.stdout

    # The range above is committed-only. Include tracked index/worktree changes
    # as well so callers that persist a diff fingerprint can reject resumes
    # after the live workspace has changed.
    worktree_args = ["diff", head]
    if exclude:
        worktree_args.append("--")
        worktree_args.append(".")
        worktree_args.extend(f":(exclude){p.rstrip('/')}" for p in exclude)
    proc = _run_git(repo, worktree_args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git diff {head} failed: {proc.stderr.strip()}")
    result += proc.stdout

    return result


def diff_name_only(repo: Path, base: str, head: str = "HEAD") -> list[str]:
    """Return the list of paths changed between *base* and *head*.

    Uses ``git diff --name-only base..head`` (two-dot) so the result is the
    direct set of files differing between the two refs at archive time.

    Soft-failure semantics mirror :func:`merge_base`: returns an empty list
    when either ref cannot be resolved or the subprocess fails. Callers in
    archive paths should not propagate git transients into manifest failure.

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


def capture_recommended_patch(
    repo: Path, base_ref: str | None, out_path: Path, *, preexisting_untracked: set[str] | None = None
) -> bool:
    """Write daydream's proposed diff (``base_ref`` → working tree) to *out_path*.

    Captures the *recommended-change patch*: the difference between the pre-fix
    tree named by *base_ref* and the current working tree — i.e. the edits
    daydream applied during the fix phase. This is distinct from ``diff.patch``,
    which is the PR-under-review diff captured before any fix ran.

    *base_ref* must be resolved by the caller BEFORE fixes run, as
    ``pre_fix_snapshot or pre_fix_head_sha``: :func:`stash_create` returns
    ``None`` on a clean tree (the common pre-fix case), so the pre-fix ``HEAD``
    SHA is the fallback base. The ``HEAD`` SHA must be captured before fixes
    because the commit phase advances ``HEAD`` past the fix.

    Best-effort: writes nothing and returns ``False`` when *base_ref* is
    ``None`` or git fails. When the diff is empty (no fix landed) it writes an
    *empty* marker file (and returns ``False``) so the run is distinguishable
    from a legacy archive, which has no ``recommended.patch`` at all —
    otherwise ``_read_recommended_patch`` falls back to ``diff.patch`` (the
    PR-under-review diff) and mislabels a no-recommendation run as "applied".
    Never raises.

    Args:
        base_ref: Pre-fix base ref (a ``stash create`` SHA or a ``HEAD`` SHA),
            or ``None`` when no pre-fix snapshot could be taken.
        preexisting_untracked: Set of repo-relative paths that were untracked
            BEFORE the fix ran. Matching untracked files contribute no creation
            hunk (they are not daydream's changes); files absent from the set
            still do. Order follows :func:`list_untracked`, never re-sorted.
            ``None`` (default) excludes nothing, preserving legacy behavior.

    Returns:
        ``True`` when a non-empty patch was written, else ``False``.
    """
    if not base_ref:
        return False
    try:
        recommended = diff_worktree_against(repo, base_ref, ["."])
    except GitError:
        return False
    # `git diff <base_ref>` only reports tracked-file changes, so new files the
    # fix phase created are absent. Append a creation hunk for each via
    # `git diff --no-index /dev/null <file>` (exit 1 = "files differ", expected).
    # Pre-fix untracked files (preexisting_untracked) are filtered out first so
    # files that were already untracked before the run never enter the patch.
    try:
        untracked = _filter_preexisting_untracked(list_untracked(repo), preexisting_untracked)
        for rel in untracked:
            proc = _run_git(repo, ["diff", "--no-index", "/dev/null", rel], timeout=30, retries=0)
            if proc.returncode in (0, 1):
                recommended += proc.stdout
    except GitError:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not recommended.strip():
        # Empty marker: a present-but-empty recommended.patch distinguishes a
        # no-fix run from a legacy archive (no file at all), preventing the
        # diff.patch fallback in _read_recommended_patch. Returns False since
        # no non-empty patch was written.
        out_path.write_text("")
        return False
    out_path.write_text(recommended)
    return True


def capture_recommended_patch_with_base(
    repo: Path,
    pre_fix_snapshot: str | None,
    pre_fix_head: str | None,
    out_path: Path,
    *,
    preexisting_untracked: set[str] | None = None,
) -> bool:
    """Capture the recommended patch, resolving the pre-fix base centrally.

    Thin wrapper over :func:`capture_recommended_patch` that resolves the
    pre-fix base as ``pre_fix_snapshot or pre_fix_head`` in one place, so the
    deep and shallow fix paths cannot drift apart in how they pick the base.

    *pre_fix_snapshot* is a ``stash create`` SHA of the tracked tree taken
    before fixes; :func:`stash_create` returns ``None`` on a clean tree (the
    common pre-fix case), so *pre_fix_head* -- the pre-fix ``HEAD`` SHA -- is
    the fallback base. *pre_fix_head* is therefore only consulted when the
    snapshot is ``None`` and need not be captured otherwise. Both must be
    captured *before* the fix/commit phase, which advances ``HEAD`` past the
    fix.

    Best-effort: never raises (see :func:`capture_recommended_patch`).

    Args:
        pre_fix_snapshot: ``stash create`` SHA, or ``None`` when the tree was
            clean or the snapshot failed.
        pre_fix_head: Pre-fix ``HEAD`` SHA, or ``None``. Used only when
            *pre_fix_snapshot* is ``None``.
        preexisting_untracked: Set of repo-relative paths that were untracked
            BEFORE the fix ran, forwarded to
            :func:`capture_recommended_patch` so they contribute no creation
            hunk. ``None`` (default) excludes nothing.

    Returns:
        ``True`` when a non-empty patch was written, else ``False``.
    """
    base_ref = pre_fix_snapshot or pre_fix_head
    return capture_recommended_patch(
        repo, base_ref, out_path, preexisting_untracked=preexisting_untracked
    )


def log(repo: Path, base: str, head: str = "HEAD") -> str:
    """Return the one-line commit log for ``base..head``.

    Returns:
        Stripped ``--oneline`` log output. Empty string when no commits.

    Raises:
        GitError: If ``git log`` fails.
    """
    proc = _run_git(repo, ["log", f"{base}..{head}", "--oneline"], timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git log {base}..{head} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def log_shas(repo: Path, ref: str, *, since: str) -> list[str] | None:
    """Return full SHAs of commits on ``since..ref``.

    Soft-failure semantics distinguish "we looked and found nothing" from
    "we could not look": a git failure (commonly a deleted branch ref or a
    squash-merged SHA that no longer resolves) returns ``None``, never
    ``[]``. Collapsing the two lets callers read an unanswerable query as a
    negative answer.

    Args:
        since: Base ref or SHA; commits reachable from *ref* but not from
            *since* are returned (``git log since..ref``).

    Returns:
        List of 40-character SHA strings in ``git log`` output order
        (newest first), possibly empty. ``None`` if the query could not be
        answered.
    """
    try:
        proc = _run_git(repo, ["log", "--pretty=%H", f"{since}..{ref}"], timeout=30)
    except GitTimeoutError:
        _logger.warning(
            "log_shas: git log %s..%s timed out after retries; "
            "returning None (commit window unavailable)",
            since,
            ref,
        )
        return None
    except GitError as exc:
        _logger.warning(
            "log_shas: git log %s..%s failed: %s; "
            "returning None (commit window unavailable)",
            since,
            ref,
            exc,
        )
        return None
    if proc.returncode != 0:
        _logger.warning(
            "log_shas: git log %s..%s exited non-zero (%d); "
            "returning None (commit window unavailable)",
            since,
            ref,
            proc.returncode,
        )
        return None
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def log_shas_since(repo: Path, head: str, base: str) -> list[str]:
    """Return full SHAs of commits on ``head..base``.

    The ``head..base`` range already bounds the walk; no ``--since`` date
    filter is needed (it was redundant and caused timeouts on large
    monorepos — see #167).

    Soft-failure semantics: returns ``[]`` on git error, but logs a
    warning so a degraded fix-applied verdict is not silent.

    Args:
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
    except GitTimeoutError:
        _logger.warning(
            "log_shas_since: git log %s..%s timed out after retries; "
            "returning empty window (fix-applied verdict may degrade to unknown)",
            head,
            base,
        )
        return []
    except GitError as exc:
        _logger.warning(
            "log_shas_since: git log %s..%s failed: %s; "
            "returning empty window (fix-applied verdict may degrade to unknown)",
            head,
            base,
            exc,
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

    Raises:
        GitError: If ``git show`` fails (e.g. path missing at that revision).
    """
    proc = _run_git(repo, ["show", f"{ref}:{path}"], timeout=30, capture_bytes=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
        raise GitError(f"git show {ref}:{path} failed: {stderr.strip()}")
    return proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()


def grep(
    repo: Path,
    pattern: str,
    *,
    word: bool = False,
    pathspecs: Sequence[str] | None = None,
) -> list[str]:
    """Return file paths matching *pattern* via ``git grep -l``.

    Only tracked, non-ignored files are searched (``git grep`` semantics).

    Args:
        repo: Repository root to search.
        pattern: Basic-regex pattern passed to ``git grep``.
        word: When true, match only at word boundaries (``-w``) so ``app``
            does not also match ``application`` or ``mapping``.
        pathspecs: Optional ``git`` pathspecs limiting the search to matching
            files (e.g. ``("*.py", "*.ts")``). When omitted, every tracked
            file is searched.

    Raises:
        GitError: If ``git grep`` exits with an unexpected status.  Exit code
            ``1`` is "no matches" and is treated as success (empty list).
    """
    args = ["grep", "-l"]
    if word:
        args.append("-w")
    args.extend(["-e", pattern])
    if pathspecs:
        args.append("--")
        args.extend(pathspecs)
    proc = _run_git(repo, args, timeout=30)
    # git grep returns 1 when there are simply no matches.
    if proc.returncode not in (0, 1):
        raise GitError(f"git grep {pattern!r} failed: {proc.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def grep_fixed_matches(
    repo: Path,
    patterns: Sequence[str],
    *,
    word: bool = False,
    pathspecs: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(path, matched_pattern)`` pairs from one batched ``git grep``.

    Runs a single ``git grep -F -o -z -f <file>`` so that every pattern is
    matched in one process invocation, emitting one ``(path, pattern)`` tuple
    per match. Patterns are deduplicated in input order; patterns containing
    NUL, CR, or LF are skipped (they cannot be expressed as a single fixed
    string in the patterns file).

    Args:
        repo: Repository root to search.
        patterns: Fixed-string patterns matched literally (``-F``).
        word: When true, match only at word boundaries (``-w``) so ``app``
            does not also match ``application`` or ``mapping``.
        pathspecs: Optional ``git`` pathspecs limiting the search to matching
            files (e.g. ``("*.py", "*.ts")``). When omitted, every tracked
            file is searched.

    Returns:
        List of ``(path, matched_pattern)`` pairs parsed from ``git grep``
        output — one tuple per match occurrence, so the same pattern may
        appear multiple times for the same file even when the matches share a
        single line (``git grep -o`` emits a record per occurrence). Only the
        input patterns are deduplicated. Empty when there are no matches or
        no usable patterns.

    Raises:
        GitError: If ``git grep`` exits with an unexpected status or emits a
            malformed record. Exit code ``1`` is "no matches" and is treated
            as success (empty list).
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
        proc = _run_git(repo, args, timeout=30, capture_bytes=True)
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
    """Return ``git status --porcelain`` output.

    Returns:
        Porcelain-formatted status text. Empty when the tree is clean.

    Raises:
        GitError: If ``git status`` fails.
    """
    proc = _run_git(repo, ["status", "--porcelain"], timeout=10)
    if proc.returncode != 0:
        raise GitError(f"git status failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout


def staged_patch(repo: Path) -> bytes:
    """Return the staged index as a binary patch (``git diff --cached --binary``).

    Byte-captured so binary content round-trips unchanged. The strict-query
    counterpart of :func:`status_porcelain`: a non-zero exit raises
    :class:`GitError` with the captured stderr text. Patch bytes are never
    embedded in exception text.

    Returns:
        The staged patch as raw bytes. Empty bytes when nothing is staged.

    Raises:
        GitError: If ``git diff --cached --binary`` fails.
    """
    proc = _run_git(repo, ["diff", "--cached", "--binary"], timeout=30, capture_bytes=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
        raise GitError(f"git diff --cached --binary failed in {repo}: {stderr.strip()}")
    return proc.stdout


def changed_files(repo: Path, *, preexisting_untracked: set[str] | None = None) -> list[str]:
    """Return repo-relative paths of files changed in the working tree.

    Best-effort: combines staged + unstaged changes (``git diff --name-only
    HEAD``) with new untracked files (``git ls-files --others
    --exclude-standard``). When ``preexisting_untracked`` is supplied, only
    untracked paths absent from that pre-fix snapshot are included. Files
    created during a daydream fix are still untracked at abort time, so the
    diff alone would omit them and leave the handoff missing critical context.

    Soft-failure semantics: returns an empty list when git is unavailable, the
    repo has no commits yet, or either subcommand fails.  Individual
    subcommand failures are logged and skipped — the other subcommand's
    results are still returned.

    Returns:
        De-duplicated list of repo-relative path strings.  Empty on error.
    """
    names: list[str] = []
    seen: set[str] = set()
    try:
        proc = _run_git(repo, ["diff", "--name-only", "HEAD"], timeout=10)
        tracked = proc.stdout.splitlines() if proc.returncode == 0 else []
    except GitError:
        tracked = []
    untracked = _filter_preexisting_untracked(list_untracked(repo), preexisting_untracked)
    for line in [*tracked, *untracked]:
        name = line.strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


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
    proc = _run_git(repo, ["diff", "--name-only", ref], timeout=10)
    if proc.returncode != 0:
        raise GitError(f"git diff --name-only {ref} failed in {repo}: {proc.stderr.strip()}")
    untracked_proc = _run_git(
        repo, ["ls-files", "--others", "--exclude-standard"], timeout=10,
    )
    if untracked_proc.returncode != 0:
        raise GitError(f"git ls-files --others failed in {repo}: {untracked_proc.stderr.strip()}")

    names: list[str] = []
    seen: set[str] = set()
    untracked = [line.strip() for line in untracked_proc.stdout.splitlines() if line.strip()]
    untracked = _filter_preexisting_untracked(untracked, preexisting_untracked)
    for line in [*proc.stdout.splitlines(), *untracked]:
        name = line.strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def diff_name_only_strict(repo: Path, from_ref: str, to_ref: str) -> list[str]:
    """Return repo-relative paths that differ between two refs' trees.

    Strict counterpart to :func:`diff_name_only` (which soft-fails to an empty
    list) for post-commit tree checks: ``git diff --name-only -z <from_ref>
    <to_ref>`` over the two commit trees. Raises :class:`GitError` when the
    diff cannot be computed, so callers can distinguish "no differences" from
    "unknown" (e.g. when verifying that a commit agent did not broaden a
    commit beyond the pre-staged set).

    Returns:
        Exact filesystem-decoded repo-relative paths, without quoting or trimming.
    """
    proc = _run_git(
        repo, ["diff", "--name-only", "-z", from_ref, to_ref],
        timeout=10, capture_bytes=True,
    )
    if proc.returncode != 0:
        raise GitError(
            f"git diff --name-only -z {from_ref} {to_ref} failed in {repo}: "
            f"{os.fsdecode(proc.stderr).strip()}"
        )
    return _decode_nul_paths(proc.stdout)


def list_untracked(repo: Path, *, strict: bool = False) -> list[str]:
    """Return repo-relative paths of untracked, non-ignored files.

    Soft-failure semantics mirror :func:`changed_files`: returns ``[]`` on any
    git error or non-zero exit. Used to snapshot the untracked set before a fix
    pass so newly-orphaned files created by a failed group can be detected.

    Args:
        strict: When True, a git error or non-zero exit raises
            :class:`GitError` instead of soft-failing to ``[]`` — used by the
            disposable read-only-checkout prep, whose error-propagation contract
            needs a reliable enumeration (a silent ``[]`` would produce a clone
            missing the untracked files).
    """
    try:
        proc = _run_git(repo, ["ls-files", "--others", "--exclude-standard"], timeout=10)
    except GitError:
        if strict:
            raise
        return []
    if proc.returncode != 0:
        if strict:
            raise GitError(
                f"git ls-files --others --exclude-standard failed in {repo}: {proc.stderr.strip()}"
            )
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _filter_preexisting_untracked(
    untracked: list[str], preexisting_untracked: set[str] | None
) -> list[str]:
    """Drop paths that were already untracked before a fix ran.

    Filters *untracked* against the *preexisting_untracked* snapshot so files
    that existed before the run (e.g. a user's scratch file) are never
    misattributed to daydream. ``None`` passes the list through unchanged,
    preserving legacy behavior. Order of *untracked* is preserved, never
    re-sorted.
    """
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


def _git_path_parent_is_confined(repo: Path, value: str) -> bool:
    """Confinement check that may inspect, but never follows, the leaf."""
    if not value or "\0" in value or value.startswith("/"):
        return False
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    root = repo.resolve()
    candidate = repo
    for part in parts[:-1]:
        candidate /= part
        try:
            if candidate.is_symlink():
                return False
            if not candidate.exists():
                break
        except OSError:
            return False
    try:
        return candidate.resolve(strict=False).is_relative_to(root)
    except OSError:
        return False


def _require_git_path_confined(repo: Path, path: str, *, allow_leaf_symlink: bool = False) -> None:
    from daydream.repository_paths import git_observed_path_is_confined

    confined = (
        _git_path_parent_is_confined(repo, path)
        if allow_leaf_symlink
        else git_observed_path_is_confined(repo, path)
    )
    if not confined:
        raise GitError("Git-observed path is not confined to the repository")


def _is_untracked_runtime_artifact(path: str) -> bool:
    from daydream.config import REVIEW_OUTPUT_FILE

    return path.startswith(".daydream/") or path == REVIEW_OUTPUT_FILE


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
    proc = _run_git(repo, ["diff", "--name-only", "-z", ref], timeout=10, capture_bytes=True)
    if proc.returncode != 0:
        stderr = os.fsdecode(proc.stderr)
        raise GitError(f"git diff --name-only -z {ref} failed in {repo}: {stderr.strip()}")
    paths = _decode_nul_paths(proc.stdout)
    if include_untracked:
        others = _run_git(
            repo,
            ["ls-files", "--others", "--exclude-standard", "-z"],
            timeout=10,
            capture_bytes=True,
        )
        if others.returncode != 0:
            stderr = os.fsdecode(others.stderr)
            raise GitError(f"git ls-files --others -z failed in {repo}: {stderr.strip()}")
        paths.extend(
            path for path in _decode_nul_paths(others.stdout)
            if include_runtime_artifacts or not _is_untracked_runtime_artifact(path)
        )
    unique = dict.fromkeys(paths)
    for path in unique:
        _require_git_path_confined(repo, path, allow_leaf_symlink=True)
    return list(unique)


def _write_git_blob(repo: Path, content: bytes) -> str:
    proc = _run_git(
        repo,
        ["hash-object", "-w", "--stdin"],
        timeout=30,
        capture_bytes=True,
        input_bytes=content,
    )
    if proc.returncode != 0:
        raise GitError(f"git hash-object failed in {repo}: {os.fsdecode(proc.stderr).strip()}")
    oid = os.fsdecode(proc.stdout).strip()
    if not oid:
        raise GitError("git hash-object returned no object id")
    return oid


def _read_git_blob(repo: Path, oid: str) -> bytes:
    proc = _run_git(repo, ["cat-file", "blob", oid], timeout=30, capture_bytes=True)
    if proc.returncode != 0:
        raise GitError(f"git cat-file blob failed in {repo}: {os.fsdecode(proc.stderr).strip()}")
    return proc.stdout


def _index_mode_for_path(repo: Path, path: str) -> int | None:
    proc = _run_git(
        repo,
        ["ls-files", "--stage", "-z", "--", _literal_pathspec(path)],
        capture_bytes=True,
    )
    if proc.returncode != 0:
        raise GitError(f"git ls-files --stage failed in {repo}: {os.fsdecode(proc.stderr).strip()}")
    records = [record for record in proc.stdout.split(b"\0") if record]
    if not records:
        return None
    if len(records) != 1:
        raise GitError("index contains unresolved entries for a captured path")
    metadata, separator, raw_path = records[0].partition(b"\t")
    if not separator or os.fsdecode(raw_path) != path:
        raise GitError("git returned an unexpected index path")
    mode, _oid, stage = metadata.decode("ascii").split(" ")
    if stage != "0":
        raise GitError("index contains an unresolved entry")
    return int(mode, 8)


def _snapshot_worktree_path(
    repo: Path,
    path: str,
    *,
    allow_leaf_symlink: bool,
) -> GitPathState:
    _require_git_path_confined(repo, path, allow_leaf_symlink=allow_leaf_symlink)
    mode_from_index = _index_mode_for_path(repo, path)
    absolute_bytes = os.fsencode(repo) + b"/" + os.fsencode(path)
    try:
        metadata = os.lstat(absolute_bytes)
    except FileNotFoundError:
        return GitPathState(path=path, state="missing", mode=None, digest=None)
    except OSError as exc:
        raise GitError("could not inspect a confined worktree path") from exc

    if mode_from_index == 0o160000 and stat.S_ISDIR(metadata.st_mode):
        nested = repo / path
        try:
            assert_is_worktree(nested)
        except NotAWorktreeError as exc:
            raise GitError("gitlink working tree is unavailable for exact evidence") from exc
        dirty = _run_git(
            nested,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
            capture_bytes=True,
        )
        if dirty.returncode != 0:
            raise GitError("could not inspect gitlink working tree for exact evidence")
        if dirty.stdout:
            # A commit OID cannot identify additional worktree content. Refuse
            # stale test evidence instead of recursively snapshotting submodules.
            raise GitError("dirty gitlink cannot provide commit-only test evidence")
        oid = head_sha(nested)
        return GitPathState(path=path, state="gitlink", mode=0o160000, digest=oid)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(absolute_bytes)
        content = target if isinstance(target, bytes) else os.fsencode(target)
        return GitPathState(path=path, state="symlink", mode=0o120000, digest=_write_git_blob(repo, content))
    if stat.S_ISREG(metadata.st_mode):
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(absolute_bytes, flags)
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode):
                    raise GitError("captured worktree path changed type during read")
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise GitError("could not read a confined worktree path") from exc
        permissions = stat.S_IMODE(metadata.st_mode)
        if mode_from_index is not None:
            permissions = 0o755 if permissions & 0o111 else 0o644
        mode = stat.S_IFREG | permissions
        return GitPathState(path=path, state="regular", mode=mode, digest=_write_git_blob(repo, b"".join(chunks)))
    raise GitError("unsupported worktree path type")


def snapshot_untracked_paths(
    repo: Path, *, include_runtime_artifacts: bool = True,
) -> dict[str, GitPathState]:
    """Capture actual untracked content/type/mode, optionally omitting runtime output."""
    proc = _run_git(
        repo,
        ["ls-files", "--others", "--exclude-standard", "-z"],
        timeout=10,
        capture_bytes=True,
    )
    if proc.returncode != 0:
        raise GitError(f"git ls-files --others -z failed in {repo}: {os.fsdecode(proc.stderr).strip()}")
    paths = _decode_nul_paths(proc.stdout)
    return {
        path: _snapshot_worktree_path(repo, path, allow_leaf_symlink=True)
        for path in paths
        if include_runtime_artifacts or not _is_untracked_runtime_artifact(path)
    }


def snapshot_worktree_paths(repo: Path, paths: Iterable[str]) -> tuple[GitPathState, ...]:
    """Capture binary-safe worktree states for exact confined paths."""
    unique = sorted(set(paths), key=_path_sort_key)
    return tuple(
        _snapshot_worktree_path(repo, path, allow_leaf_symlink=False)
        for path in unique
    )


def snapshot_worktree_gitlinks(repo: Path) -> tuple[GitPathState, ...]:
    """Capture every tracked gitlink's actual clean checked-out commit."""
    proc = _run_git(repo, ["ls-files", "--stage", "-z"], capture_bytes=True)
    if proc.returncode != 0:
        raise GitError(
            f"git ls-files --stage failed in {repo}: {os.fsdecode(proc.stderr).strip()}"
        )
    paths: list[str] = []
    for record in (record for record in proc.stdout.split(b"\0") if record):
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise GitError("git returned malformed staged path data")
        mode, _oid, stage = metadata.decode("ascii").split(" ")
        if mode == "160000" and stage == "0":
            paths.append(os.fsdecode(raw_path))
    return snapshot_worktree_paths(repo, paths)


def snapshot_worktree_delta(
    repo: Path,
    ref: str,
    *,
    preexisting_untracked: dict[str, GitPathState],
    preexisting_gitlinks: tuple[GitPathState, ...] = (),
) -> tuple[GitPathState, ...]:
    """Capture source delta plus protected paths, not changing runtime output.

    Explicitly protected paths remain visible regardless of namespace. This
    makes user-file changes invalidate evidence without audit/trace writes
    recursively invalidating the evidence they describe.
    """
    paths = (
        set(changed_paths_z(repo, ref, include_runtime_artifacts=False))
        | set(preexisting_untracked)
        | {state.path for state in preexisting_gitlinks}
    )
    return tuple(
        _snapshot_worktree_path(
            repo,
            path,
            allow_leaf_symlink=path in preexisting_untracked,
        )
        for path in sorted(paths, key=_path_sort_key)
    )


def _snapshot_git_tree_paths(
    repo: Path,
    args: list[str],
    paths: Iterable[str],
) -> tuple[GitPathState, ...]:
    unique = sorted(set(paths), key=_path_sort_key)
    for path in unique:
        _require_git_path_confined(repo, path, allow_leaf_symlink=True)
    if not unique:
        return ()
    proc = _run_git(
        repo,
        [*args, "-z", "--", *(_literal_pathspec(path) for path in unique)],
        capture_bytes=True,
    )
    if proc.returncode != 0:
        raise GitError(f"git tree-state query failed in {repo}: {os.fsdecode(proc.stderr).strip()}")
    found: dict[str, GitPathState] = {}
    for record in (record for record in proc.stdout.split(b"\0") if record):
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise GitError("git returned malformed tree-state output")
        path = os.fsdecode(raw_path)
        if path not in unique:
            raise GitError("git returned an unexpected tree-state path")
        fields = metadata.decode("ascii").split(" ")
        if args[0] == "ls-files":  # ls-files --stage: mode oid stage
            if len(fields) != 3:
                raise GitError("git returned malformed index-state metadata")
            mode_text, oid, stage_text = fields
            if stage_text != "0":
                raise GitError("index contains unresolved entries")
        elif len(fields) == 3:  # ls-tree: mode type oid
            mode_text, _object_type, oid = fields
        else:
            raise GitError("git returned malformed tree-state metadata")
        mode = int(mode_text, 8)
        state: Literal["regular", "symlink", "gitlink"]
        if mode == 0o120000:
            state = "symlink"
        elif mode == 0o160000:
            state = "gitlink"
        else:
            state = "regular"
        found[path] = GitPathState(path=path, state=state, mode=mode, digest=oid)
    return tuple(
        found.get(path, GitPathState(path=path, state="missing", mode=None, digest=None))
        for path in unique
    )


def snapshot_index(repo: Path) -> IndexSnapshot:
    """Capture the complete index tree without changing the worktree."""
    tree = _run_git(repo, ["write-tree"], timeout=30)
    if tree.returncode != 0:
        raise GitError(f"git write-tree failed in {repo}: {tree.stderr.strip()}")
    changed = _run_git(
        repo,
        ["diff", "--cached", "--name-only", "-z", "HEAD"],
        capture_bytes=True,
    )
    if changed.returncode != 0:
        raise GitError(f"git diff --cached failed in {repo}: {os.fsdecode(changed.stderr).strip()}")
    paths = tuple(sorted(set(_decode_nul_paths(changed.stdout)), key=_path_sort_key))
    return IndexSnapshot(tree_sha=tree.stdout.strip(), paths=paths)


def snapshot_index_paths(repo: Path, paths: Iterable[str]) -> tuple[GitPathState, ...]:
    """Capture exact path states from the current index."""
    return _snapshot_git_tree_paths(repo, ["ls-files", "--stage"], paths)


def snapshot_commit_paths(repo: Path, ref: str, paths: Iterable[str]) -> tuple[GitPathState, ...]:
    """Capture exact path states from a commit/tree ref."""
    return _snapshot_git_tree_paths(repo, ["ls-tree", ref], paths)


def tree_key(states: Iterable[GitPathState]) -> str:
    """Hash a canonical binary-safe sequence of content-only path states."""
    ordered = sorted(states, key=lambda state: _path_sort_key(state.path))
    digest = hashlib.sha256()
    for state in ordered:
        path = os.fsencode(state.path)
        state_bytes = state.state.encode("ascii")
        mode = b"-" if state.mode is None else format(state.mode, "o").encode("ascii")
        object_digest = b"-" if state.digest is None else state.digest.encode("ascii")
        for field_value in (path, state_bytes, mode, object_digest):
            digest.update(len(field_value).to_bytes(8, "big"))
            digest.update(field_value)
    return digest.hexdigest()


def ls_files(repo: Path, *, strict: bool = False) -> list[str]:
    """Return repo-relative paths of tracked files.

    Enumerates every file git knows about via ``git ls-files -z``, NUL-split
    and ``surrogateescape``-decoded so filenames containing newlines or invalid
    UTF-8 survive intact. This is the canonical primitive for tracked-file
    enumeration; the improve recon flow and repo-scoped exploration share it.

    Soft-failure semantics mirror :func:`list_untracked`: returns ``[]`` on any
    non-zero ``git`` exit. :class:`GitError` (timeout, missing binary)
    propagates so callers that distinguish "looked and found nothing" from
    "could not look" can catch it; best-effort callers wrap the call in
    ``try/except``.

    Args:
        strict: When True, a non-zero ``git`` exit also raises
            :class:`GitError` (instead of returning ``[]``) — used by the
            disposable read-only-checkout prep, whose error-propagation contract
            needs a reliable enumeration.

    Returns:
        Repo-relative path strings in ``git ls-files`` output order. Empty
        list on a non-zero exit (unless *strict*).
    """
    proc = _run_git(repo, ["ls-files", "-z"], capture_bytes=True)
    if proc.returncode != 0:
        if strict:
            stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
            raise GitError(f"git ls-files failed in {repo}: {stderr.strip()}")
        return []
    stdout = proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    return [
        path.decode("utf-8", errors="surrogateescape")
        for path in stdout.split(b"\0")
        if path
    ]


def stash_create(repo: Path) -> str | None:
    """Capture tracked working-tree + index changes as a dangling commit.

    Runs ``git stash create``, which builds a stash commit object WITHOUT
    touching the working tree or the stash ref list. The returned SHA names a
    snapshot of the current tracked state; pass it to :func:`diff_worktree_against`
    or :func:`restore_paths_from_ref` to capture or restore a single path's
    pre-mutation content. Untracked files are NOT included (``git stash create``
    ignores them), so callers track those separately via :func:`list_untracked`.

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


_FULL_OBJECT_ID_RE = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")


def _validate_pr_base_ref(repo: Path, ref: str, *, prefix: str) -> None:
    if not ref.startswith(prefix):
        raise GitError(f"invalid PR base ref {ref[:120]!r} in {repo}")
    proc = _run_git(repo, ["check-ref-format", ref], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"invalid PR base ref {ref[:120]!r} in {repo}")


def _merge_base_strict(repo: Path, base_ref: str, head_sha: str) -> str:
    proc = _run_git(repo, ["merge-base", base_ref, head_sha], timeout=10)
    merge_sha = proc.stdout.strip()
    if proc.returncode != 0 or _FULL_OBJECT_ID_RE.fullmatch(merge_sha) is None:
        raise GitError(
            f"no merge-base for PR head {head_sha} and base {base_ref} in {repo}"
        )
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
    head_proc = _run_git(repo, ["rev-parse", "--verify", f"{head_sha}^{{commit}}"], timeout=5)
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
        check = _run_git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"], timeout=5)
        if check.returncode == 0:
            present_remote_refs.append(ref)

    if present_remote_refs:
        bases = {
            ref: _merge_base_strict(repo, ref, head_sha)
            for ref in present_remote_refs
        }
        distinct = set(bases.values())
        if len(distinct) != 1:
            refs = ", ".join(present_remote_refs)
            raise GitError(
                f"matching PR base remotes disagree ({refs}) in {repo}; "
                "fetch/align the base remote refs"
            )
        return next(iter(distinct))

    local_check = _run_git(
        repo, ["rev-parse", "--verify", f"{local_base_ref}^{{commit}}"], timeout=5
    )
    if local_check.returncode != 0:
        raise GitError(
            f"PR base {local_base_ref} is unavailable for head {head_sha} in {repo}"
        )
    return _merge_base_strict(repo, local_base_ref, head_sha)


def upstream_ahead_count(repo: Path, branch: str) -> int:
    """Return the number of commits ``<branch>@{upstream}`` is ahead of *branch*.

    Returns:
        The right-side count from ``rev-list --left-right --count``. Returns
        ``0`` when *branch* has no configured upstream.
    """
    return _upstream_and_ahead(repo, branch)[1]


def check_ignore(repo: Path, path: str) -> bool:
    """Return True iff ``git check-ignore`` says *path* is ignored.

    Soft-failure semantics: returns ``False`` on any subprocess error (timeout,
    missing binary, OS-level failure) to avoid blocking callers that use this
    for optional file-copy filtering.
    """
    try:
        proc = _run_git(repo, ["check-ignore", "--quiet", path], timeout=5)
    except GitError:
        return False
    return proc.returncode == 0


# --- Mutating ----------------------------------------------------------------


def fetch(repo: Path, remote: str = "origin") -> None:
    """Run ``git fetch`` against *remote*.

    Raises:
        GitError: If the fetch fails.
    """
    proc = _run_git(repo, ["fetch", remote], timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git fetch {remote} failed in {repo}: {proc.stderr.strip()}")


def remove_remote(repo: Path, remote: str = "origin") -> None:
    """Remove *remote* from *repo* (``git remote remove``).

    Mutating wrapper — ``retries=0`` so a timed-out removal is never re-run.
    Callers only invoke this on a fresh clone, which always has *remote*
    configured, so no soft "no such remote" path is needed.

    Raises:
        GitError: If ``git remote remove`` fails.
    """
    proc = _run_git(repo, ["remote", "remove", remote], timeout=10, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git remote remove {remote} failed in {repo}: {proc.stderr.strip()}")


_OBJECT_ID_RE = re.compile(r"[0-9a-fA-F]{40}")


def _validate_ref_oid_pair(ref: str, oid: str) -> None:
    """Validate one *ref* -> *oid* pair with the shared fail-closed guards.

    Every ref-writing mutator runs these checks **before** any shell-out, so
    the two mutators (:func:`update_ref` and :func:`update_refs`) can never
    drift apart. In order: *oid* must be a full 40-hex object ID; *ref* must
    not start with ``-`` (git would parse it as an option, never as a ref);
    *ref*'s final component must not end in the literal lowercase ``.lock``
    suffix — git's own rule, which accepts case-variants like
    ``release.LOCK``/``x.lOck``, so those stay valid; and *ref* must not
    contain whitespace or control characters (the ``check-ref-format`` rules
    that, in the batch variant, would otherwise split a line or smuggle extra
    ``update`` lines into the ``update-ref --stdin`` payload).

    Raises:
        GitError: If *ref* or *oid* is invalid, naming the offender.
    """
    if _OBJECT_ID_RE.fullmatch(oid) is None:
        raise GitError(f"invalid OID: {oid}")
    if ref.startswith("-"):
        raise GitError(f"invalid ref name: {ref}")
    if ref.rsplit("/", 1)[-1].endswith(".lock"):
        raise GitError(f"invalid ref name: {ref}")
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in ref):
        raise GitError(f"invalid ref name: {ref}")


def update_ref(repo: Path, ref: str, oid: str) -> None:
    """Point *ref* at the explicit *oid* in *repo* (``git update-ref <ref> <oid>``).

    Fail-closed mutating wrapper — ``retries=0`` so a timed-out ref mutation is
    never re-run. Both arguments are validated **before** any shell-out:

    * *ref* is checked with ``git check-ref-format`` (git's own ref-format
      authority), plus the shared Python-side guards of
      :func:`_validate_ref_oid_pair` (also used by :func:`update_refs`):
      names beginning with ``-`` (git would parse them as options, never as a
      ref), names whose final component ends in the literal lowercase
      ``.lock`` suffix — git forbids exactly that, so ``unlock``, ``block``,
      ``deadlock`` and ``xLock`` are valid names and snapshot cleanly, while
      ``topic.lock`` stays rejected and case-variants git accepts
      (``topic.LOCK``, ``x.lOck``, ``release.LOCK``) stay valid — and names
      containing whitespace or control characters. Anything invalid raises
      ``GitError`` without invoking ``update-ref``.

    * *oid* must be a full 40-character hex object ID (upper- or lowercase);
      anything else raises ``GitError`` and is never passed through to git.

    The ``update-ref`` subprocess never substitutes a fallback value: a
    non-zero exit raises ``GitError`` with git's stderr.

    Raises:
        GitError: If *ref* or *oid* is invalid or ``git update-ref`` fails.
    """
    _validate_ref_oid_pair(ref, oid)
    # check-ref-format is a read-only query, so it inherits the retrying
    # default; only the update-ref mutation below passes retries=0.
    proc = _run_git(repo, ["check-ref-format", ref], timeout=10)
    if proc.returncode != 0:
        raise GitError(f"invalid ref name: {ref}")
    proc = _run_git(repo, ["update-ref", ref, oid], timeout=10, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git update-ref {ref} failed in {repo}: {proc.stderr.strip()}")


def update_refs(repo: Path, ref_oids: dict[str, str]) -> None:
    """Point many *ref* -> *oid* pairs at once (``git update-ref --stdin``).

    Fail-closed mutating wrapper for the branch-snapshot loop — ``retries=0``
    so a timed-out ref mutation is never re-run. Every pair is validated
    **before** any shell-out with the same shared guards as
    :func:`update_ref` via :func:`_validate_ref_oid_pair` (full 40-hex OID;
    ref must not start with ``-``; final component must not end in the literal
    ``.lock`` suffix — git's own rule, which accepts case-variants like
    ``topic.LOCK``/``x.lOck`` so those stay valid; no whitespace or control
    characters in the ref). Git's own ref-format validation — the same authority
    ``check-ref-format`` consults — then gates the whole batch as a single
    transaction, so either every ref is written or (if any line is invalid)
    none are: a half-written snapshot is never produced. An empty *ref_oids*
    is a no-op.

    Raises:
        GitError: If any ref or OID is invalid or the batch fails.
    """
    if not ref_oids:
        return
    for ref, oid in ref_oids.items():
        _validate_ref_oid_pair(ref, oid)
    lines = "".join(f"update {ref} {oid}\n" for ref, oid in ref_oids.items())
    proc = _run_git(
        repo, ["update-ref", "--stdin"], timeout=30, retries=0, input_text=lines
    )
    if proc.returncode != 0:
        raise GitError(f"git update-ref --stdin failed in {repo}: {proc.stderr.strip()}")


def apply_staged_patch(repo: Path, patch: bytes) -> None:
    """Apply a staged index patch to *repo*'s index (``git apply --cached --binary``).

    Mutating wrapper — ``retries=0`` so a timed-out apply is never re-run.
    The patch is written to a temp file (``_run_git`` cannot pass stdin) and
    unlinked in a ``finally``. Patch bytes are never embedded in exception
    text; a ``NamedTemporaryFile`` write failure raises ``OSError`` uncaught.

    Raises:
        GitError: If ``git apply --cached --binary`` fails.
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".patch") as fh:
            fh.write(patch)
            tmp_path = fh.name
        proc = _run_git(repo, ["apply", "--cached", "--binary", tmp_path], timeout=30, retries=0)
        if proc.returncode != 0:
            raise GitError(f"git apply --cached --binary failed in {repo}: {proc.stderr.strip()}")
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)


def fetch_ref(repo: Path, refspec: str, remote: str = "origin", *, timeout: int = 300) -> None:
    """Fetch a single *refspec* from *remote* into *repo*.

    Useful for fetching refs that are not covered by the default fetch
    configuration, such as ``refs/pull/<N>/head`` on GitHub.

    Args:
        timeout: Subprocess timeout in seconds. Defaults to 300 s to
            accommodate first-run blobless fetches of large repositories.

    Raises:
        GitError: If the fetch fails for any reason.
    """
    proc = _run_git(repo, ["fetch", remote, refspec], timeout=timeout, retries=0)
    if proc.returncode != 0:
        from daydream.trajectory import redact_text

        raise GitError(
            f"git fetch {_safe_url_desc(remote)} {refspec} failed in {repo}: {redact_text(proc.stderr.strip())}"
        )


def checkout_detach(repo: Path, sha: str, *, timeout: int = 300) -> None:
    """Detach HEAD onto *sha* in *repo*.

    Args:
        timeout: Subprocess timeout in seconds.  Defaults to 300 s because
            detaching HEAD in a blobless clone triggers lazy blob fetches that
            can take several minutes on large repositories.

    Raises:
        GitError: If the checkout fails.
    """
    proc = _run_git(repo, ["checkout", "--detach", sha], timeout=timeout, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git checkout --detach {sha} failed in {repo}: {proc.stderr.strip()}")


def _safe_url_desc(url: str) -> str:
    """Reduce a remote URL to a credential-free description (host/path).

    Issue #981: error messages must never echo a URL that may carry userinfo
    credentials. HTTP(S) URLs are reduced to ``host/path``; anything else
    (local paths, scp-form remotes) passes through :func:`redact_text`.
    """
    from daydream.trajectory import redact_text

    try:
        parsed = urlparse(url)
    except ValueError:
        return redact_text(url)
    if parsed.scheme and parsed.hostname:
        return f"{parsed.hostname}{parsed.path}"
    return redact_text(url)


def _run_clone(
    remote_url: str, cmd: list[str], timeout: int, env: dict[str, str] | None = None
) -> None:
    """Run a prepared ``git clone`` argv and map failures to :class:`GitError`.

    Shared by :func:`clone` and :func:`clone_with_token`, which differ only in
    argv prefix and environment. ``env=None`` inherits the parent environment,
    matching :func:`clone`'s current behavior.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
            cmd,  # noqa: S607 - git is a trusted command
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        from daydream.trajectory import redact_text

        raise GitError(
            f"git clone {_safe_url_desc(remote_url)} failed: {type(exc).__name__}: {redact_text(str(exc))}"
        ) from exc
    if proc.returncode != 0:
        from daydream.trajectory import redact_text

        raise GitError(
            f"git clone {_safe_url_desc(remote_url)} failed: {redact_text(proc.stderr.strip())}"
        )


def clone_with_token(
    remote_url: str,
    target: Path,
    token: str | None = None,
    *,
    blobless: bool = False,
    timeout: int = 300,
) -> None:
    """Clone a reconstructed identity URL with optional out-of-band auth.

    Issue #981: the URL must already be credential-free (a canonical HTTPS
    identity like ``https://github.com/owner/repo``). When *token* is set,
    auth is injected out-of-band via the git-config environment variables
    (``GIT_CONFIG_*``) carrying the base64 ``Authorization`` header — never on
    argv, in the URL, or in a config file. Without a token, plain clone
    applies (ambient credential helper). ``GIT_TERMINAL_PROMPT=0`` fails
    closed on auth errors instead of prompting.
    """
    cmd = ["git"]
    env: dict[str, str] = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if token:
        # Issue #981: auth travels out-of-band via git config environment
        # variables, never on argv. The base64 Authorization header is
        # trivially recoverable, so putting it on argv (via -c
        # http.extraHeader) would leak the token; GIT_CONFIG_* keeps it out of
        # the command line, honoring the contract that no token lands on argv.
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
            }
        )
    cmd.append("clone")
    if blobless:
        cmd.append("--filter=blob:none")
    cmd += [remote_url, str(target)]
    _run_clone(remote_url, cmd, timeout, env=env)


def clone(remote_url: str, target: Path, *, blobless: bool = False, timeout: int = 300) -> None:
    """Run ``git clone <remote_url> <target>``.

    Args:
        remote_url: Remote URL or local path to clone from.
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
    _run_clone(remote_url, cmd, timeout)


def checkout_paths(repo: Path, paths: list[Path]) -> None:
    """Run ``git checkout -- <paths>`` to discard local changes for *paths*.

    Args:
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


def restore_worktree_paths_from_ref(repo: Path, ref: str, paths: Iterable[str]) -> None:
    """Restore exact paths from *ref* without changing the index."""
    unique = sorted(set(paths), key=_path_sort_key)
    if not unique:
        return
    expected = snapshot_commit_paths(repo, ref, unique)
    for state in expected:
        _require_git_path_confined(repo, state.path, allow_leaf_symlink=state.state == "symlink")
    proc = _run_git(
        repo,
        [
            "restore",
            f"--source={ref}",
            "--worktree",
            "--no-overlay",
            "--",
            *(_literal_pathspec(path) for path in unique),
        ],
        timeout=30,
        retries=0,
    )
    if proc.returncode != 0:
        raise GitError(f"git restore --worktree from ref failed in {repo}: {proc.stderr.strip()}")


def _remove_confined_leaf(repo: Path, path: str) -> None:
    _require_git_path_confined(repo, path, allow_leaf_symlink=True)
    absolute = os.fsencode(repo) + b"/" + os.fsencode(path)
    try:
        metadata = os.lstat(absolute)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise GitError("could not inspect path before confined removal") from exc
    try:
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            shutil.rmtree(absolute)
        else:
            os.unlink(absolute)
    except OSError as exc:
        raise GitError("could not remove confined worktree path") from exc


def _restore_path_state(
    repo: Path,
    state: GitPathState,
    *,
    allow_leaf_type_replacement: bool,
) -> None:
    if state.state == "missing":
        _remove_confined_leaf(repo, state.path)
        return
    _require_git_path_confined(
        repo,
        state.path,
        allow_leaf_symlink=allow_leaf_type_replacement or state.state == "symlink",
    )
    if state.state == "gitlink":
        nested = _preflight_gitlink_restore(repo, state)
        assert state.digest is not None
        proc = _run_git(
            nested,
            ["checkout", "--detach", state.digest],
            timeout=30,
            retries=0,
        )
        if proc.returncode != 0:
            raise GitError(
                f"could not restore gitlink {state.path!r} to its captured commit: "
                f"{proc.stderr.strip()}"
            )
        if head_sha(nested) != state.digest:
            raise GitError(f"gitlink {state.path!r} did not reach its captured commit")
        _preflight_gitlink_restore(repo, state)
        return
    if state.digest is None or state.mode is None:
        raise GitError("restorable path state is incomplete")
    content = _read_git_blob(repo, state.digest)
    _remove_confined_leaf(repo, state.path)
    absolute = os.fsencode(repo) + b"/" + os.fsencode(state.path)
    parent = os.path.dirname(absolute)
    try:
        os.makedirs(parent, exist_ok=True)
        if state.state == "symlink":
            os.symlink(content, absolute)
            return
        descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IMODE(state.mode))
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
        finally:
            os.close(descriptor)
        os.chmod(absolute, stat.S_IMODE(state.mode))
    except OSError as exc:
        raise GitError("could not restore confined worktree path") from exc


def _preflight_gitlink_restore(repo: Path, state: GitPathState) -> Path:
    """Prove a gitlink can be restored without discarding nested user state."""
    if state.digest is None or state.mode != 0o160000:
        raise GitError("restorable gitlink state is incomplete")
    _require_git_path_confined(repo, state.path, allow_leaf_symlink=False)
    nested = repo / state.path
    try:
        assert_is_worktree(nested)
    except NotAWorktreeError as exc:
        raise GitError("gitlink working tree is unavailable for exact restoration") from exc
    dirty = _run_git(
        nested,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
        capture_bytes=True,
    )
    if dirty.returncode != 0:
        raise GitError("could not inspect gitlink before exact restoration")
    if dirty.stdout:
        raise GitError("dirty gitlink cannot be restored without discarding nested user state")
    target = _run_git(
        nested,
        ["cat-file", "-e", f"{state.digest}^{{commit}}"],
        timeout=30,
        retries=0,
    )
    if target.returncode != 0:
        raise GitError(f"captured gitlink commit is unavailable for {state.path!r}")
    return nested


def restore_group_from_snapshot(
    repo: Path,
    snapshot: WorktreeRollbackSnapshot,
    paths: Iterable[str],
) -> None:
    """Restore every requested group path and the supplied round index."""
    requested = sorted(set(paths), key=_path_sort_key)
    tracked = {state.path: state for state in snapshot.path_states}
    committed = {
        state.path: state
        for state in snapshot_commit_paths(
            repo,
            snapshot.ref,
            [path for path in requested if path not in tracked and path not in snapshot.untracked],
        )
    }
    restore_states: list[tuple[GitPathState, bool]] = []
    for path in requested:
        if path in snapshot.untracked:
            state = snapshot.untracked[path]
            replace_type = True
        elif path in tracked:
            state = tracked[path]
            replace_type = False
        else:
            state = committed[path]
            replace_type = state.state == "missing"
        restore_states.append((state, replace_type))

    try:
        # Preflight every nested repository before changing any worktree path.
        # A dirty or unavailable gitlink is a fail-closed condition, never
        # grounds for a forced checkout that could destroy user content.  The
        # complete parent index is still restored by ``finally``.
        for state, _replace_type in restore_states:
            if state.state == "gitlink":
                _preflight_gitlink_restore(repo, state)
        for state, replace_type in restore_states:
            _restore_path_state(
                repo,
                state,
                allow_leaf_type_replacement=replace_type,
            )
    finally:
        restore_index(repo, snapshot.index)


def restore_index(repo: Path, snapshot: IndexSnapshot) -> None:
    """Restore a complete index tree without modifying the worktree."""
    proc = _run_git(repo, ["read-tree", snapshot.tree_sha], timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git read-tree failed in {repo}: {proc.stderr.strip()}")


def build_recommended_patch_strict(
    repo: Path,
    base_ref: str,
    retained_paths: Iterable[str],
) -> bytes:
    """Build deterministic binary-capable presentation output for exact paths."""
    unique = sorted(set(retained_paths), key=_path_sort_key)
    if not unique:
        return b""
    base_states = {state.path: state for state in snapshot_commit_paths(repo, base_ref, unique)}
    current_states = {state.path: state for state in snapshot_worktree_paths(repo, unique)}
    chunks: list[bytes] = []
    for path in unique:
        if base_states[path].state == "missing" and current_states[path].state != "missing":
            proc = _run_git(
                repo,
                ["diff", "--no-index", "--binary", "--full-index", "--", "/dev/null", path],
                timeout=30,
                capture_bytes=True,
            )
            if proc.returncode not in {0, 1}:
                raise GitError(f"git diff --no-index --binary failed in {repo}: {os.fsdecode(proc.stderr).strip()}")
        else:
            proc = _run_git(
                repo,
                [
                    "diff",
                    "--binary",
                    "--full-index",
                    "--no-ext-diff",
                    base_ref,
                    "--",
                    _literal_pathspec(path),
                ],
                timeout=30,
                capture_bytes=True,
            )
            if proc.returncode != 0:
                raise GitError(f"git diff --binary failed in {repo}: {os.fsdecode(proc.stderr).strip()}")
        chunks.append(proc.stdout)
    return b"".join(chunks)


def clean_untracked(repo: Path) -> None:
    """Run ``git clean -fd`` to remove untracked files and directories.

    Raises:
        GitError: If the clean fails.
    """
    proc = _run_git(repo, ["clean", "-fd"], timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git clean -fd failed in {repo}: {proc.stderr.strip()}")


def worktree_add(
    repo: Path,
    path: Path,
    ref: str,
    *,
    detach: bool = True,
    lock_reason: str | None = None,
) -> None:
    """Create a new worktree at *path* pointing at *ref*.

    Args:
        path: Filesystem path for the new worktree (must not already exist).
        detach: When True, pass ``--detach`` so the new worktree is detached.
        lock_reason: When set, pass ``--lock --reason <lock_reason>`` so the
            worktree is created already-locked in the same ``git worktree add``
            invocation. Arming the lock atomically with creation closes the
            window in which a concurrent prune could force-remove the fresh
            worktree between an add and a separate lock call.

    Raises:
        GitError: If ``git worktree add`` fails.
    """
    args = ["worktree", "add"]
    if detach:
        args.append("--detach")
    if lock_reason is not None:
        args.append("--lock")
        args.extend(["--reason", lock_reason])
    args.extend([str(path), ref])
    proc = _run_git(repo, args, timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git worktree add {path} {ref} failed: {proc.stderr.strip()}")


def worktree_remove(repo: Path, path: Path, *, force: bool = True) -> None:
    """Remove the worktree at *path*.

    Args:
        repo: The source repository (or any worktree linked to the same repo).
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


def worktree_remove_unlocked(repo: Path, path: Path, *, force: bool = True) -> None:
    """Unlock *path* (if locked), then remove the worktree.

    git refuses to remove a locked worktree even with ``--force``, so any
    removal of a possibly-locked worktree must release the lock first. This is
    the single place encoding that unlock-before-remove ordering; callers that
    remove a worktree which may still hold a lock use this instead of inlining
    ``worktree_unlock`` + ``worktree_remove``.

    The unlock is best-effort: ``git worktree unlock`` fails on an
    already-unlocked worktree, which is expected here and ignored. The removal
    itself is authoritative and raises :class:`GitError` on failure.

    Raises:
        GitError: If ``git worktree remove`` fails.
    """
    try:
        worktree_unlock(repo, path)
    except GitError:
        pass
    worktree_remove(repo, path, force=force)


def worktree_lock(repo: Path, path: Path, *, reason: str | None = None) -> None:
    """Lock the worktree at *path* so git refuses to remove it.

    Args:
        reason: Human-readable lock reason (e.g. the run_id), shown by
            ``git worktree list --porcelain`` and stored in the ``locked`` file.

    Raises:
        GitError: If ``git worktree lock`` fails.
    """
    args = ["worktree", "lock"]
    if reason is not None:
        args.extend(["--reason", reason])
    args.append(str(path))
    proc = _run_git(repo, args, timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git worktree lock {path} failed: {proc.stderr.strip()}")


def worktree_unlock(repo: Path, path: Path) -> None:
    """Unlock the worktree at *path*, releasing git's removal guard.

    Raises:
        GitError: If ``git worktree unlock`` fails.
    """
    proc = _run_git(repo, ["worktree", "unlock", str(path)], timeout=30, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git worktree unlock {path} failed: {proc.stderr.strip()}")


def worktree_lock_mtime(repo: Path, path: Path) -> float | None:
    """Return the lock-armed time of the worktree at *path*, or None if unlocked.

    The lock file lives at ``<git_dir>/worktrees/<path.name>/locked``; its mtime
    is when the lock was armed, and its absence means the worktree is unlocked.
    Returns ``None`` only for a genuinely absent lock file, never on a git
    failure (which propagates as :class:`GitError`).

    Raises:
        GitError: If ``git rev-parse --git-common-dir`` fails.
    """
    proc = _run_git(repo, ["rev-parse", "--git-common-dir"], timeout=5)
    if proc.returncode != 0:
        raise GitError(
            f"git rev-parse --git-common-dir failed in {repo}: {proc.stderr.strip()}"
        )
    git_dir = Path(proc.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    locked = git_dir / "worktrees" / path.name / "locked"
    if not locked.is_file():
        return None
    return locked.stat().st_mtime


def create_branch(repo: Path, name: str) -> None:
    """Create and check out a new branch *name* via ``git checkout -b``.

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


def stage_paths(repo: Path, paths: list[Path]) -> None:
    """Stage exactly *paths* into the index — never ``-A`` / ``--all``.

    Runs ``git add <paths…>`` so only the named files enter the index; any
    other working-tree changes (including pre-existing untracked files) stay
    unstaged.

    Args:
        paths: Repo-relative paths to stage. Must be non-empty.

    Raises:
        GitError: If *paths* is empty, or the ``git add`` call fails.
    """
    if not paths:
        raise GitError("stage_paths requires at least one path")
    normalized = [p.as_posix() for p in paths]
    for path in normalized:
        _require_git_path_confined(repo, path)
    add = _run_git(
        repo,
        ["--literal-pathspecs", "add", "--", *normalized],
        timeout=30,
        retries=0,
    )
    if add.returncode != 0:
        raise GitError(f"git add {paths} failed in {repo}: {add.stderr.strip()}")


def commit_staged(repo: Path, message: str) -> None:
    """Commit the already-validated index without staging again."""
    identity_ok = (
        _run_git(repo, ["config", "user.email"], timeout=5).returncode == 0
        and _run_git(repo, ["config", "user.name"], timeout=5).returncode == 0
    )
    commit_args = ["commit", "-m", message]
    if not identity_ok:
        commit_args = [
            "-c",
            "user.email=daydream@localhost",
            "-c",
            "user.name=daydream",
            *commit_args,
        ]
    commit = _run_git(repo, commit_args, timeout=30, retries=0)
    if commit.returncode != 0:
        raise GitError(f"git commit failed in {repo}: {commit.stderr.strip()}")


def commit_paths(repo: Path, paths: list[Path], message: str) -> None:
    """Stage only *paths* and commit them with *message*.

    Stages exactly the named paths via ``git add <paths…>`` (never ``-A`` /
    ``--all``) so only the intended files are committed, then commits with
    ``git commit -m <message>``.

    Args:
        paths: Repo-relative paths to stage and commit. Must be non-empty.

    Raises:
        GitError: If *paths* is empty, or the ``git add`` / ``git commit`` call
            fails.
    """
    # Empty-path guard lives in stage_paths (identical GitError).
    stage_paths(repo, paths)
    commit_staged(repo, message)


def push_branch(repo: Path, branch: str, *, remote: str = "origin") -> None:
    """Push *branch* to *remote*, setting upstream tracking.

    Runs ``git push -u <remote> <branch>``.

    Raises:
        GitError: If the push fails (propagates stderr).
    """
    proc = _run_git(repo, ["push", "-u", remote, branch], timeout=60, retries=0)
    if proc.returncode != 0:
        raise GitError(f"git push -u {remote} {branch} failed in {repo}: {proc.stderr.strip()}")


# --- gh wrappers -------------------------------------------------------------


GH_PR_VIEW_FIELDS: tuple[str, ...] = (
    "number",
    "title",
    "body",
    "state",
    "headRefName",
    "baseRefName",
    "headRefOid",
    "url",
    "headRepository",
    "headRepositoryOwner",
)
GH_PR_LIST_FIELDS: tuple[str, ...] = (
    "number",
    "headRefOid",
    "baseRefName",
    "url",
    "headRepository",
    "headRepositoryOwner",
)
_GH_DIAGNOSTIC_LIMIT = 2_000
_MISSING_BRANCH_PR_RE = re.compile(r'^no pull requests found for branch "[^"\r\n]+"$')


def _safe_gh_diagnostic(stderr: str) -> str:
    """Return a redacted, bounded one-command diagnostic."""
    redacted = _redact_sensitive_text(stderr.strip())
    if len(redacted) <= _GH_DIAGNOSTIC_LIMIT:
        return redacted
    return redacted[:_GH_DIAGNOSTIC_LIMIT] + "...[truncated]"


def _pr_view_is_absent(stderr: str, pr: int | None) -> bool:
    """Recognize only gh's two established PR-absence diagnostics."""
    diagnostic = stderr.strip()
    if pr is None:
        return _MISSING_BRANCH_PR_RE.fullmatch(diagnostic) is not None
    return diagnostic == (
        f"GraphQL: Could not resolve to a PullRequest with the number of {pr}. "
        "(repository.pullRequest)"
    )


def gh_pr_view(repo: Path, pr: int | None = None) -> dict[str, Any] | None:
    """Return ``gh pr view`` output, or ``None`` only when the PR is absent.

    When *pr* is ``None``, ``gh pr view`` infers the PR from the currently
    checked-out branch. This mirrors the auto-detection flow used by the CLI
    when the user does not pass an explicit PR number.

    Returns:
        Parsed JSON dict, or ``None`` for a recognized missing PR diagnostic.

    Raises:
        GitError: If gh fails for any other reason or returns malformed output.
    """
    args = ["pr", "view"]
    if pr is not None:
        args.append(str(pr))
    args.extend(
        [
            "--json",
            ",".join(GH_PR_VIEW_FIELDS),
        ]
    )
    proc = _run_gh(repo, args, retries=_gh_retries())
    if proc.returncode != 0:
        if _pr_view_is_absent(proc.stderr, pr):
            return None
        diagnostic = _safe_gh_diagnostic(proc.stderr) or "no diagnostic"
        raise _gh_error_for(f"gh pr view failed: {diagnostic}", proc.stderr)
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GitError("gh pr view returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise GitError("gh pr view expected a JSON object")
    return data


def gh_pr_list_for_branch(repo: Path, branch: str) -> list[dict[str, Any]]:
    """List open PRs whose head ref is *branch*.

    Returns:
        List of PR dicts. An empty list means the successful query had no rows.

    Raises:
        GitError: If gh fails or returns malformed output.
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
            ",".join(GH_PR_LIST_FIELDS),
        ],
        retries=_gh_retries(),
    )
    if proc.returncode != 0:
        diagnostic = _safe_gh_diagnostic(proc.stderr) or "no diagnostic"
        raise _gh_error_for(f"gh pr list failed: {diagnostic}", proc.stderr)
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GitError("gh pr list returned invalid JSON") from exc
    if not isinstance(rows, list):
        raise GitError("gh pr list expected a JSON list")
    if any(not isinstance(row, dict) for row in rows):
        raise GitError("gh pr list returned a non-object row")
    return rows


def gh_pr_diff(repo: Path, pr: int) -> str:
    """Return the unified diff for *pr* as text.

    Raises:
        GitError: If ``gh pr diff`` fails.
    """
    proc = _run_gh(repo, ["pr", "diff", str(pr)], retries=_gh_retries())
    if proc.returncode != 0:
        raise GitError(f"gh pr diff {pr} failed: {proc.stderr.strip()}")
    return proc.stdout


def split_owner_repo(slug: str) -> tuple[str, str] | None:
    """Split an ``"owner/repo"`` slug into its components.

    Returns:
        A ``(owner, repo)`` tuple when *slug* contains exactly one ``"/"``
        and both parts are non-empty, or ``None`` otherwise.
    """
    if slug.count("/") != 1:
        return None
    owner, _, repo = slug.partition("/")
    if not owner or not repo or any(char.isspace() for char in owner + repo):
        return None
    return owner, repo


def _credential_helper_args(helper: str | None = None) -> list[str]:
    """Return the ``-c credential.helper=...`` fragment for one git command.

    Git calls the shell ``!``-form helper itself, driving ``get``/``store``/
    ``erase`` with ``protocol``/``host`` on stdin — the credential is scoped to
    this single command, never written to global or local config. ``GH_CREDENTIAL_HELPER``
    is read at call time so a subprocess contract test can monkeypatch it to a
    recording wrapper.
    """
    return ["-c", f"credential.helper={helper or GH_CREDENTIAL_HELPER}"]


def git_ls_remote(repo: Path, url: str) -> str:
    """Authenticated read of *url*'s refs via ``git ls-remote``.

    Pins ``GIT_TERMINAL_PROMPT=0`` (command-scoped, never a global config
    change) so a credential failure surfaces as a git error rather than a
    stuck stdin prompt. Git itself drives the command-scoped credential helper
    (``gh auth git-credential``); the credential never appears in a URL or on
    argv. Raises :class:`GitError` on failure.
    """
    args = [*_credential_helper_args(), "ls-remote", url]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    proc = _run_git(repo, args, env_cmd=env)
    if proc.returncode != 0:
        raise GitError(f"git ls-remote {url} failed: {proc.stderr.strip()}")
    return proc.stdout


def remote_contains_commit(repo: Path, branch: str, sha: str, *, remote: str = "origin") -> bool:
    """Return ``True`` iff ``remote``'s ``refs/heads/<branch>`` reports *sha*.

    Uses the same authenticated ``git ls-remote`` invocation as
    :func:`git_ls_remote` (``GIT_TERMINAL_PROMPT=0`` pinned). A git error in
    the ls-remote itself raises :class:`GitError`; empty ref output is
    ``False``, never ``True``.
    """
    proc = _run_git(
        repo,
        ["ls-remote", remote, f"refs/heads/{branch}"],
        env_cmd={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if proc.returncode != 0:
        raise GitError(
            f"git ls-remote {remote} refs/heads/{branch} failed: {proc.stderr.strip()}"
        )
    return any(line.split()[0] == sha for line in proc.stdout.splitlines() if line.strip())


def gh_repo_view(repo: Path) -> tuple[str, str] | None:
    """Return the ``(owner, name)`` slug for the current repository.

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


def gh_repo_view_required(repo: Path) -> tuple[str, str]:
    """Return the current repository slug, raising on command or shape failure."""
    proc = _run_gh(
        repo,
        ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        retries=_gh_retries(),
    )
    if proc.returncode != 0:
        diagnostic = _safe_gh_diagnostic(proc.stderr) or "no diagnostic"
        raise _gh_error_for(f"gh repo view failed: {diagnostic}", proc.stderr)
    raw_slug = proc.stdout.rstrip("\r\n")
    slug = split_owner_repo(raw_slug)
    if slug is None:
        raise GitError("gh repo view returned an invalid repository slug")
    return slug


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
    redacted_message = _redact_sensitive_text(message)
    is_rate_limit = (
        any(marker in lowered for marker in _RATE_LIMIT_MARKERS)
        or re.search(r"\b429\b", lowered) is not None
        or ("403" in lowered and "rate" in lowered)
    )
    if not is_rate_limit:
        return GitError(redacted_message)
    retry_after: float | None = None
    match = re.search(r"retry[- ]after[:\s]+(\d+)", lowered)
    if match:
        retry_after = float(match.group(1))
    return RateLimitError(redacted_message, retry_after=retry_after)


def _parse_gh_json(stdout: str, jq: str | None, endpoint: str, *, payload_note: str = "") -> Any:
    """Parse ``gh api`` stdout: NDJSON list with *jq*, single JSON value otherwise.

    Raises:
        GitError: If the output is not valid JSON; *payload_note* is appended
            to the message.
    """
    try:
        if jq is not None:
            return [json.loads(line) for line in stdout.splitlines() if line.strip()]
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise GitError(
            _redact_sensitive_text(f"gh api {endpoint} returned invalid JSON: {exc}{payload_note}")
        ) from exc


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
        paginate: When True, pass ``--paginate`` to walk all result pages.
        input_data: Optional JSON-serialisable payload. When provided, it is
            written to a temporary file and passed via ``--input <path>`` and
            the call uses ``--method <method>`` (gh's preferred form). On
            success the tempfile is removed; on failure it is preserved and
            its path is included in the raised :class:`GitError` so callers
            can inspect the exact request body that was sent.
        jq: Optional ``gh --jq`` filter. Each filtered value is JSON-encoded,
            then parsed as NDJSON and returned as a list. With
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
    output_args: list[str] = []
    if paginate:
        output_args.append("--paginate")
    if jq is not None:
        output_args.extend(["--jq", f"({jq}) | @json"])
    retries = _gh_retries() if idempotent else 0

    if input_data is None:
        method_args = ["-X", method.upper()] if method.upper() != "GET" else []
        args = ["api", *header_args, *method_args, *output_args, endpoint]
        proc = _run_gh(repo, args, retries=retries)
        if proc.returncode != 0:
            raise _gh_error_for(f"gh api {endpoint} failed: {proc.stderr.strip()}", proc.stderr)
        return _parse_gh_json(proc.stdout, jq, endpoint)

    # input_data path: serialise to a tempfile and shell out via `--input`.
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - lifecycle managed below
        suffix=".json", mode="w", delete=False, encoding="utf-8"
    )
    tmp_path = Path(tmp.name)
    payload_note = f" (request payload preserved at {tmp_path})"
    succeeded = False
    try:
        try:
            json.dump(input_data, tmp)
        finally:
            tmp.close()
        args = ["api", *header_args, endpoint, "--method", method.upper(), "--input", str(tmp_path), *output_args]
        proc = _run_gh(repo, args, retries=retries)
        if proc.returncode != 0:
            raise _gh_error_for(
                f"gh api {endpoint} failed: {proc.stderr.strip()}{payload_note}",
                proc.stderr,
            )
        result = _parse_gh_json(proc.stdout, jq, endpoint, payload_note=payload_note)
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
    """Set an Actions secret via ``gh secret set <name>`` with the value on stdin.

    The *value* is piped on **stdin** (never the argument vector) so secret
    material such as a PEM private key cannot leak into process listings.

    Args:
        org: Set at the organization scope (``--org``).
        repo_slug: Set at the repository scope (``--repo <owner/repo>``).

    Raises:
        GitError: If neither/both scopes are given, or the ``gh`` call fails.
    """
    args = ["secret", "set", name, *_scope_args(org, repo_slug)]
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
        org: List at the organization scope (``--org``).
        repo_slug: List at the repository scope (``--repo <owner/repo>``).

    Raises:
        GitError: If neither/both scopes are given, or the ``gh`` call fails.
    """
    return _gh_name_list(repo, "variable", org, repo_slug)


def gh_pr_create(
    repo: Path, *, head: str, base: str, title: str, body: str, repo_slug: str | None = None
) -> str:
    """Open a pull request via ``gh pr create`` and return its URL.

    Args:
        repo_slug: Explicit ``owner/repo`` target (``--repo``).  When *None*
            the ambient ``gh`` context (cwd) is used.

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


def gh_issue_create(
    repo: Path,
    *,
    title: str,
    body: str,
    repo_slug: str | None = None,
    labels: list[str] | None = None,
) -> str:
    """Open a GitHub issue via ``gh issue create`` and return its URL.

    Used by the fix loop (issue #336) to route out-of-scope-but-valid findings
    — files outside the reviewed diff or residuals after a fix round — into a
    tracked issue instead of auto-applying them to the PR.

    The body is written to a temp file and passed via ``--body-file`` so it
    never appears on the process argument vector (process-list hygiene; bodies
    can be large). Same pattern as ``gh secret set``'s stdin path.

    Args:
        repo: Worktree the call is rooted in (cwd for ``gh``).
        title: Issue title (inline ``--title``).
        body: Issue body markdown (written to a tempfile, passed as
            ``--body-file``).
        repo_slug: Explicit ``owner/repo`` target (``--repo``). When *None*
            the ambient ``gh`` context (cwd) is used.
        labels: Optional labels applied verbatim as repeated ``--label`` flags.
            *None* (the default) emits no ``--label`` flag.

    Returns:
        The created issue's URL (parsed from ``gh``'s stdout).

    Raises:
        GitError: If the ``gh issue create`` call fails (stderr included).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as bf:
        bf.write(body)
        body_path = bf.name
    args: list[str] = ["issue", "create"]
    if repo_slug is not None:
        args += ["--repo", repo_slug]
    args += ["--title", title, "--body-file", body_path]
    if labels:
        for label in labels:
            args += ["--label", label]
    try:
        proc = _run_gh(repo, args)
    finally:
        try:
            Path(body_path).unlink()
        except OSError:
            pass
    if proc.returncode != 0:
        raise _gh_error_for(f"gh issue create failed: {proc.stderr.strip()}", proc.stderr)
    return proc.stdout.strip()


def gh_issue_list(
    repo: Path,
    *,
    state: str = "open",
    search: str | None = None,
    limit: int = 100,
    repo_slug: str | None = None,
) -> list[dict[str, Any]]:
    """List issues via ``gh issue list``; best-effort (empty list on failure).

    Used by the fix loop (issue #336) for cross-run dedup of out-of-scope
    findings filed as issues: before filing, the caller checks whether an open
    issue already carries the finding's fingerprint marker, so a re-run/resume
    does not re-file the same finding (GitHub is the store — same stateless
    cross-run dedup model as :mod:`daydream.reconcile`). Best-effort by design
    so a failed ``gh issue list`` (no auth, offline, cross-org) degrades to
    filing rather than blocking the scope decision.

    Args:
        repo: Worktree the call is rooted in (cwd for ``gh``).
        state: Issue state filter (``--state``); defaults to ``open``.
        search: Optional ``--search`` qualifier to narrow results.
        limit: Cap on the number of issues returned (``--limit``).
        repo_slug: Explicit ``owner/repo`` target (``--repo``). When *None*
            the ambient ``gh`` context (cwd) is used.

    Returns:
        List of issue dicts (``number``, ``title``, ``body``, ``url``) — empty
        when no issues match or the call fails.
    """
    args: list[str] = [
        "issue",
        "list",
        "--state",
        state,
        "--json",
        "number,title,body,url",
        "--limit",
        str(limit),
    ]
    if search:
        args += ["--search", search]
    if repo_slug is not None:
        args += ["--repo", repo_slug]
    try:
        proc = _run_gh(repo, args, retries=_gh_retries())
    except GitError as exc:
        _logger.warning("gh issue list failed (%s, returning []): %s", type(exc).__name__, exc)
        return []
    if proc.returncode != 0:
        return []
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def gh_issue_list_strict(
    repo: Path,
    *,
    state: str = "all",
    repo_slug: str,
) -> list[dict[str, Any]]:
    """List every issue through the paginated REST API, failing closed.

    Unlike :func:`gh_issue_list`, this helper is for workflows where an empty
    result after a failed lookup could cause a duplicate write. It therefore
    raises on transport, API, and response-shape failures. GitHub's REST issue
    endpoint also returns pull requests; those rows are deliberately excluded.

    Args:
        repo: Worktree the call is rooted in (cwd for ``gh``).
        state: One of ``"open"``, ``"closed"``, or ``"all"``.
        repo_slug: Explicit GitHub ``owner/repo`` target.

    Returns:
        Normalized issue dictionaries containing ``number``, ``title``,
        ``body``, ``url``, and ``state``. Pagination is unbounded rather than
        stopping at GitHub's 100-item page size.

    Raises:
        GitError: If the arguments are invalid, lookup fails, or GitHub returns
            an unexpected response shape.
    """
    if state not in {"open", "closed", "all"}:
        raise GitError(f"invalid issue state {state!r}")
    parsed = split_owner_repo(repo_slug)
    if parsed is None or "/" in parsed[1]:
        raise GitError(f"invalid GitHub repository slug {repo_slug!r}")
    owner, name = parsed
    endpoint = f"repos/{owner}/{name}/issues?state={state}&per_page=100"
    rows = gh_api(
        repo,
        endpoint,
        paginate=True,
        jq=".[]",
        idempotent=True,
    )
    if not isinstance(rows, list):
        raise GitError("GitHub issue lookup returned a non-list response")

    issues: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise GitError("GitHub issue lookup returned a non-object row")
        if row.get("pull_request") is not None:
            continue
        number = row.get("number")
        title = row.get("title")
        body = row.get("body")
        url = row.get("html_url", row.get("url"))
        row_state = row.get("state")
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not isinstance(title, str)
            or body is not None
            and not isinstance(body, str)
            or not isinstance(url, str)
            or not url
            or not isinstance(row_state, str)
        ):
            raise GitError("GitHub issue lookup returned a malformed issue row")
        issues.append(
            {
                "number": number,
                "title": title,
                "body": body or "",
                "url": url,
                "state": row_state,
            }
        )
    return issues
