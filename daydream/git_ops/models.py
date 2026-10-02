"""Git and GitHub errors, request budgets, and repository state values."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# --- Errors ------------------------------------------------------------------


class GitError(Exception):
    """Base class for all git/gh failures raised by :mod:`daydream.git_ops`.

    ``preserved_payload_path`` is set by ``gh_api`` when its failed request
    retains an input file. Callers can report that path without parsing or
    exposing the exception's diagnostic text.
    """

    preserved_payload_path: Path | None = None


class GitTimeoutError(GitError):
    """A subprocess timeout after its bounded retry budget, distinct from ordinary Git
    failures.
    """


class PathAbsentError(GitError):
    """A path proven absent at a ref by show or gh_file_at_ref. Transport, authentication,
    missing tools, corrupt storage, and malformed responses must remain read errors,
    never absence.
    """


class DeadlineExpired(GitError):
    """Raised before a GitHub request when its shared deadline has expired."""


@dataclass(frozen=True)
class GitHubRequestBudget:
    """Shared absolute deadline and per-request cap for GitHub reads."""

    deadline: float
    per_request_seconds: float
    monotonic: Callable[[], float]

    def next_timeout(self) -> float:
        """Return the current positive min(cap, remaining), else raise."""
        remaining = self.deadline - self.monotonic()
        timeout = min(self.per_request_seconds, remaining)
        if not math.isfinite(timeout) or timeout <= 0:
            raise DeadlineExpired("GitHub request deadline expired")
        return timeout


@dataclass(frozen=True)
class GitHubPageLimits:
    """Manual pagination limits for bounded GitHub collection reads."""

    per_page: int = 100
    max_pages: int = 10


class RateLimitError(GitError):
    """GitHub API rate limit with optional ``retry_after`` seconds parsed from gh stderr."""

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


class SnapshotPreparationError(GitError):
    """A standalone repository snapshot could not be established safely."""


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
class RawIndexSnapshot:
    """Exact owner index bytes, including intent-to-add and entry flags."""

    content: bytes | None
    mode: int | None


@dataclass(frozen=True)
class WorktreeRollbackSnapshot:
    """One round's tracked, untracked, and index rollback point."""

    ref: str
    index: IndexSnapshot
    path_states: tuple[GitPathState, ...]
    untracked: dict[str, GitPathState]


@dataclass(frozen=True)
class IndependentSnapshot:
    """Standalone repository and lexical locations of its outward symlinks."""

    repo: Path
    outward_symlinks: frozenset[Path]
