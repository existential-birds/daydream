"""Canonical repository-path grammar and confinement validation (core).

Model-produced ``file``/``path``/``working_directory`` fields are untrusted.
This package-root module owns the lexical grammar (relative POSIX segments,
no parent traversal, no absolute paths) and the filesystem confinement check
(symlink-at-any-prefix walk plus resolved-root containment) that every
consumer — core phases and the improve command contract — relies on. Living
at package root keeps flow subpackages dependent on core, never the reverse.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

_PATH_SEGMENT = (
    r"(?:[A-Za-z0-9_+@$-][A-Za-z0-9._+@$-]*|"
    r"\.[A-Za-z0-9_+@$-][A-Za-z0-9._+@$-]*|"
    r"\.\.[A-Za-z0-9._+@$-]+)"
)
REPOSITORY_FILE_PATH_PATTERN = rf"^{_PATH_SEGMENT}(?:/{_PATH_SEGMENT})*$"
DIRECTORY_SCOPE_PATTERN = rf"^{_PATH_SEGMENT}(?:/{_PATH_SEGMENT})*/?$"

REPOSITORY_FILE_PATH_SCHEMA: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "maxLength": 512,
    "pattern": REPOSITORY_FILE_PATH_PATTERN,
}
DIRECTORY_SCOPE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "maxLength": 512,
    "pattern": DIRECTORY_SCOPE_PATTERN,
}

_REPOSITORY_FILE_PATH = re.compile(REPOSITORY_FILE_PATH_PATTERN)
_DIRECTORY_SCOPE = re.compile(DIRECTORY_SCOPE_PATTERN)


def valid_repository_file_path(value: str) -> bool:
    """Return whether ``value`` has the safe repository-file grammar."""
    return bool(_REPOSITORY_FILE_PATH.fullmatch(value))


def valid_directory_scope_lexical(value: str) -> bool:
    """Return whether ``value`` is a safe file-or-directory scope."""
    return bool(_DIRECTORY_SCOPE.fullmatch(value))


def _strip_prefix(
    parts: tuple[str, ...], base: tuple[str, ...]
) -> tuple[str, ...] | None:
    """Return ``parts`` with a leading ``base`` removed, else None."""
    if parts[: len(base)] == base:
        return parts[len(base) :]
    return None


def _walk_components(candidate: Path, parts: Sequence[str]) -> Path | None:
    """Return ``candidate`` advanced through ``parts``, or None on a crossing.

    Stops at the first component that does not exist (the containment check
    below settles it). None when a component is a symlink or cannot be
    inspected — both are treated as an escape.
    """
    for part in parts:
        candidate /= part
        try:
            if candidate.is_symlink():
                return None
            if not candidate.exists():
                break
        except OSError:
            return None
    return candidate


def path_is_confined(
    repo: Path,
    value: str,
    *,
    directory_scope: bool = False,
    allow_absolute: bool = False,
) -> bool:
    """Return whether a repository path crosses no symlink/root edge.

    When ``allow_absolute`` is set and ``value`` starts with ``"/"``, the
    relative-lexical gate is skipped and the confinement loop alone decides:
    it walks the components at or below the repo root and settles containment
    with the final ``is_relative_to`` check against the resolved repo root.
    Components above the repo root are deliberately not symlink-tested — a
    symlinked ancestor of the repo (e.g. ``/tmp`` on macOS) would otherwise
    reject the absolute spelling of a path the identical relative spelling
    accepts, and skipping that test cannot escape containment. Relative paths
    and the default ``allow_absolute=False`` keep today's exact behavior.
    """
    validator = (
        valid_directory_scope_lexical
        if directory_scope
        else valid_repository_file_path
    )
    if (
        value != "."
        and not (allow_absolute and value.startswith("/"))
        and not validator(value)
    ):
        return False
    root = repo.resolve()
    candidate = repo
    if value != ".":
        parts = PurePosixPath(value.rstrip("/")).parts
        if allow_absolute and value.startswith("/"):
            # PurePosixPath.parts begins with "/", so a bare component-wise
            # division would reset candidate to the filesystem root and
            # re-descend the repo's own ancestors — symlink-testing each one
            # and rejecting the absolute spelling of an in-repo path whenever
            # an ancestor is a symlink (e.g. /tmp -> /private/tmp). Ancestors
            # are adjudicated by the containment check below; walk only the
            # components at or below the repo root, exactly like the relative
            # form does.
            stripped = _strip_prefix(parts, PurePosixPath(repo).parts)
            if stripped is None:
                stripped = _strip_prefix(
                    parts, PurePosixPath(str(root)).parts
                )
            if stripped is not None:
                parts = stripped
        walked = _walk_components(candidate, parts)
        if walked is None:
            return False
        candidate = walked
    try:
        return candidate.resolve(strict=False).is_relative_to(root)
    except OSError:
        return False


def canonicalize_directory_scope(value: str) -> str:
    """Canonicalize the sole lossless scope spelling difference."""
    return value.rstrip("/")


__all__ = [
    "DIRECTORY_SCOPE_PATTERN",
    "DIRECTORY_SCOPE_SCHEMA",
    "REPOSITORY_FILE_PATH_PATTERN",
    "REPOSITORY_FILE_PATH_SCHEMA",
    "canonicalize_directory_scope",
    "path_is_confined",
    "valid_directory_scope_lexical",
    "valid_repository_file_path",
]
