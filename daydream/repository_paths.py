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


def path_is_confined(
    repo: Path,
    value: str,
    *,
    directory_scope: bool = False,
) -> bool:
    """Return whether a lexical repository path crosses no symlink/root edge."""
    validator = (
        valid_directory_scope_lexical
        if directory_scope
        else valid_repository_file_path
    )
    if value != "." and not validator(value):
        return False
    root = repo.resolve()
    candidate = repo
    if value != ".":
        for part in PurePosixPath(value.rstrip("/")).parts:
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
