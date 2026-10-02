"""Shared path grammar, filesystem confinement, and lexical test-path detection.

Model-produced paths are untrusted. Core review and improve share these rules;
Git-observed paths have a separate grammar boundary."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

_SEG_NONDOT = r"[\w #%~!&()+\-@]"               # non-dot segment char, no $/backtick
_SEG_CHAR   = r"[\w .#%~!&()+\-@]"              # segment char incl. dot
_PATH_SEGMENT = (
    rf"(?:{_SEG_NONDOT}{_SEG_CHAR}*"
    rf"|\.{_SEG_NONDOT}{_SEG_CHAR}*"            # single leading dot + non-dot body (.foo)
    rf"|\.\.{_SEG_CHAR}+)")            # two leading dots + >=1 char (..cache, ...)
# Anchor-free grammar core shared by every schema: one or more segments. No
# ^/$ anchors and no leading "./" prefix — consumers layer those per
# alternative, so a prefix meant for relative spellings cannot leak into an
# absolute alternative (e.g. WORKING_DIRECTORY_SCHEMA's "/…" form).
REPOSITORY_FILE_PATH_SEGMENTS = rf"{_PATH_SEGMENT}(?:/{_PATH_SEGMENT})*"
REPOSITORY_FILE_PATH_PATTERN = rf"^(?:\./)?{REPOSITORY_FILE_PATH_SEGMENTS}$"
DIRECTORY_SCOPE_PATTERN      = rf"^(?:\./)?{REPOSITORY_FILE_PATH_SEGMENTS}/?$"

# POSIX PATH_MAX (4096) is a byte budget; the lexical gates measure UTF-8 bytes.
REPOSITORY_FILE_PATH_MAX_LENGTH = 4096

# Conventional test-directory names, matched case-insensitively against every
# parent segment of a path by ``is_test_path``.
_TEST_DIRECTORY_NAMES = {"__tests__", "spec", "specs", "test", "tests"}

# \A/\Z are not ECMA-262-valid (Codex/OpenAI strict mode rejects them), so the
# patterns anchor with ^/$; Python re.search lets $ match before a trailing
# newline, so the schema alone cannot reject trailing-newline spellings — the
# fullmatch lexical gates (valid_repository_file_path et al.) are the
# enforcement point.
REPOSITORY_FILE_PATH_SCHEMA: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "maxLength": REPOSITORY_FILE_PATH_MAX_LENGTH,
    "pattern": REPOSITORY_FILE_PATH_PATTERN,
}
DIRECTORY_SCOPE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "maxLength": REPOSITORY_FILE_PATH_MAX_LENGTH,
    "pattern": DIRECTORY_SCOPE_PATTERN,
}

_REPOSITORY_FILE_PATH = re.compile(REPOSITORY_FILE_PATH_PATTERN)
_DIRECTORY_SCOPE = re.compile(DIRECTORY_SCOPE_PATTERN)


class InvalidRepositoryFilePath(ValueError):
    """Unsafe model-authored path; errors never reflect untrusted control characters."""


def _valid_lexical(value: str, pattern: re.Pattern[str]) -> bool:
    """Return whether ``value`` fits the byte budget and *pattern* grammar."""
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return len(encoded) <= REPOSITORY_FILE_PATH_MAX_LENGTH and bool(pattern.fullmatch(value))


def valid_repository_file_path(value: str) -> bool:
    """Return whether ``value`` has the safe repository-file grammar."""
    return _valid_lexical(value, _REPOSITORY_FILE_PATH)


def strip_dot_slash(value: str) -> str:
    """Normalize the one allowed ./ prefix to match reviewed paths and receipts."""
    return value[2:] if value.startswith("./") else value


def canonicalize_repository_file_path(repo: Path, value: object) -> str:
    """Validate a model path and strip only its optional ./ prefix.

    Reject absolute paths, traversal, malformed values, and current symlink crossings
    with a non-reflective error."""
    if not isinstance(value, str) or not valid_repository_file_path(value):
        raise InvalidRepositoryFilePath("invalid repository file path")
    canonical = strip_dot_slash(value)
    if not canonical or not path_is_confined(repo, canonical):
        raise InvalidRepositoryFilePath("invalid repository file path")
    return canonical


def git_observed_path_is_confined(
    repo: Path, value: str, *, allow_leaf_symlink: bool = False
) -> bool:
    """Confine a Git-observed path without applying the narrower model-output grammar.

    Reject absolute/NUL/empty/dot/parent components and current symlink crossings.
    allow_leaf_symlink checks only parents for callers that inspect/replace the leaf."""
    if not isinstance(value, str) or not value or "\0" in value or value.startswith("/"):
        return False
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    root = repo.resolve()
    walked = _walk_components(repo, parts[:-1] if allow_leaf_symlink else parts)
    if walked is None:
        return False
    try:
        return walked.resolve(strict=False).is_relative_to(root)
    except OSError:
        return False


def valid_directory_scope_lexical(value: str) -> bool:
    """Return whether ``value`` is a safe file-or-directory scope."""
    return _valid_lexical(value, _DIRECTORY_SCOPE)


def _repo_relative_parts(
    parts: tuple[str, ...], repo: Path, root: Path
) -> tuple[str, ...] | None:
    """Strip the unresolved or resolved repo prefix; return None if neither matches."""
    for base in (PurePosixPath(repo).parts, PurePosixPath(str(root)).parts):
        if parts[: len(base)] == base:
            return parts[len(base) :]
    return None


def _walk_components(candidate: Path, parts: Sequence[str]) -> Path | None:
    """Walk until a nonexistent component; reject symlinks and inspection failures.

    The caller’s final resolved-containment check handles the nonexistent suffix."""
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
    """Check path components for symlinks and resolve final root containment.

    Allowed absolute paths skip the relative grammar and inspect only components
    at/below the repo root. Symlinked ancestors such as macOS /tmp remain valid;
    final containment still prevents escape."""
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
            stripped = _repo_relative_parts(parts, repo, root)
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
    """Canonicalize the lossless scope spelling differences."""
    return strip_dot_slash(value.rstrip("/"))


def canonicalize_working_directory(repo: Path, value: str) -> str:
    """Normalize an already-confined absolute or relative directory to repo-relative POSIX."""
    if value.startswith("/"):
        parts = PurePosixPath(value.rstrip("/")).parts
        remainder = _repo_relative_parts(parts, repo, repo.resolve())
        if remainder is not None:
            # Parent-traversal segments name an in-repo directory for
            # confinement-checked values (the walk resolves each ``..``
            # against a real in-repo dir), so collapse them lexically to keep
            # "sub/../sub" and "sub" on one dedup key.
            return posixpath.normpath("/".join(remainder)) or "."
        # Not-under-repo absolute spellings are unreachable for
        # confinement-checked callers; return unchanged as a defensive identity.
        return value
    return canonicalize_directory_scope(value)


def is_test_path(path: str) -> bool:
    """Classify tests lexically; never trust an exploration model’s role annotation.

    Match conventional test directories, test/tests stems, test_/_test names,
    Test/Tests camel-case suffixes, and .test./.spec. infixes. Only camel-case suffixes
    retain case sensitivity. Review and improve gates share this policy."""
    candidate = Path(path)
    parts = {part.casefold() for part in candidate.parts[:-1]}
    name = candidate.name.casefold()
    stem = candidate.stem.casefold()
    return bool(parts & _TEST_DIRECTORY_NAMES) or (
        stem in {"test", "tests"}
        or stem.startswith("test_")
        or stem.endswith("_test")
        or candidate.stem.endswith(("Test", "Tests"))
        or ".test." in name
        or ".spec." in name
    )


__all__ = [
    "DIRECTORY_SCOPE_PATTERN",
    "DIRECTORY_SCOPE_SCHEMA",
    "REPOSITORY_FILE_PATH_PATTERN",
    "REPOSITORY_FILE_PATH_SCHEMA",
    "REPOSITORY_FILE_PATH_SEGMENTS",
    "InvalidRepositoryFilePath",
    "canonicalize_repository_file_path",
    "canonicalize_directory_scope",
    "canonicalize_working_directory",
    "git_observed_path_is_confined",
    "is_test_path",
    "path_is_confined",
    "valid_directory_scope_lexical",
    "valid_repository_file_path",
]
