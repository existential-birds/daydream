"""Canonical repository-path grammar and confinement validation (core).

Model-produced ``file``/``path``/``working_directory`` fields are untrusted.
This package-root module owns the lexical grammar (relative POSIX segments,
no parent traversal, no absolute paths) and the filesystem confinement check
(symlink-at-any-prefix walk plus resolved-root containment) that every
consumer — core phases and the improve command contract — relies on. Living
at package root keeps flow subpackages dependent on core, never the reverse.

It also owns ``is_test_path``, the deterministic test-vs-production path
classifier, for the same reason: two flows (improve plan gates and
grounded-diagram eligibility) must answer that question identically.
"""

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


def valid_repository_file_path(value: str) -> bool:
    """Return whether ``value`` has the safe repository-file grammar."""
    return len(value.encode("utf-8")) <= REPOSITORY_FILE_PATH_MAX_LENGTH and bool(
        _REPOSITORY_FILE_PATH.fullmatch(value)
    )


def valid_directory_scope_lexical(value: str) -> bool:
    """Return whether ``value`` is a safe file-or-directory scope."""
    return len(value.encode("utf-8")) <= REPOSITORY_FILE_PATH_MAX_LENGTH and bool(
        _DIRECTORY_SCOPE.fullmatch(value)
    )


def _strip_prefix(
    parts: tuple[str, ...], base: tuple[str, ...]
) -> tuple[str, ...] | None:
    """Return ``parts`` with a leading ``base`` removed, else None."""
    if parts[: len(base)] == base:
        return parts[len(base) :]
    return None


def _repo_relative_parts(
    parts: tuple[str, ...], repo: Path, root: Path
) -> tuple[str, ...] | None:
    """Return ``parts`` minus the repo prefix, trying unresolved then resolved.

    ``root`` is ``repo.resolve()``; the second base strips absolute spellings
    of a symlinked repo against its resolved spelling. None when neither base
    prefixes ``parts``.
    """
    for base in (PurePosixPath(repo).parts, PurePosixPath(str(root)).parts):
        remainder = _strip_prefix(parts, base)
        if remainder is not None:
            return remainder
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
    stripped = value.rstrip("/")
    if stripped.startswith("./"):
        stripped = stripped[2:]
    return stripped


def canonicalize_working_directory(repo: Path, value: str) -> str:
    """Return ``value`` as the repo-relative posix form (``"."`` for the root).

    Absolute, ``./``-prefixed, and plain relative spellings of the same
    directory collapse to one repo-relative key. Accepted values are already
    confinement-checked (``path_is_confined``), so the transform is lossless.
    """
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
    """Return whether ``path`` names a test file by repository convention.

    The single deterministic test-vs-production path classifier, shared by the
    improve plan gates (which must not accept a production deletion as
    self-justifying) and grounded-diagram eligibility (which counts changed
    *production* code files, issue #1113). Deliberately deterministic and
    lexical: the exploration pass's ``role == "test"`` is model output, not a
    fact, so nothing that gates behavior may depend on it.

    A path is a test path when any parent directory is a conventional test
    directory (``__tests__``, ``spec``, ``specs``, ``test``, ``tests``, matched
    case-insensitively), or when the file's own name follows a test-naming
    convention: stem ``test``/``tests``, a ``test_`` prefix or ``_test`` suffix
    (Python, Go, Rust), a ``Test``/``Tests`` camel-case suffix (JVM/Swift), or a
    ``.test.``/``.spec.`` infix (JS/TS). Case-folded throughout except the
    camel-case suffix, which is meaningful only in its original casing.
    """
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
    "canonicalize_directory_scope",
    "canonicalize_working_directory",
    "is_test_path",
    "path_is_confined",
    "valid_directory_scope_lexical",
    "valid_repository_file_path",
]
