"""Shared tree-sitter version safety guard (#1087).

Tree-sitter 0.26.0 ships a native Point-getter regression (py-tree-sitter#472)
that can SIGSEGV the whole process during coordinate access.  Every native
analysis entry point consults this module before constructing a parser: a
known-bad installed version raises :class:`TreeSitterBadVersionError`, which
callers' existing fail-open wrappers convert into an explicit, auditable
"gate unavailable" outcome instead of a process crash.
"""

from __future__ import annotations

import importlib.metadata

#: The single shared list of known-bad tree-sitter versions (S1).  Adding a
#: future bad version (or blessing a fixed 0.26.x) is a one-line change here.
KNOWN_BAD_TREE_SITTER_VERSIONS: frozenset[str] = frozenset({"0.26.0"})

_UPSTREAM_ISSUE = "py-tree-sitter#472"


class TreeSitterBadVersionError(RuntimeError):
    """Raised when the installed tree-sitter version is known to crash natively."""


def installed_tree_sitter_version() -> str | None:
    """Return the installed tree-sitter version, or ``None`` if not installed.

    Uses ``importlib.metadata`` (the Task 0 spike confirmed it reports
    correctly on both 0.25.2 and 0.26.0).  ``tree_sitter.__version__`` is
    deliberately not read: it is absent on 0.25.2, so the package attribute
    is not a valid detector.  Only ``PackageNotFoundError`` is swallowed.
    """
    try:
        return importlib.metadata.version("tree-sitter")
    except importlib.metadata.PackageNotFoundError:
        return None


def tree_sitter_unavailable_reason() -> str | None:
    """Return a human-readable reason when the installed version is bad.

    Reports ``None`` when the guard would pass -- a missing or unknown-good
    install is the existing degrade path, not a known-bad one.
    """
    try:
        version = installed_tree_sitter_version()
    except Exception:
        # Fail open, never crash the process the guard exists to protect:
        # any inability to resolve the installed version is the *unknown*
        # path, which the spec treats as not-bad (no false positives on
        # exotic installs).
        return None
    if version in KNOWN_BAD_TREE_SITTER_VERSIONS:
        return (
            f"tree-sitter {version} is known to SIGSEGV the process "
            f"(py-tree-sitter issue {_UPSTREAM_ISSUE}); install a fixed "
            "version such as 0.25.2 to enable quality analysis."
        )
    return None


def assert_tree_sitter_safe() -> None:
    """Raise :class:`TreeSitterBadVersionError` on a known-bad install.

    Returns normally when the installed version is not in the known-bad set
    (including an unresolvable/not-installed version) -- the guard never
    treats an unknown version as bad.
    """
    reason = tree_sitter_unavailable_reason()
    if reason is not None:
        version = installed_tree_sitter_version()
        raise TreeSitterBadVersionError(
            f"tree-sitter {version} is in the known-bad set ({sorted(KNOWN_BAD_TREE_SITTER_VERSIONS)}): {reason}"
        )
