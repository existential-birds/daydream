"""Guard parser construction against tree-sitter 0.26.0's native Point-getter crash.

Every native entry checks before constructing a parser. Existing fail-open
wrappers turn TreeSitterBadVersionError into an auditable unavailable gate.
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
    """Read installed metadata; return None only for PackageNotFoundError.

    tree_sitter.__version__ is absent on 0.25.2, so it cannot identify safe installs.
    """
    try:
        return importlib.metadata.version("tree-sitter")
    except importlib.metadata.PackageNotFoundError:
        return None


def tree_sitter_unavailable_reason() -> str | None:
    """Describe a known-bad install; missing or unknown versions return None."""
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
    """Raise TreeSitterBadVersionError only for an explicitly known-bad version."""
    reason = tree_sitter_unavailable_reason()
    if reason is not None:
        # The reason already names the offending version, so there is no need
        # to re-read the installed version here (and no way for that unguarded
        # second read to escape this guard).
        raise TreeSitterBadVersionError(
            f"tree-sitter is in the known-bad set ({sorted(KNOWN_BAD_TREE_SITTER_VERSIONS)}): {reason}"
        )
