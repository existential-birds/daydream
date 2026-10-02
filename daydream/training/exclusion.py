"""C5 exclusions and C8 copyleft opt-ins from packaged schema/ lists.

Resolve paths relative to this module for installed use. Re-read on every
call to avoid stale state; do not add a process cache.
"""

from __future__ import annotations

from pathlib import Path

EXCLUSION_PATH = Path(__file__).parent / "schema" / "exclusion.txt"
COPYLEFT_PATH = Path(__file__).parent / "schema" / "copyleft.txt"


def _load_repo_slugs(path: Path) -> frozenset[str]:
    """Read a one-slug-per-line file into a frozenset.

    Blank lines and lines whose stripped form starts with ``#`` are skipped.
    """
    with path.open("r", encoding="utf-8") as fh:
        return frozenset(
            line for raw in fh if (line := raw.strip()) and not line.startswith("#")
        )


def load_exclusion_list() -> frozenset[str]:
    """Load the always-enforced exclusion list (C5).

    Returns:
        Frozen set of ``owner/repo`` slugs that must never appear in the
        exported training corpus, regardless of CLI flags.
    """
    return _load_repo_slugs(EXCLUSION_PATH)


def load_copyleft_list() -> frozenset[str]:
    """Load the known GPL/AGPL repo list (C8).

    Returns:
        Frozen set of ``owner/repo`` slugs that must be skipped unless the
        caller explicitly opts in via ``--allow-copyleft``.
    """
    return _load_repo_slugs(COPYLEFT_PATH)


def is_copyleft(
    repo_slug: str | None,
    allow_list: frozenset[str],
    *,
    copyleft_list: frozenset[str] | None = None,
) -> bool:
    """Return whether a non-None slug requires a copyleft opt-in it does not have.

    Pass a preloaded copyleft_list for loops; otherwise read it on every call.
    """
    if repo_slug is None:
        return False
    if repo_slug in allow_list:
        return False
    known = copyleft_list if copyleft_list is not None else load_copyleft_list()
    return repo_slug in known
