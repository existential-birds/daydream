"""Exclusion-list and copyleft license helpers for the training exporter.

C5 (always-enforced exclusion) and C8 (GPL/AGPL opt-in) from the SPEC are
implemented here. The two backing files live alongside this module under
`schema/` so the package works correctly when installed (paths resolve
relative to `__file__`, not the current working directory).

The loaders deliberately re-read the on-disk files on every call. The files
are tiny and re-reading avoids stale-state surprises in tests; do not add
`functools.lru_cache` here.
"""

from __future__ import annotations

from pathlib import Path

EXCLUSION_PATH = Path(__file__).parent / "schema" / "exclusion.txt"
COPYLEFT_PATH = Path(__file__).parent / "schema" / "copyleft.txt"


def _load_repo_slugs(path: Path) -> frozenset[str]:
    """Read a one-slug-per-line file into a frozenset.

    Blank lines and lines whose stripped form starts with ``#`` are skipped.
    """
    slugs: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            slugs.add(stripped)
    return frozenset(slugs)


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


def is_excluded(repo_slug: str | None) -> bool:
    """Check whether a repo slug is on the C5 exclusion list.

    Returns:
        ``True`` if the slug is on the always-enforced exclusion list;
        ``False`` when the slug is ``None`` or absent from the list.
    """
    if repo_slug is None:
        return False
    return repo_slug in load_exclusion_list()


def is_copyleft(
    repo_slug: str | None,
    allow_list: frozenset[str],
    *,
    copyleft_list: frozenset[str] | None = None,
) -> bool:
    """Check whether a repo slug should be skipped under C8.

    Args:
        repo_slug: ``owner/repo`` to check, or ``None``.
        allow_list: Frozen set of slugs the caller has explicitly opted in
            via ``--allow-copyleft``.
        copyleft_list: Pre-loaded copyleft set.  When ``None`` (the default)
            the list is loaded from disk on every call (same as before).
            Pass the result of ``load_copyleft_list()`` when checking many
            rows in a loop to avoid O(N) redundant file opens.

    Returns:
        ``True`` only when ``repo_slug`` is on the copyleft list and is not
        in ``allow_list``. ``None`` short-circuits to ``False``.
    """
    if repo_slug is None:
        return False
    if repo_slug in allow_list:
        return False
    known = copyleft_list if copyleft_list is not None else load_copyleft_list()
    return repo_slug in known
