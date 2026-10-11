"""C5 benchmark exclusions from the packaged schema list.

Resolve paths relative to this module for installed use. Re-read on every
call to avoid stale state; do not add a process cache.
"""

from __future__ import annotations

from pathlib import Path

EXCLUSION_PATH = Path(__file__).parent / "schema" / "exclusion.txt"


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
