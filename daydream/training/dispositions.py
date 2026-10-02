"""Shared decisive/nondecisive classification for queues, materialization, and projection tiers. Keep
this at training package level so projection avoids adjudication's eager import chain.
"""

from typing import Final

__all__ = ["DECISIVE_DISPOSITIONS", "NON_DECISIVE_DISPOSITIONS", "is_decisive"]

NON_DECISIVE_DISPOSITIONS: Final[frozenset[str]] = frozenset({"ambiguous", "unanswered", "missing"})
DECISIVE_DISPOSITIONS: Final[frozenset[str]] = frozenset({"accepted", "rejected"})


def is_decisive(disposition: str) -> bool:
    """Return True when ``disposition`` is a decisive outcome (accepted/rejected)."""
    return disposition in DECISIVE_DISPOSITIONS
