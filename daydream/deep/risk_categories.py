"""Single source of truth for the mandatory risk-category vocabulary.

Leaf module: it declares the five categories that force conservative
verification and the trigger words that identify each from changed text. The
diff-surface classifier (:mod:`daydream.deep.latency`) and the per-item
selection classifier both read this one declaration, so a trigger can never
diverge between them.
"""

from __future__ import annotations

from collections.abc import Iterable

MANDATORY_RISK_CATEGORIES: tuple[str, ...] = (
    "security",
    "concurrency",
    "persistence",
    "public-interface",
    "migration",
)
"""The issue's five mandatory risk categories, in fixed declaration order.

These can never be removed by configuration: a run that names extra categories
widens the set, never narrows it.
"""

RISK_CATEGORIES: tuple[str, ...] = MANDATORY_RISK_CATEGORIES
"""The full declared category vocabulary; any configured name is drawn from here."""

# Calibration surfaces: match only changed content and post-state paths, never context.
CATEGORY_TRIGGERS: dict[str, tuple[str, ...]] = {
    "security": (
        "authenticate", "authorization", "authz", "password", "secret", "credential", "token", "permission",
    ),
    "concurrency": (
        "threading.lock", "asyncio.lock", "mutex", "semaphore", "deadlock", "atomic", "race condition",
    ),
    "persistence": (
        "alter table", "create table", "drop table", "begin;", "commit;", "rollback;", "transaction",
    ),
    "public-interface": ("@app.route", "@router.", "openapi", "proto3", "endpoint", "api/v", "grpc"),
    "migration": ("migrations/", "migration", "alembic", "schema_version"),
}
"""Category id -> trigger words, matched as case-insensitive substrings."""

SURFACE_CATEGORY: dict[str, str] = {
    "security_surface": "security",
    "concurrency_surface": "concurrency",
    "persistence_surface": "persistence",
    "interface_surface": "public-interface",
    "migration_surface": "migration",
}
"""The diff-surface signal names from :class:`daydream.deep.latency.DiffSignals`
to the category ids they represent, so floors and per-item classification read
one vocabulary.
"""


class UnknownRiskCategoryError(ValueError):
    """Raised when a configured risk category is not in the declared vocabulary."""


def category_matches(category: str, text: str) -> bool:
    """Return whether ``text`` contains any trigger for ``category`` (case-insensitive)."""
    lowered = text.lower()
    return any(trigger in lowered for trigger in CATEGORY_TRIGGERS[category])


def categories_in(text: str) -> tuple[str, ...]:
    """Return every declared category whose triggers occur in ``text``.

    Deterministic: categories are returned in declaration order.
    """
    return tuple(category for category in RISK_CATEGORIES if category_matches(category, text))


def resolve_mandatory_categories(extra: Iterable[str]) -> tuple[str, ...]:
    """Return the built-in categories followed by first-seen, deduplicated extras.

    Never drops a built-in and never returns an empty set. An unknown name raises
    :class:`UnknownRiskCategoryError` naming the offending value; it is never
    coerced to a default or silently narrowed.
    """
    resolved: list[str] = list(MANDATORY_RISK_CATEGORIES)
    for category in extra:
        if category not in RISK_CATEGORIES:
            declared = ", ".join(RISK_CATEGORIES)
            raise UnknownRiskCategoryError(
                f"Unknown risk category: {category!r}; declared vocabulary: {declared}"
            )
        if category not in resolved:
            resolved.append(category)
    return tuple(resolved)
