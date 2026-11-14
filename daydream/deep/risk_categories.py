"""Single source of truth for the mandatory risk-category vocabulary.

Leaf module: it declares the five categories that force conservative
verification and the trigger words that identify each from changed text. The
per-item verifier selection reads this one declaration.
"""

from __future__ import annotations

MANDATORY_RISK_CATEGORIES: tuple[str, ...] = (
    "security",
    "concurrency",
    "persistence",
    "public-interface",
    "migration",
)
"""The issue's five mandatory risk categories, in fixed declaration order."""

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



def category_matches(category: str, text: str) -> bool:
    """Return whether ``text`` contains any trigger for ``category`` (case-insensitive)."""
    lowered = text.lower()
    return any(trigger in lowered for trigger in CATEGORY_TRIGGERS[category])


def categories_in(text: str) -> tuple[str, ...]:
    """Return every declared category whose triggers occur in ``text``.

    Deterministic: categories are returned in declaration order.
    """
    return tuple(category for category in MANDATORY_RISK_CATEGORIES if category_matches(category, text))
