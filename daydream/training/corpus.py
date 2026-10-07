"""Shared identity and gold-admission helpers for training projections.

Posterior leakage is excluded by valid_at as well as the snapshot's observed_at
pin; intrinsic capture-time rewards survive. Callers normalize as_of once so
Snapshot selection and chronological leakage checks share the same boundary.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any


def _is_posterior_leak(annotation: dict[str, Any] | None, as_of: str | None) -> bool:
    """Exclude posterior evidence whose valid_at is later than the as_of pin.

    The snapshot already filters observed_at. Compare parsed datetimes so offsets,
    Z/+00:00 spelling and subsecond precision cannot reorder evidence. Equality
    is in-time; a missing annotation, pin or valid_at applies no exclusion."""
    if annotation is None or as_of is None:
        return False
    valid_at = annotation.get("valid_at")
    if valid_at is None:
        return False
    return datetime.fromisoformat(valid_at) > datetime.fromisoformat(as_of)


_OUTCOME_GOLD_LABELS = frozenset({"accepted"})


def _is_admitted_outcome_gold(
    label: str | None,
    has_posterior: bool,
    labeler_policy_version: str | None,
    decisive_mix: bool,
    decisive_only: bool,
) -> bool:
    """Require accepted, posterior-backed, decisive-only evidence with a known classifier policy.

    Missing policy versions and contested/non-decisive mixes never qualify."""
    return (
        label is not None
        and label in _OUTCOME_GOLD_LABELS
        and has_posterior
        and labeler_policy_version is not None
        and not decisive_mix
        and decisive_only
    )



def _trajectory_set_hash(session_ids: list[str]) -> str:
    """Hash sorted session IDs joined by newlines, without a trailing newline."""
    joined = "\n".join(sorted(session_ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()
