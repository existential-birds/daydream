"""Content-derived, frozen, disjoint split assignment (corpus v2, D5).

A record's split is a deterministic function of its content-derived record id
and the pinned salt — no RNG, no state, no call-order dependence. The unit
interval is sliced into ``holdout`` / ``validation`` / ``train`` in that
order, so the three membership sets are disjoint by construction.
"""

import hashlib
from typing import Literal

__all__ = ["assign_split"]

Split = Literal["train", "validation", "holdout"]


def assign_split(record_id: str, *, holdout_rate: float, val_rate: float, salt: str) -> Split:
    """Assign one record to a split from its record id (D5).

    The bucket digest is ``sha256(salt <US> record_id)``; its leading 256 bits
    are interpreted as a uniform ``u`` over ``[0, 1)``:

    - ``u < holdout_rate`` → ``holdout``
    - ``u < holdout_rate + val_rate`` → ``validation``
    - otherwise → ``train``

    Deterministic, content-derived, no RNG. Disjoint by construction: the
    three predicates partition the unit interval. ``val_rate`` must be
    non-negative and ``holdout_rate + val_rate <= 1``.
    """
    if holdout_rate < 0.0 or val_rate < 0.0 or holdout_rate + val_rate > 1.0:
        raise ValueError(
            f"assign_split: invalid rates holdout_rate={holdout_rate!r} "
            f"val_rate={val_rate!r} (require non-negative and holdout+val <= 1)"
        )
    digest = hashlib.sha256(f"{salt}\x1f{record_id}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:32], "big") / 2**256
    if u < holdout_rate:
        return "holdout"
    if u < holdout_rate + val_rate:
        return "validation"
    return "train"
