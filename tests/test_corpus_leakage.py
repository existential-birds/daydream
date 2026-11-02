"""Posterior-derived labels must not leak into a corpus pinned before their outcome.

The shared projection guard compares parsed datetimes, including equivalent zone
spellings, fractional seconds, and non-UTC offsets.
"""
from __future__ import annotations

# _is_posterior_leak boundary semantics (unit): valid_at == as_of is in-time,
# strictly greater is a leak, and no ISO-8601 spelling difference mis-orders it.

AS_OF = "2026-04-01T00:00:00+00:00"
