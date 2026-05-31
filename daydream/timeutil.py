"""Low-level time parsing helpers shared across daydream.

Provide a single canonical parser for the ISO 8601 timestamps (with optional
``Z`` UTC suffix) emitted in ATIF trajectories and read back by the eval,
archive, and PR-comment layers. This module depends only on the standard
library so any daydream module may import it without risk of a cycle.

Exports:
    parse_iso_timestamp: Parse an ISO 8601 timestamp string to a datetime.
"""

from datetime import datetime


def parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp into a datetime.

    Accepts a trailing ``Z`` UTC designator by normalizing it to ``+00:00``
    before delegating to :func:`datetime.fromisoformat`.

    Args:
        value: An ISO 8601 timestamp string, optionally ``Z``-suffixed.

    Returns:
        The parsed datetime; timezone-aware when the input carries an offset.

    Raises:
        ValueError: If ``value`` is not a valid ISO 8601 timestamp.
    """
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
