"""Low-level time parsing helpers shared across daydream."""

from datetime import datetime


def parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp, normalizing a trailing ``Z`` to ``+00:00``."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
