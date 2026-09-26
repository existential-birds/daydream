"""Low-level time parsing helpers shared across daydream."""

from datetime import datetime, timezone


def parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp, normalizing a trailing ``Z`` to ``+00:00``."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def now_iso_utc() -> str:
    """Return the current UTC time as a second-precision ISO-8601 ``Z`` string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
