"""Privacy checks for raw records, without constructing archive directories."""
from __future__ import annotations

from typing import Any

from daydream.archive.git_safe import classify_remote_url, normalize_remote_url
from daydream.archive.scan import scan_serialized_record
from daydream.redaction import redact_value


def sanitize_evidence(value: Any) -> Any:
    """Copy/redact producer evidence before its captured-content digests are made."""
    def normalize(child: Any) -> Any:
        if isinstance(child, dict):
            return {key: normalize(item) for key, item in child.items()}
        if isinstance(child, (list, tuple)):
            return [normalize(item) for item in child]
        if isinstance(child, str) and classify_remote_url(child):
            _identity, canonical = normalize_remote_url(child)
            return canonical if canonical is not None else child
        return child

    return redact_value(normalize(value))


def record_is_private(value: dict[str, Any], serialized: str) -> bool:
    """Refuse unsafe immutable input rather than silently changing its digests."""
    try:
        if sanitize_evidence(value) != value:
            return False
        if "[REDACTION_FAILED]" in serialized:
            return False
        return not scan_serialized_record(serialized).blocking
    except Exception:  # noqa: BLE001 - privacy is fail closed
        return False
