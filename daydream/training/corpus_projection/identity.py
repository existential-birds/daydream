"""Projection record identity is SHA-256 over the four signature components joined with ASCII US and
encoded as UTF-8.
"""

import hashlib

_SEPARATOR = "\x1f"


def record_id(session_id: str, trajectory_id: str, segment_id: str, fingerprint: str) -> str:
    """Return the lowercase 64-hex identity; missing or empty components raise ValueError naming the
    component.
    """
    for name, value in (
        ("session_id", session_id),
        ("trajectory_id", trajectory_id),
        ("segment_id", segment_id),
        ("fingerprint", fingerprint),
    ):
        if not value:
            raise ValueError(f"record_id: missing required component {name!r}")
    payload = _SEPARATOR.join((session_id, trajectory_id, segment_id, fingerprint))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
