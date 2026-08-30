"""Record identity for corpus v2 (Req 2, D2).

Each per-finding training record is identified by a sha256 digest over the
canonical four-component join (US-separated, UTF-8) of the session id,
trajectory id, segment id, and finding fingerprint — mirroring v1's
``_trajectory_set_hash`` canonical-string digest pattern.
"""

import hashlib

_SEPARATOR = "\x1f"


def record_id(session_id: str, trajectory_id: str, segment_id: str, fingerprint: str) -> str:
    """Return the 64-hex lowercase record id for one per-finding record.

    Raises ``ValueError`` naming the offending component when any component is
    missing or empty — no silent empty-string coercion.
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
