"""Pure core for importing surviving local archive/backup label observations.

The importer's logic is deterministic and wall-clock-free; the CLI shell owns
all I/O beyond the read-only SQLite connects performed by the inventory. This
module currently provides identity linkage (M2): every imported session is
resolved to a Hub session — via the hydrated staging index join on
``session_id`` + derivative content digest, falling back to
``repo_slug`` + ``base_sha`` + ``head_sha`` — and every unresolvable session
lands in a reason-coded bucket. Nothing is silently dropped (M2): the result
always accounts for every record.
"""

from __future__ import annotations

from typing import Any

__all__ = ["link_session_identity"]

_REASON_UNMATCHED = "no_hub_entry"
_REASON_CONFLICT = "derivative_digest_conflict"


def link_session_identity(
    records: list[dict[str, Any]],
    *,
    hydrated_index: dict[str, dict[str, Any]],
    repo_slug_sha_lookup: dict[tuple[str, str, str], Any],
) -> dict[str, dict[str, Any]]:
    """Resolve each imported record to a Hub session identity.

    Args:
        records: Inventory rows, each carrying ``session_id``, an optional
            ``derivative_digest``, and the optional fallback fields
            ``repo_slug`` / ``base_sha`` / ``head_sha``.
        hydrated_index: ``{session_id: {"derivative_digest": ..., "record_id": ...}}``
            — the hydrate import-ledger join shape.
        repo_slug_sha_lookup: ``{(repo_slug, base_sha, head_sha): hub_session_id}``
            (a dict with ``"hub_session_id"`` is also accepted).

    Returns:
        ``{"linked": {sid: {"hub_session_id", "matched_by"}},
        "unmatched": {sid: reason}, "identity_conflict": {sid: reason}}``.

    Primary rule: ``session_id`` present in ``hydrated_index`` with a matching
    derivative content digest links ``by session_id``. Fallback rule: the
    session_id is absent (or the digest conflicts) and the record carries
    ``repo_slug`` + ``base_sha`` + ``head_sha`` matching the lookup, links
    ``by repo_slug_sha``. A session matching with a conflicting digest routes
    to ``identity_conflict``; a session matching neither routes to
    ``unmatched`` — both with distinct reason strings, never silently skipped.

    Raises:
        ValueError: When a record is unresolvable via the primary rule and is
            missing the session fields required for the fallback — no
            placeholder linkage is produced.
    """
    linked: dict[str, dict[str, str]] = {}
    unmatched: dict[str, str] = {}
    identity_conflict: dict[str, str] = {}

    def _hub_id_from_lookup(value: Any) -> str:
        if isinstance(value, dict):
            return str(value["hub_session_id"])
        return str(value)

    for record in records:
        session_id = str(record["session_id"])
        digest = record.get("derivative_digest")
        hub_entry = hydrated_index.get(session_id)

        if hub_entry is not None:
            hub_digest = hub_entry.get("derivative_digest")
            if hub_digest is not None and hub_digest == digest:
                linked[session_id] = {"hub_session_id": session_id, "matched_by": "session_id"}
                continue
            if hub_digest is not None and digest is not None and hub_digest != digest:
                identity_conflict[session_id] = _REASON_CONFLICT
                continue

        # Fallback: repo_slug + base_sha + head_sha must all be present.
        repo_slug = record.get("repo_slug")
        base_sha = record.get("base_sha")
        head_sha = record.get("head_sha")
        if not repo_slug or not base_sha or not head_sha:
            msg = (
                f"session {session_id!r} has no Hub index entry and is missing "
                "the repo_slug/base_sha/head_sha fields required for the "
                "identity fallback"
            )
            raise ValueError(msg)
        fallback = repo_slug_sha_lookup.get((str(repo_slug), str(base_sha), str(head_sha)))
        if fallback is None:
            unmatched[session_id] = _REASON_UNMATCHED
        else:
            linked[session_id] = {
                "hub_session_id": _hub_id_from_lookup(fallback),
                "matched_by": "repo_slug_sha",
            }

    return {"linked": linked, "unmatched": unmatched, "identity_conflict": identity_conflict}
