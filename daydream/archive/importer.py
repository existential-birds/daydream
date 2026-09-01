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

import hashlib
import json
from typing import Any

__all__ = ["canonical_payload_digest", "dedupe_observations", "link_session_identity"]

_REASON_UNMATCHED = "no_hub_entry"
_REASON_CONFLICT = "derivative_digest_conflict"

# The writer's versioned auto-dedup tuple (daydream/archive/index.py): the
# evidence identity key. A policy-version bump, reward-version bump, or
# edited-reply digest change appends a new generation instead of deduping.
_TUPLE_FIELDS = (
    "evidence_sha",
    "labeler_policy_version",
    "reply_evidence_digest",
    "labels",
    "has_posterior",
    "reward_version",
)


def canonical_payload_digest(row: dict[str, Any], *, include_observed_at: bool) -> str:
    """Content digest over the canonical-JSON observation payload.

    The bitemporal ``observed_at`` stamp is excluded for auto rows — identical
    evidence captured by two overlapping backups at different capture times
    must dedupe regardless of ``observed_at`` (the SQLite PK
    ``(session_id, observed_at)`` in the target handles identical-timestamp
    overlap). Human rows include it: they are never auto-deduped by the
    writer, so only byte-identical human rows collapse.
    """
    payload = {k: v for k, v in row.items() if include_observed_at or k != "observed_at"}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dedup_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["source"],
        *(row.get(field) for field in _TUPLE_FIELDS),
    )


def dedupe_observations(inventories: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Merge inventory rows across overlapping backups, deduping by content.

    Args:
        inventories: One list of ``label_observations`` rows per backup root,
            each row carrying the table's columns (including ``source``).

    Returns:
        ``{"rows": [...], "deduped_count": int, "content_conflict": [...]}``
        where ``rows`` is the deterministic merged row set and
        ``content_conflict`` holds rows whose dedup tuple matched across
        inventories but whose immutable payload digests disagreed — ambiguous
        evidence, reported rather than silently resolved. ``deduped_count``
        counts dropped duplicate rows (the conflict bucket keeps its rows, so
        bucket rows + surviving rows always account for every input row).

    Dedupe key: the writer's versioned auto-dedup tuple plus the canonical
    payload digest. Identical evidence with identical payload dedupes
    regardless of ``observed_at``; the surviving row keeps the earliest
    ``observed_at``. Distinct evidence generations are never collapsed —
    every generation survives (M3, never keep-latest).

    Deterministic: the merged rows are sorted by ``session_id`` then
    ``observed_at`` (then payload digest), so inventory input order cannot
    affect the output — byte-identical re-import by construction (M4/AC1).
    """
    # Group every input row by dedup tuple, carrying its payload digest.
    groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], str]]] = {}
    for inventory in inventories:
        for row in inventory:
            human = row["source"] != "auto"
            key = _dedup_tuple(row)
            if human:
                # Human rows are never auto-deduped by the writer: only
                # byte-identical rows (including observed_at) collapse, and
                # distinct stamps are legitimate generations — never
                # content conflicts.
                key = (*key, canonical_payload_digest(row, include_observed_at=True))
            groups.setdefault(key, []).append(
                (row, canonical_payload_digest(row, include_observed_at=human))
            )

    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    deduped_count = 0
    for group in groups.values():
        digests = {digest for _, digest in group}
        if len(digests) > 1:
            # Same dedup tuple, differing immutable payloads: ambiguous.
            conflicts.extend(row for row, _ in group)
            continue
        # Byte-identical payload: keep one representative with the earliest
        # observed_at; drop the rest as duplicates.
        ordered = sorted(group, key=lambda item: (str(item[0]["observed_at"]), item[1]))
        rows.append(ordered[0][0])
        deduped_count += len(ordered) - 1

    rows.sort(key=lambda r: (str(r["session_id"]), str(r["observed_at"]), r["source"]))
    conflicts.sort(key=lambda r: (str(r["session_id"]), str(r["observed_at"]), r["source"]))
    return {"rows": rows, "deduped_count": deduped_count, "content_conflict": conflicts}


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
