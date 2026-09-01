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
import re
from pathlib import Path
from typing import Any

from daydream.archive.hydrate_rules import (
    HYDRATION_INDEX_SCHEMA_VERSION,
    REASON_CODE_IMPORT_DECISIVE_PER_FINDING,
    REASON_CODE_IMPORT_IDENTITY_CONFLICT,
    REASON_CODE_IMPORT_INVALID_VERSION,
    REASON_CODE_IMPORT_RUN_LEVEL_ONLY,
    REASON_CODE_IMPORT_STALE_EVIDENCE,
    REASON_CODE_IMPORT_UNMATCHED_SESSION,
    REASON_CODE_IMPORT_UNREDACTABLE_METADATA,
)
from daydream.archive.index import append_label_observation
from daydream.archive.known_versions import KNOWN_LABELER_VERSIONS
from daydream.archive.sanitize import _sanitize_json_value
from daydream.archive.scan import scan_run_dir
from daydream.trajectory import redact_value

__all__ = [
    "IMPORT_REASON_CODES",
    "REDACTED_PATH",
    "accounting",
    "build_import_ledger",
    "canonical_payload_digest",
    "classify_run_level",
    "dedupe_observations",
    "gold_eligible",
    "link_session_identity",
    "merge_imported_observations",
    "redact_metadata_value",
    "run_pure_import",
]

# The fixed six-bucket import accounting registry (M7): every imported
# observation row lands in exactly one of these stable reason codes.
IMPORT_REASON_CODES = (
    REASON_CODE_IMPORT_UNMATCHED_SESSION,
    REASON_CODE_IMPORT_IDENTITY_CONFLICT,
    REASON_CODE_IMPORT_STALE_EVIDENCE,
    REASON_CODE_IMPORT_INVALID_VERSION,
    REASON_CODE_IMPORT_DECISIVE_PER_FINDING,
    REASON_CODE_IMPORT_RUN_LEVEL_ONLY,
)

# Marker replacing redacted absolute local paths. Carries the ``[REDACTED_``
# prefix the scanner treats as already-safe output, so redacted payloads do
# not flag their own markers.
REDACTED_PATH = "[REDACTED_PATH]"

# Absolute local path embedded inside a larger string. The lookbehind blocks
# scheme separators (``https://…``) and UNC double slashes, so canonical git
# URLs survive untouched; standalone path values are caught by the
# startswith("/") check in :func:`_redact_path_string`.
_EMBEDDED_ABSOLUTE_PATH_RE = re.compile(r"(?<![:/\w])(/[\w.-]+(?:/[\w.-]+)+)")

# Writer columns consumed by append_label_observation; every other key on an
# import row (legacy, payload_digest, ...) is importer metadata, not payload.
_WRITER_FIELDS = (
    "labels",
    "pr_state",
    "labeler_version",
    "evidence_sha",
    "rubric_json",
    "valid_at",
    "reward_version",
    "reward_json",
    "composite_reward",
    "reviewer_logins",
    "has_posterior",
    "source",
    "reply_classifier_version",
    "reply_evidence_digest",
)

_REASON_UNMATCHED = "no_hub_entry"
_REASON_CONFLICT = "derivative_digest_conflict"
_REASON_RUN_LEVEL_ONLY = "no_projected_findings"
_REASON_AMBIGUOUS = "ambiguous_finding_mapping"

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
    overlap). The collapsed ``valid_at`` stamp is excluded alongside it: the
    writer folds ``valid_at=None`` onto ``observed_at`` (index.py), so two
    captures of identical auto evidence carry byte-different ``valid_at``
    values that must not split the dedupe into a content conflict. Human rows
    include both stamps: they are never auto-deduped by the writer, so only
    byte-identical human rows collapse.
    """
    excluded = set()
    if not include_observed_at:
        excluded |= {"observed_at", "valid_at"}
    payload = {k: v for k, v in row.items() if k not in excluded}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _redact_path_string(value: str) -> str:
    """Redact absolute local paths in one string value (never raises)."""
    if value.startswith("/"):
        return REDACTED_PATH
    return _EMBEDDED_ABSOLUTE_PATH_RE.sub(REDACTED_PATH, value)


def _redact_json_blob(value: Any, *, field: str, session_id: str) -> Any:
    """Redact one ``rubric_json``/``reward_json`` payload (URL creds, secrets,
    absolute local paths).

    Raises:
        ValueError: When a JSON-encoded string blob is malformed — never
            silently substituted (fail-closed, names the row + field).
    """
    if value is None:
        return None
    if isinstance(value, str):
        was_string = True
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            msg = (
                f"imported row for session {session_id!r} has malformed "
                f"{field} JSON: {exc}"
            )
            raise ValueError(msg) from exc
    else:
        was_string = False
    if not isinstance(value, (dict, list)):
        return redact_value(_redact_path_string(str(value)))

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: _walk(child) for key, child in node.items()}
        if isinstance(node, list):
            return [_walk(child) for child in node]
        if isinstance(node, str):
            return redact_value(_redact_path_string(_sanitize_json_value(node)))
        return node

    redacted = _walk(value)
    if was_string and isinstance(redacted, (dict, list)):
        # Preserve the column's JSON-encoded string representation.
        return json.dumps(redacted, sort_keys=True, default=str)
    return redacted


def redact_metadata_value(value: Any) -> Any:
    """Redact one credential-bearing metadata field (URL userinfo, absolute paths).

    Applies the same ``sanitize`` + ``redact_value`` chain
    :func:`redact_imported_metadata` uses per row field, so a persisted runs
    row's ``remote_url``/``source_path`` carry the same fail-closed scrubbing
    the observation rows do. Non-string values pass through unchanged.
    """
    if isinstance(value, str):
        return redact_value(_redact_path_string(_sanitize_json_value(value)))
    return value


def redact_imported_metadata(rows: list[dict[str, Any]], *, scan_dir: Path) -> dict[str, Any]:
    """Redact pre-publication metadata and fail closed on uncleanable rows (M9, AC6).

    Every row's ``remote_url`` and ``source_path`` are rewritten through the
    sanitize module's URL authority plus absolute-path redaction, and the
    ``rubric_json``/``reward_json`` blobs are walked string-leaf by string-leaf
    through ``sanitize._sanitize_json_value`` + ``redact_value`` (reuse, not
    reimplementation). The redacted payload is then serialized to
    ``scan_dir/payload.json`` and re-scanned with the fail-closed
    :func:`daydream.archive.scan.scan_run_dir`.

    Returns:
        ``{"payload": [...], "blocked": bool, "scan_summary": str,
        "blocked_reasons": [...]}``. ``blocked`` is ``True`` only when the
        post-redaction scan is dirty — the payload cannot be published; the
        offending rows are kept in ``payload`` (never dropped silently) and
        ``blocked_reasons`` carries the stable
        ``import_unredactable_metadata`` reason code. Redaction runs before
        ``publish_annotation_state`` (which re-scans and hard-fails on dirty).

    Raises:
        ValueError: When a ``rubric_json``/``reward_json`` blob is malformed
            JSON — redaction errors propagate, never placeholder-substituted.
    """
    payload: list[dict[str, Any]] = []
    for row in rows:
        session_id = str(row.get("session_id", ""))
        redacted = dict(row)
        for field in ("remote_url", "source_path"):
            redacted[field] = redact_metadata_value(redacted.get(field))
        redacted["rubric_json"] = _redact_json_blob(
            redacted.get("rubric_json"), field="rubric_json", session_id=session_id
        )
        redacted["reward_json"] = _redact_json_blob(
            redacted.get("reward_json"), field="reward_json", session_id=session_id
        )
        payload.append(redacted)

    scan_dir.mkdir(parents=True, exist_ok=True)
    payload_path = scan_dir / "payload.json"
    payload_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    scan = scan_run_dir(scan_dir)
    blocked = not scan.clean
    return {
        "payload": payload,
        "blocked": blocked,
        "scan_summary": scan.summary(),
        "blocked_reasons": [REASON_CODE_IMPORT_UNREDACTABLE_METADATA] if blocked else [],
    }


def _planned_append(row: dict[str, Any]) -> dict[str, Any]:
    """Shape one import row into append_label_observation keyword arguments."""
    plan: dict[str, Any] = {
        "session_id": row["session_id"],
        "observed_at": row["observed_at"],
    }
    for field in _WRITER_FIELDS:
        value = row.get(field)
        # Rows read from SQLite carry JSON-encoded list columns; the writer
        # expects the decoded Python values.
        if field in ("labels", "reviewer_logins") and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ValueError(
                    f"imported row for session {row['session_id']!r} at "
                    f"{row['observed_at']!r} has malformed {field} JSON: {value!r}"
                ) from exc
        plan[field] = value
    plan["has_posterior"] = bool(plan["has_posterior"])
    return plan


def merge_imported_observations(
    archive_dir: Path,
    linked_imports: list[dict[str, Any]],
    *,
    observations_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Append deduped, identity-linked import rows through the archive writer.

    Every row goes through :func:`append_label_observation` — the single
    canonical writer — so the versioned auto-dedup tuple, the human-never-dedup
    rule, and the precedence projection that refreshes the denormalized
    ``runs.outcome_labels`` cache all apply unchanged. The importer only ever
    appends: it never overwrites or deletes a newer existing observation (M3),
    and winners are decided by the writer's projection, not by the import
    order.

    Args:
        archive_dir: Target archive root (the Hub-side index being merged into).
        linked_imports: Deduped inventory rows with ``session_id`` already
            remapped to the linked Hub session id. Rows may carry a
            ``payload_digest`` recorded at inventory time.
        observations_path: Optional JSON file of additional import rows (same
            shape), loaded and appended to ``linked_imports``.
        dry_run: When ``True``, return the planned append set without writing
            any state (S2).

    Returns:
        ``{"dry_run": bool, "planned": [...], "appended": int, "deduped": int}``
        where ``planned`` holds one writer-kwarg dict per import row in
        deterministic ``(session_id, observed_at, source)`` order and
        ``appended``/``deduped`` count writer outcomes (both 0 on dry run).

    Raises:
        ValueError: Fail-closed, before any write, when a row's recomputed
            canonical payload digest disagrees with its inventory-time
            ``payload_digest`` (evidence drifted between inventory and merge),
            when a row's JSON list columns are malformed, or when the writer
            rejects the row (unknown session, non-ISO-8601 ``observed_at``) —
            each error names the offending row. Never silently defaulted.
    """
    imports = list(linked_imports)
    if observations_path is not None and observations_path.is_file():
        loaded = json.loads(observations_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(
                f"observations file {observations_path} must contain a JSON list of rows"
            )
        imports.extend(loaded)

    imports.sort(key=lambda r: (str(r["session_id"]), str(r["observed_at"]), str(r["source"])))

    # Fail-closed drift gate before any write: recompute the same canonical
    # payload digest the inventory recorded and reject any drifted row
    # (mirrors run_canonical_harvest's AnnotationDriftError path).
    for row in imports:
        if "payload_digest" not in row:
            continue
        payload = {k: v for k, v in row.items() if k != "payload_digest"}
        fresh = canonical_payload_digest(
            payload, include_observed_at=row["source"] != "auto"
        )
        if fresh != row["payload_digest"]:
            raise ValueError(
                f"imported observation for session {row['session_id']!r} at "
                f"{row['observed_at']!r} drifted from its inventory payload digest "
                f"(expected {row['payload_digest']}, recomputed {fresh}); "
                f"re-inventory the source before merging"
            )

    planned = [_planned_append(row) for row in imports]
    if dry_run:
        return {"dry_run": True, "planned": planned, "appended": 0, "deduped": 0}

    appended = 0
    deduped = 0
    for plan in planned:
        if append_label_observation(archive_dir, plan.pop("session_id"), **plan):
            appended += 1
        else:
            deduped += 1
    return {"dry_run": False, "planned": planned, "appended": appended, "deduped": deduped}


def _dedup_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source", "auto"),
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
            human = row.get("source", "auto") != "auto"
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

    rows.sort(key=lambda r: (str(r["session_id"]), str(r["observed_at"]), r.get("source", "auto")))
    conflicts.sort(key=lambda r: (str(r["session_id"]), str(r["observed_at"]), r.get("source", "auto")))
    return {"rows": rows, "deduped_count": deduped_count, "content_conflict": conflicts}


def link_session_identity(
    records: list[dict[str, Any]],
    *,
    hydrated_index: dict[str, dict[str, Any]],
    repo_slug_sha_lookup: dict[tuple[str, str, str], Any],
    unmatched_identity_less: bool = False,
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
        unmatched_identity_less: When ``True`` (the import CLI's wiring over
            local-only archive roots with an empty hydrated index), an
            identity-less record — one absent from ``hydrated_index`` and
            missing the fallback session fields — routes to ``unmatched``
            instead of raising, so a single repo-less local run cannot abort
            a whole-archive import.

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
            placeholder linkage is produced. (Suppressed in favor of
            ``unmatched`` only when ``unmatched_identity_less`` is ``True``.)
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
            if unmatched_identity_less:
                unmatched[session_id] = _REASON_UNMATCHED
                continue
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


def classify_run_level(
    records: list[dict[str, Any]],
    *,
    projector_findings: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Partition session-scoped rows into run-level vs per-finding evidence (M5, AC2).

    Args:
        records: Merged inventory rows. A row without a truthy ``record_id``
            is a run-level (session-scoped) label; a row carrying one is
            already per-finding evidence.
        projector_findings: ``{session_id: [finding, ...]}`` mirroring the
            ``corpus_v2.projector.project_findings`` enumeration — the single
            authority for the non-decisive set. Each finding dict must carry
            ``record_id`` and ``evidence_sha``.

    Returns:
        ``{"per_finding": {sid: [obs, ...]}, "run_level_only":
        {sid: reason}, "ambiguous_run_mapping": {sid: reason}}`` — a
        deterministic partition accounting for every input row (M7).

    Behavior:
        A run-level label is emitted as **run-level evidence only** — never
        copied onto every finding of the run (AC2). A row with no projected
        findings for its session is run-level-only. A row whose session has
        multiple candidate findings with no unique decisive match on identity
        + evidence digest routes to ``ambiguous_run_mapping`` — feeding the
        per-finding adjudication queue, not a fan-out. Only a row whose
        ``evidence_sha`` matches exactly one projected finding lands in
        ``per_finding`` (the sole path into the ``_is_admitted_outcome_gold``
        / ``_rubric_decisive_only`` semantics).

    Raises:
        ValueError: When a row's ``labels`` field is malformed JSON (naming
            the session_id), or when a referenced ``projector_findings``
            entry is missing ``record_id``/``evidence_sha`` — never silently
            substituted.
    """
    per_finding: dict[str, list[dict[str, Any]]] = {}
    run_level_only: dict[str, str] = {}
    ambiguous: dict[str, str] = {}

    for record in records:
        session_id = str(record["session_id"])
        if record.get("record_id"):
            # Already per-finding evidence: accounted in per_finding, never
            # routed through the run-level buckets.
            per_finding.setdefault(session_id, []).append(record)
            continue

        labels = record.get("labels")
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except json.JSONDecodeError as exc:
                msg = f"session {session_id!r} has malformed labels JSON: {exc}"
                raise ValueError(msg) from exc
        if not isinstance(labels, list):
            msg = f"session {session_id!r} has non-list labels field: {labels!r}"
            raise ValueError(msg)

        findings = projector_findings.get(session_id)
        if not findings:
            run_level_only[session_id] = _REASON_RUN_LEVEL_ONLY
            continue

        evidence_sha = record.get("evidence_sha")
        matches = []
        for finding in findings:
            if "record_id" not in finding or "evidence_sha" not in finding:
                msg = (
                    f"session {session_id!r} references a malformed projected "
                    f"finding (missing 'record_id'/'evidence_sha'): {finding!r}"
                )
                raise ValueError(msg)
            if finding["evidence_sha"] == evidence_sha:
                matches.append(finding)

        if len(matches) == 1:
            # Decisive identity+evidence-digest match on exactly one finding.
            per_finding.setdefault(session_id, []).append(record)
        else:
            # No match, or the digest matches more than one finding: the
            # run<->finding mapping is ambiguous — adjudication queue, not a
            # fan-out.
            ambiguous[session_id] = _REASON_AMBIGUOUS

    return {
        "per_finding": per_finding,
        "run_level_only": run_level_only,
        "ambiguous_run_mapping": ambiguous,
    }


def _row_reason_codes(
    merged_rows: list[dict[str, Any]],
    *,
    content_conflict: list[dict[str, Any]],
    link_result: dict[str, Any],
    run_level_result: dict[str, Any],
) -> list[tuple[dict[str, Any], str]]:
    """Classify every row into exactly one import reason code (M7).

    Precedence: dedupe/identity conflicts first (the row never got a home),
    then the version gate, then the run-level routing. Raises ``ValueError``
    naming any row that cannot be classified — never an implicit drop.
    """
    conflict_ids = {id(row) for row in content_conflict}
    per_finding_ids = {
        id(row)
        for rows in run_level_result["per_finding"].values()
        for row in rows
    }
    unmatched_sids = set(link_result["unmatched"])
    link_conflict_sids = set(link_result["identity_conflict"])
    run_level_only_sids = set(run_level_result["run_level_only"])
    ambiguous_sids = set(run_level_result["ambiguous_run_mapping"])

    classified: list[tuple[dict[str, Any], str]] = []
    for row in (*content_conflict, *merged_rows):
        session_id = str(row["session_id"])
        if id(row) in conflict_ids or session_id in link_conflict_sids:
            code = REASON_CODE_IMPORT_IDENTITY_CONFLICT
        elif session_id in unmatched_sids:
            code = REASON_CODE_IMPORT_UNMATCHED_SESSION
        elif not gold_eligible(row):
            code = REASON_CODE_IMPORT_INVALID_VERSION
        elif id(row) in per_finding_ids:
            code = REASON_CODE_IMPORT_DECISIVE_PER_FINDING
        elif session_id in run_level_only_sids:
            code = REASON_CODE_IMPORT_RUN_LEVEL_ONLY
        elif session_id in ambiguous_sids:
            code = REASON_CODE_IMPORT_STALE_EVIDENCE
        else:
            msg = (
                f"imported observation for session {session_id!r} at "
                f"{row.get('observed_at')!r} cannot be classified into an "
                "import bucket; refusing to drop the row silently"
            )
            raise ValueError(msg)
        classified.append((row, code))
    return classified


def accounting(
    merged_rows: list[dict[str, Any]],
    *,
    content_conflict: list[dict[str, Any]],
    link_result: dict[str, Any],
    run_level_result: dict[str, Any],
) -> dict[str, int]:
    """Map every merged inventory row to exactly one import reason code (M7).

    Args:
        merged_rows: The surviving deduped rows (from
            :func:`dedupe_observations`), each carrying ``session_id``.
        content_conflict: Rows routed to the dedupe content-conflict bucket —
            their evidence identity is ambiguous, so they account as
            ``import_identity_conflict``.
        link_result: The :func:`link_session_identity` result.
        run_level_result: The :func:`classify_run_level` result.

    Returns:
        ``{reason_code: row_count}`` over :data:`IMPORT_REASON_CODES` whose
        values sum to ``len(merged_rows) + len(content_conflict)``.

    Raises:
        ValueError: When a row cannot be classified into a named bucket —
            never an implicit drop.
    """
    counts = {code: 0 for code in IMPORT_REASON_CODES}
    for _row, code in _row_reason_codes(
        merged_rows,
        content_conflict=content_conflict,
        link_result=link_result,
        run_level_result=run_level_result,
    ):
        counts[code] += 1
    return counts


def run_pure_import(
    inventories: list[list[dict[str, Any]]],
    *,
    hydrated_index: dict[str, dict[str, Any]],
    repo_slug_sha_lookup: dict[tuple[str, str, str], Any],
    projector_findings: dict[str, list[dict[str, Any]]],
    unmatched_identity_less: bool = False,
) -> dict[str, Any]:
    """Compose the pure import pipeline: dedupe -> link -> run-level -> accounting.

    Args:
        inventories: One list of ``label_observations`` rows per backup root.
        hydrated_index: See :func:`link_session_identity`.
        repo_slug_sha_lookup: See :func:`link_session_identity`.
        projector_findings: See :func:`classify_run_level`.
        unmatched_identity_less: Threaded to :func:`link_session_identity`;
            see its annotation.

    Returns:
        ``{"rows", "deduped_count", "content_conflict", "link",
        "run_level", "accounting", "ledger"}`` — the deterministic pipeline
        result, where ``accounting`` sums to the full source row inventory
        count (deduped rows + dedupe conflicts) and ``ledger`` is the
        :func:`build_import_ledger` shape.
    """
    merged = dedupe_observations(inventories)
    link_result = link_session_identity(
        merged["rows"],
        hydrated_index=hydrated_index,
        repo_slug_sha_lookup=repo_slug_sha_lookup,
        unmatched_identity_less=unmatched_identity_less,
    )
    run_level_result = classify_run_level(
        merged["rows"], projector_findings=projector_findings
    )
    counts = accounting(
        merged["rows"],
        content_conflict=merged["content_conflict"],
        link_result=link_result,
        run_level_result=run_level_result,
    )
    result: dict[str, Any] = {
        "rows": merged["rows"],
        "deduped_count": merged["deduped_count"],
        "content_conflict": merged["content_conflict"],
        "link": link_result,
        "run_level": run_level_result,
        "accounting": counts,
    }
    result["ledger"] = build_import_ledger(result)
    return result


def build_import_ledger(result: dict[str, Any]) -> dict[str, Any]:
    """Shape the import result into the hydrate import-ledger format (KD5).

    Mirrors ``daydream/archive/hydrate.py``'s ledger shape — a fixed schema
    version plus ``{session_id, reason_code, ...}`` entries — so existing
    ledger consumers see one format. Every source row appears exactly once;
    the accounting sum is re-verified here and any mismatch raises (M7:
    no silent drops).

    Raises:
        ValueError: When the bucket sum does not equal the accounted
            observation count.
    """
    accounting = result["accounting"]
    observations = sorted(
        (
            {
                "session_id": str(row["session_id"]),
                "observed_at": str(row["observed_at"]),
                "source": str(row.get("source", "")),
                "reason_code": code,
            }
            for row, code in _row_reason_codes(
                result["rows"],
                content_conflict=result["content_conflict"],
                link_result=result["link"],
                run_level_result=result["run_level"],
            )
        ),
        key=lambda entry: (entry["session_id"], entry["observed_at"], entry["source"]),
    )
    if sum(accounting.values()) != len(observations):
        msg = (
            f"import accounting mismatch: buckets account for "
            f"{sum(accounting.values())} row(s) but the ledger carries "
            f"{len(observations)} observation(s)"
        )
        raise ValueError(msg)
    return {
        "schema_version": HYDRATION_INDEX_SCHEMA_VERSION,
        "accounting": dict(accounting),
        "observations": observations,
    }


def gold_eligible(observation: dict[str, Any]) -> bool:
    """Version gate for gold eligibility (M6, KD3).

    Returns True iff every version axis on the observation — labeler
    (rubric), policy, and classifier versions — is in the known-versions
    allowlist. A missing/``None`` version field, or the ``"legacy"`` sentinel
    stamp, makes the row non-gold (safe default: unknown provenance is never
    decisive). Such rows still import as evidence.
    """
    for field in ("labeler_version", "labeler_policy_version", "reply_classifier_version"):
        value = observation.get(field)
        if not value or str(value) not in KNOWN_LABELER_VERSIONS:
            return False
    return True
