"""Deterministic linkage, deduplication, and accounting for imported observations.

Match hydrated sessions by identity plus content digest, with repo/base/head
fallback. Every record survives or receives a reason code; unresolved evidence
is never silently dropped. Inventory SQLite access is read-only.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
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
from daydream.archive.index import LABEL_OBSERVATION_NAMES, append_label_observation
from daydream.archive.sanitize import _sanitize_url_string
from daydream.archive.scan import scan_run_dir
from daydream.training.labeler_versions import KNOWN_LABELER_VERSIONS, STALE_LEGACY
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

# Every surviving observation gets one stable reason; accounted plus deduped equals the
# full source inventory.
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

# Match embedded local paths while excluding scheme/UNC separators. Standalone
# absolute paths are handled by _redact_path_string's leading-slash check.
_EMBEDDED_ABSOLUTE_PATH_RE = re.compile(r"(?<![:/\w])(/[\w.-]+(?:/[\w.-]+)+)")

# Derive writer fields from the schema; preserve policy provenance and compute legacy
# separately. Importer metadata is excluded.
_WRITER_FIELDS: tuple[str, ...] = tuple(
    name
    for name in LABEL_OBSERVATION_NAMES
    if name not in ("session_id", "observed_at", "legacy")
)

_REASON_UNMATCHED = "no_hub_entry"
_REASON_CONFLICT = "derivative_digest_conflict"
_REASON_RUN_LEVEL_ONLY = "no_projected_findings"
_REASON_AMBIGUOUS = "ambiguous_finding_mapping"

# Keep this evidence identity aligned with the canonical writer auto-dedup tuple.
_TUPLE_FIELDS = (
    "evidence_sha",
    "labeler_policy_version",
    "reply_evidence_digest",
    "labels",
    "has_posterior",
    "reward_version",
)


def canonical_payload_digest(row: dict[str, Any], *, include_observed_at: bool) -> str:
    """Hash the canonical observation payload, optionally including transaction time."""
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
    """Redact a JSON payload; malformed encoded JSON raises with its row and field identity."""
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
            return redact_value(_redact_path_string(_sanitize_url_string(node)))
        return node

    redacted = _walk(value)
    if was_string and isinstance(redacted, (dict, list)):
        # Preserve the column's JSON-encoded string representation.
        return json.dumps(redacted, sort_keys=True, default=str)
    return redacted


def redact_metadata_value(value: Any) -> Any:
    """Scrub URL credentials and absolute paths through the shared sanitizers; retain non-strings."""
    if isinstance(value, str):
        return redact_value(_redact_path_string(_sanitize_url_string(value)))
    return value


def redact_imported_metadata(rows: list[dict[str, Any]], *, scan_dir: Path) -> dict[str, Any]:
    """Scrub imported rows, write the payload, and apply the publication scan."""
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
    # Report advisory findings without exposing values; only blocking findings withhold
    # publication.
    blocked = scan.blocking
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
    # Preserve source legacy markers; missing policy provenance must remain ineligible
    # for gold.
    legacy = row.get("legacy")
    if legacy is None:
        legacy = "legacy" if row.get("labeler_policy_version") == STALE_LEGACY else "auto"
    plan["legacy"] = legacy
    return plan


def merge_imported_observations(
    archive_dir: Path,
    linked_imports: list[dict[str, Any]],
    *,
    observations_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and append linked rows through the canonical archive writer."""
    imports = list(linked_imports)
    if observations_path is not None and observations_path.is_file():
        loaded = json.loads(observations_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(
                f"observations file {observations_path} must contain a JSON list of rows"
            )
        imports.extend(loaded)

    imports.sort(key=lambda r: (str(r["session_id"]), str(r["observed_at"]), str(r["source"])))

    # Validate all recorded inventory digests before writing any row.
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

    # Validate every timestamp before writing, preventing malformed later rows from
    # leaving a partial merge.
    for row in imports:
        for field in ("observed_at", "valid_at"):
            value = row.get(field)
            if value is None:
                continue
            try:
                parsed = datetime.fromisoformat(str(value))
            except ValueError:
                raise ValueError(
                    f"imported observation for session {row['session_id']!r} at "
                    f"{row['observed_at']!r} has a non-ISO-8601 {field}: {value!r}"
                ) from None
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError(
                    f"imported observation for session {row['session_id']!r} at "
                    f"{row['observed_at']!r} has a naive {field} ({value!r}): "
                    "an explicit UTC offset is required"
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
    """Merge overlapping inventories without collapsing distinct evidence generations."""
    # Group every input row by dedup tuple, carrying its payload digest.
    groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], str]]] = {}
    for inventory in inventories:
        for row in inventory:
            human = row.get("source", "auto") != "auto"
            digest = canonical_payload_digest(row, include_observed_at=human)
            key = _dedup_tuple(row)
            if human:
                # Human generations dedupe only when content and observation time are
                # identical.
                key = (*key, digest)
            groups.setdefault(key, []).append((row, digest))

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
    """Link records to hydrated sessions by identity/digest, then repo/base/head."""
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
    """Partition observations into per-finding, run-level-only, or ambiguous evidence."""
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
            # Ambiguous finding attribution goes to adjudication and never fans out.
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
    """Classify every row: conflicts, then version eligibility, then run-level routing; reject gaps."""
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
    """Count surviving and conflicting rows by import reason code."""
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
    """Compose dedupe, identity linkage, finding classification, accounting, and ledger."""
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
    """Build the versioned hydration-compatible ledger and verify complete row accounting."""
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
    """Admit gold only when labeler, policy, and classifier versions are allowlisted."""
    for field in ("labeler_version", "labeler_policy_version", "reply_classifier_version"):
        value = observation.get(field)
        if not value or str(value) not in KNOWN_LABELER_VERSIONS:
            return False
    return True
