"""Inventory, identity linkage, and append-only merge of local adjudication histories."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from daydream.archive.importer import (
    canonical_payload_digest,
    merge_imported_observations,
    redact_imported_metadata,
    redact_metadata_value,
)
from daydream.archive.index import LABEL_OBSERVATION_NAMES, _get_connection, readonly_connection
from daydream.training.adjudication.preview import _SESSIONS_OUT_FILENAME
from daydream.training.labeler_versions import STALE_LEGACY

_IMPORT_VERSION_COLUMNS = (
    "labeler_policy_version",
    "reply_classifier_version",
    "reply_evidence_digest",
)


def _inventory_import_root(root: Path) -> dict[str, Any]:
    """Read a checkpointed archive without SQLite sidecars, preserving row order.

    Fill absent legacy version/source columns, attach run identity and derivative
    digests, and return rows, the index digest, and the complete run inventory.
    Missing index or observation tables raise ValueError naming the root."""
    from daydream.archive.sanitize import _derivative_digest
    from daydream.trajectory import run_directory

    db_path = root / "index.db"
    if not db_path.is_file():
        raise ValueError(
            f"archive root {root} has no index.db; not a daydream archive/backup root"
        )
    source_digest = hashlib.sha256(db_path.read_bytes()).hexdigest()
    with closing(readonly_connection(root)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "label_observations" not in tables:
            raise ValueError(
                f"archive root {root} has no label_observations table in index.db"
            )
        columns = [str(row[1]) for row in conn.execute("PRAGMA table_info(label_observations)")]
        selected = [column for column in LABEL_OBSERVATION_NAMES if column in columns]
        rows = [
            dict(row)
            for row in conn.execute(
                f"SELECT {', '.join(selected)} FROM label_observations ORDER BY observed_at ASC"
            )
        ]
        runs = (
            {str(run["session_id"]): dict(run) for run in conn.execute("SELECT * FROM runs")}
            if "runs" in tables
            else {}
        )
    for row in rows:
        for column in _IMPORT_VERSION_COLUMNS:
            if column not in row:
                row[column] = STALE_LEGACY
        if "source" not in row:
            # A pre-``source`` legacy label_observations table: default the
            # precedence marker to the writer's own default ("auto") so no
            # downstream ``row["source"]`` read raises KeyError on a legacy row.
            row["source"] = "auto"
        runs_dir = run_directory(root, str(row["session_id"]))
        if runs_dir.is_dir():
            # Derivative content digest for identity linkage: the hydrated
            # index side derives the same digest over its own runs/<sid>
            # directory, so a matching pair links by session_id.
            row["derivative_digest"] = _derivative_digest(runs_dir)
        run = runs.get(str(row["session_id"]))
        if run is not None:
            for field in ("repo_slug", "base_sha", "head_sha", "remote_url", "source_path"):
                row[field] = run.get(field)
    return {"rows": rows, "source_digest": source_digest, "runs": runs}


def _seed_target_runs(
    state_dir: Path, sessions: set[str], runs: dict[str, dict[str, Any]]
) -> None:
    """Seed parent runs in session order, filling only NULL target columns.

    Existing status, profile, costs, and denormalized label caches remain the
    projection authority after an overlapping import. Redact remote_url and
    source_path before either insert or fill."""
    with closing(_get_connection(state_dir)) as conn:
        for session_id in sorted(sessions):
            run = dict(runs[session_id])
            for field in ("remote_url", "source_path"):
                if field in run:
                    run[field] = redact_metadata_value(run[field])
            columns = list(run)
            values = [run[column] for column in columns]
            placeholders = ", ".join("?" for _ in columns)
            # A legacy source may omit today's required columns. Only validate
            # insert constraints for absent runs; existing runs just fill NULLs.
            conn.execute(
                f"INSERT INTO runs ({', '.join(columns)}) SELECT {placeholders} "
                "WHERE NOT EXISTS (SELECT 1 FROM runs WHERE session_id = ?)",
                [*values, session_id],
            )
            assignments = ", ".join(f"{column} = COALESCE({column}, ?)" for column in columns)
            conn.execute(
                f"UPDATE runs SET {assignments} WHERE session_id = ?", [*values, session_id],
            )
        conn.commit()


class _ImportGateError(Exception):
    """An import prerequisite failed; its message is displayed verbatim."""


def _inventory_import_roots(
    roots: Sequence[Path], *, console: Any | None = None
) -> dict[str, Any]:
    """Collect rows and source digests, retaining the first run per session.

    Optional progress output is disabled by JSON callers. Inventory failures
    propagate before any merge writes."""
    inventories: list[list[dict[str, Any]]] = []
    sources: list[dict[str, Any]] = []
    runs_by_session: dict[str, dict[str, Any]] = {}
    for root in roots:
        inventory = _inventory_import_root(root)
        inventories.append(inventory["rows"])
        sources.append(
            {
                "archive_root": str(root),
                "row_count": len(inventory["rows"]),
                "source_digest": inventory["source_digest"],
            }
        )
        for session_id, run in inventory["runs"].items():
            runs_by_session.setdefault(session_id, run)
        if console is not None:  # per-root progress (S3)
            console.print(
                f"import: inventoried {len(inventory['rows'])} label_observations(s) "
                f"from {root}"
            )
    return {
        "inventories": inventories,
        "sources": sources,
        "runs_by_session": runs_by_session,
    }


def _link_imported_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Remap only identity-linked rows to Hub sessions and pin their payloads.

    Unmatched/conflicting identities remain in the reason-coded ledger. The
    merge recomputes these digests before writing."""
    linked_rows: list[dict[str, Any]] = []
    for row in result["rows"]:
        link = result["link"]["linked"].get(str(row["session_id"]))
        if link is None:
            continue
        merged_row = dict(row)
        merged_row["session_id"] = link["hub_session_id"]
        merged_row["payload_digest"] = canonical_payload_digest(
            merged_row, include_observed_at=merged_row["source"] != "auto"
        )
        linked_rows.append(merged_row)
    return linked_rows


def _load_import_index_sessions(index_root: Path) -> list[dict[str, Any]]:
    """Read sessions from a materialized snapshot or immutable hydrated archive.

    Neither input present is an identity-derivation failure, never an empty
    inventory that could authorize unrelated backup rows."""
    if not (index_root / _SESSIONS_OUT_FILENAME).is_file() and not (index_root / "index.db").is_file():
        raise ValueError(
            f"import index root {index_root} has neither sessions.jsonl nor index.db; "
            "not a hydrated index or materialized snapshot"
        )
    from daydream.training.adjudication.materialize import index_sessions

    return index_sessions(index_root)[0]


def _load_import_index_runs(
    index_root: Path, sessions: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Read every eligible parent run from the pinned archive, without sidecars.

    A sessions-only snapshot has no authoritative runs table. A hydrated index
    must contain every eligible session, including sessions with no surviving
    backup observation, so a resumed harvest can append their first labels.
    Seeding these parents never creates observation history."""
    db_path = index_root / "index.db"
    if not db_path.is_file():
        return {}
    from daydream.training.adjudication.materialize import _readonly_query

    available = {
        str(row["session_id"]): row
        for row in _readonly_query(db_path, "SELECT * FROM runs")
    }
    eligible = {str(session["session_id"]) for session in sessions}
    missing = sorted(eligible - available.keys())
    if missing:
        raise ValueError(
            "hydrated import index is missing runs for eligible session(s): "
            + ", ".join(missing)
        )
    return {session_id: available[session_id] for session_id in sorted(eligible)}


def _pinned_identity_lookup(
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str, str], str]:
    """Only unique identities in the pinned curation may link a backup row."""
    candidates: dict[tuple[str, str, str], list[str]] = {}
    for session_id, row in runs.items():
        slug, base, head = (row.get(key) for key in ("repo_slug", "base_sha", "head_sha"))
        if slug and base and head:
            candidates.setdefault((str(slug), str(base), str(head)), []).append(session_id)
    return {key: sessions[0] for key, sessions in candidates.items() if len(sessions) == 1}


def _hydrated_identity_index(
    sessions: list[dict[str, Any]], index_root: Path
) -> dict[str, dict[str, Any]]:
    """Map pinned sessions to their derivative digest (or None) and record id."""
    from daydream.archive.sanitize import _derivative_digest
    from daydream.trajectory import run_directory

    hydrated: dict[str, dict[str, Any]] = {}
    for session in sessions:
        session_id = str(session["session_id"])
        runs_dir = run_directory(index_root, session_id)
        hydrated[session_id] = {
            "derivative_digest": _derivative_digest(runs_dir) if runs_dir.is_dir() else None,
            "record_id": session_id,
        }
    return hydrated


def _projector_findings_map(
    sessions: list[dict[str, Any]],
    runs_by_session: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Project finding identities and evidence anchors from pinned sessions.

    Prefer the run head_sha, matching the archive writer's evidence_sha. With no
    run anchor, use the reply digest; this intentionally leaves imported rows
    ambiguous because the archive never stored that digest as its anchor."""
    from daydream.training.corpus_projection.projector import project_findings
    from daydream.training.labeler_versions import reply_evidence_digest

    findings_map: dict[str, list[dict[str, Any]]] = {}
    for session in sessions:
        session_id = str(session["session_id"])
        run = (runs_by_session or {}).get(session_id) or {}
        run_anchor = run.get("head_sha")
        rows: list[dict[str, Any]] = []
        for finding in project_findings(session):
            evidence = finding["evidence"]
            if not isinstance(evidence, list):
                raise ValueError(
                    f"session {session_id!r}: projected finding "
                    f"{finding['finding_fingerprint']!r} carries malformed evidence"
                )
            rows.append(
                {
                    "record_id": finding["record_id"],
                    "evidence_sha": (
                        str(run_anchor) if run_anchor else reply_evidence_digest(evidence)
                    ),
                    "fingerprint": finding["finding_fingerprint"],
                }
            )
        findings_map[session_id] = rows
    return findings_map


def _identity_summary(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Report each session's linkage method and finding-validation outcome.

    Sorted session ids keep reports digest-stable; unlinked identities remain
    unmatched regardless of any run-level evidence."""
    link = result["link"]
    run_level = result["run_level"]
    summary: dict[str, dict[str, Any]] = {}
    for session_id in sorted(
        set(link["linked"]) | set(link["unmatched"]) | set(link["identity_conflict"])
    ):
        if session_id in link["linked"]:
            matched_by: str | None = link["linked"][session_id]["matched_by"]
            if session_id in run_level["per_finding"]:
                outcome = "matched"
            elif session_id in run_level["ambiguous_run_mapping"]:
                outcome = "ambiguous"
            else:
                outcome = "run_level_only"
        else:
            matched_by = None
            outcome = "unmatched"
        summary[session_id] = {"matched_by": matched_by, "validation_outcome": outcome}
    return summary


def _write_import_merge(
    archive_dir: Path,
    state_dir: Path,
    linked_rows: list[dict[str, Any]],
    runs_by_session: dict[str, dict[str, Any]],
    index_runs_by_session: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Redact and validate linked rows before seeding or appending to archive_dir.

    Raw-row digest/timestamp validation and metadata redaction precede archive
    writes. Re-pin transformed payloads before the real merge. Pinned eligible
    runs override backup metadata; unrelated backup runs stay excluded unless
    a linked observation needs its parent. state_dir is scratch.
    """
    redacted_rows = redact_imported_metadata(linked_rows)
    merge_imported_observations(state_dir, linked_rows, dry_run=True)
    for row in redacted_rows:
        row["payload_digest"] = canonical_payload_digest(
            {k: v for k, v in row.items() if k != "payload_digest"},
            include_observed_at=row["source"] != "auto",
        )
    seed_runs = dict(runs_by_session)
    seed_runs.update(index_runs_by_session)
    seed_sessions = set(index_runs_by_session)
    seed_sessions.update(str(row["session_id"]) for row in redacted_rows)
    _seed_target_runs(archive_dir, seed_sessions, seed_runs)
    merged = merge_imported_observations(archive_dir, redacted_rows, dry_run=False)
    return {
        "planned": merged["planned"],
        "appended": merged["appended"],
        "deduped": merged["deduped"],
        "payload": redacted_rows,
    }


def _build_import_report(
    sources: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    dry_run: bool,
    merge_state: dict[str, Any],
    identity_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compose deterministic accounting and identity results."""
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "sources": sources,
        "deduped_count": result["deduped_count"],
        "accounting": dict(result["accounting"]),
        "identity_summary": identity_summary,
        "merge": {
            "planned": len(merge_state["planned"]),
            "appended": merge_state["appended"],
            "deduped": merge_state["deduped"],
        },
    }
    return report
