"""Inventory, identity linkage, and append-only merge of local adjudication histories."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from daydream.archive.importer import (
    ImportPlan,
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


def _load_import_index(index_root: Path) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]],
    dict[tuple[str, str, str], str], dict[str, list[dict[str, Any]]],
]:
    """Acquire pinned sessions and their authoritative run/derivative/finding identities.

    Never authorize unrelated backup rows: eligible parents come from current source
    findings, triplet aliases must be unique, and missing run metadata fails closed.
    Sessions-only snapshots retain digest-less and run-anchor fallback behavior.
    Prefer run head_sha to match archive evidence_sha. The reply-digest fallback
    intentionally remains ambiguous: the archive never stored that anchor.
    """
    from daydream.archive.sanitize import _derivative_digest
    from daydream.training.adjudication.materialize import index_sessions
    from daydream.training.adjudication.snapshot import FindingRecord
    from daydream.training.labeler_versions import reply_evidence_digest
    from daydream.trajectory import run_directory

    if not (index_root / _SESSIONS_OUT_FILENAME).is_file() and not (index_root / "index.db").is_file():
        raise ValueError(
            f"import index root {index_root} has neither sessions.jsonl nor index.db; "
            "not a hydrated index or materialized snapshot"
        )
    sessions, _revision, available = index_sessions(index_root)
    if available is None and (index_root / "index.db").is_file():
        # A materialized snapshot may sit beside its parent archive; only import
        # needs that optional table, while materialize/preview keep their file policy.
        with closing(readonly_connection(index_root)) as conn:
            available = {str(row["session_id"]): dict(row) for row in conn.execute("SELECT * FROM runs")}
    eligible = {str(session["session_id"]) for session in sessions}
    missing = sorted(eligible - available.keys()) if available is not None else []
    if missing:
        raise ValueError(
            "hydrated import index is missing runs for eligible session(s): " + ", ".join(missing)
        )
    runs = {sid: available[sid] for sid in sorted(eligible)} if available is not None else {}
    candidates: dict[tuple[str, str, str], list[str]] = {}
    for sid, run in runs.items():
        slug, base, head = (run.get(key) for key in ("repo_slug", "base_sha", "head_sha"))
        if slug and base and head:
            candidates.setdefault((str(slug), str(base), str(head)), []).append(sid)
    aliases = {key: ids[0] for key, ids in candidates.items() if len(ids) == 1}
    hydrated: dict[str, dict[str, Any]] = {}
    findings: dict[str, list[dict[str, Any]]] = {}
    for session in sessions:
        sid = str(session["session_id"])
        run_dir = run_directory(index_root, sid)
        hydrated[sid] = {
            "derivative_digest": _derivative_digest(run_dir) if run_dir.is_dir() else None,
            "record_id": sid,
        }
        anchor = runs.get(sid, {}).get("head_sha")
        findings[sid] = [
            {
                "record_id": finding.record_id,
                "evidence_sha": str(anchor) if anchor else reply_evidence_digest(finding.evidence),
                "fingerprint": finding.fingerprint,
            }
            for finding in FindingRecord.from_session(session)
        ]
    return sessions, runs, hydrated, aliases, findings


def _write_import_merge(
    archive_dir: Path,
    state_dir: Path,
    linked_rows: list[dict[str, Any]],
    runs_by_session: dict[str, dict[str, Any]],
    index_runs_by_session: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Redact and validate linked rows before seeding or appending to archive_dir.

    Both the raw-row drift/timestamp gate and the redacted-payload secret gate
    precede archive writes. Delete a blocked scan artifact so later scans do not
    inherit its dirty bytes. Re-pin redacted payloads before the real merge.
    Pinned eligible runs override backup metadata; unrelated backup runs stay
    excluded unless a linked observation needs its parent. state_dir is scratch."""
    scan = redact_imported_metadata(linked_rows, scan_dir=state_dir / "import-scan")
    merge_imported_observations(state_dir, linked_rows, dry_run=True)
    if scan["blocked"]:
        (state_dir / "import-scan" / "payload.json").unlink(missing_ok=True)
        message = "; ".join(scan["blocked_reasons"]) + f" ({scan['scan_summary']})"
        raise _ImportGateError(message)
    redacted_rows = scan["payload"]
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
        "scan": scan,
    }


def _build_import_report(
    sources: list[dict[str, Any]],
    result: ImportPlan,
    *,
    dry_run: bool,
    merge_state: dict[str, Any],
) -> dict[str, Any]:
    """Compose deterministic accounting and identity results.

    Only a real merge includes the redaction block; dry-run has no scan output."""
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "sources": sources,
        "deduped_count": result.deduped_count,
        "accounting": dict(result.ledger["accounting"]),
        "identity_summary": result.identity_summary,
        "merge": {
            "planned": len(merge_state["planned"]),
            "appended": merge_state["appended"],
            "deduped": merge_state["deduped"],
        },
    }
    if not dry_run:
        report["redaction"] = {
            "blocked": bool(merge_state["scan"]["blocked"]),
            "reasons": list(merge_state["scan"]["blocked_reasons"]),
        }
    return report
