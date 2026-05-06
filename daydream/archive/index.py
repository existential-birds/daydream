"""SQLite index for cross-project querying of archived daydream runs.

Manages a SQLite database at ``~/.daydream/archive/index.db`` that indexes
all archived runs by their manifest metadata. The schema is created
idempotently on every connection open, so the database is self-bootstrapping.

Exports:
    SCHEMA_VERSION: Current schema version integer.
    upsert_run: Insert or replace a run from a Manifest.
    update_labels: Update outcome labels for a session (supports prefix matching).
    query_runs: Query runs with optional WHERE clause.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from daydream.archive.manifest import Manifest

SCHEMA_VERSION = 1

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    session_id TEXT PRIMARY KEY,
    archived_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'complete',
    run_flow TEXT NOT NULL,
    skill TEXT,
    model TEXT,
    backend TEXT NOT NULL DEFAULT 'claude',
    review_backend TEXT,
    fix_backend TEXT,
    test_backend TEXT,
    review_only INTEGER NOT NULL DEFAULT 0,
    deep INTEGER NOT NULL DEFAULT 0,
    loop INTEGER NOT NULL DEFAULT 0,
    remote_url TEXT,
    repo_slug TEXT,
    branch TEXT,
    base_branch TEXT,
    head_sha TEXT,
    pr_number INTEGER,
    pr_repo TEXT,
    total_cost_usd REAL,
    total_findings INTEGER,
    grounding_rate REAL,
    coverage_ratio REAL,
    cost_per_finding_usd REAL,
    wall_clock_seconds REAL,
    total_prompt_tokens INTEGER,
    total_completion_tokens INTEGER,
    total_cached_tokens INTEGER,
    outcome_labels TEXT NOT NULL DEFAULT '[]',
    labeled_at TEXT,
    archive_path TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_repo_slug ON runs(repo_slug)",
    "CREATE INDEX IF NOT EXISTS idx_runs_archived_at ON runs(archived_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_outcome ON runs(outcome_labels)",
]

_UPSERT_SQL = """
INSERT OR REPLACE INTO runs (
    session_id, archived_at, status, run_flow, skill, model, backend,
    review_backend, fix_backend, test_backend,
    review_only, deep, loop, remote_url, repo_slug, branch, base_branch,
    head_sha, pr_number, pr_repo, total_cost_usd, total_findings,
    grounding_rate, coverage_ratio, cost_per_finding_usd, wall_clock_seconds,
    total_prompt_tokens, total_completion_tokens, total_cached_tokens,
    outcome_labels, labeled_at, archive_path, schema_version
) VALUES (
    :session_id, :archived_at, :status, :run_flow, :skill, :model, :backend,
    :review_backend, :fix_backend, :test_backend,
    :review_only, :deep, :loop, :remote_url, :repo_slug, :branch, :base_branch,
    :head_sha, :pr_number, :pr_repo, :total_cost_usd, :total_findings,
    :grounding_rate, :coverage_ratio, :cost_per_finding_usd, :wall_clock_seconds,
    :total_prompt_tokens, :total_completion_tokens, :total_cached_tokens,
    :outcome_labels, :labeled_at, :archive_path, :schema_version
)
"""


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns that exist in _CREATE_TABLE but are missing from the live DB."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    migrations: list[tuple[str, str]] = [
        ("review_backend", "TEXT"),
        ("fix_backend", "TEXT"),
        ("test_backend", "TEXT"),
    ]
    for col, col_type in migrations:
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise


def _get_connection(archive_dir: Path) -> sqlite3.Connection:
    """Open the index database, creating schema if needed.

    Enables WAL mode for concurrent read access and sets a busy timeout
    to handle contention from parallel daydream runs.

    Args:
        archive_dir: Path to the archive root (e.g. ``~/.daydream/archive``).

    Returns:
        An open sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    db_path = archive_dir / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(_CREATE_TABLE)
    _migrate_schema(conn)
    for idx_sql in _CREATE_INDEXES:
        conn.execute(idx_sql)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def upsert_run(archive_dir: Path, manifest: Manifest) -> None:
    """Insert or replace a run entry from a Manifest.

    Bool fields (review_only, deep, loop) are mapped to integers (0/1)
    for SQLite storage.

    Args:
        archive_dir: Path to the archive root.
        manifest: The Manifest to index.
    """
    conn = _get_connection(archive_dir)
    try:
        conn.execute(
            _UPSERT_SQL,
            {
                "session_id": manifest.session_id,
                "archived_at": manifest.archived_at,
                "status": manifest.status,
                "run_flow": manifest.run_flow,
                "skill": manifest.skill,
                "model": manifest.model,
                "backend": manifest.backend,
                "review_backend": manifest.review_backend,
                "fix_backend": manifest.fix_backend,
                "test_backend": manifest.test_backend,
                "review_only": int(manifest.review_only),
                "deep": int(manifest.deep),
                "loop": int(manifest.loop),
                "remote_url": manifest.remote_url,
                "repo_slug": manifest.repo_slug,
                "branch": manifest.branch,
                "base_branch": manifest.base_branch,
                "head_sha": manifest.head_sha,
                "pr_number": manifest.pr_number,
                "pr_repo": manifest.pr_repo,
                "total_cost_usd": manifest.total_cost_usd,
                "total_findings": manifest.total_findings,
                "grounding_rate": manifest.grounding_rate,
                "coverage_ratio": manifest.coverage_ratio,
                "cost_per_finding_usd": manifest.cost_per_finding_usd,
                "wall_clock_seconds": manifest.wall_clock_seconds,
                "total_prompt_tokens": manifest.total_prompt_tokens,
                "total_completion_tokens": manifest.total_completion_tokens,
                "total_cached_tokens": manifest.total_cached_tokens,
                "outcome_labels": manifest.outcome_labels,
                "labeled_at": manifest.labeled_at,
                "archive_path": manifest.archive_path,
                "schema_version": SCHEMA_VERSION,
            },
        )
        conn.commit()
    finally:
        conn.close()


def update_labels(archive_dir: Path, session_id: str, labels: list[str]) -> bool:
    """Update outcome labels for a session, supporting prefix matching.

    The session_id can be a prefix (e.g. first 8 chars of the UUID). If the
    prefix matches exactly one row, that row is updated. If it matches
    multiple rows, a ValueError is raised asking for a longer prefix.

    Args:
        archive_dir: Path to the archive root.
        session_id: Full or prefix session ID to match.
        labels: List of label strings to set.

    Returns:
        True if a row was updated, False if no matching session was found.

    Raises:
        ValueError: If the prefix matches more than one session.
    """
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            "SELECT session_id FROM runs WHERE session_id LIKE ? || '%'",
            (session_id,),
        )
        matches = cursor.fetchall()

        if not matches:
            return False

        if len(matches) > 1:
            matched_ids = [row["session_id"] for row in matches]
            msg = f"Prefix '{session_id}' matches {len(matches)} sessions: {matched_ids}. Provide a longer prefix."
            raise ValueError(msg)

        full_id = matches[0]["session_id"]
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE runs SET outcome_labels = ?, labeled_at = ? WHERE session_id = ?",
            (json.dumps(labels), now, full_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def query_runs(archive_dir: Path, where: str = "", params: tuple = ()) -> list[dict]:
    """Query the runs index with an optional WHERE clause.

    Args:
        archive_dir: Path to the archive root.
        where: Optional SQL WHERE clause (without the ``WHERE`` keyword).
            Example: ``"repo_slug = ? AND status = ?"``.
        params: Parameter tuple to bind to the WHERE clause placeholders.

    Returns:
        List of row dicts, one per matching run.
    """
    conn = _get_connection(archive_dir)
    try:
        sql = "SELECT * FROM runs"
        if where:
            sql += f" WHERE {where}"
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
