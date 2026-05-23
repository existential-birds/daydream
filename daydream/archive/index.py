"""SQLite index for cross-project querying of archived daydream runs.

Manages a SQLite database at ``~/.daydream/archive/index.db`` that indexes
all archived runs by their manifest metadata. The schema is created
idempotently on every connection open, so the database is self-bootstrapping.

Exports:
    SCHEMA_VERSION: Current schema version integer.
    upsert_run: Insert or replace a run from a Manifest.
    update_labels: Update outcome labels for a session (supports prefix matching).
    query_runs: Query runs with optional WHERE clause.
    count_runs: Count rows matching an optional WHERE clause.
    append_label_observation: Append a row to the immutable label_observations
        history and refresh the denormalized runs cache.
    latest_label_observation: Return the most recent label_observations row for
        a session, optionally constrained by an ``as_of`` cutoff timestamp.
    label_observation_history: Return the full label_observations history for
        a session in chronological order.
    label_count_summary: Return label counts for all runs in a single aggregate
        query (replaces N+1 per-session lookups).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from daydream.archive.manifest import Manifest

SCHEMA_VERSION = 2

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
    base_sha TEXT,
    changed_files TEXT,
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
    rubric_json TEXT,
    archive_path TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1
)
"""

_CREATE_LABEL_OBSERVATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS label_observations (
    session_id      TEXT NOT NULL,
    observed_at     TEXT NOT NULL,
    labels          TEXT NOT NULL,
    pr_state        TEXT,
    labeler_version TEXT NOT NULL,
    evidence_sha    TEXT,
    rubric_json     TEXT,
    PRIMARY KEY (session_id, observed_at)
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_repo_slug ON runs(repo_slug)",
    "CREATE INDEX IF NOT EXISTS idx_runs_archived_at ON runs(archived_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_outcome ON runs(outcome_labels)",
    "CREATE INDEX IF NOT EXISTS idx_label_obs_observed_at ON label_observations(observed_at)",
    "CREATE INDEX IF NOT EXISTS idx_label_obs_session ON label_observations(session_id)",
]

_UPSERT_SQL = """
INSERT OR REPLACE INTO runs (
    session_id, archived_at, status, run_flow, skill, model, backend,
    review_backend, fix_backend, test_backend,
    review_only, deep, loop, remote_url, repo_slug, branch, base_branch,
    head_sha, base_sha, changed_files, pr_number, pr_repo, total_cost_usd, total_findings,
    grounding_rate, coverage_ratio, cost_per_finding_usd, wall_clock_seconds,
    total_prompt_tokens, total_completion_tokens, total_cached_tokens,
    outcome_labels, labeled_at, archive_path, schema_version
) VALUES (
    :session_id, :archived_at, :status, :run_flow, :skill, :model, :backend,
    :review_backend, :fix_backend, :test_backend,
    :review_only, :deep, :loop, :remote_url, :repo_slug, :branch, :base_branch,
    :head_sha, :base_sha, :changed_files, :pr_number, :pr_repo, :total_cost_usd, :total_findings,
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
        ("rubric_json", "TEXT"),
        ("base_sha", "TEXT"),
        ("changed_files", "TEXT"),
    ]
    for col, col_type in migrations:
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")  # noqa: S608 - col/col_type are module-local constants
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
    conn.execute(_CREATE_LABEL_OBSERVATIONS_TABLE)
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
                "base_sha": manifest.base_sha,
                "changed_files": json.dumps(manifest.changed_files),
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


def append_label_observation(
    archive_dir: Path,
    session_id: str,
    *,
    labels: list[str],
    pr_state: str | None,
    labeler_version: str,
    evidence_sha: str | None,
    rubric_json: str | None = None,
) -> None:
    """Append a row to the immutable ``label_observations`` history.

    Writes a single ``(session_id, observed_at)`` row capturing the current
    label decision and, in the same transaction, refreshes the denormalized
    ``runs.outcome_labels`` / ``runs.labeled_at`` / ``runs.rubric_json`` cache.

    Args:
        archive_dir: Path to the archive root.
        session_id: Full session UUID — must already exist in ``runs``.
        labels: List of label strings; serialised as a JSON array.
        pr_state: One of ``open``/``merged``/``closed``/``reverted`` or
            ``None`` when not applicable (e.g. local-branch runs).
        labeler_version: Free-form version tag of the labeler that produced
            this observation (e.g. ``2026.05.22`` or ``legacy``).
        evidence_sha: Optional commit SHA / artifact hash that grounds the
            decision; ``None`` when no concrete evidence applies.
        rubric_json: Optional JSON-serialised rubric (``Rubric.to_dict()``).

    Raises:
        ValueError: When ``session_id`` is not present in the ``runs`` table.
    """
    observed_at = datetime.now(timezone.utc).isoformat()
    labels_json = json.dumps(labels)
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            "SELECT session_id FROM runs WHERE session_id = ?",
            (session_id,),
        )
        if cursor.fetchone() is None:
            msg = f"Unknown session {session_id!r}"
            raise ValueError(msg)
        conn.execute(
            "INSERT INTO label_observations "
            "(session_id, observed_at, labels, pr_state, labeler_version, evidence_sha, rubric_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                observed_at,
                labels_json,
                pr_state,
                labeler_version,
                evidence_sha,
                rubric_json,
            ),
        )
        conn.execute(
            "UPDATE runs SET outcome_labels = ?, labeled_at = ?, rubric_json = ? WHERE session_id = ?",
            (labels_json, observed_at, rubric_json, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def latest_label_observation(
    archive_dir: Path,
    session_id: str,
    *,
    as_of: str | None = None,
) -> dict | None:
    """Return the most recent label observation for ``session_id``.

    When ``as_of`` is provided, the result is the most recent observation
    whose ``observed_at <= as_of`` — enabling reproducible corpus pinning.

    Args:
        archive_dir: Path to the archive root.
        session_id: Full session UUID.
        as_of: Optional ISO 8601 cutoff timestamp.

    Returns:
        The row as a dict, or ``None`` when no matching observation exists.
    """
    conn = _get_connection(archive_dir)
    try:
        if as_of is None:
            cursor = conn.execute(
                "SELECT * FROM label_observations WHERE session_id = ? "
                "ORDER BY observed_at DESC LIMIT 1",
                (session_id,),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM label_observations WHERE session_id = ? AND observed_at <= ? "
                "ORDER BY observed_at DESC LIMIT 1",
                (session_id, as_of),
            )
        row = cursor.fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def label_observation_history(archive_dir: Path, session_id: str) -> list[dict]:
    """Return the full label history for ``session_id`` in chronological order.

    Args:
        archive_dir: Path to the archive root.
        session_id: Full session UUID.

    Returns:
        List of row dicts ordered by ``observed_at`` ascending.
    """
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            "SELECT * FROM label_observations WHERE session_id = ? ORDER BY observed_at ASC",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def update_labels(archive_dir: Path, session_id: str, labels: list[str]) -> bool:
    """Update outcome labels for a session, supporting prefix matching.

    Backwards-compatible thin wrapper around :func:`append_label_observation`.
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
    finally:
        conn.close()

    if not matches:
        return False

    if len(matches) > 1:
        matched_ids = [row["session_id"] for row in matches]
        msg = f"Prefix '{session_id}' matches {len(matches)} sessions: {matched_ids}. Provide a longer prefix."
        raise ValueError(msg)

    full_id = matches[0]["session_id"]
    append_label_observation(
        archive_dir,
        full_id,
        labels=labels,
        pr_state=None,
        labeler_version="legacy",
        evidence_sha=None,
    )
    return True


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
            sql += f" WHERE {where}"  # noqa: S608 - caller-supplied SQL fragment with bound params
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def label_count_summary(
    archive_dir: Path,
    as_of: str | None = None,
) -> dict[str, int]:
    """Return label counts for all runs in a single aggregate query.

    For each run in ``runs``, finds the most recent ``label_observations`` row
    whose ``observed_at <= as_of`` (or the most recent overall when ``as_of``
    is ``None``), extracts the first label, and tallies counts.  Runs with no
    qualifying observation are counted under ``"unlabeled"``.

    This replaces the N+1 pattern of calling
    :func:`latest_label_observation` once per run.

    Args:
        archive_dir: Path to the archive root.
        as_of: Optional ISO 8601 cutoff timestamp. When ``None``, the
            most recent observation for each session is used regardless of
            ``observed_at``.

    Returns:
        Dict mapping label string → count.  Always includes at least one key
        when the archive is non-empty.
    """
    conn = _get_connection(archive_dir)
    try:
        if as_of is None:
            best_sql = (
                "SELECT session_id, MAX(observed_at) AS max_at "
                "FROM label_observations "
                "GROUP BY session_id"
            )
            params: tuple = ()
        else:
            best_sql = (
                "SELECT session_id, MAX(observed_at) AS max_at "
                "FROM label_observations "
                "WHERE observed_at <= ? "
                "GROUP BY session_id"
            )
            params = (as_of,)

        cursor = conn.execute(
            f"SELECT lo.labels "  # noqa: S608
            f"FROM runs r "
            f"LEFT JOIN ({best_sql}) best ON r.session_id = best.session_id "
            f"LEFT JOIN label_observations lo "
            f"    ON lo.session_id = best.session_id AND lo.observed_at = best.max_at",
            params,
        )
        counts: dict[str, int] = {}
        for (labels_raw,) in cursor.fetchall():
            label = "unlabeled"
            if labels_raw:
                try:
                    parsed = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
                    if isinstance(parsed, list) and parsed and parsed[0]:
                        label = str(parsed[0])
                except (json.JSONDecodeError, TypeError):
                    pass
            counts[label] = counts.get(label, 0) + 1
        return counts
    finally:
        conn.close()


def count_runs(archive_dir: Path, where: str = "", params: tuple = ()) -> int:
    """Return the number of runs matching an optional WHERE clause.

    Uses ``SELECT COUNT(*)`` so no rows are materialised.

    Args:
        archive_dir: Path to the archive root.
        where: Optional SQL WHERE clause (without the ``WHERE`` keyword).
        params: Parameter tuple to bind to the WHERE clause placeholders.

    Returns:
        Integer count of matching rows.
    """
    conn = _get_connection(archive_dir)
    try:
        sql = "SELECT COUNT(*) FROM runs"
        if where:
            sql += f" WHERE {where}"  # noqa: S608 - caller-supplied SQL fragment with bound params
        cursor = conn.execute(sql, params)
        return cursor.fetchone()[0]
    finally:
        conn.close()
