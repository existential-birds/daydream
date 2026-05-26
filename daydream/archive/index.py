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
    append_label_observation: Append a row to the immutable bitemporal
        label_observations history (``observed_at`` transaction time,
        ``valid_at`` valid time, plus reward columns) and refresh the
        denormalized runs cache.
    latest_label_observation: Return the most recent label_observations row for
        a session, optionally constrained by an ``as_of`` cutoff timestamp.
    bulk_latest_label_observations: Return the most recent label_observations
        row for each session in a collection — single round-trip alternative to
        calling ``latest_label_observation`` in a loop.
    label_observation_history: Return the full label_observations history for
        a session in chronological order.
    label_count_summary: Return label counts for all runs in a single aggregate
        query (replaces N+1 per-session lookups).
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

from daydream.archive.manifest import Manifest

SCHEMA_VERSION = 3

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
    composite_reward REAL,
    archive_path TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1
)
"""

# Append-only bitemporal annotation history. ``observed_at`` is transaction
# time (when the annotation was recorded); ``valid_at`` is valid time (when the
# outcome the annotation describes became true, e.g. a PR merge timestamp). The
# reward columns (``reward_version``, ``reward_json``, ``composite_reward``)
# carry the full ``RewardBreakdown`` plus its cached composite scalar so a
# corpus re-projection has every axis and each annotation generation is
# self-describing (the ``runs.composite_reward`` mirror remains the SQL-threshold
# cache). See spec ``corpus-pipeline-architecture`` (silver layer).
_CREATE_LABEL_OBSERVATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS label_observations (
    session_id       TEXT NOT NULL,
    observed_at      TEXT NOT NULL,
    labels           TEXT NOT NULL,
    pr_state         TEXT,
    labeler_version  TEXT NOT NULL,
    evidence_sha     TEXT,
    rubric_json      TEXT,
    valid_at         TEXT,
    reward_version   TEXT,
    reward_json      TEXT,
    composite_reward REAL,
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
    outcome_labels, labeled_at, composite_reward, archive_path, schema_version
) VALUES (
    :session_id, :archived_at, :status, :run_flow, :skill, :model, :backend,
    :review_backend, :fix_backend, :test_backend,
    :review_only, :deep, :loop, :remote_url, :repo_slug, :branch, :base_branch,
    :head_sha, :base_sha, :changed_files, :pr_number, :pr_repo, :total_cost_usd, :total_findings,
    :grounding_rate, :coverage_ratio, :cost_per_finding_usd, :wall_clock_seconds,
    :total_prompt_tokens, :total_completion_tokens, :total_cached_tokens,
    :outcome_labels, :labeled_at, :composite_reward, :archive_path, :schema_version
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
        ("composite_reward", "REAL"),
    ]
    for col, col_type in migrations:
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")  # noqa: S608 - col/col_type are module-local constants
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise


def _recreate_label_observations_if_stale(conn: sqlite3.Connection) -> None:
    """Drop and recreate ``label_observations`` if it predates the bitemporal columns.

    The bitemporal/reward columns are part of the table's primary structure, so
    rather than `ALTER TABLE ADD COLUMN` (which cannot retrofit them cleanly for
    the spec's clean-recreate guarantee), a stale table is dropped and rebuilt.
    Dev label rows are discarded (spec-sanctioned — repopulate via ``harvest``).
    The ``runs`` table is never touched. Idempotent: after a recreate the
    ``valid_at`` column exists, so subsequent calls are a no-op.

    Args:
        conn: An open connection whose ``label_observations`` table exists.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(label_observations)").fetchall()}
    if existing and "valid_at" not in existing:
        warnings.warn(
            "label_observations table predates bitemporal columns and will be dropped "
            "and recreated. Existing label rows will be lost — repopulate via `harvest`.",
            stacklevel=2,
        )
        conn.execute("DROP TABLE label_observations")
        conn.execute(_CREATE_LABEL_OBSERVATIONS_TABLE)


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
    _recreate_label_observations_if_stale(conn)
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
                "composite_reward": manifest.composite_reward,
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
    valid_at: str | None = None,
    reward_version: str | None = None,
    reward_json: str | None = None,
    composite_reward: float | None = None,
) -> None:
    """Append a row to the immutable ``label_observations`` history.

    Writes a single ``(session_id, observed_at)`` row capturing the current
    label decision plus the bitemporal valid time and reward breakdown, and in
    the same transaction refreshes the denormalized
    ``runs.outcome_labels`` / ``runs.labeled_at`` / ``runs.rubric_json`` /
    ``runs.composite_reward`` cache.

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
        valid_at: ISO 8601 valid time — when the outcome the annotation
            describes became true (e.g. a PR merge timestamp). ``None`` for
            non-PR/local runs, in which case it collapses to ``observed_at``
            so an ``as_of``-pinned corpus never spuriously drops the run (Q2).
        reward_version: Version tag of the reward reducer that produced
            ``reward_json`` (``RewardBreakdown.reward_version``); ``None`` when
            no reward was scored.
        reward_json: Full ``RewardBreakdown.to_dict()`` serialised as JSON so a
            corpus re-projection has every axis; ``None`` when unscored.
        composite_reward: The cached composite reward scalar. Persisted on the
            ``label_observations`` row (so each annotation generation is
            self-describing) and mirrored onto ``runs.composite_reward`` for
            SQL thresholding; ``None`` when uncomputable.

    Raises:
        ValueError: When ``session_id`` is not present in the ``runs`` table.
    """
    observed_at = datetime.now(timezone.utc).isoformat()
    valid_at_value = valid_at if valid_at is not None else observed_at
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
            "(session_id, observed_at, labels, pr_state, labeler_version, evidence_sha, rubric_json, "
            "valid_at, reward_version, reward_json, composite_reward) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                observed_at,
                labels_json,
                pr_state,
                labeler_version,
                evidence_sha,
                rubric_json,
                valid_at_value,
                reward_version,
                reward_json,
                composite_reward,
            ),
        )
        conn.execute(
            "UPDATE runs SET outcome_labels = ?, labeled_at = ?, rubric_json = ?, composite_reward = ? "
            "WHERE session_id = ?",
            (labels_json, observed_at, rubric_json, composite_reward, session_id),
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


def bulk_latest_label_observations(
    archive_dir: Path,
    session_ids: list[str],
    *,
    as_of: str | None = None,
) -> dict[str, dict]:
    """Return the most recent label observation for each session in *session_ids*.

    Fetches all matching rows in a single SQL query instead of one query per
    session, eliminating the N+1 pattern when building a corpus.

    When ``as_of`` is provided, only observations whose ``observed_at <= as_of``
    are considered — the same temporal constraint applied by
    :func:`latest_label_observation`.

    Args:
        archive_dir: Path to the archive root.
        session_ids: Collection of session UUIDs to look up.
        as_of: Optional ISO 8601 cutoff timestamp.

    Returns:
        Mapping of ``session_id`` → row dict for every session that has at
        least one qualifying observation.  Sessions with no observation are
        absent from the returned dict (callers should treat them as ``None``).
    """
    if not session_ids:
        return {}
    placeholders = ",".join("?" * len(session_ids))
    conn = _get_connection(archive_dir)
    try:
        if as_of is None:
            cursor = conn.execute(
                f"""
                SELECT *
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY session_id
                               ORDER BY observed_at DESC
                           ) AS _rn
                    FROM label_observations
                    WHERE session_id IN ({placeholders})
                )
                WHERE _rn = 1
                """,
                session_ids,
            )
        else:
            cursor = conn.execute(
                f"""
                SELECT *
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY session_id
                               ORDER BY observed_at DESC
                           ) AS _rn
                    FROM label_observations
                    WHERE session_id IN ({placeholders})
                      AND observed_at <= ?
                )
                WHERE _rn = 1
                """,
                [*session_ids, as_of],
            )
        return {row["session_id"]: dict(row) for row in cursor.fetchall()}
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
                except (json.JSONDecodeError, TypeError) as exc:
                    warnings.warn(
                        f"Invalid labels payload {labels_raw!r}: {exc}",
                        stacklevel=2,
                    )
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
