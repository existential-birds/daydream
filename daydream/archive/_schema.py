"""Archive index DDL and idempotent migrations, owned by daydream.archive.index."""

from __future__ import annotations

import sqlite3
import warnings
from collections.abc import Iterable, Sequence
from typing import NamedTuple

SCHEMA_VERSION = 8

_PRECEDENCE_ORDER = "CASE WHEN source = 'human' THEN 1 ELSE 0 END DESC, observed_at DESC"
"""SQL ORDER BY expression that ranks label_observations by human-first precedence then recency.

Used identically across append_label_observation and latest_label_observation —
centralised here so all callers stay in sync if the precedence rule ever changes.
"""

class Column(NamedTuple):
    """SQL column declaration with additive-migration and run-upsert membership flags."""

    name: str
    definition: str
    additive: bool = False
    upserted: bool = False


class RunColumn(Column):
    """A runs column in canonical fresh-database order."""


RUNS_COLUMNS: tuple[RunColumn, ...] = (
    RunColumn("session_id", "TEXT PRIMARY KEY", False, True),
    RunColumn("archived_at", "TEXT NOT NULL", False, True),
    RunColumn("status", "TEXT NOT NULL DEFAULT 'complete'", False, True),
    RunColumn("archive_status", "TEXT NOT NULL DEFAULT 'complete'", True, True),
    RunColumn("pipeline_status", "TEXT NOT NULL DEFAULT 'unknown'", True, True),
    RunColumn("phase_states", "TEXT", True, True),
    RunColumn("daydream_version", "TEXT", True, True),
    RunColumn("daydream_install_source", "TEXT", True, True),
    RunColumn("daydream_commit", "TEXT", True, True),
    RunColumn("daydream_dirty", "INTEGER", True, True),
    RunColumn("daydream_container_digest", "TEXT", True, True),
    RunColumn("run_flow", "TEXT NOT NULL", False, True),
    RunColumn("skill", "TEXT", False, True),
    RunColumn("model", "TEXT", False, True),
    RunColumn("backend", "TEXT NOT NULL DEFAULT 'claude'", False, True),
    RunColumn("review_backend", "TEXT", True, True),
    RunColumn("fix_backend", "TEXT", True, True),
    RunColumn("test_backend", "TEXT", True, True),
    RunColumn("per_stack_review_backend", "TEXT", True, True),
    RunColumn("per_stack_review_model", "TEXT", True, True),
    RunColumn("review_only", "INTEGER NOT NULL DEFAULT 0", False, True),
    RunColumn("deep", "INTEGER NOT NULL DEFAULT 0", False, True),
    RunColumn("remote_url", "TEXT", False, True),
    RunColumn("repo_slug", "TEXT", False, True),
    RunColumn("source_path", "TEXT", True, True),
    RunColumn("branch", "TEXT", False, True),
    RunColumn("base_branch", "TEXT", False, True),
    RunColumn("head_sha", "TEXT", False, True),
    RunColumn("base_sha", "TEXT", True, True),
    RunColumn("changed_files", "TEXT", True, True),
    RunColumn("pr_number", "INTEGER", False, True),
    RunColumn("pr_repo", "TEXT", False, True),
    RunColumn("total_cost_usd", "REAL", False, True),
    RunColumn("total_findings", "INTEGER", False, True),
    RunColumn("cost_per_finding_usd", "REAL", False, True),
    RunColumn("wall_clock_seconds", "REAL", False, True),
    RunColumn("erosion", "REAL", True, True),
    RunColumn("verbosity", "REAL", True, True),
    RunColumn("location_in_hunk_rate", "REAL", True, True),
    RunColumn("shipped_duplicate_pairs", "INTEGER", True, True),
    RunColumn("fix_quality_gate", "TEXT", True, True),
    RunColumn("recommended_patch_capture", "TEXT", True, True),
    RunColumn("total_prompt_tokens", "INTEGER", False, True),
    RunColumn("total_completion_tokens", "INTEGER", False, True),
    RunColumn("total_cached_tokens", "INTEGER", False, True),
    RunColumn("outcome_labels", "TEXT NOT NULL DEFAULT '[]'", False, True),
    RunColumn("labeled_at", "TEXT", False, True),
    RunColumn("rubric_json", "TEXT", True, False),
    RunColumn("composite_reward", "REAL", True, True),
    RunColumn("has_posterior", "INTEGER NOT NULL DEFAULT 0", True, False),
    RunColumn("archive_path", "TEXT NOT NULL", False, True),
    RunColumn("schema_version", "INTEGER NOT NULL DEFAULT 1", False, True),
    RunColumn("profile_schema_version", "INTEGER", True, True),
    RunColumn("profile_name", "TEXT", True, True),
    RunColumn("profile_source_kind", "TEXT", True, True),
    RunColumn("profile_digest", "TEXT", True, True),
)


def _create_table_sql(
    columns: Iterable[Column],
    table: str = "runs",
    table_constraints: Sequence[str] = (),
) -> str:
    """Render verbatim column definitions followed by table constraints."""
    lines = [f"    {col.name} {col.definition}" for col in columns]
    lines += [f"    {constraint}" for constraint in table_constraints]
    return f"\nCREATE TABLE IF NOT EXISTS {table} (\n" + ",\n".join(lines) + "\n)\n"


def _upsert_sql(columns: Iterable[RunColumn]) -> str:
    """Generate matching column and parameter lists for declared upsert columns."""
    participating = tuple(col.name for col in columns if col.upserted)
    parameters = tuple(f":{name}" for name in participating)
    return (
        "\nINSERT OR REPLACE INTO runs (\n"
        + "    " + ", ".join(participating)
        + "\n) VALUES (\n"
        + "    " + ", ".join(parameters)
        + "\n)\n"
    )


_CREATE_TABLE = _create_table_sql(RUNS_COLUMNS)

# Append-only history: observed_at is transaction time; valid_at is outcome time.
# Reward JSON retains every axis; runs.composite_reward caches the winning score.
# reviewer_logins names posterior contributors; has_posterior separates populations.
# Human observations outrank auto regardless of recency. Auto dedup binds evidence,
# labeler/reply policy, reply digest, labels, and population membership.
# Additive migrations default old source to auto and mark missing-policy rows legacy
# once, without rewriting their labels, timestamps, or rubric.
LABEL_OBSERVATION_COLUMNS: tuple[Column, ...] = (
    Column("session_id", "TEXT NOT NULL"),
    Column("observed_at", "TEXT NOT NULL"),
    Column("labels", "TEXT NOT NULL"),
    Column("pr_state", "TEXT"),
    Column("labeler_version", "TEXT NOT NULL"),
    Column("evidence_sha", "TEXT"),
    Column("rubric_json", "TEXT"),
    Column("valid_at", "TEXT"),
    Column("reward_version", "TEXT"),
    Column("reward_json", "TEXT"),
    Column("composite_reward", "REAL"),
    Column("reviewer_logins", "TEXT"),
    Column("has_posterior", "INTEGER NOT NULL DEFAULT 0"),
    Column("source", "TEXT NOT NULL DEFAULT 'auto'", True),
    Column("labeler_policy_version", "TEXT", True),
    Column("reply_classifier_version", "TEXT", True),
    Column("reply_evidence_digest", "TEXT", True),
    Column("legacy", "TEXT NOT NULL DEFAULT 'auto'", True),
)
LABEL_OBSERVATION_NAMES: tuple[str, ...] = tuple(col.name for col in LABEL_OBSERVATION_COLUMNS)

_CREATE_LABEL_OBSERVATIONS_TABLE = _create_table_sql(
    LABEL_OBSERVATION_COLUMNS,
    table="label_observations",
    table_constraints=("PRIMARY KEY (session_id, observed_at)",),
)

# Both observation append and metadata re-admission must expose the same winner.
_REFRESH_RUN_LABELS_SQL = f"""
UPDATE runs SET (outcome_labels, labeled_at, rubric_json, composite_reward, has_posterior) = (
    SELECT labels, observed_at, rubric_json, composite_reward, has_posterior
    FROM label_observations WHERE session_id = runs.session_id
    ORDER BY {_PRECEDENCE_ORDER} LIMIT 1
) WHERE EXISTS (SELECT 1 FROM label_observations WHERE session_id = runs.session_id)
"""


def _install_label_projection(conn: sqlite3.Connection) -> None:
    """Own human-first label projection on append and current-run re-admission."""
    installed = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'project_runs_labels'"
    ).fetchone()
    for table in ("runs", "label_observations"):
        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS project_{table}_labels AFTER INSERT ON {table}
            BEGIN {_REFRESH_RUN_LABELS_SQL} AND session_id = NEW.session_id; END
        """)
    if installed is None:
        conn.execute(_REFRESH_RUN_LABELS_SQL)


_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_repo_slug ON runs(repo_slug)",
    "CREATE INDEX IF NOT EXISTS idx_runs_archived_at ON runs(archived_at)",
    "CREATE INDEX IF NOT EXISTS idx_runs_outcome ON runs(outcome_labels)",
    "CREATE INDEX IF NOT EXISTS idx_label_obs_observed_at ON label_observations(observed_at)",
    "CREATE INDEX IF NOT EXISTS idx_label_obs_session ON label_observations(session_id)",
]

_UPSERT_SQL = _upsert_sql(RUNS_COLUMNS)


def _alter_add_missing(
    conn: sqlite3.Connection,
    table: str,
    migrations: list[tuple[str, str]],
) -> None:
    """Add missing columns; tolerate concurrent openers racing to add the same column."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}  # noqa: S608 - table is a module-local constant at every call site
    for col, col_type in migrations:
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")  # noqa: S608 - col/col_type are module-local constants
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise


def _migration_entries(columns: Iterable[Column]) -> list[tuple[str, str]]:
    """Select additive columns for ALTER ADD in declaration order."""
    return [(col.name, col.definition) for col in columns if col.additive]


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns that exist in _CREATE_TABLE but are missing from the live DB."""
    _alter_add_missing(conn, "runs", _migration_entries(RUNS_COLUMNS))


def _recreate_label_observations_if_stale(conn: sqlite3.Connection) -> None:
    """Recreate observations missing structural valid_at or has_posterior columns."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(label_observations)").fetchall()}
    if existing and ("valid_at" not in existing or "has_posterior" not in existing):
        warnings.warn(
            "label_observations table predates bitemporal/posterior columns and will be dropped "
            "and recreated. Existing label rows will be lost — repopulate via `harvest`.",
            stacklevel=2,
        )
        conn.execute("DROP TABLE label_observations")
        conn.execute(_CREATE_LABEL_OBSERVATIONS_TABLE)


def _migrate_label_observations_schema(conn: sqlite3.Connection) -> None:
    """Add missing observation columns without deleting rows; legacy source defaults to auto."""
    _alter_add_missing(
        conn,
        "label_observations",
        _migration_entries(LABEL_OBSERVATION_COLUMNS),
    )
    # Only missing-policy rows still marked auto become legacy; this guard prevents
    # repeat WAL writes and preserves labels, observed_at, and rubric_json.
    conn.execute(
        "UPDATE label_observations SET legacy = 'legacy' "
        "WHERE labeler_policy_version IS NULL AND legacy = 'auto'"
    )
