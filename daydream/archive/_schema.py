"""SQLite schema definitions and migration helpers for the daydream archive index.

Centralises all DDL constants (CREATE TABLE, CREATE INDEX, UPSERT) and the
idempotent migration helpers that bring a live database up to the current
schema version. Imported exclusively by ``daydream.archive.index``; callers
outside that module should not depend on anything in this file directly.
"""

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

_REVIEWER_PENALTY_MAP: dict[str, float] = {
    "accepted": 0.0,
    "contested": 0.5,
    "rejected": 1.0,
}
"""Maintainer outcome label → false-positive penalty, mirroring
``daydream.training.reward._FP_PENALTY_MAP``.  Defined here so the archive
layer does not depend on the training layer."""

class Column(NamedTuple):
    """One table column: its SQL name, DDL body text, and run-migration flags.

    ``additive``/``upserted`` default to ``False`` so non-``runs`` tables (e.g.
    ``label_observations``) declare only a name and definition.
    """

    name: str
    definition: str
    additive: bool = False
    upserted: bool = False


class RunColumn(Column):
    """One column of the ``runs`` table.

    ``RUNS_COLUMNS`` is the *only* declaration of the runs column set — the
    ``CREATE TABLE`` text (``_CREATE_TABLE``), the ``ALTER TABLE ADD COLUMN``
    entries ``_migrate_schema`` applies, and the run-upsert statement
    (``_UPSERT_SQL``) are all generated from it. Its order is the canonical
    fresh-database column order; upgraded databases are migrated *by name*
    (see ``_alter_add_missing``), so their resulting column order is not a
    contract, and every runs read is ``SELECT *`` materialised by column name.

    ``additive`` marks a column that appended databases migrate onto;
    ``upserted`` marks a column the run upsert writes (the label-observation
    paths own the rest and maintain them separately).
    """


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
    RunColumn("grounding_rate", "REAL", False, True),
    RunColumn("coverage_ratio", "REAL", False, True),
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
    """Render a CREATE TABLE statement from *columns* and *table_constraints*.

    Every line carries a trailing comma except the last, matching the canonical
    fresh-database ``runs`` DDL byte-for-byte. The definition text is emitted
    verbatim. ``table_constraints`` are appended after the columns.
    """
    lines = [f"    {col.name} {col.definition}" for col in columns]
    lines += [f"    {constraint}" for constraint in table_constraints]
    return f"\nCREATE TABLE IF NOT EXISTS {table} (\n" + ",\n".join(lines) + "\n)\n"


_UPSERT_LINE_WIDTH = 92
"""Maximum length of a generated upsert column/parameter line (4-space indent included)."""


def _wrap_tokens(tokens: tuple[str, ...]) -> str:
    """Greedily wrap *tokens* into 4-space-indented comma-separated lines.

    The single formatting rule shared by both halves of the run upsert: fill a
    line until the next token would push it past ``_UPSERT_LINE_WIDTH``, then
    break *before* that token. Every line ends with a comma except the last.
    """
    lines: list[str] = []
    current = "    "
    for token in tokens:
        separator = ", " if current.strip() else ""
        candidate = current + separator + token
        if current.strip() and len(candidate) > _UPSERT_LINE_WIDTH:
            lines.append(current + ",")
            current = "    " + token
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def _upsert_sql(columns: Iterable[RunColumn]) -> str:
    """Render the run upsert statement from *columns*.

    Only the columns declared ``upserted`` participate, in declaration order;
    the column block and the ``:name`` parameter block are wrapped by the same
    ``_wrap_tokens`` rule so their token sequences stay aligned. No trailing
    comma on either block's last token.
    """
    participating = tuple(col.name for col in columns if col.upserted)
    parameters = tuple(f":{name}" for name in participating)
    return (
        "\nINSERT OR REPLACE INTO runs (\n"
        + _wrap_tokens(participating)
        + "\n) VALUES (\n"
        + _wrap_tokens(parameters)
        + "\n)\n"
    )


_CREATE_TABLE = _create_table_sql(RUNS_COLUMNS)

# Append-only bitemporal annotation history. ``observed_at`` is transaction
# time (when the annotation was recorded); ``valid_at`` is valid time (when the
# outcome the annotation describes became true, e.g. a PR merge timestamp). The
# reward columns (``reward_version``, ``reward_json``, ``composite_reward``)
# carry the full ``RewardBreakdown`` plus its cached composite scalar so a
# corpus re-projection has every axis and each annotation generation is
# self-describing (the ``runs.composite_reward`` mirror remains the SQL-threshold
# cache). ``reviewer_logins`` is a JSON array of the human GitHub accounts whose
# review/reply outcomes seeded the posterior axis (empty/``None`` for non-PR
# runs); ``has_posterior`` is the population discriminator (1 when the row
# carries a ``PosteriorBreakdown``, mirrored onto ``runs`` so SQL consumers can
# split labeled/unlabeled populations without parsing ``reward_json``). See spec
# ``corpus-pipeline-architecture`` (silver layer) and ``reward-posterior-corrections`` (C3).
# ``source`` is the precedence marker (``'auto'`` for automated rubric labels,
# ``'human'`` for maintainer overrides) — human-sourced rows win in the "latest
# label" projections regardless of recency. Pre-existing rows default to ``'auto'``
# via the additive ``_migrate_label_observations_schema`` ALTER-ADD migration.
# ``labeler_policy_version`` mirrors ``labeler_version`` so the policy axis is an
# explicit column; ``reply_classifier_version`` / ``reply_evidence_digest`` carry
# the reply-classifier version and a stable digest over the combined reply
# evidence — together with ``evidence_sha`` they form the auto-dedup key
# ``(evidence_sha, labeler_policy_version, reply_evidence_digest, labels,
# has_posterior)``, replacing the older ``(evidence_sha, reward_version)`` key.
# ``legacy`` marks provenance generation: new rows default ``'auto'``
# (current-generation); the additive migration stamps every pre-existing row
# (``labeler_policy_version IS NULL``) ``'legacy'`` exactly once, never touching
# its labels/observed_at/rubric_json.
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
    Column("source", "TEXT NOT NULL DEFAULT 'auto'"),
    Column("labeler_policy_version", "TEXT"),
    Column("reply_classifier_version", "TEXT"),
    Column("reply_evidence_digest", "TEXT"),
    Column("legacy", "TEXT NOT NULL DEFAULT 'auto'"),
)
LABEL_OBSERVATION_NAMES: tuple[str, ...] = tuple(col.name for col in LABEL_OBSERVATION_COLUMNS)

_CREATE_LABEL_OBSERVATIONS_TABLE = _create_table_sql(
    LABEL_OBSERVATION_COLUMNS,
    table="label_observations",
    table_constraints=("PRIMARY KEY (session_id, observed_at)",),
)

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
    """ALTER TABLE *table* to add any columns in *migrations* that are absent.

    Each entry in *migrations* is a ``(column_name, column_type)`` pair.
    Missing columns are added idempotently; a ``duplicate column name`` error
    (raised by a race between concurrent openers) is silently swallowed, which
    preserves the warn-and-continue semantics of the callers it replaces.
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}  # noqa: S608 - table is a module-local constant at every call site
    for col, col_type in migrations:
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")  # noqa: S608 - col/col_type are module-local constants
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise


def _migration_entries(columns: Iterable[RunColumn]) -> list[tuple[str, str]]:
    """Render the ``(name, type)`` ALTER-ADD entries for the additive *columns*.

    Non-additive entries are omitted: they are present in every legacy shape by
    construction (``archived_at``, ``run_flow`` and ``archive_path`` are
    ``NOT NULL`` without defaults and cannot be added to a populated table). The
    emitted order follows the declaration; ``_alter_add_missing`` filters by
    live ``PRAGMA table_info``, so order is not a contract.
    """
    return [(col.name, col.definition) for col in columns if col.additive]


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns that exist in _CREATE_TABLE but are missing from the live DB."""
    _alter_add_missing(conn, "runs", _migration_entries(RUNS_COLUMNS))


def _recreate_label_observations_if_stale(conn: sqlite3.Connection) -> None:
    """Drop and recreate ``label_observations`` if it predates the bitemporal/posterior columns.

    The bitemporal/reward/posterior columns are part of the table's primary
    structure, so rather than `ALTER TABLE ADD COLUMN` (which cannot retrofit
    them cleanly for the spec's clean-recreate guarantee), a stale table is
    dropped and rebuilt. A table missing either ``valid_at`` (pre-bitemporal) or
    ``has_posterior`` (pre-reward-posterior-corrections) is considered stale. Dev
    label rows are discarded (spec-sanctioned — repopulate via ``harvest``). The
    ``runs`` table is never touched. Idempotent: after a recreate both columns
    exist, so subsequent calls are a no-op.
    """
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
    """Additively ALTER-ADD columns missing from a live ``label_observations`` table.

    Unlike ``_recreate_label_observations_if_stale`` (which drop-recreates for
    structural bitemporal/posterior columns), this is non-destructive: the
    ``source`` precedence marker is additive, so pre-existing rows are preserved
    and default to ``'auto'``. Delegates to ``_alter_add_missing`` (the shared
    helper that also backs ``_migrate_schema`` for the runs table).
    """
    _alter_add_missing(
        conn,
        "label_observations",
        [
            ("source", "TEXT NOT NULL DEFAULT 'auto'"),
            ("labeler_policy_version", "TEXT"),
            ("reply_classifier_version", "TEXT"),
            ("reply_evidence_digest", "TEXT"),
            ("legacy", "TEXT NOT NULL DEFAULT 'auto'"),
        ],
    )
    # Stamp history exactly once (M17): rows written before the reply-label
    # columns existed have ``labeler_policy_version IS NULL``. The additional
    # ``legacy = 'auto'`` condition is the actual idempotency guard — once a
    # row is marked ``legacy`` it stops matching, so later connection opens do
    # not re-execute the UPDATE (or rewrite matched rows into the WAL) even
    # though ``labeler_policy_version`` stays NULL. No row's labels/observed_at/
    # rubric_json is ever written here.
    conn.execute(
        "UPDATE label_observations SET legacy = 'legacy' "
        "WHERE labeler_policy_version IS NULL AND legacy = 'auto'"
    )
