"""SQLite schema definitions and migration helpers for the daydream archive index.

Centralises all DDL constants (CREATE TABLE, CREATE INDEX, UPSERT) and the
idempotent migration helpers that bring a live database up to the current
schema version. Imported exclusively by ``daydream.archive.index``; callers
outside that module should not depend on anything in this file directly.
"""

from __future__ import annotations

import sqlite3
import warnings
from collections.abc import Iterable
from typing import NamedTuple

SCHEMA_VERSION = 8

_PRECEDENCE_ORDER = "CASE WHEN source = 'human' THEN 1 ELSE 0 END DESC, observed_at DESC"
"""SQL ORDER BY expression that ranks label_observations by human-first precedence then recency.

Used identically across append_label_observation, latest_label_observation,
bulk_latest_label_observations, and label_count_summary — centralised here so
all callers stay in sync if the precedence rule ever changes.
"""

_REVIEWER_PENALTY_MAP: dict[str, float] = {
    "accepted": 0.0,
    "contested": 0.5,
    "rejected": 1.0,
}
"""Maintainer outcome label → false-positive penalty, mirroring
``daydream.training.reward._FP_PENALTY_MAP``.  Defined here so the archive
layer does not depend on the training layer."""

class RunColumn(NamedTuple):
    """One column of the ``runs`` table.

    ``RUNS_COLUMNS`` is the *only* declaration of the runs column set — the
    ``CREATE TABLE`` text (``_CREATE_TABLE``), the ``ALTER TABLE ADD COLUMN``
    entries ``_migrate_schema`` applies, and the run-upsert statement
    (``_UPSERT_SQL``) are all generated from it. Its order is the canonical
    fresh-database column order; upgraded databases are migrated *by name*
    (see ``_alter_add_missing``), so their resulting column order is not a
    contract, and every runs read is ``SELECT *`` materialised by column name.

    ``definition`` is the DDL body text after the column name (including any
    constraint clauses). ``additive`` marks a column that appended databases
    migrate onto; ``upserted`` marks a column the run upsert writes (the
    label-observation paths own the rest and maintain them separately).
    """

    name: str
    definition: str
    additive: bool
    upserted: bool


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


def _create_table_sql(columns: Iterable[RunColumn]) -> str:
    """Render the ``runs`` CREATE TABLE text from *columns*.

    Every line carries a trailing comma except the last, matching the canonical
    fresh-database DDL byte-for-byte. The definition text is emitted verbatim.
    """
    columns = tuple(columns)
    lines = [
        f"    {col.name} {col.definition}" + ("," if i < len(columns) - 1 else "")
        for i, col in enumerate(columns)
    ]
    return "\nCREATE TABLE IF NOT EXISTS runs (\n" + "\n".join(lines) + "\n)\n"


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
    reviewer_logins  TEXT,
    has_posterior    INTEGER NOT NULL DEFAULT 0,
    source           TEXT NOT NULL DEFAULT 'auto',
    labeler_policy_version TEXT,
    reply_classifier_version TEXT,
    reply_evidence_digest  TEXT,
    legacy           TEXT NOT NULL DEFAULT 'auto',
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
    session_id, archived_at, status, archive_status, pipeline_status, phase_states,
    daydream_version, daydream_install_source, daydream_commit, daydream_dirty,
    daydream_container_digest, run_flow, skill, model, backend,
    review_backend, fix_backend, test_backend, per_stack_review_backend, per_stack_review_model,
    review_only, deep, remote_url, repo_slug, source_path, branch, base_branch,
    head_sha, base_sha, changed_files, pr_number, pr_repo, total_cost_usd, total_findings,
    grounding_rate, coverage_ratio, cost_per_finding_usd, wall_clock_seconds,
    erosion, verbosity, location_in_hunk_rate, shipped_duplicate_pairs,
    fix_quality_gate, recommended_patch_capture,
    total_prompt_tokens, total_completion_tokens, total_cached_tokens,
    outcome_labels, labeled_at, composite_reward, archive_path, schema_version,
    profile_schema_version, profile_name, profile_source_kind, profile_digest
) VALUES (
    :session_id, :archived_at, :status, :archive_status, :pipeline_status, :phase_states,
    :daydream_version, :daydream_install_source, :daydream_commit, :daydream_dirty,
    :daydream_container_digest, :run_flow, :skill, :model, :backend,
    :review_backend, :fix_backend, :test_backend, :per_stack_review_backend, :per_stack_review_model,
    :review_only, :deep, :remote_url, :repo_slug, :source_path, :branch, :base_branch,
    :head_sha, :base_sha, :changed_files, :pr_number, :pr_repo, :total_cost_usd, :total_findings,
    :grounding_rate, :coverage_ratio, :cost_per_finding_usd, :wall_clock_seconds,
    :erosion, :verbosity, :location_in_hunk_rate, :shipped_duplicate_pairs,
    :fix_quality_gate, :recommended_patch_capture,
    :total_prompt_tokens, :total_completion_tokens, :total_cached_tokens,
    :outcome_labels, :labeled_at, :composite_reward, :archive_path, :schema_version,
    :profile_schema_version, :profile_name, :profile_source_kind, :profile_digest
)
"""


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


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns that exist in _CREATE_TABLE but are missing from the live DB."""
    _alter_add_missing(
        conn,
        "runs",
        [
            ("review_backend", "TEXT"),
            ("fix_backend", "TEXT"),
            ("test_backend", "TEXT"),
            ("per_stack_review_backend", "TEXT"),
            ("per_stack_review_model", "TEXT"),
            ("rubric_json", "TEXT"),
            ("base_sha", "TEXT"),
            ("changed_files", "TEXT"),
            ("composite_reward", "REAL"),
            ("source_path", "TEXT"),
            ("has_posterior", "INTEGER NOT NULL DEFAULT 0"),
            ("erosion", "REAL"),
            ("verbosity", "REAL"),
            ("location_in_hunk_rate", "REAL"),
            ("shipped_duplicate_pairs", "INTEGER"),
            ("fix_quality_gate", "TEXT"),
            ("recommended_patch_capture", "TEXT"),
            ("archive_status", "TEXT NOT NULL DEFAULT 'complete'"),
            ("pipeline_status", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("phase_states", "TEXT"),
            ("daydream_version", "TEXT"),
            ("daydream_install_source", "TEXT"),
            ("daydream_commit", "TEXT"),
            ("daydream_dirty", "INTEGER"),
            ("daydream_container_digest", "TEXT"),
            ("profile_schema_version", "INTEGER"),
            ("profile_name", "TEXT"),
            ("profile_source_kind", "TEXT"),
            ("profile_digest", "TEXT"),
        ],
    )


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
