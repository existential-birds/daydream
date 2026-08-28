"""SQLite schema definitions and migration helpers for the daydream archive index.

Centralises all DDL constants (CREATE TABLE, CREATE INDEX, UPSERT) and the
idempotent migration helpers that bring a live database up to the current
schema version. Imported exclusively by ``daydream.archive.index``; callers
outside that module should not depend on anything in this file directly.
"""

from __future__ import annotations

import sqlite3
import warnings

SCHEMA_VERSION = 7

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

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    session_id TEXT PRIMARY KEY,
    archived_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'complete',
    archive_status TEXT NOT NULL DEFAULT 'complete',
    pipeline_status TEXT NOT NULL DEFAULT 'unknown',
    phase_states TEXT,
    daydream_version TEXT,
    daydream_install_source TEXT,
    daydream_commit TEXT,
    daydream_dirty INTEGER,
    daydream_container_digest TEXT,
    run_flow TEXT NOT NULL,
    skill TEXT,
    model TEXT,
    backend TEXT NOT NULL DEFAULT 'claude',
    review_backend TEXT,
    fix_backend TEXT,
    test_backend TEXT,
    per_stack_review_backend TEXT,
    per_stack_review_model TEXT,
    review_only INTEGER NOT NULL DEFAULT 0,
    deep INTEGER NOT NULL DEFAULT 0,
    remote_url TEXT,
    repo_slug TEXT,
    source_path TEXT,
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
    erosion REAL,
    verbosity REAL,
    fix_quality_gate TEXT,
    recommended_patch_capture TEXT,
    total_prompt_tokens INTEGER,
    total_completion_tokens INTEGER,
    total_cached_tokens INTEGER,
    outcome_labels TEXT NOT NULL DEFAULT '[]',
    labeled_at TEXT,
    rubric_json TEXT,
    composite_reward REAL,
    has_posterior INTEGER NOT NULL DEFAULT 0,
    archive_path TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    profile_schema_version INTEGER,
    profile_name TEXT,
    profile_source_kind TEXT,
    profile_digest TEXT
)
"""

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
    erosion, verbosity, fix_quality_gate, recommended_patch_capture,
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
    :erosion, :verbosity, :fix_quality_gate, :recommended_patch_capture,
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
    # columns existed have ``labeler_policy_version IS NULL`` — the condition is
    # its own idempotency guard, so pre-existing rows are marked ``legacy`` on
    # first open and never rewritten afterwards. No row's labels/observed_at/
    # rubric_json is ever written here.
    conn.execute("UPDATE label_observations SET legacy = 'legacy' WHERE labeler_policy_version IS NULL")
