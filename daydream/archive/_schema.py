"""Current archive index DDL, owned by daydream.archive.index."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple

SCHEMA_VERSION = 8

_PRECEDENCE_ORDER = "CASE WHEN source = 'human' THEN 1 ELSE 0 END DESC, observed_at DESC"
"""SQL ORDER BY expression that ranks label_observations by human-first precedence then recency.

Used identically across append_label_observation and latest_label_observation —
centralised here so all callers stay in sync if the precedence rule ever changes.
"""

class Column(NamedTuple):
    """SQL column declaration."""

    name: str
    definition: str
    upserted: bool = False


class RunColumn(Column):
    """A runs column in canonical fresh-database order.

    The declaration generates CREATE and UPSERT SQL.
    Existing column order is unconstrained: reads use names.
    Label-observation writers own columns excluded from run upserts.
    """


RUNS_COLUMNS: tuple[RunColumn, ...] = (
    RunColumn("session_id", "TEXT PRIMARY KEY", True),
    RunColumn("archived_at", "TEXT NOT NULL", True),
    RunColumn("status", "TEXT NOT NULL DEFAULT 'complete'", True),
    RunColumn("archive_status", "TEXT NOT NULL DEFAULT 'complete'", True),
    RunColumn("pipeline_status", "TEXT NOT NULL DEFAULT 'unknown'", True),
    RunColumn("phase_states", "TEXT", True),
    RunColumn("daydream_version", "TEXT", True),
    RunColumn("daydream_install_source", "TEXT", True),
    RunColumn("daydream_commit", "TEXT", True),
    RunColumn("daydream_dirty", "INTEGER", True),
    RunColumn("daydream_container_digest", "TEXT", True),
    RunColumn("run_flow", "TEXT NOT NULL", True),
    RunColumn("skill", "TEXT", True),
    RunColumn("model", "TEXT", True),
    RunColumn("backend", "TEXT NOT NULL DEFAULT 'claude'", True),
    RunColumn("review_backend", "TEXT", True),
    RunColumn("fix_backend", "TEXT", True),
    RunColumn("test_backend", "TEXT", True),
    RunColumn("per_stack_review_backend", "TEXT", True),
    RunColumn("per_stack_review_model", "TEXT", True),
    RunColumn("review_only", "INTEGER NOT NULL DEFAULT 0", True),
    RunColumn("deep", "INTEGER NOT NULL DEFAULT 0", True),
    RunColumn("remote_url", "TEXT", True),
    RunColumn("repo_slug", "TEXT", True),
    RunColumn("source_path", "TEXT", True),
    RunColumn("branch", "TEXT", True),
    RunColumn("base_branch", "TEXT", True),
    RunColumn("head_sha", "TEXT", True),
    RunColumn("base_sha", "TEXT", True),
    RunColumn("changed_files", "TEXT", True),
    RunColumn("pr_number", "INTEGER", True),
    RunColumn("pr_repo", "TEXT", True),
    RunColumn("total_cost_usd", "REAL", True),
    RunColumn("total_findings", "INTEGER", True),
    RunColumn("cost_per_finding_usd", "REAL", True),
    RunColumn("wall_clock_seconds", "REAL", True),
    RunColumn("erosion", "REAL", True),
    RunColumn("verbosity", "REAL", True),
    RunColumn("location_in_hunk_rate", "REAL", True),
    RunColumn("shipped_duplicate_pairs", "INTEGER", True),
    RunColumn("fix_quality_gate", "TEXT", True),
    RunColumn("recommended_patch_capture", "TEXT", True),
    RunColumn("total_prompt_tokens", "INTEGER", True),
    RunColumn("total_completion_tokens", "INTEGER", True),
    RunColumn("total_cached_tokens", "INTEGER", True),
    RunColumn("outcome_labels", "TEXT NOT NULL DEFAULT '[]'", True),
    RunColumn("labeled_at", "TEXT", True),
    RunColumn("rubric_json", "TEXT", False),
    RunColumn("composite_reward", "REAL", True),
    RunColumn("has_posterior", "INTEGER NOT NULL DEFAULT 0", False),
    RunColumn("archive_path", "TEXT NOT NULL", True),
    RunColumn("schema_version", "INTEGER NOT NULL DEFAULT 1", True),
    RunColumn("profile_schema_version", "INTEGER", True),
    RunColumn("profile_name", "TEXT", True),
    RunColumn("profile_source_kind", "TEXT", True),
    RunColumn("profile_digest", "TEXT", True),
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


_UPSERT_LINE_WIDTH = 92
"""Maximum length of a generated upsert column/parameter line (4-space indent included)."""


def _wrap_tokens(tokens: tuple[str, ...]) -> str:
    """Wrap comma-separated tokens at _UPSERT_LINE_WIDTH using four-space indentation."""
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
    """Generate matching column and parameter lists for declared upsert columns."""
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
# label" projections regardless of recency.
# ``labeler_policy_version`` mirrors ``labeler_version`` so the policy axis is an
# explicit column; ``reply_classifier_version`` / ``reply_evidence_digest`` carry
# the reply-classifier version and a stable digest over the combined reply
# evidence — together with ``evidence_sha`` they form the auto-dedup key
# ``(evidence_sha, labeler_policy_version, reply_evidence_digest, labels,
# has_posterior)``, replacing the older ``(evidence_sha, reward_version)`` key.
# ``legacy`` marks provenance generation: new rows default ``'auto'``;
# previously persisted ``'legacy'`` rows retain their historical provenance.
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

INDEXES = (
    ("idx_runs_repo_slug", "runs", "repo_slug"),
    ("idx_runs_archived_at", "runs", "archived_at"),
    ("idx_runs_outcome", "runs", "outcome_labels"),
    ("idx_label_obs_observed_at", "label_observations", "observed_at"),
    ("idx_label_obs_session", "label_observations", "session_id"),
)
_CREATE_INDEXES = [f"CREATE INDEX IF NOT EXISTS {name} ON {table}({column})" for name, table, column in INDEXES]


_UPSERT_SQL = _upsert_sql(RUNS_COLUMNS)
