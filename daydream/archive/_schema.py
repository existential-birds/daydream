"""Current archive index DDL, owned by daydream.archive.index."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

SCHEMA_VERSION = 9

class RunColumn(NamedTuple):
    """SQL column of the local diagnostic run index."""

    name: str
    definition: str


RUNS_COLUMNS: tuple[RunColumn, ...] = (
    RunColumn("session_id", "TEXT PRIMARY KEY"),
    RunColumn("archived_at", "TEXT NOT NULL"),
    RunColumn("status", "TEXT NOT NULL DEFAULT 'complete'"),
    RunColumn("archive_status", "TEXT NOT NULL DEFAULT 'complete'"),
    RunColumn("pipeline_status", "TEXT NOT NULL DEFAULT 'unknown'"),
    RunColumn("phase_states", "TEXT"),
    RunColumn("daydream_version", "TEXT"),
    RunColumn("daydream_install_source", "TEXT"),
    RunColumn("daydream_commit", "TEXT"),
    RunColumn("daydream_dirty", "INTEGER"),
    RunColumn("daydream_container_digest", "TEXT"),
    RunColumn("run_flow", "TEXT NOT NULL"),
    RunColumn("skill", "TEXT"),
    RunColumn("model", "TEXT"),
    RunColumn("backend", "TEXT NOT NULL DEFAULT 'claude'"),
    RunColumn("review_backend", "TEXT"),
    RunColumn("fix_backend", "TEXT"),
    RunColumn("test_backend", "TEXT"),
    RunColumn("per_stack_review_backend", "TEXT"),
    RunColumn("per_stack_review_model", "TEXT"),
    RunColumn("review_only", "INTEGER NOT NULL DEFAULT 0"),
    RunColumn("deep", "INTEGER NOT NULL DEFAULT 0"),
    RunColumn("remote_url", "TEXT"),
    RunColumn("repo_slug", "TEXT"),
    RunColumn("source_path", "TEXT"),
    RunColumn("branch", "TEXT"),
    RunColumn("base_branch", "TEXT"),
    RunColumn("head_sha", "TEXT"),
    RunColumn("base_sha", "TEXT"),
    RunColumn("changed_files", "TEXT"),
    RunColumn("pr_number", "INTEGER"),
    RunColumn("pr_repo", "TEXT"),
    RunColumn("total_cost_usd", "REAL"),
    RunColumn("total_findings", "INTEGER"),
    RunColumn("cost_per_finding_usd", "REAL"),
    RunColumn("wall_clock_seconds", "REAL"),
    RunColumn("erosion", "REAL"),
    RunColumn("verbosity", "REAL"),
    RunColumn("location_in_hunk_rate", "REAL"),
    RunColumn("shipped_duplicate_pairs", "INTEGER"),
    RunColumn("fix_quality_gate", "TEXT"),
    RunColumn("recommended_patch_capture", "TEXT"),
    RunColumn("total_prompt_tokens", "INTEGER"),
    RunColumn("total_completion_tokens", "INTEGER"),
    RunColumn("total_cached_tokens", "INTEGER"),
    RunColumn("archive_path", "TEXT NOT NULL"),
    RunColumn("schema_version", "INTEGER NOT NULL DEFAULT 1"),
    RunColumn("profile_schema_version", "INTEGER"),
    RunColumn("profile_name", "TEXT"),
    RunColumn("profile_source_kind", "TEXT"),
    RunColumn("profile_digest", "TEXT"),
)


def _create_table_sql(columns: Iterable[RunColumn]) -> str:
    """Render local run declarations verbatim."""
    lines = [f"    {col.name} {col.definition}" for col in columns]
    return "\nCREATE TABLE IF NOT EXISTS runs (\n" + ",\n".join(lines) + "\n)\n"


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
    participating = tuple(col.name for col in columns)
    parameters = tuple(f":{name}" for name in participating)
    return (
        "\nINSERT OR REPLACE INTO runs (\n"
        + _wrap_tokens(participating)
        + "\n) VALUES (\n"
        + _wrap_tokens(parameters)
        + "\n)\n"
    )


_CREATE_TABLE = _create_table_sql(RUNS_COLUMNS)

INDEXES = (
    ("idx_runs_repo_slug", "runs", "repo_slug"),
    ("idx_runs_archived_at", "runs", "archived_at"),
)
_CREATE_INDEXES = [f"CREATE INDEX IF NOT EXISTS {name} ON {table}({column})" for name, table, column in INDEXES]


_UPSERT_SQL = _upsert_sql(RUNS_COLUMNS)
