"""The runs column set is declared once in ``_schema.RUNS_COLUMNS``.

Every schema path — the ``CREATE TABLE`` text, the ``ALTER TABLE ADD COLUMN``
entries ``_migrate_schema`` applies, and the run-upsert statement — is generated
from that one declaration, so a column can no longer exist for fresh databases
and be missing for every existing corpus. The witnesses below are frozen
historical facts about the schema at the time of the refactor (issue #1219); a
deliberate edit is required to change them, which is the point.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3

import pytest

from daydream.archive import _schema
from daydream.archive._schema import RUNS_COLUMNS

# Frozen witness: the whole generated CREATE TABLE text at the refactor commit.
FROZEN_DDL_SHA256 = "eac468a7be8b7830b245925e55e3be2d03abceeb3f443d9332404e0ff36e68c8"

# Frozen witness: the columns of the original v1 runs table (every column that
# is already present in any legacy database by construction, so it must never
# be declared additive).
V1_BASELINE_NAMES = frozenset(
    {
        "archive_path",
        "archived_at",
        "backend",
        "base_branch",
        "branch",
        "cost_per_finding_usd",
        "coverage_ratio",
        "deep",
        "grounding_rate",
        "head_sha",
        "labeled_at",
        "model",
        "outcome_labels",
        "pr_number",
        "pr_repo",
        "remote_url",
        "repo_slug",
        "review_only",
        "run_flow",
        "schema_version",
        "session_id",
        "skill",
        "status",
        "total_cached_tokens",
        "total_completion_tokens",
        "total_cost_usd",
        "total_findings",
        "total_prompt_tokens",
        "wall_clock_seconds",
    }
)

# Frozen witness: the (name, column type) pairs the migration applied before the
# refactor — the equivalence bar for the generated entries.
FROZEN_MIGRATION_ENTRIES = frozenset(
    {
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
    }
)

# Frozen witness: the columns no run upsert writes — they are maintained by the
# label-observation paths, which own their values.
WRITER_OWNED = frozenset({"rubric_json", "has_posterior"})

ADDITIVE_NAMES = frozenset(name for name, _ in FROZEN_MIGRATION_ENTRIES)
ALL_NAMES = V1_BASELINE_NAMES | ADDITIVE_NAMES
UPSERT_NAMES = ALL_NAMES - WRITER_OWNED


def test_the_declaration_lists_every_column_exactly_once() -> None:
    names = [col.name for col in RUNS_COLUMNS]
    assert set(names) == ALL_NAMES
    assert len(names) == len(set(names)) == 58
    assert all(col.definition.strip() for col in RUNS_COLUMNS)


def test_the_declaration_partitions_additive_and_writer_owned_columns() -> None:
    assert {col.name for col in RUNS_COLUMNS if col.additive} == ADDITIVE_NAMES
    # The v1 witness: a column declared "already present" cannot be new.
    assert {col.name for col in RUNS_COLUMNS if not col.additive} == V1_BASELINE_NAMES
    assert {col.name for col in RUNS_COLUMNS if col.upserted} == UPSERT_NAMES


def test_create_table_is_generated_and_byte_identical_to_the_frozen_text() -> None:
    assert _schema._CREATE_TABLE == _schema._create_table_sql(RUNS_COLUMNS)
    assert (
        hashlib.sha256(_schema._CREATE_TABLE.encode()).hexdigest() == FROZEN_DDL_SHA256
    ), f"generated CREATE TABLE drifted from the frozen text:\n{_schema._CREATE_TABLE}"
    lines = _schema._CREATE_TABLE.splitlines()
    assert lines[0] == "" and lines[1] == "CREATE TABLE IF NOT EXISTS runs ("
    assert lines[-1] == ")"
    body = lines[2:-1]
    assert len(body) == len(RUNS_COLUMNS)
    assert all(line.startswith("    ") and not line.startswith("     ") for line in body)
    assert [line.strip().rstrip(",").split()[0] for line in body] == [col.name for col in RUNS_COLUMNS]


def test_migration_entries_are_generated_and_match_the_frozen_pairs() -> None:
    entries = _schema._migration_entries(RUNS_COLUMNS)
    assert set(entries) == FROZEN_MIGRATION_ENTRIES
    assert entries == [(col.name, col.definition) for col in RUNS_COLUMNS if col.additive]


def test_migrate_schema_applies_the_generated_entries_in_declaration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, list[tuple[str, str]]]] = []

    def _capture(conn: sqlite3.Connection, table: str, migrations: list[tuple[str, str]]) -> None:
        captured.append((table, list(migrations)))

    monkeypatch.setattr(_schema, "_alter_add_missing", _capture)
    _schema._migrate_schema(sqlite3.connect(":memory:"))
    assert captured == [("runs", _schema._migration_entries(RUNS_COLUMNS))]


def _split_sql_columns(block: str) -> list[str]:
    return [token.strip() for line in block.splitlines() for token in line.split(",") if token.strip()]


def _upsert_statement_names() -> tuple[list[str], list[str]]:
    """Return (column names, parameter names) in the generated upsert statement."""
    column_block = re.search(r"runs \(\n(.*?)\n\) VALUES \(", _schema._UPSERT_SQL, re.S)
    param_block = re.search(r"VALUES \(\n(.*?)\n\)\n", _schema._UPSERT_SQL, re.S)
    assert column_block is not None and param_block is not None
    return _split_sql_columns(column_block.group(1)), [
        token.lstrip(":") for token in _split_sql_columns(param_block.group(1))
    ]


def test_upsert_statement_is_generated_from_the_declaration() -> None:
    columns, params = _upsert_statement_names()
    expected = [col.name for col in RUNS_COLUMNS if col.upserted]
    assert columns == params == expected
    assert _schema._UPSERT_SQL == _schema._upsert_sql(RUNS_COLUMNS)


def test_writer_owned_columns_are_never_written_by_the_upsert() -> None:
    columns, params = _upsert_statement_names()
    assert not WRITER_OWNED & set(columns)
    assert not WRITER_OWNED & set(params)
