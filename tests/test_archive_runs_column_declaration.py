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
from pathlib import Path
from typing import Any

import pytest

from daydream.archive import _schema
from daydream.archive._schema import RUNS_COLUMNS, RunColumn
from daydream.archive.index import _get_connection, _run_upsert_values
from tests.harness.trajectory import make_manifest

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


def test_upsert_values_mapping_covers_exactly_the_declared_upsert_columns() -> None:
    assert set(_run_upsert_values(make_manifest())) == UPSERT_NAMES


def test_declaration_is_importable_from_both_module_paths() -> None:
    from daydream.archive import index

    assert index.RUNS_COLUMNS is RUNS_COLUMNS
    assert "RUNS_COLUMNS" in index.__all__


def _v1_baseline_sql() -> str:
    return _schema._create_table_sql([col for col in RUNS_COLUMNS if not col.additive])


def _pre_v4_sql() -> str:
    return _schema._create_table_sql([col for col in RUNS_COLUMNS if col.name != "has_posterior"])


def _open_legacy(archive_dir: Path, ddl: str) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(archive_dir / "index.db"))
    try:
        conn.execute(ddl)
        conn.execute(
            "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) VALUES (?, ?, ?, ?)",
            ("legacy-row", "2026-01-01T00:00:00+00:00", "normal", "/x"),
        )
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
    finally:
        conn.close()


def _runs_columns(archive_dir: Path) -> set[str]:
    conn = _get_connection(archive_dir)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    finally:
        conn.close()


@pytest.mark.parametrize("column", [col for col in RUNS_COLUMNS if col.additive], ids=lambda col: col.name)
def test_every_additive_column_is_alter_eligible_on_a_populated_table(column: RunColumn) -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(_v1_baseline_sql())
        conn.execute(
            "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) VALUES (?, ?, ?, ?)",
            ("populated", "2026-01-01T00:00:00+00:00", "normal", "/x"),
        )
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column.name} {column.definition}")
        except sqlite3.OperationalError as exc:  # pragma: no cover - failure path
            raise AssertionError(
                f"declared additive column {column.name!r} cannot be added to a populated "
                f"table with definition {column.definition!r}: {exc}"
            ) from exc
    finally:
        conn.close()


@pytest.mark.parametrize("build_legacy", [_v1_baseline_sql, _pre_v4_sql], ids=["v1-baseline", "pre-v4"])
def test_fresh_and_upgraded_databases_end_with_the_same_column_set(
    tmp_path: Path, build_legacy: Any
) -> None:
    legacy_dir = tmp_path / "legacy"
    fresh_dir = tmp_path / "fresh"
    _open_legacy(legacy_dir, build_legacy())

    assert _runs_columns(legacy_dir) == _runs_columns(fresh_dir) == set(ALL_NAMES)


def test_generation_tracks_a_mutated_declaration() -> None:
    source = Path(_schema.__file__).read_text()
    start = source.index("RUNS_COLUMNS: tuple[RunColumn, ...] = (")
    open_paren = source.index("(", start)
    depth = 0
    close_paren = -1
    for index in range(open_paren, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                close_paren = index
                break
    assert close_paren > open_paren
    probe = '    RunColumn("zzz_probe", "TEXT NOT NULL DEFAULT \'probe\'", True, True),\n'
    mutated = source[:close_paren] + probe + source[close_paren:]

    namespace: dict[str, Any] = {"__name__": "schema_mutated", "__file__": _schema.__file__}
    exec(compile(mutated, "<schema_mutated>", "exec"), namespace)  # noqa: S102 - test-local namespace

    mutated_columns = namespace["RUNS_COLUMNS"]
    assert len(mutated_columns) == len(RUNS_COLUMNS) + 1
    assert namespace["_CREATE_TABLE"] == namespace["_create_table_sql"](mutated_columns)
    assert "    zzz_probe TEXT NOT NULL DEFAULT 'probe'\n" in namespace["_CREATE_TABLE"]
    assert ("zzz_probe", "TEXT NOT NULL DEFAULT 'probe'") in namespace["_migration_entries"](mutated_columns)
    assert ":zzz_probe" in namespace["_UPSERT_SQL"]
