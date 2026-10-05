"""Fresh schema and upserts share RUNS_COLUMNS.

Independent frozen column/type witnesses make compatibility changes explicit.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.archive import _schema, index
from daydream.archive._schema import RUNS_COLUMNS
from daydream.archive.index import _get_connection, _run_upsert_values
from tests.harness.trajectory import make_manifest

# Current fresh-database shape after removing reviewer-read metrics.
CURRENT_DDL_SHA256 = "92d502a2e38d7147c30f0ca5c1d8ab89da92cc8f9c734fe84a9c3ebb6d4db41a"

# Independent witness of current columns and label-writer ownership.
ALL_NAMES = frozenset({
    'archive_path', 'archive_status', 'archived_at', 'backend', 'base_branch',
    'base_sha', 'branch', 'changed_files', 'composite_reward', 'cost_per_finding_usd',
    'daydream_commit', 'daydream_container_digest', 'daydream_dirty', 'daydream_install_source', 'daydream_version',
    'deep', 'erosion', 'fix_backend', 'fix_quality_gate', 'has_posterior',
    'head_sha', 'labeled_at', 'location_in_hunk_rate', 'model', 'outcome_labels',
    'per_stack_review_backend', 'per_stack_review_model', 'phase_states', 'pipeline_status', 'pr_number',
    'pr_repo', 'profile_digest', 'profile_name', 'profile_schema_version', 'profile_source_kind',
    'recommended_patch_capture', 'remote_url', 'repo_slug', 'review_backend', 'review_only',
    'rubric_json', 'run_flow', 'schema_version', 'session_id', 'shipped_duplicate_pairs',
    'skill', 'source_path', 'status', 'test_backend', 'total_cached_tokens',
    'total_completion_tokens', 'total_cost_usd', 'total_findings', 'total_prompt_tokens', 'verbosity',
    'wall_clock_seconds',
})
WRITER_OWNED = frozenset({"rubric_json", "has_posterior"})
UPSERT_NAMES = ALL_NAMES - WRITER_OWNED

def test_the_declaration_lists_every_column_exactly_once() -> None:
    names = [col.name for col in RUNS_COLUMNS]
    assert set(names) == ALL_NAMES
    assert len(names) == len(set(names)) == 56
    assert all(col.definition.strip() for col in RUNS_COLUMNS)


def test_create_table_is_generated_with_obsolete_metrics_removed() -> None:
    assert _schema._CREATE_TABLE == _schema._create_table_sql(RUNS_COLUMNS)
    assert (hashlib.sha256(_schema._CREATE_TABLE.encode()).hexdigest() == CURRENT_DDL_SHA256
    ), f"generated CREATE TABLE drifted from the current text:\n{_schema._CREATE_TABLE}"
    lines = _schema._CREATE_TABLE.splitlines()
    assert lines[0] == "" and lines[1] == "CREATE TABLE IF NOT EXISTS runs ("
    assert lines[-1] == ")"
    body = lines[2:-1]
    assert len(body) == len(RUNS_COLUMNS)
    assert all(line.startswith("    ") and not line.startswith("     ") for line in body)
    assert [line.strip().rstrip(",").split()[0] for line in body] == [col.name for col in RUNS_COLUMNS]


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
    assert index.RUNS_COLUMNS is RUNS_COLUMNS
    assert "RUNS_COLUMNS" in index.__all__


def test_generation_tracks_a_mutated_declaration() -> None:
    source = Path(_schema.__file__).read_text()
    start = source.index("RUNS_COLUMNS: tuple[RunColumn, ...] = (")
    open_paren = source.index("(", start)
    depth = 0
    close_paren = -1
    for idx in range(open_paren, len(source)):
        if source[idx] == "(":
            depth += 1
        elif source[idx] == ")":
            depth -= 1
            if depth == 0:
                close_paren = idx
                break
    assert close_paren > open_paren
    probe = '    RunColumn("zzz_probe", "TEXT NOT NULL DEFAULT \'probe\'", True),\n'
    mutated = source[:close_paren] + probe + source[close_paren:]

    namespace: dict[str, Any] = {"__name__": "schema_mutated", "__file__": _schema.__file__}
    exec(compile(mutated, "<schema_mutated>", "exec"), namespace)  # noqa: S102 - test-local namespace

    mutated_columns = namespace["RUNS_COLUMNS"]
    assert len(mutated_columns) == len(RUNS_COLUMNS) + 1
    assert namespace["_CREATE_TABLE"] == namespace["_create_table_sql"](mutated_columns)
    assert "    zzz_probe TEXT NOT NULL DEFAULT 'probe'\n" in namespace["_CREATE_TABLE"]
    assert ":zzz_probe" in namespace["_UPSERT_SQL"]



def _current_database(archive_dir: Path) -> None:
    with closing(sqlite3.connect(archive_dir / "index.db")) as conn, conn:
        conn.execute(_schema._CREATE_TABLE)
        conn.execute(_schema._CREATE_LABEL_OBSERVATIONS_TABLE)
        for sql in _schema._CREATE_INDEXES:
            conn.execute(sql)
        conn.execute(f"PRAGMA user_version = {index.SCHEMA_VERSION}")
        conn.execute(
            "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) VALUES (?, ?, ?, ?)",
            ("historical", "2026-01-01T00:00:00Z", "normal", "/historical"),
        )
        conn.execute(
            "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, legacy) "
            "VALUES ('historical', '2026-01-01T00:00:00Z', '[\"accepted\"]', 'old-policy', 'legacy')"
        )


def test_fresh_empty_database_is_initialized_with_current_defaults(tmp_path: Path) -> None:
    # Existing empty SQLite databases and absent database files are both fresh.
    for root in (tmp_path / "absent", tmp_path / "empty"):
        root.mkdir()
        if root.name == "empty":
            sqlite3.connect(root / "index.db").close()
        conn = _get_connection(root)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == index.SCHEMA_VERSION
            assert {row[1] for row in conn.execute("PRAGMA table_info(runs)")} == ALL_NAMES
            assert {row[1] for row in conn.execute("PRAGMA table_info(label_observations)")} == set(
                index.LABEL_OBSERVATION_NAMES
            )
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()
        index.upsert_run(root, make_manifest(session_id="fresh"))
        row = index.query_runs(root)[0]
        assert row["has_posterior"] == 0 and row["rubric_json"] is None


@pytest.mark.parametrize("unsupported", [
    "old", "future", "zero-populated", "old-empty", "future-empty", "missing-runs", "missing-observations",
    "missing-run-column", "missing-observation-column", "missing-index", "wrong-index", "wrong-default",
    "wrong-type", "missing-primary-key", "zero-unrelated-table", "wrong-index-table", "unique-index", "partial-index",
])
@pytest.mark.parametrize("readonly", [False, True], ids=["write", "read"])
def test_unsupported_schema_is_rejected_without_changing_database(
    tmp_path: Path, unsupported: str, readonly: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _current_database(tmp_path)
    db_path = tmp_path / "index.db"
    with closing(sqlite3.connect(db_path)) as conn, conn:
        if unsupported in {"old", "future", "zero-populated", "old-empty", "future-empty", "zero-unrelated-table"}:
            version = {"old": 7, "future": 9, "zero-populated": 0, "old-empty": 7, "future-empty": 9,
                       "zero-unrelated-table": 0}[unsupported]
            conn.execute(f"PRAGMA user_version = {version}")
        if unsupported in {"old-empty", "future-empty", "zero-unrelated-table"}:
            conn.execute("DROP TABLE runs")
            conn.execute("DROP TABLE label_observations")
        elif unsupported == "missing-runs":
            conn.execute("DROP TABLE runs")
        elif unsupported == "missing-observations":
            conn.execute("DROP TABLE label_observations")
        elif unsupported == "missing-run-column":
            conn.execute("ALTER TABLE runs DROP COLUMN profile_digest")
        elif unsupported == "missing-observation-column":
            conn.execute("ALTER TABLE label_observations DROP COLUMN legacy")
        elif unsupported in {"missing-index", "wrong-index"}:
            conn.execute("DROP INDEX idx_runs_repo_slug")
            if unsupported == "wrong-index":
                conn.execute("CREATE INDEX idx_runs_repo_slug ON runs(session_id)")
        elif unsupported == "wrong-index-table":
            conn.execute("DROP INDEX idx_label_obs_session")
            conn.execute("CREATE INDEX idx_label_obs_session ON runs(session_id)")
        elif unsupported in {"unique-index", "partial-index"}:
            conn.execute("DROP INDEX idx_runs_repo_slug")
            if unsupported == "unique-index":
                conn.execute("CREATE UNIQUE INDEX idx_runs_repo_slug ON runs(repo_slug)")
            else:
                conn.execute("CREATE INDEX idx_runs_repo_slug ON runs(repo_slug) WHERE repo_slug IS NOT NULL")
        elif unsupported in {"wrong-default", "wrong-type", "missing-primary-key"}:
            conn.execute("DROP TABLE label_observations")
            ddl = _schema._CREATE_LABEL_OBSERVATIONS_TABLE
            if unsupported == "wrong-default":
                ddl = ddl.replace("DEFAULT 'auto'", "DEFAULT 'human'")
            elif unsupported == "wrong-type":
                ddl = ddl.replace("labels TEXT", "labels INTEGER")
            else:
                ddl = ddl.replace(",\n    PRIMARY KEY (session_id, observed_at)", "")
            conn.execute(ddl)
        # Unrelated historical tables must also survive rejection.
        conn.execute("CREATE TABLE historical_notes (note TEXT)")
        conn.execute("INSERT INTO historical_notes VALUES ('keep historical evidence')")
        conn.commit()
        before_dump = list(conn.iterdump())
        before_journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    before_bytes = db_path.read_bytes()
    before_tree = sorted(p.name for p in tmp_path.iterdir())
    opened: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def tracked_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = cast(sqlite3.Connection, connect(*args, **kwargs))
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    opener = index.readonly_connection if readonly else _get_connection
    with pytest.raises(ValueError, match="unsupported archive index schema.*fresh archive directory"):
        opener(tmp_path)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")
    assert db_path.read_bytes() == before_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == before_tree
    with closing(connect(db_path)) as conn, conn:
        assert list(conn.iterdump()) == before_dump
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == before_journal == "delete"


def test_current_database_preserves_observation_history_and_extra_tables(tmp_path: Path) -> None:
    _current_database(tmp_path)
    with closing(sqlite3.connect(tmp_path / "index.db")) as conn, conn:
        conn.execute("ALTER TABLE runs ADD COLUMN retired_metric REAL")
        conn.execute("UPDATE runs SET retired_metric = 0.75")
        conn.execute("CREATE TABLE historical_notes (note TEXT)")
        conn.execute("INSERT INTO historical_notes VALUES ('keep me')")
        # A current-schema row with unknown policy must not be stamped on open.
        conn.execute(
            "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, source) "
            "VALUES ('historical', '2026-02-01T00:00:00Z', '[\"rejected\"]', 'human', 'human')"
        )
        conn.commit()
        before = conn.execute("SELECT * FROM label_observations ORDER BY observed_at").fetchall()
    index.upsert_run(tmp_path, make_manifest(session_id="new"))
    assert index.append_label_observation(
        tmp_path, "historical", labels=["contested"], pr_state="open", labeler_version="new-policy",
        evidence_sha="new-evidence", source="auto",
    )
    with closing(sqlite3.connect(tmp_path / "index.db")) as conn, conn:
        assert conn.execute("SELECT * FROM label_observations ORDER BY observed_at").fetchall()[:2] == before
        assert conn.execute("SELECT note FROM historical_notes").fetchone()[0] == "keep me"
        assert conn.execute("SELECT retired_metric FROM runs WHERE session_id='historical'").fetchone()[0] == 0.75
    winner = index.latest_label_observation(tmp_path, "historical")
    assert winner is not None and winner["source"] == "human" and winner["labels"] == '["rejected"]'
    assert index.query_runs(tmp_path, "session_id = ?", ("historical",))[0]["outcome_labels"] == '["rejected"]'
    conn = index.readonly_connection(tmp_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == index.SCHEMA_VERSION
    finally:
        conn.close()
