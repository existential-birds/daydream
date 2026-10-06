"""Local diagnostic schemas fail closed without rewriting historical archives."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.archive import index
from tests.harness.trajectory import make_manifest


def _current_database(root: Path) -> None:
    index.upsert_run(root, make_manifest(session_id="historical", repo_slug="org/repo"))


def test_local_diagnostic_index_records_runs_without_annotation_tables(tmp_path: Path) -> None:
    _current_database(tmp_path)
    with closing(sqlite3.connect(tmp_path / "index.db")) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 9
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("runs",)]
    row = index.query_runs(tmp_path)[0]
    assert row["session_id"] == "historical"
    assert row["status"] == "complete" and row["pipeline_status"] == "unknown"
    assert row["repo_slug"] == "org/repo"


@pytest.mark.parametrize("unsupported", [
    "old", "future", "zero-populated", "old-empty", "future-empty", "missing-runs",
    "missing-column", "missing-index", "wrong-index", "wrong-default", "wrong-type",
    "missing-primary-key", "extra-primary-key", "zero-unrelated-table", "wrong-index-table",
    "unique-index", "partial-index",
])
@pytest.mark.parametrize("readonly", [False, True], ids=["write", "read"])
def test_unsupported_schema_is_rejected_without_changing_database(
    tmp_path: Path, unsupported: str, readonly: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _current_database(tmp_path)
    db_path = tmp_path / "index.db"
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        if unsupported in {"old", "future", "zero-populated", "old-empty", "future-empty", "zero-unrelated-table"}:
            version = {"old": 8, "future": 10, "zero-populated": 0, "old-empty": 8, "future-empty": 10,
                       "zero-unrelated-table": 0}[unsupported]
            connection.execute(f"PRAGMA user_version = {version}")
        if unsupported in {"old-empty", "future-empty", "zero-unrelated-table", "missing-runs"}:
            connection.execute("DROP TABLE runs")
        elif unsupported == "missing-column":
            connection.execute("ALTER TABLE runs DROP COLUMN profile_digest")
        elif unsupported in {"missing-index", "wrong-index", "unique-index", "partial-index"}:
            connection.execute("DROP INDEX idx_runs_repo_slug")
            if unsupported == "wrong-index":
                connection.execute("CREATE INDEX idx_runs_repo_slug ON runs(session_id)")
            elif unsupported == "unique-index":
                connection.execute("CREATE UNIQUE INDEX idx_runs_repo_slug ON runs(repo_slug)")
            elif unsupported == "partial-index":
                connection.execute("CREATE INDEX idx_runs_repo_slug ON runs(repo_slug) WHERE repo_slug IS NOT NULL")
        elif unsupported == "wrong-index-table":
            connection.execute("DROP INDEX idx_runs_repo_slug")
            connection.execute("CREATE TABLE unrelated (repo_slug TEXT)")
            connection.execute("CREATE INDEX idx_runs_repo_slug ON unrelated(repo_slug)")
        else:
            ddl = connection.execute("SELECT sql FROM sqlite_master WHERE name='runs'").fetchone()[0]
            indexes = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL").fetchall()
            connection.execute("DROP TABLE runs")
            if unsupported == "wrong-default":
                ddl = ddl.replace("DEFAULT 'unknown'", "DEFAULT 'complete'")
            elif unsupported == "wrong-type":
                ddl = ddl.replace("model TEXT", "model INTEGER")
            elif unsupported == "missing-primary-key":
                ddl = ddl.replace("session_id TEXT PRIMARY KEY", "session_id TEXT")
            else:
                ddl = ddl.replace("session_id TEXT PRIMARY KEY", "session_id TEXT")
                ddl = ddl.rstrip().removesuffix(")") + ", extra TEXT, PRIMARY KEY (session_id, extra))"
            connection.execute(ddl)
            for (sql,) in indexes:
                connection.execute(sql)
        connection.execute("CREATE TABLE historical_notes (note TEXT)")
        connection.execute("INSERT INTO historical_notes VALUES ('keep historical evidence')")
        connection.commit()
        before_dump = list(connection.iterdump())
    before_bytes = db_path.read_bytes()
    before_tree = sorted(path.name for path in tmp_path.iterdir())
    opened: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def tracked_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = cast(sqlite3.Connection, connect(*args, **kwargs))
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    opener = index.readonly_connection if readonly else index._get_connection
    with pytest.raises(ValueError, match="unsupported archive index schema.*fresh archive directory"):
        opener(tmp_path)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")
    assert db_path.read_bytes() == before_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == before_tree
    with closing(connect(db_path)) as connection:
        assert list(connection.iterdump()) == before_dump
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_local_diagnostic_queries_preserve_unrelated_history_and_files(tmp_path: Path) -> None:
    _current_database(tmp_path)
    db_path = tmp_path / "index.db"
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute("ALTER TABLE runs ADD COLUMN retired_metric REAL")
        connection.execute("UPDATE runs SET retired_metric = 0.75")
        connection.execute("CREATE TABLE historical_notes (note TEXT)")
        connection.execute("INSERT INTO historical_notes VALUES ('keep me')")
    before = db_path.read_bytes()
    tree = sorted(path.name for path in tmp_path.iterdir())
    assert index.query_runs(tmp_path, "session_id = ?", ("historical",))[0]["retired_metric"] == 0.75
    assert db_path.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == tree
    index.upsert_run(tmp_path, make_manifest(session_id="new"))
    with closing(sqlite3.connect(db_path)) as connection:
        assert connection.execute("SELECT note FROM historical_notes").fetchone()[0] == "keep me"
        assert connection.execute("SELECT retired_metric FROM runs WHERE session_id='historical'").fetchone()[0] == 0.75
