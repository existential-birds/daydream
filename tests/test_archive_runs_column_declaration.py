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


def test_unsupported_schema_is_rejected_without_changing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _current_database(tmp_path)
    db_path = tmp_path / "index.db"
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("DROP INDEX idx_runs_repo_slug")
        connection.execute("CREATE TABLE unrelated (repo_slug TEXT)")
        connection.execute("CREATE INDEX idx_runs_repo_slug ON unrelated(repo_slug)")
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
    with pytest.raises(ValueError, match="unsupported archive index schema.*fresh archive directory"):
        index.upsert_run(tmp_path, make_manifest(session_id="refused"))
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")
    assert db_path.read_bytes() == before_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == before_tree
    with closing(connect(db_path)) as connection:
        assert list(connection.iterdump()) == before_dump
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
