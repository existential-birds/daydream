"""Local diagnostic index of frozen archive manifests.

Training annotations and HF publication read the canonical record store. This
index records finalized run metadata for diagnostics without authoring annotation history.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from daydream.archive._schema import (
    _CREATE_INDEXES,
    _CREATE_TABLE,
    _UPSERT_SQL,
    INDEXES,
    RUNS_COLUMNS,
    SCHEMA_VERSION,
)
from daydream.archive.git_safe import normalize_remote_url
from daydream.archive.manifest import Manifest


def _check_schema(conn: sqlite3.Connection, db_path: Path, *, allow_empty: bool = False) -> bool:
    """Validate without writes, returning whether an empty database needs initialization."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT GLOB 'sqlite_*'"
    )}
    if allow_empty and version == 0 and not tables:
        return True
    compatible = version == SCHEMA_VERSION
    runs_table = "runs"
    declarations = {runs_table: RUNS_COLUMNS}
    for table, columns in declarations.items():
        existing = {row[1]: row for row in conn.execute(f"PRAGMA table_info({table})")}  # noqa: S608
        primary_key = ("session_id",)
        actual_primary_key = tuple(row[1] for row in sorted(existing.values(), key=lambda row: row[5]) if row[5])
        compatible = compatible and actual_primary_key == primary_key
        for col in columns:
            row = existing.get(col.name)
            default = col.definition.split(" DEFAULT ", 1)[1] if " DEFAULT " in col.definition else None
            compatible = compatible and row is not None and (
                row[2] == col.definition.split()[0]
                and bool(row[3]) == ("NOT NULL" in col.definition)
                and row[4] == default
            )
    indexes = {row[0]: row[1] for row in conn.execute(
        "SELECT name, tbl_name FROM sqlite_master WHERE type = 'index'"
    )}
    for name, table, column in INDEXES:
        index_columns = [row[2] for row in conn.execute(f"PRAGMA index_info({name})")]  # noqa: S608
        properties = next((row for row in conn.execute(f"PRAGMA index_list({table})") if row[1] == name), None)  # noqa: S608
        compatible = compatible and (
            indexes.get(name) == table and index_columns == [column]
            and properties is not None and properties[2] == 0 and properties[4] == 0
        )
    if not compatible:
        raise ValueError(
            f"unsupported archive index schema at {db_path} (version {version}); "
            f"expected complete schema version {SCHEMA_VERSION} or an empty version-zero database. "
            "Keep this database for historical access and use a fresh archive directory."
        )
    return False


def _get_connection(archive_dir: Path) -> sqlite3.Connection:
    """Open a current index or initialize an empty one; never migrate existing data."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    db_path = archive_dir / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        fresh = _check_schema(conn, db_path, allow_empty=True)
        # Compatibility must be established before WAL or any persistent writes.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        if fresh:
            conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                conn.execute(idx_sql)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        return conn
    except BaseException:
        conn.close()
        raise


@contextmanager
def _connection(archive_dir: Path) -> Iterator[sqlite3.Connection]:
    """Yield a writer connection and close it on exit."""
    conn = _get_connection(archive_dir)
    try:
        yield conn
    finally:
        conn.close()


def _project_daydream(daydream: Any) -> dict[str, Any]:
    """Project executable provenance; absent provenance and non-bool dirty states become NULL."""
    values = {
        f"daydream_{name}": getattr(daydream, name) if daydream is not None else None
        for name in ("version", "install_source", "commit", "dirty", "container_digest")
    }
    dirty = values["daydream_dirty"]
    values["daydream_dirty"] = int(dirty) if isinstance(dirty, bool) else None
    return values


def _run_upsert_values(manifest: Manifest) -> dict[str, Any]:
    """Project declared upsert columns, normalizing credentials, JSON, booleans, and provenance."""
    # Normalize again at persistence even if capture bypassed URL sanitization.
    normalized_slug, normalized_url = (
        (manifest.repo_slug, None)
        if manifest.remote_url is None
        else normalize_remote_url(manifest.remote_url)
    )
    overrides = {
        **_project_daydream(manifest.daydream),
        "review_only": int(manifest.review_only),
        "deep": int(manifest.deep),
        "remote_url": normalized_url,
        "repo_slug": normalized_slug,
        "changed_files": json.dumps(manifest.changed_files),
        "phase_states": json.dumps(manifest.phase_states) if manifest.phase_states is not None else None,
        "fix_quality_gate": json.dumps(manifest.fix_quality_gate)
        if manifest.fix_quality_gate is not None
        else None,
        "schema_version": SCHEMA_VERSION,
    }
    return {
        col.name: overrides[col.name] if col.name in overrides else getattr(manifest, col.name)
        for col in RUNS_COLUMNS
    }


def upsert_run(archive_dir: Path, manifest: Manifest) -> None:
    """Insert or replace a manifest using the schema declaration for columns and parameters."""
    values = _run_upsert_values(manifest)
    with _connection(archive_dir) as conn:
        conn.execute(_UPSERT_SQL, values)
        conn.commit()

