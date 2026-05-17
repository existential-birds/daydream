"""Tests for daydream.training.export end-to-end orchestration (Wave 6).

Drives :func:`run_export` against the §9 fixture matrix materialized via
``tests.fixtures.training.build_archive.build_fixture_archive`` — a real
SQLite index built with the production ``upsert_run`` helper. No mocking
of SQLite, the archive layer, or the filesystem.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema

from daydream.training.export import ExportConfig, ExportFilters, run_export
from tests.fixtures.training.build_archive import build_fixture_archive

SCHEMA_PATH = Path(__file__).parent.parent / "daydream" / "training" / "schema" / "v1.json"


def _cfg(tmp_path: Path, **overrides: Any) -> ExportConfig:
    """Build an ExportConfig pointing at ``tmp_path`` with sensible defaults."""
    base: dict[str, Any] = {
        "out_path": tmp_path / "out.jsonl",
        "filters": ExportFilters(),
        "archive_dir": tmp_path,
    }
    base.update(overrides)
    return ExportConfig(**base)


def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_export_emits_valid_jsonl(tmp_path: Path) -> None:
    """Every emitted line is valid JSON and validates against schema v1."""
    build_fixture_archive(tmp_path)
    summary = run_export(_cfg(tmp_path))
    assert summary["emitted"] > 0

    schema = _load_schema()
    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines, "expected at least one emitted record"
    for line in lines:
        record = json.loads(line)
        jsonschema.validate(record, schema)


def test_export_record_fields_present(tmp_path: Path) -> None:
    """Every required schema field appears on the first emitted record."""
    build_fixture_archive(tmp_path)
    run_export(_cfg(tmp_path))

    schema = _load_schema()
    first_line = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    for field in schema["required"]:
        assert field in record, f"missing required field: {field}"


def test_export_deterministic_output(tmp_path: Path) -> None:
    """Two runs against the same archive produce byte-identical JSONL."""
    build_fixture_archive(tmp_path)
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    run_export(_cfg(tmp_path, out_path=out_a))
    run_export(_cfg(tmp_path, out_path=out_b))

    digest_a = hashlib.sha256(out_a.read_bytes()).hexdigest()
    digest_b = hashlib.sha256(out_b.read_bytes()).hexdigest()
    assert digest_a == digest_b


def test_export_schema_json_emitted_next_to_out(tmp_path: Path) -> None:
    """``schema.json`` lands next to ``out_path`` and matches the source."""
    build_fixture_archive(tmp_path)
    config = _cfg(tmp_path)
    run_export(config)

    schema_dst = config.out_path.parent / "schema.json"
    assert schema_dst.exists()
    assert schema_dst.read_text(encoding="utf-8") == SCHEMA_PATH.read_text(encoding="utf-8")


def test_export_dry_run_writes_nothing(tmp_path: Path) -> None:
    """``dry_run=True`` skips file writes and reports ``emitted=0``."""
    build_fixture_archive(tmp_path)
    config = _cfg(tmp_path, dry_run=True)
    summary = run_export(config)

    assert not config.out_path.exists()
    assert summary["emitted"] == 0
    assert summary["after_filters"] > 0  # filter pipeline still ran


def test_export_missing_trajectory_logged_and_skipped(tmp_path: Path) -> None:
    """A row with no ``trajectory.json`` is skipped (not crashed) and absent."""
    build_fixture_archive(tmp_path)
    (tmp_path / "runs" / "aaa-python-accepted" / "trajectory.json").unlink()

    config = _cfg(tmp_path)
    summary = run_export(config)
    assert summary["emitted"] > 0  # other rows still made it

    emitted_ids = {
        json.loads(line)["session_id"]
        for line in config.out_path.read_text(encoding="utf-8").splitlines()
    }
    assert "aaa-python-accepted" not in emitted_ids


def test_export_emit_schema_only_writes_schema_no_records(tmp_path: Path) -> None:
    """``emit_schema_only=True`` writes only ``schema.json``; no JSONL."""
    config = _cfg(tmp_path, emit_schema_only=True)
    summary = run_export(config)

    assert (config.out_path.parent / "schema.json").exists()
    assert not config.out_path.exists()
    assert summary == {
        "total_runs_in_index": 0,
        "after_filters": 0,
        "after_stratify": 0,
        "emitted": 0,
    }
