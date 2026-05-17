"""Tests for daydream.training.export query + filter pipeline (Wave 4).

Uses the §9 fixture matrix materialized via
``tests.fixtures.training.build_archive.build_fixture_archive`` — a real
SQLite index built with the production ``upsert_run`` helper. No mocking
of SQLite, the archive layer, or the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.training.export import ExportFilters, _query_index
from tests.fixtures.training.build_archive import build_fixture_archive


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """Build the §9 fixture matrix into ``tmp_path`` and return the root."""
    build_fixture_archive(tmp_path)
    return tmp_path


@pytest.fixture
def copyleft_seeded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the C8 loader at a temp file that lists ``gnu/coreutils``.

    The committed ``schema/copyleft.txt`` is intentionally empty (R1 ships
    no known copyleft repos), so the copyleft tests have to seed the file
    themselves. Mirrors the pattern used by
    ``tests/test_training_exclusion.py::test_is_copyleft_opt_in_overrides_skip``.
    """
    copyleft_file = tmp_path / "_copyleft.txt"
    copyleft_file.write_text("gnu/coreutils\n", encoding="utf-8")
    monkeypatch.setattr("daydream.training.exclusion.COPYLEFT_PATH", copyleft_file)


def _ids(rows: list[dict]) -> list[str]:
    return [row["session_id"] for row in rows]


def test_query_excludes_c5_repos_unconditionally(archive: Path) -> None:
    rows = _query_index(archive, ExportFilters(include_all_labels=True))
    assert "ddd-on-exclusion" not in _ids(rows)


def test_query_skill_filter(archive: Path) -> None:
    unfiltered = _query_index(archive, ExportFilters())
    python_only = _query_index(archive, ExportFilters(skill="beagle-python:review-python"))

    assert len(python_only) < len(unfiltered)
    assert all(row["skill"] == "beagle-python:review-python" for row in python_only)
    # Sanity: at least one python row survives default filters.
    assert "aaa-python-accepted" in _ids(python_only)


def test_query_min_grounding_filter(archive: Path) -> None:
    rows = _query_index(archive, ExportFilters(min_grounding=0.5))
    ids = _ids(rows)
    assert "aaa-python-accepted" in ids
    assert "ccc-python-low-grounding" not in ids


def test_query_label_filter_defaults_to_accepted(archive: Path) -> None:
    rows = _query_index(archive, ExportFilters())
    assert "bbb-react-rejected" not in _ids(rows)


def test_query_include_all_labels_overrides_default(archive: Path) -> None:
    rows = _query_index(archive, ExportFilters(include_all_labels=True))
    assert "bbb-react-rejected" in _ids(rows)


def test_query_copyleft_skipped_by_default(archive: Path, copyleft_seeded: None) -> None:
    rows = _query_index(archive, ExportFilters())
    assert "eee-copyleft" not in _ids(rows)


def test_query_copyleft_opt_in(archive: Path, copyleft_seeded: None) -> None:
    rows = _query_index(
        archive,
        ExportFilters(allow_copyleft=frozenset({"gnu/coreutils"})),
    )
    assert "eee-copyleft" in _ids(rows)


def test_query_orders_by_session_id(archive: Path) -> None:
    rows = _query_index(archive, ExportFilters())
    ids = _ids(rows)
    assert ids == sorted(ids)


def test_query_attaches_stack(archive: Path) -> None:
    rows = _query_index(archive, ExportFilters(include_all_labels=True))
    by_id = {row["session_id"]: row for row in rows}
    assert by_id["aaa-python-accepted"]["stack"] == "python"
    assert by_id["bbb-react-rejected"]["stack"] == "react"
