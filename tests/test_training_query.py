"""Tests for daydream.training.corpus query + filter pipeline.

Uses the §9 fixture matrix materialized via
``tests.fixtures.training.build_archive.build_fixture_archive`` — a real
SQLite index built with the production ``upsert_run`` helper plus one silver
annotation per session. No mocking of SQLite, the archive layer, or the
filesystem.

``_query_index`` applies only the label-independent SQL filters (status, C5
exclusion, skill, repos, min_grounding) and the post-query C8 copyleft skip.
Label admission (C9 accepted-only / min-reward) moved into
``run_build_corpus`` because the label now comes from the ``as_of``-pinned
silver annotation, not the denormalized ``runs.outcome_labels`` cache — so the
label tests assert against the projection output, not ``_query_index``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.training.corpus import BuildCorpusConfig, CorpusFilters, _query_index, run_build_corpus
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


def _emitted_ids(out_path: Path) -> list[str]:
    if not out_path.exists():
        return []
    return [json.loads(line)["session_id"] for line in out_path.read_text(encoding="utf-8").splitlines()]


def test_query_excludes_c5_repos_unconditionally(archive: Path) -> None:
    rows = _query_index(archive, CorpusFilters(include_all_labels=True))
    assert "ddd-on-exclusion" not in _ids(rows)


def test_query_skill_filter(archive: Path) -> None:
    unfiltered = _query_index(archive, CorpusFilters(include_all_labels=True))
    python_only = _query_index(
        archive, CorpusFilters(skill="beagle-python:review-python", include_all_labels=True)
    )

    assert len(python_only) < len(unfiltered)
    assert all(row["skill"] == "beagle-python:review-python" for row in python_only)
    # Sanity: at least one python row survives the SQL filters.
    assert "aaa-python-accepted" in _ids(python_only)


def test_query_min_grounding_filter(archive: Path) -> None:
    rows = _query_index(archive, CorpusFilters(min_grounding=0.5, include_all_labels=True))
    ids = _ids(rows)
    assert "aaa-python-accepted" in ids
    assert "ccc-python-low-grounding" not in ids


def test_label_filter_defaults_to_accepted(archive: Path, tmp_path: Path) -> None:
    """Accepted-only is the C9 default: the rejected row is not emitted."""
    out = tmp_path / "out.jsonl"
    run_build_corpus(BuildCorpusConfig(out_path=out, archive_dir=archive, filters=CorpusFilters()))
    assert "bbb-react-rejected" not in _emitted_ids(out)


def test_include_all_labels_overrides_default(archive: Path, tmp_path: Path) -> None:
    """``include_all_labels`` admits the rejected row into the corpus."""
    out = tmp_path / "out.jsonl"
    run_build_corpus(
        BuildCorpusConfig(out_path=out, archive_dir=archive, filters=CorpusFilters(include_all_labels=True))
    )
    assert "bbb-react-rejected" in _emitted_ids(out)


def test_query_copyleft_skipped_by_default(archive: Path, copyleft_seeded: None) -> None:
    rows = _query_index(archive, CorpusFilters(include_all_labels=True))
    assert "eee-copyleft" not in _ids(rows)


def test_query_copyleft_opt_in(archive: Path, copyleft_seeded: None) -> None:
    rows = _query_index(
        archive,
        CorpusFilters(allow_copyleft=frozenset({"gnu/coreutils"}), include_all_labels=True),
    )
    assert "eee-copyleft" in _ids(rows)


def test_query_orders_by_session_id(archive: Path) -> None:
    rows = _query_index(archive, CorpusFilters(include_all_labels=True))
    ids = _ids(rows)
    assert ids == sorted(ids)


def test_query_attaches_stack(archive: Path) -> None:
    rows = _query_index(archive, CorpusFilters(include_all_labels=True))
    by_id = {row["session_id"]: row for row in rows}
    assert by_id["aaa-python-accepted"]["stack"] == "python"
    assert by_id["bbb-react-rejected"]["stack"] == "react"


def test_query_warns_on_unknown_skill(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A row whose skill isn't in REVIEW_SKILLS triggers a one-time warning."""
    from daydream.archive.index import upsert_run
    from daydream.archive.manifest import Manifest

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    session_dir = runs_dir / "zzz-mystery"
    session_dir.mkdir()
    manifest = Manifest(
        session_id="zzz-mystery",
        archived_at="2026-05-17T00:00:00+00:00",
        status="complete",
        skill="beagle-zig:review-zig",
        repo_slug="someorg/zig-app",
        branch="feat/x",
        base_branch="main",
        head_sha="abc",
        grounding_rate=0.9,
        outcome_labels=json.dumps(["accepted"]),
        archive_path=str(session_dir),
    )
    upsert_run(tmp_path, manifest)

    rows = _query_index(tmp_path, CorpusFilters(include_all_labels=True))
    captured = capsys.readouterr()

    assert any(row["session_id"] == "zzz-mystery" and row["stack"] is None for row in rows)
    assert "beagle-zig:review-zig" in captured.out
