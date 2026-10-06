"""Diagnostic dumps preserve finalized assembly bytes at the finalization seam."""

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from daydream.archive import scan
from daydream.archive.errors import ArchivePublicationError
from daydream.run_config import RunConfig
from tests.test_archive import _setup_bundle, _strict_archive, _write_snapshot


def _contents(path: Path) -> dict[str, bytes]:
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def test_dump_preserves_credentials_and_binary_without_scanning(
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "session-1"
    source, _, recorder = _setup_bundle(tmp_path, session_id)
    (source / ".daydream" / "diff.patch").write_bytes(b"+TOKEN=ghp_canaryfake123\n")
    (source / ".daydream" / "deep" / "evidence.bin").write_bytes(b"\xff\x00ghp_binarycanary")
    stage = tmp_path / "late"
    stage.mkdir()
    (stage / "unrelated.txt").write_bytes(b"prior destination")
    original = _contents(source)

    def unexpected_scan(_path: Path) -> scan.ScanResult:
        pytest.fail("raw dumps must not invoke the scanner")

    monkeypatch.setattr(scan, "scan_run_dir", unexpected_scan)
    _strict_archive(
        target=source, session_id=session_id, config=RunConfig(target=str(source), dump_artifacts=str(stage)),
        write_snapshot=_write_snapshot(recorder), dump_path=stage,
    )
    archived = _contents(archive_dir / "runs" / session_id)
    assert _contents(stage) == {**archived, "unrelated.txt": b"prior destination"}
    assert (stage / "diff.patch").read_bytes() == b"+TOKEN=ghp_canaryfake123\n"
    assert (stage / "deep" / "evidence.bin").read_bytes() == b"\xff\x00ghp_binarycanary"
    assert _contents(source) == original
    assert json.loads((stage / "manifest.json").read_text())["session_id"] == session_id
    with closing(sqlite3.connect(f"{(archive_dir / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT session_id FROM runs").fetchall() == [(session_id,)]


@pytest.mark.parametrize("patch", ["+print('safe')\n", '+SORT_KEY = "created_at"\n'])
def test_dump_retains_exact_bundle_bytes(tmp_path: Path, archive_dir: Path, patch: str) -> None:
    source, _, recorder = _setup_bundle(tmp_path)
    (source / ".daydream" / "diff.patch").write_text(patch)
    stage = tmp_path / "late"
    stage.mkdir()
    original = _contents(source)
    _strict_archive(
        target=source, session_id=recorder.session_id,
        config=RunConfig(target=str(source), dump_artifacts=str(stage)),
        write_snapshot=_write_snapshot(recorder), dump_path=stage,
    )
    assert _contents(stage) == _contents(archive_dir / "runs" / recorder.session_id)
    assert (stage / "diff.patch").read_text() == patch
    assert _contents(source) == original


@pytest.mark.parametrize("patch", ["+print('safe')\n", "+ghp_canaryfake123\n"])
def test_destination_publication_error_stays_fatal(tmp_path: Path, archive_dir: Path, patch: str) -> None:
    source, _, recorder = _setup_bundle(tmp_path)
    (source / ".daydream" / "diff.patch").write_text(patch)
    stage = tmp_path / "late"
    # A real destination filesystem conflict inside the owned private stage.
    (stage / "manifest.json" / "manifest.json").mkdir(parents=True)
    (stage / "manifest.json" / "keep.txt").write_text("prior evidence")
    original = _contents(source)
    with pytest.raises(ArchivePublicationError, match="dump publication failed") as raised:
        _strict_archive(
            target=source, session_id=recorder.session_id,
            config=RunConfig(target=str(source), dump_artifacts=str(stage)),
            write_snapshot=_write_snapshot(recorder), dump_path=stage,
        )
    assert isinstance(raised.value.__cause__, shutil.Error)
    assert _contents(stage) == {}
    assert stage.stat().st_mode & 0o777 == 0o700
    assert _contents(source) == original
    assert not (archive_dir / "runs" / recorder.session_id).exists()
    assert not (archive_dir / "index.db").exists()
    assert not list((archive_dir / "runs").glob(".*.finalizing"))
