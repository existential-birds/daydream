"""Diagnostic dumps preserve finalized assembly bytes at the finalization seam."""

import shutil
from pathlib import Path

import pytest

from daydream.archive.errors import ArchivePublicationError
from daydream.run_config import RunConfig
from tests.test_archive import _setup_bundle, _strict_archive, _write_snapshot


def _contents(path: Path) -> dict[str, bytes]:
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def test_destination_publication_error_stays_fatal(tmp_path: Path, archive_dir: Path) -> None:
    patch = "+ghp_canaryfake123\n"
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
