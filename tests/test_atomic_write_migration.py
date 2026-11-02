"""Byte, mode, durability and failure-cleanup contract for the eight corpus and
benchmark artifact writers (issue #1215).

Every test enters from a production entry point and asserts what a consumer can
observe on disk -- exact bytes, permission bits, surviving temp files -- never
that a function was merely called.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from daydream.commands.corpus import _handle_corpus_command
from daydream.training.calibration import run_calibration
from daydream.training.corpus_projection.projector import build_frozen_corpus
from tests.harness.record_projection import projection_config, seed_projection_store
from tests.test_calibration import _build_fixture, _config


def _fail_all_renames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every rename fail (ENOSPC) — the failure mode the temp cleanup exists for."""

    def failing(src: object, dst: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", failing)


def test_projection_and_calibration_failures_leave_no_stray_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = seed_projection_store(tmp_path, dispositions=("accepted", "rejected"))
    out = tmp_path / "proj"
    projection = projection_config(store, tmp_path, out_dir=out)
    (tmp_path / "calibration").mkdir()
    calibration = _config(_build_fixture(tmp_path / "calibration"), tmp_path / "calibration")
    _fail_all_renames(monkeypatch)
    with pytest.raises(OSError, match="No space left"):
        build_frozen_corpus(projection)
    assert list(out.iterdir()) == []       # projector's unlink-on-failure must not be weakened
    with pytest.raises(OSError, match="No space left"):
        run_calibration(calibration)
    assert list(calibration.out_dir.iterdir()) == []   # its try/finally already unlinks both temps


class TestRecordExportPersistence:
    def test_queue_publication_private_and_fault_preserves_prior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = seed_projection_store(tmp_path)
        config = projection_config(store, tmp_path)
        state = tmp_path / "state"
        args = ["adjudicate", "build", "--store", str(store.root), "--snapshot-id", config.snapshot_id,
                "--state-dir", str(state)]
        assert _handle_corpus_command(args) == 0
        before = {path.name: path.read_bytes() for path in state.iterdir()}
        assert "queue.json" in before and "reference.json" in before
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in state.iterdir())
        _fail_all_renames(monkeypatch)
        assert _handle_corpus_command(args) == 1
        assert {path.name: path.read_bytes() for path in state.iterdir()} == before
