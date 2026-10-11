"""Byte, mode, durability and failure-cleanup contract for the eight corpus and
benchmark artifact writers (issue #1215).

Every test enters from a production entry point and asserts what a consumer can
observe on disk -- exact bytes, permission bits, surviving temp files -- never
that a function was merely called.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.harbor import candidate
from daydream.commands.corpus import _handle_corpus_command
from daydream.json_utils import atomic_write_bytes, atomic_write_pair
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.calibration import run_calibration
from daydream.training.corpus_projection import projector
from daydream.training.corpus_projection.projector import build_frozen_corpus
from tests.harness.record_projection import projection_config, seed_projection_store
from tests.test_calibration import _build_fixture, _config


@contextmanager
def _umask(value: int) -> Iterator[None]:
    """Pin the process umask so a umask-derived mode is characterized, not inherited."""
    prior = os.umask(value)
    try:
        yield
    finally:
        os.umask(prior)


def _canonical_line(row: dict[str, object]) -> str:
    """The repo's canonical JSONL row: sorted keys, compact separators, non-ASCII kept."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _fail_all_renames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every rename fail (ENOSPC) — the failure mode the temp cleanup exists for."""

    def failing(src: object, dst: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", failing)


def _instrument(monkeypatch: pytest.MonkeyPatch, module: str) -> list[tuple[Path, bytes, dict[str, Any]]]:
    """Record the primitive's calls at *module*'s binding, delegating to the real one.

    Not a mock: the real ``json_utils.atomic_write_bytes`` runs, so the file, its mode
    and its temp-cleanup behaviour are all genuine. Only the kwargs each site chose are
    captured -- which is exactly the contract this migration is about.
    """
    calls: list[tuple[Path, bytes, dict[str, Any]]] = []
    real = atomic_write_bytes

    def spy(path: Path, content: bytes, **kwargs: Any) -> None:
        calls.append((Path(path), content, kwargs))
        real(path, content, **kwargs)

    monkeypatch.setattr(f"{module}.atomic_write_bytes", spy)
    return calls

class TestWriterCharacterization:




    def test_projection_bytes_and_private_mode(self, tmp_path: Path) -> None:
        store = seed_projection_store(tmp_path, dispositions=("accepted", "rejected"))
        out = tmp_path / "proj"
        build_frozen_corpus(projection_config(store, tmp_path, out_dir=out))
        records = [json.loads(line) for line in (out / "corpus.jsonl").read_text().splitlines()]
        assert (out / "corpus.jsonl").read_bytes() == "".join(_canonical_line(r) for r in records).encode("utf-8")
        assert (out / "_SUCCESS").read_bytes() == b"ok\n"
        # schema.json is a text-mode read of the packaged schema (CRLF would already be
        # normalized today); the migration must not switch it to a read_bytes() copy.
        schema_src = Path(projector.__file__).parent.parent / "schema" / "record-schema.json"
        assert (out / "schema.json").read_bytes() == schema_src.read_text(encoding="utf-8").encode("utf-8")
        assert (out / "lineage.json").read_bytes().endswith(b"\n")
        for path in out.iterdir():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name

    def test_calibration_two_file_publish_bytes_and_mode(self, tmp_path: Path) -> None:
        fixture_dir = _build_fixture(tmp_path)
        config = _config(fixture_dir, tmp_path)
        with _umask(0o022):
            run_calibration(config)
        raw = (config.out_dir / "calibration.json").read_bytes()
        assert raw == (json.dumps(json.loads(raw.decode()), sort_keys=True, indent=2) + "\n").encode("utf-8")
        assert (config.out_dir / "report.md").read_text(encoding="utf-8").strip()
        assert sorted(p.name for p in config.out_dir.iterdir()) == ["calibration.json", "report.md"]
        for name in ("report.md", "calibration.json"):
            assert stat.S_IMODE((config.out_dir / name).stat().st_mode) == 0o644


    def test_candidate_bytes_mode_and_no_stray_temp(self, tmp_path: Path) -> None:
        artifact = candidate.build_candidate_artifact("case-abc123def456", [])
        dest = tmp_path / "logs" / "artifacts" / "review.json"
        with _umask(0o022):
            candidate.write_candidate_artifact_atomic(dest, artifact)
        # bare json.dumps: default separators, insertion order, no trailing newline.
        assert dest.read_bytes() == json.dumps(artifact).encode("utf-8")
        assert not dest.read_bytes().endswith(b"\n")
        assert stat.S_IMODE(dest.stat().st_mode) == 0o644
        assert list(dest.parent.glob("*.tmp")) == []

    def test_projection_failure_leaves_no_stray_temp(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        store = seed_projection_store(tmp_path, dispositions=("accepted", "rejected"))
        out = tmp_path / "proj"
        config = projection_config(store, tmp_path, out_dir=out)
        _fail_all_renames(monkeypatch)
        with pytest.raises(OSError, match="No space left"):
            build_frozen_corpus(config)
        assert list(out.iterdir()) == []       # projector's unlink-on-failure must not be weakened

    def test_calibration_failure_leaves_no_stray_temp(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(_build_fixture(tmp_path), tmp_path)
        _fail_all_renames(monkeypatch)
        with pytest.raises(OSError, match="No space left"):
            run_calibration(config)
        assert list(config.out_dir.iterdir()) == []   # its try/finally already unlinks both temps





class TestProjectorKnobs:
    def test_projection_routes_every_member_through_the_primitive(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = seed_projection_store(tmp_path, dispositions=("accepted", "rejected"))
        out = tmp_path / "proj"
        calls = _instrument(monkeypatch, "daydream.training.corpus_projection.projector")
        build_frozen_corpus(projection_config(store, tmp_path, out_dir=out))
        assert {target.name for target, _c, _k in calls} >= {"corpus.jsonl", "adjudication-report.json", "schema.json",
            "lineage.json", "_SUCCESS",
        }
        # mode=None keeps mkstemp's 0600: this site must NOT be widened to the
        # umask-derived 0644 the other seven get.
        assert all(kwargs == {"fsync": False, "dir_fsync": False, "mode": None} for _t, _c, kwargs in calls)
        content_by_name = {target.name: content for target, content, _k in calls}
        assert content_by_name["_SUCCESS"] == b"ok\n"
        for target, content, _kwargs in calls:
            assert target.read_bytes() == content

class TestCalibrationKnobs:
    def test_calibration_calls_the_primitive_report_first(self, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(_build_fixture(tmp_path), tmp_path)
        calls: list[tuple[Path, bytes, dict[str, Any]]] = []
        real = atomic_write_pair
        def spy(first: tuple[Path, bytes], second: tuple[Path, bytes], **kwargs: Any) -> None:
            calls.append((Path(first[0]), first[1], kwargs))
            calls.append((Path(second[0]), second[1], kwargs))
            real(first, second, **kwargs)
        monkeypatch.setattr("daydream.training.calibration.atomic_write_pair", spy)
        with _umask(0o022):
            run_calibration(config)
        # M8: report.md is published before calibration.json, and the pair goes
        # through the primitive with the same explicit knobs.
        assert [target.name for target, _c, _k in calls] == ["report.md", "calibration.json"]
        assert all(kwargs == {"fsync": False, "dir_fsync": False, "mode": 0o644} for _t, _c, kwargs in calls)
        for target, content, _kwargs in calls:
            assert target.read_bytes() == content
        assert sorted(p.name for p in config.out_dir.iterdir()) == ["calibration.json", "report.md"]

class TestCandidateKnobs:
    def test_candidate_calls_the_primitive_with_the_documented_knobs(self, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        artifact = candidate.build_candidate_artifact("case-abc123def456", [])
        dest = tmp_path / "review.json"
        calls = _instrument(monkeypatch, "daydream.benchmark.harbor.candidate")
        with _umask(0o077):
            candidate.write_candidate_artifact_atomic(dest, artifact)
        [(target, content, kwargs)] = calls
        assert target == dest
        assert content == json.dumps(artifact).encode("utf-8")
        assert kwargs == {"fsync": False, "dir_fsync": False, "mode": 0o600}

    def test_candidate_failure_stays_typed_and_leaves_no_temp(self, tmp_path: Path) -> None:
        # Isolate the destination so the autouse archive_dir fixture's tmp_path/archive
        # does not pollute the "no temp survives" observation.
        out = tmp_path / "out"
        out.mkdir()
        dest = out / "existing-dir"
        dest.mkdir()                                   # a directory -> the rename cannot land
        with pytest.raises(candidate.CandidateError) as failure:
            candidate.write_candidate_artifact_atomic(dest, {"schema_version": 1, "findings": []})
        assert failure.value.kind == "write_failure"
        assert isinstance(failure.value.__cause__, OSError)
        assert list(out.iterdir()) == [dest]           # no uuid temp survives the failure



class TestRecordExportPersistence:
    def test_materialize_bytes_private_modes_and_fault_preserve_prior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = seed_projection_store(tmp_path, dispositions=("accepted", "rejected"))
        config = projection_config(store, tmp_path)
        out = tmp_path / "mat"
        args = ["adjudicate", "materialize", "--store", str(store.root), "--snapshot-id", config.snapshot_id,
                "--out-dir", str(out)]
        assert _handle_corpus_command(args) == 0
        rows = [json.loads(line) for line in (out / "annotations.jsonl").read_text().splitlines()]
        assert (out / "annotations.jsonl").read_bytes() == "".join(_canonical_line(r) for r in rows).encode()
        before = {path.name: path.read_bytes() for path in out.iterdir()}
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in out.iterdir())
        _fail_all_renames(monkeypatch)
        assert _handle_corpus_command(args) == 1
        assert {path.name: path.read_bytes() for path in out.iterdir()} == before

    def test_materialize_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        store = seed_projection_store(tmp_path)
        config = projection_config(store, tmp_path)
        out = tmp_path / "mat"
        run_materialize(store.root, config.snapshot_id, out, dry_run=True)
        assert not out.exists()

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
