"""Byte, mode, durability and failure-cleanup contract for the eight corpus and
benchmark artifact writers (issue #1215).

Every test enters from a production entry point and asserts what a consumer can
observe on disk -- exact bytes, permission bits, surviving temp files -- never
that a function was merely called.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from daydream import cli
from daydream.benchmark.harbor import candidate
from daydream.json_utils import atomic_write_bytes
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.export import write_export_rows
from daydream.training.adjudication.final_bundle import build_final_bundle
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.calibration import run_calibration
from daydream.training.corpus_projection import projector
from daydream.training.corpus_projection.projector import build_frozen_corpus
from tests.test_calibration import _build_fixture, _config
from tests.test_cli_adjudicate import _seed_adjudicated, _write_sessions
from tests.test_corpus_projection import _cfg, _write_annotations_snapshot, _write_bundle
from tests.test_training_adjudication_final_bundle import seed_final_bundle_state
from tests.test_training_adjudication_materialize import _PIN, _index


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
    def test_materialize_bytes_and_umask_mode(self, tmp_path: Path) -> None:
        root = _index(tmp_path)
        out = tmp_path / "mat"
        with _umask(0o022):
            run_materialize(root, out, pin=_PIN)
        rows = [json.loads(line) for line in (out / "sessions.jsonl").read_text().splitlines()]
        assert (out / "sessions.jsonl").read_bytes() == "".join(_canonical_line(r) for r in rows).encode("utf-8")
        # preview-manifest.json is canonical JSON with NO trailing newline (M4).
        manifest = (out / "preview-manifest.json").read_bytes()
        assert not manifest.endswith(b"\n")
        assert manifest == json.dumps(json.loads(manifest), sort_keys=True,
                                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        assert sorted(p.name for p in out.iterdir()) == ["preview-manifest.json", "sessions.jsonl"]
        for name in ("sessions.jsonl", "preview-manifest.json"):
            assert stat.S_IMODE((out / name).stat().st_mode) == 0o644

    def test_export_bytes_mode_and_digest_contract(self, tmp_path: Path) -> None:
        root, state = _seed_adjudicated(tmp_path)
        out = tmp_path / "export.jsonl"
        with _umask(0o022):
            assert cli._handle_corpus_command(
                ["adjudicate", "export", "--index-root", str(root),
                 "--state-dir", str(state), "--out", str(out)]
            ) == 0
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        raw = out.read_bytes()
        assert raw == "".join(_canonical_line(r) for r in rows).encode("utf-8")
        # write_export_rows stays public, keeps its signature, and still returns the
        # sha256 of the written bytes (M8).
        assert write_export_rows(rows, tmp_path / "again.jsonl") == hashlib.sha256(raw).hexdigest()
        assert stat.S_IMODE(out.stat().st_mode) == 0o644

    def test_canonical_annotations_bytes_and_mode(self, tmp_path: Path) -> None:
        index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
        with _umask(0o022):
            run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
        rows = [json.loads(line) for line in (mat / "annotations.jsonl").read_text().splitlines()]
        assert (mat / "annotations.jsonl").read_bytes() == "".join(_canonical_line(r) for r in rows).encode("utf-8")
        assert stat.S_IMODE((mat / "annotations.jsonl").stat().st_mode) == 0o644

    def test_final_bundle_copies_are_verbatim_and_mode_is_explicit(self, tmp_path: Path) -> None:
        index_root, mat, archive_dir, _pin = seed_final_bundle_state(tmp_path)
        run_canonical_harvest(index_root, mat, archive_dir, observations_path=None)
        out = tmp_path / "final-bundle"
        with _umask(0o022):
            build_final_bundle(index_root=index_root, materialize_dir=mat,
                               archive_dir=archive_dir, out_dir=out)
        for name in ("annotations.jsonl", "sessions.jsonl", "preview-manifest.json"):
            assert (out / name).read_bytes() == (mat / name).read_bytes(), name
        assert (out / "coverage-report.json").read_bytes().endswith(b"\n")
        for name in ("annotations.jsonl", "sessions.jsonl", "label-observations.jsonl",
                     "coverage-report.json", "lineage.json", "preview-manifest.json",
                     "policy-binding.json"):
            assert stat.S_IMODE((out / name).stat().st_mode) == 0o644, name

    def test_projection_bytes_and_private_mode(self, tmp_path: Path) -> None:
        bundle_dir = _write_bundle(tmp_path)
        snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted", "rejected"])
        out = tmp_path / "proj"
        build_frozen_corpus(_cfg(out, bundle_dir, snap))
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

    def test_queue_bytes_and_private_mode(self, tmp_path: Path) -> None:
        _write_sessions(tmp_path)
        state = tmp_path / "adj"
        assert cli._handle_corpus_command(
            ["adjudicate", "build", "--index-root", str(tmp_path), "--state-dir", str(state)]
        ) == 0
        raw = (state / "queue.json").read_bytes()
        assert raw == (json.dumps(json.loads(raw.decode()), indent=2, sort_keys=True) + "\n").encode("utf-8")
        assert stat.S_IMODE((state / "queue.json").stat().st_mode) == 0o600
        assert sorted(p.name for p in state.iterdir()) == ["queue.json"]

    def test_candidate_bytes_mode_and_no_stray_temp(self, tmp_path: Path) -> None:
        artifact = candidate.build_candidate_artifact("case-abc123def456", [])
        dest = tmp_path / "logs" / "artifacts" / "review.json"
        with _umask(0o022):
            candidate.write_candidate_artifact_atomic(dest, artifact)
        # bare json.dumps: default separators, insertion order, no trailing newline.
        assert dest.read_bytes() == json.dumps(artifact).encode("utf-8")
        assert not dest.read_bytes().endswith(b"\n")
        assert stat.S_IMODE(dest.stat().st_mode) == 0o644
        assert list(dest.parent.glob("review.json*")) == [dest]

    def test_projection_failure_leaves_no_stray_temp(self, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        bundle_dir = _write_bundle(tmp_path)
        snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted", "rejected"])
        out = tmp_path / "proj"
        _fail_all_renames(monkeypatch)
        with pytest.raises(OSError, match="No space left"):
            build_frozen_corpus(_cfg(out, bundle_dir, snap))
        assert list(out.iterdir()) == []       # projector's unlink-on-failure must not be weakened

    def test_calibration_failure_leaves_no_stray_temp(self, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
        config = _config(_build_fixture(tmp_path), tmp_path)
        _fail_all_renames(monkeypatch)
        with pytest.raises(OSError, match="No space left"):
            run_calibration(config)
        assert list(config.out_dir.iterdir()) == []   # its try/finally already unlinks both temps


class TestExportKnobs:
    def test_export_calls_the_primitive_with_the_documented_knobs(self, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _instrument(monkeypatch, "daydream.training.adjudication.export")
        root, state = _seed_adjudicated(tmp_path)
        out = tmp_path / "export.jsonl"
        assert cli._handle_corpus_command(
            ["adjudicate", "export", "--index-root", str(root),
             "--state-dir", str(state), "--out", str(out)]
        ) == 0
        [(target, content, kwargs)] = calls
        assert target == out
        assert content == out.read_bytes()          # the site hands over exactly what lands on disk
        assert kwargs == {"fsync": False, "dir_fsync": False, "mode": 0o644}

    def test_export_failure_keeps_prior_bytes_and_leaves_no_temp(self, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        root, state = _seed_adjudicated(tmp_path)
        out = tmp_path / "export.jsonl"
        out.write_bytes(b"prior\n")
        _fail_all_renames(monkeypatch)
        with pytest.raises(OSError, match="No space left"):
            cli._handle_corpus_command(
                ["adjudicate", "export", "--index-root", str(root),
                 "--state-dir", str(state), "--out", str(out)]
            )
        assert out.read_bytes() == b"prior\n"
        assert list(tmp_path.glob("export.jsonl*")) == [out]
