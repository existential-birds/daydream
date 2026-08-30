"""Tests for ``daydream corpus calibrate-reward`` CLI registration (issue #999, M1).

Drives ``cli.main`` through ``sys.argv`` (the production entrypoint),
mocking only the ``run_calibration`` seam. The validation-failure path runs
the real fail-closed gates against the checked-in tampered-digest fixture
variant and asserts the nonzero exit plus the gate message on stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from daydream import cli

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "training" / "calibration"


def _run_main(argv: list[str]) -> int:
    """Drive ``cli.main`` with ``argv`` and return its exit code."""
    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:  # main() always exits via sys.exit
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    return 0


@pytest.fixture
def fixture_corpus() -> Path:
    return FIXTURE_DIR


def test_corpus_calibrate_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixture_corpus: Path) -> None:
    called: dict[str, Any] = {}

    def _fake_run(cfg: Any) -> dict[str, Any]:
        called["cfg"] = cfg
        return {"total_records": 2, "out": str(tmp_path)}

    monkeypatch.setattr("daydream.training.calibration.run_calibration", _fake_run)
    rc = _run_main(
        [
            "corpus",
            "calibrate-reward",
            "--corpus-dir",
            str(fixture_corpus / "corpus"),
            "--gold-labels",
            str(fixture_corpus / "gold.json"),
            "--breakdowns",
            str(fixture_corpus / "breakdowns.json"),
            "--run-id",
            "cal-1",
            "--seed",
            "7",
            "--candidate",
            "w_fp=0.1,0.2,0.3",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    cfg = called["cfg"]
    assert cfg.candidates["w_fp"] == [0.1, 0.2, 0.3]
    assert cfg.run_id == "cal-1"
    assert cfg.seed == 7
    assert cfg.corpus_dir == fixture_corpus / "corpus"
    assert cfg.out_dir == tmp_path / "out"


def test_calibrate_validation_failure_exits_nonzero(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, fixture_corpus: Path
) -> None:
    # variants/digest holds a tampered corpus.jsonl against a pristine
    # SHA256SUMS manifest: the real digest gate must fire before anything else.
    rc = _run_main(
        [
            "corpus",
            "calibrate-reward",
            "--corpus-dir",
            str(fixture_corpus / "variants" / "digest"),
            "--gold-labels",
            str(fixture_corpus / "gold.json"),
            "--breakdowns",
            str(fixture_corpus / "breakdowns.json"),
            "--run-id",
            "cal-1",
            "--seed",
            "7",
            "--candidate",
            "w_fp=0.1,0.2,0.3",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1
    # print_error renders a rich panel on the shared console (stdout);
    # assert the gate message on the captured output stream.
    captured = capsys.readouterr()
    assert "digest mismatch" in captured.out + captured.err
    assert not (tmp_path / "out").exists()


def test_calibrate_unknown_candidate_flag_exits_nonzero(
    tmp_path: Path, fixture_corpus: Path
) -> None:
    rc = _run_main(
        [
            "corpus",
            "calibrate-reward",
            "--corpus-dir",
            str(fixture_corpus / "corpus"),
            "--gold-labels",
            str(fixture_corpus / "gold.json"),
            "--breakdowns",
            str(fixture_corpus / "breakdowns.json"),
            "--run-id",
            "cal-1",
            "--seed",
            "7",
            "--candidate",
            "no_equals_sign",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1
