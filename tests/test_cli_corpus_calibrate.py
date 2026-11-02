"""Test calibration CLI registration and real tampered-digest refusal.

Mock only run_calibration on success; invalid fixtures must exit nonzero with
the gate message on stderr.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.scripts import cli_main as _run_main

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "training" / "calibration"


def _base_argv(tmp_path: Path, *, corpus_dir: str = "corpus", candidate: str = "w_fp=0.1,0.2,0.3",) -> list[str]:
    return ["corpus", "calibrate-reward", "--corpus-dir", str(FIXTURE_DIR / corpus_dir), "--gold-labels",
        str(FIXTURE_DIR / "gold.json"), "--breakdowns", str(FIXTURE_DIR / "breakdowns.json"), "--run-id", "cal-1",
        "--seed", "7", "--candidate", candidate, "--out", str(tmp_path / "out"),
    ]

def test_calibrate_validation_failure_exits_nonzero(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    # Tamper with corpus content while retaining pristine SHA256SUMS; the digest gate must fire
    # first.
    rc = _run_main(_base_argv(tmp_path, corpus_dir="variants/digest"))
    assert rc == 1
    # Rich error panels are written to stdout.
    captured = capsys.readouterr()
    assert "digest mismatch" in captured.out + captured.err
    assert not (tmp_path / "out").exists()

def test_calibrate_unknown_candidate_flag_exits_nonzero(tmp_path: Path) -> None:
    rc = _run_main(_base_argv(tmp_path, candidate="no_equals_sign"))
    assert rc == 1
