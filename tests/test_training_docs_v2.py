"""Documentation-contract test for the corpus-v2 real-corpus train flow (issue #1081).

Pins that docs/training-launch.md and docs/runbooks/annotation-final-publish.md
state the exact real-corpus v2 build + train commands and the immutable-input
contract (verification markers, lineage, split digests, git SHAs, C5/C8
re-application, fail-closed drift). Mirrors the doc-contract pattern in
tests/test_calibration_docs.py (which itself mirrors tests/test_benchmark_docs.py,
the #991 pattern).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "docs" / "training-launch.md"
RUNBOOK = ROOT / "docs" / "runbooks" / "annotation-final-publish.md"


def test_training_launch_documents_v2_command_and_immutable_inputs() -> None:
    text = LAUNCH.read_text()
    assert "corpus build-v2" in text
    assert "daydream train --corpus-v2" in text
    assert "_SUCCESS" in text and "lineage" in text
    assert "base_sha" in text and "head_sha" in text
    assert "C5" in text and "C8" in text
    assert "fail-closed" in text
    assert "drift" in text


def test_annotation_runbook_points_to_v2_train_command() -> None:
    text = RUNBOOK.read_text()
    assert "corpus build-v2" in text
    assert "daydream train --corpus-v2" in text
    assert "_SUCCESS" in text and "lineage" in text
    assert "base_sha" in text and "head_sha" in text
    assert "C5" in text and "C8" in text
    assert "fail-closed" in text
    assert "drift" in text
