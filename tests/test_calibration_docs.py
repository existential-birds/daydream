"""Documentation-contract test for the reward-calibration runbook (issue #999, M9).

Pins that docs/calibration.md states the exact CLI invocation and the #114
scope boundary. Mirrors the doc-contract pattern in tests/test_benchmark_docs.py.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "calibration.md"


def test_docs_state_exact_command_and_boundary() -> None:
    text = DOCS.read_text()
    assert "daydream corpus calibrate-reward" in text
    assert "issue #114" in text
    assert "choosing values" in text  # boundary statement present
    assert "--corpus-dir" in text
    assert "--seed" in text
