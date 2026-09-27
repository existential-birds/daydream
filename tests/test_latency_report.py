"""Per-profile comparison and lens-to-shipped attribution (issue #732, MH13)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.eval.latency_report import attribute_shipped_lens, build_report, main

_MANIFEST = Path(__file__).resolve().parent / "fixtures" / "latency_profiles" / "manifest.json"


def test_attribution_counts_shipped_items_per_lens_and_reports_coverage() -> None:
    items = [
        {"file": "api.py", "line": 3, "severity": "high", "lens": "per-stack",
         "rationale": "why (Sources: python-records item 6)"},
        {"file": "api.py", "line": 9, "severity": "medium", "lens": "wonder",
         "rationale": "why, no citation"},
        {"file": "App.tsx", "line": 2, "severity": "medium", "lens": "cross-stack",
         "rationale": "why (Sources: alternatives item 4, react-records item 1)"},
    ]
    attribution = attribute_shipped_lens(items)
    assert attribution["by_lens"] == {"per-stack": 1, "wonder": 1, "cross-stack": 1, "structural": 0}
    assert attribution["citations_present"] == 2
    assert attribution["citation_coverage"] == round(2 / 3, 4)


def test_report_is_complete_per_profile_from_the_committed_corpus() -> None:
    report = build_report(json.loads(_MANIFEST.read_text()))
    assert set(report["profiles"]) == {"fast", "balanced", "forensic"}
    balanced = report["profiles"]["balanced"]
    assert balanced["phase_latency_seconds"]["wonder"]["p50"] >= 0
    assert balanced["phase_latency_seconds"]["arbiter"]["p90"] >= balanced["phase_latency_seconds"]["arbiter"]["p50"]
    assert 0.0 <= balanced["high_severity_recall"] <= 1.0
    assert 0.0 <= balanced["false_positive_rate"] <= 1.0
    assert balanced["contested"]["kept"] >= 0
    assert balanced["shipped_by_lens"]["wonder"] >= 1
    assert report["citations"]["coverage"] > 0
    assert report["calibration"]["surface_signals"]           # A3: the report states its signal lists


def test_main_emits_the_report_from_a_clean_checkout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--corpus", str(_MANIFEST)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profiles"]["fast"]["runs"] == 1
