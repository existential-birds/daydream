"""Per-profile comparison and lens-to-shipped attribution (issue #732, MH13)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.eval.latency_report import (
    _phase_timings,
    attribute_shipped_lens,
    build_report,
    main,
)

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


def test_arbiter_phase_latency_reads_the_deep_bucket_real_runs_write() -> None:
    """MH13: real runs key the arbiter wall-clock under ``deep``, never ``arbiter``.

    Every arbiter call runs inside ``phase_scope(DaydreamPhase.DEEP,
    stage="arbiter")`` and ``compute_timing_summary`` keys ``phase_timings`` by
    the phase value alone, so the report must read the deep bucket instead of
    silently reporting an absent one as 0.0. The legacy ``arbiter`` bucket stays
    honoured so hand-authored corpora keep reporting their own number.
    """
    real = {
        "timing": {
            "phase_timings": {
                "alternatives": {"wall_clock_seconds": 20.0},
                "deep": {"wall_clock_seconds": 90.0},
            }
        }
    }
    assert _phase_timings(real) == {"wonder": 20.0, "arbiter": 90.0}

    legacy = {"timing": {"phase_timings": {"arbiter": {"wall_clock_seconds": 30.0}}}}
    assert _phase_timings(legacy) == {"wonder": None, "arbiter": 30.0}

    assert _phase_timings({"timing": {"phase_timings": {}}}) == {"wonder": None, "arbiter": None}


def test_report_separates_cold_and_warm_samples_and_names_its_corpus(tmp_path: Path) -> None:
    """MH14: a cold run and a warm loop are reported separately, against the target.

    The sample group is a property of the case, not of the profile: two cases that
    declare ``sample_group`` are aggregated apart so a warm rerun's lower wall
    clock cannot be averaged into the cold baseline. The header names the corpus,
    the sample size actually observed per group, and the stated target, so the
    report is self-describing without the docs.
    """
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "corpus": "review-runtime-sample",
        "cases": [
            {"name": "c1", "sample_group": "cold", "profiles": {"per_stack_review": [600, 620, 700]}},
            {"name": "c2", "sample_group": "warm", "profiles": {"per_stack_review": [1, 1, 1]}},
        ],
    }))
    report = build_report(json.loads(manifest.read_text()), corpus_dir=tmp_path)
    runtime = report["review_runtime"]
    assert runtime["corpus"] == "review-runtime-sample"
    assert runtime["groups"]["cold"]["n"] == 1
    assert runtime["groups"]["warm"]["n"] == 1
    header = runtime["header"]
    assert "review-runtime-sample" in header and "n=1" in header
    assert "cold" in header and "warm" in header
    assert "p50=620" in header and "p50=1" in header
    assert "Target: 5-15 min for a small follow-up fix" in header
