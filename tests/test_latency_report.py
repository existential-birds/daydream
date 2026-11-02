"""Per-profile comparison and lens-to-shipped attribution (issue #732, MH13)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from daydream.eval.latency_report import (
    build_report,
    main,
)

_MANIFEST = Path(__file__).resolve().parent / "fixtures" / "latency_profiles" / "manifest.json"
CORPUS = _MANIFEST.parent


def load_manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_MANIFEST.read_text())
    return data

def test_main_emits_the_report_from_a_clean_checkout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--corpus", str(_MANIFEST)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["profiles"]["fast"]["runs"] == 1
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
    block = build_report(load_manifest())["verify_selection"]
    # Every declared selection case carries the artifacts the predicate reads.
    assert block["cases"] and not block["skipped"]
    assert set(block["modes"]) == {"current", "proposed"}
    current, proposed = block["modes"]["current"], block["modes"]["proposed"]
    assert current["items"] >= proposed["items"]  # the optimised mode verifies no more
    assert proposed["calls"] in (0, 1) and current["calls"] in (0, 1)
    assert set(current["latency_seconds"]) == {"p50", "p90"}
    # single-arm corpus: labelled, never passed off as measured
    assert proposed["latency_seconds_projected"] is True
    assert proposed["high_severity_recall"] == 1.0  # the recall anchor is selected in every case
    assert proposed["contradictory_fixes"] == 0  # no skipped item archived as contradicts/uncertain
    assert isinstance(proposed["reverted_and_failed_fixes"], int)
    assert block["flip_allowed"] is (proposed["contradictory_fixes"] == 0
        and proposed["high_severity_recall"] >= current["high_severity_recall"]
    )


def test_report_separates_cold_and_warm_samples_and_names_its_corpus(tmp_path: Path) -> None:
    """Group samples by case so warm reruns cannot lower a cold baseline.
    Keep corpus, observed sample size and stated target in the report."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"corpus": "review-runtime-sample",
        "cases": [{"name": "c1", "sample_group": "cold", "profiles": {"per_stack_review": [600, 620, 700]}},
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


def test_selection_gate_reports_unreadable_cases_and_blocks_contradicted_flips(tmp_path: Path) -> None:
    manifest: dict[str, Any] = {"corpus": "broken", "selection_cases": [{"name": "gone", "profile": "forensic"}]}
    report = build_report(manifest, corpus_dir=tmp_path)
    block = report["verify_selection"]
    assert block["cases"] == ["gone"]
    assert block["skipped"] and block["skipped"][0]["path"].endswith("runs/gone/forensic")
    assert block["modes"]["current"]["items"] == 0
    assert block["modes"]["current"]["calls"] == 0
    assert isinstance(block["flip_allowed"], bool)
    run_dir = tmp_path / "runs" / "counter" / "forensic"
    shutil.copytree(CORPUS / "runs" / "routine-skip" / "forensic", run_dir)
    verdicts_path = run_dir / "recommendation-verdicts.json"
    payload = json.loads(verdicts_path.read_text())
    payload["verdicts"] = [entry for entry in payload["verdicts"] if entry["issue_id"] == 1]
    payload["verdicts"][0]["verdict"] = "contradicts"
    verdicts_path.write_text(json.dumps(payload))
    manifest = {"corpus": "counter",
        "selection_cases": [{"name": "counter", "profile": "forensic", "golden_high_severity": [["api.py", 5]]}],
    }
    report = build_report(manifest, corpus_dir=tmp_path)
    block = report["verify_selection"]
    assert block["modes"]["proposed"]["skipped"] == 1
    assert block["modes"]["proposed"]["contradictory_fixes"] == 1
    assert block["modes"]["proposed"]["high_severity_recall"] == 1.0
    assert block["flip_allowed"] is False
