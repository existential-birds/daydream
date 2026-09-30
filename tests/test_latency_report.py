"""Per-profile comparison and lens-to-shipped attribution (issue #732, MH13)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from daydream.eval.latency_report import (
    _phase_timings,
    attribute_shipped_lens,
    build_report,
    main,
)

_MANIFEST = Path(__file__).resolve().parent / "fixtures" / "latency_profiles" / "manifest.json"
CORPUS = _MANIFEST.parent


def load_manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_MANIFEST.read_text())
    return data


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
    assert _phase_timings(real) == {"wonder": 20.0, "arbiter": 90.0, "verify": None}

    legacy = {"timing": {"phase_timings": {"arbiter": {"wall_clock_seconds": 30.0}}}}
    assert _phase_timings(legacy) == {"wonder": None, "arbiter": 30.0, "verify": None}

    assert _phase_timings({"timing": {"phase_timings": {}}}) == {
        "wonder": None,
        "arbiter": None,
        "verify": None,
    }


def test_selection_corpus_cases_carry_every_artifact_the_predicate_reads() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    cases = manifest["selection_cases"]
    assert cases, "the MH16 corpus must declare at least one selection case"
    required = {"merged-items.json", "recommendation-verdicts.json", "adjudication-provenance.json",
                "diff.patch", "fix-outcomes.json", "evaluation.json"}
    for case in cases:
        run_dir = CORPUS / "runs" / case["name"] / case["profile"]
        present = {p.name for p in run_dir.iterdir()}
        assert required <= present, f"{run_dir} is missing {sorted(required - present)}"
        verdicts = json.loads((run_dir / "recommendation-verdicts.json").read_text())["verdicts"]
        # The archived arm is the conservative one: every non-structural item has a verdict.
        assert verdicts, f"{run_dir} has no archived verdicts to compare against"
        assert case["golden_high_severity"], f"{case['name']} declares no high-severity anchor"


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


def test_selection_report_states_both_modes_and_their_gate() -> None:
    report = build_report(load_manifest())
    block = report["verify_selection"]
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
    assert block["flip_allowed"] is (
        proposed["contradictory_fixes"] == 0
        and proposed["high_severity_recall"] >= current["high_severity_recall"]
    )


def test_selection_block_is_absent_without_selection_cases() -> None:
    report = build_report({"corpus": "no-selection", "cases": []}, corpus_dir=Path("/nonexistent"))
    assert "verify_selection" not in report
    # adding the verify timing to _PHASE_TIMING_KEYS must not move the per-profile output
    assert all(
        set(profile["phase_latency_seconds"]) == {"wonder", "arbiter"}
        for profile in report["profiles"].values()
    )


def test_selection_report_reports_unreadable_case_instead_of_dropping_it(tmp_path: Path) -> None:
    manifest = {"corpus": "broken", "selection_cases": [{"name": "gone", "profile": "forensic"}]}
    report = build_report(manifest, corpus_dir=tmp_path)
    block = report["verify_selection"]
    assert block["cases"] == ["gone"]
    assert block["skipped"] and block["skipped"][0]["path"].endswith("runs/gone/forensic")
    assert block["modes"]["current"]["items"] == 0
    assert block["modes"]["current"]["calls"] == 0
    assert isinstance(block["flip_allowed"], bool)


def test_selection_report_does_not_count_lens_exemptions_as_skips(tmp_path: Path) -> None:
    """A lens exemption is never reported as a skip the mode chose to make.

    ``verify_all`` never selection-skips a non-exempt item, so its ``skipped``
    figure must stay zero even when the corpus carries a structural item (which
    the selective rule exempts rather than skips), and the selective arm's own
    skip count must not grow by the exempt count.
    """
    case = {"name": "case", "profile": "forensic", "golden_high_severity": [["api.py", 5]]}
    base_root = tmp_path / "base"
    exempt_root = tmp_path / "exempt"
    for root in (base_root, exempt_root):
        shutil.copytree(
            CORPUS / "runs" / "routine-skip" / "forensic",
            root / "runs" / "case" / "forensic",
        )
    merged_path = exempt_root / "runs" / "case" / "forensic" / "merged-items.json"
    merged = json.loads(merged_path.read_text())
    merged["items"].append(
        {
            "id": 3, "item_uid": "structure:9", "lens": "structural", "file": "api.py",
            "line": 2, "confidence": "HIGH", "severity": "low", "related_files": None,
            "description": "duplicated helper", "rationale": "why (Sources: structure-records item 9)",
            "evidence": "api.py:2 duplicated helper", "source_uids": ["structure:9"],
        }
    )
    merged_path.write_text(json.dumps(merged))
    manifest = {"corpus": "exemptions", "selection_cases": [case]}
    base = build_report(manifest, corpus_dir=base_root)["verify_selection"]
    with_exempt = build_report(manifest, corpus_dir=exempt_root)["verify_selection"]
    assert base["modes"]["current"]["skipped"] == 0
    assert with_exempt["modes"]["current"]["skipped"] == 0
    assert with_exempt["modes"]["proposed"]["skipped"] == base["modes"]["proposed"]["skipped"]
    assert with_exempt["modes"]["proposed"]["items"] == base["modes"]["proposed"]["items"]


def test_selection_gate_blocks_a_flip_when_a_skipped_item_contradicts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "counter" / "forensic"
    shutil.copytree(CORPUS / "runs" / "routine-skip" / "forensic", run_dir)
    verdicts_path = run_dir / "recommendation-verdicts.json"
    payload = json.loads(verdicts_path.read_text())
    payload["verdicts"] = [entry for entry in payload["verdicts"] if entry["issue_id"] == 1]
    payload["verdicts"][0]["verdict"] = "contradicts"
    verdicts_path.write_text(json.dumps(payload))
    manifest = {
        "corpus": "counter",
        "selection_cases": [
            {"name": "counter", "profile": "forensic", "golden_high_severity": [["api.py", 5]]}
        ],
    }
    report = build_report(manifest, corpus_dir=tmp_path)
    block = report["verify_selection"]
    assert block["modes"]["proposed"]["skipped"] == 1
    assert block["modes"]["proposed"]["contradictory_fixes"] == 1
    assert block["modes"]["proposed"]["high_severity_recall"] == 1.0
    assert block["flip_allowed"] is False
