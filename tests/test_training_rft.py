"""Replay RFT against frozen base/head/diff identity. Require deterministic winners that pass breakdown
thresholds; missing identity or invalid thresholds fail closed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from daydream.training.reward import REWARD_VERSION
from daydream.training.rft import RftConfig, run_rft


def _record(rid: str, **overrides: object) -> dict[str, object]:
    rec: dict[str, object] = {"id": rid, "repo_slug": "owner/repo", "base_sha": "a" * 40, "head_sha": "b" * 40,
        "diff": f"diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ for {rid}\n",
        "findings": [{"id": f"{rid}-f1", "text": "Fix the off-by-one in the loop bound.",
             "grounded": True, "verdict": "consistent"},
            {"id": f"{rid}-f2", "text": "Add a guard for empty input.", "grounded": True, "verdict": "consistent"},
        ], "format_valid": True, "length": 400,
    }
    rec.update(overrides)
    return rec


def _write_corpus(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "rft-inputs.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture()
def frozen_rft_inputs(tmp_path: Path) -> Path:
    return _write_corpus(tmp_path, [_record("r1"), _record("r2")])


def test_winners_byte_identical_on_rerun(frozen_rft_inputs: Path, tmp_path: Path) -> None:
    """One frozen RFT run owns threshold admission, provenance and deterministic winner output."""
    cfg_a = RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "a",
                      model_id="test-model", min_breakdown={"composite": 0.6, "correctness_per_finding": 0.5})
    cfg_b = RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "b",
                      model_id="test-model", min_breakdown={"composite": 0.6, "correctness_per_finding": 0.5})
    w1, w2 = run_rft(cfg_a), run_rft(cfg_b)
    assert w1.winners_path.read_bytes() == w2.winners_path.read_bytes()
    assert w1.records and w1.inputs_sha256
    payload = json.loads(w1.winners_path.read_text())
    header = payload["header"]
    assert header["seed"] == 11 and header["rubric_version"] == "2026.08.29-1"
    assert header["model_id"] == "test-model" and header["inputs_sha256"] == w1.inputs_sha256
    assert header["temperature"] == "0.000000"
    assert header["min_breakdown"] == {"composite": "0.600000", "correctness_per_finding": "0.500000"}
    assert [w["record_id"] for w in payload["winners"]] == sorted(w["record_id"] for w in payload["winners"])
    for winner in w1.records:
        assert winner.breakdown.composite is not None and winner.breakdown.composite >= 0.6
        correctness = winner.breakdown.correctness_per_finding
        assert correctness and sum(correctness) / len(correctness) >= 0.5


def test_shipped_threshold_example_replays_current_reward(frozen_rft_inputs: Path, tmp_path: Path,) -> None:
    example = Path(__file__).resolve().parents[1] / "rl/train/rft.toml"
    match = re.search(r"e\.g\. (\{[^\n]+\})", example.read_text())
    assert match is not None, "the reference recipe must document a winner threshold"
    spec = json.loads(match.group(1))
    winners = run_rft(RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="current",
        output_dir=tmp_path / "shipped-example", min_breakdown=spec,
    ))
    assert winners.records
    assert all(w.breakdown.composite is not None and w.breakdown.composite >= spec["composite"]
               for w in winners.records)

def test_spec_axes_match_score_trajectory_breakdown_fields(frozen_rft_inputs: Path, tmp_path: Path) -> None:
    # Filter axes must name real RewardBreakdown fields: correctness_per_finding is valid;
    # correctness is not.
    with pytest.raises(TypeError, match="unknown axis"):
        RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="v", output_dir=tmp_path / "axis-bogus",
            min_breakdown={"correctness": 0.5},
        )
    cfg = RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="v", output_dir=tmp_path / "axis-real",
        min_breakdown={"correctness_per_finding": 0.5},
    )
    winners = run_rft(cfg)
    for w in winners.records:
        assert w.breakdown.correctness_per_finding is not None
        assert min(w.breakdown.correctness_per_finding) >= 0.5

def test_scalar_threshold_is_rejected(frozen_rft_inputs: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="min_breakdown"):
        RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="v", output_dir=tmp_path / "d",
            min_breakdown=0.6,  # type: ignore[arg-type]
        )

def test_missing_identity_fails_closed_naming_the_record(tmp_path: Path) -> None:
    path = _write_corpus(tmp_path, [_record("ok"), _record("broken", base_sha="")])
    with pytest.raises(ValueError, match="broken"):
        run_rft(RftConfig(inputs=path, seed=11, rubric_version="v", output_dir=tmp_path / "e"))


def test_sampled_findings_drive_breakdown_variance(tmp_path: Path) -> None:
    """Varying the sampled finding subset must vary the breakdown while identical samples remain
    deterministic.
    """
    rec: dict[str, object] = _record("r-vary",
        findings=[{"id": "r-vary-f1", "text": "grounded fix A", "grounded": True, "verdict": "consistent"},
            {"id": "r-vary-f2", "text": "ungrounded guess B", "grounded": False, "verdict": "contradicts"},
        ],
    )
    path = _write_corpus(tmp_path, [rec,
        _record("r-gold", length=4000, findings=[{"verdict": "consistent"}, {"verdict": "uncertain"}]),
        _record("r-invalid-format", findings=[], verifier_verdicts=None, format_valid=False, length=50),
        _record("r-missing-credit", findings=[], verifier_verdicts=None, length=None)])
    out_a = run_rft(RftConfig(inputs=path, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "a"))
    out_b = run_rft(RftConfig(inputs=path, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "b"))
    assert out_a.winners_path.read_bytes() == out_b.winners_path.read_bytes()
    assert {winner.record_id for winner in out_a.records} == {"r-vary", "r-gold", "r-invalid-format"}
    full = {winner.record_id: winner.breakdown for winner in out_a.records if winner.candidate_index == 0}
    assert full["r-gold"].correctness_per_finding == [1.0, 0.5]
    assert full["r-gold"].composite == 0.7 and full["r-gold"].reward_version == REWARD_VERSION
    assert full["r-invalid-format"].composite == 0.0 and full["r-invalid-format"].format_valid is False
    breakpoints = {(w.candidate_index, w.breakdown.composite, tuple(w.breakdown.correctness_per_finding or []))
                   for w in out_a.records if w.record_id == "r-vary"}
    assert len(breakpoints) > 1
