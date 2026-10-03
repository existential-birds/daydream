"""Replay RFT against frozen base/head/diff identity. Require deterministic winners that pass breakdown
thresholds; missing identity or invalid thresholds fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

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
    cfg_a = RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "a")
    cfg_b = RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "b")
    w1 = run_rft(cfg_a)
    w2 = run_rft(cfg_b)
    assert w1.winners_path.read_bytes() == w2.winners_path.read_bytes()
    assert w1.records, "expected at least one winner from a well-formed corpus"

def test_filter_threshold_reads_breakdown(frozen_rft_inputs: Path, tmp_path: Path) -> None:
    cfg = RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "c",
        min_breakdown={"composite": 0.6, "correctness_per_finding": 0.5},
    )
    winners = run_rft(cfg)
    assert winners.records
    for w in winners.records:
        assert w.breakdown.composite is not None  # every winner went through score_trajectory's breakdown
        assert w.breakdown.correctness_per_finding
        assert sum(w.breakdown.correctness_per_finding) / len(w.breakdown.correctness_per_finding) >= 0.5

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

def test_winners_header_stamps_provenance(frozen_rft_inputs: Path, tmp_path: Path) -> None:
    result = run_rft(RftConfig(inputs=frozen_rft_inputs, seed=11, rubric_version="2026.08.29-1",
                               output_dir=tmp_path / "f", model_id="test-model"))
    payload = json.loads(result.winners_path.read_text())
    header = payload["header"]
    assert header["seed"] == 11
    assert header["rubric_version"] == "2026.08.29-1"
    assert header["model_id"] == "test-model"
    assert header["inputs_sha256"]
    ids = [w["record_id"] for w in payload["winners"]]
    assert ids == sorted(ids)

def test_sampled_findings_drive_breakdown_variance(tmp_path: Path) -> None:
    """Varying the sampled finding subset must vary the breakdown while identical samples remain
    deterministic.
    """
    rec: dict[str, object] = _record("r-vary",
        findings=[{"id": "r-vary-f1", "text": "grounded fix A", "grounded": True, "verdict": "consistent"},
            {"id": "r-vary-f2", "text": "ungrounded guess B", "grounded": False, "verdict": "contradicts"},
        ],
    )
    path = _write_corpus(tmp_path, [rec])
    out_a = run_rft(RftConfig(inputs=path, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "a"))
    out_b = run_rft(RftConfig(inputs=path, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "b"))
    assert out_a.winners_path.read_bytes() == out_b.winners_path.read_bytes()
    breakpoints = {(w.candidate_index, w.breakdown.composite, tuple(w.breakdown.correctness_per_finding or []))
                   for w in out_a.records}
    assert len(breakpoints) > 1


def test_replay_header_binds_captured_input_when_file_is_replaced(
    frozen_rft_inputs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import daydream.training.rft as rft
    from daydream.training.reward import score_trajectory

    captured = frozen_rft_inputs.read_bytes()
    score = score_trajectory
    replaced = False

    def score_after_replacement(*args: Any, **kwargs: Any) -> Any:
        nonlocal replaced
        if not replaced:
            replacement = tmp_path / "replacement.jsonl"
            replacement.write_text(json.dumps(_record("later-generation")) + "\n")
            replacement.replace(frozen_rft_inputs)
            replaced = True
        return score(*args, **kwargs)

    monkeypatch.setattr(rft, "score_trajectory", score_after_replacement)
    result = run_rft(RftConfig(
        inputs=frozen_rft_inputs, seed=11, rubric_version="current", output_dir=tmp_path / "bound",
    ))
    expected = hashlib.sha256(captured).hexdigest()
    assert result.inputs_sha256 == expected
    assert json.loads(result.winners_path.read_text())["header"]["inputs_sha256"] == expected
    assert {winner.record_id for winner in result.records} == {"r1", "r2"}
    assert hashlib.sha256(frozen_rft_inputs.read_bytes()).hexdigest() != expected
