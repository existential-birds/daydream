"""Stage-2 deterministic RFT (M11, M12): offline replay, breakdown-filtered
byte-identical winners.

The fixture freezes base/head/diff identity (M16: tasks are rebuilt from the
record's frozen inputs, never live repo state) in a temp JSONL corpus and the
tests assert observable outcomes: byte-identical winners on rerun, every
winner passing through ``score_trajectory``'s breakdown, and fail-closed
propagation on missing identity / scalar thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from daydream.training.rft import RftConfig, run_rft


@dataclass(frozen=True)
class FrozenRftInputs:
    """A frozen replay corpus plus the identity fields the records carry."""

    path: Path
    base_sha: str
    head_sha: str
    diff: str


def _record(rid: str, **overrides: object) -> dict[str, object]:
    rec: dict[str, object] = {
        "id": rid,
        "repo_slug": "owner/repo",
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "diff": f"diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ for {rid}\n",
        "findings": [
            {"id": f"{rid}-f1", "text": "Fix the off-by-one in the loop bound.",
             "grounded": True, "verdict": "consistent"},
            {"id": f"{rid}-f2", "text": "Add a guard for empty input.",
             "grounded": True, "verdict": "consistent"},
        ],
        "format_valid": True,
        "length": 400,
    }
    rec.update(overrides)
    return rec


def _write_corpus(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "rft-inputs.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture()
def frozen_rft_inputs(tmp_path: Path) -> FrozenRftInputs:
    return FrozenRftInputs(
        path=_write_corpus(tmp_path, [_record("r1"), _record("r2")]),
        base_sha="a" * 40,
        head_sha="b" * 40,
        diff="diff --git a/f.py b/f.py\n",
    )


def test_winners_byte_identical_on_rerun(frozen_rft_inputs: FrozenRftInputs, tmp_path: Path) -> None:
    cfg_a = RftConfig(inputs=frozen_rft_inputs.path, seed=11, rubric_version="2026.08.29-1",
                      output_dir=tmp_path / "a")
    cfg_b = RftConfig(inputs=frozen_rft_inputs.path, seed=11, rubric_version="2026.08.29-1",
                      output_dir=tmp_path / "b")
    w1 = run_rft(cfg_a)
    w2 = run_rft(cfg_b)
    # AC6: byte-identical winners given the same inputs/model/seeds/rubric version.
    assert w1.winners_path.read_bytes() == w2.winners_path.read_bytes()
    assert w1.records, "expected at least one winner from a well-formed corpus"


def test_filter_threshold_reads_breakdown(frozen_rft_inputs: FrozenRftInputs, tmp_path: Path) -> None:
    cfg = RftConfig(
        inputs=frozen_rft_inputs.path,
        seed=11,
        rubric_version="2026.08.29-1",
        output_dir=tmp_path / "c",
        min_breakdown={"composite": 0.6, "grounding": 0.5},
    )
    winners = run_rft(cfg)
    assert winners.records
    for w in winners.records:
        assert w.breakdown.composite is not None  # every winner went through score_trajectory's breakdown
        assert w.breakdown.grounding is not None and w.breakdown.grounding >= 0.5


def test_spec_axes_match_score_trajectory_breakdown_fields(
    frozen_rft_inputs: FrozenRftInputs, tmp_path: Path
) -> None:
    """Stage-boundary contract: allowed spec axes are the breakdown's actual attribute names."""
    # "correctness" is not a field of score_trajectory's RewardBreakdown — the real
    # axis is correctness_per_finding. A spec naming a non-axis must be rejected at
    # config time (fail closed), and the real axis name must filter.
    with pytest.raises(TypeError, match="unknown axis"):
        RftConfig(
            inputs=frozen_rft_inputs.path,
            seed=11,
            rubric_version="v",
            output_dir=tmp_path / "axis-bogus",
            min_breakdown={"correctness": 0.5},
        )
    cfg = RftConfig(
        inputs=frozen_rft_inputs.path,
        seed=11,
        rubric_version="v",
        output_dir=tmp_path / "axis-real",
        min_breakdown={"correctness_per_finding": 0.5},
    )
    winners = run_rft(cfg)
    for w in winners.records:
        assert w.breakdown.correctness_per_finding is not None
        assert min(w.breakdown.correctness_per_finding) >= 0.5


def test_scalar_threshold_is_rejected(frozen_rft_inputs: FrozenRftInputs, tmp_path: Path) -> None:
    # M12: the filter threshold names axes, never a bare scalar.
    with pytest.raises(TypeError, match="min_breakdown"):
        RftConfig(
            inputs=frozen_rft_inputs.path,
            seed=11,
            rubric_version="v",
            output_dir=tmp_path / "d",
            min_breakdown=0.6,  # type: ignore[arg-type]
        )


def test_missing_identity_fails_closed_naming_the_record(tmp_path: Path) -> None:
    path = _write_corpus(tmp_path, [_record("ok"), _record("broken", base_sha="")])
    with pytest.raises(ValueError, match="broken"):
        run_rft(RftConfig(inputs=path, seed=11, rubric_version="v", output_dir=tmp_path / "e"))


def test_winners_header_stamps_provenance(frozen_rft_inputs: FrozenRftInputs, tmp_path: Path) -> None:
    result = run_rft(RftConfig(inputs=frozen_rft_inputs.path, seed=11, rubric_version="2026.08.29-1",
                               output_dir=tmp_path / "f", model_id="test-model"))
    payload = json.loads(result.winners_path.read_text())
    header = payload["header"]
    assert header["seed"] == 11
    assert header["rubric_version"] == "2026.08.29-1"
    assert header["model_id"] == "test-model"
    assert header["inputs_sha256"]
    ids = [w["record_id"] for w in payload["winners"]]
    assert ids == sorted(ids)


def test_rft_toml_carries_sampling() -> None:
    import tomllib

    cfg = tomllib.loads((Path(__file__).parents[1] / "rl" / "train" / "rft.toml").read_text())
    # Deterministic replay: temperature pinned to 0, explicit seeds.
    assert cfg["sampling"]["temperature"] == 0.0
    assert "seed" in cfg["sampling"]


def test_sampled_findings_drive_breakdown_variance(tmp_path: Path) -> None:
    """Issue 6: scoring inputs derive from the sampled findings subset, so candidates
    that differ only in their findings score differently (breakdown filter can prefer
    one sampled completion). Determinism on identical inputs is preserved."""
    rec: dict[str, object] = _record(
        "r-vary",
        findings=[
            {"id": "r-vary-f1", "text": "grounded fix A", "grounded": True, "verdict": "consistent"},
            {"id": "r-vary-f2", "text": "ungrounded guess B", "grounded": False, "verdict": "contradicts"},
        ],
    )
    path = _write_corpus(tmp_path, [rec])
    out_a = run_rft(RftConfig(inputs=path, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "a"))
    out_b = run_rft(RftConfig(inputs=path, seed=11, rubric_version="2026.08.29-1", output_dir=tmp_path / "b"))
    # Byte-identical on rerun (M11) while the per-candidate breakdowns vary.
    assert out_a.winners_path.read_bytes() == out_b.winners_path.read_bytes()
    breakpoints = {(w.candidate_index, w.breakdown.grounding, tuple(w.breakdown.correctness_per_finding or []))
                   for w in out_a.records}
    # With mixed finding signals, sampled subsets yield distinct breakdowns, not
    # byte-identical duplicates differing only in candidate_index.
    assert len(breakpoints) > 1
