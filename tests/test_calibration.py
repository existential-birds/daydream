"""Task-1 tests: fail-closed validation matrix (M2) for ``calibrate-reward``.

``fixture_corpus`` builds a minimal synthetic corpus-v2 bundle in ``tmp_path``;
corruption variants are injected through ``CalibrationConfig.corruptions`` so
every gate is exercised against otherwise-identical inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.training.calibration import CalibrationConfig, CalibrationError, run_calibration

AS_OF = "2026-01-01T00:00:00+00:00"
VALID_AT = "2025-12-01T00:00:00+00:00"
SALT = "cal-salt-1"
LABEL_STAMPS = {
    "labeler_policy_version": "980-policy-r1",
    "reply_classifier_version": "980-classifier-r1",
    "rubric_schema_version": "980-rubric-r2",
}
REWARD_VERSION = "2026.05.28-2"

# Corruption flags the fixture understands; these are test seams, never CLI surface.
CORRUPTION_FLAGS = frozenset(
    {
        "schema_version",
        "posterior",
        "label-version",
        "c5-repo",
        "license",
        "digest",
        "split-overlap",
        "drop-session",
    }
)


def _record(i: int) -> dict[str, Any]:
    return {
        "schema_version": "2",
        "record_id": f"rec-{i:04d}",
        "session_id": f"sess-{i:04d}",
        "repo_slug": "acme/widgets",
        "reward_version": REWARD_VERSION,
        "lineage": {
            "split": "train",
            "as_of": AS_OF,
            "valid_at": VALID_AT,
            "license_decision": "allow",
            **LABEL_STAMPS,
        },
    }


def _build_fixture(tmp_path: Path) -> Path:
    """Write corpus.jsonl + lineage.json + SHA256SUMS + gold.json + breakdowns.json."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    records = [_record(i) for i in range(4)]
    corpus_bytes = ("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n").encode()
    (corpus_dir / "corpus.jsonl").write_bytes(corpus_bytes)
    (corpus_dir / "lineage.json").write_text(
        json.dumps(
            {
                "schema_version": "corpus-v2",
                "salt": SALT,
                "holdout_rate": 0.1,
                "val_rate": 0.1,
                "as_of": AS_OF,
                "valid_at": VALID_AT,
                "content_digests": {},
            },
            sort_keys=True,
        )
        + "\n"
    )
    (corpus_dir / "SHA256SUMS").write_text(
        f"{hashlib.sha256(corpus_bytes).hexdigest()}  corpus.jsonl\n"
    )
    gold = {f"rec-{i:04d}": {"accepted": i % 2 == 0} for i in range(4)}
    (tmp_path / "gold.json").write_text(json.dumps(gold, sort_keys=True) + "\n")
    # w_fp is partially but not perfectly separable so bootstrap CIs are non-degenerate.
    w_fp = [0.55, 0.5, 0.65, 0.58]
    breakdowns = {
        f"rec-{i:04d}": {"fidelity": 0.2 + 0.1 * i, "specificity": 0.8 - 0.1 * i, "w_fp": w_fp[i]}
        for i in range(4)
    }
    (tmp_path / "breakdowns.json").write_text(json.dumps(breakdowns, sort_keys=True) + "\n")
    return tmp_path


@pytest.fixture
def fixture_corpus(tmp_path: Path) -> Path:
    return _build_fixture(tmp_path)


def _config(fixture_dir: Path, tmp_path: Path, **overrides: object) -> CalibrationConfig:
    corrupt = frozenset(k for k, v in overrides.items() if k in CORRUPTION_FLAGS and v)
    base: dict[str, Any] = dict(
        corpus_dir=fixture_dir / "corpus",
        gold_labels=fixture_dir / "gold.json",
        breakdowns=fixture_dir / "breakdowns.json",
        out_dir=tmp_path / "out",
        run_id="cal-test-1",
        seed=7,
        candidates={"w_fp": [0.1, 0.3]},
        corruptions=corrupt,
    )
    base.update({k: v for k, v in overrides.items() if k not in CORRUPTION_FLAGS})
    return CalibrationConfig(**base)


@pytest.mark.parametrize(
    "corrupt,expected",
    [
        ("schema_version", "schema_version"),  # record stamped other than "2"
        ("posterior", "valid_at"),  # valid_at posterior to as_of
        ("label-version", "unrecognized label"),  # absent/unknown version stamp
        ("c5-repo", "excluded repository"),
        ("license", "license decision"),
        ("digest", "digest mismatch"),
        ("split-overlap", "split membership overlaps"),
    ],
)
def test_validation_fails_closed(
    tmp_path: Path, fixture_corpus: Path, corrupt: str, expected: str
) -> None:
    with pytest.raises(CalibrationError, match=expected):
        run_calibration(_config(fixture_corpus, tmp_path, **{corrupt: True}))
    assert not (tmp_path / "out" / "calibration.json").exists()  # no partial artifact


def test_clean_corpus_passes_gates(tmp_path: Path, fixture_corpus: Path) -> None:
    result = run_calibration(_config(fixture_corpus, tmp_path))
    assert result["run_id"] == "cal-test-1"
    assert result["record_count"] == 4
    artifact = tmp_path / "out" / "calibration.json"
    assert artifact.exists()
    assert json.loads(artifact.read_text())["schema_version"] == "calibration-artifact-v1"


def test_statistics_are_exact_and_deterministic(fixture_corpus: Path, tmp_path: Path) -> None:
    result = run_calibration(_config(fixture_corpus, tmp_path, seed=42))
    m = result["metrics"]
    assert m["length_distribution"]["median"] == round(m["length_distribution"]["median"], 4)
    assert {"median", "iqr"} <= set(m["length_distribution"])
    assert set(m["class_balance"]) == {"accepted", "rejected"}
    assert m["per_axis_correlations"]["w_fp_axis"] <= 1.0
    lo, hi = m["auc_bootstrap_ci"]["w_fp=0.3"]
    assert lo < hi
    # same seed ⇒ identical bytes
    run_calibration(_config(fixture_corpus, tmp_path / "r2", out_dir=tmp_path / "r2", seed=42))
    assert (tmp_path / "out" / "calibration.json").read_bytes() == \
        (tmp_path / "r2" / "calibration.json").read_bytes()


def test_candidate_without_range_fails_closed(tmp_path: Path, fixture_corpus: Path) -> None:
    with pytest.raises(CalibrationError, match="w_fp"):
        run_calibration(_config(fixture_corpus, tmp_path, candidates={"w_fp": []}))


def test_unknown_candidate_axis_fails_closed(tmp_path: Path, fixture_corpus: Path) -> None:
    with pytest.raises(CalibrationError, match="w_ghost"):
        run_calibration(_config(fixture_corpus, tmp_path, candidates={"w_ghost": [0.1, 0.2]}))


def test_missing_required_field_fails_closed(tmp_path: Path, fixture_corpus: Path) -> None:
    with pytest.raises(CalibrationError, match="missing required field"):
        run_calibration(_config(fixture_corpus, tmp_path, **{"drop-session": True}))
