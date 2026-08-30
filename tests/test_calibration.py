"""Task-1 tests: fail-closed validation matrix (M2) for ``calibrate-reward``.

``fixture_corpus`` builds a minimal synthetic corpus-v2 bundle in ``tmp_path``;
corruption variants are injected through ``CalibrationConfig.corruptions`` so
every gate is exercised against otherwise-identical inputs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
        f"rec-{i:04d}": {
            "fidelity": 0.2 + 0.1 * i,
            "specificity": 0.8 - 0.1 * i,
            "correctness": 0.4 + 0.05 * i,
            "grounding": 0.7 - 0.05 * i,
            "w_fp": w_fp[i],
        }
        for i in range(4)
    }
    (tmp_path / "breakdowns.json").write_text(json.dumps(breakdowns, sort_keys=True) + "\n")
    # Stage-0 score files: aligned (one entry per record) and misaligned (unknown record_id).
    scores = {
        f"rec-{i:04d}": {"score": 0.3 + 0.1 * i, "model_digest": "sha256:stage0-model-1"}
        for i in range(4)
    }
    (tmp_path / "stage0-scores-aligned.json").write_text(json.dumps(scores, sort_keys=True) + "\n")
    misaligned = dict(scores)
    misaligned["rec-ghost"] = {"score": 0.9, "model_digest": "sha256:stage0-model-1"}
    (tmp_path / "stage0-scores-misaligned.json").write_text(
        json.dumps(misaligned, sort_keys=True) + "\n"
    )
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


# --- Task 3 (M4): stage-0 marginal analysis + explicit unavailable marker ---


def test_stage0_scores_report_marginal_value(fixture_corpus: Path, tmp_path: Path) -> None:
    scores = fixture_corpus / "stage0-scores-aligned.json"  # keyed by record_id + model_digest
    result = run_calibration(_config(fixture_corpus, tmp_path, stage0_scores=scores))
    s0 = result["stage0_analysis"]
    assert s0["status"] == "ok"
    assert set(s0["marginal_value_per_axis"]) >= {"correctness", "grounding"}


def test_stage0_absent_is_explicit_unavailable(fixture_corpus: Path, tmp_path: Path) -> None:
    result = run_calibration(_config(fixture_corpus, tmp_path))
    assert result["stage0_analysis"] == {"status": "unavailable"}


def test_stage0_misaligned_records_are_refused(fixture_corpus: Path, tmp_path: Path) -> None:
    scores = fixture_corpus / "stage0-scores-misaligned.json"
    with pytest.raises(CalibrationError, match="stage-0 score.*no matching record"):
        run_calibration(_config(fixture_corpus, tmp_path, stage0_scores=scores))


# --- Task 4 (M8): committed synthetic fixture under tests/fixtures/training ---

COMMITTED_FIXTURE = Path(__file__).parent / "fixtures" / "training" / "calibration"
#: Verbatim from daydream/training/schema/exclusion.txt (C5 always-excluded list).
C5_SLUG = "getsentry/sentry"


@pytest.fixture
def committed_fixture() -> Path:
    assert (COMMITTED_FIXTURE / "build_fixture.py").exists(), "committed calibration fixture missing"
    return COMMITTED_FIXTURE


def test_committed_fixture_calibrates_clean(committed_fixture: Path, tmp_path: Path) -> None:
    """The committed bundle passes every gate end-to-end through run_calibration."""
    result = run_calibration(
        _config(
            committed_fixture,
            tmp_path,
            corpus_dir=committed_fixture / "corpus",
            gold_labels=committed_fixture / "gold.json",
            breakdowns=committed_fixture / "breakdowns.json",
        )
    )
    assert result["record_count"] == 12
    assert result["metrics"]["class_balance"] == {"accepted": 6, "rejected": 6}
    artifact = json.loads((tmp_path / "out" / "calibration.json").read_text())
    assert artifact["schema_version"] == "calibration-artifact-v1"


def test_committed_fixture_contains_both_classes_and_c5_repo(committed_fixture: Path) -> None:
    """Both gold classes are present, and the C5-excluded slug appears only in
    its corruption variant — never in the clean bundle."""
    records = [
        json.loads(line)
        for line in (committed_fixture / "corpus" / "corpus.jsonl").read_text().splitlines()
        if line.strip()
    ]
    gold = json.loads((committed_fixture / "gold.json").read_text())
    labels = {gold[rid]["accepted"] for rid in gold}
    assert labels == {True, False}
    assert len(records) == 12
    assert all(r["repo_slug"] != C5_SLUG for r in records)
    variant_records = [
        json.loads(line)
        for line in (committed_fixture / "variants" / "c5-excluded" / "corpus.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert any(r["repo_slug"] == C5_SLUG for r in variant_records)


def test_committed_fixture_variant_c5_excluded_fails_closed(
    committed_fixture: Path, tmp_path: Path
) -> None:
    with pytest.raises(CalibrationError, match=r"excluded repository list \(C5\)"):
        run_calibration(
            _config(
                committed_fixture,
                tmp_path,
                corpus_dir=committed_fixture / "variants" / "c5-excluded",
                gold_labels=committed_fixture / "gold.json",
                breakdowns=committed_fixture / "breakdowns.json",
            )
        )


def test_committed_fixture_variant_posterior_fails_closed(
    committed_fixture: Path, tmp_path: Path
) -> None:
    with pytest.raises(CalibrationError, match="posterior to as_of"):
        run_calibration(
            _config(
                committed_fixture,
                tmp_path,
                corpus_dir=committed_fixture / "variants" / "posterior",
                gold_labels=committed_fixture / "gold.json",
                breakdowns=committed_fixture / "breakdowns.json",
            )
        )


def test_committed_fixture_variant_digest_tamper_fails_closed(
    committed_fixture: Path, tmp_path: Path
) -> None:
    with pytest.raises(CalibrationError, match="digest mismatch"):
        run_calibration(
            _config(
                committed_fixture,
                tmp_path,
                corpus_dir=committed_fixture / "variants" / "digest",
                gold_labels=committed_fixture / "gold.json",
                breakdowns=committed_fixture / "breakdowns.json",
            )
        )


def test_committed_stage0_files_join_and_misalign_refused(
    committed_fixture: Path, tmp_path: Path
) -> None:
    shared = dict(
        corpus_dir=committed_fixture / "corpus",
        gold_labels=committed_fixture / "gold.json",
        breakdowns=committed_fixture / "breakdowns.json",
    )
    result = run_calibration(
        _config(
            committed_fixture,
            tmp_path,
            stage0_scores=committed_fixture / "stage0-scores-aligned.json",
            **shared,
        )
    )
    assert result["stage0_analysis"]["status"] == "ok"
    with pytest.raises(CalibrationError, match="no matching record"):
        run_calibration(
            _config(
                committed_fixture,
                tmp_path / "r2",
                stage0_scores=committed_fixture / "stage0-scores-misaligned.json",
                **shared,
            )
        )


def test_build_fixture_replays_byte_identical(committed_fixture: Path, tmp_path: Path) -> None:
    """The checked-in generator regenerates the committed files byte-for-byte."""
    replay = tmp_path / "replay"
    subprocess.run(
        [sys.executable, str(committed_fixture / "build_fixture.py"), "--out", str(replay)],
        check=True,
        cwd=Path(__file__).parent.parent,
    )
    committed = [
        p.relative_to(committed_fixture)
        for p in committed_fixture.rglob("*")
        if p.is_file() and p.name != "build_fixture.py" and "__pycache__" not in p.parts
    ]
    assert committed, "committed fixture is empty"
    for rel in committed:
        assert (replay / rel).read_bytes() == (committed_fixture / rel).read_bytes(), rel
