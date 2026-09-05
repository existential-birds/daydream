"""Task-1 tests: fail-closed validation matrix (M2) for ``calibrate-reward``.

``fixture_corpus`` builds a minimal synthetic corpus-v2 bundle in ``tmp_path``;
corruption variants are injected through ``CalibrationConfig.corruptions`` so
every gate is exercised against otherwise-identical inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream.training.calibration import (
    CalibrationConfig,
    CalibrationError,
    assign_split,
    run_calibration,
)
from daydream.training.reward import REWARD_VERSION as _PRODUCTION_REWARD_VERSION

AS_OF = "2026-01-01T00:00:00+00:00"
VALID_AT = "2025-12-01T00:00:00+00:00"
#: Salt chosen so the derived holdout keeps >2 records with both gold classes:
#: a 1-2 record holdout forces every stage-0 marginal to exactly 0.0 / +/-1,
#: hiding regressions in the residualization math.
SALT = "cal-salt-2"
LABEL_STAMPS = {
    "labeler_policy_version": "980-policy-r1",
    "reply_classifier_version": "980-classifier-r1",
    "rubric_schema_version": "980-rubric-r2",
}
#: Imported, never duplicated as a literal: ``build_fixture.py`` stamps the
#: committed corpora from this same production constant, so a local copy
#: would silently break every gate on a legitimate bump. Drift is caught by
#: ``test_build_fixture_replays_byte_identical``, not by a second literal.
REWARD_VERSION = _PRODUCTION_REWARD_VERSION

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
            # Stored split must equal the split re-derived by run_calibration
            # from the fixture salt + split rates (0.1/0.1 in lineage.json).
            "split": assign_split(f"rec-{i:04d}", holdout_rate=0.1, val_rate=0.1, salt=SALT),
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
    # Stage-0 score files: aligned (one entry per record) and misaligned (unknown
    # record_id, plus one corpus record whose score is dropped).
    scores = {
        f"rec-{i:04d}": {"score": 0.3 + 0.1 * i, "model_digest": "sha256:stage0-model-1"}
        for i in range(4)
    }
    (tmp_path / "stage0-scores-aligned.json").write_text(json.dumps(scores, sort_keys=True) + "\n")
    misaligned = dict(scores)
    del misaligned["rec-0000"]  # drop a corpus record's score (missing-score direction)
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
    candidates = {"w_fp": [0.3, 0.6]}  # straddles the fixture's w_fp values
    result = run_calibration(_config(fixture_corpus, tmp_path, seed=42, candidates=candidates))
    m = result["metrics"]
    assert m["length_distribution"]["median"] == round(m["length_distribution"]["median"], 4)
    assert {"median", "iqr"} <= set(m["length_distribution"])
    assert set(m["class_balance"]) == {"accepted", "rejected"}
    assert m["per_axis_correlations"]["w_fp_axis"] <= 1.0
    lo, hi = m["auc_bootstrap_ci"]["w_fp"]
    assert lo < hi
    # the candidate grid genuinely differentiates the per-point metrics
    assert m["threshold_metrics"]["w_fp=0.3"] != m["threshold_metrics"]["w_fp=0.6"]
    # same seed ⇒ identical bytes
    run_calibration(
        _config(fixture_corpus, tmp_path / "r2", out_dir=tmp_path / "r2", seed=42, candidates=candidates)
    )
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


def test_stored_split_mismatch_fails_closed(tmp_path: Path) -> None:
    """A stored lineage.split that diverges from the re-derived split is refused."""
    root = _build_fixture(tmp_path)
    corpus_dir = root / "corpus"
    records = [
        json.loads(line)
        for line in (corpus_dir / "corpus.jsonl").read_text().splitlines()
        if line.strip()
    ]
    lineage = json.loads((corpus_dir / "lineage.json").read_text())
    rid = str(records[0]["record_id"])
    derived = assign_split(
        rid,
        holdout_rate=float(lineage["holdout_rate"]),
        val_rate=float(lineage["val_rate"]),
        salt=str(lineage["salt"]),
    )
    records[0]["lineage"]["split"] = "val" if derived != "val" else "holdout"
    payload = "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    (corpus_dir / "corpus.jsonl").write_text(payload)
    (corpus_dir / "SHA256SUMS").write_text(
        f"{hashlib.sha256(payload.encode()).hexdigest()}  corpus.jsonl\n"
    )
    with pytest.raises(CalibrationError, match="stored split.*does not match"):
        run_calibration(_config(root, tmp_path))


# --- Task 3 (M4): stage-0 marginal analysis + explicit unavailable marker ---


def test_stage0_scores_report_marginal_value(fixture_corpus: Path, tmp_path: Path) -> None:
    scores = fixture_corpus / "stage0-scores-aligned.json"  # keyed by record_id + model_digest
    result = run_calibration(
        _config(
            fixture_corpus,
            tmp_path,
            stage0_scores=scores,
            model_digest="sha256:stage0-model-1",
        )
    )
    s0 = result["stage0_analysis"]
    assert s0["status"] == "ok"
    marginal = s0["marginal_value_per_axis"]
    assert set(marginal) >= {"correctness", "grounding"}
    # Non-degenerate: a tiny holdout would force every marginal to exactly
    # 0.0 / +/-1 and hide regressions, so at least one axis must be interior.
    assert any(-1.0 < v < 1.0 for v in marginal.values())


def test_stage0_scores_require_model_digest(tmp_path: Path, fixture_corpus: Path) -> None:
    # --model-digest is documented as required with --stage0-scores; a missing
    # digest must fail closed instead of silently skipping verification.
    scores = fixture_corpus / "stage0-scores-aligned.json"
    with pytest.raises(CalibrationError, match="model-digest is required"):
        run_calibration(_config(fixture_corpus, tmp_path, stage0_scores=scores))


def test_stage0_absent_is_explicit_unavailable(fixture_corpus: Path, tmp_path: Path) -> None:
    result = run_calibration(_config(fixture_corpus, tmp_path))
    assert result["stage0_analysis"] == {"status": "unavailable"}


def test_stage0_misaligned_records_are_refused(fixture_corpus: Path, tmp_path: Path) -> None:
    scores = fixture_corpus / "stage0-scores-misaligned.json"
    with pytest.raises(CalibrationError, match="stage-0 score.*no matching record"):
        run_calibration(
            _config(
                fixture_corpus,
                tmp_path,
                stage0_scores=scores,
                model_digest="sha256:stage0-model-1",
            )
        )


def test_stage0_missing_score_for_corpus_record_is_refused(
    fixture_corpus: Path, tmp_path: Path
) -> None:
    """Reverse no-partial-join direction: a corpus record with no stage-0 score
    fails closed (the gate at calibration.py:467), not the unknown-record_id gate."""
    raw = json.loads((fixture_corpus / "stage0-scores-misaligned.json").read_text())
    del raw["rec-ghost"]  # keep only the dropped-record direction
    dropped = tmp_path / "stage0-scores-dropped.json"
    dropped.write_text(json.dumps(raw, sort_keys=True) + "\n")
    with pytest.raises(
        CalibrationError, match="missing stage-0 score for corpus record 'rec-0000'"
    ):
        run_calibration(
            _config(
                fixture_corpus,
                tmp_path / "r2",
                stage0_scores=dropped,
                model_digest="sha256:stage0-model-1",
            )
        )


# --- Task 5 (M5, M6, S1, S2): versioned artifact + reproducible report ---


def test_artifact_schema_and_byte_replay(fixture_corpus: Path, tmp_path: Path) -> None:
    run_calibration(_config(fixture_corpus, tmp_path, seed=7))
    art = json.loads((tmp_path / "out" / "calibration.json").read_text())
    assert art["schema_version"] == "calibration-artifact-v1"
    assert art["tool_version"] and art["resampling_seed"] == 7
    assert art["input_digests"]["corpus.jsonl"].startswith("sha256:")
    assert art["corpus_digest"] and art["split_digest"]  # S1
    assert {str(c) for c in art["candidate_settings"]} >= {"w_fp"}
    first = (tmp_path / "out" / "calibration.json").read_bytes()
    run_calibration(_config(fixture_corpus, tmp_path, out_dir=tmp_path / "replay", seed=7))
    assert (tmp_path / "replay" / "calibration.json").read_bytes() == first  # M6
    assert (tmp_path / "out" / "report.md").read_text().count("w_fp") >= 1  # M5b


def test_report_is_reproducible_too(fixture_corpus: Path, tmp_path: Path) -> None:  # S2
    run_calibration(_config(fixture_corpus, tmp_path, seed=7))
    a = (tmp_path / "out" / "report.md").read_bytes()
    run_calibration(_config(fixture_corpus, tmp_path, out_dir=tmp_path / "r2", seed=7))
    assert (tmp_path / "r2" / "report.md").read_bytes() == a


def test_no_artifact_on_failure(tmp_path: Path, fixture_corpus: Path) -> None:
    with pytest.raises(CalibrationError):
        run_calibration(_config(fixture_corpus, tmp_path, out_dir=tmp_path / "out", digest=True))
    assert not (tmp_path / "out" / "calibration.json").exists()
    assert not (tmp_path / "out" / "report.md").exists()


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
            model_digest="sha256:calibration-stage0-model-1",
            **shared,
        )
    )
    assert result["stage0_analysis"]["status"] == "ok"
    # The committed fixture's derived holdout must stay non-degenerate; an
    # all-+/-1 marginal means a fixture change collapsed the holdout draw and
    # regressions would go undetected.
    marginal = result["stage0_analysis"]["marginal_value_per_axis"]
    assert any(-1.0 < v < 1.0 for v in marginal.values())
    with pytest.raises(CalibrationError, match="no matching record"):
        run_calibration(
            _config(
                committed_fixture,
                tmp_path / "r2",
                stage0_scores=committed_fixture / "stage0-scores-misaligned.json",
                model_digest="sha256:calibration-stage0-model-1",
                **shared,
            )
        )
    # Reverse direction of the same no-partial-join gate: drop the unknown
    # record_id so only the corpus record missing its score remains.
    raw = json.loads((committed_fixture / "stage0-scores-misaligned.json").read_text())
    del raw["rec-ghost"]
    dropped = tmp_path / "stage0-scores-dropped.json"
    dropped.write_text(json.dumps(raw, sort_keys=True) + "\n")
    with pytest.raises(CalibrationError, match="missing stage-0 score for corpus record 'rec-0000'"):
        run_calibration(
            _config(
                committed_fixture,
                tmp_path / "r3",
                stage0_scores=dropped,
                model_digest="sha256:calibration-stage0-model-1",
                **shared,
            )
        )


def test_build_fixture_replays_byte_identical(committed_fixture: Path, tmp_path: Path) -> None:
    """The checked-in generator regenerates the committed files byte-for-byte."""
    replay = tmp_path / "replay"
    # The generator imports the top-level ``daydream`` package, which would
    # resolve to an installed copy (or nothing) rather than this checkout;
    # pin the child interpreter to the repo root so replay is always enforced.
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, str(committed_fixture / "build_fixture.py"), "--out", str(replay)],
        check=True,
        cwd=Path(__file__).parent.parent,
        env=env,
    )
    committed = [
        p.relative_to(committed_fixture)
        for p in committed_fixture.rglob("*")
        if p.is_file() and p.name != "build_fixture.py" and "__pycache__" not in p.parts
    ]
    assert committed, "committed fixture is empty"
    for rel in committed:
        assert (replay / rel).read_bytes() == (committed_fixture / rel).read_bytes(), rel


def test_output_collision_with_different_run_identity_is_refused(
    fixture_corpus: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_calibration(_config(fixture_corpus, tmp_path, out_dir=out, run_id="cal-1"))
    with pytest.raises(CalibrationError, match="run identity"):
        run_calibration(_config(fixture_corpus, tmp_path, out_dir=out, run_id="cal-2"))


def test_same_run_identity_rerun_is_allowed(fixture_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_calibration(_config(fixture_corpus, tmp_path, out_dir=out, run_id="cal-1"))
    run_calibration(_config(fixture_corpus, tmp_path, out_dir=out, run_id="cal-1"))  # resume/overwrite ok
