"""Task-1 tests: fail-closed validation matrix (M2) for ``calibrate-reward``.

``fixture_corpus`` builds a minimal synthetic projected-corpus bundle in ``tmp_path``;
corruption variants are written to bundle files so every gate is exercised
through the same input boundary as the CLI.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream.training.calibration import (
    CalibrationConfig,
    CalibrationError,
    run_calibration,
)
from daydream.training.corpus_projection.splits import assign_split
from daydream.training.reward import REWARD_VERSION as _PRODUCTION_REWARD_VERSION

AS_OF = "2026-01-01T00:00:00+00:00"
VALID_AT = "2025-12-01T00:00:00+00:00"
#: Salt chosen so the derived holdout keeps >2 records with both gold classes:
#: a 1-2 record holdout forces every stage-0 marginal to exactly 0.0 / +/-1,
#: hiding regressions in the residualization math.
SALT = "cal-salt-2"
LABEL_STAMPS = {"labeler_policy_version": "980-policy-r1", "reply_classifier_version": "980-classifier-r1",
    "rubric_schema_version": "980-rubric-r2",
}
#: Synthetic fresh corpora use current semantics; committed fixture remains frozen.
REWARD_VERSION = _PRODUCTION_REWARD_VERSION

CORRUPTION_FLAGS = frozenset(
    {
        "schema_version",
        "posterior",
        "label-version",
        "c5-repo",
        "identity",
        "digest",
        "split-overlap",
        "drop-session",
    }
)


def _record(i: int) -> dict[str, Any]:
    return {
        "schema_version": "3", "record_id": f"rec-{i:04d}", "session_id": f"sess-{i:04d}",
        "reward_version": REWARD_VERSION,
        "lineage": {
            # Stored split must equal the split re-derived by run_calibration
            # from the fixture salt + split rates (0.1/0.1 in lineage.json).
            "split": assign_split(f"rec-{i:04d}", holdout_rate=0.1, val_rate=0.1, salt=SALT), "as_of": AS_OF,
            "valid_at": VALID_AT, "repo_slug": "acme/widgets",
            **LABEL_STAMPS,
        },
    }


def _build_fixture(tmp_path: Path) -> Path:
    """Write corpus.jsonl + lineage.json + SHA256SUMS + gold.json + breakdowns.json."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    records = [_record(i) for i in range(4)]
    _write_corpus(corpus_dir, records)
    (corpus_dir / "lineage.json").write_text(json.dumps({
                "schema_version": "lineage", "salt": SALT, "holdout_rate": 0.1, "val_rate": 0.1, "as_of": AS_OF,
                "valid_at": VALID_AT, "content_digests": {},
            }, sort_keys=True,
        )
        + "\n"
    )
    gold = {f"rec-{i:04d}": {"accepted": i % 2 == 0} for i in range(4)}
    (tmp_path / "gold.json").write_text(json.dumps(gold, sort_keys=True) + "\n")
    # w_fp is partially but not perfectly separable so bootstrap CIs are non-degenerate.
    w_fp = [0.55, 0.5, 0.65, 0.58]
    breakdowns = {f"rec-{i:04d}": {
            "fidelity": 0.2 + 0.1 * i, "specificity": 0.8 - 0.1 * i, "correctness": 0.4 + 0.05 * i, "w_fp": w_fp[i],
        }
        for i in range(4)
    }
    (tmp_path / "breakdowns.json").write_text(json.dumps(breakdowns, sort_keys=True) + "\n")
    # Stage-0 score files: aligned (one entry per record) and misaligned (unknown
    # record_id, plus one corpus record whose score is dropped).
    scores = {f"rec-{i:04d}": {"score": 0.3 + 0.1 * i, "model_digest": "sha256:stage0-model-1"} for i in range(4)}
    (tmp_path / "stage0-scores-aligned.json").write_text(json.dumps(scores, sort_keys=True) + "\n")
    misaligned = dict(scores)
    del misaligned["rec-0000"]  # drop a corpus record's score (missing-score direction)
    misaligned["rec-ghost"] = {"score": 0.9, "model_digest": "sha256:stage0-model-1"}
    (tmp_path / "stage0-scores-misaligned.json").write_text(json.dumps(misaligned, sort_keys=True) + "\n")
    return tmp_path


@pytest.fixture
def fixture_corpus(tmp_path: Path) -> Path:
    return _build_fixture(tmp_path)


def _write_corpus(corpus_dir: Path, records: list[dict[str, Any]], *, update_digest: bool = True) -> None:
    payload = ("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n").encode()
    (corpus_dir / "corpus.jsonl").write_bytes(payload)
    if update_digest:
        (corpus_dir / "SHA256SUMS").write_text(f"{hashlib.sha256(payload).hexdigest()}  corpus.jsonl\n")


def _corrupt_fixture(corpus_dir: Path, flags: frozenset[str]) -> None:
    if not flags:
        return
    records = [json.loads(line) for line in (corpus_dir / "corpus.jsonl").read_text().splitlines()]
    first = records[0]
    if "schema_version" in flags:
        first["schema_version"] = "1"
    if "posterior" in flags:
        first["lineage"]["valid_at"] = "2026-06-01T00:00:00+00:00"
    if "label-version" in flags:
        del first["lineage"]["labeler_policy_version"]
    if "c5-repo" in flags:
        first["lineage"]["repo_slug"] = "getsentry/sentry"
    if "identity" in flags:
        first["lineage"]["repo_slug"] = "owner/repo/extra"
    if "split-overlap" in flags:
        records.append(json.loads(json.dumps(records[-1])))
    if "drop-session" in flags:
        del first["session_id"]
    if "digest" in flags:
        records = [{**records[0], "session_id": "sess-tampered"}]
    _write_corpus(corpus_dir, records, update_digest="digest" not in flags)


def _config(fixture_dir: Path, tmp_path: Path, **overrides: object) -> CalibrationConfig:
    corrupt = frozenset(k for k, v in overrides.items() if k in CORRUPTION_FLAGS and v)
    base: dict[str, Any] = dict(corpus_dir=fixture_dir / "corpus", gold_labels=fixture_dir / "gold.json",
        breakdowns=fixture_dir / "breakdowns.json", out_dir=tmp_path / "out", run_id="cal-test-1", seed=7,
        candidates={"w_fp": [0.1, 0.3]},
    )
    base.update({k: v for k, v in overrides.items() if k not in CORRUPTION_FLAGS})
    _corrupt_fixture(base["corpus_dir"], corrupt)
    return CalibrationConfig(**base)


def _mixed_reward_version(root: Path) -> None:
    corpus = root / "corpus" / "corpus.jsonl"
    records = [json.loads(line) for line in corpus.read_text().splitlines()]
    records[0]["reward_version"] = "2026.09.04-1"
    _write_corpus(corpus.parent, records)


def _stored_split_mismatch(root: Path) -> None:
    corpus_dir = root / "corpus"
    records = [json.loads(line) for line in (corpus_dir / "corpus.jsonl").read_text().splitlines()]
    lineage = json.loads((corpus_dir / "lineage.json").read_text())
    derived = assign_split(str(records[0]["record_id"]), holdout_rate=float(lineage["holdout_rate"]),
                           val_rate=float(lineage["val_rate"]), salt=str(lineage["salt"]))
    records[0]["lineage"]["split"] = "validation" if derived != "validation" else "holdout"
    _write_corpus(corpus_dir, records)


def _numeric_repo_slug(root: Path) -> None:
    records = [json.loads(line) for line in (root / "corpus/corpus.jsonl").read_text().splitlines()]
    records[0]["lineage"]["repo_slug"] = 17
    _write_corpus(root / "corpus", records)


def test_calibration_fails_closed_before_any_output(tmp_path: Path) -> None:
    """Every input gate refuses before calibration.json (or the out dir) is written."""
    cases: list[tuple[str, dict[str, Any], Any, str]] = [
        ("schema", {"schema_version": True}, None, "schema_version"),  # record stamped other than "3"
        ("posterior", {"posterior": True}, None, "valid_at"),  # valid_at posterior to as_of
        ("label", {"label-version": True}, None, "unrecognized label"),  # absent/unknown version stamp
        ("c5", {"c5-repo": True}, None, "excluded repository"),
        ("overlap", {"split-overlap": True}, None, "split membership overlaps"),
        ("identity", {"identity": True}, None, "repo_slug"),
        ("numeric-slug", {}, _numeric_repo_slug, "repo_slug"),
        ("drop-session", {"drop-session": True}, None, "missing required field"),
        ("empty-range", {"candidates": {"w_fp": []}}, None, "w_fp"),
        ("unknown-axis", {"candidates": {"w_ghost": [0.1, 0.2]}}, None, "w_ghost"),
        ("split", {}, _stored_split_mismatch, "stored split.*does not match"),
        ("reward", {}, _mixed_reward_version, "mixes reward versions"),
        # --model-digest is required with --stage0-scores; never silently skip verification.
        ("stage0", {"stage0_scores": "stage0-scores-aligned.json"}, None, "model-digest is required"),
    ]
    for name, overrides, mutate, expected in cases:
        root = tmp_path / name
        root.mkdir()
        _build_fixture(root)
        if mutate is not None:
            mutate(root)
        if "stage0_scores" in overrides:
            overrides = {"stage0_scores": root / overrides["stage0_scores"]}
        with pytest.raises(CalibrationError, match=expected):
            run_calibration(_config(root, root, **overrides))
        assert not (root / "out").exists(), name  # no partial artifact


# --- Task 4 (M8): committed synthetic fixture under tests/fixtures/training ---

COMMITTED_FIXTURE = Path(__file__).parent / "fixtures" / "training" / "calibration"
@pytest.fixture
def committed_fixture() -> Path:
    assert (COMMITTED_FIXTURE / "build_fixture.py").exists(), "committed calibration fixture missing"
    return COMMITTED_FIXTURE


def test_committed_fixture_replays_and_stage0_files_join_or_refuse_misalignment(
    committed_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", ["build_fixture.py", "--out", str(replay)])
    runpy.run_path(str(committed_fixture / "build_fixture.py"), run_name="__main__")
    committed = [path.relative_to(committed_fixture) for path in committed_fixture.rglob("*")
                 if path.is_file() and path.name != "build_fixture.py" and "__pycache__" not in path.parts]
    assert committed, "committed fixture is empty"
    for rel in committed:
        assert (replay / rel).read_bytes() == (committed_fixture / rel).read_bytes(), rel
    # Both gold classes are present; the C5-excluded slug appears only in its corruption variant.
    def slugs(corpus: Path) -> list[str]:
        return [json.loads(line)["lineage"]["repo_slug"] for line in corpus.read_text().splitlines() if line.strip()]
    gold = json.loads((committed_fixture / "gold.json").read_text())
    assert {row["accepted"] for row in gold.values()} == {True, False}
    clean = slugs(committed_fixture / "corpus" / "corpus.jsonl")
    assert len(clean) == 12 and "getsentry/sentry" not in clean
    assert "getsentry/sentry" in slugs(committed_fixture / "variants" / "c5-excluded" / "corpus.jsonl")
    shared = dict(corpus_dir=committed_fixture / "corpus", gold_labels=committed_fixture / "gold.json",
        breakdowns=committed_fixture / "breakdowns.json",
    )
    result = run_calibration(_config(
            committed_fixture, tmp_path, stage0_scores=committed_fixture / "stage0-scores-aligned.json",
            model_digest="sha256:calibration-stage0-model-1", **shared,
        )
    )
    assert result["stage0_analysis"]["status"] == "ok"
    # The committed fixture's derived holdout must stay non-degenerate; an
    # all-+/-1 marginal means a fixture change collapsed the holdout draw and
    # regressions would go undetected.
    marginal = result["stage0_analysis"]["marginal_value_per_axis"]
    assert "correctness" in marginal and "grounding" not in marginal
    assert any(-1.0 < v < 1.0 for v in marginal.values())
    with pytest.raises(CalibrationError, match="no matching record"):
        run_calibration(_config(
                committed_fixture, tmp_path / "r2", stage0_scores=committed_fixture / "stage0-scores-misaligned.json",
                model_digest="sha256:calibration-stage0-model-1", **shared,
            )
        )
    # Reverse direction of the same no-partial-join gate: drop the unknown
    # record_id so only the corpus record missing its score remains.
    raw = json.loads((committed_fixture / "stage0-scores-misaligned.json").read_text())
    del raw["rec-ghost"]
    dropped = tmp_path / "stage0-scores-dropped.json"
    dropped.write_text(json.dumps(raw, sort_keys=True) + "\n")
    with pytest.raises(CalibrationError, match="missing stage-0 score for corpus record 'rec-0000'"):
        run_calibration(_config(committed_fixture, tmp_path / "r3", stage0_scores=dropped,
                model_digest="sha256:calibration-stage0-model-1", **shared,
            )
        )


def test_output_collision_with_different_run_identity_is_refused(fixture_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_calibration(_config(fixture_corpus, tmp_path, out_dir=out, run_id="cal-1"))
    with pytest.raises(CalibrationError, match="run identity"):
        run_calibration(_config(fixture_corpus, tmp_path, out_dir=out, run_id="cal-2"))
