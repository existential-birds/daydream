"""Build the synthetic corpus-v2 calibration fixture (issue #999, M8).

Regenerates every file under ``tests/fixtures/training/calibration/``
byte-for-byte deterministically (no timestamps, sorted keys, fixed seed-free
formulas) so replay diffs are reviewable:

    uv run python tests/fixtures/training/calibration/build_fixture.py
    uv run python tests/fixtures/training/calibration/build_fixture.py --out /tmp/replay

Layout (consumed by ``daydream.training.calibration.run_calibration``):

- ``corpus/``          — clean bundle: ``corpus.jsonl``, ``lineage.json``,
                         ``curation-manifest.json``, ``SHA256SUMS``, ``_SUCCESS``
- ``gold.json``        — gold labels keyed by record_id: ``{"accepted": bool}``
- ``breakdowns.json``  — per-axis intrinsic breakdowns keyed by record_id
- ``stage0-scores-aligned.json``    — one score per corpus record
- ``stage0-scores-misaligned.json`` — aligned scores plus one unknown record_id and one corpus record dropped
- ``variants/``        — deliberate corruption bundles, one gate each:
                         ``c5-excluded/`` (C5 exclusion list slug),
                         ``posterior/`` (valid_at posterior to as_of),
                         ``digest/`` (tampered corpus, pristine SHA256SUMS)

Version stamps (labeler/reward versions) are imported from the production
modules, so a version bump requires re-running this generator and committing
the diff — which is exactly the reviewability the fixture wants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from daydream.training.calibration import assign_split
from daydream.training.exclusion import load_exclusion_list
from daydream.training.labeler_versions import (
    LABELER_POLICY_VERSION,
    REPLY_CLASSIFIER_VERSION,
    RUBRIC_SCHEMA_VERSION,
)
from daydream.training.reward import REWARD_VERSION

AS_OF = "2026-01-01T00:00:00+00:00"
VALID_AT = "2025-12-01T00:00:00+00:00"
#: Salt chosen so the derived holdout split is non-degenerate for the 12-record
#: corpus (>=3 records, mixed gold labels): a 2-record holdout forces every
#: stage-0 marginal point-biserial to exactly +/-1, hiding regressions.
SALT = "calibration-fixture-salt-nondegenerate"
RECORD_COUNT = 12
STAGE0_MODEL_DIGEST = "sha256:calibration-stage0-model-1"

#: A slug that is on the C5 exclusion list in the repo schema — used only in
#: the ``variants/c5-excluded`` corruption bundle, never in the clean corpus.
C5_SLUG = "getsentry/sentry"

REPO_SLUGS = [f"acme/widgets-{i % 3}" for i in range(RECORD_COUNT)]


def _record(i: int) -> dict[str, Any]:
    return {
        "schema_version": "2",
        "record_id": f"rec-{i:04d}",
        "session_id": f"sess-{i:04d}",
        "repo_slug": REPO_SLUGS[i],
        "reward_version": REWARD_VERSION,
        "lineage": {
            # Stored split must equal the split run_calibration re-derives from
            # the bundle salt + split rates (0.2/0.2 in _lineage()); the
            # stored-split gate compares them fail-closed.
            "split": assign_split(f"rec-{i:04d}", holdout_rate=0.2, val_rate=0.2, salt=SALT),
            "as_of": AS_OF,
            "valid_at": VALID_AT,
            "license_decision": "allow",
            "labeler_policy_version": LABELER_POLICY_VERSION,
            "reply_classifier_version": REPLY_CLASSIFIER_VERSION,
            "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
        },
    }


def _records() -> list[dict[str, Any]]:
    return [_record(i) for i in range(RECORD_COUNT)]


def _gold() -> dict[str, dict[str, Any]]:
    # Interleaved 6/6 class balance.
    return {f"rec-{i:04d}": {"accepted": i % 2 == 0} for i in range(RECORD_COUNT)}


def _breakdowns() -> dict[str, dict[str, float]]:
    # Partially separable so bootstrap CIs and correlations are non-degenerate.
    w_fp = [0.55, 0.50, 0.65, 0.58, 0.52, 0.61, 0.48, 0.57, 0.63, 0.51, 0.59, 0.47]
    return {
        f"rec-{i:04d}": {
            "fidelity": round(0.2 + 0.05 * i, 4),
            "specificity": round(0.8 - 0.04 * i, 4),
            "correctness": round(0.4 + 0.03 * i, 4),
            "grounding": round(0.7 - 0.02 * i, 4),
            "w_fp": w_fp[i],
        }
        for i in range(RECORD_COUNT)
    }


def _stage0_scores(aligned: bool) -> dict[str, dict[str, Any]]:
    scores = {
        f"rec-{i:04d}": {"score": round(0.3 + 0.05 * i, 4), "model_digest": STAGE0_MODEL_DIGEST}
        for i in range(RECORD_COUNT)
    }
    if not aligned:
        del scores["rec-0000"]  # drop a corpus record's score (missing-score direction)
        scores["rec-ghost"] = {"score": 0.9, "model_digest": STAGE0_MODEL_DIGEST}
    return scores


def _lineage() -> dict[str, Any]:
    return {
        "schema_version": "corpus-v2",
        "salt": SALT,
        "holdout_rate": 0.2,
        "val_rate": 0.2,
        "as_of": AS_OF,
        "valid_at": VALID_AT,
        "content_digests": {},
    }


def _manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    gold = _gold()
    excluded_hits = sorted(set(load_exclusion_list()) & {r["repo_slug"] for r in records})
    c5_claim = (
        f"contains excluded repo slug(s): {', '.join(excluded_hits)}"
        if excluded_hits
        else "clean corpus repo slugs are synthetic and absent from the exclusion list"
    )
    return {
        "description": "Synthetic corpus-v2 bundle for calibrate-reward fixtures (issue #999)",
        "record_count": RECORD_COUNT,
        "accepted_count": sum(1 for v in gold.values() if v["accepted"]),
        "rejected_count": sum(1 for v in gold.values() if not v["accepted"]),
        "constraints": {
            "C5_exclusion": c5_claim,
            "C8_copyleft": "clean corpus repo slugs are absent from the copyleft list",
            "gpu_free": True,
        },
    }


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n").encode()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _sha256sums(corpus_dir: Path, names: list[str]) -> None:
    lines = [
        f"{hashlib.sha256((corpus_dir / name).read_bytes()).hexdigest()}  {name}" for name in names
    ]
    (corpus_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def _write_bundle(corpus_dir: Path, records: list[dict[str, Any]], *, sums_over_tampered: bool = False) -> None:
    """Write corpus.jsonl + lineage.json + SHA256SUMS (+ curation-manifest).

    With ``sums_over_tampered`` (the digest variant), the corpus is tampered
    *after* the pristine SHA256SUMS is computed — the deliberate corruption.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus.jsonl").write_bytes(_jsonl_bytes(records))
    _write_json(corpus_dir / "lineage.json", _lineage())
    _write_json(corpus_dir / "curation-manifest.json", _manifest(records))
    (corpus_dir / "_SUCCESS").write_text("")
    _sha256sums(corpus_dir, ["corpus.jsonl", "lineage.json", "curation-manifest.json"])
    if sums_over_tampered:
        tampered = [json.loads(line) for line in (corpus_dir / "corpus.jsonl").read_text().splitlines()]
        tampered[0]["session_id"] = "sess-tampered"
        (corpus_dir / "corpus.jsonl").write_bytes(_jsonl_bytes(tampered))


def build(out: Path) -> None:
    excluded = load_exclusion_list()
    assert C5_SLUG in excluded, f"{C5_SLUG} is no longer on the C5 exclusion list; pick another slug"
    assert not any(slug in excluded for slug in set(REPO_SLUGS)), "clean corpus slug hit the exclusion list"

    records = _records()

    # Clean bundle.
    _write_bundle(out / "corpus", records)

    # Join files.
    _write_json(out / "gold.json", _gold())
    _write_json(out / "breakdowns.json", _breakdowns())
    _write_json(out / "stage0-scores-aligned.json", _stage0_scores(aligned=True))
    _write_json(out / "stage0-scores-misaligned.json", _stage0_scores(aligned=False))

    # Corruption variants: identical shapes, one deliberate defect each.
    c5 = [json.loads(json.dumps(r)) for r in records]
    c5[0]["repo_slug"] = C5_SLUG
    _write_bundle(out / "variants" / "c5-excluded", c5)

    posterior = [json.loads(json.dumps(r)) for r in records]
    posterior[0]["lineage"]["valid_at"] = "2026-06-01T00:00:00+00:00"
    _write_bundle(out / "variants" / "posterior", posterior)

    _write_bundle(out / "variants" / "digest", records, sums_over_tampered=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory (default: the committed fixture directory)",
    )
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
