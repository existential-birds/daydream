import json
from pathlib import Path

import pytest

from daydream.training.reward_model import score_comment, train_outcome_model


def _pairs(tmp_path: Path, n: int = 20) -> Path:
    rows = []
    for i in range(n):
        rows.append({"comment_id": f"a{i}", "text": f"solid grounding {i}", "label": "accepted",
                     "labeler_policy_version": "980-policy-r1"})
        rows.append({"comment_id": f"r{i}", "text": f"noise noise {i}", "label": "rejected",
                     "labeler_policy_version": "980-policy-r1"})
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_trains_on_both_classes_and_ranks(tmp_path: Path) -> None:
    p = _pairs(tmp_path)
    model = train_outcome_model(p, split={"train": 0.8, "held_out": 0.2}, seed=0)
    assert model.label_ratio_reported  # S2: actual ratio at training time, not a stale figure
    assert score_comment(model, "grounded, references line 42 of the diff") > score_comment(
        model, "nit: lol looks fine"
    )


def test_refuses_single_class_training(tmp_path: Path) -> None:
    rows = [{"comment_id": "a", "text": "x", "label": "accepted", "labeler_policy_version": "980-policy-r1"}]
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="both classes"):
        train_outcome_model(p, split={"train": 0.8, "held_out": 0.2}, seed=0)  # C9: cannot rank on one class


def test_deterministic_given_seed(tmp_path: Path) -> None:
    p = _pairs(tmp_path, n=8)
    m1 = train_outcome_model(p, split={"train": 0.8, "held_out": 0.2}, seed=7)
    m2 = train_outcome_model(p, split={"train": 0.8, "held_out": 0.2}, seed=7)
    assert score_comment(m1, "some comment") == score_comment(m2, "some comment")


def test_reads_production_export_shape(tmp_path: Path) -> None:
    """Issue 2: gold admission reads the run_build_corpus export keys
    (outcome_label/review_output/session_id) as well as the fixture keys."""
    rows = []
    for i in range(12):
        rows.append({"session_id": f"a{i}", "review_output": f"solid grounding {i}",
                     "outcome_label": "accepted", "labeler_policy_version": "980-policy-r1"})
        rows.append({"session_id": f"r{i}", "review_output": f"noise noise {i}",
                     "outcome_label": "rejected", "labeler_policy_version": "980-policy-r1"})
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    model = train_outcome_model(p, split={"train": 0.8, "held_out": 0.2}, seed=0)
    assert model.label_ratio_reported
    assert score_comment(model, "grounded, references line 42 of the diff") > score_comment(
        model, "nit: lol looks fine"
    )


def test_refuses_legacy_row_without_policy_version(tmp_path: Path) -> None:
    """Issue 17/21: a row with no labeler_policy_version is refused as legacy,
    never silently admitted via a fallback version."""
    rows = [
        {"comment_id": "a0", "text": "solid grounding 0", "label": "accepted",
         "labeler_policy_version": "980-policy-r1"},
        {"comment_id": "r0", "text": "noise noise 0", "label": "rejected"},  # legacy: no version
    ]
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="refused by the gold-outcome gate"):
        train_outcome_model(p, split={"train": 0.8, "held_out": 0.2}, seed=0)
