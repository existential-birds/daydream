from pathlib import Path

import pytest

from daydream.training.gate import _build_frozen_split
from daydream.training.reward_model import score_comment, train_outcome_model
from tests.test_training_gate import _frozen_pairs


def test_trains_on_both_classes_and_ranks(tmp_path: Path) -> None:
    frozen = _frozen_pairs(tmp_path, n=20, seed=0)
    model = train_outcome_model(frozen, seed=0)
    assert model.label_ratio_reported  # S2: actual ratio at training time, not a stale figure
    assert score_comment(model, "grounded, references line 42 of the diff") > score_comment(model, "nit: lol looks fine"
    )

def test_refuses_single_class_training(tmp_path: Path) -> None:
    rows = [{"comment_id": "a", "text": "x", "label": "accepted", "labeler_policy_version": "980-policy-r1"}]
    with pytest.raises(ValueError, match="both classes"):
        _build_frozen_split(tmp_path / "labels.jsonl", train_rows=rows, held_out_rows=rows,
                            seed=0, held_out_fraction=0.5)

def test_deterministic_given_seed(tmp_path: Path) -> None:
    frozen = _frozen_pairs(tmp_path, n=8, seed=7)
    m1 = train_outcome_model(frozen, seed=7)
    m2 = train_outcome_model(_frozen_pairs(tmp_path, n=8, seed=7), seed=7)
    assert score_comment(m1, "some comment") == score_comment(m2, "some comment")

def test_reads_production_export_shape(tmp_path: Path) -> None:
    """Admit production export keys alongside fixture keys for gold outcome training."""
    rows = []
    for i in range(12):
        rows.append({"session_id": f"a{i}", "review_output": f"solid grounding {i}",
                     "outcome_label": "accepted", "labeler_policy_version": "980-policy-r1"})
        rows.append({"session_id": f"r{i}", "review_output": f"noise noise {i}",
                     "outcome_label": "rejected", "labeler_policy_version": "980-policy-r1"})
    split = _build_frozen_split(tmp_path / "labels.jsonl", train_rows=rows[:20], held_out_rows=rows[20:],
                                seed=0, held_out_fraction=0.2)
    model = train_outcome_model(split, seed=0)
    assert model.label_ratio_reported
    assert score_comment(model, "grounded, references line 42 of the diff") > score_comment(model, "nit: lol looks fine"
    )

def test_refuses_legacy_row_without_policy_version(tmp_path: Path) -> None:
    rows = [{"comment_id": "a0", "text": "solid grounding 0", "label": "accepted",
         "labeler_policy_version": "980-policy-r1"},
        {"comment_id": "r0", "text": "noise noise 0", "label": "rejected"},  # legacy: no version
    ]
    with pytest.raises(ValueError, match="refused by the gold-outcome gate"):
        _build_frozen_split(tmp_path / "legacy.jsonl", train_rows=rows[:1], held_out_rows=rows[1:],
                            seed=0, held_out_fraction=0.5)
