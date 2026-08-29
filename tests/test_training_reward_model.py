import json
from pathlib import Path

import pytest

from daydream.training.reward_model import score_comment, train_outcome_model


def _pairs(tmp_path: Path, n: int = 20) -> Path:
    rows = []
    for i in range(n):
        rows.append({"comment_id": f"a{i}", "text": f"solid grounding {i}", "label": "accepted"})
        rows.append({"comment_id": f"r{i}", "text": f"noise noise {i}", "label": "rejected"})
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
    rows = [{"comment_id": "a", "text": "x", "label": "accepted"}]
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="both classes"):
        train_outcome_model(p, split={"train": 0.8, "held_out": 0.2}, seed=0)  # C9: cannot rank on one class


def test_deterministic_given_seed(tmp_path: Path) -> None:
    p = _pairs(tmp_path, n=8)
    m1 = train_outcome_model(p, split={"train": 0.8, "held_out": 0.2}, seed=7)
    m2 = train_outcome_model(p, split={"train": 0.8, "held_out": 0.2}, seed=7)
    assert score_comment(m1, "some comment") == score_comment(m2, "some comment")
