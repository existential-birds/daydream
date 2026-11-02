import json
from pathlib import Path

import pytest

from daydream.training.gate import GateConfig, evaluate_gate, freeze_split
from daydream.training.reward_model import OutcomeModel, train_outcome_model


def _pairs(tmp_path: Path, n: int = 25) -> Path:
    rows = []
    for i in range(n):
        rows.append({"comment_id": f"a{i}", "text": f"solid grounding {i}", "label": "accepted",
                     "labeler_policy_version": "980-policy-r1"})
        rows.append({"comment_id": f"r{i}", "text": f"noise noise {i}", "label": "rejected",
                     "labeler_policy_version": "980-policy-r1"})
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


@pytest.fixture
def trained_model(tmp_path: Path) -> OutcomeModel:
    labels = _pairs(tmp_path)
    return train_outcome_model(labels, split=freeze_split(labels, held_out_fraction=0.2, seed=3), seed=3)


def test_gate_refuses_when_evidence_missing(trained_model: OutcomeModel) -> None:
    with pytest.raises(RuntimeError, match="gate evidence"):
        evaluate_gate(trained_model, None, GateConfig())


def test_gate_config_rejects_out_of_range_thresholds() -> None:
    with pytest.raises(ValueError):
        GateConfig(min_separation=0.0)
    with pytest.raises(ValueError):
        GateConfig(min_separation=1.0)
    with pytest.raises(ValueError):
        GateConfig(min_calibration=-0.1)
    with pytest.raises(ValueError):
        GateConfig(min_calibration=1.5)

def test_gate_refuses_single_class_held_out(tmp_path: Path) -> None:
    # A held-out split with only one class cannot measure separation: refuse closed.
    rows = [{"comment_id": f"a{i}", "text": f"solid grounding {i}", "label": "accepted",
             "labeler_policy_version": "980-policy-r1"} for i in range(4)]
    rows += [{"comment_id": f"r{i}", "text": f"noise noise {i}", "label": "rejected",
              "labeler_policy_version": "980-policy-r1"} for i in range(46)]
    labels = tmp_path / "skew.jsonl"
    labels.write_text("\n".join(json.dumps(r) for r in rows))
    frozen = freeze_split(labels, held_out_fraction=0.5, seed=14)
    model = train_outcome_model(labels, split=frozen, seed=11)
    with pytest.raises(RuntimeError, match="class"):
        evaluate_gate(model, frozen, GateConfig())
