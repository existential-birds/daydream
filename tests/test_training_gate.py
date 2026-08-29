import json
from pathlib import Path

import pytest

from daydream.training.gate import FrozenSplit, GateConfig, evaluate_gate, freeze_split
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
def frozen_split(tmp_path: Path) -> FrozenSplit:
    return freeze_split(_pairs(tmp_path), held_out_fraction=0.2, seed=3)


@pytest.fixture
def trained_model(tmp_path: Path) -> OutcomeModel:
    return train_outcome_model(_pairs(tmp_path), split={"train": 0.8, "held_out": 0.2}, seed=3)


def test_split_frozen_before_training(tmp_path: Path) -> None:
    labels = _pairs(tmp_path)
    frozen = freeze_split(labels, held_out_fraction=0.2, seed=3)
    assert frozen.fingerprint != ""
    assert (tmp_path / frozen.digest_path).exists()
    # frozen split is content-addressed: same seed+labels -> same digest
    again = freeze_split(labels, held_out_fraction=0.2, seed=3)
    assert again.digest == frozen.digest
    # digest file records the split digest for the resume guard (M18)
    payload = json.loads((tmp_path / frozen.digest_path).read_text())
    assert payload["digest"] == frozen.digest
    # a second freeze with a different seed rewrites the sidecar with its own digest
    other = freeze_split(labels, held_out_fraction=0.2, seed=4)
    assert other.digest != frozen.digest
    assert json.loads((tmp_path / frozen.digest_path).read_text())["digest"] == other.digest


def test_gate_pass_separates_classes(frozen_split: FrozenSplit, trained_model: OutcomeModel) -> None:
    report = evaluate_gate(
        trained_model, frozen_split, GateConfig(min_separation=0.1, min_calibration=0.5)
    )
    assert report.passed
    assert report.separation > 0
    assert report.evidence_digest
    assert report.to_dict()["separation"] == report.separation  # JSON-serializable


def test_gate_refuses_when_evidence_missing(trained_model: OutcomeModel) -> None:
    with pytest.raises(RuntimeError, match="gate evidence"):
        evaluate_gate(trained_model, None, GateConfig())


def test_label_ratio_reported_not_stale(frozen_split: FrozenSplit, trained_model: OutcomeModel) -> None:
    report = evaluate_gate(trained_model, frozen_split, GateConfig())
    assert report.accepted_ratio is not None  # S2: measured at gate time
    assert 0.0 <= report.accepted_ratio <= 1.0


def test_gate_fails_below_thresholds(frozen_split: FrozenSplit, trained_model: OutcomeModel) -> None:
    report = evaluate_gate(
        trained_model, frozen_split, GateConfig(min_separation=0.99, min_calibration=0.99)
    )
    assert not report.passed
    assert report.thresholds == {"min_separation": 0.99, "min_calibration": 0.99}


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
    model = train_outcome_model(labels, split={"train": 0.5, "held_out": 0.5}, seed=11)
    with pytest.raises(RuntimeError, match="class"):
        evaluate_gate(model, frozen, GateConfig())
