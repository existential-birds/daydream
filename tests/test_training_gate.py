import json
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.training.gate import FrozenSplit, GateConfig, _build_frozen_split, evaluate_gate
from daydream.training.reward_model import OutcomeModel, train_outcome_model


def _pairs(n: int = 25) -> list[dict[str, Any]]:
    rows = []
    for i in range(n):
        rows.append({"comment_id": f"a{i}", "text": f"solid grounding {i}", "label": "accepted",
                     "labeler_policy_version": "980-policy-r1"})
        rows.append({"comment_id": f"r{i}", "text": f"noise noise {i}", "label": "rejected",
                     "labeler_policy_version": "980-policy-r1"})
    return rows


def _frozen_pairs(tmp_path: Path, n: int = 25, seed: int = 3) -> FrozenSplit:
    rows = _pairs(n)
    boundary = 2 * (n - max(1, n // 5))
    return _build_frozen_split(
        tmp_path / "labels.jsonl", train_rows=rows[:boundary], held_out_rows=rows[boundary:],
        seed=seed, held_out_fraction=0.2,
    )


@pytest.fixture
def frozen_split(tmp_path: Path) -> FrozenSplit:
    return _frozen_pairs(tmp_path)


@pytest.fixture
def trained_model(frozen_split: FrozenSplit) -> OutcomeModel:
    return train_outcome_model(frozen_split, seed=3)


def test_split_frozen_before_training(tmp_path: Path) -> None:
    frozen = _frozen_pairs(tmp_path)
    assert frozen.fingerprint != ""
    assert frozen.digest_path == "labels.jsonl.gate-split.json"
    sidecar_path = tmp_path / frozen.digest_path
    assert sidecar_path.exists()
    # Pinned membership and the seed retain the existing sidecar content address.
    first_bytes = sidecar_path.read_bytes()
    again = _frozen_pairs(tmp_path)
    assert again.digest == frozen.digest
    assert sidecar_path.read_bytes() == first_bytes
    expected = {"digest": frozen.digest, "held_out_fraction": 0.2,
        "held_out_ids": sorted(str(row["comment_id"]) for row in frozen.held_out_rows), "seed": 3,
        "train_ids": sorted(str(row["comment_id"]) for row in frozen.train_rows),
    }
    assert first_bytes == json.dumps(expected, indent=2, sort_keys=True).encode()
    assert not first_bytes.endswith(b"\n")
    assert json.loads(first_bytes) == expected
    other = _frozen_pairs(tmp_path, seed=4)
    assert other.digest != frozen.digest
    assert json.loads(sidecar_path.read_text())["digest"] == other.digest

def test_gate_pass_separates_classes(frozen_split: FrozenSplit, trained_model: OutcomeModel) -> None:
    report = evaluate_gate(trained_model, frozen_split, GateConfig(min_separation=0.1, min_calibration=0.5))
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
    report = evaluate_gate(trained_model, frozen_split, GateConfig(min_separation=0.99, min_calibration=0.99))
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
    rows = _pairs(2)
    frozen = _build_frozen_split(
        tmp_path / "labels.jsonl", train_rows=rows[:3], held_out_rows=rows[3:],
        seed=14, held_out_fraction=0.5,
    )
    model = train_outcome_model(frozen, seed=11)
    with pytest.raises(RuntimeError, match="class"):
        evaluate_gate(model, frozen, GateConfig())


@pytest.mark.parametrize("change", [
    {"comment_id": None}, {"comment_id": 42}, {"text": None}, {"text": 42},
    {"label": "ambiguous"}, {"label": []},
    {"has_posterior": False}, {"labeler_policy_version": None},
    {"decisive_mix": True}, {"decisive_only": False},
])
def test_split_refuses_invalid_outcomes_before_publication(tmp_path: Path, change: dict[str, Any]) -> None:
    rows = _pairs(2)
    rows[0].update(change)
    with pytest.raises(ValueError):
        _build_frozen_split(tmp_path / "labels.jsonl", train_rows=rows[:2], held_out_rows=rows[2:],
                            seed=0, held_out_fraction=0.5)
    assert not (tmp_path / "labels.jsonl.gate-split.json").exists()


def test_split_freezes_the_population_used_for_counts_training_and_gate(tmp_path: Path) -> None:
    rows = _pairs(2)
    train, held_out = rows[:2], rows[2:]
    split = _build_frozen_split(tmp_path / "labels.jsonl", train_rows=train, held_out_rows=held_out,
                                seed=0, held_out_fraction=0.5)
    before = train_outcome_model(split, seed=0)
    report = evaluate_gate(before, split, GateConfig())
    rows[0]["label"] = "rejected"
    rows[2]["text"] = "changed after admission"
    train.clear()
    held_out.clear()
    assert train_outcome_model(split, seed=0).state_dict() == before.state_dict()
    assert before.label_ratio_reported == 0.5
    assert evaluate_gate(before, split, GateConfig()) == report
    with pytest.raises(TypeError):
        cast(Any, split.train_rows)[0]["label"] = "rejected"
    with pytest.raises(TypeError):
        cast(Any, split.held_out_rows)[0]["text"] = "changed"


@pytest.mark.parametrize("rows", [[], [None]])
def test_split_refuses_empty_or_nonobject_holdout(tmp_path: Path, rows: list[Any]) -> None:
    with pytest.raises(ValueError, match="held-out rows|must be an object"):
        _build_frozen_split(tmp_path / "labels.jsonl", train_rows=_pairs(1), held_out_rows=rows,
                            seed=0, held_out_fraction=0.5)
    assert not (tmp_path / "labels.jsonl.gate-split.json").exists()
