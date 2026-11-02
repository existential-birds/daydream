"""Fail-open, session-bound quality gate evidence at its owner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.deep.artifacts import DeepArtifact
from daydream.deep.quality_gate import QualityGateThresholds, _evaluate_quality_gate
from daydream.eval import analyzer as analyzer_mod
from tests.deep_orchestrator.support import _capture_warnings


async def test_quality_gate_records_fail_open_session_bound_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every unavailable stage, prior-artifact defect, and flag rule persists auditable evidence.

    Runner fix tests reach the gate but cannot cheaply vary analyzer failures,
    foreign-session artifacts, or each per-file flag rule; a regression that reads
    an undefined metric or a broken file as a clean pass would only show here.
    """
    warnings = _capture_warnings(monkeypatch, "daydream.deep.quality_gate.print_warning")
    dd = tmp_path / "deep"
    dd.mkdir()
    gate = DeepArtifact.FIX_QUALITY_GATE.at(dd)
    thresholds = QualityGateThresholds(
        erosion_delta=0.1, verbosity_delta=0.1, erosion_absolute=1.0, verbosity_absolute=1.0,
    )
    after: list[Any] = []

    def analyze(_daydream_dir: Path, _candidates: set[str] | None, **_kwargs: Any) -> Any:
        if not after:
            raise RuntimeError("analyzer down")
        return after[0]

    monkeypatch.setattr(analyzer_mod, "analyze_quality", analyze)

    async def evaluate(**overrides: Any) -> dict[str, Any]:
        await _evaluate_quality_gate(**{
            "enabled": True, "thresholds": thresholds, "daydream_dir": tmp_path, "code_workspace": tmp_path,
            "dd": dd, "candidates": {"a.py"}, "before": {"per_file": {}}, "before_unavailable_reason": None,
            "iteration": None, **overrides,
        })
        payload: dict[str, Any] = json.loads(gate.read_text())
        return payload

    assert await evaluate(enabled=False) == {"enabled": False, "session_id": None}

    gate.write_text(json.dumps({"session_id": "prior-run", "rounds": [{"round": 1}]}))
    persisted = await evaluate(candidates=None)
    assert persisted["rounds"] == [{"round": 1, "unavailable": {
        "stage": "candidates",
        "reason": "could not enumerate files changed by the fix pass against the pre-fix snapshot"}}]
    assert persisted["erosion_absolute_threshold"] == 1.0
    assert any("another run's session" in warning for warning in warnings)

    gate.write_text("{malformed")
    persisted = await evaluate(before=None, before_unavailable_reason="snapshot failed", iteration=3)
    assert persisted["rounds"] == [{"round": 3, "unavailable": {"stage": "before", "reason": "snapshot failed"}}]

    gate.write_text(json.dumps({"session_id": None, "rounds": "invalid"}))
    persisted = await evaluate()
    assert persisted["rounds"][0]["unavailable"] == {"stage": "after", "reason": "RuntimeError: analyzer down"}

    gate.write_text(json.dumps({"session_id": None, "rounds": [{"round": 1, "per_file": {}}, "invalid"]}))
    after.append({"per_file": {
        "delta.py": {"erosion": 1.5, "verbosity": 0.2}, "verbose.py": {"erosion": 0.5, "verbosity": 0.8},
        "new-baseline.py": {"erosion": 0.1, "verbosity": 0.1}, "absolute.py": {"erosion": 2.0, "verbosity": 0.5},
        "absolute-verbosity.py": {"erosion": 0.1, "verbosity": 2.0}, "clean.py": {"erosion": 0.5, "verbosity": 0.5},
    }})
    persisted = await evaluate(
        candidates={"delta.py", "verbose.py", "broken.py", "new-baseline.py", "absolute.py",
                    "absolute-verbosity.py", "clean.py", "absent.py"},
        before={"per_file": {
            "delta.py": {"erosion": 1.0, "verbosity": 0.2}, "verbose.py": {"erosion": 0.5, "verbosity": 0.5},
            "broken.py": {"erosion": 0.1, "verbosity": 0.1}, "absolute.py": {"erosion": None, "verbosity": 0.5},
            "absolute-verbosity.py": {"erosion": 0.1, "verbosity": None},
            "clean.py": {"erosion": 0.5, "verbosity": 0.5},
        }},
    )
    assert [row["round"] for row in persisted["rounds"]] == [1, 2]
    per_file = persisted["rounds"][1]["per_file"]
    assert "absent.py" not in per_file
    assert {name: entry["flagged"] for name, entry in per_file.items()} == {
        "absolute-verbosity.py": True, "absolute.py": True, "broken.py": True, "clean.py": False,
        "delta.py": True, "new-baseline.py": True, "verbose.py": True,
    }
    assert per_file["broken.py"]["unparseable"] is True and per_file["delta.py"]["erosion_delta"] == 0.5
    assert "missing pre-fix baseline" in per_file["new-baseline.py"]["reason"]
    assert any("dropping them" in warning for warning in warnings)
    assert any("flagged 6 file(s)" in warning and "erosion 1.0 -> 1.5" in warning for warning in warnings)

    after[0] = ["not a quality payload"]
    persisted = await evaluate(iteration=4)
    assert persisted["rounds"][-1]["unavailable"]["stage"] == "persist"

    gate.unlink()
    gate.mkdir()
    await _evaluate_quality_gate(
        enabled=True, thresholds=thresholds, daydream_dir=tmp_path, code_workspace=tmp_path, dd=dd,
        candidates=None, before=None, before_unavailable_reason=None, iteration=None,
    )
    assert "could not persist unavailable verdict" in warnings[-1]
    assert any("rounds list" in warning for warning in warnings)
    assert any("malformed" in warning for warning in warnings)
