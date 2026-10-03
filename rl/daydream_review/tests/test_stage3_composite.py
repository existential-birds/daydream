"""Verify Stage-0 rubric composition and the shipped training configs' Pi backend and gate-report pins."""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes.subprocess import SubprocessRuntime

from daydream_review.taskset import (
    DaydreamReviewConfig,
    DaydreamReviewState,
    DaydreamReviewTaskset,
    stage0_composite_terms,
)

MODEL = "some-org/some-policy-model"

RL_TRAIN_DIR = Path(__file__).resolve().parents[2] / "train"

SESSION_ID = "9b36227a-9f80-41e5-a419-5cfed5a34b5b"


@pytest.fixture
def mini_taskset(fixture_manifest_path: Path, stage0_gate_report: Path, outcome_model_path: Path,
) -> DaydreamReviewTaskset:
    return DaydreamReviewTaskset(DaydreamReviewConfig(id="daydream-review",
                        manifest_path=fixture_manifest_path,
            use_images=False, gate_report_path=stage0_gate_report, outcome_model_path=outcome_model_path,
        )
    )


@pytest.fixture
def rl_train_configs() -> list[dict[str, Any]]:
    return [tomllib.loads(p.read_text(encoding="utf-8")) for p in sorted(RL_TRAIN_DIR.glob("*.toml"))]


def _stage_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    deep = run_dir / "deep"
    deep.mkdir(parents=True)
    (deep / "merged-items.json").write_text(json.dumps({"items": [{
                        "id": 1, "description": "off-by-one in add() makes every sum wrong", "file": "calc.py",
                        "line": 4, "confidence": "HIGH", "rationale": "test contradicts implementation",
                        "evidence": "test_add fails", "lens": "per-stack", "severity": "high", "related_files": None,
                    }
                ]
            }
        ), encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
    return run_dir


async def test_env_scores_with_stage0_composite(
    mini_taskset: DaydreamReviewTaskset, tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path,
    outcome_model_path: Path,
) -> None:
    """Drive task.score with outcome_model_path and require the stage0 breakdown, proving the path survives
    the loader-to-reward handoff.
    """
    tasks = mini_taskset.load()
    assert tasks, "gate passed but no tasks loaded"
    task = tasks[0]
    assert task.config._outcome_scorer is not None
    assert "_outcome_scorer" not in task.config.model_dump()

    archive_root = tmp_path / "archive"
    dest = archive_root / "runs" / SESSION_ID
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(rundir_golden, dest)
    trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(model=MODEL), state=DaydreamReviewState(),
    )
    trace.info["daydream_archive_root"] = str(archive_root)
    trace.info["daydream_repo_path"] = str(tmp_path / "repo")

    await task.score(trace, runtime)

    breakdown = trace.info["reward_breakdown"]
    stage0 = breakdown.get("stage0")
    assert stage0 is not None, ("outcome_model_path was dropped before the reward: no rubric composite composed (M13)"
    )
    assert "learned_outcome" in stage0["terms"]
    assert "fp_penalty" in stage0["terms"]
    assert "localization" not in stage0["terms"]
    assert stage0["composite"] is not None
    assert stage0["reward_version"]  # rubric version stamped for provenance
    assert stage0["terms"]["intrinsic_composite"] is None  # no verifier verdicts in this rollout
    assert trace.rewards["intrinsic_composite"] == stage0["composite"]

def test_stage0_composition_absent_without_model(tmp_path: Path) -> None:
    assert stage0_composite_terms(None, _stage_run_dir(tmp_path)) is None

def test_backend_config_pi_only(rl_train_configs: list[dict[str, Any]]) -> None:
    checked = 0
    for cfg in rl_train_configs:
        envs: list[dict[str, Any]] = cfg.get("orchestrator", {}).get("train", {}).get("env", [])
        for env in envs:
            assert env["harness"]["backend"] == "pi"
            checked += 1
    assert checked >= 1, "no train env found — the backend pin test must not pass vacuously"

def test_train_envs_carry_stage0_gate(rl_train_configs: list[dict[str, Any]]) -> None:
    checked = 0
    for cfg in rl_train_configs:
        envs: list[dict[str, Any]] = cfg.get("orchestrator", {}).get("train", {}).get("env", [])
        for env in envs:
            taskset_cfg = env["taskset"]
            assert taskset_cfg.get("gate_report_path"), f"train env {env.get('name')!r} has no gate_report_path"
            assert taskset_cfg.get("outcome_model_path"), f"train env {env.get('name')!r} has no outcome_model_path"
            checked += 1
    assert checked >= 1, "no train env found — the gate config test must not pass vacuously"


async def _native_model_score(task: Any, root: Path, runtime: SubprocessRuntime, golden: Path) -> float:
    archive_root = root / "archive"
    dest = archive_root / "runs" / SESSION_ID
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(golden, dest)
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(model=MODEL), state=DaydreamReviewState(),
    )
    trace.info["daydream_archive_root"] = str(archive_root)
    trace.info["daydream_repo_path"] = str(root / "repo")
    await task.score(trace, runtime)
    return float(trace.info["reward_breakdown"]["stage0"]["terms"]["learned_outcome"])


async def test_admitted_checkpoint_survives_replacement_before_first_score(
    mini_taskset: DaydreamReviewTaskset, tmp_path: Path, runtime: SubprocessRuntime,
    rundir_golden: Path, outcome_model_path: Path,
) -> None:
    from daydream.training.reward_model import OutcomeModel, score_comment

    task = mini_taskset.load()[0]
    model = OutcomeModel(**json.loads(outcome_model_path.read_text()))
    state = model.state_dict() | {"weights": {}, "bias": 8.0, "model_fingerprint": "unbound"}
    outcome_model_path.write_text(json.dumps(state))
    value = await _native_model_score(task, tmp_path, runtime, rundir_golden)
    descriptions = json.loads((rundir_golden / "deep/merged-items.json").read_text())["items"]
    expected = sum(score_comment(model, str(row["description"])) for row in descriptions) / len(descriptions)
    assert value == pytest.approx(expected)
    assert value != pytest.approx(1.0)


async def test_tasksets_capture_distinct_gate_bound_models_at_same_path(
    mini_taskset: DaydreamReviewTaskset, tmp_path: Path, runtime: SubprocessRuntime,
    rundir_golden: Path, outcome_model_path: Path, stage0_gate_report: Path,
) -> None:
    from daydream.training.gate import _evidence_digest
    from daydream.training.reward_model import OutcomeModel

    def publish(bias: float) -> None:
        model = OutcomeModel({}, bias, "fixture-split-digest", 0.5, 10, 4, 0.75)
        outcome_model_path.write_text(json.dumps(model.state_dict()))
        report = json.loads(stage0_gate_report.read_text())
        evidence = {
            key: report[key]
            for key in ("thresholds", "held_out_rows", "separation", "calibration", "accepted_ratio")
        }
        evidence.update(split_digest=model.split_digest, model_fingerprint=model.model_fingerprint)
        report["evidence_digest"] = _evidence_digest(evidence)
        stage0_gate_report.write_text(json.dumps(report))

    publish(-8.0)
    task_a = mini_taskset.load()[0]
    score_a = await _native_model_score(task_a, tmp_path / "a", runtime, rundir_golden)
    publish(8.0)
    task_b = mini_taskset.load()[0]
    score_b = await _native_model_score(task_b, tmp_path / "b", runtime, rundir_golden)
    score_a_again = await _native_model_score(task_a, tmp_path / "a-again", runtime, rundir_golden)
    assert score_a < 0.01
    assert score_b > 0.99
    assert score_a_again == score_a


async def test_admitted_checkpoint_does_not_require_file_during_scoring(
    mini_taskset: DaydreamReviewTaskset, tmp_path: Path, runtime: SubprocessRuntime,
    rundir_golden: Path, outcome_model_path: Path,
) -> None:
    task = mini_taskset.load()[0]
    outcome_model_path.unlink()
    assert await _native_model_score(task, tmp_path, runtime, rundir_golden) > 0.0


async def test_intrinsic_only_taskset_clears_reused_capture(
    mini_taskset: DaydreamReviewTaskset, tmp_path: Path, runtime: SubprocessRuntime,
    rundir_golden: Path,
) -> None:
    admitted = mini_taskset.load()[0]
    config = mini_taskset.config.model_copy(update={"task": admitted.config, "outcome_model_path": Path("")})
    task = DaydreamReviewTaskset(config).load()[0]
    assert task.config._outcome_scorer is None
    assert admitted.config._outcome_scorer is not None
    archive_root = tmp_path / "archive"
    dest = archive_root / "runs" / SESSION_ID
    dest.parent.mkdir(parents=True)
    shutil.copytree(rundir_golden, dest)
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(model=MODEL), state=DaydreamReviewState(),
    )
    trace.info.update(daydream_archive_root=str(archive_root), daydream_repo_path=str(tmp_path / "repo"))
    await task.score(trace, runtime)
    assert "stage0" not in trace.info["reward_breakdown"]
