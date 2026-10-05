"""Verify Stage-0 rubric composition and the shipped training configs' Pi backend and gate-report pins."""

from __future__ import annotations

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
    assert task.config.outcome_model_path == outcome_model_path

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
    # No outcome model short-circuits before any run artifact is read, so a real
    # (empty) directory is the whole input this branch needs.
    assert stage0_composite_terms(Path(""), tmp_path) is None

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
