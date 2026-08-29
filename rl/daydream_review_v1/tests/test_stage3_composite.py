"""Stage-3 composite integration (M13) + pi-only train backend pin (M24).

The env's scoring path must compose the validated Stage-0 rubric (rubric_v2)
with the intrinsic composite when an outcome model is configured, and every
train environment in ``rl/train/`` must pin ``backend = "pi"`` and carry a
Stage-0 gate report path — the M4 refusal has to hold for the shipped configs,
not just the loader.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes.subprocess import SubprocessRuntime

from daydream_review_v1.taskset import (
    DaydreamReviewConfig,
    DaydreamReviewState,
    DaydreamReviewTaskset,
    stage0_composite_terms,
)

RL_TRAIN_DIR = Path(__file__).resolve().parents[2] / "train"

SESSION_ID = "9b36227a-9f80-41e5-a419-5cfed5a34b5b"


@pytest.fixture
def mini_taskset(
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
    stage0_gate_report: Path,
    outcome_model_path: Path,
) -> DaydreamReviewTaskset:
    return DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
            use_images=False,
            gate_report_path=stage0_gate_report,
            outcome_model_path=outcome_model_path,
        )
    )


@pytest.fixture
def rl_train_configs() -> list[dict[str, Any]]:
    """Every shipped training config, parsed."""
    return [
        tomllib.loads(p.read_text(encoding="utf-8")) for p in sorted(RL_TRAIN_DIR.glob("*.toml"))
    ]


def _stage_run_dir(tmp_path: Path) -> Path:
    """A minimal archived run dir: merged findings + a manifest with grounding."""
    run_dir = tmp_path / "run"
    deep = run_dir / "deep"
    deep.mkdir(parents=True)
    (deep / "merged-items.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": 1,
                        "description": "off-by-one in add() makes every sum wrong",
                        "file": "calc.py",
                        "line": 4,
                        "confidence": "HIGH",
                        "rationale": "test contradicts implementation",
                        "evidence": "test_add fails",
                        "lens": "per-stack",
                        "severity": "high",
                        "related_files": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"metrics": {"grounding_rate": 1.0}}), encoding="utf-8"
    )
    return run_dir


async def test_env_scores_with_stage0_composite(
    mini_taskset: DaydreamReviewTaskset,
    tmp_path: Path,
    runtime: SubprocessRuntime,
    rundir_golden: Path,
    outcome_model_path: Path,
) -> None:
    """The scoring path composes rubric_v2 terms, not intrinsic-only (M13).

    Drives the reward through the production entrypoint (``await task.score``),
    where ``intrinsic_composite`` composes the Stage-0 rubric only because the
    per-task config carried ``outcome_model_path``. A caller regression that
    drops that path before the reward reads it fails here — the asserted
    ``stage0`` breakdown would be absent and the reward would be intrinsic-only.
    """
    tasks = mini_taskset.load()
    assert tasks, "gate passed but no tasks loaded"
    task = tasks[0]
    # The taskset stamps outcome_model_path onto each per-task config; the
    # reward composes rubric_v2 only when it actually reads the path (M13).
    assert task.config.outcome_model_path == outcome_model_path

    archive_root = tmp_path / "archive"
    dest = archive_root / "runs" / SESSION_ID
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(rundir_golden, dest)
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=DaydreamReviewState(),
    )
    trace.info["daydream_archive_root"] = str(archive_root)
    trace.info["daydream_repo_path"] = str(tmp_path / "repo")

    await task.score(trace, runtime)

    breakdown = trace.info["reward_breakdown"]
    stage0 = breakdown.get("stage0")
    assert stage0 is not None, (
        "outcome_model_path was dropped before the reward: no rubric composite composed (M13)"
    )
    # The scored breakdown carries the rubric_v2 terms, not intrinsic-only.
    assert "learned_outcome" in stage0["terms"]
    assert "fp_penalty" in stage0["terms"]
    assert "localization" in stage0["terms"]
    assert stage0["composite"] is not None
    assert stage0["reward_version"]  # rubric version stamped for provenance
    # M13: the reward IS the rubric composite (which carries the intrinsic
    # composite as one weighted term), not the intrinsic-only value.
    assert stage0["terms"]["intrinsic_composite"] is not None
    assert trace.rewards["intrinsic_composite"] == stage0["composite"]


def test_stage0_composition_absent_without_model(tmp_path: Path) -> None:
    """No outcome model configured → no Stage-0 composition (intrinsic-only)."""
    assert stage0_composite_terms(Path(""), _stage_run_dir(tmp_path)) is None


def test_backend_config_pi_only(rl_train_configs: list[dict[str, Any]]) -> None:
    """M24: only pi may be configured for training runs."""
    checked = 0
    for cfg in rl_train_configs:
        envs: list[dict[str, Any]] = cfg.get("orchestrator", {}).get("train", {}).get("env", [])
        for env in envs:
            assert env["harness"]["backend"] == "pi"
            checked += 1
    assert checked >= 1, "no train env found — the backend pin test must not pass vacuously"


def test_train_envs_carry_stage0_gate(rl_train_configs: list[dict[str, Any]]) -> None:
    """M4 at the config level: shipped train envs name a Stage-0 gate report."""
    checked = 0
    for cfg in rl_train_configs:
        envs: list[dict[str, Any]] = cfg.get("orchestrator", {}).get("train", {}).get("env", [])
        for env in envs:
            taskset_cfg = env["taskset"]
            assert taskset_cfg.get("gate_report_path"), f"train env {env.get('name')!r} has no gate_report_path"
            assert taskset_cfg.get("outcome_model_path"), f"train env {env.get('name')!r} has no outcome_model_path"
            checked += 1
    assert checked >= 1, "no train env found — the gate config test must not pass vacuously"
