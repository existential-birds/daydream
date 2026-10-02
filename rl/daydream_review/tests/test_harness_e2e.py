"""Run complete rollouts through real interception, CLI, archive, and scoring boundaries.

The canned upstream proves environment injection, authentication, trace capture,
artifact retrieval, intrinsic reward, and suite telemetry wiring; its reward
values carry no quality evidence. Only the opt-in live rollout uses a real model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import PROJECT_ROOT

from daydream_review.fixture import build_fixture_repo

REQUIRED_REWARDS = {"intrinsic_composite"}


def _stage(root: Path) -> dict[str, Path]:
    repo = root / "repo"
    build_fixture_repo(repo)
    archive = root / "archive"
    home = root / "home"
    for path in (archive, home):
        path.mkdir(parents=True, exist_ok=True)
    return {"repo": repo, "archive": archive, "home": home, "out": root / "out"}


def _run_eval(paths: dict[str, Path], *, model: str, base_url: str | None, backend: str | None = None,
) -> subprocess.CompletedProcess[str]:
    argv = ["uv", "run", "eval", "@", "configs/eval-stub.toml", "-m", model, "--no-rich", "-o", str(paths["out"]),
        "--harness.repo-path", str(paths["repo"]), "--harness.archive-root", str(paths["archive"]), "--harness.home",
        str(paths["home"]),
    ]
    if base_url is not None:
        argv += ["--client.base-url", base_url]
    if backend is not None:
        argv += ["--harness.backend", backend]
    return subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)


def _assert_reward_bounds(trace: dict[str, Any]) -> None:
    assert REQUIRED_REWARDS <= set(trace["rewards"]), trace["rewards"]
    reward = trace["rewards"]["intrinsic_composite"]
    score = reward["score"] if isinstance(reward, dict) else reward
    assert 0.0 <= score <= 1.0, trace["rewards"]
    assert trace["metrics"]["suite_non_regression"] in (0.0, 1.0), trace["metrics"]


def _sole_trace(paths: dict[str, Path]) -> dict[str, Any]:
    # verifiers 0.2.1 appends one full trace per line (no Episode wrapper)
    out = paths["out"]
    traces = sorted(out.rglob("traces.jsonl"))
    assert traces, f"no traces.jsonl under {out}"
    lines = traces[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, f"expected one rollout, got {len(lines)}"
    return dict(json.loads(lines[0]))


@pytest.mark.skipif(shutil.which("claude") is None, reason="the claude CLI is not on PATH")
def test_stub_rollout_scores_without_crash(tmp_path: Path, stub_upstream: str) -> None:
    paths = _stage(tmp_path)

    result = _run_eval(paths, model="stub/canned", base_url=stub_upstream)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    episode = _sole_trace(paths)
    assert episode["is_completed"] is True, episode.get("errors")
    assert episode["errors"] == []
    trace = episode
    # Sampled turns prove both interception and the expected tool dialect; request counts alone
    # cannot establish either.
    sampled = [node for node in trace["nodes"] if node.get("sampled")]
    assert sampled, "no sampled assistant turns — endpoint injection or the dialect did not work"
    _assert_reward_bounds(trace)
    assert trace["info"]["daydream_backend"] == "claude"
    assert trace["info"]["daydream_exit_code"] == 0
    assert trace["info"]["reward_breakdown"]["reward_version"]
    assert list((paths["archive"] / "runs").iterdir()), "daydream archived nothing"

@pytest.mark.skipif(not os.environ.get("DAYDREAM_RL_LIVE_E2E"),
    reason="set DAYDREAM_RL_LIVE_E2E=1, DAYDREAM_RL_LIVE_MODEL and DAYDREAM_RL_LIVE_BASE_URL to run",
)
def test_live_rollout(tmp_path: Path) -> None:
    """One full deep rollout against a real model. Never runs in CI.

    ``DAYDREAM_RL_LIVE_BACKEND`` selects the strategy (default claude);
    ``DAYDREAM_RL_LIVE_BASE_URL`` and ``DAYDREAM_RL_LIVE_MODEL`` name the
    upstream the interception server forwards to.
    """
    paths = _stage(tmp_path)
    # Require an explicit live model to avoid sending billed traffic to a guessed endpoint.
    model = os.environ["DAYDREAM_RL_LIVE_MODEL"]
    base_url = os.environ.get("DAYDREAM_RL_LIVE_BASE_URL")
    backend = os.environ.get("DAYDREAM_RL_LIVE_BACKEND", "claude")

    result = _run_eval(paths, model=model, base_url=base_url, backend=backend)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    trace = _sole_trace(paths)
    _assert_reward_bounds(trace)
    assert trace["info"]["daydream_backend"] == backend
