"""Phase 4: whole rollouts, through the real verifiers entrypoint.

``test_stub_rollout_scores_without_crash`` is the resilience path: everything
real except the model. A canned upstream answers every request with the same
useless sentence, so the rewards are meaningless — what it proves is the
plumbing, and it proves all of it at once. Env injection reaches the CLI, the CLI
speaks the Anthropic dialect to the interception server and authenticates with
the rollout secret, every turn lands in the trace DAG, daydream archives a run,
the run dir comes back out of the sandbox, and the single intrinsic reward
scores alongside the suite_non_regression metric.

``test_live_rollout`` is the only test that needs a real model, and it is the
only one whose reward values mean anything.
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

from daydream_review_v1.fixture import build_fixture_repo

REQUIRED_REWARDS = {"intrinsic_composite"}


def _stage(root: Path) -> dict[str, Path]:
    """Lay out the three directories the harness config points at."""
    repo = root / "repo"
    build_fixture_repo(repo)
    archive = root / "archive"
    home = root / "home"
    for path in (archive, home):
        path.mkdir(parents=True, exist_ok=True)
    return {"repo": repo, "archive": archive, "home": home, "out": root / "out"}


def _run_eval(paths: dict[str, Path], *, model: str, base_url: str | None) -> subprocess.CompletedProcess[str]:
    argv = [
        "uv",
        "run",
        "eval",
        "@",
        "configs/eval-stub.toml",
        "-m",
        model,
        "--no-rich",
        "-o",
        str(paths["out"]),
        "--env.agent.harness.repo-path",
        str(paths["repo"]),
        "--env.agent.harness.archive-root",
        str(paths["archive"]),
        "--env.agent.harness.home",
        str(paths["home"]),
    ]
    if base_url is not None:
        argv += ["--client.base-url", base_url]
    return subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)


def _sole_trace(paths: dict[str, Path]) -> dict[str, Any]:
    # verifiers 0.3.1 groups runs: traces land at output_dir/run.dir/traces.jsonl
    out = paths["out"]
    traces = sorted(out.rglob("traces.jsonl"))
    assert traces, f"no traces.jsonl under {out}"
    lines = traces[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, f"expected one rollout, got {len(lines)}"
    # json.loads yields Any; we only need a shallow dict view.
    return dict(json.loads(lines[0]))


@pytest.mark.skipif(shutil.which("claude") is None, reason="the claude CLI is not on PATH")
def test_stub_rollout_scores_without_crash(tmp_path: Path, stub_upstream: str) -> None:
    paths = _stage(tmp_path)

    result = _run_eval(paths, model="stub/canned", base_url=stub_upstream)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    episode = _sole_trace(paths)
    # 0.3.1 wraps traces in an Episode: episode.ok, episode.errors, one trace inside
    assert episode["ok"] is True, episode.get("errors")
    assert episode["errors"] == []
    trace = episode["traces"][0]
    # Sampled assistant nodes exist only when a model turn went through the
    # interception server AND the dialect parsed the reply. A harness that
    # reached a provider directly records nothing; a stub whose payload fails the
    # dialect's strict model records calls but no sampled turns, which would let
    # the run exercise the retry path while claiming to prove the dialect.
    sampled = [node for node in trace["nodes"] if node.get("sampled")]
    assert sampled, "no sampled assistant turns — endpoint injection or the dialect did not work"
    assert REQUIRED_REWARDS <= set(trace["rewards"]), trace["rewards"]
    rw = trace["rewards"]["intrinsic_composite"]
    rw = rw["score"] if isinstance(rw, dict) else rw
    assert 0.0 <= rw <= 1.0, trace["rewards"]
    assert trace["metrics"]["suite_non_regression"] in (0.0, 1.0), trace["metrics"]
    assert trace["info"]["daydream_backend"] == "claude"
    assert trace["info"]["daydream_exit_code"] == 0
    # Scoring really read daydream's archived run dir out of the sandbox.
    assert trace["info"]["reward_breakdown"]["reward_version"]
    assert list((paths["archive"] / "runs").iterdir()), "daydream archived nothing"


@pytest.mark.skipif(
    not os.environ.get("DAYDREAM_RL_LIVE_E2E"),
    reason="set DAYDREAM_RL_LIVE_E2E=1, DAYDREAM_RL_LIVE_MODEL and DAYDREAM_RL_LIVE_BASE_URL to run",
)
def test_live_rollout(tmp_path: Path) -> None:
    """One full deep rollout against a real model. Never runs in CI.

    ``DAYDREAM_RL_LIVE_BACKEND`` selects the strategy (default claude);
    ``DAYDREAM_RL_LIVE_BASE_URL`` and ``DAYDREAM_RL_LIVE_MODEL`` name the
    upstream the interception server forwards to.
    """
    paths = _stage(tmp_path)
    # No default: a model id baked into this repo would be exactly the hardcoding
    # SPEC C1 forbids, and a wrong one silently bills the wrong endpoint.
    model = os.environ["DAYDREAM_RL_LIVE_MODEL"]
    base_url = os.environ.get("DAYDREAM_RL_LIVE_BASE_URL")
    backend = os.environ.get("DAYDREAM_RL_LIVE_BACKEND", "claude")

    argv_extra = ["--env.agent.harness.backend", backend]
    result = subprocess.run(
        [
            "uv", "run", "eval", "@", "configs/eval-stub.toml",
            "-m", model, "--no-rich", "-o", str(paths["out"]),
            "--env.agent.harness.repo-path", str(paths["repo"]),
            "--env.agent.harness.archive-root", str(paths["archive"]),
            "--env.agent.harness.home", str(paths["home"]),
            *(["--client.base-url", base_url] if base_url else []),
            *argv_extra,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    trace = _sole_trace(paths)
    assert REQUIRED_REWARDS <= set(trace["rewards"]), trace["rewards"]
    rw2 = trace["rewards"]["intrinsic_composite"]
    rw2 = rw2["score"] if isinstance(rw2, dict) else rw2
    assert 0.0 <= rw2 <= 1.0
    assert trace["metrics"]["suite_non_regression"] in (0.0, 1.0)
    assert trace["info"]["daydream_backend"] == backend
