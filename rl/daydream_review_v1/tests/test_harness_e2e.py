"""Phase 4: whole rollouts, through the real verifiers entrypoint.

``test_stub_rollout_scores_without_crash`` is the resilience path: everything
real except the model. A canned upstream answers every request with the same
useless sentence, so the rewards are meaningless — what it proves is the
plumbing, and it proves all of it at once. Env injection reaches the CLI, the CLI
speaks the Anthropic dialect to the interception server and authenticates with
the rollout secret, every turn lands in the trace DAG, daydream archives a run,
the run dir comes back out of the sandbox, and both reward axes score.

``test_live_rollout`` is the only test that needs a real model, and it is the
only one whose reward values mean anything.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import PROJECT_ROOT

from daydream_review_v1.fixture import build_fixture_repo

REQUIRED_REWARDS = {"intrinsic_composite", "fix_tests_pass"}


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
        "--harness.repo-path",
        str(paths["repo"]),
        "--harness.archive-root",
        str(paths["archive"]),
        "--harness.home",
        str(paths["home"]),
    ]
    if base_url is not None:
        argv += ["--client.base-url", base_url]
    return subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)


def _sole_trace(paths: dict[str, Path]) -> dict:
    lines = (paths["out"] / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, f"expected one rollout, got {len(lines)}"
    return json.loads(lines[0])


@pytest.mark.skipif(shutil.which("claude") is None, reason="the claude CLI is not on PATH")
def test_stub_rollout_scores_without_crash(tmp_path: Path, stub_upstream: str) -> None:
    paths = _stage(tmp_path)

    result = _run_eval(paths, model="stub/canned", base_url=stub_upstream)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    trace = _sole_trace(paths)
    assert trace["stop_condition"] == "agent_completed"
    assert trace["errors"] == []
    # Sampled assistant nodes exist only when a model turn went through the
    # interception server AND the dialect parsed the reply. A harness that
    # reached a provider directly records nothing; a stub whose payload fails the
    # dialect's strict model records calls but no sampled turns, which would let
    # the run exercise the retry path while claiming to prove the dialect.
    sampled = [node for node in trace["nodes"] if node.get("sampled")]
    assert sampled, "no sampled assistant turns — endpoint injection or the dialect did not work"
    assert REQUIRED_REWARDS <= set(trace["rewards"]), trace["rewards"]
    assert trace["info"]["daydream_backend"] == "claude"
    assert trace["info"]["daydream_exit_code"] == 0
    # Scoring really read daydream's archived run dir out of the sandbox.
    assert trace["info"]["reward_breakdown"]["reward_version"]
    assert list((paths["archive"] / "runs").iterdir()), "daydream archived nothing"


@pytest.mark.skipif(
    not os.environ.get("DAYDREAM_RL_LIVE_E2E"),
    reason="set DAYDREAM_RL_LIVE_E2E=1 (plus a real upstream in DAYDREAM_RL_LIVE_BASE_URL) to run",
)
def test_live_rollout(tmp_path: Path) -> None:
    """One full deep rollout against a real model. Never runs in CI.

    ``DAYDREAM_RL_LIVE_BACKEND`` selects the strategy (default claude);
    ``DAYDREAM_RL_LIVE_BASE_URL`` and ``DAYDREAM_RL_LIVE_MODEL`` name the
    upstream the interception server forwards to.
    """
    paths = _stage(tmp_path)
    model = os.environ.get("DAYDREAM_RL_LIVE_MODEL", "claude-sonnet-5")
    base_url = os.environ.get("DAYDREAM_RL_LIVE_BASE_URL")
    backend = os.environ.get("DAYDREAM_RL_LIVE_BACKEND", "claude")

    argv_extra = ["--harness.backend", backend]
    result = subprocess.run(
        [
            "uv", "run", "eval", "@", "configs/eval-stub.toml",
            "-m", model, "--no-rich", "-o", str(paths["out"]),
            "--harness.repo-path", str(paths["repo"]),
            "--harness.archive-root", str(paths["archive"]),
            "--harness.home", str(paths["home"]),
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
    assert trace["rewards"]["fix_tests_pass"] in (0.0, 1.0)
    assert trace["info"]["daydream_backend"] == backend
