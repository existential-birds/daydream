"""Parametrized cross-driver real-path test — one body, driver is a parameter.

Drives the genuine backend (``CodexBackend`` *and* ``ClaudeBackend``) end-to-end
through ``runner.run`` on a single-pass shallow run. A single shared
``PHASE_SCRIPTS`` (the proven map from ``test_codex_realpath``) renders to BOTH
drivers' native formats via the harness; only each driver's external boundary
(the Codex subprocess / the Claude SDK client) is stubbed per firing phase.

Assertions pin OBSERVABLE outcomes only: exit code 0 and a non-empty on-disk
ATIF trajectory. ``assume="no"`` declines the post-pass commit gate so the run
finishes on the four named phases without a real ``git push``.
"""

import json
from pathlib import Path

import pytest

from daydream.runner import RunConfig, run
from tests.harness.phase_replay import replay_through_runner
from tests.test_codex_realpath import PHASE_SCRIPTS


@pytest.mark.parametrize("driver", ["codex", "claude"])
@pytest.mark.asyncio
async def test_realpath_shallow_run_per_driver(driver: str, feature_branch_repo: Path, tmp_path: Path) -> None:
    """Single-pass shallow run through the real backend for *driver*.

    The shared ``PHASE_SCRIPTS`` renders to the driver's native message stream;
    the run completes with exit 0 and writes a non-empty ATIF trajectory.
    ``assume="no"`` declines the post-pass commit gate (no real ``git push``).
    """
    traj = tmp_path / "trajectory.json"
    config = RunConfig(
        target=str(feature_branch_repo),
        skill="python",
        quiet=True,
        cleanup=False,
        shallow=True,
        loop=False,
        backend=driver,
        assume="no",
        trajectory_path=traj,
    )

    with replay_through_runner(driver, PHASE_SCRIPTS):
        exit_code = await run(config)

    assert exit_code == 0
    assert json.loads(traj.read_text(encoding="utf-8"))["steps"]
