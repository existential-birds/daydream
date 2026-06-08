"""Real-path test for the shared ``PhaseDispatchBackend`` (Task 10).

Drives the production shallow loop through ``runner.run`` with the shared
phase-dispatch fake injected at the ``daydream.runner.create_backend`` seam.
Asserts the observable outcome (exit code + parse-call count proving the loop
sequenced twice: an issue on iteration 1, clean on iteration 2 → exit 0).
"""

import pytest

from daydream.runner import RunConfig, run
from tests.harness.phase_backend import PhaseDispatchBackend

# Minimal FEEDBACK_SCHEMA issue record.
ISSUE = {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}


@pytest.fixture
def mock_ui_loop(monkeypatch):
    """Decline interactive gates so the loop runs unattended (mirrors test_loop.py)."""
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *a, **kw: "n")


@pytest.mark.asyncio
async def test_shared_phase_backend_drives_shallow_loop(feature_branch_repo, mock_ui_loop, monkeypatch):
    """An issue on iteration 1, clean on iteration 2 → the loop runs twice and exits 0."""
    backend = PhaseDispatchBackend(parse_results=[[ISSUE], []])
    monkeypatch.setattr("daydream.runner.create_backend", lambda n, model=None: backend)

    exit_code = await run(
        RunConfig(
            target=str(feature_branch_repo),
            skill="python",
            quiet=True,
            cleanup=False,
            loop=True,
            max_iterations=5,
            shallow=True,
        )
    )

    assert exit_code == 0
    assert backend.parse_calls == 2
