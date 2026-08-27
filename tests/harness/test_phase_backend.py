"""Real-path test for the shared ``PhaseDispatchBackend`` (Task 10).

Drives the production shallow mode through ``runner.run`` with the shared
phase-dispatch fake injected at the ``daydream.runner.create_backend`` seam.
Asserts the observable outcome (exit code + parse-call count proving the
single review→parse pass ran once and the run completed cleanly — ``--loop``
was removed in the single-flow collapse (#330)).
"""

import pytest

from daydream.runner import RunConfig, run
from tests.harness.phase_backend import PhaseDispatchBackend

# Minimal FEEDBACK_SCHEMA issue record.
ISSUE = {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}


@pytest.fixture
def mock_ui_loop(monkeypatch):
    """Decline interactive gates so the run runs unattended."""
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *a, **kw: "n")


@pytest.mark.asyncio
async def test_shared_phase_backend_drives_shallow_pass(feature_branch_repo, mock_ui_loop, monkeypatch):  # noqa
    """One issue on the single pass → the shallow deep run completes and exits 0."""
    backend = PhaseDispatchBackend(parse_results=[[ISSUE]])
    monkeypatch.setattr("daydream.runner.create_backend", lambda n, model=None, **kwargs: backend)

    exit_code = await run(
        RunConfig(
            target=str(feature_branch_repo),
            stack="python",
            quiet=True,
            cleanup=False,
            shallow=True,
        )
    )

    assert exit_code == 0
    # Issue #745: the per-stack reviewers emit PER_STACK_RECORD_SCHEMA records
    # directly -- the spine no longer has a parse phase. The single shallow
    # pass still fires the per-stack reviews for the combined python stack +
    # the structural meta-stack.
    assert backend.parse_calls == 0
    assert len(backend.review_prompts) == 2

