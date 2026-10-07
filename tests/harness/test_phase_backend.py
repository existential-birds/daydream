"""Real-path tests for the shared ``PhaseDispatchBackend``.

Drives the production shallow mode through ``runner.run`` with the shared
phase-dispatch fake injected at the ``daydream.runner.create_backend`` seam.
Asserts the observable outcome and that the removed parse phase is not invoked.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.phase_backend import PhaseDispatchBackend

# Minimal FEEDBACK_SCHEMA issue record.
ISSUE = {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}


@pytest.fixture
def mock_ui_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decline interactive gates so the run runs unattended."""
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "n")

@pytest.mark.asyncio
async def test_shared_phase_backend_drives_shallow_pass(feature_branch_repo: Path,
    mock_ui_loop: Any,  # noqa: F841
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One issue on the single pass → the shallow deep run completes and exits 0."""
    backend = PhaseDispatchBackend(parse_results=[[ISSUE]])
    monkeypatch.setattr("daydream.runner.create_backend", lambda n, model=None, **kwargs: backend)
    exit_code = await run(
        RunConfig(target=str(feature_branch_repo), stack="python", quiet=True, cleanup=False, shallow=True,)
    )

    assert exit_code == 0
    assert "Add type hints" in (feature_branch_repo / ".review-output.md").read_text()
    # The public scopes each receive one first pass; structure also integrates
    # their changed behavior before deterministic publication.
    assert backend.parse_calls == 0
    stages = [json.JSONDecoder().raw_decode(prompt.split("Host review stage:\n", 1)[1])[0]
              for prompt in backend.review_prompts]
    assert sorted((stage["scope_id"], stage["stage"]) for stage in stages) == [
        ("python", "first_pass"), ("structure", "first_pass"), ("structure", "integration"),
    ]
