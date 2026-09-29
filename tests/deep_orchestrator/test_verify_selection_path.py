"""Real-path verify-selection config tests (issue #735).

``runner.run`` is the production entrypoint; only the external backend is
stubbed (``_install_accept_gate_pipeline``). The assertions read the persisted
``recommendation-verdicts.json`` artifact, so they observe the wiring from the
``RunConfig`` knob through ``_step_verify`` to the phase's ``selection`` block.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.deep.artifacts import verdicts_path
from daydream.runner import RunConfig, run
from tests.deep_orchestrator.support import _install_accept_gate_pipeline
from tests.test_deep_orchestrator import Mute


async def _run_deep_with(target: Path, **overrides: Any) -> int:
    """Run the deep pipeline through ``runner.run`` with explicit RunConfig overrides."""
    config = RunConfig(
        target=str(target),
        start_at="review",
        cleanup=False,
        **overrides,
    )
    return await run(config)


async def test_verify_all_reproduces_the_conservative_item_set(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
) -> None:
    """MH13 real-path: verify_all renders every non-structural item and marks none skipped."""
    _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)
    exit_code = await _run_deep_with(multi_stack_target, verify_all=True)
    assert exit_code == 0
    payload = json.loads(verdicts_path(multi_stack_target / ".daydream" / "deep").read_text())
    assert payload["selection"]["mode"] == "verify_all"
    assert payload["selection"]["skipped"] == 0
    assert all(
        d["reason_code"] in {"verify_all", "exempt:structural", "exempt:wonder"}
        for d in payload["selection"]["decisions"]
    )
