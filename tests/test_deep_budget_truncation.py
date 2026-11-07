"""Deep non-wonder budget/truncation integration tests."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import install_stub_backend, silence
from tests.test_deep_orchestrator import _pin_findings_pr


async def test_single_stack_alternatives_timeout_still_emits_findings(
    tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: Callable[..., RunConfig],
) -> None:
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, tiny_diff_target)
    stub.runaway_alternatives = True
    _pin_findings_pr(monkeypatch, tiny_diff_target)
    out = tiny_diff_target / "findings.json"
    assert await run(make_config(
        tiny_diff_target, pr_number=7, findings_out=str(out), review_profile=independent_alternatives_profile(),
    )) == 0
    artifact = json.loads(out.read_text())
    assert artifact["review_warnings"] == ["alternatives: tool_call_budget_exceeded"]
