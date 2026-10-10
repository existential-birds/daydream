"""Deep arbiter-session continuation integration tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.harness.review_result import saved_coverage
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence
from tests.test_deep_orchestrator import _run_deep


def _merge_call(stub: StubBackend) -> dict[str, Any]:
    return next(c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower())

async def test_merge_resumes_arbiter_session(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The merge call resumes the arbiter's session and warns about stale records."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.arbiter_session_id = "arb-sess"
    assert await _run_deep(multi_stack_target) == 0
    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is not None
    assert merge_call["continuation"].data["session_id"] == "arb-sess"
    assert "re-read" in merge_call["prompt"].lower()

async def test_merge_cold_when_arbiter_mints_no_token(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No arbiter token means merge runs cold with today's prompt, no addendum."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.arbiter_session_id = None
    assert await _run_deep(multi_stack_target) == 0
    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is None
    assert "re-read" not in merge_call["prompt"].lower()

async def test_merge_cold_when_arbiter_skipped_on_resume(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--start-at merge past a completed adjudication runs merge cold."""

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.arbiter_session_id = "arb-sess"
    stub.parse_severity = "high"
    assert await _run_deep(multi_stack_target) == 0
    assert _merge_call(stub)["continuation"].data["session_id"] == "arb-sess"
    deep = multi_stack_target / ".daydream/deep"
    assert saved_coverage(deep).phases["arbiter"]["status"] == "complete"
    stub.calls.clear()
    assert await _run_deep(multi_stack_target, start_at="merge") == 0
    assert [c for c in stub.calls if "you are the arbiter" in c["prompt"].lower()] == []
    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is None
    assert "re-read" not in merge_call["prompt"].lower()
