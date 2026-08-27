"""Deep arbiter-session continuation integration tests."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from tests.harness.stub_backend import StubBackend, install_stub_backend, silence


def _default_strategy(stage: str) -> str:
    from daydream import review_profile as _rp

    return _rp.build_default_profile().strategies[stage].content


def _merge_call(stub: StubBackend) -> dict[str, Any]:
    return next(c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower())


async def test_merge_resumes_arbiter_session(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge call resumes the arbiter's session and warns about stale records."""
    from tests.test_deep_orchestrator import _run_deep

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.arbiter_session_id = "arb-sess"
    assert await _run_deep(multi_stack_target) == 0
    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is not None
    assert merge_call["continuation"].data["session_id"] == "arb-sess"
    assert "re-read" in merge_call["prompt"].lower()


async def test_merge_cold_when_arbiter_mints_no_token(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No arbiter token means merge runs cold with today's prompt, no addendum."""
    from tests.test_deep_orchestrator import _run_deep

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.arbiter_session_id = None
    assert await _run_deep(multi_stack_target) == 0
    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is None
    assert "re-read" not in merge_call["prompt"].lower()


async def test_merge_cold_when_arbiter_skipped_on_resume(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--start-at merge past a completed adjudication runs merge cold."""
    from tests.test_deep_orchestrator import _prime_merge_resume, _record, _run_deep

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.arbiter_session_id = "arb-sess"
    deep = _prime_merge_resume(
        multi_stack_target,
        python=[_record(description="py issue", severity="high")],
        react=[_record(description="tsx issue", severity="high")],
        generic=[_record(description="md issue", severity="high")],
        structure=[],
    )
    (deep / "arbiter-complete.marker").write_text("done")
    # Fresh runs write the key before producing each prerequisite artifact.
    future = time.time() + 1
    for artifact in [
        deep / "intent.md",
        deep / "alternatives.json",
        *deep.glob("stack-*-records.json"),
    ]:
        os.utime(artifact, (future, future))
    assert await _run_deep(multi_stack_target, start_at="merge") == 0
    assert [c for c in stub.calls if "you are the arbiter" in c["prompt"].lower()] == []
    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is None
    assert "re-read" not in merge_call["prompt"].lower()


def test_merge_prompt_cold_path_is_byte_identical(tmp_path: Path) -> None:
    """resumed_from_arbiter=False reproduces today's prompt exactly."""
    from daydream.deep.prompts import build_merge_prompt

    kwargs: dict[str, Any] = dict(
        strategy=_default_strategy("merge"),
        per_stack_records_paths=[tmp_path / "stack-python-records.json"],
        intent_path=tmp_path / "intent.md", alternatives_path=tmp_path / "alternatives.json",
        dedup_candidates_path=tmp_path / "dedup.json", output_path=tmp_path / "out.md",
    )
    omitted = build_merge_prompt(**kwargs)
    explicit_false = build_merge_prompt(**kwargs, resumed_from_arbiter=False)
    resumed = build_merge_prompt(**kwargs, resumed_from_arbiter=True)
    assert omitted == explicit_false
    assert resumed != omitted
    assert resumed.startswith(omitted)
