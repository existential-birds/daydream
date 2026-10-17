"""Deep arbiter-session continuation integration tests."""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ResultEvent, ToolStartEvent
from tests.harness.review_result import merge_result, saved_coverage
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence
from tests.test_deep_orchestrator import _run_deep


def _merge_call(stub: StubBackend) -> dict[str, Any]:
    return next(c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower())

@pytest.mark.parametrize("finalize", [False, True])
async def test_merge_resumes_arbiter_session(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, finalize: bool,
) -> None:
    """A resumed merge receives retained structural records, including during bounded finalization."""
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.arbiter_session_id = "arb-sess"
    execute = stub.execute
    supplied: list[dict[str, Any]] = []
    finalization_calls: list[dict[str, Any]] = []

    async def capture_merge(cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if kwargs.pop("finalization", False):
            finalization_calls.append({"prompt": prompt,
                "continuation": args[1] if len(args) > 1 else kwargs.get("continuation")})
            yield ResultEvent(structured_output=merge_result([]), continuation=None)
            return
        is_merge = "cross-stack merge agent" in prompt.lower()
        if is_merge:
            source = re.search(r"^- structural-records: (.+)$", prompt, flags=re.MULTILINE)
            if source is not None:
                supplied.extend(json.loads(Path(source[1]).read_text())["issues"])
        async for event in execute(cwd, prompt, *args, **kwargs):
            if is_merge and finalize and isinstance(event, ResultEvent):
                # Exhaust the real merge allowance without a completed checkpoint.
                for index in range(17):
                    yield ToolStartEvent(id=f"merge-{index}", name="Read", input={"file_path": "api.py"})
            else:
                yield event

    monkeypatch.setattr(stub, "execute", capture_merge)
    monkeypatch.setattr(stub, "supports_finalization", True, raising=False)
    assert await _run_deep(multi_stack_target) == 0
    merge_call = _merge_call(stub)
    assert merge_call["continuation"] is not None
    assert merge_call["continuation"].data["session_id"] == "arb-sess"
    assert "re-read" in merge_call["prompt"].lower()
    retained = json.loads((multi_stack_target / ".daydream/deep/stack-structure-records.json").read_text())["issues"]
    assert retained
    assert supplied == retained
    if finalize:
        assert len(finalization_calls) == 1
        final_call = finalization_calls[0]
        assert final_call["continuation"] is None
        assert "Do not call tools or obtain new evidence" in final_call["prompt"]
        assert "Input 'structural-records'" in final_call["prompt"]
        for record in retained:
            assert record["uid"] in final_call["prompt"]
            assert record["description"] in final_call["prompt"]
        assert saved_coverage(multi_stack_target / ".daydream/deep").phases["merge"]["status"] == "incomplete"
    else:
        assert finalization_calls == []

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
