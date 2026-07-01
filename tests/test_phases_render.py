"""Phase-seam render tests for the de-silenced structured phases (Task 3).

After Task 1 suppressed raw structured-output JSON in ``run_agent``, the
structured phases (parse-feedback, cross-stack merge, arbiter) produced no
visible terminal output. These tests drive the real phase entrypoints with a
``MockBackend`` whose ``ResultEvent.structured_output`` matches each phase's
schema and assert the restored summaries render observable content (counts,
a table) without dumping raw JSON.

Verified harness (from Task 0): record console is
``Console(record=True, force_terminal=True, width=100)``; tests patch the
importing module's binding via ``monkeypatch.setattr("daydream.phases.console", rec)``.
``MockBackend`` mirrors tests/test_agent_recorder_integration.py:61-96 and
accounts for ``run_agent``'s keyword-only ``phase`` argument (passed by the
phases themselves, not the backend).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from test_agent_recorder_integration import MockBackend

from daydream.backends import AgentEvent, ContinuationToken, ResultEvent
from daydream.deep.detection import StackAssignment
from daydream.phases import (
    phase_arbiter_review,
    phase_cross_stack_merge,
    phase_parse_feedback,
    phase_per_stack_reviews,
)


def _rec(monkeypatch: Any) -> Console:
    rec = Console(record=True, force_terminal=True, width=100)
    monkeypatch.setattr("daydream.phases.console", rec)
    return rec


async def test_parse_feedback_renders_issue_table(monkeypatch, tmp_path, make_work):
    rec = _rec(monkeypatch)
    payload = {
        "issues": [
            {
                "id": 1,
                "description": "Missing null check",
                "file": "f.py",
                "line": 3,
                "confidence": "HIGH",
                "rationale": "crashes on None",
            }
        ]
    }
    backend = MockBackend([ResultEvent(structured_output=payload, continuation=None)])

    items = await phase_parse_feedback(backend, make_work(tmp_path))

    out = rec.export_text()
    assert items == payload["issues"]
    assert "Found 1 actionable issue" in out
    assert "f.py" in out
    assert "Missing null check" in out
    assert "{" not in out


async def test_merge_prints_item_count(monkeypatch, tmp_path, make_work):
    rec = _rec(monkeypatch)
    items = [
        {
            "id": i,
            "description": f"issue {i}",
            "file": f"f{i}.py",
            "line": i,
            "confidence": "HIGH",
            "rationale": "r",
            "evidence": f"f{i}.py:{i}",
            "lens": "per-stack",
            "severity": "high",
        }
        for i in range(1, 4)
    ]
    backend = MockBackend([ResultEvent(structured_output={"items": items}, continuation=None)])

    dd = tmp_path / ".daydream" / "deep"
    dd.mkdir(parents=True, exist_ok=True)
    intent = dd / "intent.md"
    intent.write_text("intent")
    alts = dd / "alternatives.json"
    alts.write_text("[]")
    dedup = dd / "dedup-candidates.json"
    dedup.write_text("[]")

    await phase_cross_stack_merge(
        backend,
        make_work(tmp_path),
        per_stack_records_paths=[],
        intent_path=intent,
        alternatives_path=alts,
        dedup_candidates_path=dedup,
    )

    out = rec.export_text()
    assert "Merged into 3 items" in out
    assert "{" not in out


async def test_arbiter_prints_kept_dropped(monkeypatch, tmp_path, make_work):
    rec = _rec(monkeypatch)
    selected = [
        {
            "file": f"f{i}.py",
            "line": i,
            "severity": "high",
            "confidence": "HIGH",
            "description": f"d{i}",
            "rationale": "r",
        }
        for i in range(1, 4)
    ]
    findings = [
        {"arb_id": 1, "keep": True, "severity": "high", "confidence": "HIGH", "description": "d1", "rationale": "r"},
        {"arb_id": 2, "keep": True, "severity": "high", "confidence": "HIGH", "description": "d2", "rationale": "r"},
        {"arb_id": 3, "keep": False, "severity": "low", "confidence": "LOW", "description": "d3", "rationale": "r"},
    ]
    backend = MockBackend([ResultEvent(structured_output={"findings": findings}, continuation=None)])

    dd = tmp_path / ".daydream" / "deep"
    dd.mkdir(parents=True, exist_ok=True)
    diff = dd / "diff.patch"
    diff.write_text("diff")
    intent = dd / "intent.md"
    intent.write_text("intent")
    alts = dd / "alternatives.json"
    alts.write_text("[]")

    verdicts = await phase_arbiter_review(
        backend,
        make_work(tmp_path),
        selected_records=selected,
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    out = rec.export_text()
    assert len(verdicts) == 3
    assert "kept" in out.lower()
    assert "2" in out
    assert "1" in out


@dataclass
class _PerStackBackend:
    """Backend that raises for stacks whose name appears in ``fail_for``.

    ``phase_per_stack_reviews`` passes each stack's output path (which embeds the
    stack name, e.g. ``stack-stack-a-review.md``) into the per-stack prompt, so the
    stub keys its raise/succeed decision off the prompt text. Mirrors the
    three-method Backend protocol (test_agent_recorder_integration:61-96).
    """

    model = "mock-model"
    fail_for: set[str]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        should_fail = any(f"stack-{name}-review.md" in prompt for name in self.fail_for)

        async def _gen() -> AsyncIterator[AgentEvent]:
            if should_fail:
                raise RuntimeError("agent boom")
            yield ResultEvent(structured_output=None, continuation=None)

        return _gen()

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


async def test_per_stack_failures_summarized_once(monkeypatch, tmp_path, make_work):
    rec = _rec(monkeypatch)
    work = make_work(tmp_path)
    (tmp_path / ".daydream" / "deep").mkdir(parents=True, exist_ok=True)
    diff = tmp_path / ".daydream" / "deep" / "diff.patch"
    diff.write_text("diff")
    intent = tmp_path / ".daydream" / "deep" / "intent.md"
    intent.write_text("intent")
    alts = tmp_path / ".daydream" / "deep" / "alternatives.json"
    alts.write_text("[]")

    stacks = [
        StackAssignment(stack_name="stack-a", skill_invocation=None, files=["a.py"]),
        StackAssignment(stack_name="stack-b", skill_invocation=None, files=["b.py"]),
        StackAssignment(stack_name="stack-c", skill_invocation=None, files=["c.py"]),
    ]
    backend = _PerStackBackend(fail_for={"stack-a", "stack-b"})

    successes, failures = await phase_per_stack_reviews(
        backend,
        work,
        stacks,
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    out = rec.export_text()
    assert set(failures) == {"stack-a", "stack-b"}
    assert set(successes) == {"stack-c"}
    # ONE consolidated end-of-phase summary names BOTH failed stacks -- not two
    # scattered inline warnings. The summary header is emitted exactly once.
    assert "stack-a" in out and "stack-b" in out
    assert out.count("failures will be passed to the merge step") >= 1
    assert out.count("Per-stack reviews failed") == 1
