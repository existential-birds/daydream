"""Phase-seam render tests for the de-silenced structured phases.

These tests drive the real phase entrypoints with a ``ScriptedBackend`` whose
``ResultEvent.structured_output`` matches each phase's schema and assert the
restored summaries render observable content (counts, a table) without dumping
raw JSON.
"""
from __future__ import annotations

from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest
from rich.console import Console

from daydream.backends import Backend, ResultEvent
from daydream.deep.detection import StackAssignment
from daydream.phases import (
    phase_arbiter_review,
    phase_cross_stack_merge,
    phase_per_stack_reviews,
)
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend


def _rec(monkeypatch: Any) -> Console:
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=100, height=25)
    monkeypatch.setattr("daydream.phases.console", rec)
    return rec


def _seed_deep(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Seed ``.daydream/deep`` with the intent/alternatives files every render
    phase requires; returns ``(deep_dir, intent_path, alternatives_path)``."""
    dd = tmp_path / ".daydream" / "deep"
    dd.mkdir(parents=True, exist_ok=True)
    intent = dd / "intent.md"
    intent.write_text("intent")
    alts = dd / "alternatives.json"
    alts.write_text("[]")
    return dd, intent, alts


async def test_merge_prints_item_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_work: Callable[..., WorkContext],
) -> None:
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
    backend = ScriptedBackend(
        events=[ResultEvent(structured_output={"items": items}, continuation=None)],
        model="mock-model",
    )

    dd, intent, alts = _seed_deep(tmp_path)
    dedup = dd / "dedup-candidates.json"
    dedup.write_text("[]")

    await phase_cross_stack_merge(
        backend,
        make_work(tmp_path),
        per_stack_records_paths=[],
        intent_path=intent,
        alternatives_path=alts,
        allow_standalone=True,
        dedup_candidates_path=dedup,
    )

    out = rec.export_text()
    assert "Merged into 3 items" in out
    assert "{" not in out


async def test_arbiter_prints_kept_dropped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_work: Callable[..., WorkContext],
) -> None:
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
    backend = ScriptedBackend(
        events=[ResultEvent(structured_output={"findings": findings}, continuation=None)],
        model="mock-model",
    )

    dd, intent, alts = _seed_deep(tmp_path)
    diff = dd / "diff.patch"
    diff.write_text("diff")

    verdicts, _ = await phase_arbiter_review(
        backend,
        make_work(tmp_path),
        selected_records=selected,
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
        allow_standalone=True,
    )

    out = rec.export_text()
    assert len(verdicts) == 3
    assert "Arbiter: kept 2, dropped 1" in out


def _per_stack_backend(fail_for: set[str]) -> ScriptedBackend:
    """Responder-backed fake raising for stacks whose name appears in ``fail_for``.

    ``phase_per_stack_reviews`` passes each stack's output path (which embeds the
    stack name, e.g. ``stack-stack-a-review.md``) into the per-stack prompt, so the
    fake keys its raise/succeed decision off the prompt text.
    """

    def respond(cwd: Any, prompt: str, *args: Any) -> list[Any]:
        if any(f"stack-{name}-review.md" in prompt for name in fail_for):
            return [RuntimeError("agent boom")]
        # Issue #745: per-stack reviewers must emit PER_STACK_RECORD_SCHEMA
        # structured output (issues + verdicts) to be recorded as a success.
        return [ResultEvent(structured_output={"issues": [], "verdicts": []}, continuation=None)]

    return ScriptedBackend(responder=respond, model="mock-model")


async def test_per_stack_failures_summarized_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_work: Callable[..., WorkContext],
) -> None:
    rec = _rec(monkeypatch)
    work = make_work(tmp_path)
    dd, intent, alts = _seed_deep(tmp_path)
    diff = dd / "diff.patch"
    diff.write_text("diff")

    stacks = [
        StackAssignment(stack_name="stack-a", files=["a.py"]),
        StackAssignment(stack_name="stack-b", files=["b.py"]),
        StackAssignment(stack_name="stack-c", files=["c.py"]),
    ]
    backend = _per_stack_backend({"stack-a", "stack-b"})

    successes, failures = await phase_per_stack_reviews(
        cast(Backend, backend),
        work,
        stacks,
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
        allow_standalone=True,
    )

    out = rec.export_text()
    assert set(failures) == {"stack-a", "stack-b"}
    assert set(successes) == {"stack-c"}
    # ONE consolidated end-of-phase summary names BOTH failed stacks -- not two
    # scattered inline warnings. The summary header is emitted exactly once.
    assert "stack-a" in out and "stack-b" in out
    assert out.count("failures will be passed to the merge step") >= 1
    assert out.count("Per-stack reviews failed") == 1
