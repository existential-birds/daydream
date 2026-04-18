"""Cross-stack merge prompt + invocation tests (D-23..D-27, D-38)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.backends import ResultEvent, TextEvent
from daydream.deep.prompts import build_merge_prompt
from daydream.phases import phase_cross_stack_merge


def test_merge_prompt_specifies_report_format(tmp_path: Path) -> None:
    """D-25: the flat numbered ## Issues + ## Cross-Stack Issues structure is specified."""
    intent = tmp_path / "intent.md"
    alts = tmp_path / "alts.json"
    dedup = tmp_path / "dedup.json"
    out = tmp_path / ".review-output.md"
    records = [tmp_path / "r1.json", tmp_path / "r2.json"]
    prompt = build_merge_prompt(
        per_stack_records_paths=records,
        intent_path=intent,
        alternatives_path=alts,
        dedup_candidates_path=dedup,
        output_path=out,
    )
    assert "## Issues" in prompt
    assert "## Cross-Stack Issues" in prompt
    assert "continues the SAME numbering" in prompt


def test_merge_prompt_mandates_cross_stack_prefix(tmp_path: Path) -> None:
    """D-26: every cross-stack title must begin with [cross-stack]."""
    prompt = build_merge_prompt(
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
        output_path=tmp_path / ".review-output.md",
    )
    assert "[cross-stack]" in prompt
    assert "MUST begin with the literal prefix [cross-stack]" in prompt


def test_merge_prompt_references_records_by_path(tmp_path: Path) -> None:
    """D-22: prompt references records by path, not embedded content."""
    records = [
        tmp_path / "deep" / "stack-python-records.json",
        tmp_path / "deep" / "stack-react-records.json",
    ]
    prompt = build_merge_prompt(
        per_stack_records_paths=records,
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
        output_path=tmp_path / ".review-output.md",
    )
    for r in records:
        assert str(r) in prompt


def test_merge_prompt_mentions_dedup_candidates(tmp_path: Path) -> None:
    """D-27: merger is told to read dedup-candidates and adjudicate."""
    prompt = build_merge_prompt(
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "dedup-candidates.json",
        output_path=tmp_path / ".review-output.md",
    )
    assert "dedup-candidates.json" in prompt or "candidate pair" in prompt
    assert (
        "adjudication" in prompt.lower()
        or "adjudicate" in prompt.lower()
        or "decide" in prompt.lower()
    )


class _RecordingBackend:
    """Records every execute call; verifies no `agents` kwarg was passed."""

    def __init__(self) -> None:
        self.agents_seen: list[Any] = []
        self.prompts: list[str] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
    ):
        self.prompts.append(prompt)
        self.agents_seen.append(agents)
        yield TextEvent(text="merged")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


async def test_phase_cross_stack_merge_returns_output_path(tmp_path: Path) -> None:
    """D-24: merged report path is cwd / REVIEW_OUTPUT_FILE."""
    from daydream.config import REVIEW_OUTPUT_FILE

    backend = _RecordingBackend()
    result = await phase_cross_stack_merge(
        backend,
        tmp_path,
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
    )
    assert result == tmp_path / REVIEW_OUTPUT_FILE


async def test_phase_cross_stack_merge_no_agents_kwarg(tmp_path: Path) -> None:
    """D-38: no agents= kwarg (Codex compatibility)."""
    backend = _RecordingBackend()
    await phase_cross_stack_merge(
        backend,
        tmp_path,
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
    )
    assert all(a is None for a in backend.agents_seen)
