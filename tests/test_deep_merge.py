"""Cross-stack merge prompt + invocation tests (D-23..D-27, D-38)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.backends import ResultEvent, TextEvent
from daydream.deep.prompts import build_merge_prompt
from daydream.phases import phase_cross_stack_merge


def test_merge_prompt_specifies_structured_item_list(tmp_path: Path) -> None:
    """D-25: the agent is told to return a schema item list with lens + severity."""
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
    assert '{"items": [' in prompt  # structured JSON list, not markdown
    assert "lens" in prompt
    assert "severity" in prompt
    # The agent must NOT be told to write a markdown report to a file anymore.
    assert "write the complete report to" not in prompt.lower()


def test_merge_prompt_mandates_cross_stack_lens(tmp_path: Path) -> None:
    """D-26: cross-stack concerns are tagged via the cross-stack lens."""
    prompt = build_merge_prompt(
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
        output_path=tmp_path / ".review-output.md",
    )
    assert "cross-stack" in prompt
    assert "spanning multiple stacks" in prompt


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

    model = "test-model"

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
        max_turns: Any = None,
        read_only=False,
    ):
        self.prompts.append(prompt)
        self.agents_seen.append(agents)
        # The merge agent returns a schema item list; the host renders the report.
        yield TextEvent(text="merged")
        yield ResultEvent(
            structured_output={
                "items": [
                    {
                        "id": 1,
                        "lens": "per-stack",
                        "file": "api.py",
                        "line": 1,
                        "severity": "low",
                        "description": "issue",
                        "confidence": "LOW",
                        "rationale": "r",
                    }
                ]
            },
            continuation=None,
        )

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


async def test_phase_cross_stack_merge_returns_output_path(tmp_path: Path, make_work) -> None:
    """D-24: merged report path is work.repo / REVIEW_OUTPUT_FILE."""
    from daydream.config import REVIEW_OUTPUT_FILE

    backend = _RecordingBackend()
    result = await phase_cross_stack_merge(
        backend,
        make_work(tmp_path),
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
    )
    assert result == tmp_path / REVIEW_OUTPUT_FILE


async def test_phase_cross_stack_merge_no_agents_kwarg(tmp_path: Path, make_work) -> None:
    """D-38: no agents= kwarg (Codex compatibility)."""
    backend = _RecordingBackend()
    await phase_cross_stack_merge(
        backend,
        make_work(tmp_path),
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
    )
    assert all(a is None for a in backend.agents_seen)
