"""Cross-stack merge prompt + invocation tests (D-23..D-27, D-38)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from daydream.backends import ResultEvent, TextEvent
from daydream.deep.prompts import build_merge_prompt
from daydream.phases import phase_cross_stack_merge
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend, Turn


def _default_strategy(stage: str) -> str:
    from daydream import review_profile as _rp

    return _rp.build_default_profile().strategies[stage].content


def test_merge_prompt_mandates_cross_stack_lens(tmp_path: Path) -> None:
    """D-26: cross-stack concerns are tagged via the cross-stack lens."""
    prompt = build_merge_prompt(
        strategy=_default_strategy("merge"),
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
        strategy=_default_strategy("merge"),
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
        strategy=_default_strategy("merge"),
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


# The merge agent returns a schema item list; the host renders the report.
_MERGE_TURN: Turn = [
    TextEvent(text="merged"),
    ResultEvent(
        structured_output={
            "items": [
                {
                    "id": 1,
                    "lens": "per-stack",
                    "file": "api.py",
                    "line": 1,
                    "severity": "low",
                    "description": "issue",
                    "confidence": "MEDIUM",
                    "rationale": "r",
                    "evidence": "api.py:1",
                }
            ]
        },
        continuation=None,
    ),
]


async def test_phase_cross_stack_merge_returns_output_path(
    tmp_path: Path,
    make_work: Callable[..., WorkContext],
) -> None:
    """D-24: merged report path is work.repo / REVIEW_OUTPUT_FILE."""
    from daydream.config import REVIEW_OUTPUT_FILE

    backend = ScriptedBackend(events=_MERGE_TURN)
    result = await phase_cross_stack_merge(
        backend,
        make_work(tmp_path),
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
    )
    assert result == tmp_path / REVIEW_OUTPUT_FILE


async def test_phase_cross_stack_merge_no_agents_kwarg(tmp_path: Path, make_work: Callable[..., WorkContext]) -> None:
    """D-38: no agents= kwarg (Codex compatibility)."""
    backend = ScriptedBackend(events=_MERGE_TURN)
    await phase_cross_stack_merge(
        backend,
        make_work(tmp_path),
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
    )
    assert all(c["agents"] is None for c in backend.calls)


def test_merge_prompt_accepts_shard_records_paths(tmp_path: Path) -> None:
    """Issue #731 (P2): merge prompt references synthetic ``#`` shard paths."""
    records = [tmp_path / "deep" / "stack-python#0-records.json",
               tmp_path / "deep" / "stack-python#1-records.json"]
    prompt = build_merge_prompt(
        strategy=_default_strategy("merge"),
        per_stack_records_paths=records,
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
        output_path=tmp_path / "o.md",
    )
    for r in records:
        assert str(r) in prompt


def test_merge_prompt_tags_alternatives_items_as_wonder(tmp_path: Path) -> None:
    prompt = build_merge_prompt(
        strategy=_default_strategy("merge"),
        per_stack_records_paths=[tmp_path / "r.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
        output_path=tmp_path / ".review-output.md",
    )
    assert '"wonder"' in prompt      # the lens value the agent must emit for alt items
    assert "alternatives" in prompt


def test_merge_prompt_demands_verbatim_source_uids(tmp_path: Path) -> None:
    """#1111: the merge prompt asks for machine-readable provenance, verbatim.

    ``source_uids`` is strict-mode required, so the model emits the key whether
    or not it was told what to put in it -- meaning a prompt that never explains
    the field yields a schema-valid run in which every item is attributed to
    nothing, and the host-side validator has nothing to validate. Two halves of
    the instruction are load-bearing and pinned here: copy the uid VERBATIM (a
    reformatted uid is indistinguishable from an invented one to the validator,
    which keeps only exact pool members), and list ALL contributing uids on a
    deduplicated item (the multi-record case is the whole reason the field is a
    list rather than a single handle).
    """
    prompt = build_merge_prompt(
        strategy=_default_strategy("merge"),
        per_stack_records_paths=[tmp_path / "stack-python-records.json"],
        intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=tmp_path / "d.json",
        output_path=tmp_path / ".review-output.md",
    )
    assert "source_uids" in prompt
    assert "VERBATIM" in prompt
    assert "list ALL contributing" in prompt
    # The consequence of inventing one is stated, because the host silently
    # drops an unknown uid rather than failing the merge over it.
    assert "not in the records is discarded" in prompt
    # ...and the human-readable citation is still required alongside it, so the
    # machine-readable field does not quietly replace the reader's version.
    assert "(Sources: ...)" in prompt
