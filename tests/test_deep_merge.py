"""Cross-stack merge prompt + invocation tests (D-23..D-27, D-38)."""
from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path

import pytest

from daydream.backends import ResultEvent, TextEvent
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.deep.artifacts import (
    DeepArtifact,
    deep_dir,
)
from daydream.deep.prompts import build_merge_prompt
from daydream.extensions import get_registry
from daydream.phases import phase_cross_stack_merge
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend, Turn
from tests.harness.review_profile import default_strategy as _default_strategy
from tests.harness.review_result import merge_result


@pytest.mark.parametrize("payload", [[], {"issues": []}])
async def test_empty_merge_host_noop_requires_current_record_envelope(
    tmp_path: Path, make_work: Callable[..., WorkContext], payload: object,
) -> None:
    dd = deep_dir(tmp_path, allow_standalone=True)
    records = dd / "stack-python-records.json"
    records.write_text(json.dumps(payload))
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    backend = ScriptedBackend(events=[ResultEvent(structured_output={"items": []}, continuation=None)])

    report = await phase_cross_stack_merge(
        backend, make_work(tmp_path), per_stack_records_paths=[records], intent_path=dd / "intent.md",
        alternatives_path=alternatives, dedup_candidates_path=dd / "dedup-candidates.json", allow_standalone=True,
    )

    assert backend.call_count == int(isinstance(payload, list))
    assert json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text()) == {"items": []}
    assert report.read_text() == DeepArtifact.MERGED_REPORT.at(dd).read_text()


@pytest.mark.parametrize("input_kind", [
    "nonempty-records", "nonempty-alternatives", "missing-records", "missing-alternatives",
    "malformed-records", "malformed-alternatives", "wrong-records-shape", "wrong-alternatives-shape",
    "envelope-alternatives", "no-record-paths", "custom-strategy", "custom-builder",
])
async def test_empty_merge_requires_completed_inputs_and_builtin_contract(
    tmp_path: Path, make_work: Callable[..., WorkContext], monkeypatch: pytest.MonkeyPatch, input_kind: str,
) -> None:
    dd = deep_dir(tmp_path, allow_standalone=True)
    records = dd / "stack-python-records.json"
    records.write_text('{"issues": []}')
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    strategy: str | None = None
    if input_kind == "nonempty-records":
        records.write_text('[{"id": 1, "description": "review finding"}]')
    elif input_kind == "nonempty-alternatives":
        alternatives.write_text('[{"title": "independent design finding", "files": ["api.py"]}]')
    elif input_kind.startswith("missing-"):
        (records if input_kind == "missing-records" else alternatives).unlink()
    elif input_kind.startswith("malformed-"):
        (records if input_kind == "malformed-records" else alternatives).write_text("{bad json")
    elif input_kind.startswith("wrong-"):
        (records if input_kind == "wrong-records-shape" else alternatives).write_text('{"issues": null}')
    elif input_kind == "envelope-alternatives":
        alternatives.write_text('{"issues": []}')
    elif input_kind == "custom-strategy":
        strategy = "Perform an independent custom merge inspection."
    elif input_kind == "custom-builder":
        registry = get_registry()
        registry.override_prompt("merge", lambda **kwargs: build_merge_prompt(**kwargs))
        monkeypatch.setattr("daydream.extensions.loader._REGISTRY_VAR", ContextVar("merge-registry", default=registry))
    backend = ScriptedBackend(events=[ResultEvent(structured_output={"items": []}, continuation=None)])

    await phase_cross_stack_merge(
        backend, make_work(tmp_path), per_stack_records_paths=[] if input_kind == "no-record-paths" else [records],
        intent_path=dd / "intent.md", alternatives_path=alternatives,
        dedup_candidates_path=dd / "dedup-candidates.json", strategy=strategy, allow_standalone=True,
    )

    assert backend.call_count == 1


async def test_empty_merge_keeps_structural_findings_and_clears_stale_outputs(
    tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    dd = deep_dir(tmp_path, allow_standalone=True)
    records = dd / "stack-python-records.json"
    records.write_text('{"issues": []}')
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    structural = dd / "stack-structure-records.json"
    structural.write_text(json.dumps({"issues": [{
        "id": 42, "uid": "structure:42", "item_uid": "stable-structural-item", "file": "design.py", "line": 0,
        "description": "concrete structural finding", "evidence": "Documented dependency contradicts contract",
        "rationale": "The dependency must honor the documented boundary",
        "severity": "medium", "confidence": "MEDIUM",
    }]}))
    for path in (DeepArtifact.MERGED_ITEMS.at(dd), DeepArtifact.MERGED_REPORT.at(dd), tmp_path / REVIEW_OUTPUT_FILE,
                 dd / "dropped-speculative.json", dd / "folded-structural.json"):
        path.write_text("stale result")
    backend = ScriptedBackend(events=[AssertionError("empty language merge must not dispatch")])

    await phase_cross_stack_merge(
        backend, make_work(tmp_path), per_stack_records_paths=[records], intent_path=dd / "intent.md",
        alternatives_path=alternatives, dedup_candidates_path=dd / "dedup-candidates.json",
        structural_records_path=structural, strategy=_default_strategy("merge"), allow_standalone=True,
    )

    assert backend.call_count == 0
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text())["items"]
    assert len(items) == 1
    assert items[0]["id"] == 1
    assert items[0]["uid"] == "structure:42"
    assert items[0]["item_uid"] == "stable-structural-item"
    assert items[0]["source_uids"] == ["structure:42"]
    assert items[0]["lens"] == "structural"
    assert items[0]["severity"] == "medium"
    assert items[0]["confidence"] == "MEDIUM"
    assert not (dd / "dropped-speculative.json").exists()
    assert not (dd / "folded-structural.json").exists()
    assert "stale result" not in (tmp_path / REVIEW_OUTPUT_FILE).read_text()


async def test_empty_merge_exposes_failed_stack_coverage(
    tmp_path: Path, make_work: Callable[..., WorkContext], capsys: pytest.CaptureFixture[str],
) -> None:
    dd = deep_dir(tmp_path, allow_standalone=True)
    records = dd / "stack-python-records.json"
    records.write_text('{"issues": []}')
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    backend = ScriptedBackend(events=[AssertionError("empty merge must not dispatch")])

    await phase_cross_stack_merge(
        backend, make_work(tmp_path), per_stack_records_paths=[records], intent_path=dd / "intent.md",
        alternatives_path=alternatives, dedup_candidates_path=dd / "dedup-candidates.json",
        failed_stacks={"react": "review budget exhausted"}, allow_standalone=True,
    )

    assert backend.call_count == 0
    assert "react" in capsys.readouterr().out


def _merge_prompt(tmp_path: Path, records: list[Path] | None = None, *, dedup_candidates_path: Path | None = None,
) -> str:
    return build_merge_prompt(strategy=_default_strategy("merge"),
        per_stack_records_paths=records if records is not None else [tmp_path / "r.json"],
        intent_path=tmp_path / "i.md", alternatives_path=tmp_path / "a.json",
        dedup_candidates_path=dedup_candidates_path if dedup_candidates_path is not None else tmp_path / "d.json",
    )

def test_merge_prompt_mandates_cross_stack_lens(tmp_path: Path) -> None:
    prompt = _merge_prompt(tmp_path)
    assert "cross-stack" in prompt
    assert "spanning multiple stacks" in prompt

def test_merge_prompt_references_records_by_path(tmp_path: Path) -> None:
    records = [tmp_path / "deep" / "stack-python-records.json", tmp_path / "deep" / "stack-react-records.json",]
    prompt = _merge_prompt(tmp_path, records)
    for r in records:
        assert str(r) in prompt

def test_merge_prompt_mentions_dedup_candidates(tmp_path: Path) -> None:
    prompt = _merge_prompt(tmp_path, dedup_candidates_path=tmp_path / "dedup-candidates.json")
    assert "dedup-candidates.json" in prompt or "candidate pair" in prompt
    assert ("adjudication" in prompt.lower() or "adjudicate" in prompt.lower() or "decide" in prompt.lower())

# The merge agent returns a schema item list; the host renders the report.
_MERGE_TURN: Turn = [TextEvent(text="merged"),
    ResultEvent(structured_output=merge_result([{
                    "id": 1, "lens": "per-stack", "file": "api.py", "line": 1, "severity": "low",
                    "description": "issue", "confidence": "MEDIUM", "rationale": "r", "evidence": "api.py:1",
                }
            ]), continuation=None,
    ),
]

async def test_phase_cross_stack_merge_returns_output_path(tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    backend = ScriptedBackend(events=_MERGE_TURN)
    result = await phase_cross_stack_merge(
        backend, make_work(tmp_path), per_stack_records_paths=[tmp_path / "r.json"], intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json", dedup_candidates_path=tmp_path / "d.json", allow_standalone=True,
    )
    assert result == tmp_path / REVIEW_OUTPUT_FILE

async def test_phase_cross_stack_merge_no_agents_kwarg(tmp_path: Path, make_work: Callable[..., WorkContext]) -> None:
    """D-38: no agents= kwarg (Codex compatibility)."""
    backend = ScriptedBackend(events=_MERGE_TURN)
    await phase_cross_stack_merge(
        backend, make_work(tmp_path), per_stack_records_paths=[tmp_path / "r.json"], intent_path=tmp_path / "i.md",
        alternatives_path=tmp_path / "a.json", dedup_candidates_path=tmp_path / "d.json", allow_standalone=True,
    )
    assert all(c["agents"] is None for c in backend.calls)

def test_merge_prompt_accepts_shard_records_paths(tmp_path: Path) -> None:
    records = [tmp_path / "deep" / "stack-python#0-records.json",
               tmp_path / "deep" / "stack-python#1-records.json"]
    prompt = _merge_prompt(tmp_path, records)
    for r in records:
        assert str(r) in prompt

def test_merge_prompt_tags_alternatives_items_as_wonder(tmp_path: Path) -> None:
    prompt = _merge_prompt(tmp_path)
    assert '"wonder"' in prompt      # the lens value the agent must emit for alt items
    assert "alternatives" in prompt

def test_merge_prompt_demands_verbatim_source_uids(tmp_path: Path) -> None:
    """Require exact source uids and every contributor to a merged item.

    The schema alone cannot ensure attribution: reformatted or invented uids are
    discarded by host validation, while an empty list still satisfies the schema.
    """
    prompt = _merge_prompt(tmp_path, [tmp_path / "stack-python-records.json"])
    assert "source_uids" in prompt
    assert "VERBATIM" in prompt
    assert "list ALL contributing" in prompt
    # Unknown uids are discarded without failing the merge.
    assert "not in the records is discarded" in prompt
    # Machine provenance must coexist with human-readable citations.
    assert "(Sources: ...)" in prompt
