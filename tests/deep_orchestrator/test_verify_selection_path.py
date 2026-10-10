"""Real runner tests of config-to-verifier wiring via persisted selection decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.deep.artifacts import DeepArtifact
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.deep_orchestrator.support import _install_accept_gate_pipeline
from tests.test_deep_orchestrator import Mute, _run_deep


def _mutate_item_evidence(target: Path, *, item_uid: str, evidence: str) -> None:
    """Rewrite one canonical item's verifier-relevant text in merged-items.json."""
    path = target / ".daydream" / "deep" / "merged-items.json"
    payload = json.loads(path.read_text())
    for item in payload["items"]:
        if item.get("item_uid") == item_uid:
            item["evidence"] = evidence
            path.write_text(json.dumps(payload))
            return
    raise AssertionError(f"no merged item with item_uid {item_uid!r}")


def _is_verifier_prompt(prompt: str) -> bool:
    return "recommendation-verifier" in prompt.lower()


async def _run_deep_with(target: Path, **overrides: Any) -> int:
    """Run the deep pipeline through ``runner.run`` with explicit RunConfig overrides."""
    config = RunConfig(target=str(target), start_at="review", cleanup=False, **overrides,)
    return await run(config)

async def test_start_at_fix_resume_reuses_unchanged_verdicts_and_reverifies_changed_ones(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
) -> None:
    """MH14 real-path: the second run makes one backend verify call, marked reused for the unchanged item."""
    stub = _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)
    assert await _run_deep(multi_stack_target) == 0
    verifier_calls_first = [c for c in stub.calls if _is_verifier_prompt(c["prompt"])]
    # Rewrite the merged items so item:2's verifier-relevant text changes, then resume at the fix gate.
    _mutate_item_evidence(multi_stack_target, item_uid="item:2", evidence="a.py:1 rewritten by hand")
    stub.calls.clear()
    assert await _run_deep(multi_stack_target, start_at="fix") == 0
    second = json.loads(DeepArtifact.VERDICTS.at(multi_stack_target / ".daydream" / "deep").read_text())
    assert len([c for c in stub.calls if _is_verifier_prompt(c["prompt"])]) == len(verifier_calls_first)
    assert any(d["verdict_reused"] for d in second["selection"]["decisions"])

async def test_verify_all_reproduces_the_conservative_item_set(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
) -> None:
    """MH13 real-path: verify_all renders every non-structural item and marks none skipped."""
    stub = _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)
    exit_code = await _run_deep_with(multi_stack_target, verify_all=True)
    assert exit_code == 0
    payload = json.loads(DeepArtifact.VERDICTS.at(multi_stack_target / ".daydream" / "deep").read_text())
    assert payload["selection"]["mode"] == "verify_all"
    assert payload["selection"]["skipped"] == 0
    assert all(d["reason_code"] in {"verify_all", "exempt:structural", "exempt:wonder"}
        for d in payload["selection"]["decisions"]
    )
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(multi_stack_target / ".daydream" / "deep").read_text())["items"]
    structural = [item for item in items if item['lens'] == 'structural']
    language = [item for item in items if item['lens'] == 'per-stack']
    assert structural and language
    prompts = [call['prompt'] for call in stub.calls if _is_verifier_prompt(call['prompt'])]
    assert prompts
    assert all(item['description'] not in prompt for item in structural for prompt in prompts)
    assert all(any(item['description'] in prompt for prompt in prompts) for item in language)
