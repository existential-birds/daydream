"""Real runner tests of config-to-verifier wiring via persisted selection decisions."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ResultEvent
from daydream.deep.artifacts import DeepArtifact
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.deep_orchestrator.support import _install_accept_gate_pipeline
from tests.test_deep_orchestrator import Mute, _PromptHookStub, _run_deep


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
    fields = {"test_command": f"{sys.executable} -c pass", **overrides}
    config = RunConfig(target=str(target), start_at="review", cleanup=False, **fields)
    return await run(config)

async def test_start_at_fix_resume_reuses_unchanged_verdicts_and_reverifies_changed_ones(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute
) -> None:
    """MH14 real-path: the second run makes one backend verify call, marked reused for the unchanged item."""
    _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)

    class CompleteVerifier(_PromptHookStub):
        def intercept(self, cwd: Path, prompt: str) -> list[ResultEvent] | None:
            if not _is_verifier_prompt(prompt):
                return None
            self.calls.append({"prompt": prompt, "model": self.model})
            ids = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", prompt)]
            assert ids, "the verifier must receive numbered findings"
            return [ResultEvent(structured_output={"verdicts": [
                {"issue_id": item_id, "verdict": "consistent", "evidence": "stub"} for item_id in ids
            ]}, continuation=None)]

    stub = CompleteVerifier(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_a, **_kw: stub)
    assert await _run_deep(multi_stack_target) == 0
    verifier_calls_first = [c for c in stub.calls if _is_verifier_prompt(c["prompt"])]
    assert len(verifier_calls_first) == 1
    path = DeepArtifact.VERDICTS.at(multi_stack_target / ".daydream" / "deep")
    first = json.loads(path.read_text())
    selected = {d["item_uid"]: d for d in first["selection"]["decisions"] if d["selected"]}
    assert "item:2" in selected
    assert selected.keys() - {"item:2"}, "the resume must also prove reuse of an unchanged selected item"
    assert {verdict["issue_id"] for verdict in first["verdicts"]} == {d["item_id"] for d in selected.values()}
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(multi_stack_target / ".daydream" / "deep").read_text())["items"]
    descriptions = {item["item_uid"]: item["description"] for item in items}
    # Rewrite the merged items so item:2's verifier-relevant text changes, then resume at the fix gate.
    changed_evidence = "App.tsx:1 rewritten by hand"
    _mutate_item_evidence(multi_stack_target, item_uid="item:2", evidence=changed_evidence)
    stub.calls.clear()
    assert await _run_deep(multi_stack_target, start_at="fix") == 0
    second = json.loads(path.read_text())
    resumed = [c["prompt"] for c in stub.calls if _is_verifier_prompt(c["prompt"])]
    assert len(resumed) == 1
    assert {int(value) for value in re.findall(r"(?m)^(\d+)\. \[", resumed[0])} == {selected["item:2"]["item_id"]}
    assert descriptions["item:2"] in resumed[0]
    assert all(descriptions[uid] not in resumed[0] for uid in selected if uid != "item:2")
    assert {d["item_uid"]: d["verdict_reused"] for d in second["selection"]["decisions"] if d["selected"]} == {
        uid: uid != "item:2" for uid in selected
    }
    assert {d["item_uid"]: d["content_digest"] != selected[d["item_uid"]]["content_digest"]
            for d in second["selection"]["decisions"] if d["selected"]} == {
        uid: uid == "item:2" for uid in selected
    }

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
