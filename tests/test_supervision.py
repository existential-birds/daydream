"""Tests for runtime findings and tool supervision."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

import pytest

from daydream import review_profile
from daydream.backends import ResultEvent
from daydream.deep.artifacts import DeepArtifact, deep_dir, persist_review_coverage
from daydream.deep.merge_steps import _step_supervise
from daydream.deep.prompts import build_supervise_prompt
from daydream.extensions import get_registry
from daydream.flows.engine import FlowContext
from daydream.phases import phase_supervise_review
from daydream.review_result import ReasonCode
from daydream.run_config import RunConfig
from daydream.supervision import revise_finding_fields
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.review_result import review_coverage


def test_revise_finding_fields_updates_whitelist_only() -> None:
    item = {"id": 7, "severity": "high", "confidence": "medium", "description": "original", "rationale": "because",
        "evidence": ["line 1"], "file": "safe.py", "line": 12,
    }
    item_id = id(item)
    revise_finding_fields(item, {"severity": "low", "file": "hacked.py", "line": 999, "reason": "x"},)

    assert id(item) == item_id
    assert item["severity"] == "low"
    assert item["file"] == "safe.py"
    assert item["line"] == 12
    assert item["id"] == 7


@pytest.mark.parametrize("host_stage", [False, True], ids=["provider-contract", "host-coverage-completion"])
@pytest.mark.parametrize("contract", ["default", "packaged", "custom-strategy", "custom-builder", "nonempty"])
async def test_supervise_empty_builtin_skips_provider_and_preserves_input_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext], contract: str,
    host_stage: bool,
) -> None:
    work = make_work(tmp_path)
    dd = deep_dir(work.repo, allow_standalone=True)
    (dd / "supervise-input.json").write_text('[{"id": 99, "description": "stale"}]')
    coverage = review_coverage(phases=("supervision", "arbiter"))
    for phase, message in (("supervision", "stale timeout"), ("arbiter", "unresolved findings")):
        coverage.record_phase(phase, "incomplete", reasons=(ReasonCode.HOST_WALL_BUDGET_EXHAUSTION,),
                              diagnostic=message)
    persist_review_coverage(dd, coverage)
    diff = dd / "diff.patch"
    diff.write_text("diff --git a/foo.py b/foo.py\n+value = 2\n")
    intent = dd / "intent.md"
    intent.write_text("Update value")
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    items = [{"id": 1, "item_uid": "item:1", "lens": "per-stack", "file": "foo.py", "line": 1,
              "description": "finding", "severity": "high", "confidence": "HIGH", "rationale": "rationale",
              "evidence": "foo.py:1", "related_files": None, "source_uids": None}] if contract == "nonempty" else []
    strategy: str | None = review_profile.build_default_profile().strategies["supervision"].content
    if contract == "default":
        strategy = None
    elif contract == "custom-strategy":
        strategy = "Perform a custom supervision pass over {supervise_input_path}."
    elif contract == "custom-builder":
        registry = get_registry()
        registry.override_prompt("supervise", lambda **kwargs: build_supervise_prompt(**kwargs))
        monkeypatch.setattr("daydream.extensions.loader._REGISTRY_VAR", ContextVar("test-registry", default=registry))
    returned_verdicts = [
        {"id": 1, "action": "edit", "reason": "Reduce severity", "severity": "low",
         "confidence": None, "description": None, "rationale": None, "evidence": None},
        {"id": 99, "action": "drop", "reason": "Stale target", "severity": None,
         "confidence": None, "description": None, "rationale": None, "evidence": None},
    ] if contract == "nonempty" else []
    backend = ScriptedBackend(events=[
        ResultEvent(structured_output={"verdicts": returned_verdicts}, continuation=None),
    ])

    if host_stage:
        profile = review_profile.build_default_profile()
        supervision = profile.strategies["supervision"]
        profile = replace(profile, strategies={**profile.strategies, "supervision":
                          replace(supervision, content=strategy or supervision.content)})
        items_file = DeepArtifact.MERGED_ITEMS.at(dd)
        items_file.write_text(json.dumps({"items": items, "held": []}))
        ctx = FlowContext(config=RunConfig(target=str(work.repo)), work=work, registry=get_registry(),
            review_profile=review_profile.ResolvedProfile(profile, "test"), allow_standalone_artifacts=True,
            data={"dd": dd, "review_coverage": coverage, "items_file": items_file, "diff_path": diff,
                  "intent_path": intent, "alts_path": alternatives, "exploration_dir": None,
                  "merged_report": dd / "public-report.md"},
            _backend_factory=lambda *_args: backend)
        await _step_supervise(ctx)
        outcome = coverage.phases["supervision"]
        assert outcome["status"] == ("incomplete" if contract == "nonempty" else "complete")
        assert outcome["reason_codes"] == ([ReasonCode.EVIDENCE_INCOMPLETE] if contract == "nonempty" else [])
        assert json.loads(items_file.read_text())["items"] == ([{**items[0], "severity": "low"}] if items else [])
    else:
        verdicts = await phase_supervise_review(
            backend, work, items=items, diff_path=diff, intent_path=intent, alternatives_path=alternatives,
            strategy=strategy, allow_standalone=True,
        )
        expected = {1: {"id": 1, "action": "edit", "reason": "Reduce severity", "severity": "low"}}
        assert verdicts == (expected if contract == "nonempty" else {})
    assert json.loads((dd / "supervise-input.json").read_text()) == items
    diagnostics = {"arbiter": "unresolved findings"}
    if not host_stage:
        diagnostics["supervision"] = "stale timeout"
    elif contract == "nonempty":
        diagnostics["supervision"] = "evidence_incomplete"
    persisted = json.loads(DeepArtifact.REVIEW_COVERAGE.at(dd).read_text())
    assert coverage.diagnostics["phases"] == persisted["diagnostics"]["phases"] == diagnostics
    assert backend.call_count == (0 if contract in {"default", "packaged"} else 1)
