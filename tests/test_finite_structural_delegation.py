"""Finite primary reviews may own the default structural lens explicitly."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends.pi import PiBackend
from daydream.deep.detection import StackAssignment
from daydream.deep.finite_review import FiniteResult, FiniteReview
from daydream.deep.prompts import build_per_stack_prompt
from daydream.extensions import Registry
from daydream.phases import phase_per_stack_reviews
from daydream.review_profile import build_default_profile
from daydream.run_context import InteractionPolicy, RunContext
from daydream.workspace import WorkContext


@pytest.mark.parametrize("mode", [
    "complete", "incomplete", "custom_structure", "custom_primary", "custom_builder", "unowned", "large",
])
async def test_structural_delegation_requires_complete_default_primary_packets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext], mode: str,
) -> None:
    files = ["api.py", "web.ts"]
    for path in files:
        (tmp_path / path).write_text("value = 1\n")
    diff = tmp_path / "diff.patch"
    diff.write_text("".join(
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-value = 0\n+value = 1\n"
        for path in files
    ) + (" " * 65536 if mode == "large" else ""))
    intent = tmp_path / "intent.md"
    intent.write_text("Preserve the interface.")
    alternatives = tmp_path / "alternatives.json"
    alternatives.write_text("[]")
    strategies = {name: strategy.content for name, strategy in build_default_profile().strategies.items()}
    if mode == "custom_structure":
        strategies["discovery.structural"] += "\nCUSTOM POLICY"
    if mode == "custom_primary":
        strategies["discovery.per_stack"] += "\nCUSTOM POLICY"
    if mode == "custom_builder":
        registry = Registry()
        registry.override_prompt("structural", lambda **_: "CUSTOM STRUCTURAL BUILDER")
        registry.override_prompt("per-stack", build_per_stack_prompt)
        monkeypatch.setattr("daydream.deep.finite_review.get_registry", lambda: registry)
    structural_files = files + (["unowned.py"] if mode == "unowned" else [])
    stacks = [StackAssignment("python", [files[0]]), StackAssignment("react", [files[1]]),
              StackAssignment("structure", structural_files)]
    packets: list[FiniteReview] = []
    traditional: list[str] = []

    async def finite(*args: Any, **kwargs: Any) -> FiniteResult:
        review: FiniteReview = args[2]
        packets.append(review)
        incomplete = mode == "incomplete" and review.sources[0].path == files[0]
        return FiniteResult(
            {"issues": [], "verdicts": [{"path": source.path, "lines_read": 1,
             "verdict": "not_reviewed" if incomplete else "clean", "n_findings": 0} for source in review.sources]},
            "evidence_incomplete" if incomplete else None,
            frozenset() if incomplete else frozenset(source.path for source in review.sources),
            tuple(source.metadata() for source in review.sources),
        )

    async def normal(*args: Any, **kwargs: Any) -> Any:
        traditional.append(args[2])
        return {"issues": [], "verdicts": []}, None, None

    monkeypatch.setattr("daydream.deep.finite_review.run_finite_review", finite)
    monkeypatch.setattr("daydream.phases.run_agent", normal)
    results, failures = await phase_per_stack_reviews(
        PiBackend(model="test", reasoning_effort="high"), make_work(tmp_path), stacks,
        diff_path=diff, diff_text=diff.read_text(), intent_path=intent,
        alternatives_path=alternatives, strategies=strategies, allow_standalone=True,
        write_coverage_receipts=True,
        run_context=RunContext(InteractionPolicy(interactive=False)),
    )
    deep = tmp_path / ".daydream/deep"
    delegation = deep / "structural-delegation.json"
    if mode not in {"complete", "incomplete"}:
        assert traditional
        assert not delegation.exists()
        return
    assert traditional == []
    assert len(packets) == 2
    assert set(results) == {"python", "react", "structure"}
    for packet in packets:
        assert "global_changed_file_partition" in packet.prompt
        assert all(path in packet.prompt for path in files)
        assert "canonical" in packet.system_instructions
        assert "shared contracts" in packet.system_instructions
    saved = json.loads(delegation.read_text())
    assert saved["primary_scopes"] == {"python": ["api.py"], "react": ["web.ts"]}
    assert saved["structural_files"] == files
    assert json.loads((deep / "stack-structure-records.json").read_text())["verdicts"] == []
    assert ("python" in failures) == (mode == "incomplete")
