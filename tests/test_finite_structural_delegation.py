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
from daydream.review_profile import FOLDED_ALTERNATIVES_INSTRUCTION, build_default_profile
from daydream.run_context import InteractionPolicy, RunContext
from daydream.workspace import WorkContext


@pytest.mark.parametrize("mode", [
    "complete", "incomplete", "custom_structure", "custom_primary", "custom_builder", "unowned", "large",
    "exception", "timeout", "budget", "invalid", "no_output", "fallback_budget", "fallback_exception", "folded",
    "write_record", "write_markdown", "write_marker",
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
    registry = None
    if mode == "custom_builder":
        registry = Registry()
        registry.override_prompt("structural", lambda **_: "CUSTOM STRUCTURAL BUILDER")
        registry.override_prompt("per-stack", build_per_stack_prompt)
    if mode == "folded":
        strategies["discovery.structural"] += "\n\n" + FOLDED_ALTERNATIVES_INSTRUCTION
    structural_files = files + (["unowned.py"] if mode == "unowned" else [])
    stacks = [StackAssignment("python", [files[0]]), StackAssignment("react", [files[1]]),
              StackAssignment("structure", structural_files)]
    packets: list[FiniteReview] = []
    traditional: list[str] = []
    completed: list[str] = []
    deep = tmp_path / ".daydream/deep"
    delegation = deep / "structural-delegation.json"
    structural_record = deep / "stack-structure-records.json"
    structural_report = deep / "stack-structure-review.md"
    fallback_modes = {"incomplete", "exception", "timeout", "budget", "invalid", "no_output",
                      "fallback_budget", "fallback_exception", "folded"}
    write_failure_modes = {"write_record", "write_markdown", "write_marker"}
    original_write = Path.write_text
    original_replace = Path.replace

    def write(path: Path, *args: Any, **kwargs: Any) -> int:
        if (mode == "write_record" and path == structural_record
                or mode == "write_markdown" and path == structural_report):
            raise OSError("compatibility write failed")
        return original_write(path, *args, **kwargs)

    def replace(path: Path, target: Any) -> Path:
        if target == delegation:
            assert structural_record.exists() and structural_report.exists()
            assert set(completed) == set(files)
            if mode == "write_marker":
                raise OSError("commit marker replacement failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(Path, "replace", replace)

    async def finite(*args: Any, **kwargs: Any) -> FiniteResult:
        review: FiniteReview = args[2]
        packets.append(review)
        completed.append(review.sources[0].path)
        incomplete = mode in fallback_modes and review.sources[0].path == files[0]
        if incomplete and mode == "exception":
            raise RuntimeError("primary failed")
        if incomplete and mode == "timeout":
            raise TimeoutError("primary timed out")
        output: Any = {"issues": [], "verdicts": [{"path": source.path, "lines_read": 1,
            "verdict": "not_reviewed" if incomplete else "clean", "n_findings": 0} for source in review.sources]}
        if incomplete and mode in {"invalid", "no_output"}:
            output = {"issues": "invalid", "verdicts": []} if mode == "invalid" else None
        return FiniteResult(
            output,
            ("wall_budget" if mode == "budget" else "evidence_incomplete")
            if incomplete and mode not in {"invalid", "no_output"} else None,
            frozenset() if incomplete else frozenset(source.path for source in review.sources),
            tuple(source.metadata() for source in review.sources),
        )

    async def normal(*args: Any, **kwargs: Any) -> Any:
        traditional.append(args[2])
        if mode in fallback_modes:
            assert set(completed) == set(files)
            assert not delegation.exists()
        if mode == "fallback_exception":
            raise RuntimeError("structural fallback failed")
        issues = [{"id": 1, "file": "api.py", "line": 1, "description": "Shared contract mismatch",
                   "severity": "high", "confidence": "HIGH", "rationale": "Boundary differs",
                   "evidence": "api.py:1 and web.ts:1"}] if mode == "fallback_budget" else []
        return {"issues": issues, "verdicts": []}, None, "wall_budget" if mode == "fallback_budget" else None

    monkeypatch.setattr("daydream.deep.finite_review.run_finite_review", finite)
    monkeypatch.setattr("daydream.phases.run_agent", normal)
    results, failures = await phase_per_stack_reviews(
        PiBackend(model="test", reasoning_effort="high"), make_work(tmp_path), stacks,
        diff_path=diff, diff_text=diff.read_text(), intent_path=intent,
        alternatives_path=alternatives, strategies=strategies, allow_standalone=True,
        write_coverage_receipts=True,
        **({"registry": registry} if registry is not None else {}),
        run_context=RunContext(InteractionPolicy(interactive=False)),
    )
    if mode in fallback_modes:
        assert len(traditional) == 1
        assert "python" in failures
        assert not delegation.exists()
        if mode == "fallback_exception":
            assert "structure" in failures
            assert "structure" not in results
        else:
            assert "structure" in results
            assert ("structure" in failures) == (mode == "fallback_budget")
            record = json.loads(structural_record.read_text())
            assert "delegated_to" not in record
            if mode == "fallback_budget":
                assert record["incomplete"] is True
                assert record["issues"][0]["uid"] == "structure:1"
        if mode == "folded":
            assert FOLDED_ALTERNATIVES_INSTRUCTION in traditional[0]
        return
    if mode in write_failure_modes:
        assert traditional == []
        assert "structure" in failures
        assert "structure" not in results
        assert not any(path.exists() for path in (delegation, structural_record, structural_report))
        assert not list(deep.glob("structural-delegation*.tmp"))
        return
    if mode == "custom_builder":
        assert traditional == ["CUSTOM STRUCTURAL BUILDER"]
        assert len(packets) == 2
    if mode != "complete":
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
        assert "Review the repository-wide interactions" not in packet.system_instructions
        assert "global context does not add review targets" in packet.system_instructions
    saved = json.loads(delegation.read_text())
    assert saved["primary_scopes"] == {"python": ["api.py"], "react": ["web.ts"]}
    assert saved["structural_files"] == files
    assert json.loads((deep / "stack-structure-records.json").read_text())["verdicts"] == []
    assert ("python" in failures) == (mode == "incomplete")
