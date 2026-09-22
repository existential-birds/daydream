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
        registry=registry,
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
        assert 'lens="per-stack"' in packet.system_instructions
        assert 'lens="structural"' in packet.system_instructions
        assert "Review the repository-wide interactions" not in packet.system_instructions
        assert "global context does not add review targets" in packet.system_instructions
    saved = json.loads(delegation.read_text())
    assert saved["primary_scopes"] == {"python": ["api.py"], "react": ["web.ts"]}
    assert saved["structural_files"] == files
    assert json.loads((deep / "stack-structure-records.json").read_text())["verdicts"] == []
    assert ("python" in failures) == (mode == "incomplete")


@pytest.mark.parametrize("mode", ["mixed", "structural_only", "missing_lens", "invalid_lens", "incomplete"])
async def test_delegated_findings_route_by_lens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Callable[..., WorkContext], mode: str,
) -> None:
    import copy

    import anyio

    from daydream.phases import PER_STACK_RECORD_SCHEMA

    scopes = {"python": ["api.py"], "react": ["web.ts"]}
    for files in scopes.values():
        (tmp_path / files[0]).write_text("value = 1\n")
    diff = tmp_path / "diff.patch"
    diff.write_text("".join(
        f"diff --git a/{file} b/{file}\n--- a/{file}\n+++ b/{file}\n@@ -1 +1 @@\n-value = 0\n+value = 1\n"
        for files in scopes.values() for file in files
    ))
    intent = tmp_path / "intent.md"
    intent.write_text("Preserve shared contracts")
    alternatives = tmp_path / "alternatives.json"
    alternatives.write_text("[]")
    original_schema = copy.deepcopy(PER_STACK_RECORD_SCHEMA)
    outputs: list[dict[str, Any]] = []
    schemas: list[dict[str, Any]] = []
    finished: list[str] = []
    fallback: list[str] = []

    async def finite(*args: Any, **kwargs: Any) -> FiniteResult:
        review = args[2]
        file = review.sources[0].path
        if file == "api.py":
            await anyio.sleep(0.01)  # Complete in reverse scope order.
        schemas.append(kwargs["schema"])
        issue = {"id": 1, "file": file, "line": 1, "description": f"{file} boundary mismatch",
                 "severity": "high", "confidence": "HIGH", "rationale": "Shared contract differs",
                 "evidence": f"{file}:1", "lens": "structural"}
        issues = [issue]
        if mode != "structural_only":
            issues.append({**issue, "description": f"{file} local defect", "lens": "per-stack"})
        if file == "api.py" and mode == "missing_lens":
            issue.pop("lens")
        if file == "api.py" and mode == "invalid_lens":
            issue["lens"] = "unknown"
        incomplete = file == "api.py" and mode == "incomplete"
        output = {"issues": issues, "verdicts": [{"path": file, "lines_read": 1,
                  "verdict": "not_reviewed" if incomplete else "has_findings", "n_findings": len(issues)}]}
        outputs.append(output)
        finished.append(file)
        return FiniteResult(output, "evidence_incomplete" if incomplete else None,
                            frozenset() if incomplete else frozenset([file]),
                            tuple(source.metadata() for source in review.sources))

    async def normal(*args: Any, **kwargs: Any) -> Any:
        fallback.append(args[2])
        return {"issues": [{"id": 1, "file": "api.py", "line": 1, "description": "Fallback boundary finding",
                            "severity": "high", "confidence": "HIGH", "rationale": "Independent fallback",
                            "evidence": "api.py:1"}], "verdicts": []}, None, None

    monkeypatch.setattr("daydream.deep.finite_review.run_finite_review", finite)
    monkeypatch.setattr("daydream.phases.run_agent", normal)
    _, failures = await phase_per_stack_reviews(
        PiBackend(model="test", reasoning_effort="high"), make_work(tmp_path),
        [*(StackAssignment(name, files) for name, files in scopes.items()),
         StackAssignment("structure", ["api.py", "web.ts"])],
        diff_path=diff, diff_text=diff.read_text(), intent_path=intent, alternatives_path=alternatives,
        allow_standalone=True, run_context=RunContext(InteractionPolicy(interactive=False)),
    )
    deep = tmp_path / ".daydream/deep"
    structural = json.loads((deep / "stack-structure-records.json").read_text())
    assert PER_STACK_RECORD_SCHEMA == original_schema
    if mode in {"missing_lens", "invalid_lens", "incomplete"}:
        assert "python" in failures
        assert len(fallback) == 1
        assert not (deep / "structural-delegation.json").exists()
        assert "delegated_to" not in structural
        assert [issue["description"] for issue in structural["issues"]] == ["Fallback boundary finding"]
        return
    assert failures == {}
    assert fallback == []
    assert finished == ["web.ts", "api.py"]
    assert structural["delegated_to"] == list(scopes)
    assert structural["verdicts"] == []
    assert [issue["uid"] for issue in structural["issues"]] == ["structure:1", "structure:2"]
    assert [issue["description"] for issue in structural["issues"]] == [
        "api.py boundary mismatch", "web.ts boundary mismatch",
    ]
    for name, files in scopes.items():
        saved = json.loads((deep / f"stack-{name}-records.json").read_text())
        assert [issue["uid"] for issue in saved["issues"]] == ([] if mode == "structural_only" else [f"{name}:1"])
        assert all(issue["description"] == f"{files[0]} local defect" for issue in saved["issues"])
        assert saved["verdicts"][0]["n_findings"] == (0 if mode == "structural_only" else 1)
        assert saved["verdicts"][0]["verdict"] == ("clean" if mode == "structural_only" else "has_findings")
        assert all("lens" not in issue for issue in saved["issues"])
    assert all("lens" not in issue for issue in structural["issues"])
    assert all("lens" in issue and "uid" not in issue for output in outputs for issue in output["issues"])
    for schema in schemas:
        item = schema["properties"]["issues"]["items"]
        assert item["properties"]["lens"]["enum"] == ["per-stack", "structural"]
        assert "lens" in item["required"]
