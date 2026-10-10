"""Review Merge And Verifier."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import runner as _runner
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.config_file import DaydreamFileConfig
from daydream.deep import dedup as _dedup, detection as _detection, prompts as _prompts
from daydream.deep.artifacts import (
    DeepArtifact,
    deep_dir,
    per_stack_records_path,
)
from daydream.deep.diff import _diff_changed_files
from daydream.findings import load_findings_artifact
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.deep_orchestrator.support import (
    _install_accept_gate_pipeline,
    _install_post_recorder,
)
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.review_result import saved_coverage
from tests.test_deep_orchestrator import (
    _TWIN_DESCRIPTION,
    Mute,
    _force_interactive,
    _install_model_capturing_stubs,
    _install_stub_backend,
    _pin_findings_pr,
    _prime_merge_resume,
    _profile_with_pipeline,
    _record,
    _run_deep,
    _silence,
    _twin_parse_by_stack,
    _write_plugin_registry,
)


async def test_cold_empty_builtin_merge_uses_no_provider_and_writes_canonical_report(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The real runner must synthesize a clean cold review without merge calls."""
    backend = EmptyReviewBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    trajectory = tmp_path / "empty-trajectory.json"

    assert await _runner.run(empty_review_config(multi_stack_target, trajectory)) == 0

    assert not any("cross-stack merge agent" in call["prompt"].lower() for call in backend.calls)
    assert not any("supervisor adjudication" in call["prompt"].lower() for call in backend.calls)
    deep = multi_stack_target / ".daydream" / "deep"
    payload = json.loads((deep / "merged-items.json").read_text())
    assert payload == {"items": [], "held": []}
    assert DeepArtifact.MERGED_REPORT.at(deep).is_file()
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).read_text() == DeepArtifact.MERGED_REPORT.at(deep).read_text()
    assert not (deep / "merge-failed.txt").exists()
    assert not saved_coverage(deep).unfinished_scopes
    events = json.loads(trajectory.read_text())["extra"]["phase_events"]
    for phase, stage in (("merge", "cross-stack-agent"), ("deep", "supervise")):
        lifecycle = [event for event in events if event["phase"] == phase
                     and event.get("metadata", {}).get("stage") == stage]
        assert [event["event"] for event in lifecycle] == ["phase_start", "phase_end"]
        assert lifecycle[-1]["status"] == "succeeded"


async def test_cold_structural_only_merge_preserves_identity_and_supervises_findings(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    structural = _record(description="Structural boundary erosion", file="api.py", line=1,
                         severity="medium", confidence="MEDIUM", rationale="boundary coupling", evidence="api.py:1")
    backend = EmptyReviewBackend(multi_stack_target, forbid_supervise=False,
                                 review_by_stack={"structure": [structural]})
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)

    assert await _runner.run(empty_review_config(multi_stack_target, tmp_path / "trajectory.json")) == 0

    assert not any("cross-stack merge agent" in call["prompt"].lower() for call in backend.calls)
    assert sum("supervisor adjudication" in call["prompt"].lower() for call in backend.calls) == 1
    deep = multi_stack_target / ".daydream" / "deep"
    records = json.loads(per_stack_records_path(deep, "structure").read_text())["issues"]
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(deep).read_text())["items"]
    assert len(items) == 1
    assert items[0]["description"] == structural["description"]
    assert items[0]["lens"] == "structural"
    assert items[0]["source_uids"] == [records[0]["uid"]]
    assert items[0]["file"] == structural["file"]
    assert items[0]["line"] == structural["line"]
    assert structural["description"] in (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()


@pytest.mark.parametrize("budget_stop", [False, True], ids=["provider-failure", "incomplete-coverage"])
async def test_cold_empty_merge_preserves_failed_stack_diagnostics(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], budget_stop: bool,
) -> None:
    backend = EmptyReviewBackend(multi_stack_target, fail_stack=None if budget_stop else "python",
                                 incomplete_stack="python" if budget_stop else None)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)

    policy = DaydreamFileConfig(supervisor="llm", tool_supervisor="rules", tool_bash_deny=["blocked-review-tool"])
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    output = tmp_path / "findings.json"
    config = empty_review_config(multi_stack_target, tmp_path / "trajectory.json", file_config=policy,
                                 findings_out=str(output), pr_number=pr.number)
    assert await _runner.run(config) == 0

    assert not any("cross-stack merge agent" in call["prompt"].lower() for call in backend.calls)
    assert not any("supervisor adjudication" in call["prompt"].lower() for call in backend.calls)
    deep = multi_stack_target / ".daydream" / "deep"
    assert json.loads(DeepArtifact.MERGED_ITEMS.at(deep).read_text()) == {"items": [], "held": []}
    failures = saved_coverage(deep).unfinished_scopes
    assert "python" in failures
    assert "python" in capsys.readouterr().out
    assert "Review incomplete" in (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "python" in DeepArtifact.MERGED_REPORT.at(deep).read_text()
    loaded = load_findings_artifact(output, expected_repo="o/r", expected_pr_number=pr.number,
                                    expected_head_sha=pr.head_sha)
    assert loaded.findings == []
    assert any("python" in warning for warning in loaded.review_warnings)
    public = json.loads(output.read_text())
    assert public["schema_version"] == 2
    result = public["terminal_result"]
    assert result["analysis_state"] == "incomplete"
    assert result["pipeline_state"] == "completed"
    assert set(result["completed_stacks"]) == {"generic", "react", "structure"}
    failed_scope = next(row for row in result["stack_outcomes"] if row["scope_id"] == "python")
    assert failed_scope["reason_codes"] == ["policy_veto" if budget_stop else "backend_failure"]
    if budget_stop:
        assert failures["python"].startswith("budget exhausted:")
    else:
        assert "review provider unavailable" in failures["python"]


@pytest.mark.parametrize("input_kind", ["language", "alternatives", "custom"])
async def test_cold_merge_with_model_owned_inputs_or_custom_strategy_dispatches(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, input_kind: str,
) -> None:
    backend = EmptyReviewBackend(multi_stack_target, forbid_merge=False, forbid_supervise=False)
    config_overrides: dict[str, Any] = {}
    if input_kind == "language":
        backend.review_by_stack = {"python": [_record(description="Language defect", file="api.py", severity="medium",
                                                      confidence="MEDIUM", rationale="stub", evidence="api.py:1")]}
        backend.merge_echo_records = True
    elif input_kind == "alternatives":
        backend.alternatives = [{"id": 1, "title": "Alternative design", "description": "Missing reusable boundary",
                                 "recommendation": "Extract boundary", "severity": "low", "files": ["api.py"],
                                 "confidence": "MEDIUM", "rationale": "Boundary is reused", "evidence": "api.py:1"}]
        config_overrides["review_profile"] = independent_alternatives_profile()
    else:
        resolved = _profile_with_pipeline()
        strategies = dict(resolved.profile.strategies)
        strategies["merge"] = replace(strategies["merge"], content=strategies["merge"].content + "\nCustom synthesis.")
        config_overrides["review_profile"] = replace(resolved, profile=replace(resolved.profile, strategies=strategies))
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)

    config = empty_review_config(multi_stack_target, tmp_path / "trajectory.json", **config_overrides)
    assert await _runner.run(config) == 0

    assert sum("cross-stack merge agent" in call["prompt"].lower() for call in backend.calls) == 1
    deep = multi_stack_target / ".daydream" / "deep"
    assert DeepArtifact.MERGED_ITEMS.at(deep).is_file()
    if input_kind == "language":
        items = json.loads(DeepArtifact.MERGED_ITEMS.at(deep).read_text())["items"]
        assert items[0]["description"] == "Language defect"
        supervisors = sum("supervisor adjudication" in call["prompt"].lower() for call in backend.calls)
        assert supervisors == 1
    if input_kind == "alternatives":
        assert json.loads((deep / "alternatives.json").read_text()) == backend.alternatives


def _install_merge_captures(
    monkeypatch: pytest.MonkeyPatch, *, captured_merge: dict[str, Any], captured_record_dedup: dict[str, Any],
    captured_dedup_records: dict[str, Any] | None = None,
) -> None:
    """Wrap the merge/dedup builders to capture their arguments."""
    real_build_merge = _prompts.build_merge_prompt
    real_build_dedup = _dedup.build_dedup_candidates
    real_build_record_dedup = _dedup.build_record_dedup_candidates

    def _capture_merge(**kwargs: Any) -> Any:
        captured_merge.update(kwargs)
        return real_build_merge(**kwargs)

    def _capture_dedup(records: Any, alt_issues: Any) -> Any:
        if captured_dedup_records is not None:
            captured_dedup_records["records"] = list(records)
        return real_build_dedup(records, alt_issues)

    def _capture_record_dedup(records: Any, sources: Any) -> Any:
        captured_record_dedup["records"] = list(records)
        captured_record_dedup["sources"] = list(sources)
        return real_build_record_dedup(records, sources=sources)

    monkeypatch.setattr("daydream.deep.prompts.build_merge_prompt", _capture_merge)
    monkeypatch.setattr("daydream.deep.merge_steps.build_dedup_candidates", _capture_dedup)
    monkeypatch.setattr("daydream.deep.merge_steps.build_record_dedup_candidates", _capture_record_dedup,)

def test_run_deep_routes_detected_react_to_react_stack_without_plugin(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:


    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target, pin_skill_availability=False)
    # A populated or empty plugin registry must not change built-in routing (M1):
    # react stays its own stack even with only python installed.
    _write_plugin_registry(tmp_path, ["beagle-python"])
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    captured: dict[str, list[_detection.StackAssignment]] = {}
    real_detect = _detection.detect_stacks

    def _spy(files: list[str], **kwargs: Any) -> list[_detection.StackAssignment]:
        result = real_detect(files, **kwargs)
        captured["stacks"] = result
        return result

    monkeypatch.setattr("daydream.deep.orchestrator.detect_stacks", _spy)

    exit_code = anyio.run(_run_deep, multi_stack_target)
    assert exit_code == 0

    stacks = {s.stack_name for s in captured["stacks"]}
    # M1/M3: built-in routing is registry-independent and never degrades a
    # detected stack to generic -- react stays its own stack regardless of
    # which plugins are installed.
    assert "python" in stacks
    assert "react" in stacks

def test_diff_changed_files_rename_single_entry() -> None:
    """Rename diff contributes only the destination path, not both sides."""
    rename_diff = (
        "diff --git a/foo.py b/foo.ts\n"
        "similarity index 85%\n"
        "rename from foo.py\n"
        "rename to foo.ts\n"
        "--- a/foo.py\n"
        "+++ b/foo.ts\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+const x = 1;\n"
    )
    assert _diff_changed_files(rename_diff) == ["foo.ts"]

def test_diff_changed_files_handles_modify_add_delete_binary() -> None:
    """Non-rename diff shapes emit exactly one path each."""
    mixed = (
        "diff --git a/keep.py b/keep.py\n"
        "--- a/keep.py\n"
        "+++ b/keep.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
        "diff --git a/old.py b/old.py\n"
        "deleted file mode 100644\n"
        "--- a/old.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-x = 1\n"
        "diff --git a/logo.png b/logo.png\n"
        "index 1234..5678 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    assert _diff_changed_files(mixed) == ["keep.py", "new.py", "old.py", "logo.png"]

async def test_failed_per_stack_surfaces_to_merge_prompt_and_persists(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-stack agent failure must: 1) persist to typed review coverage under .daydream/deep/, 2) appear in
    the merge prompt under an 'Uncovered stacks' block, so the merge agent can call it out instead of silently
    ignoring the gap."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # Wrap execute so only the REACT per-stack prompt raises; everything else
    # keeps the stub's normal behavior.
    original_execute = stub.execute

    def _maybe_fail(cwd: Any, prompt: str, output_schema: Any = None, continuation: Any = None, agents: Any = None,
        max_turns: Any = None, read_only: Any = False, persist_session: Any = True,
    ) -> Any:
        pl = prompt.lower()
        if "you are reviewing the react stack" in pl:
            async def _raise() -> None:
                raise RuntimeError("simulated react failure")

            async def _fail() -> AsyncIterator[Any]:
                await _raise()
                yield  # pragma: no cover -- unreachable; satisfies async-gen typing

            return _fail()
        return original_execute(
            cwd, prompt, output_schema, continuation, agents, max_turns=max_turns, read_only=read_only,
            persist_session=persist_session,
        )

    stub.execute = _maybe_fail  # type: ignore[method-assign]
    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    failures_payload = saved_coverage(multi_stack_target / ".daydream/deep").unfinished_scopes
    assert "react" in failures_payload
    assert "simulated react failure" in failures_payload["react"]

    merge_prompts = [c["prompt"] for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert merge_prompts, "merge agent was not invoked"
    prompt = merge_prompts[0]
    assert "Uncovered stacks" in prompt
    assert "react" in prompt
    assert "simulated react failure" in prompt

async def test_resume_merge_errors_on_missing_stack_records(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    # Records for python only; react and generic are missing.
    _prime_merge_resume(multi_stack_target, python=[_record(description="py issue")])
    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 1
    # Merge agent must NOT have run -- the orchestrator bailed before it.
    merge_calls = [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert merge_calls == []

async def test_resume_merge_allows_missing_records_for_failed_stacks(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # No records for the generic bucket, but it's listed as a prior failure.
    deep = _prime_merge_resume(multi_stack_target, python=[_record(description="py issue")],
        react=[_record(description="tsx issue", file="App.tsx")], structure=[_record(description="structural issue")],
    )
    coverage = saved_coverage(deep)
    coverage.record_scope("generic", "failed", reasons=("backend_failure",),
                          diagnostic="simulated generic failure")
    (deep / "review-coverage.json").write_text(json.dumps(coverage.to_dict()))

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    merge_calls = [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert len(merge_calls) == 1

async def test_orchestrator_partitions_structural_records_from_merge(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_merge: dict[str, Any] = {}
    captured_dedup_records: dict[str, Any] = {}
    captured_record_dedup: dict[str, Any] = {}
    _install_merge_captures(monkeypatch, captured_merge=captured_merge, captured_dedup_records=captured_dedup_records,
        captured_record_dedup=captured_record_dedup,
    )

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    # The structural record carries a sentinel id so we can verify it never lands
    # in the dedup input lists.
    _prime_merge_resume(multi_stack_target, python=[_record(id=1, description="py issue")],
        react=[_record(id=2, description="tsx issue", file="App.tsx")],
        generic=[_record(id=3, description="docs issue", file="README.md")],
        structure=[_record(id=4, description="1000-line file budget violated")],
    )

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    # (1) per_stack_records_paths must NOT include the structural file (the host
    #     appends it separately).
    per_stack_paths = captured_merge["per_stack_records_paths"]
    assert all(p.name != "stack-structure-records.json" for p in per_stack_paths), (
        f"structural records must be partitioned out: {per_stack_paths}"
    )

    # (2) The structural sentinel record must NOT appear in either dedup input.
    def _has_structure(records: list[dict[str, Any]]) -> bool:
        return any(r.get("id") == 4 for r in records)

    assert not _has_structure(captured_dedup_records["records"]), (
        f"structural records leaked into build_dedup_candidates: {captured_dedup_records['records']}"
    )
    assert not _has_structure(captured_record_dedup["records"]), (
        f"structural records leaked into build_record_dedup_candidates: {captured_record_dedup['records']}"
    )
    # And the sources list must stay parallel to the filtered records list.
    assert len(captured_record_dedup["sources"]) == len(captured_record_dedup["records"])

async def test_orchestrator_partitions_structural_records_from_merge_fresh_run(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_merge: dict[str, Any] = {}
    captured_record_dedup: dict[str, Any] = {}
    _install_merge_captures(monkeypatch, captured_merge=captured_merge, captured_record_dedup=captured_record_dedup,)

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # Structural records file lives under the deep artifact dir.
    per_stack_paths = captured_merge["per_stack_records_paths"]
    assert all(p.name != "stack-structure-records.json" for p in per_stack_paths), (
        f"structural records must be partitioned out (fresh run): {per_stack_paths}"
    )

    # Fresh-run populates record_sources with stack_name, so the partition drops
    # every entry whose source == 'structure'; sources stay parallel to records.
    assert "structure" not in captured_record_dedup["sources"]
    assert len(captured_record_dedup["sources"]) == len(captured_record_dedup["records"])

@pytest.mark.parametrize("structural_line", [1, 0], ids=["same-line", "whole-file"])
async def test_structural_language_twin_is_arbitrated_and_reported_once(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, structural_line: int,
) -> None:
    """Same-line and whole-file structural twins collapse to one high-severity finding."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_echo_records = True
    stub.parse_by_stack = _twin_parse_by_stack(structural_line=structural_line)
    assert await _run_deep(multi_stack_target) == 0

    deep = deep_dir(multi_stack_target, allow_standalone=True)
    arbiter_input = json.loads(DeepArtifact.ARBITER_INPUT.at(deep).read_text())
    observed = sorted((record["file"], record["line"], record["severity"]) for record in arbiter_input)
    expected = sorted([("api.py", 1, "medium"), ("api.py", structural_line, "high")])
    assert observed == expected, arbiter_input
    structural = json.loads(per_stack_records_path(deep, "structure").read_text())
    assert structural["issues"][0]["description"].startswith("ARBITRATED: ")

    items = json.loads(DeepArtifact.MERGED_ITEMS.at(deep).read_text())["items"]
    twins = [item for item in items if item["file"] == "api.py" and _TWIN_DESCRIPTION in item["description"]]
    assert len(twins) == 1, twins
    assert twins[0]["severity"] == "high"
    assert {item["file"] for item in items} == {"api.py", "App.tsx", "README.md"}

async def test_precision_mode_suppression_never_sees_structural_records(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural records rejoin adjudication for the arbiter only (issue #1103)."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_echo_records = True
    stub.suppression_keep = False
    stub.parse_by_stack = {"python": {"severity": "low", "confidence": "MEDIUM", "file": "api.py", "line": 1,
            "description": "Borderline python nit the suppression pass rejects",
        },
        "structure": {"severity": "low", "confidence": "MEDIUM", "file": "api.py", "line": 1,
            "description": "Low-severity structural erosion in this module",
        },
        "react": {"severity": "medium", "confidence": "MEDIUM", "file": "App.tsx", "line": 1,
            "description": "Unrelated React concern",
        },
        "generic": {"severity": "medium", "confidence": "MEDIUM", "file": "README.md", "line": 1,
            "description": "Unrelated docs concern",
        },
    }

    assert await _run_deep(multi_stack_target, precision_mode=True) == 0

    deep = deep_dir(multi_stack_target, allow_standalone=True)
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(deep).read_text())["items"]
    descriptions = [i["description"] for i in items]
    # The borderline LANGUAGE finding is suppressed (that is the pass working).
    assert not any("Borderline python nit" in d for d in descriptions), descriptions
    # The equally borderline STRUCTURAL finding is not -- it was never eligible.
    assert any("Low-severity structural erosion" in d for d in descriptions), descriptions

async def test_precision_suppression_preserves_structural_records_on_resume(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_echo_records = True
    stub.suppression_keep = False
    deep = _prime_merge_resume(multi_stack_target,
        python=[_record(description="Local borderline finding", severity="low", confidence="MEDIUM")],
        react=[], generic=[],
        structure=[_record(description="Structural boundary mismatch", severity="low", confidence="MEDIUM",
                           evidence="api.py:1")],
    )
    assert await _run_deep(multi_stack_target, start_at="merge", precision_mode=True) == 0
    items = json.loads((deep / "merged-items.json").read_text())["items"]
    assert not any(item["description"] == "Local borderline finding" for item in items)
    structural = [item for item in items if item["description"] == "Structural boundary mismatch"]
    assert len(structural) == 1
    assert structural[0]["lens"] == "structural"
    assert structural[0]["source_uids"] == ["structure:1"]

async def test_distinct_structural_finding_survives_the_fold(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fold is a duplicate check, not a structural-lens filter."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_echo_records = True
    overrides = _twin_parse_by_stack(structural_line=1)
    overrides["structure"]["description"] = "Module layering inverted: api.py now imports the CLI"
    stub.parse_by_stack = overrides
    assert await _run_deep(multi_stack_target) == 0
    dd = deep_dir(multi_stack_target, allow_standalone=True)
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text())["items"]
    api_items = sorted(i["description"] for i in items if i["file"] == "api.py")
    assert len(api_items) == 2, f"a distinct structural finding was folded away: {api_items}"
    assert any(i["lens"] == "structural" for i in items)
    assert "## Structural Review" in DeepArtifact.MERGED_REPORT.at(dd).read_text()

async def test_resume_fix_skips_pr_post(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    post_calls: list[dict[str, Any]] = []
    _install_post_recorder(monkeypatch, post_calls)

    # Prime the fix-resume artifacts: the verifier and fix gate both read the
    # canonical merged-items.json, so prime it alongside the markdown report.
    deep = _prime_merge_resume(multi_stack_target)
    (multi_stack_target / REVIEW_OUTPUT_FILE).write_text(
        "# Review\n\n## Issues\n\n1. [api.py:1] primed issue\n   rationale\n"
    )
    (deep / "merged-items.json").write_text(json.dumps({"items": [{
                        "id": 1, "lens": "per-stack", "file": "api.py", "line": 1, "severity": "medium",
                        "description": "primed issue", "confidence": "MEDIUM", "rationale": "rationale",
                    }
                ]
            }
        )
    )

    exit_code = await _run_deep(multi_stack_target, start_at="fix")
    assert exit_code == 0
    assert post_calls == [], (
        f"post_review_to_pr_from_report should be skipped on --start-at fix, got {len(post_calls)} call(s)"
    )

async def test_resolve_backend_called_with_each_phase_in_deep_flow(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute,
) -> None:
    seen_phases: list[str] = []
    original = _runner._resolve_backend

    def spy(config: Any, phase: Any, cache: Any = None, *, cwd: Any = None, audit_workspace: Any = None,
        effort_override: Any = None,
    ) -> Any:
        seen_phases.append(phase)
        return original(config, phase, cache, cwd=cwd, audit_workspace=audit_workspace, effort_override=effort_override,
        )

    # run_deep imports _resolve_backend from daydream.runner, so patching it there
    # intercepts every call site under per-phase resolution.
    monkeypatch.setattr("daydream.runner._resolve_backend", spy)

    # Accept the fix gate so fix/test/commit run; pin interactivity so the "y"
    # stub is honoured instead of the unattended decline default.
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.deep.review_steps.print_stage_progress", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")

    _install_stub_backend(monkeypatch, multi_stack_target)

    # Stub the outward-facing tail phases (they still trigger their resolver call)
    # plus phase_fix, so the fix loop doesn't mutate the workspace.
    mute_side_effects()

    async def _stub_fix(backend: Any, work: Any, item: Any, idx: Any, total: Any, **kwargs: Any) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr("daydream.phases.fix.phase_fix", _stub_fix)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # Issue #745: the pre-merge parse-<stack> stage was removed (reviewers emit
    # records directly), so "parse" is no longer a resolved deep phase.
    expected_phases = {"intent", "per_stack_review", "merge", "fix", "test", "verify"}
    captured = set(seen_phases)
    missing = expected_phases - captured
    assert not missing, f"Deep orchestrator missing per-phase resolver calls for {missing}; got {sorted(captured)}"
    assert "wonder" not in captured  # The default design lens shares per_stack_review.

async def test_intent_phase_runs_on_sonnet_through_runner_run(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#171 real-path: the intent phase Sonnet downgrade must be observable through the runner.run production
    entrypoint, not only at the unit seam."""
    _silence(monkeypatch)
    calls = _install_model_capturing_stubs(
        monkeypatch, multi_stack_target, parse_severity="high", merge_echo_records=True
    )

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # "understand the intent of these changes" is unique to build_intent_prompt
    # (phases.py) -- the established intent-phase prompt discriminator.
    intent_models = [c["model"] for c in calls if "understand the intent of these changes" in c["prompt"].lower()]
    assert intent_models, "intent phase did not execute through runner.run"
    assert set(intent_models) == {"claude-sonnet-5"}, (
        f"intent phase should run on claude-sonnet-5 (mid tier), got {sorted(intent_models)!r}"
    )

async def test_verifier_runs_after_merge_before_fix(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute,
) -> None:
    # phase_fix stays REAL so verdict propagation is observable.
    stub = _install_accept_gate_pipeline(monkeypatch, multi_stack_target, mute_side_effects)
    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    merge_idx: int | None = None
    verifier_idx: int | None = None
    first_fix_idx: int | None = None
    for idx, call in enumerate(stub.calls):
        pl = call["prompt"].lower()
        if merge_idx is None and "cross-stack merge agent" in pl:
            merge_idx = idx
        elif verifier_idx is None and "recommendation-verifier" in pl:
            verifier_idx = idx
        elif first_fix_idx is None and pl.startswith("fix this issue:"):
            first_fix_idx = idx

    assert verifier_idx is not None, "verifier prompt was not dispatched"
    assert merge_idx is not None, "merge prompt was not dispatched"
    assert first_fix_idx is not None, "no fix prompt dispatched -- fix loop did not run"
    assert merge_idx < verifier_idx < first_fix_idx, (
        f"expected merge ({merge_idx}) < verifier ({verifier_idx}) < first fix ({first_fix_idx})"
    )

    # Verdicts JSON lands on disk at the orchestrator-controlled path.
    expected_path = DeepArtifact.VERDICTS.at(multi_stack_target / ".daydream" / "deep")
    assert expected_path == multi_stack_target / ".daydream" / "deep" / "recommendation-verdicts.json"
    assert expected_path.is_file(), f"verdicts file missing at {expected_path}"


    payload = json.loads(expected_path.read_text())
    assert payload["verdicts"] == [
        {"issue_id": 1, "verdict": "consistent", "evidence": "stub", "unverified_assumptions": []}
    ]
    assert payload["selection"]["decisions"]
