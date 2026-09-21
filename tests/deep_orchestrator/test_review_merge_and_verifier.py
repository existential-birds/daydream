"""Review Merge And Verifier."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import pytest

from tests.deep_orchestrator.support import (
    _install_accept_gate_pipeline,
)
from tests.harness.review_profile import default_strategy as _default_strategy
from tests.test_deep_orchestrator import (
    _TWIN_DESCRIPTION,
    Mute,
    _force_interactive,
    _install_model_capturing_stubs,
    _install_stub_backend,
    _prime_merge_resume,
    _record,
    _run_deep,
    _silence,
    _twin_parse_by_stack,
    _write_plugin_registry,
)

if TYPE_CHECKING:
    from daydream.pr_review import ReviewRenderers
    from daydream.run_context import RunContext


def _install_merge_captures(
    monkeypatch: pytest.MonkeyPatch,
    *,
    captured_merge: dict[str, Any],
    captured_record_dedup: dict[str, Any],
    captured_dedup_records: dict[str, Any] | None = None,
) -> None:
    """Wrap the merge/dedup builders to capture their arguments."""
    from daydream.deep import dedup as _dedup
    from daydream.deep import prompts as _prompts

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
    monkeypatch.setattr(
        "daydream.deep.merge_steps.build_record_dedup_candidates",
        _capture_record_dedup,
    )


def test_run_deep_routes_detected_react_to_react_stack_without_plugin(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Detected React files retain their own stack without the plugin registry."""

    from daydream.deep import detection as _detection

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
    from daydream.deep.diff import _diff_changed_files

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
    from daydream.deep.diff import _diff_changed_files

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


async def test_merge_prompt_lists_records_in_sorted_order(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The merge prompt lists per-stack records in stable order."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    merge_prompts = [c["prompt"] for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert merge_prompts, "merge agent was not invoked"
    prompt = merge_prompts[0]

    # Records appear under "Per-stack parsed records:" as "  - <path>" lines.
    lines = prompt.splitlines()
    start = next((i for i, line in enumerate(lines) if "per-stack parsed records:" in line.lower()), None)
    assert start is not None, "merge prompt missing per-stack records block"

    record_paths: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("  - "):
            record_paths.append(line[4:].strip())
        elif line.strip() == "":
            break
        else:
            break

    assert record_paths, "no record paths found in merge prompt"
    assert record_paths == sorted(record_paths), f"records not in sorted order: {record_paths}"


def test_merge_prompt_emits_related_files_instruction() -> None:
    from pathlib import Path

    from daydream.deep.prompts import build_merge_prompt

    prompt = build_merge_prompt(
        strategy=_default_strategy("merge"),
        per_stack_records_paths=[Path("a-records.json")],
        intent_path=Path("intent.md"),
        alternatives_path=Path("alt.md"),
        dedup_candidates_path=Path("dedup.json"),
    )
    assert "related_files" in prompt


async def test_failed_per_stack_surfaces_to_merge_prompt_and_persists(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-stack agent failure must: 1) persist to per-stack-failures.json under .daydream/deep/, 2) appear in
    the merge prompt under an 'Uncovered stacks' block, so the merge agent can call it out instead of silently
    ignoring the gap."""
    import json as _json

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # Wrap execute so only the REACT per-stack prompt raises; everything else
    # keeps the stub's normal behavior.
    original_execute = stub.execute

    def _maybe_fail(
        cwd: Any,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: Any = False,
        persist_session: Any = True,
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
            cwd,
            prompt,
            output_schema,
            continuation,
            agents,
            max_turns=max_turns,
            read_only=read_only,
            persist_session=persist_session,
        )

    stub.execute = _maybe_fail  # type: ignore[method-assign]

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    failures_p = multi_stack_target / ".daydream" / "deep" / "per-stack-failures.json"
    assert failures_p.is_file(), "failures file should be persisted for merge-resume"
    failures_payload = _json.loads(failures_p.read_text())
    assert "react" in failures_payload
    assert "simulated react failure" in failures_payload["react"]

    merge_prompts = [c["prompt"] for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert merge_prompts, "merge agent was not invoked"
    prompt = merge_prompts[0]
    assert "Uncovered stacks" in prompt
    assert "react" in prompt
    assert "simulated react failure" in prompt


async def test_resume_merge_errors_on_missing_stack_records(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--start-at merge must fail loudly when a detected stack has no records."""
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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stack listed in per-stack-failures.json is allowed to be missing."""
    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)

    # No records for the generic bucket, but it's listed as a prior failure.
    deep = _prime_merge_resume(
        multi_stack_target,
        python=[_record(description="py issue")],
        react=[_record(description="tsx issue", file="App.tsx")],
        structure=[_record(description="structural issue")],
    )
    (deep / "per-stack-failures.json").write_text(json.dumps({"generic": "simulated generic failure"}))

    exit_code = await _run_deep(multi_stack_target, start_at="merge")
    assert exit_code == 0

    merge_calls = [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert len(merge_calls) == 1


async def test_orchestrator_partitions_structural_records_from_merge(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural records are partitioned out of the dedup pre-filter pool."""

    captured_merge: dict[str, Any] = {}
    captured_dedup_records: dict[str, Any] = {}
    captured_record_dedup: dict[str, Any] = {}
    _install_merge_captures(
        monkeypatch,
        captured_merge=captured_merge,
        captured_dedup_records=captured_dedup_records,
        captured_record_dedup=captured_record_dedup,
    )

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    # The structural record carries a sentinel id so we can verify it never lands
    # in the dedup input lists.
    _prime_merge_resume(
        multi_stack_target,
        python=[_record(id="py-1", description="py issue")],
        react=[_record(id="react-1", description="tsx issue", file="App.tsx")],
        generic=[_record(id="generic-1", description="docs issue", file="README.md")],
        structure=[_record(id="structure-1", description="1000-line file budget violated")],
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
        return any(str(r.get("id", "")).startswith("structure") for r in records)

    assert not _has_structure(captured_dedup_records["records"]), (
        f"structural records leaked into build_dedup_candidates: {captured_dedup_records['records']}"
    )
    assert not _has_structure(captured_record_dedup["records"]), (
        f"structural records leaked into build_record_dedup_candidates: {captured_record_dedup['records']}"
    )
    # And the sources list must stay parallel to the filtered records list.
    assert len(captured_record_dedup["sources"]) == len(captured_record_dedup["records"])


async def test_orchestrator_partitions_structural_records_from_merge_fresh_run(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh-run path (no start_at) applies the same structural partition."""

    captured_merge: dict[str, Any] = {}
    captured_record_dedup: dict[str, Any] = {}
    _install_merge_captures(
        monkeypatch,
        captured_merge=captured_merge,
        captured_record_dedup=captured_record_dedup,
    )

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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    structural_line: int,
) -> None:
    """Same-line and whole-file structural twins collapse to one high-severity finding."""
    from daydream.deep.artifacts import (
        arbiter_input_path,
        deep_dir,
        merged_items_path,
        per_stack_records_path,
    )

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_echo_records = True
    stub.parse_by_stack = _twin_parse_by_stack(structural_line=structural_line)
    assert await _run_deep(multi_stack_target) == 0

    deep = deep_dir(multi_stack_target, allow_standalone=True)
    arbiter_input = json.loads(arbiter_input_path(deep).read_text())
    observed = sorted((record["file"], record["line"], record["severity"]) for record in arbiter_input)
    expected = sorted([("api.py", 1, "medium"), ("api.py", structural_line, "high")])
    assert observed == expected, arbiter_input
    structural = json.loads(per_stack_records_path(deep, "structure").read_text())
    assert structural["issues"][0]["description"].startswith("ARBITRATED: ")

    items = json.loads(merged_items_path(deep).read_text())["items"]
    twins = [item for item in items if item["file"] == "api.py" and _TWIN_DESCRIPTION in item["description"]]
    assert len(twins) == 1, twins
    assert twins[0]["severity"] == "high"
    assert {item["file"] for item in items} == {"api.py", "App.tsx", "README.md"}


async def test_precision_mode_suppression_never_sees_structural_records(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural records rejoin adjudication for the arbiter only (issue #1103)."""
    from daydream.deep.artifacts import deep_dir, merged_items_path

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_echo_records = True
    stub.suppression_keep = False
    stub.parse_by_stack = {
        "python": {
            "severity": "low",
            "confidence": "MEDIUM",
            "file": "api.py",
            "line": 1,
            "description": "Borderline python nit the suppression pass rejects",
        },
        "structure": {
            "severity": "low",
            "confidence": "MEDIUM",
            "file": "api.py",
            "line": 1,
            "description": "Low-severity structural erosion in this module",
        },
        "react": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "App.tsx",
            "line": 1,
            "description": "Unrelated React concern",
        },
        "generic": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "README.md",
            "line": 1,
            "description": "Unrelated docs concern",
        },
    }

    assert await _run_deep(multi_stack_target, precision_mode=True) == 0

    items = json.loads(merged_items_path(deep_dir(multi_stack_target, allow_standalone=True)).read_text())["items"]
    descriptions = [i["description"] for i in items]
    # The borderline LANGUAGE finding is suppressed (that is the pass working).
    assert not any("Borderline python nit" in d for d in descriptions), descriptions
    # The equally borderline STRUCTURAL finding is not -- it was never eligible.
    assert any("Low-severity structural erosion" in d for d in descriptions), descriptions


async def test_distinct_structural_finding_survives_the_fold(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fold is a duplicate check, not a structural-lens filter."""
    from daydream.deep.artifacts import deep_dir, merged_items_path, merged_report_path

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_echo_records = True
    overrides = _twin_parse_by_stack(structural_line=1)
    overrides["structure"]["description"] = "Module layering inverted: api.py now imports the CLI"
    stub.parse_by_stack = overrides

    assert await _run_deep(multi_stack_target) == 0

    dd = deep_dir(multi_stack_target, allow_standalone=True)
    items = json.loads(merged_items_path(dd).read_text())["items"]
    api_items = sorted(i["description"] for i in items if i["file"] == "api.py")
    assert len(api_items) == 2, f"a distinct structural finding was folded away: {api_items}"
    assert any(i["lens"] == "structural" for i in items)
    assert "## Structural Review" in merged_report_path(dd).read_text()


async def test_resume_fix_skips_pr_post(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--start-at fix must not call post_review_to_pr_from_report."""
    from daydream.config import REVIEW_OUTPUT_FILE

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    post_calls: list[dict[str, Any]] = []

    async def _spy(
        target_dir: Path,
        merged_items_path: Path,
        *,
        console: Any,
        run_info: str,
        renderers: ReviewRenderers,
        post: bool = False,
        approve_on_clean: bool = False,
        pr_number: int | None = None,
        diagram_blocks: str | None = None,
        run_context: RunContext | None = None,
        auth: Any,
    ) -> None:
        post_calls.append({"target_dir": target_dir, "report_path": merged_items_path})

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _spy)

    # Prime the fix-resume artifacts: the verifier and fix gate both read the
    # canonical merged-items.json, so prime it alongside the markdown report.
    deep = _prime_merge_resume(multi_stack_target)
    (multi_stack_target / REVIEW_OUTPUT_FILE).write_text(
        "# Review\n\n## Issues\n\n1. [api.py:1] primed issue\n   rationale\n"
    )
    (deep / "merged-items.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": 1,
                        "lens": "per-stack",
                        "file": "api.py",
                        "line": 1,
                        "severity": "medium",
                        "description": "primed issue",
                        "confidence": "MEDIUM",
                        "rationale": "rationale",
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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """The deep orchestrator resolves a backend for each required phase."""
    from daydream import runner as _runner

    seen_phases: list[str] = []
    original = _runner._resolve_backend

    def spy(
        config: Any,
        phase: Any,
        cache: Any = None,
        *,
        cwd: Any = None,
        audit_workspace: Any = None,
    ) -> Any:
        seen_phases.append(phase)
        return original(
            config,
            phase,
            cache,
            cwd=cwd,
            audit_workspace=audit_workspace,
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

    monkeypatch.setattr("daydream.phases.phase_fix", _stub_fix)

    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # Issue #745: the pre-merge parse-<stack> stage was removed (reviewers emit
    # records directly), so "parse" is no longer a resolved deep phase.
    expected_phases = {"intent", "per_stack_review", "merge", "fix", "test", "verify"}
    captured = set(seen_phases)
    missing = expected_phases - captured
    assert not missing, f"Deep orchestrator missing per-phase resolver calls for {missing}; got {sorted(captured)}"
    assert "wonder" not in captured  # The default design lens shares per_stack_review.


def test_intent_phase_resolves_to_sonnet_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: the ``intent`` phase resolves to ``claude-sonnet-5`` by default."""
    from daydream.runner import RunConfig, _resolve_backend

    captured: dict[str, Any] = {}

    class _B:
        def __init__(self, model: str | None) -> None:
            self.model = model

    def fake_create(name: str, model: str | None = None, **kwargs: object) -> _B:  # noqa: ARG001
        captured["model"] = model
        return _B(model)

    monkeypatch.setattr("daydream.runner.create_backend", fake_create)

    # Default: intent lands on Sonnet (mid tier), not Opus.
    backend = _resolve_backend(RunConfig(), "intent", {})
    assert backend.model == "claude-sonnet-5", f"intent phase default should be claude-sonnet-5, got {backend.model!r}"

    # An explicit global model override still wins over the phase default.
    backend_override = _resolve_backend(RunConfig(model="claude-opus-5"), "intent", {})
    assert backend_override.model == "claude-opus-5", (
        f"RunConfig(model=...) override should win for intent, got {backend_override.model!r}"
    )


async def test_intent_phase_runs_on_sonnet_through_runner_run(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> None:
    """Recommendation verifier runs as a sub-step of the fix gate."""
    from daydream.deep.artifacts import verdicts_path

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
    expected_path = verdicts_path(multi_stack_target / ".daydream" / "deep")
    assert expected_path == multi_stack_target / ".daydream" / "deep" / "recommendation-verdicts.json"
    assert expected_path.is_file(), f"verdicts file missing at {expected_path}"

    import json as _json

    payload = _json.loads(expected_path.read_text())
    assert payload == {
        "verdicts": [
            {
                "issue_id": 1,
                "verdict": "consistent",
                "evidence": "stub",
                "unverified_assumptions": [],
            }
        ]
    }
