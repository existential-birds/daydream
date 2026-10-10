"""Tests for exploration subagent prompts and the pre-scan orchestrator."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import anyio
import pytest

import daydream.exploration_runner as er
from daydream.backends import AgentEvent, Backend, ResultEvent, TextEvent
from daydream.exploration import ExplorationContext, FileInfo
from daydream.exploration_runner import (
    pre_scan,
    repo_scan,
)
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES
from daydream.prompts.exploration_subagents import (
    DEPENDENCY_TRACER_SCHEMA,
    PATTERN_SCANNER_SCHEMA,
    TEST_MAPPER_SCHEMA,
    build_dependency_tracer_prompt,
    build_pattern_scanner_prompt,
)
from daydream.prompts.grounding import (
    CWD_GROUNDING_INSTRUCTION,
    UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY,
)
from tests.harness.backend import Responder, ScriptedBackend
from tests.harness.fake_clock import FakeClock
from tests.harness.review_profile import default_strategy as _default_strategy, exploration_strategies
from tests.harness.trajectory import (
    dispatch_descriptors as _ref_descriptors,
    dispatch_encloses_children as _dispatch_encloses_children,
    make_recorder,
    read_trajectory,
)

FIXTURES = Path(__file__).parent / "fixtures" / "diffs"

# Shelfspace diff paths reused by the mapper-targeting and static-prescan tests.
SHELFSPACE_PATHS = [".github/workflows/daydream.yml", "Makefile", "docs/README.md", "docs/daydream-review.md",
    "frontend/app/components/modals/__tests__/CreateShelfModal.test.tsx",
    "frontend/app/components/shelf/ShelfNameInput.tsx",
    "frontend/tests/components/taste-reveal/RevealFlowContainer.test.tsx",
    "quality-workspaces.json", "scripts/daydream-workflow.test.mjs", "scripts/github-actions-workflow-policy.mjs",
]


def _dispatch_steps(trajectory: dict[str, Any], *, phase: str) -> list[dict[str, Any]]:
    """Return identified deterministic dispatch steps for one phase."""
    return [step
        for step in trajectory["steps"]
        if step.get("llm_call_count") == 0
        and step.get("extra", {}).get("daydream_phase") == phase
        and "dispatch_id" in step.get("extra", {})
    ]


# Subagent prompt sanity checks (Plan 03)
def test_pattern_scanner_prompt_includes_guideline_files() -> None:
    files = [FileInfo("daydream/foo.py", "modified")]
    cwd = Path("/repo")
    dynamic = build_pattern_scanner_prompt(
        files, "main...HEAD", cwd=cwd, strategy=_default_strategy("exploration.pattern_scan")
    )
    assert "CLAUDE.md" in dynamic
    assert "main...HEAD" in dynamic
    assert "daydream/foo.py" in dynamic
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=cwd) in dynamic
    # Regression guard: raw diff text must never be embedded.
    assert "<diff>" not in dynamic
    # git diff must be Bash-conditional -- read_only backends (Pi) have no Bash.
    assert "If a Bash tool is available, you may also run `git diff" in dynamic

# Orchestrator helpers and mock backend
_VALID_ENVELOPE: dict[str, Any] = {"pattern_scanner": {
        "conventions": [{"name": "snake_case", "description": "snake_case for funcs", "source": "CLAUDE.md"}],
        "guidelines": ["use type hints"],
    }, "dependency_tracer": {"affected_files": [{"path": "daydream/extra.py", "role": "imports", "summary": "helper"}],
        "dependencies": [{"source": "daydream/a.py", "target": "daydream/extra.py", "relationship": "imports"}],
    },
    "test_mapper": {"affected_files": [
            {"path": "tests/test_a.py", "role": "test", "summary": "covers a.py", "source_file": "daydream/a.py"},
        ],
    },
}


def _specialist_backend(results: dict[str, dict[str, Any]] | None = None, *, responder: Responder | None = None,
) -> ScriptedBackend:
    """Schema-keyed specialist backend: each output_schema gets its envelope slice.

    Mirrors the old per-schema branches: a schema absent from the envelope gets
    an empty payload, and an unmatched schema falls back to the whole envelope.
    An optional ``responder`` hook runs before schema selection (so it can fail
    one specialist or block it without changing the rest).
    """
    payload = results or _VALID_ENVELOPE
    return ScriptedBackend(responses_by_schema=[(PATTERN_SCANNER_SCHEMA,
                [ResultEvent(structured_output=payload.get("pattern_scanner", {}), continuation=None)],
            ), (DEPENDENCY_TRACER_SCHEMA,
                [ResultEvent(structured_output=payload.get("dependency_tracer", {}), continuation=None)],
            ),
            (TEST_MAPPER_SCHEMA, [ResultEvent(structured_output=payload.get("test_mapper", {}), continuation=None)],),
            (None, [ResultEvent(structured_output=payload, continuation=None)]),
        ], responder=responder,
    )


async def specialist_pre_scan(*args: Any, **kwargs: Any) -> ExplorationContext:
    """Exercise opted-in specialist behavior rather than the default static shortcut."""
    strategies = exploration_strategies()
    strategies["exploration.dependency_trace"] += "\nCustom specialist dispatch requested."
    kwargs.setdefault("strategies", strategies)
    return await pre_scan(*args, **kwargs)


# Pure helpers
def _py_ts_diff() -> str:
    """Python + TypeScript multifile fixtures: 4 files, so the parallel tier runs."""
    return ((FIXTURES / "python_multifile.diff").read_text()
            + (FIXTURES / "typescript_multifile.diff").read_text())


# Orchestrator tier dispatch
def test_skip_tier_no_subagents(tmp_path: Path) -> None:
    diff_text = (FIXTURES / "trivial_single.diff").read_text()
    backend = _specialist_backend()
    ctx = anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff_text))

    assert backend.calls == []
    assert ctx is not None

def test_single_tier_dependency_tracer_only(tmp_path: Path) -> None:
    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    backend = _specialist_backend()
    ctx = anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff_text))

    assert backend.call_count == 1
    assert backend.calls[0]["output_schema"] == DEPENDENCY_TRACER_SCHEMA
    assert all(call["read_only"] is True for call in backend.calls)
    paths = {f.path for f in ctx.affected_files}
    assert "daydream/extra.py" in paths

@pytest.mark.parametrize(("investigation_s", "finalization_s", "completed", "has_dependencies", "calls"),
    [(180, 0, True, True, 1), (301, 60, False, True, 2), (301, 121, False, False, 2)],
)
async def test_pre_scan_retains_slow_dependency_mapping_with_bounded_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, investigation_s: float, finalization_s: float, completed: bool,
    has_dependencies: bool, calls: int,
) -> None:
    """Pi can need over two minutes to investigate and over 30s to finalize."""
    clock = FakeClock().install(monkeypatch)
    async def responder(*args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        finalizing = "INVESTIGATION HAS ENDED" in args[1]
        clock.advance(finalization_s if finalizing else investigation_s)
        yield ResultEvent(structured_output=_VALID_ENVELOPE["dependency_tracer"], continuation=None)
    backend = _specialist_backend(responder=responder)
    context = await specialist_pre_scan(
        cast(Backend, backend), tmp_path, (FIXTURES / "python_multifile.diff").read_text(),
    )

    assert context.completed is completed
    assert bool(context.dependencies) is has_dependencies
    assert backend.call_count == calls

def test_specialist_rows_carry_llm_provenance(tmp_path: Path) -> None:
    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    backend = _specialist_backend(results=_VALID_ENVELOPE)
    ctx = anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff_text))
    by_path = {f.path: f for f in ctx.affected_files}
    assert by_path["daydream/extra.py"].provenance == "llm"

@pytest.mark.parametrize("oversized", [False, True])
def test_pre_scan_supplies_small_diff_without_requiring_bash(tmp_path: Path, oversized: bool) -> None:

    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    if oversized:
        diff_text += "+" + "x" * INLINE_DIFF_BUDGET_BYTES
    backend = _specialist_backend()
    anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff_text))
    prompt = backend.calls[0]["prompt"]
    if oversized:
        assert "<change_diff>" not in prompt
        assert diff_text not in prompt
        assert "<change_overview>" in prompt
        overview = prompt.split("<change_overview>")[1].split("</change_overview>")[0]
        assert len(overview.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES
        assert "omitted" in overview
    else:
        assert f"<change_diff>\n{diff_text}\n</change_diff>" in prompt
        assert "git diff" not in prompt

def test_large_diff_overview_preserves_small_edits_after_large_additions() -> None:

    large = "diff --git a/workflow.yml b/workflow.yml\n--- /dev/null\n+++ b/workflow.yml\n@@ -0,0 +1,500 @@\n"
    large += "+" + "workflow step " * 30 + "\n"
    large += "+" + "x" * 32_000 + "\n"
    small = (
        "diff --git a/component.tsx b/component.tsx\n--- a/component.tsx\n+++ b/component.tsx\n"
        "@@ -20,2 +20,3 @@\n <input\n+aria-label={label}\n />\n"
    )
    prompt = build_dependency_tracer_prompt(
        [FileInfo("workflow.yml", "modified"), FileInfo("component.tsx", "modified")],
        "main...HEAD", cwd=Path("/repo"), strategy="custom-strategy", inline_diff=large + small,
    )
    assert "<change_overview>" in prompt
    overview = prompt.split("<change_overview>")[1].split("</change_overview>")[0]
    assert len(overview.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES
    assert "+++ b/workflow.yml" in overview and "+++ b/component.tsx" in overview
    assert "@@ -20,2 +20,3 @@" in overview and "+aria-label={label}" in overview
    assert "omitted" in overview
    assert "not a complete patch or a review decision" in overview

def test_change_overview_byte_limit_handles_many_long_unicode_paths() -> None:

    diff = "".join(
        f"diff --git a/{'界' * 100}/{i}.py b/{'界' * 100}/{i}.py\n"
        f"--- a/{'界' * 100}/{i}.py\n+++ b/{'界' * 100}/{i}.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
        for i in range(100)
    )
    prompt = build_dependency_tracer_prompt([], "main...HEAD", cwd=Path("/repo"), strategy="", inline_diff=diff)
    overview = prompt[prompt.index("<change_overview>"):prompt.index("</change_overview>") + len("</change_overview>")]
    assert len(overview.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES
    assert "file excerpts omitted" in overview
    assert "@@" not in overview  # No dangling hunks without their full file headers.

def test_pre_scan_skips_mapper_when_no_changed_source_targets(tmp_path: Path) -> None:
    diff = _multifile_diff(["docs/a.md", "package.json", "Makefile", "tests/foo.test.ts"])
    backend = _specialist_backend()
    anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff))
    assert backend.call_count == 2
    assert {json.dumps(call["output_schema"], sort_keys=True) for call in backend.calls} == {
        json.dumps(PATTERN_SCANNER_SCHEMA, sort_keys=True), json.dumps(DEPENDENCY_TRACER_SCHEMA, sort_keys=True),
    }

def test_modest_default_prescan_is_static_and_captures_guidance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = SHELFSPACE_PATHS
    memberships = [FileInfo(path, "modified") for path in paths] + [FileInfo("frontend/constants.ts", "imports")]
    monkeypatch.setattr("daydream.exploration_runner.detect_affected_files", lambda *_: memberships)
    (tmp_path / "AGENTS.md").write_text("Guideline evidence: keep shared helpers canonical.")
    backend = _specialist_backend()
    context = anyio.run(lambda: pre_scan(cast(Backend, backend), tmp_path, _multifile_diff(paths)))
    assert backend.calls == []
    assert context.affected_files == memberships
    assert context.dependencies == []  # Membership does not establish a source-target edge.
    assert context.completed is True
    context.write_to_dir(tmp_path / "exploration")
    summary = (tmp_path / "exploration/summary.md").read_text()
    assert "Guideline evidence: keep shared helpers canonical." in summary
    assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in summary
    assert "not correctness review or coverage evidence" in summary

def test_custom_mapper_runs_without_default_source_targets(tmp_path: Path) -> None:
    strategies = exploration_strategies()
    strategies["exploration.test_mapping"] = "custom mapping of config contract tests"
    backend = _specialist_backend()
    diff = _multifile_diff(["a.json", "b.json", "c.json", "d.json"])
    anyio.run(lambda: pre_scan(cast(Backend, backend), tmp_path, diff, strategies=strategies))
    assert backend.call_count == 3
    mapper = next(call["prompt"] for call in backend.calls if call["output_schema"] == TEST_MAPPER_SCHEMA)
    targets = mapper.split("<affected_files>\n")[1].split("</affected_files>")[0]
    assert all(path in targets for path in ("a.json", "b.json", "c.json", "d.json"))
    assert "custom mapping of config contract tests" in mapper
    assert "Do not hunt for tests of documentation or manifests" not in mapper

def test_static_guidance_is_bounded_and_confined_to_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("界" * 10_000)
    outside = tmp_path / "outside.md"
    outside.write_text("outside guidance must not be captured")
    (repo / "CLAUDE.md").symlink_to(outside)
    (repo / ".editorconfig").write_bytes(b"root=true\n\xff")
    (repo / "src").mkdir()
    (repo / "src/AGENTS.md").write_text("nested guideline evidence")
    backend = _specialist_backend()
    context = anyio.run(lambda: pre_scan(cast(Backend, backend), repo, _multifile_diff(["src/a.py", "src/b.py"])))
    assert backend.calls == []
    assert "outside guidance" not in context.raw_notes
    assert "nested guideline evidence" in context.raw_notes
    assert "truncated" in context.raw_notes
    assert len(context.raw_notes.encode("utf-8")) <= 8192

def test_static_guidance_includes_ancestors_of_non_source_changes(tmp_path: Path) -> None:
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / ".github/AGENTS.md").write_text("Workflow-specific convention evidence")
    backend = _specialist_backend()
    context = anyio.run(lambda: pre_scan(
        cast(Backend, backend), tmp_path, _multifile_diff([".github/workflows/check.yml", "config.json"]),
    ))
    assert backend.calls == []
    assert "Workflow-specific convention evidence" in context.raw_notes

def test_parse_envelope_handles_missing_keys(tmp_path: Path) -> None:
    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    envelope = {"dependency_tracer": {
            "affected_files": [{"path": "daydream/x.py", "role": "imports", "summary": ""}], "dependencies": [],
        }
    }
    backend = _specialist_backend(results=envelope)
    ctx = anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff_text))
    assert any(f.path == "daydream/x.py" for f in ctx.affected_files)
    assert ctx.conventions == []

def test_specialist_failure_doesnt_cancel_others(tmp_path: Path) -> None:
    diff_text = _py_ts_diff()
    def responder(*args: Any, **kwargs: Any) -> Any:
        if args[2] == PATTERN_SCANNER_SCHEMA:
            raise RuntimeError("pattern scanner exploded")
        return None
    backend = _specialist_backend(responder=responder)
    ctx = anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff_text))

    # Pattern scanner failed, but others should have run
    assert backend.call_count == 3
    assert not ctx.completed
    assert ctx.conventions == []  # pattern scanner failed
    assert any(f.path == "daydream/extra.py" for f in ctx.affected_files)


def _multifile_diff(paths: list[str]) -> str:
    return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-old\n+new\n" for p in paths)

async def test_pre_scan_dispatch_interval_timeout_dispatch_keeps_completed_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-scan timeout is terminal evidence and retains only completed refs."""

    async def _never_yield() -> AsyncIterator[AgentEvent]:
        await anyio.sleep_forever()
        yield ResultEvent(structured_output=None, continuation=None)

    def responder(*args: Any, **kwargs: Any) -> Any:
        if args[2] != DEPENDENCY_TRACER_SCHEMA:
            return _never_yield()
        return None

    monkeypatch.setattr(er, "_PRE_SCAN_TIMEOUT_SECONDS", 0.05)
    diff_text = _multifile_diff([f"src/file_{index}.py" for index in range(4)])
    recorder = make_recorder(tmp_path)

    async with recorder:
        context = await specialist_pre_scan(
            cast(Backend, _specialist_backend(responder=responder)), tmp_path, diff_text,
        )

    assert context.completed is False
    assert context.dependencies
    step = _dispatch_steps(read_trajectory(recorder.path), phase="exploration")[0]
    assert _ref_descriptors(step) == ["explore-pattern_scanner", "explore-dependency_tracer", "explore-test_mapper"]
    assert _dispatch_encloses_children(step, recorder.target_dir)
    assert step["extra"]["dispatch_status"] == "timed_out"
    assert step["extra"]["reason_code"] == "timed_out"
    assert step["extra"]["planned_count"] == 3
    assert step["extra"]["attempted_count"] == 3
    # Cancelled specialists still write their bounded child evidence; semantic
    # completion is represented by the dispatch's timed-out terminal.
    assert step["extra"]["completed_count"] == 3

async def test_repo_scan_dispatch_records_survey_failure(tmp_path: Path) -> None:
    backend = ScriptedBackend(events=[TextEvent(text="Starting repository survey"), RuntimeError("survey failed")])
    recorder = make_recorder(tmp_path)
    async with recorder:
        context = await repo_scan(cast(Backend, backend), tmp_path,)

    assert context.conventions == []
    step = _dispatch_steps(read_trajectory(recorder.path), phase="exploration")[0]
    # run_agent records its handled backend failure in a durable child document.
    assert _ref_descriptors(step) == ["explore-repo_survey"]
    assert _dispatch_encloses_children(step, recorder.target_dir)
    assert step["extra"]["dispatch_status"] == "failed"
    assert step["extra"]["reason_code"] == "all_children_failed"
    assert step["extra"]["planned_count"] == 1
    assert step["extra"]["attempted_count"] == 1
    assert step["extra"]["completed_count"] == 1

def test_pre_scan_fallback_uses_rename_new_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the fallback path with a genuine static-analysis failure.
    def analyzer_failure(diff_text: str, repo_root: Path) -> list[FileInfo]:
        raise OSError("tree-sitter source unavailable")
    monkeypatch.setattr(er, "detect_affected_files", analyzer_failure)
    # A rename plus a second file => 2 files => single tier => dependency_tracer runs.
    rename_diff = (
        "diff --git a/services/taste/old_name.py b/services/taste/new_name.py\n"
        "similarity index 95%\n"
        "rename from services/taste/old_name.py\n"
        "rename to services/taste/new_name.py\n"
        "--- a/services/taste/old_name.py\n"
        "+++ b/services/taste/new_name.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    diff_text = rename_diff + _multifile_diff(["services/taste/other.py"])
    backend = _specialist_backend()
    anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff_text))
    dep_prompt = backend.calls[0]["prompt"]
    targets = dep_prompt.split("<affected_files>")[1].split("</affected_files>")[0]
    assert str(tmp_path / "services/taste/new_name.py") in targets
    assert "old_name.py" not in targets

def test_test_mapper_source_file_flows_through_pre_into_test_map_json(tmp_path: Path) -> None:
    diff_text = _py_ts_diff()
    backend = _specialist_backend(results=_VALID_ENVELOPE)
    ctx = anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff_text))
    exploration_dir = tmp_path / "exploration"
    ctx.write_to_dir(exploration_dir)
    data = json.loads((exploration_dir / "test-map.json").read_text())
    assert {"test_file": "tests/test_a.py", "source_file": "daydream/a.py"} in data["test_mapping"]
