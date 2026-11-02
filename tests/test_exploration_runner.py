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
from tests.conftest import _feature_repo
from tests.harness.backend import Responder, ScriptedBackend
from tests.harness.git_helpers import git as _git
from tests.harness.review_profile import default_strategy as _default_strategy, exploration_strategies
from tests.harness.trajectory import (
    dispatch_descriptors as _ref_descriptors,
    dispatch_encloses_children as _dispatch_encloses_children,
    make_recorder,
    read_trajectory,
)

FIXTURES = Path(__file__).parent / "fixtures" / "diffs"

def _dispatch_steps(trajectory: dict[str, Any], *, phase: str) -> list[dict[str, Any]]:
    """Return identified deterministic dispatch steps for one phase."""
    return [step
        for step in trajectory["steps"]
        if step.get("llm_call_count") == 0
        and step.get("extra", {}).get("daydream_phase") == phase
        and "dispatch_id" in step.get("extra", {})
    ]


# Subagent prompt sanity checks (Plan 03)
def test_specialist_prompts_name_guidelines_and_bound_the_change_overview() -> None:
    files = [FileInfo("daydream/foo.py", "modified")]
    cwd = Path("/repo")
    dynamic = build_pattern_scanner_prompt(
        files, "main...HEAD", cwd=cwd, strategy=_default_strategy("exploration.pattern_scan")
    )
    assert "AGENTS.md" in dynamic
    assert "CLAUDE.md" in dynamic
    assert "main...HEAD" in dynamic
    assert "daydream/foo.py" in dynamic
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=cwd) in dynamic
    # Regression guard: raw diff text must never be embedded.
    assert "<diff>" not in dynamic
    # git diff must be Bash-conditional -- read_only backends (Pi) have no Bash.
    assert "If a Bash tool is available, you may also run `git diff" in dynamic

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

# Orchestrator helpers and mock backend
_VALID_ENVELOPE: dict[str, Any] = {"pattern_scanner": {
        "conventions": [{"name": "snake_case", "description": "snake_case for funcs", "source": "AGENTS.md"}],
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
def test_skip_tier_runs_no_subagents_and_one_specialist_failure_cancels_no_sibling(tmp_path: Path) -> None:
    diff_text = (FIXTURES / "trivial_single.diff").read_text()
    backend = _specialist_backend()
    ctx = anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, diff_text))

    assert backend.calls == []
    assert ctx is not None

    def responder(*args: Any, **kwargs: Any) -> Any:
        if args[2] == PATTERN_SCANNER_SCHEMA:
            raise RuntimeError("pattern scanner exploded")
        return None
    backend = _specialist_backend(responder=responder)
    ctx = anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), tmp_path, _py_ts_diff()))

    # Pattern scanner failed, but others should have run
    assert backend.call_count == 3
    assert not ctx.completed
    assert ctx.conventions == []  # pattern scanner failed
    assert any(f.path == "daydream/extra.py" for f in ctx.affected_files)
    # A malformed specialist section contributes nothing rather than failing its siblings.
    malformed = er._parse_envelope({**_VALID_ENVELOPE, "dependency_tracer": "not an object"})
    assert malformed.dependencies == [] and malformed.conventions

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
    (repo / ".github/workflows").mkdir(parents=True)
    (repo / ".github/AGENTS.md").write_text("Workflow-specific convention evidence")
    backend = _specialist_backend()
    context = anyio.run(lambda: pre_scan(cast(Backend, backend), repo,
        _multifile_diff(["src/a.py", "src/b.py", ".github/workflows/check.yml"])))
    assert backend.calls == []
    assert "outside guidance" not in context.raw_notes
    assert "nested guideline evidence" in context.raw_notes
    assert "Workflow-specific convention evidence" in context.raw_notes
    assert "truncated" in context.raw_notes
    assert len(context.raw_notes.encode("utf-8")) <= 8192


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


def test_native_repository_exploration_persists_static_and_specialist_facts(tmp_path: Path) -> None:
    targets = {
        "package/__init__.py": "", "package/feature/__init__.py": "",
        "package/feature/nested/__init__.py": "", "package/feature/nested/deep/__init__.py": "",
        "package/parentonly.py": "class User: pass\n", "package/grandparent.py": "class User: pass\n",
        "package/ancestor.py": "thing = 1\n", "package/absolute.py": "VALUE = 1\n",
        "package/feature/local.py": "class User: pass\n", "package/feature/something.py": "thing = 2\n",
        "package/services/__init__.py": "", "package/extra.py": "EXTRA = 1\n",
        "tests/test_relative_api.py": "def test_api(): pass\n",
    }
    sources = {
        "package/direct.py": "import package.absolute\n",
        "package/feature/simple.py": "from .local import User\n",
        "package/feature/api.py": "from ..parentonly import User\n",
        "package/feature/nested/api.py": "from ...grandparent import User\n",
        "package/feature/nested/deep/api.py": "from ....ancestor import thing\n",
        "package/feature/siblings.py": "from . import something\n",
        "package/feature/parent.py": "from .. import services\n",
        "package/parentonly.py": "class User: pass\n",
        "web/api.ts": 'import { User } from "./models"; export function getUser(): User { return new User(); }\n',
        "web/models.ts": "export class User {}\n",
        "src/api.rs": "use crate::models::User; pub fn get_user() -> User { User }\n",
        "src/models.rs": "pub struct User;\n",
    }
    repo = _feature_repo(
        tmp_path, "exploration-population", {**targets, **dict.fromkeys(sources, "before = 0\n")}, sources,
    )
    backend = _specialist_backend(results={
        "pattern_scanner": _VALID_ENVELOPE["pattern_scanner"],
        "dependency_tracer": {
            "affected_files": [
                {"path": "web/models.ts", "role": "imports", "summary": "longer model detail"},
                {"path": "package/extra.py", "role": "imports", "summary": "indirect helper"},
            ],
            "dependencies": [{"source": "web/api.ts", "target": "web/models.ts", "relationship": "imports"}],
        },
        "test_mapper": {"affected_files": [{"path": "tests/test_relative_api.py", "role": "test",
            "summary": "covers relative api", "source_file": "package/feature/api.py"}]},
    })
    context = anyio.run(lambda: specialist_pre_scan(cast(Backend, backend), repo, _git(repo, "diff", "main")))
    rows = {(row.path, row.role): row for row in context.affected_files}
    assert {row.path for row in context.affected_files if row.role == "modified"} == set(sources)
    assert {row.path for row in context.affected_files if row.role == "imports"} == {
        "package/__init__.py", "package/feature/__init__.py", "package/absolute.py",
        "package/feature/local.py", "package/parentonly.py", "package/grandparent.py",
        "package/ancestor.py", "package/feature/something.py", "package/services/__init__.py",
        "web/models.ts", "src/models.rs", "package/extra.py",
    }
    assert rows[("web/models.ts", "imports")].summary == "longer model detail"
    assert rows[("web/models.ts", "imports")].provenance == "static"
    assert rows[("package/extra.py", "imports")].provenance == "llm"
    assert rows[("tests/test_relative_api.py", "test")].provenance == "llm"
    assert all(row.provenance == "static" for row in context.affected_files
        if (row.path, row.role) not in {("package/extra.py", "imports"), ("tests/test_relative_api.py", "test")})
    assert context.completed is True
    assert backend.call_count == 3
    assert all(call["read_only"] is True for call in backend.calls)
    exploration = context.write_to_dir(tmp_path / "exploration")
    assert json.loads((exploration / "test-map.json").read_text()) == {"test_mapping": [
        {"test_file": "tests/test_relative_api.py", "source_file": "package/feature/api.py"}]}
    artifact_names = ("summary.md", "affected_files.md", "conventions.md", "dependencies.md")
    artifacts = {name: (exploration / name).read_text() for name in artifact_names}
    assert "affected_files.md" in artifacts["summary.md"]
    assert "| `web/models.ts` | imports | static | longer model detail |" in artifacts["affected_files.md"]
    assert "| modified | static |" in artifacts["affected_files.md"]
    assert "snake_case" in artifacts["conventions.md"] and "use type hints" in artifacts["conventions.md"]
    assert "web/api.ts" in artifacts["dependencies.md"] and "web/models.ts" in artifacts["dependencies.md"]
    assert "imports" in artifacts["dependencies.md"]
    assert all(UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in content for content in artifacts.values())
    assert (artifacts["affected_files.md"].index(UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY)
        < artifacts["affected_files.md"].index("web/models.ts"))
