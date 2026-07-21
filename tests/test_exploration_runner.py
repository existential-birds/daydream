"""Tests for exploration subagent prompts and the pre-scan orchestrator."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio

from daydream.backends import AgentEvent, ResultEvent
from daydream.exploration import FileInfo
from daydream.exploration_runner import (
    count_changed_files,
    pre_scan,
    select_tier,
)
from daydream.prompts.exploration_subagents import (
    DEPENDENCY_TRACER_SCHEMA,
    PATTERN_SCANNER_SCHEMA,
    TEST_MAPPER_SCHEMA,
    build_dependency_tracer_prompt,
    build_pattern_scanner_prompt,
    build_test_mapper_prompt,
)
from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION

FIXTURES = Path(__file__).parent / "fixtures" / "diffs"


# Subagent prompt sanity checks (Plan 03)
def test_pattern_scanner_prompt_includes_guideline_files():
    files = [FileInfo("daydream/foo.py", "modified")]
    dynamic = build_pattern_scanner_prompt(files, "main...HEAD", cwd=Path("/repo"))
    assert "CLAUDE.md" in dynamic
    assert "main...HEAD" in dynamic
    assert "daydream/foo.py" in dynamic
    # Regression guard: raw diff text must never be embedded.
    assert "<diff>" not in dynamic


def test_dependency_tracer_prompt_mentions_affected_files():
    files = [FileInfo("daydream/a.py", "modified"), FileInfo("daydream/b.py", "modified")]
    prompt = build_dependency_tracer_prompt(files, "main...HEAD", cwd=Path("/repo"))
    assert "daydream/a.py" in prompt
    assert "daydream/b.py" in prompt
    assert "main...HEAD" in prompt
    assert "```json" in prompt
    assert "<diff>" not in prompt


def test_test_mapper_prompt_instructs_mapping():
    files = [FileInfo("daydream/x.py", "modified")]
    prompt = build_test_mapper_prompt(files, "main...HEAD", cwd=Path("/repo"))
    assert "test" in prompt.lower()
    assert "daydream/x.py" in prompt
    assert "main...HEAD" in prompt
    assert "```json" in prompt
    assert "<diff>" not in prompt


def test_pattern_scanner_prompt_contains_cwd_grounding():
    cwd = Path("/tmp/linked/worktree")
    files = [FileInfo("daydream/foo.py", "modified")]
    prompt = build_pattern_scanner_prompt(files, "main...HEAD", cwd=cwd)
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=cwd) in prompt
    assert str(cwd) in prompt


def test_dependency_tracer_prompt_contains_cwd_grounding():
    cwd = Path("/tmp/linked/worktree")
    files = [FileInfo("daydream/a.py", "modified")]
    prompt = build_dependency_tracer_prompt(files, "main...HEAD", cwd=cwd)
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=cwd) in prompt
    assert str(cwd) in prompt


def test_test_mapper_prompt_contains_cwd_grounding():
    cwd = Path("/tmp/linked/worktree")
    files = [FileInfo("daydream/x.py", "modified")]
    prompt = build_test_mapper_prompt(files, "main...HEAD", cwd=cwd)
    assert CWD_GROUNDING_INSTRUCTION.format(cwd=cwd) in prompt
    assert str(cwd) in prompt


def test_schemas_are_valid_objects():
    for schema in (PATTERN_SCANNER_SCHEMA, DEPENDENCY_TRACER_SCHEMA, TEST_MAPPER_SCHEMA):
        assert schema["type"] == "object"
        assert "required" in schema
        assert isinstance(schema["required"], list)


# Orchestrator helpers and mock backend
_VALID_ENVELOPE: dict[str, Any] = {
    "pattern_scanner": {
        "conventions": [
            {"name": "snake_case", "description": "snake_case for funcs", "source": "CLAUDE.md"},
        ],
        "guidelines": ["use type hints"],
    },
    "dependency_tracer": {
        "affected_files": [
            {"path": "daydream/extra.py", "role": "imports", "summary": "helper"},
        ],
        "dependencies": [
            {"source": "daydream/a.py", "target": "daydream/extra.py", "relationship": "imports"},
        ],
    },
    "test_mapper": {
        "affected_files": [
            {"path": "tests/test_a.py", "role": "test", "summary": "covers a.py"},
        ],
    },
}


class _SpecialistMockBackend:
    """Mock backend that returns specialist results based on output_schema."""

    model = "mock-model"

    def __init__(self, results: dict[str, dict] | None = None) -> None:
        self.execute_calls: list[dict[str, Any]] = []
        self._results = results or _VALID_ENVELOPE

    async def execute(  # type: ignore[no-untyped-def]
        self,
        cwd,
        prompt,
        output_schema=None,
        continuation=None,
        agents=None,
        max_turns=None,
        read_only=False,
    ) -> AsyncIterator[AgentEvent]:
        self.execute_calls.append({"prompt": prompt, "schema": output_schema, "agents": agents})
        result: dict[str, Any] = {}
        if output_schema == PATTERN_SCANNER_SCHEMA:
            result = self._results.get("pattern_scanner", {})
        elif output_schema == DEPENDENCY_TRACER_SCHEMA:
            result = self._results.get("dependency_tracer", {})
        elif output_schema == TEST_MAPPER_SCHEMA:
            result = self._results.get("test_mapper", {})
        else:
            result = self._results
        yield ResultEvent(structured_output=result, continuation=None)

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


# Backward compat alias for tests/test_integration.py
_AgentsRecordingMockBackend = _SpecialistMockBackend


# Pure helpers
def test_count_changed_files_counts_unique_paths():
    assert count_changed_files("") == 0
    one = (FIXTURES / "trivial_single.diff").read_text()
    assert count_changed_files(one) == 1
    py = (FIXTURES / "python_multifile.diff").read_text()
    assert count_changed_files(py) == 2
    combined = py + (FIXTURES / "typescript_multifile.diff").read_text()
    assert count_changed_files(combined) == 4


def test_select_tier_thresholds():
    assert select_tier(0) == "skip"
    assert select_tier(1) == "skip"
    assert select_tier(2) == "single"
    assert select_tier(3) == "single"
    assert select_tier(4) == "parallel"
    assert select_tier(99) == "parallel"


# Orchestrator tier dispatch
def test_skip_tier_no_subagents(tmp_path):
    diff_text = (FIXTURES / "trivial_single.diff").read_text()
    backend = _SpecialistMockBackend()

    ctx = anyio.run(pre_scan, backend, tmp_path, diff_text)

    assert backend.execute_calls == []
    assert ctx is not None


def test_single_tier_dependency_tracer_only(tmp_path):
    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    backend = _SpecialistMockBackend()

    ctx = anyio.run(pre_scan, backend, tmp_path, diff_text)

    assert len(backend.execute_calls) == 1
    assert backend.execute_calls[0]["schema"] == DEPENDENCY_TRACER_SCHEMA
    paths = {f.path for f in ctx.affected_files}
    assert "daydream/extra.py" in paths


def test_parallel_tier_launches_three_agents(tmp_path):
    py = (FIXTURES / "python_multifile.diff").read_text()
    ts = (FIXTURES / "typescript_multifile.diff").read_text()
    diff_text = py + ts

    backend = _SpecialistMockBackend()
    ctx = anyio.run(pre_scan, backend, tmp_path, diff_text)

    assert len(backend.execute_calls) == 3
    schemas = {call["schema"]["type"] for call in backend.execute_calls}
    assert len(schemas) >= 1  # All are "object" type
    # No agents= passed and no raw diff leaked into specialist prompts.
    for call in backend.execute_calls:
        assert call["agents"] is None
        assert "<diff>" not in call["prompt"]
    assert any(c.name == "snake_case" for c in ctx.conventions)
    assert any(f.path == "tests/test_a.py" for f in ctx.affected_files)
    assert "use type hints" in ctx.guidelines


def test_parallel_tier_gives_every_specialist_the_same_list(tmp_path):
    py = (FIXTURES / "python_multifile.diff").read_text()
    ts = (FIXTURES / "typescript_multifile.diff").read_text()
    diff_text = py + ts

    backend = _SpecialistMockBackend()
    anyio.run(pre_scan, backend, tmp_path, diff_text)

    # Paths are cwd-absolute (issue #221): rooted at repo_root (tmp_path).
    diff_changed = {
        str(tmp_path / p)
        for p in ("daydream_demo/api.py", "daydream_demo/models.py", "src/api.ts", "src/models.ts")
    }

    calls_by_schema = {}
    for call in backend.execute_calls:
        if call["schema"] == PATTERN_SCANNER_SCHEMA:
            calls_by_schema["pattern_scanner"] = call
        elif call["schema"] == DEPENDENCY_TRACER_SCHEMA:
            calls_by_schema["dependency_tracer"] = call
        elif call["schema"] == TEST_MAPPER_SCHEMA:
            calls_by_schema["test_mapper"] = call

    assert set(calls_by_schema.keys()) == {"pattern_scanner", "dependency_tracer", "test_mapper"}

    def _affected_block(prompt: str) -> str:
        start = prompt.index("<affected_files>")
        end = prompt.index("</affected_files>", start) + len("</affected_files>")
        return prompt[start:end]

    # Consolidation: every specialist receives a byte-identical affected-files
    # block -- no per-specialist input split.
    blocks = [_affected_block(calls_by_schema[name]["prompt"]) for name in calls_by_schema]
    assert blocks[0] == blocks[1] == blocks[2]

    # And the shared block lists every changed file, role-annotated and cwd-absolute.
    for path in diff_changed:
        assert f"- {path} (modified)" in blocks[0]


def test_parse_envelope_handles_missing_keys(tmp_path):
    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    envelope = {
        "dependency_tracer": {
            "affected_files": [
                {"path": "daydream/x.py", "role": "imports", "summary": ""},
            ],
            "dependencies": [],
        }
    }
    backend = _SpecialistMockBackend(results=envelope)
    ctx = anyio.run(pre_scan, backend, tmp_path, diff_text)
    assert any(f.path == "daydream/x.py" for f in ctx.affected_files)
    assert ctx.conventions == []


def test_specialist_failure_doesnt_cancel_others(tmp_path):
    """One specialist raising doesn't cancel the others."""
    py = (FIXTURES / "python_multifile.diff").read_text()
    ts = (FIXTURES / "typescript_multifile.diff").read_text()
    diff_text = py + ts  # 4 files -> parallel tier

    call_count = 0

    class _FailingPatternScanner:
        async def execute(
            self, cwd, prompt, output_schema=None, continuation=None, agents=None,
            max_turns=None, read_only=False,
        ):
            nonlocal call_count
            call_count += 1
            if output_schema == PATTERN_SCANNER_SCHEMA:
                raise RuntimeError("pattern scanner exploded")
            result = _VALID_ENVELOPE.get("dependency_tracer", {})
            if output_schema == TEST_MAPPER_SCHEMA:
                result = _VALID_ENVELOPE.get("test_mapper", {})
            yield ResultEvent(structured_output=result, continuation=None)

        async def cancel(self):
            return None

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    backend = _FailingPatternScanner()
    ctx = anyio.run(pre_scan, backend, tmp_path, diff_text)

    # Pattern scanner failed, but others should have run
    assert call_count == 3
    assert ctx.conventions == []  # pattern scanner failed
    assert any(f.path == "daydream/extra.py" for f in ctx.affected_files)


def _multifile_diff(paths: list[str]) -> str:
    return "".join(
        f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-old\n+new\n"
        for p in paths
    )


def test_pre_scan_passes_cwd_absolute_paths(tmp_path):
    # 4 files => parallel tier => all specialists receive the affected-files list.
    paths = [f"services/taste/file{i}.py" for i in range(4)]
    diff_text = _multifile_diff(paths)
    backend = _SpecialistMockBackend()

    anyio.run(pre_scan, backend, tmp_path, diff_text)

    joined = "\n".join(c["prompt"] for c in backend.execute_calls)
    # Specialists receive cwd-absolute paths, never bare relatives.
    for p in paths:
        assert str(tmp_path / p) in joined
        assert f"- {p} (" not in joined


def test_pre_scan_passes_cwd_absolute_static_files(tmp_path, monkeypatch):
    import daydream.exploration_runner as er

    monkeypatch.setattr(
        er,
        "detect_affected_files",
        lambda diff_text, repo_root, depth: [FileInfo("services/taste/dep.py", "imports", "helper")],
    )
    # 2 files => single tier => dependency_tracer gets static_files.
    diff_text = _multifile_diff(["services/taste/a.py", "services/taste/b.py"])
    backend = _SpecialistMockBackend()

    anyio.run(pre_scan, backend, tmp_path, diff_text)

    dep_prompt = backend.execute_calls[0]["prompt"]
    assert str(tmp_path / "services/taste/dep.py") in dep_prompt
    assert "- services/taste/dep.py (" not in dep_prompt


def test_pre_scan_fallback_uses_rename_new_path(tmp_path, monkeypatch):
    """Fallback seeding is rename-aware: a renamed file seeds its new path, not the old one."""
    import daydream.exploration_runner as er

    # Force the fallback path: static resolution yields nothing.
    monkeypatch.setattr(er, "detect_affected_files", lambda diff_text, repo_root, depth: [])
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
    backend = _SpecialistMockBackend()

    anyio.run(pre_scan, backend, tmp_path, diff_text)

    dep_prompt = backend.execute_calls[0]["prompt"]
    assert str(tmp_path / "services/taste/new_name.py") in dep_prompt
    assert "old_name.py" not in dep_prompt
