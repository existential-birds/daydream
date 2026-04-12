"""Tests for exploration subagent prompts and the pre-scan orchestrator."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio

from daydream.backends import AgentEvent, ResultEvent
from daydream.exploration import FileInfo
from daydream.exploration_runner import (
    EXPLORATION_ENVELOPE_SCHEMA,
    count_changed_files,
    pre_scan,
    select_tier,
)
from daydream.prompts.exploration_subagents import (
    DEPENDENCY_TRACER_SCHEMA,
    EXPLORATION_AGENTS,
    PATTERN_SCANNER_SCHEMA,
    PATTERN_SCANNER_SYSTEM_PROMPT,
    TEST_MAPPER_SCHEMA,
    build_dependency_tracer_prompt,
    build_pattern_scanner_prompt,
    build_test_mapper_prompt,
)

FIXTURES = Path(__file__).parent / "fixtures" / "diffs"


# ---------------------------------------------------------------------------
# Subagent prompt sanity checks (Plan 03)
# ---------------------------------------------------------------------------


def test_pattern_scanner_prompt_includes_guideline_files():
    assert "CLAUDE.md" in PATTERN_SCANNER_SYSTEM_PROMPT
    assert ".coderabbit.yaml" in PATTERN_SCANNER_SYSTEM_PROMPT

    dynamic = build_pattern_scanner_prompt("diff text", ["daydream/foo.py"])
    assert "CLAUDE.md" in dynamic
    assert ".coderabbit.yaml" in dynamic
    assert "diff text" in dynamic
    assert "daydream/foo.py" in dynamic

    assert set(EXPLORATION_AGENTS.keys()) == {"pattern-scanner", "dependency-tracer", "test-mapper"}
    assert EXPLORATION_AGENTS["pattern-scanner"].model == "inherit"
    assert "Read" in EXPLORATION_AGENTS["pattern-scanner"].tools
    assert "Glob" in EXPLORATION_AGENTS["pattern-scanner"].tools
    assert "Grep" in EXPLORATION_AGENTS["pattern-scanner"].tools


def test_dependency_tracer_prompt_mentions_affected_files():
    files = [FileInfo("daydream/a.py", "modified"), FileInfo("daydream/b.py", "modified")]
    prompt = build_dependency_tracer_prompt("some diff", files)
    assert "daydream/a.py" in prompt
    assert "daydream/b.py" in prompt
    assert "some diff" in prompt
    assert "```json" in prompt


def test_test_mapper_prompt_instructs_mapping():
    prompt = build_test_mapper_prompt("diff", ["daydream/x.py"])
    assert "test" in prompt.lower()
    assert "daydream/x.py" in prompt
    assert "```json" in prompt


def test_schemas_are_valid_objects():
    for schema in (PATTERN_SCANNER_SCHEMA, DEPENDENCY_TRACER_SCHEMA, TEST_MAPPER_SCHEMA):
        assert schema["type"] == "object"
        assert "required" in schema
        assert isinstance(schema["required"], list)


# ---------------------------------------------------------------------------
# Orchestrator helpers and mock backend
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


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


def test_envelope_schema_includes_three_subagent_keys():
    props = EXPLORATION_ENVELOPE_SCHEMA["properties"]
    assert set(props.keys()) == {"pattern_scanner", "dependency_tracer", "test_mapper"}
    assert EXPLORATION_ENVELOPE_SCHEMA["type"] == "object"


# ---------------------------------------------------------------------------
# Orchestrator tier dispatch
# ---------------------------------------------------------------------------


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
    # Verify no agents= was passed
    for call in backend.execute_calls:
        assert call["agents"] is None
    # Results merged correctly
    assert any(c.name == "snake_case" for c in ctx.conventions)
    assert any(f.path == "tests/test_a.py" for f in ctx.affected_files)
    assert "use type hints" in ctx.guidelines


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
        async def execute(self, cwd, prompt, output_schema=None, continuation=None, agents=None):
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
