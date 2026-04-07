"""Tests for exploration subagent prompts and (future) orchestrator runner.

The orchestrator (daydream.exploration_runner) is implemented in Plan 04.
Tests for it use pytest.importorskip inside each test body.
"""

import pytest


def test_pattern_scanner_prompt_includes_guideline_files():
    pytest.importorskip("daydream.prompts.exploration_subagents")
    from daydream.prompts.exploration_subagents import (
        EXPLORATION_AGENTS,
        PATTERN_SCANNER_SYSTEM_PROMPT,
        build_pattern_scanner_prompt,
    )

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
    pytest.importorskip("daydream.prompts.exploration_subagents")
    from daydream.exploration import FileInfo
    from daydream.prompts.exploration_subagents import build_dependency_tracer_prompt

    files = [FileInfo("daydream/a.py", "modified"), FileInfo("daydream/b.py", "modified")]
    prompt = build_dependency_tracer_prompt("some diff", files)
    assert "daydream/a.py" in prompt
    assert "daydream/b.py" in prompt
    assert "some diff" in prompt
    assert "```json" in prompt


def test_test_mapper_prompt_instructs_mapping():
    pytest.importorskip("daydream.prompts.exploration_subagents")
    from daydream.prompts.exploration_subagents import build_test_mapper_prompt

    prompt = build_test_mapper_prompt("diff", ["daydream/x.py"])
    assert "test" in prompt.lower()
    assert "daydream/x.py" in prompt
    assert "```json" in prompt


def test_schemas_are_valid_objects():
    pytest.importorskip("daydream.prompts.exploration_subagents")
    from daydream.prompts.exploration_subagents import (
        DEPENDENCY_TRACER_SCHEMA,
        PATTERN_SCANNER_SCHEMA,
        TEST_MAPPER_SCHEMA,
    )

    for schema in (PATTERN_SCANNER_SCHEMA, DEPENDENCY_TRACER_SCHEMA, TEST_MAPPER_SCHEMA):
        assert schema["type"] == "object"
        assert "required" in schema
        assert isinstance(schema["required"], list)
