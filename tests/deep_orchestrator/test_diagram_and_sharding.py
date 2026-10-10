"""Sweep Diagram And Sharding."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream.config import (
    DEFAULT_DEEP_SHARD_ENABLED,
    DEFAULT_DEEP_SHARD_MAX_FILES,
)
from daydream.config_file import DaydreamFileConfig
from daydream.deep.orchestrator import (
    _deep_shard_enabled,
    _deep_shard_max_files,
)
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.stub_backend import install_stub_backend
from tests.test_deep_orchestrator import (
    _install_model_capturing_stubs,
    _run_deep,
    _silence,
)


def test_deep_shard_enabled_default_off(tmp_path: Path) -> None:

    assert DEFAULT_DEEP_SHARD_ENABLED is False
    # Default off (sharding-off): no RunConfig attr, no file config.
    cfg = RunConfig(target=str(tmp_path))
    assert _deep_shard_enabled(cfg) is False
    # Large diffs opt into the existing sharder unless a caller explicitly
    # selected a value.
    large_diff = "+line\n" * 6_500
    assert _deep_shard_enabled(cfg, diff=large_diff) is True
    assert _deep_shard_enabled(cfg, diff="+line\n" * 1_500) is False
    # RunConfig True (highest tier) enables.
    cfg = RunConfig(target=str(tmp_path), deep_shard_enabled=True)
    assert _deep_shard_enabled(cfg) is True
    # File-config True with no RunConfig override enables.

    fc = DaydreamFileConfig(deep_shard_enabled=True)
    cfg = RunConfig(target=str(tmp_path), file_config=fc)
    assert _deep_shard_enabled(cfg) is True
    # Explicit RunConfig False (highest tier) force-off a file-config-enabled repo.
    cfg = RunConfig(target=str(tmp_path), file_config=fc, deep_shard_enabled=False)
    assert _deep_shard_enabled(cfg) is False
    assert _deep_shard_enabled(cfg, diff=large_diff) is False

def test_deep_shard_max_files_resolves_and_coerces(tmp_path: Path) -> None:
    """The per-shard file bound resolves with RunConfig > file-config > default
    and degrades malformed ints to the named default (never raises)."""

    # Default.
    cfg = RunConfig(target=str(tmp_path))
    assert _deep_shard_max_files(cfg) == DEFAULT_DEEP_SHARD_MAX_FILES

    # RunConfig int override wins.
    cfg = RunConfig(target=str(tmp_path), deep_shard_max_files=5)
    assert _deep_shard_max_files(cfg) == 5

    # Float must degrade, not raise.
    cfg = RunConfig(target=str(tmp_path),
        deep_shard_max_files=2.5,  # type: ignore[arg-type]
    )
    assert _deep_shard_max_files(cfg) == DEFAULT_DEEP_SHARD_MAX_FILES

    # File-config override applies when no RunConfig attr is set.
    fc = DaydreamFileConfig(deep_shard_max_files=7)
    cfg = RunConfig(target=str(tmp_path), file_config=fc)
    assert _deep_shard_max_files(cfg) == 7

async def test_deep_large_diff_produces_review_and_record_shards(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch, install_backend: Callable[[object], object],
) -> None:

    install_stub_backend(monkeypatch, shard_many_python_target)
    # Sharding enabled, tiny file bound -> the python stack shards.
    rc = await run(RunConfig(
            target=str(shard_many_python_target), cleanup=False, deep_shard_enabled=True, deep_shard_max_files=1,
            deep_shard_max_bytes=10**9,
        )
    )
    assert rc == 0
    deep = shard_many_python_target / ".daydream" / "deep"
    shards = sorted(p for p in deep.glob("stack-python#*-review.md"))
    assert len(shards) >= 2  # >1 review task for one language stack
    records = sorted(p for p in deep.glob("stack-python#*-records.json"))
    assert records

async def test_deep_sharding_off_keeps_single_agent_per_stack(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch, install_backend: Callable[[object], object],
) -> None:
    install_stub_backend(monkeypatch, shard_many_python_target)
    rc = await run(RunConfig(target=str(shard_many_python_target), cleanup=False,
            deep_shard_enabled=False,  # default / sharding off
            deep_shard_max_files=1,
        )
    )  # bound ignored when off
    assert rc == 0
    deep = shard_many_python_target / ".daydream" / "deep"
    assert not list(deep.glob("stack-python#*-review.md"))  # exactly one agent per stack, as today

async def test_no_parse_phase_and_records_from_output_schema(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC4: no parse-<stack> fork exists; records come from output_schema."""
    _silence(monkeypatch)
    prompts = _install_model_capturing_stubs(monkeypatch, multi_stack_target)
    exit_code = await _run_deep(multi_stack_target)
    assert exit_code == 0

    # (1) No parse-<stack> phase: the phase_parse_feedback prompt never fires.
    assert not any("extract only actionable issues" in c["prompt"].lower() for c in prompts), (
        "a parse phase prompt fired; parse-* stage must be removed"
    )
    assert not any("parse" in str(c.get("model", "")).lower() for c in prompts)

    # (2) Per-stack records files exist and hold the reviewer's structured
    # output. Each lens is checked against its own stub wording ("Sample issue"
    # for the language stacks, a distinct description for the structural
    # meta-stack), so the assertion cannot depend on glob ordering.
    deep_dir_path = multi_stack_target / ".daydream" / "deep"
    records = sorted(deep_dir_path.glob("stack-*-records.json"))
    assert records, "expected per-stack records written by the reviewer"
    language = [p for p in records if p.name != "stack-structure-records.json"]
    assert language, "expected at least one language-stack records file"
    loaded = json.loads(language[0].read_text())
    assert loaded["issues"][0]["description"] == "Sample issue"
    structural = deep_dir_path / "stack-structure-records.json"
    assert structural.is_file(), "expected the structural meta-stack records file"
    assert json.loads(structural.read_text())["issues"][0]["description"] == "Structural maintainability concern"


    # Review requests keep the persisted changed-line authority.
    prompt = next(c["prompt"] for c in prompts if "Relevant diff hunks" in c["prompt"])
    assert "hunk-index.json" in prompt or "changed line ranges" in prompt.lower()
    assert "do NOT re-Read diff.patch" in prompt or "diff.patch" not in prompt
