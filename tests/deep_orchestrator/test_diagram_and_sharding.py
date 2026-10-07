"""Sweep Diagram And Sharding."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from daydream.config import (
    DEFAULT_DEEP_SHARD_ENABLED,
    DEFAULT_DEEP_SHARD_MAX_BYTES,
    DEFAULT_DEEP_SHARD_MAX_FILES,
)
from daydream.config_file import DaydreamFileConfig
from daydream.deep.orchestrator import (
    DIAGRAM_STEPS,
    STEPS,
    _config_pipeline,
    _deep_shard_enabled,
    _deep_shard_max_files,
    _flow_kind_for_mode,
    _flow_name_for_mode,
    _resolve_mode,
)
from daydream.extensions import Registry
from daydream.extensions.builtins import register_builtins
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.trajectory import DaydreamRunFlow
from tests.harness.stub_backend import install_stub_backend
from tests.test_deep_orchestrator import (
    _install_model_capturing_stubs,
    _profile_with_pipeline,
    _run_deep,
    _silence,
)

if TYPE_CHECKING:
    pass

def test_diagram_step_position_and_phase_key() -> None:
    names = [step.name for step in STEPS]
    assert names.index("supervise") + 1 == names.index("diagram")
    assert names.index("diagram") + 1 == names.index("findings-out")
    steps = {step.name: step for step in STEPS}
    assert steps["diagram"].phase_key == "diagram"
    # ``post-diagram`` must NOT be in STEPS: ``_register_builtin_flows``
    # derives the deep flow definition from it, so a GitHub write would be
    # spliced into every deep review.
    assert "post-diagram" not in names
    assert [step.name for step in DIAGRAM_STEPS] == ["post-diagram"]
    assert DIAGRAM_STEPS[0].phase_key == "post-diagram"

def test_diagram_flow_is_registered_with_its_three_steps() -> None:
    registry = Registry()
    register_builtins(registry)
    assert sorted(registry.flow_names()) == ["deep", "diagram", "improve"]
    assert registry.flow("diagram") == ["exploration", "diagram", "post-diagram"]

def test_resolve_mode_maps_diagram_output_mode() -> None:
    config = RunConfig(target="/tmp", output_mode="diagram", diagram="sequence")
    assert _resolve_mode(config) == "diagram"
    assert _flow_name_for_mode("diagram") == "diagram"
    assert _flow_kind_for_mode("diagram") is DaydreamRunFlow.DIAGRAM
    # ``--shallow`` must not win over an explicit diagram-only request.
    shallow = RunConfig(target="/tmp", output_mode="diagram", diagram="both", shallow=True)
    assert _resolve_mode(shallow) == "diagram"
    for mode in ("loop", "comment", "review", "shallow"):
        assert _flow_name_for_mode(mode) == "deep"

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

def test_deep_shard_default_bounds_align_with_inline_budget() -> None:
    """Issue #740: the default shard bounds retune to 5 files / 12288 bytes, and
    the byte bound equals INLINE_DIFF_BUDGET_BYTES so shards inline by construction."""
    assert DEFAULT_DEEP_SHARD_MAX_FILES == 5
    assert DEFAULT_DEEP_SHARD_MAX_BYTES == INLINE_DIFF_BUDGET_BYTES  # == 12_288

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

def test_structural_gate_resolver_reads_profile_pipeline() -> None:
    assert _config_pipeline(RunConfig(target="/tmp/x")).structural_enabled is True
    off = _profile_with_pipeline(structural_enabled=False)
    assert _config_pipeline(RunConfig(target="/tmp/x", review_profile=off)).structural_enabled is False
