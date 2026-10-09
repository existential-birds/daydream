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
from tests.harness.stub_backend import install_stub_backend, review_stage_state
from tests.test_deep_orchestrator import (
    _install_model_capturing_stubs,
    _profile_with_pipeline,
    _run_deep,
    _silence,
)

if TYPE_CHECKING:
    pass




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
    language_prompts = [c["prompt"] for c in prompts if (stage := review_stage_state(c["prompt"])) is not None
                        and stage["stage"] == "first_pass"]
    assert language_prompts
    for prompt in language_prompts:
        from tests.deep_orchestrator.test_review_capture_and_retry import supporting_contents
        state = review_stage_state(prompt)
        assert state is not None
        supporting = supporting_contents(prompt)
        assert set(json.loads(supporting['hunk-index'])) == set(state['assigned_files'])
        assert all(f'diff --git a/{path} b/{path}' in supporting['diff'] for path in state['assigned_files'])



async def test_default_cap_coarsens_only_enough_and_preserves_scope_qualified_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections import Counter

    from tests.deep_orchestrator.test_review_capture_and_retry import supporting_contents
    from tests.deep_orchestrator.test_review_completion import scopes
    from tests.deep_orchestrator.test_review_investigation import InvestigationRun
    from tests.harness.git_helpers import seed_feature_branch

    before = {f'module_{index:02}.py': 'VALUE = 0\n' for index in range(41)}
    before.update({f'guide_{index}.md': '# Guide\n' for index in range(7)})
    before.update({'App.tsx': 'export const App = () => <div/>;\n',
                   'Other.tsx': 'export const Other = () => <div/>;\n'})
    after = {path: text + ''.join(f'# changed {index}: ' + 'x' * 75 + '\n' for index in range(155))
             for path, text in before.items() if not path.endswith('.tsx')}
    after.update({path: text + '// Retain component contract.\n'
                  for path, text in before.items() if path.endswith('.tsx')})
    repo = tmp_path / 'cap_workload'
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    assert await review.run() == 0
    data = review.load()
    outcomes = scopes(data)
    python = [scope for name, scope in outcomes.items() if name.startswith('python#')]
    assert len(python) == 8
    assert len(outcomes) == 17 and all(scope['status'] == 'complete' for scope in outcomes.values())
    assigned = [path for name, scope in outcomes.items() if name != 'structure' for path in scope['files']]
    assert Counter(assigned) == Counter(before.keys())
    required: set[tuple[str, str]] = set()
    for stage in review.backend.stages:
        if stage['stage'] != 'first_pass':
            continue
        for target in stage['assigned_target_ids']:
            key = (stage['scope_id'], target)
            assert key not in required
            required.add(key)
    assert len(required) > len(before)
    for call in review.backend.calls:
        call_stage = review_stage_state(call['prompt'])
        if call_stage is None or call_stage['stage'] != 'first_pass':
            continue
        contents = supporting_contents(call['prompt'])
        from daydream.prompt_budget import inline_section_emitted_bytes
        logical = [('review-assignment', contents['review-assignment'])] if 'review-assignment' in contents else [
            (label, contents[label]) for label in ('diff', 'hunk-index', 'input-binding')]
        assert inline_section_emitted_bytes([(label, len(text.encode())) for label, text in logical]) <= 12288
    coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    assert coverage['stack_outcomes'] == data['terminal_result']['stack_outcomes']
