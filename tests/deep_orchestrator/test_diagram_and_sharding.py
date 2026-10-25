"""Sweep Diagram And Sharding."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from daydream.config import (
    DEFAULT_DEEP_SHARD_ENABLED,
    DEFAULT_DEEP_SHARD_MAX_FILES,
)
from daydream.config_file import DaydreamFileConfig
from daydream.deep import orchestrator
from daydream.deep.dependency import build_import_graph
from daydream.deep.detection import StackAssignment
from daydream.deep.orchestrator import (
    _deep_shard_enabled,
    _deep_shard_max_files,
)
from daydream.deep.sharding import shard_stacks
from daydream.review_budget import review_scale_for_diff
from daydream.run_config import RunConfig
from tests.conftest import ExtDir
from tests.deep_orchestrator.test_review_completion import ReviewRun, scopes
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.stub_backend import review_stage_state
from tests.test_deep_orchestrator import _profile_with_pipeline


@pytest.mark.parametrize('shape', ['fitting', 'singleton', 'cap-coarsened', 'explicit-false'])
def test_preparation_skips_native_graph_when_sharding_cannot_help(
    sibling_frontier_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str,
) -> None:
    """Supplement runner ownership proofs with the independent graph-work contract."""
    repo = sibling_frontier_target
    if shape == 'singleton':
        repo = tmp_path / 'graph_singleton'
        seed_feature_branch(repo, base={'module.py': 'VALUE = 0\n'},
                            feature={'module.py': 'VALUE = 0\n' + '# changed\n' * 50})
    changes = git(repo, 'diff', '--name-only', 'main..HEAD').splitlines()
    diff = git(repo, 'diff', 'main..HEAD')
    config = RunConfig(target=str(repo), diagram='off', shallow_fanout_threshold=0,
        deep_shard_enabled=shape != 'explicit-false',
        deep_shard_max_files=20 if shape == 'fitting' else 1,
        deep_shard_max_bytes=10**9 if shape == 'fitting' else 1,
        deep_shard_fanout_cap=1 if shape == 'cap-coarsened' else 16)
    calls: list[list[str]] = []

    def observed_build(files: list[str], root: Path) -> dict[str, set[str]]:
        calls.append(files)
        return build_import_graph(files, root)

    monkeypatch.setattr(orchestrator, 'build_import_graph', observed_build)
    assignments, collapsed, graph = orchestrator._prepare_review_stacks(config, changes, diff, repo, 'review')
    assert collapsed is False
    assert calls == [] and graph == {}
    assert {scope.stack_name for scope in assignments} == {'structure', 'python'}
    assert Counter(path for scope in assignments if scope.stack_name != 'structure'
                   for path in scope.files) == Counter(changes)


@pytest.mark.parametrize(('run_enabled', 'file_enabled', 'should_plan'), [
    (False, True, False),
    (None, False, False),
    (True, False, True),
], ids=['runconfig-false', 'file-false', 'runconfig-true-over-file-false'])
def test_explicit_sharding_opt_out_skips_feasibility_packing(
    sibling_frontier_target: Path, monkeypatch: pytest.MonkeyPatch,
    run_enabled: bool | None, file_enabled: bool, should_plan: bool,
) -> None:
    """Supplement real runner precedence coverage with the independent packing-work contract."""
    repo = sibling_frontier_target
    changes = git(repo, 'diff', '--name-only', 'main..HEAD').splitlines()
    diff = git(repo, 'diff', 'main..HEAD')
    planned: list[None] = []

    def observed_plan(*args: Any, **kwargs: Any) -> list[StackAssignment]:
        planned.append(None)
        return shard_stacks(*args, **kwargs)

    monkeypatch.setattr(orchestrator, 'shard_stacks', observed_plan)
    config = RunConfig(target=str(repo), diagram='off', deep_shard_enabled=run_enabled,
                       file_config=DaydreamFileConfig(deep_shard_enabled=file_enabled))
    assignments, collapsed, _ = orchestrator._prepare_review_stacks(config, changes, diff, repo, 'review')
    assert collapsed is False
    assert bool(planned) is should_plan
    assert any('#' in scope.stack_name for scope in assignments) is should_plan
    assert Counter(path for scope in assignments if scope.stack_name != 'structure'
                   for path in scope.files) == Counter(changes)


def test_preparation_shares_native_graph_with_diagram_and_preserves_scopes_after_failure(
    sibling_frontier_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native builder failure retains useful shards through the outer preparation guard."""
    repo = sibling_frontier_target
    changes = git(repo, 'diff', '--name-only', 'main..HEAD').splitlines()
    diff = git(repo, 'diff', 'main..HEAD')
    config = RunConfig(target=str(repo), diagram='auto', shallow_fanout_threshold=0,
        deep_shard_enabled=True, deep_shard_max_files=5, deep_shard_max_bytes=700)
    calls: list[list[str]] = []
    fail = False

    def observed_build(files: list[str], root: Path) -> dict[str, set[str]]:
        calls.append(files)
        graph = build_import_graph(files, root)
        if fail:
            raise RuntimeError('native graph build failed outside parser recovery')
        return graph

    monkeypatch.setattr(orchestrator, 'build_import_graph', observed_build)
    normal, _, graph = orchestrator._prepare_review_stacks(config, changes, diff, repo, 'review')
    assert calls == [changes] and any(graph.values())
    assert any(scope.frontier_files for scope in normal)
    calls.clear()
    fail = True
    recovered, collapsed, failed_graph = orchestrator._prepare_review_stacks(config, changes, diff, repo, 'review')
    assert calls == [changes] and collapsed is False and failed_graph == {}
    assert [(scope.stack_name, scope.files) for scope in recovered] == [
        (scope.stack_name, scope.files) for scope in normal]
    assert [scope.stack_name for scope in recovered] == ['structure', *(f'python#{index}' for index in range(5))]
    assert Counter(path for scope in recovered if scope.stack_name != 'structure'
                   for path in scope.files) == Counter(changes)
    assert set(recovered[0].files) == set(changes)
    assert all(not scope.frontier_files for scope in recovered)


@pytest.mark.parametrize('language', ['go', 'python'])
async def test_below_scale4_overloaded_stack_publishes_automatic_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, language: str,
) -> None:
    """New and deleted tests stay primary review inputs in automatic language shards."""
    suffix = 'go' if language == 'go' else 'py'
    source = 'package sample\n\nfunc Value() int { return 0 }\n' if language == 'go' else 'VALUE = 0\n'
    old_test = 'removed_test.go' if language == 'go' else 'test_removed.py'
    new_test = 'added_test.go' if language == 'go' else 'test_added.py'
    before = {f'module_{index}.{suffix}': source.replace('Value()', f'Value{index}()') for index in range(6)}
    before[old_test] = ('package sample\n\nimport "testing"\n\n'
                        'func TestRemoved(t *testing.T) { if Value0() != 0 { t.Fatal("value") } }\n'
                        if language == 'go' else
                        'from module_0 import VALUE\n\ndef test_removed():\n    assert VALUE == 0\n')
    before.update({'README.md': '# Project\n',
                   'App.tsx': 'import React from "react"; export const App = () => <div/>;\n'})
    comment = '// ' if language == 'go' else '# '
    additions = ''.join(f'{comment}changed {index}: ' + 'x' * 70 + '\n' for index in range(110))
    after = {path: text + additions for path, text in before.items() if path.endswith(suffix) and path != old_test}
    after[new_test] = before[old_test].replace('Removed', 'Added').replace('removed', 'added') + additions
    after.update({'README.md': '# Project\nUpdated contract.\n',
                  'App.tsx': 'import React from "react"; export const App = () => <div>Updated</div>;\n'})
    repo = tmp_path / language
    seed_feature_branch(repo, base=before, feature=after)
    # seed_feature_branch writes additions/modifications; omissions are not deletions.
    (repo / old_test).unlink()
    git(repo, 'add', old_test)
    git(repo, 'commit', '--amend', '-m', 'feature with removed test')
    diff = git(repo, 'diff', 'main..HEAD')
    assert review_scale_for_diff(diff) == 2
    assert 'deleted file mode 100644\n' in diff and 'new file mode 100644\n' in diff
    changed = git(repo, 'diff', '--name-only', 'main..HEAD').splitlines()
    review = ReviewRun(repo, tmp_path, monkeypatch)
    assert await review.run() == 0
    data = review.load()
    inventory = scopes(data)
    language_scopes = [name for name in inventory if name.startswith(f'{language}#')]
    assert len(language_scopes) > 1, 'below-scale4 overloaded stack remained a serial reviewer'
    # The full persisted ~68KiB patch packs into seven byte-bounded scopes;
    # the bounded prompt retains only ~10KiB and would under-shard this work.
    assert language_scopes == [f'{language}#{index}' for index in range(7)]
    assert Counter(path for name, scope in inventory.items() if name != 'structure'
                   for path in scope['files']) == Counter(changed)
    assert set(inventory['structure']['files']) == set(changed)
    assert all(scope['status'] == 'complete' for scope in inventory.values())
    assert data['findings'] == [] and data['terminal_result']['analysis_state'] == 'complete'
    coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    assert coverage['stack_outcomes'] == data['terminal_result']['stack_outcomes']
    assert coverage['analyzed_revision'] == data['terminal_result']['analyzed_revision']
    assert [(scope['scope_id'], scope['files']) for scope in coverage['planned_scopes']] == [
        (name, scope['files']) for name, scope in inventory.items()]


@pytest.mark.parametrize(('overrides', 'expected_python'), [
    ({}, ['python']),
    ({'deep_shard_max_files': 1}, ['python#0', 'python#1', 'python#2']),
    ({'deep_shard_max_files': 0}, ['python#0', 'python#1', 'python#2']),
    ({'deep_shard_max_bytes': 0}, ['python#0', 'python#1', 'python#2']),
    ({'deep_shard_max_files': 1, 'deep_shard_fanout_cap': 0}, ['python']),
    ({'deep_shard_max_files': 5, 'file_config': DaydreamFileConfig(deep_shard_max_files=1)}, ['python']),
    ({'deep_shard_max_files': -1, 'file_config': DaydreamFileConfig(deep_shard_max_files=1)}, ['python']),
    ({'deep_shard_max_files': 1, 'shallow_fanout_threshold': 4}, ['python']),
    ({'deep_shard_max_files': 1, 'shallow': True, 'output_mode': 'loop', 'findings_out': None}, ['python']),
    ({'deep_shard_max_files': 1, 'review_profile': _profile_with_pipeline(structural_enabled=False)},
     ['python#0', 'python#1', 'python#2']),
], ids=['fitting', 'file-bound', 'zero-files', 'zero-bytes', 'zero-cap', 'runconfig-bound',
        'invalid-bound-fallback', 'tiny-collapse', 'shallow', 'structure-disabled'])
async def test_resolved_sharding_controls_publish_actual_review_scopes(
    shard_many_python_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any], expected_python: list[str],
) -> None:
    review = ReviewRun(shard_many_python_target, tmp_path, monkeypatch)
    assert await review.run(**overrides) == 0
    coverage = json.loads((shard_many_python_target / '.daydream/deep/review-coverage.json').read_text())
    inventory = {scope['scope_id']: scope for scope in coverage['stack_outcomes']}
    if overrides.get('output_mode') != 'loop':
        assert scopes(review.load()) == inventory
    if "review_profile" in overrides:
        assert 'structure' not in inventory
        assert any("evaluate the implementation" in call["prompt"].lower() for call in review.backend.calls)
        alternatives = next(phase for phase in coverage["phase_outcomes"] if phase["phase"] == "alternatives")
        assert alternatives["status"] == "complete"
    assert [name for name in inventory if name.startswith('python')] == expected_python
    changed = git(shard_many_python_target, 'diff', '--name-only', 'main..HEAD').splitlines()
    assert Counter(path for name, scope in inventory.items() if name != 'structure'
                   for path in scope['files']) == Counter(changed)
    assert all(scope['status'] == 'complete' for scope in inventory.values())
    if overrides.get('output_mode') != 'loop':
        # The published reviewer files match the resolved scopes exactly,
        # including the absence of shard artifacts when explicitly disabled.
        deep = shard_many_python_target / '.daydream/deep'
        assert {path.name for path in deep.glob('stack-python*-review.md')} == {
            f'stack-{name}-review.md' for name in expected_python}
        assert {path.name for path in deep.glob('stack-python*-records.json')} == {
            f'stack-{name}-records.json' for name in expected_python}


async def test_oversized_singleton_retains_intact_primary_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / 'singleton'
    before = {'module.py': 'VALUE = 0\n', 'README.md': '# Project\n'}
    after = {'module.py': before['module.py'] + ''.join('# ' + 'x' * 100 + '\n' for _ in range(700)),
             'README.md': '# Project\nUpdated.\n'}
    seed_feature_branch(repo, base=before, feature=after)
    review = ReviewRun(repo, tmp_path, monkeypatch)
    assert await review.run() == 0
    inventory = scopes(review.load())
    assert set(inventory) == {'python', 'generic', 'structure'}
    assert inventory['python']['files'] == ['module.py']
    assert all(scope['status'] == 'complete' for scope in inventory.values())


def test_deep_shard_enabled_overrides_and_aggregate_fallback(tmp_path: Path) -> None:

    assert DEFAULT_DEEP_SHARD_ENABLED is False
    # No assignments supplied: the aggregate fallback retains its threshold.
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


async def test_default_cap_coarsens_only_enough_and_preserves_scope_qualified_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from collections import Counter

    from tests.deep_orchestrator.test_review_capture_and_retry import supporting_contents
    from tests.deep_orchestrator.test_review_completion import scopes
    from tests.deep_orchestrator.test_review_investigation import InvestigationRun
    from tests.harness.git_helpers import seed_feature_branch

    before = {f'module_{index:02}.py': 'VALUE = 0\n' for index in range(41)}
    before.update({f'guide_{index}.md': '# Guide\n' for index in range(7)})
    before.update({'App.tsx': 'import React from "react"; export const App = () => <div/>;\n',
                   'Other.tsx': 'import React from "react"; export const Other = () => <div/>;\n'})
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
    # These medium files now fit whole in the exact-path assignment cap; scope
    # coarsening still owns every file exactly once without resetting on hunks.
    assert required == {(name, path) for name, scope in outcomes.items() if name != 'structure'
                        for path in scope['files']}
    for call in review.backend.calls:
        call_stage = review_stage_state(call['prompt'])
        if call_stage is None or call_stage['stage'] != 'first_pass':
            continue
        contents = supporting_contents(call['prompt'])
        from daydream.prompt_budget import inline_section_emitted_bytes
        logical = [('review-assignment', contents['review-assignment'])] if 'review-assignment' in contents else [
            (label, contents[label]) for label in ('diff', 'hunk-index', 'input-binding')]
        assert inline_section_emitted_bytes([(label, len(text.encode())) for label, text in logical]) <= 24 * 1024
    coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    assert coverage['stack_outcomes'] == data['terminal_result']['stack_outcomes']


@pytest.mark.parametrize('fallback', [
    'plain-typescript', 'malformed-package', 'conflicting-tsconfig', 'explicit-rule',
    'python-plain-typescript', 'python-angular',
])
async def test_automatic_tiny_review_preserves_framework_and_explicit_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, fallback: str,
) -> None:
    """Target evidence and fork ownership survive automatic tiny-diff handling."""
    repo = tmp_path / 'framework_routing'
    primary_stack = 'python' if fallback.startswith('python-') else 'react'
    component = 'api.py' if primary_stack == 'python' else 'packages/web/src/App.tsx'
    old_component = 'VALUE = "Hello"\n' if primary_stack == 'python' else (
        'export const App = () => <div>Hello</div>;\n'
    )
    secondary = 'tools/helper.ts'
    old_secondary, new_secondary = 'export const value = 1;\n', 'export const value = 2;\n'
    secondary_stack = 'generic'
    if fallback == 'python-angular':
        old_secondary = 'import {Component} from "@angular/core"; export const value = 1;\n'
        new_secondary = old_secondary.replace('1', '2')
    elif fallback == 'malformed-package':
        secondary, old_secondary, new_secondary = 'tools/package.json', '{}\n', '{broken\n'
    elif fallback == 'conflicting-tsconfig':
        secondary, old_secondary = 'tools/tsconfig.json', '{}\n'
        new_secondary = '{"compilerOptions":{"jsxImportSource":"preact"}}\n'
    elif fallback == 'explicit-rule':
        ext_dir.write_module(
            'from daydream.extensions import StackRule\n'
            'def register(registry):\n'
            '    registry.add_stack(StackRule("python", ("tools/helper.ts",)))\n'
        )
        secondary_stack = 'python'
    before = {
        'package.json': '{"dependencies":{"react":"*"}}\n',
        'packages/web/package.json': '{"dependencies":{"react":"*"}}\n',
        component: old_component,
        secondary: old_secondary,
    }
    seed_feature_branch(repo, base=before, feature={
        component: before[component].replace('Hello', 'Updated'),
        secondary: new_secondary,
    })
    review = ReviewRun(repo, tmp_path, monkeypatch)
    # ReviewRun disables tiny collapse for most tests. Use the production default:
    # the two changed files qualify, but their resolved ownership must survive.
    assert await review.run(shallow_fanout_threshold=None) == 0
    inventory = scopes(review.load())
    assert {name: scope['files'] for name, scope in inventory.items()} == {
        primary_stack: [component], secondary_stack: [secondary], 'structure': [component, secondary],
    }
    observed: dict[str, set[str]] = {}
    for call in review.backend.calls:
        stage = review_stage_state(call['prompt'])
        if stage is not None and stage['stage'] == 'first_pass':
            observed.setdefault(stage['scope_id'], set()).update(stage['assigned_target_ids'])
    assert observed == {primary_stack: {component}, secondary_stack: {secondary}}
    assert all(scope['status'] == 'complete' for scope in inventory.values())
