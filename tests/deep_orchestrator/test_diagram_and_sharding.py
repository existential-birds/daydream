"""Sweep Diagram And Sharding."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from daydream.deep import orchestrator
from daydream.deep.dependency import build_import_graph
from daydream.review_budget import review_scale_for_diff
from daydream.run_config import RunConfig
from tests.deep_orchestrator.test_review_completion import ReviewRun, scopes
from tests.harness.git_helpers import git, seed_feature_branch


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
    # An unavailable external grammar fails open to a graph without edges.
    import tree_sitter_python

    from daydream.tree_sitter_index import runtime

    def unavailable_grammar() -> None:
        raise RuntimeError('external Python grammar unavailable')

    monkeypatch.setattr(runtime, '_PARSER_CACHE', {})
    monkeypatch.setattr(tree_sitter_python, 'language', unavailable_grammar)
    assert build_import_graph(changes, repo) == {path: set() for path in changes}


async def test_below_scale4_overloaded_stack_publishes_automatic_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New and deleted tests stay primary review inputs in automatic language shards."""
    language = 'python'
    before = {f'module_{index}.py': 'VALUE = 0\n' for index in range(6)}
    before['test_removed.py'] = 'from module_0 import VALUE\n\ndef test_removed():\n    assert VALUE == 0\n'
    before.update({'README.md': '# Project\n',
                   'App.tsx': 'import React from "react"; export const App = () => <div/>;\n'})
    additions = ''.join(f'# changed {index}: ' + 'x' * 70 + '\n' for index in range(110))
    after = {path: text + additions for path, text in before.items()
             if path.endswith('.py') and path != 'test_removed.py'}
    after['test_added.py'] = before['test_removed.py'].replace('removed', 'added') + additions
    after.update({'README.md': '# Project\nUpdated contract.\n',
                  'App.tsx': 'import React from "react"; export const App = () => <div>Updated</div>;\n'})
    repo = tmp_path / language
    seed_feature_branch(repo, base=before, feature=after)
    # seed_feature_branch writes additions/modifications; omissions are not deletions.
    (repo / 'test_removed.py').unlink()
    git(repo, 'add', 'test_removed.py')
    git(repo, 'commit', '--amend', '-m', 'feature with removed test')
    diff = git(repo, 'diff', 'main..HEAD')
    assert review_scale_for_diff(diff) < 4
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




def test_tiny_diff_collapse_and_fanout_cap_preserve_routing_and_only_coarsen_the_largest_stack() -> None:
    """JavaScript-generic routing survives tiny-diff collapse; the cap coarsens only what it must.

    Runner sharding journeys use one language, so neither the guard nor the early cap exit
    is reached there.
    """
    from daydream.deep.detection import StackAssignment
    from daydream.deep.sharding import shard_stacks

    script = [StackAssignment('python', ['a.py']), StackAssignment('generic', ['b.ts'])]
    assert orchestrator._collapse_stacks_for_tiny_diff(script, ['a.py', 'b.ts'], threshold=4) == (script, False)
    # Shallow collapse cannot give one language agent two per-language scopes: it uses generic.
    languages = [StackAssignment('python', ['a.py']), StackAssignment('react', ['b.tsx'])]
    structure = StackAssignment('structure', ['a.py', 'b.tsx'])
    collapsed, single = orchestrator._collapse_stacks_for_shallow(
        [*languages, structure], ['a.py', 'b.tsx'], RunConfig(target='.'))
    assert single and [(s.stack_name, s.files) for s in collapsed] == [
        ('structure', ['a.py', 'b.tsx']), ('generic', ['a.py', 'b.tsx'])]
    python = StackAssignment('python', [f'p{index}.py' for index in range(4)])
    go = StackAssignment('go', [f'g{index}.go' for index in range(3)])
    shards = shard_stacks([python, go], '', max_files=1, max_bytes=10**9, fanout_cap=5, frontier_max=0)
    assert [(shard.stack_name, len(shard.files)) for shard in shards] == [
        ('python#0', 2), ('python#1', 2), ('go#0', 1), ('go#1', 1), ('go#2', 1)]
