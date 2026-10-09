"""Whole-change design duty proves behavior and honest folded failure lineage."""
from __future__ import annotations

import json
import runpy
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, StagedBackend, stage_ends
from tests.harness.git_helpers import seed_feature_branch
from tests.harness.stub_backend import review_stage_state, stage_result
from tests.test_deep_orchestrator import _sanctioned_inputs


@pytest.mark.parametrize(('split_owner', 'syntax_failure'), [(True, False), (False, False), (False, True)],
                         ids=['concrete-design-downside', 'single-owner-clean', 'folded-syntax-lineage'])
async def test_structure_fulfils_folded_design_duty_after_early_docs_or_propagates_its_real_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split_owner: bool, syntax_failure: bool,
) -> None:
    repo = tmp_path / 'folded_design'
    store = ('class Store:\n    def __init__(self):\n        self.cache = {}\n'
             '    def publish(self, key, value):\n        self.cache[key] = value\n'
             '    def fetch(self, key):\n        return self.cache.get(key)\n'
             '    def invalidate(self, key):\n        self.cache.pop(key, None)\n')
    previous = ('class Service:\n    def __init__(self, store):\n        self.store = store\n'
                '    def publish(self, key, value):\n        self.store.publish(key, value)\n')
    current = ('class Service:\n    def __init__(self, store):\n        self.store = store\n'
               + ('        self.private_cache = {}\n' if split_owner else '')
               + '    def publish(self, key, value):\n'
               + ('        self.private_cache[key] = value\n' if split_owner else
                  '        self.store.publish(key, value)\n')
               + '    def invalidate(self, key):\n        self.store.invalidate(key)\n')
    before = {'store.py': store, 'service.py': previous}
    after = {'store.py': store + '# Shared invalidation remains authoritative.\n', 'service.py': current}
    for index in range(14):
        path = f'00_guide_{index:02}.md'
        before[path] = '# Shared cache ownership\n'
        after[path] = before[path] + 'Publish, fetch and invalidate use the same owner.\n' * 30
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    judged: list[bool] = []
    title = 'Duplicated cache ownership bypasses shared invalidation'

    def design_response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        assert stage['stage'] == 'integration' and stage['assigned_target_ids'] == ['integration:structure']
        assert stage['folded_alternatives'] is True
        assert stage['assigned_files'][0].startswith('00_guide_')
        assert {'store.py', 'service.py'} <= set(stage['assigned_files'])
        assert stage['remaining_work']['assignment_units'] == stage['remaining_work']['stages'] == 1
        bundle = Path(stage['supporting_bundle']['path'])
        inventory = bundle.read_text()
        assert 'store.py' in inventory and 'service.py' in inventory
        yield ToolStartEvent(id='interaction-inventory', name='Read', input={'file_path': str(bundle)})
        yield ToolResultEvent(id='interaction-inventory', output=inventory, is_error=False)
        catalog_path = stage['supporting_catalog']['path']
        catalog_body = Path(catalog_path).read_text()
        yield ToolStartEvent(id='supporting-catalog', name='Read', input={'file_path': catalog_path})
        yield ToolResultEvent(id='supporting-catalog', output=catalog_body, is_error=False)
        deferred, = [part for part in json.loads(catalog_body)['parts'] if part['file'] == 'service.py']
        diff = Path(deferred['path']).read_text()
        assert 'class Service:' in diff
        yield ToolStartEvent(id='service-supporting-diff', name='Read', input={'file_path': deferred['path']})
        yield ToolResultEvent(id='service-supporting-diff', output=diff, is_error=False)
        for path in ['store.py', 'service.py']:
            window, = [window for window in stage['source_access']
                       if window['file'] == path and window['side'] == 'after']
            source = Path(window['access']['path']).read_text()
            assert source == after[path]
            yield ToolStartEvent(id=f'design-source-{path}', name='Read', input={'file_path': window['access']['path']})
            yield ToolResultEvent(id=f'design-source-{path}', output=source, is_error=False)
        Store = runpy.run_path(str(repo / 'store.py'))['Store']
        Service = runpy.run_path(str(repo / 'service.py'))['Service']
        owner = Store()
        Service(owner).publish('item', 'new')
        downside = owner.fetch('item') != 'new'
        judged.append(downside)
        owner.invalidate('item')
        assert owner.fetch('item') is None
        output['notes'] = ('Shared publish/fetch/invalidation ownership splits across incompatible stores.' if downside
                           else 'One Store owner serves publish/fetch/invalidation; an extra cache is unnecessary.')
        if downside:
            finding = {'id': 1, 'file': 'service.py', 'line': 4, 'severity': 'medium', 'confidence': 'MEDIUM',
                       'description': title,
                       'rationale': 'The service duplicates state outside the invalidation owner.',
                       'evidence': 'service.py:4 creates private_cache; store.py:9 invalidates only Store.cache.'}
            output['candidates'] = [{'candidate_id': '', 'file': 'service.py', 'line': 4,
                                     'trigger': 'Publish through Service and fetch or invalidate through Store',
                                     'consequence': 'The owners disagree about cached data',
                                     'grounds': finding['evidence'],
                                     'disposition': 'confirmed', 'finding': finding}]

    class DesignBackend(StagedBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            if stage is None or stage['scope_id'] != 'structure':
                async for event in super().execute(cwd, prompt, *args, **kwargs):
                    yield event
                return
            self.stages.append(stage)
            self.calls.append({'prompt': prompt, **kwargs})
            output = stage_result(stage)
            for event in design_response(stage, output):
                yield event
            if syntax_failure:
                yield TextEvent(text=json.dumps(output) + ', "candidates": []}')
                yield ResultEvent(structured_output=None, continuation=None)
            else:
                yield ResultEvent(structured_output=output, continuation=None)

    review.backend = DesignBackend(repo)
    data = await review.finish('structure', reason='malformed_output' if syntax_failure else None,
                               findings=(title,) if split_owner else ())
    assert judged == [split_owner]
    assert not any('evaluate the implementation' in call['prompt'].lower() for call in review.backend.calls)
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    assert phases['alternatives']['status'] == ('failed' if syntax_failure else 'complete')
    event, = stage_ends(review, 'structure')
    assert event['metadata']['attempt'] == 1 and event['metadata']['observed_tool_starts'] == 5
    if syntax_failure:
        assert phases['alternatives']['reason_codes'] == ['malformed_output']
        assert event['metadata']['failure_class'] == 'syntax_failure'
        coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
        assert coverage['diagnostics']['scopes']['structure'] in coverage['diagnostics']['phases']['alternatives']


@pytest.mark.parametrize('wired', [False, True], ids=['whole-change-defect', 'whole-change-clean'])
async def test_structure_initial_prompt_is_compact_with_complete_bounded_supporting_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wired: bool,
) -> None:
    import ast

    from tests.deep_orchestrator.test_review_native_capacity import _workload
    from tests.deep_orchestrator.test_review_native_sources import NativeSourceBackend

    before, after = _workload(wired)
    for index in range(5):
        path = f'00_context_{index}.md'
        before[path] = '# Request boundary\n'
        after[path] = before[path] + 'Preserve the dry-run request boundary.\n'
    for index in range(43):
        path = f'transport/module_{index:02}.py'
        before[path], after[path] = 'VALUE = 0\n', 'VALUE = 1\n'
    # Full enclosing source remains captured alongside deferred changed parts;
    # their combined capture exceeds the previous 4 MiB transport allowance.
    for path in before:
        context = ''.join(f'# Transport context {index:03}: ' + 'c' * 28 + '\n' for index in range(400))
        before[path] += context
        after[path] += context
    repo = tmp_path / 'large_interaction'
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    title = 'Whole-change request construction drops the parsed dry-run flag'
    judged: list[bool] = []
    prompt_sizes: list[int] = []
    catalog_paths: set[Path] = set()
    transport_sizes: list[tuple[int, int]] = []

    def catalog(path: Path, found: list[dict[str, Any]]) -> Iterable[AgentEvent]:
        catalog_paths.add(path)
        raw = path.read_text()
        assert len(raw.encode()) <= 12_000
        yield ToolStartEvent(id=f'catalog-{path.name}', name='Read', input={'file_path': str(path)})
        yield ToolResultEvent(id=f'catalog-{path.name}', output=raw, is_error=False)
        data = json.loads(raw)
        if 'parts' in data:
            found.extend(data['parts'])
        else:
            for child in data['catalogs']:
                yield from catalog(Path(child['path']), found)

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'structure':
            return
        prompt_sizes.append(len(review.backend.calls[-1]['prompt'].encode()))
        assert prompt_sizes[-1] < 128 * 1024
        assert len(stage['assigned_files']) == 90 and stage['assigned_target_ids'] == ['integration:structure']
        assert stage['folded_alternatives'] is True
        assert 'supporting_parts' not in stage
        assert all('path' not in window['access'] for window in stage['source_access'])
        guide = json.loads(stage['access_guide'])
        assert len(stage['access_guide'].encode()) <= 8192
        assert guide['source_catalog'] == stage['source_catalog']
        assert stage['source_catalog']['status'] == 'complete'
        source_paths: set[Path] = set()
        source_entries: list[dict[str, Any]] = []
        def source_catalog(path: Path) -> None:
            source_paths.add(path)
            raw = path.read_text()
            assert len(raw.encode()) <= 12000
            data = json.loads(raw)
            if 'windows' in data:
                source_entries.extend(data['windows'])
            else:
                for child in data['catalogs']:
                    source_catalog(Path(child['path']))
        source_catalog(Path(guide['source_catalog']['path']))
        recipe = review.backend.calls[-1]['source_recipe']
        assert len(source_entries) == len(recipe.windows)
        for entry in source_entries:
            assert entry['access']['tool'] == 'read_source' and 'path' not in entry['access']
            for target in entry['target_ids']:
                window = recipe.selector(target, entry['side'])
                assert window is not None
                assert window.metadata() == {key: entry[key] for key in window.metadata()}
        found: list[dict[str, Any]] = []
        yield from catalog(Path(stage['supporting_catalog']['path']), found)
        assert not source_paths.intersection(catalog_paths)
        assert len(found) == stage['supporting_catalog']['part_count'] > 90
        assert {part['file'] for part in found} == set(stage['assigned_files'])
        recipe = review.backend.calls[-1]['source_recipe']
        assert {window.file for window in recipe.windows} == set(stage['assigned_files'])
        inputs = catalog_paths | source_paths | {Path(part['path']) for part in found}
        for window in recipe.windows:
            assert window.projection is not None
            assert window.projection.read_text() == window.body
            assert window.body == (before if window.side == 'before' else after)[window.file]
            inputs.add(window.projection)
        inputs.update(_sanctioned_inputs(review.backend.calls[-1]['prompt']).values())
        # Inspect complete available bytes, including pointers omitted from the
        # compact initial prompt, at the external provider dispatch boundary.
        sizes = [len(path.read_bytes()) for path in inputs]
        transport_sizes.append((sum(sizes), len(sizes)))
        assert 4_194_304 < sum(sizes) < 8_388_608
        assert max(sizes) <= 1_048_576 and len(sizes) <= 512
        producer, consumer = 'package_0/module_00.py', 'package_8/module_40.py'
        deferred, = [part for part in found if part['file'] == consumer]
        raw = Path(deferred['path']).read_text()
        assert 'def build_request(options)' in raw
        yield ToolStartEvent(id='late-supporting-diff', name='Read', input={'file_path': deferred['path']})
        yield ToolResultEvent(id='late-supporting-diff', output=raw, is_error=False)
        bodies: dict[str, str] = {}
        for path in (producer, consumer):
            source = recipe.selector(path, 'after')
            assert source is not None
            bodies[path] = source.body
            yield ToolStartEvent(id=f'whole-source-{path}', name='read_source',
                                 input={'target_id': path, 'side': 'after'})
            yield ToolResultEvent(id=f'whole-source-{path}', is_error=False,
                                  output=json.dumps({'source': source.metadata(), 'body': source.body}))
        producer_tree = ast.parse(bodies[producer])
        assert any(isinstance(node, ast.FunctionDef) and node.name == 'parse_flags' for node in producer_tree.body)
        consumer_tree = ast.parse(bodies[consumer])
        fn = next(node for node in consumer_tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'build_request')
        returned = fn.body[0]
        assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Dict)
        carries = any(isinstance(key, ast.Constant) and key.value == 'dry_run' for key in returned.value.keys)
        judged.append(carries)
        output['notes'] = 'The parsed option was traced into the returned request across the complete inventory.'
        if not carries:
            finding = {'id': 1, 'file': consumer, 'line': 4, 'severity': 'medium', 'confidence': 'MEDIUM',
                       'description': title, 'rationale': 'The parser creates a dry-run field absent from the request.',
                       'evidence': f'{producer}: defines dry_run; {consumer}:4 returns an empty request.'}
            output['candidates'] = [{'candidate_id': '', 'file': consumer, 'line': 4,
                                     'trigger': 'Parse --dry-run and build the request',
                                     'consequence': 'The request loses the dry-run flag',
                                     'grounds': finding['evidence'],
                                     'disposition': 'confirmed', 'finding': finding}]

    review.backend.stage_response = response
    data = await review.finish('structure', findings=(title,) if not wired else ())
    assert judged == [wired] and len(prompt_sizes) == 1
    assert len(transport_sizes) == 1
    assert transport_sizes[0][0] > 32 * prompt_sizes[0]
    metadata = stage_ends(review, 'structure')[0]['metadata']
    assert 4 * 1024 * 1024 < metadata['sanctioned_input_bytes'] <= 8 * 1024 * 1024
    assert 180 < metadata['sanctioned_input_count'] <= 512
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'structure'))
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    assert phases['alternatives']['status'] == 'complete'
