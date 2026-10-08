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
        deferred, = [part for part in stage['supporting_parts'] if part['file'] == 'service.py']
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
    assert event['metadata']['attempt'] == 1 and event['metadata']['observed_tool_starts'] == 4
    if syntax_failure:
        assert phases['alternatives']['reason_codes'] == ['malformed_output']
        assert event['metadata']['failure_class'] == 'syntax_failure'
        coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
        assert coverage['diagnostics']['scopes']['structure'] in coverage['diagnostics']['phases']['alternatives']
