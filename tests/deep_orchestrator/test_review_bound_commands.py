"""Native frozen-source reads retain findings and honest coverage through runner.run."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from daydream.backends import AgentEvent, ResultEvent, ToolResultEvent, ToolStartEvent
from daydream.config_file import DaydreamFileConfig
from tests.deep_orchestrator.test_review_completion import scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, stage_ends
from tests.deep_orchestrator.test_review_native_sources import NativeSourceBackend
from tests.harness.git_helpers import seed_feature_branch, tracked_source_state
from tests.harness.stub_backend import review_stage_state, stage_result


class NativeReaderBackend(NativeSourceBackend):
    """Use the actual invocation reader at the external provider boundary."""

    read_only_disposable_clone = True

    def __init__(self, repo: Path, *, fault: str) -> None:
        super().__init__(repo)
        self.fault = fault
        self.reads: dict[str, list[dict[str, Any]]] = {'react': [], 'structure': []}
        self.decisions: list[tuple[str, str, str]] = []
        self.native_results: dict[str, list[ToolResultEvent]] = {'react': [], 'structure': []}

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        stage = review_stage_state(prompt)
        if stage is None or stage['scope_id'] not in self.reads:
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
            return
        from daydream.backends.source_reader import SourceReader

        self.stages.append(stage)
        self.calls.append({'cwd': cwd, 'prompt': prompt, **kwargs})
        scope = stage['scope_id']
        output = stage_result(stage)
        windows = [window for window in stage['source_access'] if window['side'] == 'after'
                   and window['file'].endswith('.tsx')]
        assert len(windows) == 4
        assert all(window['access']['tool'] == 'read_source' for window in windows)
        recipe = kwargs['source_recipe']
        assert recipe is not None
        captured = ''
        if self.fault in {'compound-context', 'shell-only'}:
            command = ' && '.join('cat ' + shlex.quote(window['source_path']) for window in windows)
            yield ToolStartEvent(id='compound-context', name='exec_command', input={'cmd': command})
            executed = subprocess.run(command, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
            assert executed.returncode == 0
            yield ToolResultEvent(id='compound-context', output=executed.stdout, is_error=False, exit_code=0)
            captured += executed.stdout
        if self.fault != 'shell-only':
            async with SourceReader(recipe) as reader:
                headers = {'Authorization': f'Bearer {reader.token}', 'Accept': 'application/json, text/event-stream'}
                async with httpx.AsyncClient(headers=headers) as client:
                    for index, window in enumerate(windows):
                        arguments = dict(window['access']['arguments'])
                        if self.fault == 'unknown-selector':
                            arguments['target_id'] = 'ungranted:target'
                        elif self.fault == 'hostile-arguments':
                            arguments['path'] = '../private-source'
                        self.reads[scope].append(arguments)
                        event_id = f'native-source-{index}'
                        yield ToolStartEvent(id=event_id, name='read_source', input=arguments)
                        response = await client.post(reader.url, json={
                            'jsonrpc': '2.0', 'id': index + 1, 'method': 'tools/call',
                            'params': {'name': 'read_source', 'arguments': arguments},
                        })
                        response.raise_for_status()
                        packet = response.json()
                        result = packet['result']
                        text = '\n'.join(block['text'] for block in result['content'] if block['type'] == 'text')
                        is_error = result.get('isError', False)
                        assert is_error is (self.fault in {'unknown-selector', 'hostile-arguments'})
                        if not is_error:
                            source = json.loads(text)
                            assert source['source']['file'] == window['file']
                            assert source['source']['side'] == 'after'
                            assert source['source']['revision'] == stage['analyzed_revision']['head_sha']
                            captured += source['body']
                            if self.fault == 'wrong-bytes':
                                source['body'] = 'unbound source bytes\n' + source['body']
                                text = json.dumps(source)
                        native = ToolResultEvent(
                            event_id, output=text, is_error=is_error or self.fault == 'failed',
                            truncated=self.fault == 'truncated', cancelled=self.fault == 'cancelled',
                        )
                        self.native_results[scope].append(native)
                        yield native
        actual = re.search(r'export const actualPath = ("[^"]+");', captured)
        expected = re.search(r'export const expectedPath = ("[^"]+");', captured)
        if actual is not None and expected is not None:
            values = json.loads(actual[1]), json.loads(expected[1])
            self.decisions.append((scope, *values))
            output['notes'] = 'Navigation was compared with the concrete profile route contract.'
            if scope == 'react' and values[0] != values[1]:
                finding = {'id': 1, 'file': 'App.tsx', 'line': 1, 'severity': 'medium', 'confidence': 'MEDIUM',
                           'description': 'Profile navigation loses the username required by the route',
                           'rationale': 'Navigation omits the username segment.',
                           'evidence': 'App.tsx:1 supplies /profile/; Profile.tsx:1 requires /profile/alex.'}
                output['candidates'] = [{'candidate_id': '', 'file': 'App.tsx', 'line': 1,
                                         'trigger': 'Navigate to the profile for alex',
                                         'consequence': 'The destination cannot identify the profile',
                                         'grounds': finding['evidence'], 'disposition': 'confirmed',
                                         'finding': finding}]
        yield ResultEvent(structured_output=output, continuation=None)


@pytest.mark.parametrize('fault', [
    'none', 'compound-context', 'failed', 'truncated', 'cancelled', 'wrong-bytes',
    'unknown-selector', 'hostile-arguments', 'shell-only',
])
async def test_native_source_keeps_findings_or_reports_honest_incomplete_folded_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    repo = tmp_path / 'native_source'
    route = 'user.$username.tsx'
    before = {
        route: 'export const profileNavigation = "user.$username";\n',
        'App.tsx': 'export const actualPath = "/profile/alex";\n',
        'Profile.tsx': 'export const expectedPath = "/profile/alex";\n',
        'Routes.tsx': 'export const profileRoute = "/profile/$username";\n',
        'README.md': '# Profile navigation\n',
    }
    after = {path: body + '// Changed profile boundary.\n' for path, body in before.items()}
    after['App.tsx'] = 'export const actualPath = "/profile/";\n'
    seed_feature_branch(repo, base=before, feature=after)
    original = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    backend = NativeReaderBackend(repo, fault=fault)
    review.backend = backend
    backend.sandbox = True
    complete = fault in {'none', 'compound-context'}
    assert await review.run(file_config=DaydreamFileConfig(supervisor='off')) == 0
    data = review.load()
    title = 'Profile navigation loses the username required by the route'
    assert [finding['title'] for finding in data['findings']] == ([title] if complete else [])
    assert data['terminal_result']['analysis_state'] == ('complete' if complete else 'incomplete')
    inventory = scopes(data)
    assert inventory['generic']['status'] == 'complete'
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    assert phases['alternatives']['status'] == ('complete' if complete else 'failed')
    if not complete:
        assert phases['alternatives']['reason_codes'] == ['evidence_incomplete']
    if fault in {'unknown-selector', 'hostile-arguments'}:
        assert backend.decisions == []
    else:
        assert set(backend.decisions) == {('react', '/profile/', '/profile/alex'),
                                          ('structure', '/profile/', '/profile/alex')}
    root_trajectory = json.loads((tmp_path / 'trajectory.json').read_text())
    for scope in backend.reads:
        terminal, = stage_ends(review, scope)
        metadata = terminal['metadata']
        assert len(backend.reads[scope]) == (0 if fault == 'shell-only' else 4)
        assert metadata['admitted'] is complete
        assert metadata['fresh_source_reads'] == (4 if complete else 0)
        assert metadata['attempt'] == 1
        expected_starts = 1 if fault == 'shell-only' else 5 if fault == 'compound-context' else 4
        assert metadata['remaining_tool_calls'] == 48 - expected_starts
        assert inventory[scope]['status'] in (('complete',) if complete else ('failed', 'incomplete'))
        if not complete:
            assert inventory[scope]['reason_codes'] == ['evidence_incomplete']
        sibling, = [sub for sub in root_trajectory['extra']['subtrajectories']
                    if sub.get('descriptor') == f'deep-{scope}']
        trace = json.loads((repo / '.daydream' / sibling['sibling_trajectory_ref']).read_text())
        native_results = [result for step in trace['steps']
                          for result in step.get('observation', {}).get('results', [])]
        assert len(native_results) == expected_starts
        raw = native_results[1:] if fault == 'compound-context' else native_results if fault != 'shell-only' else []
        assert [result['content'] for result in raw] == [result.output for result in backend.native_results[scope]]
        if complete:
            sources = [json.loads(result['content']) for result in raw]
            assert {source['source']['source_path'] for source in sources} == set(before) - {'README.md'}
            assert {source['body'] for source in sources} == {after[path] for path in before if path.endswith('.tsx')}
        if fault in {'failed', 'unknown-selector', 'hostile-arguments'}:
            assert all(result['extra']['is_error'] for result in raw)
        if fault in {'truncated', 'cancelled'}:
            assert all(result['extra'][fault] for result in raw)
    assert tracked_source_state(repo) == original
