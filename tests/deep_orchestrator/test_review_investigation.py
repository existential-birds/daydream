"""Bounded review stages through runner.run, real Git and the external Backend seam."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from daydream import clock
from daydream.backends import (
    AgentEvent,
    MaxTurnsError,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from tests.conftest import ExtDir
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.deep_orchestrator.test_review_completion import ReviewRun, record, scopes
from tests.harness.fake_clock import FakeClock, patch_retry_sleep
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.stub_backend import review_stage_state, stage_result
from tests.test_deep_orchestrator import _profile_with_pipeline, _sanctioned_inputs

StageResponse = Callable[[dict[str, Any], dict[str, Any]],
                         Iterable[AgentEvent] | AsyncGenerator[AgentEvent, None] | None]

class StagedBackend(EmptyReviewBackend):
    """A provider that remembers nothing beyond the current request."""

    supports_review_instructions = True
    supports_budget_preamble = False

    def __init__(self, repo: Path) -> None:
        super().__init__(repo, forbid_merge=False, forbid_supervise=False)
        self.merge_echo_records = True
        self.stages: list[dict[str, Any]] = []
        self.stage_response: StageResponse | None = None
        self.scripts: dict[str, tuple[StageResponse, bool]] = {}

    def script(self, scope_id: str = 'python', *, terminal: bool = True) -> Callable[[StageResponse], StageResponse]:
        """Register only provider behavior; the real runner owns admission and lifecycle."""
        def register(response: StageResponse) -> StageResponse:
            self.scripts[scope_id] = (response, terminal)
            return response
        return register

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        stage = review_stage_state(prompt)
        if stage is not None:
            self.stages.append(stage)
            self.calls.append({'cwd': cwd, 'prompt': prompt,
                               'output_schema': args[0] if args else kwargs.get('output_schema'), **kwargs})
            output = stage_result(stage)
            response, terminal = self.scripts.get(stage['scope_id'], (self.stage_response, True))
            events = response(stage, output) if response else None
            if isinstance(events, AsyncGenerator):
                async with aclosing(events):
                    async for event in events:
                        yield event
            else:
                for event in events or ():
                    yield event
            if not terminal:
                return
            yield ResultEvent(structured_output=output, continuation=None)
            return
        kwargs.pop('review_instructions', None)
        kwargs.pop('wall_budget_s', None)
        kwargs.pop('tool_call_budget', None)
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


class InvestigationRun(ReviewRun):
    backend: StagedBackend

    def __init__(self, repo: Path, tmp: Path, patch: pytest.MonkeyPatch) -> None:
        super().__init__(repo, tmp, patch)
        self.backend = StagedBackend(repo)

    async def finish(self, scope_id: str, *, reason: str | None = None, findings: tuple[str, ...] = (),
                     statuses: tuple[str, ...] | None = None, expected_exit: int = 0,
                     **overrides: Any) -> dict[str, Any]:
        assert await self.run(**overrides) == expected_exit
        data = self.load()
        assert [finding['title'] for finding in data['findings']] == list(findings)
        inventory = scopes(data)
        scope = inventory[scope_id]
        assert all(s['status'] == 'complete' for name, s in inventory.items() if name != scope_id)
        assert scope['status'] in (statuses or ('incomplete' if reason else 'complete',)), scope['reason_codes']
        assert scope['reason_codes'] == ([reason] if reason else [])
        assert not any(call.get('tools_disabled') for call in self.backend.calls)
        return data


@pytest.fixture
def investigation(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InvestigationRun:
    return InvestigationRun(multi_stack_target, tmp_path, monkeypatch)


def candidate(*, disposition: str = 'open', candidate_id: str = '',
              finding: dict[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    return {'candidate_id': candidate_id, 'file': 'api.py', 'line': 2,
            'trigger': 'Calling hello()', 'consequence': 'The returned greeting may violate callers.',
            'grounds': "api.py:2 returns 'universe'; current callers accept that greeting.",
            'disposition': disposition, 'finding': finding, **fields}


def read_events(repo: Path, path: str, *, event_id: str) -> Iterable[AgentEvent]:
    """Read real source, retaining the provider's start/result event boundary."""
    source = (repo / path).read_text()
    yield ToolStartEvent(id=event_id, name='Read', input={'file_path': path, 'offset': 1,
                                                       'limit': len(source.splitlines())})
    yield ToolResultEvent(id=event_id, output=source, is_error=False)


@pytest.mark.parametrize(('fault', 'reason'), [('no-tool-read', None),
    ('unknown', 'malformed_output'), ('duplicate', 'malformed_output'),
    ('omitted', 'malformed_output'), ('read-all', 'malformed_output'),
    ('handoff-bytes', 'evidence_incomplete'), ('handoff-items', 'evidence_incomplete'),
    ('missing-grounds', 'evidence_incomplete'), ('invalid-disposition', 'malformed_output'),
    ('truncated-tool-output', None),
    ('unmatched-tool-result', None), ('optional-failures', None)])
async def test_invalid_decisions_fail_but_optional_tool_events_do_not_veto_valid_output(
    investigation: InvestigationRun, fault: str, reason: str) -> None:
    @investigation.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        concrete = candidate(disposition='confirmed', finding=record())
        if fault == 'unknown':
            output['targets'][0]['target_id'] = 'foreign.py'
        elif fault == 'duplicate':
            output['targets'] *= 2
        elif fault in {'omitted', 'read-all'}:
            output['targets'] = []
        elif fault == 'handoff-bytes':
            output['notes'] = 'x' * 70_000
        else:
            output['candidates'] = [concrete] * (129 if fault == 'handoff-items' else 1)
            if fault == 'missing-grounds':
                concrete['grounds'] = ''
            elif fault == 'invalid-disposition':
                concrete['disposition'] = []
        if fault in {'read-all', 'truncated-tool-output'}:
            yield ToolStartEvent(id='read', name='Read', input={'file_path': 'api.py'})
            yield ToolResultEvent(id='read', output=(investigation.repo / 'api.py').read_text(), is_error=False,
                                  truncated=fault == 'truncated-tool-output')
        elif fault == 'unmatched-tool-result':
            yield ToolResultEvent(id='unmatched-source', is_error=False,
                                  output=(investigation.repo / 'api.py').read_text())
        elif fault == 'optional-failures':
            yield ToolStartEvent(id='optional', name='Bash', input={'command': 'rg definitely_absent .'})
            yield ToolResultEvent(id='optional', output='x' * (2 * 1024 * 1024 + 1),
                                  is_error=True, exit_code=1, truncated=True)
            yield ToolResultEvent(id='unassociated', output='failed optional lookup', is_error=True)

    data = await investigation.finish('python', reason=reason, findings=('Grounded defect',) if reason is None else (),
                                      statuses=('complete',) if reason is None else ('incomplete', 'failed'))
    terminal = stage_ends(investigation, 'python')[-1]
    assert (terminal['status'], terminal.get('reason_code')) == (
        ('succeeded', None) if reason is None else ('failed', 'domain_failure'))
    if fault == 'no-tool-read':
        assert data['terminal_result']['analysis_state'] == 'complete'
        assert terminal['metadata']['observed_tool_starts'] == 0
    if fault == 'optional-failures':
        assert terminal['metadata']['admitted'] is True and terminal['metadata']['observed_tool_starts'] == 1


def stage_ends(review: ReviewRun, scope_id: str) -> list[dict[str, Any]]:
    trajectory = json.loads((review.tmp / 'trajectory.json').read_text())
    return [event for sub in trajectory['extra']['subtrajectories'] for event in sub.get('phase_events', [])
            if event['event'] == 'phase_end' and event['metadata'].get('review_scope_id') == scope_id]


def many_file_review(tmp_path: Path, patch: pytest.MonkeyPatch, *, count: int = 17,
                     lines: int = 1, docs_count: int = 4, feature_message: str = 'feature',
                     feature_overrides: dict[str, str] | None = None,
                     supporting_sources: dict[str, str] | None = None) -> InvestigationRun:
    before = {'api.py': "def hello():\n    return 'world'\n"}
    before.update({f'module_{i:02}.py': 'VALUE = 0\n' for i in range(count - 1)})
    before.update({f'00_guide_{i}.md': '# Guide\nExisting behavior\n' for i in range(docs_count)})
    after = {name: content + ''.join(f'# changed {i}\n' for i in range(lines)) for name, content in before.items()}
    after.update(feature_overrides or {})
    before.update(supporting_sources or {})
    after.update(supporting_sources or {})
    repo = tmp_path / 'many_files'
    seed_feature_branch(repo, base=before, feature=after, feature_message=feature_message)
    review = InvestigationRun(repo, tmp_path, patch)
    # These journeys exercise serial stages under one cumulative reviewer allowance.
    review.config = replace(review.config, deep_shard_enabled=False)
    return review


@pytest.mark.parametrize('seeded_defect', [False, True], ids=['clean', 'contradictory-changed-code'])
@pytest.mark.parametrize('transport', ['exact', 'inline', 'native'])
async def test_successful_supporting_dependency_notes_reach_only_relevant_later_assignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seeded_defect: bool, transport: str) -> None:
    """A clean cross-file conclusion survives disjoint batches without deciding later work."""
    review = many_file_review(tmp_path, monkeypatch, count=9, feature_overrides={
        'api.py': "from module_03 import VALUE\n\ndef hello():\n    return 'world' if VALUE == 0 else None\n",
        'module_03.py': f'VALUE = {1 if seeded_defect else 0}\n# Assigned shared dependency\n',
    })
    review.backend.sandbox = transport == 'inline'
    finding = dict(record(1), file='module_03.py', description='Later changed-code defect',
                   evidence='module_03.py:1 changes VALUE to 1, making api.hello return None')
    if transport == 'inline':
        # Existing inline synthesis provider has no private artifact-path access.
        review.backend.merge_echo_records = False
        review.backend.merge_items = (
            [dict(finding, lens='per-stack', source_uids=['python:1'])] if seeded_defect else [])
        review.backend.responder = lambda prompt: [ResultEvent(structured_output={'verdicts': [{
            'id': 1, 'action': 'allow', 'reason': 'Source-supported changed-code defect.',
            'severity': None, 'confidence': None, 'description': None, 'rationale': None, 'evidence': None,
        }]}, continuation=None)] if 'supervisor adjudication' in prompt.lower() else None
    conclusion = 'The shared VALUE declaration is compatible with the first batch consumer.'
    received: list[dict[str, Any]] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if transport == 'native':
            yield RequestEvent(prompt=review.backend.calls[-1]['prompt'],
                               output_schema=stage['response_contract']['schema'],
                               config=PiRequestConfig(schema_emulated=False, no_tools=False))
        if 'api.py' in stage['assigned_files']:
            output.update(notes=conclusion, supporting_files=['./module_03.py'])
        elif 'module_03.py' in stage['assigned_files']:
            received.extend(stage['note_context'])
            assert len(stage['notes']) == 1
            assert stage['notes'] == [conclusion]
            note, = stage['note_context']
            assert note['note_index'] == 0
            assert note['supporting_files'] == ['module_03.py']
            assert note['origin']['stage'] == 'first_pass'
            assert note['origin']['scope_id'] == 'python'
            assert note['origin']['sequence'] == 1
            assert note['origin']['judgment_origin'] == 'reviewer'
            assert note['origin']['analyzed_revision'] == stage['analyzed_revision']
            assert 'module_03.py' not in stage['completed_target_ids']
            assert stage['candidates'] == stage['closed_decisions'] == []
            if seeded_defect:
                assert (review.repo / 'module_03.py').read_text().startswith('VALUE = 1\n')
                output['candidates'] = [candidate(disposition='confirmed', file='module_03.py', line=1,
                    finding=finding, grounds=finding['evidence'])]
        else:
            assert conclusion not in stage['notes']
        if transport == 'native':
            yield ToolStartEvent(id='submit', name='structured_output', input=output)
            yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    data = await review.finish('python', findings=('Later changed-code defect',) if seeded_defect else ())
    assert len(received) == 1
    assert data['terminal_result']['analysis_state'] == 'complete'
    saved = json.loads((review.repo / '.daydream/deep/stack-python-records.json').read_text())
    assert [item['description'] for item in saved['issues']] == ([finding['description']] if seeded_defect else [])
    # Reporting retains source conclusions without changing the original records
    # that downstream merge/adjudication calls receive as sanctioned input.
    assert set(saved) == {'issues', 'scope_id', 'analyzed_revision', 'originating_run_id'}
    sidecar = json.loads((review.repo / '.daydream/deep/stack-python-report-context.json').read_text())
    assert all(sidecar[key] == saved[key] for key in ('scope_id', 'analyzed_revision', 'originating_run_id'))
    context = sidecar['review_context']
    admitted = [event for event in stage_ends(review, 'python') if event['metadata']['admitted']]
    assigned_ids = {target for stage in review.backend.stages if stage['scope_id'] == 'python'
                    for target in stage['assigned_target_ids']}
    assert context['admitted_stages'] == len(admitted)
    assert context['target_decisions'] == {
        'required': len(assigned_ids), 'reviewed': len(assigned_ids), 'not_reviewed': 0}
    assert len(context['notes']) == len(admitted)
    assert context['notes'][0]['text'] == conclusion
    assert context['notes'][0]['origin'] == received[0]['origin']
    assert context['notes'][0]['supporting_files'] == ['module_03.py']
    assert context['notes_omitted'] == 0


@pytest.mark.parametrize('path_kind', ['plain', 'multibyte', 'escaped'])
async def test_persisted_source_assertions_are_bounded_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_kind: str) -> None:
    """Provider stage notes must not become an unbounded or credential-bearing report."""
    prefix = {'plain': '', 'multibyte': ('𐐨' * 43 + '/') * 2,
              'escaped': ('é' * 84 + '/') * 2}[path_kind]
    supporting = [prefix + f'dependency_{index}.py' for index in range(5)] if prefix else []
    count = 65 if path_kind == 'plain' else 33
    review = many_file_review(tmp_path, monkeypatch, count=count, docs_count=0,
                              supporting_sources={path: 'VALUE = 0\n' for path in supporting})

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        output['notes'] = 'Source-only assertion ghp_1234567890 {"token": "private-value"} ' + 'x' * 700
        output['supporting_files'] = supporting

    review.backend.stage_response = response
    data = await review.finish('python', review_cache_enabled=True)
    context_path = review.repo / '.daydream/deep/stack-python-report-context.json'
    original_bytes = context_path.read_bytes()
    assert len(original_bytes) <= 65536, 'producer output must fit the exact reader/cache byte cap'
    saved = json.loads(original_bytes)
    context = saved['review_context']
    admitted = [event for event in stage_ends(review, 'python') if event['metadata']['admitted']]
    assert len(admitted) > 8 and context['admitted_stages'] == len(admitted)
    assert context['target_decisions'] == {'required': count, 'reviewed': count, 'not_reviewed': 0}
    assert 0 < len(context['notes']) <= 8 and context['notes_omitted'] == len(admitted) - len(context['notes'])
    assert len(context['notes']) == 8 if path_kind == 'plain' else len(context['notes']) < 8
    assert all(len(note['text']) <= 512 and note['text'].endswith('[note clipped]') for note in context['notes'])
    assert all('ghp_1234567890' not in note['text'] and 'private-value' not in note['text']
               and '[REDACTED_API_KEY]' in note['text']
               for note in context['notes'])
    python_stages = [stage for stage in review.backend.stages if stage['scope_id'] == 'python']
    for note in context['notes']:
        source_stage = python_stages[note['origin']['sequence'] - 1]
        assert note['files'] == sorted(set(source_stage['assigned_files']) | set(supporting))[:32]
        assert note['supporting_files'] == supporting
        assert note['origin']['analyzed_revision'] == saved['analyzed_revision']
    structural = json.loads((review.repo / '.daydream/deep/stack-structure-report-context.json').read_text())[
        'review_context']
    assert structural['target_decisions'] == {'required': 1, 'reviewed': 1, 'not_reviewed': 0}
    assert len(structural['notes'][0]['files']) == 32
    python = next(scope for scope in data['review_context']['scopes'] if scope['scope_id'] == 'python')
    assert python['review_context'] == context
    assert saved['originating_run_id'] == data['terminal_result']['run_id']
    assert saved['analyzed_revision'] == python['analyzed_revision'] == data['terminal_result']['analyzed_revision']
    stages_before_reuse = len(review.backend.stages)
    reused = await review.finish('python', review_cache_enabled=True)
    assert len(review.backend.stages) == stages_before_reuse, 'optional rationale must not elicit another review'
    assert context_path.read_bytes() == original_bytes
    restored = next(scope for scope in reused['review_context']['scopes'] if scope['scope_id'] == 'python')
    assert restored == python and reused['terminal_result']['run_id'] != data['terminal_result']['run_id']


async def test_optional_supporting_paths_are_confined_without_rejecting_valid_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable references do not veto findings or acquire source/review authority."""
    review = many_file_review(tmp_path, monkeypatch, count=9)
    outside = tmp_path / 'outside.py'
    outside.write_text('PRIVATE_OUTSIDE_SOURCE\n')
    (review.repo / 'linked.py').symlink_to(outside)
    git(review.repo, 'add', 'linked.py')
    git(review.repo, 'commit', '-m', 'Add an outward symlink source control')
    review = InvestigationRun(review.repo, tmp_path, monkeypatch)
    review.config = replace(review.config, deep_shard_enabled=False)
    notes: list[dict[str, Any]] = []
    supporting_reference: dict[str, Any] = {}

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if 'api.py' in stage['assigned_files']:
            supporting_reference.update({
                'artifact_sha256': hashlib.sha256((review.repo / 'module_03.py').read_bytes()).hexdigest(),
                'part_sha256': None, 'revision': stage['analyzed_revision']['head_sha'],
                'path': 'module_03.py', 'side': 'new', 'source_range': {'start': 1, 'end': 1},
                'fragment': None, 'basis': 'source_inspection', 'role': 'supporting',
            })
            output['targets'][0]['evidence_refs'].append(supporting_reference.copy())
            output.update(notes='Supporting paths are optional model declarations.', supporting_files=[
                './module_03.py', 'module_03.py', '../outside.py', str(outside),
                'linked.py', 'missing.py', '.git/config', '.daydream/deep/stack-python-records.json',
                'module_03.py\n', 'module_03.py/../api.py'])
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        elif 'module_03.py' in stage['assigned_files']:
            notes.extend(stage['note_context'])
            assert stage['candidates'] == []
            assert 'module_03.py' not in stage['completed_target_ids']

    data = await review.finish('python', findings=('Grounded defect',))
    note, = notes
    assert note['supporting_files'] == ['module_03.py']
    assert 'api.py' in note['files']
    decisions = scopes(data)['python']['target_decisions']
    api_decision = next(decision for decision in decisions if decision['target_id'] == 'api.py')
    assert supporting_reference in api_decision['evidence_refs']
    assert any(reference['role'] == 'assigned' for reference in api_decision['evidence_refs'])
    assert all('PRIVATE_OUTSIDE_SOURCE' not in call['prompt'] for call in review.backend.calls)


@pytest.mark.parametrize(('references', 'native'), [
    ('missing', False), (None, True), ('unusable-container', False), ([None, 42, '../outside.py'], True)])
async def test_unavailable_optional_reference_metadata_preserves_primary_stage_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, references: Any, native: bool) -> None:
    """Legacy/malformed optional metadata cannot erase valid findings or trigger fresh stages."""
    review = many_file_review(tmp_path, monkeypatch, count=9)

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if native:
            yield RequestEvent(prompt=review.backend.calls[-1]['prompt'],
                               output_schema=stage['response_contract']['schema'],
                               config=PiRequestConfig(schema_emulated=False, no_tools=False))
        if 'api.py' in stage['assigned_files']:
            output['notes'] = 'Optional association is unavailable.'
            if references == 'missing':
                output.pop('supporting_files')
            else:
                output['supporting_files'] = references
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        else:
            assert 'Optional association is unavailable.' not in stage['notes']
            assert stage['note_context'] == []
        if native:
            yield ToolStartEvent(id='submit', name='structured_output', input=output)
            yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    await review.finish('python', findings=('Grounded defect',))
    ends = stage_ends(review, 'python')
    assert len(ends) == 3 and all(event['metadata']['admitted'] for event in ends)
    assert all(event['metadata']['attempt'] == 1 for event in ends)


async def test_supporting_relevance_metadata_counts_toward_the_semantic_handoff_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=129)

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        output['supporting_files'] = ['api.py', *[f'module_{i:02}.py' for i in range(128)]]

    await review.finish('python', reason='evidence_incomplete')
    ends = stage_ends(review, 'python')
    assert len(ends) == 1 and ends[0]['metadata']['admitted'] is False
    assert ends[0]['metadata']['failure_class'] == 'admission_failure'


@pytest.mark.parametrize('stop', ['wall', 'pipeline', 'outage-circuit', 'permanent'])
async def test_isolated_failure_continuation_respects_shared_deadline_and_outage_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=9)
    review.backend.fanout_concurrency = 1
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')
    timer = FakeClock(monotonic_value=clock.monotonic()).install(monkeypatch)
    patch_retry_sleep(monkeypatch, timer)
    monkeypatch.setattr('daydream.config.RETRY_CIRCUIT_FAILURE_THRESHOLD', 1)
    attempts: list[list[str]] = []

    class TemporaryFailure(RuntimeError):
        retryable = True

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if 'api.py' in stage['assigned_files']:
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        elif 'module_03.py' in stage['assigned_files']:
            attempts.append(stage['assigned_files'])
            if stop in {'wall', 'pipeline'}:
                timer.advance(481 if stop == 'wall' else 61)
                raise RuntimeError('isolated failure after the shared deadline')
            if stop == 'permanent':
                raise RuntimeError('Invalid API key')
            raise TemporaryFailure('isolated provider outage')
        else:
            pytest.fail('Later assignment dispatched despite a shared stopping condition')

    assert await review.run(review_profile=_profile_with_pipeline(
        reviewer_investigation_budget_s=480,
        review_wall_budget_s=1000 if stop == 'wall' else 60)) == 0
    data = review.load()
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    expected_reasons = ['backend_failure']
    if stop in {'wall', 'pipeline'}:
        expected_reasons.append(f'host_{stop}_budget_exhaustion')
    assert scopes(data)['python']['reason_codes'] == expected_reasons
    assert len(attempts) == (2 if stop == 'outage-circuit' else 1)
    assert scopes(data)['python']['partial_evidence'] is True
    assert [event['metadata']['admitted'] for event in stage_ends(review, 'python')] == [True, False]
    coverage = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    diagnostic = coverage['diagnostics']['scopes']['python']
    assert 'review invocation failed (backend_failure)' in diagnostic
    if stop in {'wall', 'pipeline'}:
        assert f'{stop}_budget_exceeded' in diagnostic
    elif stop == 'outage-circuit':
        assert 'provider outage circuit denied a later batch' in diagnostic


async def test_isolated_first_pass_and_triage_failures_preserve_later_independent_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed batches cannot discard prior findings, suppress later batches or decide failed triage."""
    review = many_file_review(tmp_path, monkeypatch, count=9)

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['stage'] == 'triage':
            if stage['assigned_candidate_ids'][0] == 'python:candidate:2':
                raise RuntimeError('first triage group unavailable')
            assert stage['closed_candidate_ids'] == ['python:candidate:1', 'python:candidate:11']
            output['candidates'] = [dict(item, disposition='rejected') for item in stage['candidates']]
        elif 'api.py' in stage['assigned_files']:
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            output['candidates'] += [candidate() for _ in range(9)]
        elif 'module_03.py' in stage['assigned_files']:
            output['notes'] = 'FAILED_STAGE_POISON'
            raise RuntimeError('middle first-pass batch unavailable')
        else:
            assert all('FAILED_STAGE_POISON' not in note for note in stage['notes'])
            assert not set(stage['completed_target_ids']).intersection(
                {'module_03.py', 'module_04.py', 'module_05.py', 'module_06.py'})
            finding = dict(record(1), file='module_07.py', description='Later admitted defect',
                           evidence='module_07.py:1')
            output['candidates'] = [candidate(disposition='confirmed', finding=finding,
                                             file='module_07.py', line=1, grounds=finding['evidence'])]

    data = await review.finish('python', reason='backend_failure',
                               findings=('Grounded defect', 'Later admitted defect'))
    stages = [stage for stage in review.backend.stages if stage['scope_id'] == 'python']
    assert [stage['stage'] for stage in stages] == ['first_pass'] * 3 + ['triage'] * 2
    assert [len(stage['assigned_candidate_ids']) for stage in stages[-2:]] == [8, 1]
    ends = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in ends] == [True, False, True, False, True]
    assert ends[-1]['metadata']['retained_findings'] == 2
    assert ends[-1]['metadata']['admitted_candidate_count'] == 11
    assert scopes(data)['python']['partial_evidence'] is True
    assert data['terminal_result']['analysis_state'] == 'incomplete'
    coverage = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    assert 'RuntimeError: review invocation failed (backend_failure).' in coverage['diagnostics']['scopes']['python']


@pytest.mark.parametrize('failed_stage', ['first_pass', 'triage'])
@pytest.mark.parametrize(('stop', 'reason', 'diagnostic'), [
    ('auth', 'authentication_failure', 'AuthenticationFailure: review invocation failed'),
    ('admission', 'evidence_incomplete', 'Stage evidence admission failed'),
    ('wall', 'host_wall_budget_exhaustion', 'wall_budget_exceeded'),
    ('pipeline', 'host_pipeline_budget_exhaustion', 'pipeline_budget_exceeded'),
    ('tools', 'host_tool_budget_exhaustion', 'tool_call_budget_exceeded'),
])
async def test_later_stopping_failure_retains_original_failure_and_stopping_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_stage: str,
    stop: str, reason: str, diagnostic: str,
) -> None:
    """Saved coverage must explain both lost work and why no further batch was admitted."""
    review = many_file_review(tmp_path, monkeypatch, count=13, docs_count=0)
    review.backend.fanout_concurrency = 1
    timer = FakeClock(monotonic_value=clock.monotonic()).install(monkeypatch)
    patch_retry_sleep(monkeypatch, timer)
    attempts = 0

    class AuthenticationFailure(RuntimeError):
        category = 'AUTH_CONFIG'

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal attempts
        if stage['stage'] == 'first_pass' and 'api.py' in stage['assigned_files']:
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            if failed_stage == 'triage':
                output['candidates'] += [candidate() for _ in range(17)]
            return
        if stage['stage'] != failed_stage:
            return
        attempts += 1
        if attempts == 1:
            raise RuntimeError('PRIVATE_INITIAL_PROVIDER_DETAIL')
        assert attempts == 2, 'Dispatch continued after a stopping failure'
        if stop == 'auth':
            raise AuthenticationFailure('PRIVATE_AUTH_PROVIDER_DETAIL')
        if stop == 'admission':
            if stage['stage'] == 'triage':
                output['candidates'] = [dict(item, disposition='rejected') for item in stage['candidates']]
            output['notes'] = 'x' * (64 * 1024)
        elif stop in {'wall', 'pipeline'}:
            timer.advance(481 if stop == 'wall' else 61)
        else:
            for index in range(50):
                yield from read_events(review.repo, 'api.py', event_id=f'exhaust-{index}')

    assert await review.run(review_profile=_profile_with_pipeline(
        reviewer_investigation_budget_s=480,
        review_wall_budget_s=1000 if stop == 'wall' else 60)) == 0
    data = review.load()
    assert attempts == 2
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    scope = scopes(data)['python']
    assert scope['status'] == 'incomplete' and scope['partial_evidence'] is True
    assert scope['reason_codes'] == sorted(['backend_failure', reason])
    assert set(data['terminal_result']['reason_codes']) >= {'backend_failure', reason}
    coverage = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    saved_diagnostic = coverage['diagnostics']['scopes']['python']
    assert 'RuntimeError: review invocation failed (backend_failure).' in saved_diagnostic
    assert diagnostic in saved_diagnostic
    assert 'PRIVATE_' not in saved_diagnostic
    assert stage_ends(review, 'python')[-1]['metadata']['admitted'] is False


@pytest.mark.parametrize('stop', ['wall', 'pipeline', 'outage-circuit', 'tools'])
async def test_unadmitted_backend_failure_remains_failed_after_a_shared_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str,
) -> None:
    """A refused later batch cannot relabel attempted failed work as unattempted coverage."""
    review = many_file_review(tmp_path, monkeypatch, count=9, docs_count=0)
    review.backend.fanout_concurrency = 1
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')
    monkeypatch.setattr('daydream.config.RETRY_CIRCUIT_FAILURE_THRESHOLD', 1)
    timer = FakeClock(monotonic_value=clock.monotonic()).install(monkeypatch)
    patch_retry_sleep(monkeypatch, timer)
    attempts = 0

    class TemporaryFailure(RuntimeError):
        retryable = True

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal attempts
        attempts += 1
        if stop in {'wall', 'pipeline'}:
            timer.advance(481 if stop == 'wall' else 61)
        if stop == 'outage-circuit':
            raise TemporaryFailure('PRIVATE_PROVIDER_DETAIL')
        if attempts == 1:
            raise RuntimeError('PRIVATE_PROVIDER_DETAIL')
        assert stop == 'tools' and attempts == 2
        for index in range(50):
            yield from read_events(review.repo, 'api.py', event_id=f'exhaust-{index}')

    assert await review.run(review_profile=_profile_with_pipeline(
        reviewer_investigation_budget_s=480,
        review_wall_budget_s=1000 if stop == 'wall' else 60)) == 0
    data = review.load()
    assert attempts == (2 if stop in {'outage-circuit', 'tools'} else 1)
    scope = scopes(data)['python']
    assert scope['status'] == 'failed' and scope['partial_evidence'] is False
    expected_reasons = ['backend_failure']
    if stop != 'outage-circuit':
        expected_reasons.append(f"host_{'tool' if stop == 'tools' else stop}_budget_exhaustion")
    assert scope['reason_codes'] == expected_reasons
    assert data['findings'] == []
    assert all(event['metadata']['admitted'] is False for event in stage_ends(review, 'python'))


@pytest.mark.parametrize('unfinished_target', [False, True], ids=['complete', 'unfinished-first-target'])
async def test_large_unsharded_scope_retains_coverage_without_spending_semantic_handoff_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unfinished_target: bool) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=129)

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if unfinished_target and 'api.py' in stage['assigned_files']:
            target = next(target for target in output['targets'] if target['target_id'] == 'api.py')
            target.update(status='not_reviewed', reason='The changed greeting contract remains unverified.')

    data = await review.finish('python', reason='evidence_incomplete' if unfinished_target else None,
                               deep_shard_enabled=False)
    stages = [stage for stage in review.backend.stages if stage['scope_id'] == 'python']
    assigned = [target for stage in stages for target in stage['assigned_target_ids']]
    scope = scopes(data)['python']
    assert len(scope['files']) == 129
    assert set(scope['files']) == {path.name for path in review.repo.glob('*.py')}
    assert set(assigned) == set(scope['files']) and len(assigned) == len(set(assigned))
    assert {decision['target_id'] for decision in scope['target_decisions']} == set(assigned)
    assert all(decision['evidence_refs'] for decision in scope['target_decisions'])
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))
    assert data['terminal_result']['analysis_state'] == ('incomplete' if unfinished_target else 'complete')
    assert scope['partial_evidence'] is unfinished_target


@pytest.mark.parametrize('scope_id', ['python', 'generic', 'structure'])
async def test_useful_completed_reads_borrow_cumulative_capacity_and_allow_later_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope_id: str) -> None:
    review = many_file_review(tmp_path, monkeypatch, docs_count=9)
    stage_spend: list[int] = []
    request_prompts: list[str] = []

    @review.backend.script(scope_id)
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        prompt = review.backend.calls[-1]['prompt']
        request_prompts.append(prompt)
        yield RequestEvent(prompt=prompt, output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        assert 'Host review stage budget fields are dispatch-time snapshots:' in prompt
        assert 'Reviewer dispatch-time snapshot: hard wall allowance:' in prompt
        assert prompt.count('current invocation allocation:') == 1
        assert 'Cumulative observed starts:' not in prompt
        assert 'Remaining hard cumulative tool allowance:' not in prompt
        assert 'Advisory stage tool-call target:' not in prompt
        assert 'advisory_tool_call_target guides pace rather than limiting useful assigned checks' in prompt
        assert 'submit when those checks settle. Unused capacity is expected.' in ' '.join(prompt.split())
        assert 'At zero remaining calls, capacity is exhausted' in prompt
        initial = not stage_spend
        starts = stage['advisory_tool_call_target'] + 1 if initial else 1
        stage_spend.append(starts + 1)
        assert stage['observed_tool_starts'] == sum(stage_spend[:-1])
        assert stage['remaining_tool_calls'] == 48 - sum(stage_spend[:-1])
        path = stage['assigned_files'][0]
        for index in range(starts):
            yield from read_events(review.repo, path, event_id=f"stage-{len(stage_spend)}-read-{index}")
        if initial:
            line = 3 if scope_id == 'generic' else 2
            finding = dict(record(line), file=path, evidence=f'{path}:{line}')
            output['candidates'] = [candidate(disposition='confirmed', finding=finding,
                                             file=path, line=line, grounds=finding['evidence'])]
            if scope_id == 'structure':
                output['candidates'].append(candidate(file=path, line=line))
        elif stage['stage'] == 'triage':
            output['candidates'] = [dict(item, disposition='rejected') for item in stage['candidates']]
        yield ToolStartEvent(id='submit', name='structured_output', input=output)
        yield ToolResultEvent(id='submit', output='Submitted.', is_error=False)

    await review.finish(scope_id, findings=('Grounded defect',))
    stages = [stage for stage in review.backend.stages if stage['scope_id'] == scope_id]
    metadata = [event['metadata'] for event in stage_ends(review, scope_id)]
    assert len(stages) > 1 and stage_spend[0] - 1 > stages[0]['advisory_tool_call_target']
    assert [item['observed_tool_starts'] for item in metadata] == [
        sum(stage_spend[:index + 1]) for index in range(len(stage_spend))]
    assert metadata[-1]['remaining_tool_calls'] == 48 - sum(stage_spend)
    assert all(item['admitted'] for item in metadata)

    assert metadata[-1]['remaining_tool_calls'] > 0
    assert len(request_prompts) == len(stages)

@pytest.mark.parametrize(('sandbox', 'long_grounds'), [(False, False), (True, False), (False, True)],
                         ids=['exact-path-inputs', 'inline-inputs', 'clipped-closed-conclusion'])
async def test_stage_builders_preserve_transport_and_bounded_semantic_handoffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, sandbox: bool, long_grounds: bool) -> None:
    ext_dir.write_module(
        "import json\n"
        "from daydream.deep.prompts import build_per_stack_prompt, build_structural_prompt\n"
        "def wrap(builder, kw):\n"
        "    state = kw['review_stage']\n"
        "    assert isinstance(' '.join(state['notes']), str)\n"
        "    assert kw['files'] == state['assigned_files']\n"
        "    return builder(**kw) + '\\nBuilder stage arguments:\\n' + json.dumps({\n"
        "        'files': kw['files'], 'diff': kw.get('inline_diff'), 'stage': state['stage']})\n"
        "def register(r):\n"
        "    r.override_prompt('per-stack', lambda **kw: wrap(build_per_stack_prompt, kw))\n"
        "    r.override_prompt('structural', lambda **kw: wrap(build_structural_prompt, kw))\n"
    )
    review = many_file_review(tmp_path, monkeypatch, count=9,
                             feature_message='fix: SETTLED_UNRELATED_HISTORY\n\nDaydream-Run: previous-fixture')
    review.backend.sandbox = sandbox
    if sandbox:
        review.backend.merge_echo_records = False
        review.backend.merge_items = [dict(record(), lens='per-stack', source_uids=['python:1'])]
    later_assignments: list[dict[str, Any]] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        call = review.backend.calls[-1]
        prose = call['prompt'].split('Host review stage:\n')[0]
        assert set(call['output_schema']['properties']) == {
            'targets', 'notes', 'supporting_files', 'candidates', 'contradictions'}
        assert 'with issues' not in prose
        assert 'AI-generated code' not in prose
        assert 'Be strict' not in prose
        assert 'Host review stage budget fields are dispatch-time snapshots:' in prose
        assert 'remaining_tool_calls is the reviewer' in prose
        assert 'hard_tool_call_allowance is the current invocation allocation' in prose
        assert 'advisory_tool_call_target guides pace' in prose
        assert stage['hard_tool_call_allowance'] <= stage['remaining_tool_calls']
        assert stage['advisory_tool_call_target'] <= stage['hard_tool_call_allowance']
        assert 'Error Handling Semantics (QUAL-04)' in prose
        assert 'Confidence and Convention Rules' in prose
        assert 'HIGH: directly verified by the supplied change, ordinary investigation, or Exploration Context' in prose
        if stage['stage'] == 'first_pass':
            assert 'SETTLED_UNRELATED_HISTORY' in prose
            assert len(stage['assigned_files']) <= 4
            assert stage['context_transport'] == ('inline' if sandbox else 'exact_paths')
            if sandbox:
                assert 'Sanctioned phase inputs (read only these exact files):' not in call['prompt']
                assert str(review.repo / '.daydream') not in call['prompt']
                assert 'Sanctioned phase inputs (captured verbatim):' in call['prompt']
            else:
                pointers = _sanctioned_inputs(call['prompt'])
                assert set(pointers) == set(stage['context_inputs'])
                for label in ('diff', 'hunk-index'):
                    assert pointers[label].is_file() and pointers[label].read_text()
            assigned_line = next(line for line in prose.splitlines() if 'Assigned files:' in line)
            assert assigned_line.split('Assigned files:', 1)[1].strip() == ', '.join(stage['assigned_files'])
            assert stage['candidates'] == []
            if 'api.py' in stage['assigned_files']:
                confirmed = candidate(disposition='confirmed', finding=record())
                if long_grounds:
                    confirmed['grounds'] += ' Additional explanation: ' + 'é' * 400
                rejected = candidate(disposition='rejected',
                                     grounds='The greeting is allowed by the updated consumer contract.')
                output['candidates'] = [confirmed, rejected] + [candidate() for _ in range(9)]
            if stage['progress']:
                later_assignments.append(stage)
                decisions = stage['closed_decisions']
                assert [(decision['candidate_id'], decision['file'], decision['line'], decision['disposition'])
                        for decision in decisions] == [('python:candidate:1', 'api.py', 2, 'confirmed'),
                                                      ('python:candidate:2', 'api.py', 2, 'rejected')]
                assert decisions[0]['conclusion'].startswith('Grounded defect: ' + candidate()['grounds'])
                assert len(decisions[0]['conclusion'].encode()) <= 512
                assert decisions[0]['conclusion'].endswith('[conclusion clipped]') is long_grounds
                assert decisions[1]['conclusion'] == 'The greeting is allowed by the updated consumer contract.'
                assert all('evidence_references' not in decision for decision in decisions)
                assert all(decision['judgment_origin'] == 'reviewer' for decision in decisions)
                assert stage['candidates'] == []
                instruction = review.backend.calls[-1]['review_instructions']
                assert 'The greeting is allowed by the updated consumer contract.' in instruction
                assert 'Host-retained receipt authority' not in instruction
                policy = ' '.join(instruction.split())
                assert 'Host admission validates the submission contract, not the correctness' in policy
                assert 'Source evidence can disprove the original reasoning without a changed premise' in policy
        else:
            assert stage['stage'] == 'triage'
            assert 'SETTLED_UNRELATED_HISTORY' not in prose
            assert [item['candidate_id'] for item in stage['candidates']] == stage['assigned_candidate_ids']
            assert all(item['disposition'] == 'open' for item in stage['candidates'])
            assert 'Assigned files:' not in prose
            assert stage['assigned_files'] == ['api.py']
            assert all(item['grounds'] and item['trigger'] and item['consequence'] for item in stage['candidates'])
            assert not any(label.startswith('source-') for label in stage['context_inputs'])
            assert 'Sanctioned phase inputs (read only these exact files):' not in call['prompt']
            output['candidates'] = [dict(item, disposition='rejected') for item in stage['candidates']]

    await review.finish('python', findings=('Grounded defect',))
    assert len(later_assignments) == 2
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))
    stages = [stage for stage in review.backend.stages if stage['scope_id'] == 'python']
    assert len([stage for stage in stages if stage['stage'] == 'first_pass']) > 1
    assert [stage['stage'] for stage in stages].count('triage') == 2
    assert [len(stage['assigned_candidate_ids']) for stage in stages if stage['stage'] == 'triage'] == [8, 1]


@pytest.mark.parametrize(('stop', 'reason'), [('tool', 'host_tool_budget_exhaustion'),
    ('model', 'model_budget_exhaustion'), ('backend', 'backend_failure'), ('builder', 'backend_failure'),
    ('deadline', 'host_pipeline_budget_exhaustion')])
async def test_unsuccessful_later_stage_retains_only_prior_admitted_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, stop: str, reason: str) -> None:
    if stop == 'builder':
        ext_dir.write_module(
            "from daydream.deep.prompts import build_per_stack_prompt\n"
            "def scoped(**kw):\n"
            "    if kw['review_stage']['progress']:\n"
            "        raise RuntimeError('stage builder unavailable')\n"
            "    return build_per_stack_prompt(**kw)\n"
            "def register(r): r.override_prompt('per-stack', scoped)\n"
        )
    review = many_file_review(tmp_path, monkeypatch, count=5 if stop == 'deadline' else 17)
    backend = review.backend
    dispatch_deadlines: list[float] = []
    deadline_test_wall_s = 60
    timer = FakeClock(monotonic_value=clock.monotonic())
    if stop == 'deadline':
        timer.install(monkeypatch)
    no_tool_turns = 0

    @backend.script()
    async def response(stage: dict[str, Any], output: dict[str, Any]) -> AsyncGenerator[AgentEvent, None]:
        nonlocal no_tool_turns
        prompt = backend.calls[-1]['prompt']
        if stop == 'deadline':
            allowance = re.search(r'Reviewer dispatch-time snapshot: hard wall allowance: ([\d.e+-]+) seconds', prompt)
            assert allowance is not None
            dispatch_deadlines.append(timer.monotonic() + float(allowance[1]))
        yield RequestEvent(prompt=prompt, output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        if not stage['progress']:
            for path in stage['assigned_files']:
                for event in read_events(review.repo, path, event_id=f'admitted-{path}'):
                    yield event
            assert "return 'world'" in (review.repo / 'api.py').read_text()
            output['candidates'] = [candidate(disposition='confirmed', finding=record(),
                                             grounds="api.py:1-2 defines hello() and returns 'world'.")]
            yield ToolStartEvent(id='admitted-submit', name='structured_output', input=output)
            yield ToolResultEvent(id='admitted-submit', output='Submitted.', is_error=False)
            return
        if stop == 'deadline':
            output['candidates'] = [candidate(disposition='confirmed', finding=dict(
                record(), description='Unadmitted prose checkpoint'))]
            while True:
                no_tool_turns += 1
                yield TextEvent(text=json.dumps(output))
                yield TurnEndEvent()
                timer.advance(1)
                await asyncio.sleep(0)
        yield ToolStartEvent(id='later-submit', name='structured_output', input=output)
        yield ToolResultEvent(id='later-submit', output='Submitted.', is_error=False)
        yield ResultEvent(structured_output=stage_result(stage, candidates=[
            candidate(disposition='confirmed', finding=dict(record(), description='Unadmitted defect'))]),
            continuation=None)
        if stop == 'tool':
            for i in range(stage['remaining_tool_calls']):
                yield ToolStartEvent(id=f'cutoff-{i}', name='structured_output', input={})
                yield ToolResultEvent(id=f'cutoff-{i}', output='Validation failed', is_error=True)
        else:
            raise MaxTurnsError('model exhausted') if stop == 'model' else RuntimeError('backend unavailable')

    overrides: dict[str, Any] = (
        {'review_profile': _profile_with_pipeline(review_wall_budget_s=deadline_test_wall_s)}
        if stop == 'deadline' else {}
    )
    data = await review.finish('python', reason=reason, findings=('Grounded defect',), **overrides)
    admitted, *stops = stage_ends(review, 'python')
    stopped = stops[0]
    assert admitted['status'] == 'succeeded'
    assert admitted['metadata']['observed_tool_starts'] > 0
    assert stopped['metadata']['observed_tool_starts'] == (
        49 if stop == 'tool' else admitted['metadata']['observed_tool_starts']
        + (0 if stop in {'builder', 'deadline'} else 1))
    assert scopes(data)['python']['partial_evidence'] is True
    assert all((event['status'], event.get('reason_code')) == (
        ('timed_out', 'timed_out') if stop == 'deadline' else ('failed', 'domain_failure')) for event in stops)
    assert len(stops) == (4 if stop in {'builder', 'backend'} else 1)
    assert [stage['stage'] for stage in backend.stages if stage['scope_id'] == 'python'] == (
        ['first_pass'] if stop == 'builder' else ['first_pass'] * (5 if stop == 'backend' else 2))
    if stop == 'deadline':
        assert no_tool_turns > 1
        assert stopped['metadata']['native_output'] is True
        assert stopped['metadata']['submission_starts'] == 0
        assert stopped['metadata']['admitted'] is False
        assert len(dispatch_deadlines) == 2
        assert dispatch_deadlines[1] == pytest.approx(dispatch_deadlines[0], abs=0.01)
        assert timer.monotonic() >= dispatch_deadlines[0]


@pytest.mark.parametrize(('local_s', 'pipeline_s', 'reason', 'bound'), [
    (480, 2000, 'host_wall_budget_exhaustion', 'local'),
    (900, 2000, None, 'local'),
    (900, 600, 'host_pipeline_budget_exhaustion', 'pipeline'),
])
async def test_configured_investigation_time_changes_completion_without_renewing_stage_deadlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_s: int, pipeline_s: int,
    reason: str | None, bound: str,
) -> None:
    """The same two-stage work distinguishes local configuration from the shared pipeline cap."""
    review = many_file_review(tmp_path, monkeypatch, count=5)
    review.backend.fanout_concurrency = 1
    timer = FakeClock(monotonic_value=clock.monotonic()).install(monkeypatch)
    deadlines: list[float] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        allowance = re.search(r'Reviewer dispatch-time snapshot: hard wall allowance: ([\d.e+-]+) seconds',
                              review.backend.calls[-1]['prompt'])
        assert allowance is not None
        deadlines.append(timer.monotonic() + float(allowance[1]))
        if not stage['progress']:
            assert stage['remaining_tool_calls'] == 48
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        else:
            assert stage['closed_decisions'][0]['conclusion'].startswith('Grounded defect: ')
            assert stage['remaining_tool_calls'] < 48
        yield from read_events(review.repo, stage['assigned_files'][0], event_id=f'read-{len(deadlines)}')
        timer.advance(300)
        yield TextEvent(text='Assigned source investigation completed.')

    assert await review.run(review_profile=_profile_with_pipeline(
        reviewer_investigation_budget_s=local_s, review_wall_budget_s=pipeline_s)) == 0
    data = review.load()
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    scope = scopes(data)['python']
    assert scope['status'] == ('incomplete' if reason else 'complete')
    assert scope['reason_codes'] == ([reason] if reason else [])
    assert scope['partial_evidence'] is bool(reason)
    assert len(deadlines) == 2
    assert deadlines[1] == pytest.approx(deadlines[0], abs=0.01)
    if bound == 'local':
        assert all(scope['status'] == 'complete' for name, scope in scopes(data).items() if name != 'python')
    events = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in events] == [True, reason is None]
    for event in events:
        metadata = event['metadata']
        assert metadata['investigation_base_budget_s'] == metadata['investigation_scaled_budget_s'] == local_s
        assert metadata['investigation_limit'] == (
            'reviewer_investigation_budget' if bound == 'local' else 'pipeline_budget')
        assert metadata['investigation_effective_allowance_s'] == pytest.approx(
            local_s if bound == 'local' else pipeline_s * 0.8)
    saved = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    assert next(item for item in saved['stack_outcomes'] if item['scope_id'] == 'python') == scope
    if reason:
        assert ('pipeline_budget_exceeded' if bound == 'pipeline' else 'wall_budget_exceeded') in (
            saved['diagnostics']['scopes']['python'])
        assert events[-1]['metadata']['retained_findings'] == 1


async def test_zero_configured_investigation_time_dispatches_no_review_and_reports_incomplete_coverage(
    investigation: InvestigationRun, ext_dir: ExtDir,
) -> None:
    """Zero disables staged investigation rather than silently restoring its default allowance."""
    ext_dir.write_module(
        "def forbidden(**kw): raise AssertionError('Zero allowance dispatched a review prompt builder')\n"
        "def register(r): r.override_prompt('per-stack', forbidden)\n"
    )
    assert await investigation.run(review_profile=_profile_with_pipeline(reviewer_investigation_budget_s=0)) == 0
    data = investigation.load()
    assert investigation.backend.stages == []
    assert data['findings'] == [] and data['review_warnings']
    assert data['terminal_result']['analysis_state'] == 'failed'
    assert all(scope['status'] == 'incomplete' and scope['partial_evidence'] is False
               and scope['reason_codes'] == ['host_wall_budget_exhaustion'] for scope in scopes(data).values())
    assert 'Review incomplete' in (investigation.repo / '.review-output.md').read_text()


async def test_stage_preparation_cannot_renew_the_original_invocation_wall_clamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir,
) -> None:
    """An extension's preparation time spends the same invocation cap as dispatched investigation."""
    review = many_file_review(tmp_path, monkeypatch, count=5)
    review.backend.fanout_concurrency = 1
    elapsed = tmp_path / 'preparation-elapsed.txt'
    elapsed.write_text('0')
    timer = FakeClock(monotonic_value=clock.monotonic())
    monkeypatch.setattr(clock, 'monotonic', lambda: timer.monotonic() + float(elapsed.read_text()))
    ext_dir.write_module(
        "from pathlib import Path\n"
        "from daydream.deep.prompts import build_per_stack_prompt\n"
        "def scoped(**kw):\n"
        "    if kw['review_stage']['scope_id'] == 'python':\n"
        f"        elapsed = Path({str(elapsed)!r})\n"
        "        elapsed.write_text(str(float(elapsed.read_text()) + 2000))\n"
        "    return build_per_stack_prompt(**kw)\n"
        "def register(r): r.override_prompt('per-stack', scoped)\n"
    )

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        assert stage['progress'] == [], 'A later stage dispatched after preparation spent the original cap'
        assert 'hard wall allowance: 1600 seconds' in review.backend.calls[-1]['prompt']
        yield from read_events(review.repo, 'api.py', event_id='admitted-read')
        output['candidates'] = [candidate(disposition='confirmed', finding=record())]

    data = await review.finish('python', reason='host_wall_budget_exhaustion', findings=('Grounded defect',),
                              review_profile=_profile_with_pipeline(
                                  reviewer_investigation_budget_s=10000, review_wall_budget_s=10000))
    assert float(elapsed.read_text()) == 4000
    assert len([stage for stage in review.backend.stages if stage['scope_id'] == 'python']) == 1
    events = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in events] == [True, False]
    for event in events:
        assert event['metadata']['investigation_effective_allowance_s'] == pytest.approx(3600, rel=0, abs=0.01)
        assert event['metadata']['investigation_limit'] == 'invocation_wall_budget'
    assert events[-1]['metadata']['retained_findings'] == 1
    assert scopes(data)['python']['partial_evidence'] is True
    saved = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    assert 'wall_budget_exceeded' in saved['diagnostics']['scopes']['python']


@pytest.mark.parametrize('decision', ['rejected', 'confirmed', 'unresolved'])
async def test_unresolved_discovery_reaches_triage_with_spare_capacity_and_retains_confirmed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=9)

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['stage'] == 'first_pass':
            if not stage['progress']:
                output['candidates'] = [candidate(disposition='confirmed', finding=record()),
                    candidate(disposition='unresolved', file='module_00.py', line=1,
                              grounds='module_00.py:1 defines VALUE, but its consumer contract remains unverified.')]
            yield from read_events(review.repo, stage['assigned_files'][0],
                                   event_id=f"discovery-{len(stage['progress'])}")
            return
        pending, = stage['candidates']
        assert pending['disposition'] == 'unresolved'
        assert stage['assigned_candidate_ids'] == [pending['candidate_id']]
        assert len(stage['completed_target_ids']) == 9
        assert stage['remaining_tool_calls'] > 0
        assert [item['disposition'] for item in stage['closed_decisions']] == ['confirmed']
        finding = dict(record(1), file='module_00.py', description='Resolved concern',
                       evidence='module_00.py:1') if decision == 'confirmed' else None
        output['candidates'] = [dict(pending, disposition=decision, finding=finding)]

    data = await review.finish('python', reason='evidence_incomplete' if decision == 'unresolved' else None,
                              findings=('Grounded defect', 'Resolved concern') if decision == 'confirmed'
                              else ('Grounded defect',))
    stages = [stage for stage in review.backend.stages if stage['scope_id'] == 'python']
    assert [stage['stage'] for stage in stages] == ['first_pass'] * 3 + ['triage']
    events = stage_ends(review, 'python')
    assert all(event['metadata']['admitted'] for event in events)
    assert events[-1]['metadata']['remaining_tool_calls'] > 0
    assert events[-1]['metadata']['retained_findings'] == (2 if decision == 'confirmed' else 1)
    assert data['terminal_result']['analysis_state'] == ('incomplete' if decision == 'unresolved' else 'complete')
    assert scopes(data)['python']['partial_evidence'] is (decision == 'unresolved')


@pytest.mark.parametrize(('decision', 'reason'), [
    ('rejected', None), ('confirmed', None), ('missing-finding', 'malformed_output'),
    ('unresolved', 'evidence_incomplete'), ('unknown-contradiction', 'malformed_output'),
    ('contradiction', 'evidence_incomplete'), ('unknown', 'malformed_output'),
    ('duplicate', 'malformed_output'), ('omitted', 'malformed_output'), ('extra-target', 'malformed_output')])
async def test_one_triage_round_preserves_closed_decisions_and_rejects_invalid_ids(
    investigation: InvestigationRun, decision: str, reason: str | None) -> None:
    backend = investigation.backend
    contracts: list[dict[str, Any]] = []

    @backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        assert "return 'universe'" in (investigation.repo / 'api.py').read_text()
        contract = stage['response_contract']
        contracts.append(contract)
        assert contract['schema'] == backend.calls[-1]['output_schema']
        validator = Draft202012Validator(contract['schema'])
        if stage['stage'] == 'first_pass':
            output['candidates'] = [candidate(disposition='confirmed', finding=record())
                                    if decision == 'contradiction' else candidate(disposition='rejected'), candidate()]
            assert validator.is_valid(output)
            invented = dict(output, candidates=[candidate(candidate_id='invented-native-id')])
            assert not validator.is_valid(invented)
        else:
            pending, = stage['candidates']
            closed_id, = stage['closed_candidate_ids']
            assert stage['progress'] == stage['assigned_target_ids'] == []
            assert contract['skeleton']['targets'] == []
            assert closed_id != pending['candidate_id']
            assert pending['candidate_id'] and pending['disposition'] == 'open'
            assert stage['assigned_candidate_ids'] == [pending['candidate_id']]
            disposition = ('confirmed' if decision in {'confirmed', 'missing-finding'}
                           else 'unresolved' if decision == 'unresolved' else 'rejected')
            output['candidates'] = [candidate(disposition=disposition, candidate_id=pending['candidate_id'],
                                               finding=record() if decision == 'confirmed' else None)]
            if decision == 'unknown':
                output['candidates'][0]['candidate_id'] = closed_id
            elif decision == 'duplicate':
                output['candidates'] *= 2
            elif decision == 'omitted':
                output['candidates'] = []
            elif decision == 'contradiction':
                # The same captured source refutes the earlier reasoning; no changed premise is needed.
                output['notes'] = "api.py:2 already returns a greeting that current callers accept."
                output['contradictions'] = [closed_id]
            elif decision == 'unknown-contradiction':
                output['contradictions'] = ['foreign-candidate']
            elif decision == 'extra-target':
                output['targets'] = [{'target_id': 'api.py', 'status': 'reviewed', 'reason': ''}]
            if decision in {'rejected', 'extra-target'}:
                assert validator.is_valid(output) is (decision == 'rejected')

    await investigation.finish('python', reason=reason,
                               findings=('Grounded defect',) if decision in {'confirmed', 'contradiction'} else ())
    assert [s['stage'] for s in backend.stages if s['scope_id'] == 'python'] == ['first_pass', 'triage']
    assert len(contracts) == 2 and all(contract['schema'] and contract['skeleton'] for contract in contracts)
    events = stage_ends(investigation, 'python')
    assert [event['metadata']['attempt'] for event in events] == [1, 1]
    if decision in {'rejected', 'extra-target'}:
        assert events[-1]['metadata']['failure_class'] == (
            'assignment_mismatch' if decision == 'extra-target' else None)


@pytest.mark.parametrize(('confirmed', 'fault', 'integration_starts'), [
    (False, None, 18), (True, None, 18), (False, 'cutoff', 18), (True, 'cutoff', 18), (False, 'syntax', 18),
    (True, None, 81), (True, 'cutoff', 81)])
async def test_integration_retains_admitted_findings_and_cumulative_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed: bool, fault: str | None,
    integration_starts: int) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=41, lines=35)
    cutoff = fault == 'cutoff'
    backend = review.backend
    observations: list[ToolStartEvent] = []

    @backend.script('structure', terminal=False)
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['stage'] == 'integration':
            starts = 0 if fault == 'syntax' else integration_starts
            assert stage['observed_tool_starts'] == 0
            assert stage['remaining_tool_calls'] == 96
            assert stage['assigned_target_ids'] == ['integration:structure']
            assert stage['progress'] == [] and stage['folded_alternatives'] is True
            assert stage['assigned_files'][0].startswith('00_guide_')
            assert {'api.py', 'module_00.py'} <= set(stage['assigned_files'])
            assert stage['remaining_work']['assignment_units'] == stage['remaining_work']['stages'] == 1
            output['candidates'] = [candidate(disposition='rejected'), candidate()]
        else:
            starts = stage['remaining_tool_calls'] + 1 if cutoff else 0
            assert stage['observed_tool_starts'] == integration_starts
            assert stage['remaining_tool_calls'] == stage['hard_tool_call_allowance'] == 96 - integration_starts
            assert stage['completed_target_ids'] == ['integration:structure']
            assert stage['progress'] == stage['assigned_target_ids'] == []
            assert [(item['candidate_id'], item['disposition']) for item in stage['closed_decisions']] == (
                [('structure:candidate:1', 'rejected'), ('structure:candidate:3', 'confirmed')] if confirmed
                else [('structure:candidate:1', 'rejected')])
            assert stage['closed_candidate_ids'] == [item['candidate_id'] for item in stage['closed_decisions']]
            pending, = stage['candidates']
            assert pending['disposition'] == 'open'
            assert stage['assigned_candidate_ids'] == [pending['candidate_id']] == ['structure:candidate:2']
            output['candidates'] = [candidate(disposition='rejected', candidate_id=pending['candidate_id'])]
        for i in range(starts):
            name = ('api.py', 'module_00.py')[i] if stage['stage'] == 'integration' and i < 2 else 'api.py'
            event = ToolStartEvent(id=f'observed-{len(observations)}', name='Read', input={'file_path': name})
            observations.append(event)
            yield event
            if stage['stage'] == 'integration':
                yield ToolResultEvent(id=event.id, output=(review.repo / name).read_text(), is_error=False)
        if stage['stage'] == 'integration' and confirmed:
            output['candidates'].append(candidate(disposition='confirmed', finding=record(),
                                                 grounds="api.py:2 returns 'world'."))
        if fault == 'syntax':
            yield TextEvent(text=json.dumps(output) + ', "candidates": []}')
        yield ResultEvent(structured_output=None if fault == 'syntax' else output, continuation=None)

    reason = None if fault is None else {'cutoff': 'host_tool_budget_exhaustion', 'syntax': 'malformed_output'}[fault]
    data = await review.finish('structure', reason=reason, findings=('Grounded defect',) if confirmed else ())
    assert set(scopes(data)) == {'python', 'generic', 'structure'}
    assert not any('evaluate the implementation' in call['prompt'].lower() for call in backend.calls)
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    metadata = [event['metadata'] for event in stage_ends(review, 'structure')]
    if fault == 'syntax':
        assert phases['alternatives']['status'] == 'incomplete'
        assert phases['alternatives']['reason_codes'] == ['malformed_output']
        event, = metadata
        assert event['attempt'] == 1 and event['observed_tool_starts'] == 0
        assert event['failure_class'] == 'syntax_failure'
        coverage = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
        assert coverage['diagnostics']['scopes']['structure'] in coverage['diagnostics']['phases']['alternatives']
    else:
        assert phases['alternatives']['status'] == ('incomplete' if cutoff else 'complete')
        assert len(observations) == (97 if cutoff else integration_starts)
        assert metadata[-1]['observed_tool_starts'] == (97 if cutoff else integration_starts)
        assert metadata[-1]['remaining_tool_calls'] == (0 if cutoff else 96 - integration_starts)
        assert [item['admitted'] for item in metadata] == [True, not cutoff]
        stages = [stage for stage in backend.stages if stage['scope_id'] == 'structure']
        assert [stage['stage'] for stage in stages] == ['integration', 'triage']
        assert all(stage['advisory_tool_call_target'] <= stage['remaining_tool_calls'] for stage in stages)


async def test_failed_retry_discards_poisoned_progress_but_charges_every_observed_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review = many_file_review(tmp_path, monkeypatch, lines=70)
    backend = review.backend
    backend.supports_budget_preamble = True
    timer = FakeClock(monotonic_value=clock.monotonic()).install(monkeypatch)
    patch_retry_sleep(monkeypatch, timer)
    deadlines: list[float] = []
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')
    attempts = 0

    class TransportFailure(RuntimeError):
        retryable = True

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal attempts
        if stage['scope_id'] == 'python':
            deadlines.append(timer.monotonic() + backend.calls[-1]['wall_budget_s'])
            timer.advance(5)
        if stage['scope_id'] == 'python' and not stage['progress']:
            attempts += 1
            # Starts arrive together before any completion, as parallel members may.
            paths = ['api.py'] * 2 if attempts == 1 else stage['assigned_files']
            for index, path in enumerate(paths):
                yield ToolStartEvent(id=f'attempt-{attempts}-{index}', name='Read', input={'file_path': path})
            if attempts == 1:
                yield ToolResultEvent(id='attempt-1-0', output='poisoned incomplete evidence', is_error=False)
                yield ResultEvent(structured_output=stage_result(stage, candidates=[
                    candidate(disposition='confirmed', finding=record())]), continuation=None)
                raise TransportFailure('retryable provider transport failure')
            for index, path in enumerate(paths):
                source = (review.repo / path).read_text()
                yield ToolResultEvent(id=f'attempt-{attempts}-{index}', output=source, is_error=False)
        elif stage['scope_id'] == 'python':
            assert stage['observed_tool_starts'] >= 6
            if len(stage['progress']) == 4:
                assert stage['observed_tool_starts'] == 6 and stage['remaining_tool_calls'] == 90
            assert stage['candidates'] == []
            assert all('poisoned' not in note for note in stage['notes'])
        elif stage['scope_id'] == 'structure':
            assert stage['observed_tool_starts'] == 0 and stage['remaining_tool_calls'] == 96
            assert stage['candidates'] == []

    backend.stage_response = response
    await review.finish('python', review_profile=_profile_with_pipeline(
        reviewer_investigation_budget_s=20, review_wall_budget_s=2000))
    assert attempts == 2
    assert len(deadlines) > 2
    assert deadlines == pytest.approx([deadlines[0]] * len(deadlines), abs=0.01, rel=0)
    metadata = stage_ends(review, 'python')[0]['metadata']
    assert metadata['investigation_base_budget_s'] == 20
    assert metadata['investigation_scaled_budget_s'] == 40
    assert metadata['investigation_effective_allowance_s'] == pytest.approx(40, rel=0, abs=0.01)
    assert stage_ends(review, 'python')[0]['metadata']['observed_tool_starts'] == 6
    assert stage_ends(review, 'python')[-1]['metadata']['observed_tool_starts'] == 6
    assert stage_ends(review, 'python')[-1]['metadata']['remaining_tool_calls'] == 90
    python_stages = [stage for stage in backend.stages if stage['scope_id'] == 'python']
    assert all(stage['stage'] == 'first_pass' for stage in python_stages)
    assert len(python_stages) > 2
    assert all(len(stage['assigned_files']) <= 4 for stage in python_stages)


@pytest.mark.parametrize('automatic_shards', [False, True], ids=['whole-stack', 'automatic-shards'])
async def test_runner_cancellation_closes_stream_and_does_not_admit_an_aborted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, automatic_shards: bool) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=10, docs_count=0)
    checkpoint_emitted = asyncio.Event()
    blocked_scopes: set[str] = set()
    closed_scopes: set[str] = set()
    expected = {'python#0', 'python#1'} if automatic_shards else {'python'}

    class CancellationBackend(StagedBackend):
        cancel_done = False

        async def cancel(self) -> None:
            await asyncio.sleep(0)
            self.cancel_done = True

    backend = CancellationBackend(review.repo)
    backend.fanout_concurrency = 2
    review.backend = backend

    async def response(stage: dict[str, Any], output: dict[str, Any]) -> AsyncGenerator[AgentEvent, None]:
        scope_id = stage['scope_id']
        if scope_id not in expected:
            return
        path = stage['assigned_files'][0]
        line = 2 if path == 'api.py' else 1
        finding = dict(record(line), file=path, evidence=f'{path}:{line}')
        if stage['progress']:
            # The settled first stage's finding is carried into this later request.
            assert len(stage['closed_decisions']) == 1
            assert stage['closed_decisions'][0]['conclusion'].startswith('Grounded defect: ')
            try:
                for event in read_events(review.repo, path, event_id=f'{scope_id}-cancelled-read'):
                    yield event
                output['candidates'] = [candidate(disposition='confirmed', file=path, line=line,
                    grounds=finding['evidence'], finding=dict(finding, description='Aborted checkpoint'))]
                yield ResultEvent(structured_output=output, continuation=None)
                blocked_scopes.add(scope_id)
                if blocked_scopes == expected:
                    checkpoint_emitted.set()
                await asyncio.Event().wait()
            finally:
                closed_scopes.add(scope_id)
            return
        for assigned in stage['assigned_files']:
            for event in read_events(review.repo, assigned, event_id=f'{scope_id}-admitted-{assigned}'):
                yield event
        output['candidates'] = [candidate(disposition='confirmed', file=path, line=line,
                                         grounds=finding['evidence'], finding=finding)]

    backend.stage_response = response
    task = asyncio.create_task(review.run(deep_shard_enabled=None if automatic_shards else False,
                                          review_cache_enabled=True))
    try:
        await asyncio.wait_for(checkpoint_emitted.wait(), timeout=60)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert closed_scopes == expected and backend.cancel_done
    for scope_id in expected:
        completed, interrupted = stage_ends(review, scope_id)
        assert completed['metadata']['admitted'] is True
        assert interrupted['metadata']['admitted'] is False
        assert interrupted['metadata']['failure_class'] == 'cancellation'
        assert interrupted['metadata']['stop_reason'] == 'interruption'
        assert interrupted['metadata']['retained_findings'] == completed['metadata']['retained_findings'] == 1
        assert interrupted['metadata']['observed_tool_starts'] == 5
    assert not review.output.exists()
    coverage_paths = list(review.repo.glob('.daydream/**/review-coverage.json'))
    for coverage_path in coverage_paths:
        coverage = json.loads(coverage_path.read_text())
        assert not any(scope['scope_id'] in expected and scope['status'] == 'complete'
                       for scope in coverage['stack_outcomes'])
    # Cancellation must not publish any aborted scope as a reusable complete result.
    manifests = list(review.repo.glob('.daydream/review-cache/entries/*/manifest.json'))
    assert all(json.loads(path.read_text())['unit'] not in {f'shard:{scope}' for scope in expected}
               for path in manifests)
