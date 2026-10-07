"""Bounded review stages through runner.run, real Git and the external Backend seam."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, MaxTurnsError, ResultEvent, ToolResultEvent, ToolStartEvent
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.deep_orchestrator.test_review_completion import ReviewRun, record, scopes
from tests.harness.git_helpers import seed_feature_branch
from tests.test_deep_orchestrator import _profile_with_pipeline


class StagedBackend(EmptyReviewBackend):
    """A provider that remembers nothing beyond the current request."""

    def __init__(self, repo: Path) -> None:
        super().__init__(repo)
        self.stages: list[dict[str, Any]] = []
        self.stage_response: Callable[[dict[str, Any]], Iterable[AgentEvent | BaseException]] | None = None
        self.stage_delay: Callable[[dict[str, Any]], float] | None = None

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if 'Host review stage:\n' in prompt:
            stage = json.JSONDecoder().raw_decode(prompt.split('Host review stage:\n', 1)[1])[0]
            self.stages.append(stage)
            self.calls.append({'prompt': prompt, **kwargs})
            if self.stage_delay is not None:
                await asyncio.sleep(self.stage_delay(stage))
            if self.stage_response is not None:
                for event in self.stage_response(stage):
                    if isinstance(event, BaseException):
                        raise event
                    yield event
                return
            yield ResultEvent(structured_output={
                'targets': [{'target_id': target, 'status': 'reviewed', 'reason': ''}
                            for target in stage['assigned_target_ids']],
                'notes': 'The assigned changed behavior was checked against its callers.',
                'candidates': [], 'contradictions': [],
            }, continuation=None)
            return
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


@pytest.fixture
def investigation(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReviewRun:
    review = ReviewRun(multi_stack_target, tmp_path, monkeypatch)
    review.backend = StagedBackend(review.repo)
    return review


async def test_acknowledged_empty_batches_complete_without_model_publication(investigation: ReviewRun) -> None:
    assert await investigation.run() == 0
    data = investigation.load()
    assert data['findings'] == []
    assert all(scope['status'] == 'complete' for scope in scopes(data).values())
    backend = investigation.backend
    assert isinstance(backend, StagedBackend)
    assert sorted((stage['scope_id'], stage['stage']) for stage in backend.stages) == [
        ('generic', 'first_pass'), ('python', 'first_pass'), ('react', 'first_pass'),
        ('structure', 'first_pass'), ('structure', 'integration'),
    ]
    assert all(call.get('no_tools') is not True for call in backend.calls)


def stage_result(stage: dict[str, Any], *, candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        'targets': [{'target_id': target, 'status': 'reviewed', 'reason': ''}
                    for target in stage['assigned_target_ids']],
        'notes': 'Checked changed behavior against callers.',
        'candidates': candidates or [], 'contradictions': [],
    }


def candidate(*, disposition: str = 'open', candidate_id: str = '',
              finding: dict[str, Any] | None = None) -> dict[str, Any]:
    return {'candidate_id': candidate_id, 'file': 'api.py', 'line': 2,
            'trigger': 'Calling hello()', 'consequence': 'The returned greeting may violate callers.',
            'grounds': "api.py:2 returns 'universe'; current callers accept that greeting.",
            'disposition': disposition, 'finding': finding}


async def test_open_candidate_is_triaged_once_with_host_id_and_prior_progress(investigation: ReviewRun) -> None:
    backend = investigation.backend
    assert isinstance(backend, StagedBackend)

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent]:
        candidates = []
        if stage['scope_id'] == 'python':
            if stage['stage'] == 'first_pass':
                candidates = [candidate()]
            elif stage['stage'] == 'triage':
                assert stage['progress'] == [{'target_id': 'api.py', 'status': 'reviewed', 'reason': ''}]
                assert len(stage['candidates']) == 1
                assigned = stage['candidates'][0]['candidate_id']
                assert assigned and stage['candidates'][0]['disposition'] == 'open'
                candidates = [candidate(disposition='rejected', candidate_id=assigned)]
                yield ToolStartEvent(id='targeted-reread', name='Read', input={'file_path': 'api.py'})
                yield ToolResultEvent(id='targeted-reread', output=(investigation.repo / 'api.py').read_text(),
                                      is_error=False)
        yield ResultEvent(structured_output=stage_result(stage, candidates=candidates), continuation=None)

    backend.stage_response = response
    assert await investigation.run() == 0
    data = investigation.load()
    assert data['findings'] == [] and scopes(data)['python']['status'] == 'complete'
    assert [s['stage'] for s in backend.stages if s['scope_id'] == 'python'] == ['first_pass', 'triage']
    assert all(call.get('no_tools') is not True for call in backend.calls)


@pytest.mark.parametrize('fault', ['unknown', 'duplicate', 'omitted', 'read-all'])
async def test_reads_and_invalid_target_declarations_do_not_establish_coverage(
    investigation: ReviewRun, fault: str,
) -> None:
    backend = investigation.backend
    assert isinstance(backend, StagedBackend)

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent]:
        output = stage_result(stage)
        if stage['scope_id'] == 'python':
            if fault == 'unknown':
                output['targets'][0]['target_id'] = 'foreign.py'
            elif fault == 'duplicate':
                output['targets'] *= 2
            else:
                output['targets'] = []
            if fault == 'read-all':
                yield ToolStartEvent(id='all-hunks', name='Read', input={'file_path': 'api.py'})
                yield ToolResultEvent(id='all-hunks', output=(investigation.repo / 'api.py').read_text(),
                                      is_error=False)
        yield ResultEvent(structured_output=output, continuation=None)

    backend.stage_response = response
    assert await investigation.run() == 0
    data = investigation.load()
    assert scopes(data)['python']['status'] in {'incomplete', 'failed'}
    assert scopes(data)['python']['reason_codes'] == ['malformed_output']
    assert scopes(data)['react']['status'] == 'complete' and data['findings'] == []
    if fault == 'unknown':
        terminal = stage_ends(investigation, 'python')[-1]
        assert (terminal['status'], terminal.get('reason_code')) == ('failed', 'domain_failure')


def stage_ends(review: ReviewRun, scope_id: str) -> list[dict[str, Any]]:
    trajectory = json.loads((review.tmp / 'trajectory.json').read_text())
    return [event for sub in trajectory['extra']['subtrajectories'] for event in sub.get('phase_events', [])
            if event['event'] == 'phase_end' and event['metadata'].get('review_scope_id') == scope_id]


def many_file_review(tmp_path: Path, patch: pytest.MonkeyPatch, *, count: int = 17,
                     lines: int = 1) -> ReviewRun:
    before = {'api.py': "def hello():\n    return 'world'\n"}
    before.update({f'module_{i:02}.py': 'VALUE = 0\n' for i in range(count - 1)})
    after = {name: content + ''.join(f'# changed {i}\n' for i in range(lines)) for name, content in before.items()}
    repo = tmp_path / 'many_files'
    seed_feature_branch(repo, base=before, feature=after)
    review = ReviewRun(repo, tmp_path, patch)
    review.backend = StagedBackend(repo)
    review.backend.forbid_merge = False
    review.backend.forbid_supervise = False
    review.backend.merge_echo_records = True
    return review


@pytest.mark.parametrize(('stop', 'reason'), [('tool', 'host_tool_budget_exhaustion'),
    ('model', 'model_budget_exhaustion'), ('backend', 'backend_failure')])
async def test_unsuccessful_later_stage_retains_only_prior_admitted_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str, reason: str,
) -> None:
    review = many_file_review(tmp_path, monkeypatch)
    backend = review.backend
    assert isinstance(backend, StagedBackend)

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent | BaseException]:
        if stage['scope_id'] == 'python':
            if not stage['progress']:
                yield ResultEvent(structured_output=stage_result(stage, candidates=[
                    candidate(disposition='confirmed', finding=record())]), continuation=None)
                return
            yield ResultEvent(structured_output=stage_result(stage, candidates=[
                candidate(disposition='confirmed', finding=dict(record(), description='Unadmitted defect'))]),
                continuation=None)
            if stop == 'tool':
                for i in range(17):
                    yield ToolStartEvent(id=f'cutoff-{i}', name='Read', input={'file_path': 'api.py'})
            else:
                yield MaxTurnsError('model exhausted') if stop == 'model' else RuntimeError('backend unavailable')
            return
        yield ResultEvent(structured_output=stage_result(stage), continuation=None)

    backend.stage_response = response
    assert await review.run() == 0
    data = review.load()
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    assert scopes(data)['python']['status'] == 'incomplete'
    assert scopes(data)['python']['reason_codes'] == [reason]
    assert scopes(data)['structure']['status'] == 'complete'
    admitted, stopped = stage_ends(review, 'python')
    assert admitted['status'] == 'succeeded'
    assert (stopped['status'], stopped.get('reason_code')) == ('failed', 'domain_failure')
    assert not any(call.get('no_tools') for call in backend.calls)


@pytest.mark.parametrize('fault', ['handoff-bytes', 'handoff-items', 'missing-grounds', 'truncated-grounds'])
async def test_incomplete_handoff_and_candidate_grounds_do_not_mint_findings(
    investigation: ReviewRun, fault: str,
) -> None:
    backend = investigation.backend
    assert isinstance(backend, StagedBackend)

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent]:
        output = stage_result(stage)
        if stage['scope_id'] == 'python':
            concrete = candidate(disposition='confirmed', finding=record())
            if fault == 'handoff-bytes':
                output['notes'] = 'x' * 70_000
            elif fault == 'handoff-items':
                output['candidates'] = [concrete] * 129
            else:
                output['candidates'] = [concrete]
                if fault == 'missing-grounds':
                    concrete['grounds'] = ''
                else:
                    yield ToolStartEvent(id='clipped', name='Read', input={'file_path': 'api.py'})
                    yield ToolResultEvent(id='clipped', output="def hello(): ... [truncated]", is_error=False,
                                          truncated=True)
        yield ResultEvent(structured_output=output, continuation=None)

    backend.stage_response = response
    assert await investigation.run() == 0
    data = investigation.load()
    assert data['findings'] == []
    assert scopes(data)['python']['status'] in {'incomplete', 'failed'}
    assert scopes(data)['python']['reason_codes'] == ['evidence_incomplete']
    assert scopes(data)['structure']['status'] == 'complete'


@pytest.mark.parametrize(('decision', 'reason'), [
    ('rejected', None), ('confirmed', None), ('missing-finding', 'malformed_output'),
    ('unresolved', 'evidence_incomplete'), ('unknown-contradiction', 'malformed_output'),
    ('contradiction', 'evidence_incomplete'), ('unknown', 'malformed_output'),
    ('duplicate', 'malformed_output'), ('omitted', 'malformed_output')])
async def test_one_triage_round_preserves_closed_decisions_and_rejects_invalid_ids(
    investigation: ReviewRun, decision: str, reason: str | None,
) -> None:
    backend = investigation.backend
    assert isinstance(backend, StagedBackend)

    if decision == 'confirmed':
        backend.forbid_merge = backend.forbid_supervise = False
        backend.merge_echo_records = True

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent]:
        output = stage_result(stage)
        if stage['scope_id'] == 'python':
            if stage['stage'] == 'first_pass':
                output['candidates'] = [candidate(disposition='rejected'), candidate()]
            else:
                closed, pending = stage['candidates']
                assert closed['disposition'] == 'rejected'
                assert stage['assigned_candidate_ids'] == [pending['candidate_id']]
                disposition = ('confirmed' if decision in {'confirmed', 'missing-finding'}
                               else 'unresolved' if decision == 'unresolved' else 'rejected')
                output['candidates'] = [candidate(disposition=disposition, candidate_id=pending['candidate_id'],
                                                   finding=record() if decision == 'confirmed' else None)]
                if decision == 'unknown':
                    output['candidates'][0]['candidate_id'] = closed['candidate_id']
                elif decision == 'duplicate':
                    output['candidates'] *= 2
                elif decision == 'omitted':
                    output['candidates'] = []
                elif decision == 'contradiction':
                    output['contradictions'] = [closed['candidate_id']]
                elif decision == 'unknown-contradiction':
                    output['contradictions'] = ['foreign-candidate']
        yield ResultEvent(structured_output=output, continuation=None)

    backend.stage_response = response
    assert await investigation.run() == 0
    data = investigation.load()
    assert [finding['title'] for finding in data['findings']] == (
        ['Grounded defect'] if decision == 'confirmed' else [])
    assert scopes(data)['python']['status'] == ('complete' if reason is None else 'incomplete')
    assert scopes(data)['python']['reason_codes'] == ([] if reason is None else [reason])
    assert [s['stage'] for s in backend.stages if s['scope_id'] == 'python'] == ['first_pass', 'triage']


@pytest.mark.parametrize('cutoff', [False, True])
async def test_lossy_backend_receives_cumulative_81_of_96_spend_without_fresh_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cutoff: bool,
) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=41, lines=35)
    backend = review.backend
    assert isinstance(backend, StagedBackend)
    observations: list[ToolStartEvent] = []

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent]:
        output = stage_result(stage)
        if stage['scope_id'] == 'structure':
            if stage['stage'] == 'first_pass':
                starts = 12
                assert stage['tool_call_allowance'] == 12
                assert stage['remaining_tool_calls'] == 96 - len(observations)
                if not stage['progress']:
                    output['candidates'] = [candidate(disposition='rejected'), candidate()]
            elif stage['stage'] == 'integration':
                starts = 9
                assert stage['observed_tool_starts'] == 72
                assert stage['remaining_tool_calls'] == 24
                assert len(stage['progress']) == 41
                assert stage['candidates'][0]['disposition'] == 'rejected'
            else:
                starts = 16 if cutoff else 0
                assert stage['observed_tool_starts'] == 81
                assert stage['remaining_tool_calls'] == stage['tool_call_allowance'] == 15
                closed, pending = stage['candidates']
                assert closed['disposition'] == 'rejected'
                assert stage['assigned_candidate_ids'] == [pending['candidate_id']]
                output['candidates'] = [candidate(disposition='rejected', candidate_id=pending['candidate_id'])]
            for i in range(starts):
                event = ToolStartEvent(id=f'observed-{len(observations)}', name='Read',
                                       input={'file_path': 'api.py', 'parallel_member': i})
                observations.append(event)
                yield event
        yield ResultEvent(structured_output=output, continuation=None)

    backend.stage_response = response
    assert await review.run() == 0
    data = review.load()
    assert data['findings'] == []
    assert scopes(data)['structure']['status'] == ('incomplete' if cutoff else 'complete')
    assert scopes(data)['structure']['reason_codes'] == (['host_tool_budget_exhaustion'] if cutoff else [])
    assert len(observations) == (97 if cutoff else 81)
    trajectory = json.loads((tmp_path / 'trajectory.json').read_text())
    metadata = [event['metadata'] for sub in trajectory['extra']['subtrajectories']
                for event in sub.get('phase_events', []) if event['event'] == 'phase_end'
                and event['metadata'].get('review_scope_id') == 'structure']
    assert metadata[-1]['observed_tool_starts'] == (97 if cutoff else 81)
    assert metadata[-1]['remaining_tool_calls'] == (0 if cutoff else 15)
    stages = [stage for stage in backend.stages if stage['scope_id'] == 'structure']
    assert [stage['stage'] for stage in stages] == ['first_pass'] * 6 + ['integration', 'triage']
    assert len({target for stage in stages[:6] for target in stage['assigned_target_ids']}) == 41
    assert all(len(stage['assigned_target_ids']) <= 8 for stage in stages[:6])
    assert all(call.get('no_tools') is not True for call in backend.calls)


@pytest.mark.parametrize('wired', [False, True])
async def test_structure_integrates_flag_parsing_and_request_construction_across_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wired: bool,
) -> None:
    before = {'a_flags.py': 'def parse_flags(args):\n    return {}\n',
              'z_request.py': "def build_request(options):\n    return {'legacy': False}\n"}
    before.update({f'middle_{i:02}.py': 'VALUE = 0\n' for i in range(15)})
    after = {name: content + '# changed behavior\n' for name, content in before.items()}
    after['a_flags.py'] = "def parse_flags(args):\n    return {'dry_run': '--dry-run' in args}\n"
    after['z_request.py'] = ('def build_request(options):\n'
                             + ("    return {'dry_run': options['dry_run']}\n" if wired else '    return {}\n'))
    repo = tmp_path / 'cross_batch_flag'
    seed_feature_branch(repo, base=before, feature=after)
    review = ReviewRun(repo, tmp_path, monkeypatch)
    backend = StagedBackend(repo)
    backend.forbid_merge = backend.forbid_supervise = False
    backend.merge_echo_records = True
    review.backend = backend

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent]:
        output = stage_result(stage)
        if stage['stage'] == 'first_pass':
            output['notes'] = ', '.join(stage['assigned_target_ids']) + ': changed behavior checked.'
        elif stage['scope_id'] == 'structure' and stage['stage'] == 'integration':
            assert stage['assigned_target_ids'] == ['integration:structure']
            assert {target['target_id'] for target in stage['progress']} == set(after)
            assert 'a_flags.py' in ' '.join(stage['notes']) and 'z_request.py' in ' '.join(stage['notes'])
            for name in ('a_flags.py', 'z_request.py'):
                yield ToolStartEvent(id=name, name='Read', input={'file_path': name})
                yield ToolResultEvent(id=name, output=(repo / name).read_text(), is_error=False)
            if "options['dry_run']" not in (repo / 'z_request.py').read_text():
                finding = dict(record(), file='z_request.py', line=2,
                               description='Parsed dry-run flag is dropped during request construction',
                               evidence='a_flags.py:2 parses dry_run; z_request.py:2 returns an empty request')
                output['candidates'] = [dict(candidate(disposition='confirmed', finding=finding),
                    file='z_request.py', trigger='Pass --dry-run through parse_flags and build_request',
                    consequence='Request omits dry_run and performs a live operation',
                    grounds=finding['evidence'])]
        yield ResultEvent(structured_output=output, continuation=None)

    backend.stage_response = response
    assert await review.run() == 0
    data = review.load()
    assert scopes(data)['structure']['status'] == 'complete'
    assert set(scopes(data)) == {'python', 'structure'}
    structure = [stage for stage in backend.stages if stage['scope_id'] == 'structure']
    assert [stage['stage'] for stage in structure] == ['first_pass'] * 3 + ['integration']
    assert 'a_flags.py' in structure[0]['assigned_target_ids']
    assert 'z_request.py' in structure[2]['assigned_target_ids']
    assert all(stage['tool_call_allowance'] <= 16 for stage in structure)
    assert [finding['title'] for finding in data['findings']] == (
        [] if wired else ['Parsed dry-run flag is dropped during request construction'])
    assert all(call.get('no_tools') is not True for call in backend.calls)


async def test_failed_retry_discards_poisoned_progress_but_charges_every_observed_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = many_file_review(tmp_path, monkeypatch, lines=70)
    backend = review.backend
    assert isinstance(backend, StagedBackend)
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')
    attempts = 0

    class TransportFailure(RuntimeError):
        retryable = True

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent | BaseException]:
        nonlocal attempts
        if stage['scope_id'] == 'python' and not stage['progress']:
            attempts += 1
            # Starts arrive together before any completion, as parallel members may.
            for index in range(2 if attempts == 1 else 3):
                yield ToolStartEvent(id=f'attempt-{attempts}-{index}', name='Read', input={'file_path': 'api.py'})
            if attempts == 1:
                yield ToolResultEvent(id='attempt-1-0', output='poisoned incomplete evidence', is_error=False)
                yield ResultEvent(structured_output=stage_result(stage, candidates=[
                    candidate(disposition='confirmed', finding=record())]), continuation=None)
                yield TransportFailure('retryable provider transport failure')
                return
        elif stage['scope_id'] == 'python':
            assert stage['observed_tool_starts'] == 5 and stage['remaining_tool_calls'] == 91
            assert stage['candidates'] == []
            assert all('poisoned' not in note for note in stage['notes'])
        elif stage['scope_id'] == 'structure':
            assert stage['observed_tool_starts'] == 0 and stage['remaining_tool_calls'] == 96
            assert stage['candidates'] == []
        yield ResultEvent(structured_output=stage_result(stage), continuation=None)

    backend.stage_response = response
    assert await review.run() == 0
    data = review.load()
    assert attempts == 2
    assert data['findings'] == [] and all(scope['status'] == 'complete' for scope in scopes(data).values())
    assert [stage['stage'] for stage in backend.stages if stage['scope_id'] == 'python'] == ['first_pass'] * 4


async def test_fresh_stages_share_absolute_deadline_and_preserve_admitted_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = many_file_review(tmp_path, monkeypatch)
    backend = review.backend
    assert isinstance(backend, StagedBackend)
    backend.stage_delay = lambda stage: (5.0 if stage['progress'] else 0.1) if stage['scope_id'] == 'python' else 0

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent]:
        admitted = [candidate(disposition='confirmed', finding=record())] if stage['scope_id'] == 'python' else []
        yield ResultEvent(structured_output=stage_result(stage, candidates=admitted), continuation=None)

    backend.stage_response = response
    assert await review.run(review_profile=_profile_with_pipeline(review_wall_budget_s=3)) == 0
    data = review.load()
    assert scopes(data)['python']['status'] == 'incomplete'
    assert scopes(data)['python']['reason_codes'] == ['host_pipeline_budget_exhaustion']
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    assert [stage['stage'] for stage in backend.stages if stage['scope_id'] == 'python'] == ['first_pass'] * 2
    admitted, stopped = stage_ends(review, 'python')
    assert admitted['status'] == 'succeeded'
    assert (stopped['status'], stopped.get('reason_code')) == ('timed_out', 'timed_out')
    assert not any(call.get('no_tools') for call in backend.calls)


@pytest.mark.parametrize('oversized', [False, True])
async def test_batches_use_complete_diff_weights_and_preserve_explicit_oversized_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oversized: bool,
) -> None:
    before = {'api.py': 'VALUE = 0\n', 'module_00.py': 'VALUE = 0\n'}
    after = {'api.py': "VALUE = '" + 'x' * (60_000 if oversized else 28_000) + "'\n",
             'module_00.py': "VALUE = '" + 'y' * (3 if oversized else 28_000) + "'\n"}
    repo = tmp_path / 'weighted_diff'
    seed_feature_branch(repo, base=before, feature=after)
    review = ReviewRun(repo, tmp_path, monkeypatch)
    backend = StagedBackend(repo)
    review.backend = backend

    def response(stage: dict[str, Any]) -> Iterable[AgentEvent]:
        output = stage_result(stage)
        if oversized and stage['scope_id'] == 'python':
            if stage['assigned_target_ids'] == ['api.py']:
                output['targets'][0].update(status='not_reviewed', reason='Oversized diff needs more targeted reads.')
            else:
                assert stage['progress'] == [{'target_id': 'api.py', 'status': 'not_reviewed',
                                             'reason': 'Oversized diff needs more targeted reads.'}]
        yield ResultEvent(structured_output=output, continuation=None)

    backend.stage_response = response
    assert await review.run() == 0
    data = review.load()
    assert data['findings'] == []
    assert scopes(data)['python']['status'] == ('incomplete' if oversized else 'complete')
    assert scopes(data)['python']['reason_codes'] == (['evidence_incomplete'] if oversized else [])
    stages = [stage for stage in backend.stages if stage['scope_id'] == 'python']
    assert [stage['assigned_target_ids'] for stage in stages] == [['api.py'], ['module_00.py']]
    assert all(stage['tool_call_allowance'] <= 16 for stage in stages)
    assert len((repo / '.daydream/diff.patch').read_bytes()) > 48 * 1024
