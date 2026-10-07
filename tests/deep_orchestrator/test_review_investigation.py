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
from tests.harness.stub_backend import review_stage_state, stage_result
from tests.test_deep_orchestrator import _profile_with_pipeline


class StagedBackend(EmptyReviewBackend):
    """A provider that remembers nothing beyond the current request."""

    def __init__(self, repo: Path) -> None:
        super().__init__(repo, forbid_merge=False, forbid_supervise=False)
        self.merge_echo_records = True
        self.stages: list[dict[str, Any]] = []
        self.stage_response: (
            Callable[[dict[str, Any], dict[str, Any]], Iterable[AgentEvent] | None] | None
        ) = None
        self.stage_delay: Callable[[dict[str, Any]], float] | None = None

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        stage = review_stage_state(prompt)
        if stage is not None:
            self.stages.append(stage)
            self.calls.append({'prompt': prompt, **kwargs})
            if self.stage_delay is not None:
                await asyncio.sleep(self.stage_delay(stage))
            output = stage_result(stage)
            if self.stage_response is not None:
                for event in self.stage_response(stage, output) or ():
                    yield event
            yield ResultEvent(structured_output=output, continuation=None)
            return
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


class InvestigationRun(ReviewRun):
    backend: StagedBackend

    def __init__(self, repo: Path, tmp: Path, patch: pytest.MonkeyPatch) -> None:
        super().__init__(repo, tmp, patch)
        self.backend = StagedBackend(repo)

    async def finish(self, scope_id: str, *, reason: str | None = None, findings: tuple[str, ...] = (),
                     statuses: tuple[str, ...] | None = None, **overrides: Any) -> dict[str, Any]:
        assert await self.run(**overrides) == 0
        data = self.load()
        assert [finding['title'] for finding in data['findings']] == list(findings)
        scope = scopes(data)[scope_id]
        assert scope['status'] in (statuses or ('incomplete' if reason else 'complete',))
        assert scope['reason_codes'] == ([reason] if reason else [])
        assert not any(call.get('no_tools') for call in self.backend.calls)
        return data


@pytest.fixture
def investigation(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InvestigationRun:
    return InvestigationRun(multi_stack_target, tmp_path, monkeypatch)


def candidate(*, disposition: str = 'open', candidate_id: str = '',
              finding: dict[str, Any] | None = None) -> dict[str, Any]:
    return {'candidate_id': candidate_id, 'file': 'api.py', 'line': 2,
            'trigger': 'Calling hello()', 'consequence': 'The returned greeting may violate callers.',
            'grounds': "api.py:2 returns 'universe'; current callers accept that greeting.",
            'disposition': disposition, 'finding': finding}


@pytest.mark.parametrize(('fault', 'reason'), [
    ('unknown', 'malformed_output'), ('duplicate', 'malformed_output'),
    ('omitted', 'malformed_output'), ('read-all', 'malformed_output'),
    ('handoff-bytes', 'evidence_incomplete'), ('handoff-items', 'evidence_incomplete'),
    ('missing-grounds', 'evidence_incomplete'), ('truncated-grounds', 'evidence_incomplete'),
])
async def test_invalid_stage_declarations_and_evidence_do_not_establish_coverage(
    investigation: InvestigationRun, fault: str, reason: str,
) -> None:
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
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
        if fault in {'read-all', 'truncated-grounds'}:
            yield ToolStartEvent(id='read', name='Read', input={'file_path': 'api.py'})
            yield ToolResultEvent(id='read', output=(investigation.repo / 'api.py').read_text(), is_error=False,
                                  truncated=fault == 'truncated-grounds')

    investigation.backend.stage_response = response
    data = await investigation.finish('python', reason=reason, statuses=('incomplete', 'failed'))
    assert scopes(data)['react']['status'] == scopes(data)['structure']['status'] == 'complete'
    terminal = stage_ends(investigation, 'python')[-1]
    assert (terminal['status'], terminal.get('reason_code')) == ('failed', 'domain_failure')


def stage_ends(review: ReviewRun, scope_id: str) -> list[dict[str, Any]]:
    trajectory = json.loads((review.tmp / 'trajectory.json').read_text())
    return [event for sub in trajectory['extra']['subtrajectories'] for event in sub.get('phase_events', [])
            if event['event'] == 'phase_end' and event['metadata'].get('review_scope_id') == scope_id]


def many_file_review(tmp_path: Path, patch: pytest.MonkeyPatch, *, count: int = 17,
                     lines: int = 1, wired: bool = True) -> InvestigationRun:
    before = {'api.py': "def hello():\n    return 'world'\n",
              'a_flags.py': 'def parse_flags(args):\n    return {}\n',
              'z_request.py': "def build_request(options):\n    return {'legacy': False}\n"}
    before.update({f'module_{i:02}.py': 'VALUE = 0\n' for i in range(count - 3)})
    after = {name: content + ''.join(f'# changed {i}\n' for i in range(lines)) for name, content in before.items()}
    after['a_flags.py'] = "def parse_flags(args):\n    return {'dry_run': '--dry-run' in args}\n"
    after['z_request.py'] = ('def build_request(options):\n'
                             + ("    return {'dry_run': options['dry_run']}\n" if wired else '    return {}\n'))
    repo = tmp_path / 'many_files'
    seed_feature_branch(repo, base=before, feature=after)
    return InvestigationRun(repo, tmp_path, patch)


@pytest.mark.parametrize(('stop', 'reason'), [('tool', 'host_tool_budget_exhaustion'),
    ('model', 'model_budget_exhaustion'), ('backend', 'backend_failure'),
    ('deadline', 'host_pipeline_budget_exhaustion')])
async def test_unsuccessful_later_stage_retains_only_prior_admitted_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: str, reason: str,
) -> None:
    review = many_file_review(tmp_path, monkeypatch)
    backend = review.backend

    if stop == 'deadline':
        backend.stage_delay = lambda stage: (5.0 if stage['progress'] else 0.1) if stage['scope_id'] == 'python' else 0

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] == 'python':
            if not stage['progress']:
                output['candidates'] = [candidate(disposition='confirmed', finding=record())]
                return
            yield ResultEvent(structured_output=stage_result(stage, candidates=[
                candidate(disposition='confirmed', finding=dict(record(), description='Unadmitted defect'))]),
                continuation=None)
            if stop == 'tool':
                for i in range(17):
                    yield ToolStartEvent(id=f'cutoff-{i}', name='Read', input={'file_path': 'api.py'})
            else:
                raise MaxTurnsError('model exhausted') if stop == 'model' else RuntimeError('backend unavailable')

    backend.stage_response = response
    overrides: dict[str, Any] = (
        {'review_profile': _profile_with_pipeline(review_wall_budget_s=3)} if stop == 'deadline' else {}
    )
    data = await review.finish('python', reason=reason, findings=('Grounded defect',), **overrides)
    assert scopes(data)['structure']['status'] == 'complete'
    admitted, stopped = stage_ends(review, 'python')
    assert admitted['status'] == 'succeeded'
    assert (stopped['status'], stopped.get('reason_code')) == (
        ('timed_out', 'timed_out') if stop == 'deadline' else ('failed', 'domain_failure'))
    assert [stage['stage'] for stage in backend.stages if stage['scope_id'] == 'python'] == ['first_pass'] * 2


@pytest.mark.parametrize(('decision', 'reason'), [
    ('rejected', None), ('confirmed', None), ('missing-finding', 'malformed_output'),
    ('unresolved', 'evidence_incomplete'), ('unknown-contradiction', 'malformed_output'),
    ('contradiction', 'evidence_incomplete'), ('unknown', 'malformed_output'),
    ('duplicate', 'malformed_output'), ('omitted', 'malformed_output')])
async def test_one_triage_round_preserves_closed_decisions_and_rejects_invalid_ids(
    investigation: InvestigationRun, decision: str, reason: str | None,
) -> None:
    backend = investigation.backend

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] == 'python':
            if stage['stage'] == 'first_pass':
                output['candidates'] = [candidate(disposition='rejected'), candidate()]
            else:
                closed, pending = stage['candidates']
                assert stage['progress'] == [{'target_id': 'api.py', 'status': 'reviewed', 'reason': ''}]
                assert closed['disposition'] == 'rejected'
                assert pending['candidate_id'] and pending['disposition'] == 'open'
                assert stage['assigned_candidate_ids'] == [pending['candidate_id']]
                yield ToolStartEvent(id='targeted-reread', name='Read', input={'file_path': 'api.py'})
                yield ToolResultEvent(id='targeted-reread', output=(investigation.repo / 'api.py').read_text(),
                                      is_error=False)
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

    backend.stage_response = response
    await investigation.finish('python', reason=reason,
                               findings=('Grounded defect',) if decision == 'confirmed' else ())
    assert [s['stage'] for s in backend.stages if s['scope_id'] == 'python'] == ['first_pass', 'triage']


@pytest.mark.parametrize('cutoff', [False, True])
@pytest.mark.parametrize('wired', [False, True])
async def test_cross_batch_integration_preserves_cumulative_spend_and_flag_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cutoff: bool, wired: bool,
) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=41, lines=35, wired=wired)
    backend = review.backend
    observations: list[ToolStartEvent] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] == 'structure':
            if stage['stage'] == 'first_pass':
                starts = 12
                output['notes'] = ', '.join(stage['assigned_target_ids']) + ': changed behavior checked.'
                assert stage['tool_call_allowance'] == 12
                assert stage['remaining_tool_calls'] == 96 - len(observations)
                if not stage['progress']:
                    output['candidates'] = [candidate(disposition='rejected'), candidate()]
            elif stage['stage'] == 'integration':
                starts = 9
                assert stage['observed_tool_starts'] == 72
                assert stage['remaining_tool_calls'] == 24
                assert stage['assigned_target_ids'] == ['integration:structure']
                assert {target['target_id'] for target in stage['progress']} == {
                    path.name for path in review.repo.glob('*.py')}
                assert 'a_flags.py' in ' '.join(stage['notes']) and 'z_request.py' in ' '.join(stage['notes'])
                assert len(stage['progress']) == 41
                assert stage['candidates'][0]['disposition'] == 'rejected'
            else:
                starts = 16 if cutoff else 0
                assert stage['observed_tool_starts'] == 81
                assert stage['remaining_tool_calls'] == stage['tool_call_allowance'] == 15
                closed, pending, *_confirmed = stage['candidates']
                assert closed['disposition'] == 'rejected'
                assert stage['assigned_candidate_ids'] == [pending['candidate_id']]
                output['candidates'] = [candidate(disposition='rejected', candidate_id=pending['candidate_id'])]
            for i in range(starts):
                name = ('a_flags.py', 'z_request.py')[i] if stage['stage'] == 'integration' and i < 2 else 'api.py'
                event = ToolStartEvent(id=f'observed-{len(observations)}', name='Read',
                                       input={'file_path': name, 'parallel_member': i})
                observations.append(event)
                yield event
                if stage['stage'] == 'integration' and i < 2:
                    yield ToolResultEvent(id=event.id, output=(review.repo / name).read_text(), is_error=False)
            if (stage['stage'] == 'integration'
                    and "options['dry_run']" not in (review.repo / 'z_request.py').read_text()):
                finding = dict(record(), file='z_request.py', line=2,
                               description='Parsed dry-run flag is dropped during request construction',
                               evidence='a_flags.py:2 parses dry_run; z_request.py:2 returns an empty request')
                output['candidates'] = [dict(candidate(disposition='confirmed', finding=finding),
                    file='z_request.py', trigger='Pass --dry-run through parse_flags and build_request',
                    consequence='Request omits dry_run and performs a live operation', grounds=finding['evidence'])]

    backend.stage_response = response
    data = await review.finish('structure', reason='host_tool_budget_exhaustion' if cutoff else None,
                              findings=() if wired else ('Parsed dry-run flag is dropped during request construction',))
    assert set(scopes(data)) == {'python', 'structure'}
    assert len(observations) == (97 if cutoff else 81)
    metadata = [event['metadata'] for event in stage_ends(review, 'structure')]
    assert metadata[-1]['observed_tool_starts'] == (97 if cutoff else 81)
    assert metadata[-1]['remaining_tool_calls'] == (0 if cutoff else 15)
    stages = [stage for stage in backend.stages if stage['scope_id'] == 'structure']
    assert [stage['stage'] for stage in stages] == ['first_pass'] * 6 + ['integration', 'triage']
    assert 'a_flags.py' in stages[0]['assigned_target_ids']
    assert 'z_request.py' in stages[5]['assigned_target_ids']
    assert all(len(stage['assigned_target_ids']) <= 8 for stage in stages[:6])
    assert all(stage['tool_call_allowance'] <= 16 for stage in stages)


async def test_failed_retry_discards_poisoned_progress_but_charges_every_observed_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = many_file_review(tmp_path, monkeypatch, lines=70)
    backend = review.backend
    monkeypatch.setenv('DAYDREAM_PI_RETRY_ATTEMPTS', '1')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_BASE_DELAY_S', '0')
    monkeypatch.setenv('DAYDREAM_PI_RETRY_MAX_DELAY_S', '0')
    attempts = 0

    class TransportFailure(RuntimeError):
        retryable = True

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
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
                raise TransportFailure('retryable provider transport failure')
        elif stage['scope_id'] == 'python':
            assert stage['observed_tool_starts'] == 5 and stage['remaining_tool_calls'] == 91
            assert stage['candidates'] == []
            assert all('poisoned' not in note for note in stage['notes'])
        elif stage['scope_id'] == 'structure':
            assert stage['observed_tool_starts'] == 0 and stage['remaining_tool_calls'] == 96
            assert stage['candidates'] == []

    backend.stage_response = response
    data = await review.finish('python')
    assert attempts == 2
    assert all(scope['status'] == 'complete' for scope in scopes(data).values())
    assert [stage['stage'] for stage in backend.stages if stage['scope_id'] == 'python'] == ['first_pass'] * 4


@pytest.mark.parametrize('oversized', [False, True])
async def test_batches_use_complete_diff_weights_and_preserve_explicit_oversized_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oversized: bool,
) -> None:
    before = {'api.py': 'VALUE = 0\n', 'module_00.py': 'VALUE = 0\n'}
    after = {'api.py': "VALUE = '" + 'x' * (60_000 if oversized else 28_000) + "'\n",
             'module_00.py': "VALUE = '" + 'y' * (3 if oversized else 28_000) + "'\n"}
    repo = tmp_path / 'weighted_diff'
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    backend = review.backend

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if oversized and stage['scope_id'] == 'python':
            if stage['assigned_target_ids'] == ['api.py']:
                output['targets'][0].update(status='not_reviewed', reason='Oversized diff needs more targeted reads.')
            else:
                assert stage['progress'] == [{'target_id': 'api.py', 'status': 'not_reviewed',
                                             'reason': 'Oversized diff needs more targeted reads.'}]

    backend.stage_response = response
    await review.finish('python', reason='evidence_incomplete' if oversized else None)
    stages = [stage for stage in backend.stages if stage['scope_id'] == 'python']
    assert [stage['assigned_target_ids'] for stage in stages] == [['api.py'], ['module_00.py']]
    assert all(stage['tool_call_allowance'] <= 16 for stage in stages)
    assert len((repo / '.daydream/diff.patch').read_bytes()) > 48 * 1024
