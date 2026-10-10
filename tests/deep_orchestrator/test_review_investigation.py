"""Bounded review stages through runner.run, real Git and the external Backend seam."""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

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
from tests.harness.git_helpers import seed_feature_branch
from tests.harness.stub_backend import review_stage_state, stage_result
from tests.test_deep_orchestrator import _profile_with_pipeline, _sanctioned_inputs

StageResponse = Callable[[dict[str, Any], dict[str, Any]],
                         Iterable[AgentEvent] | AsyncGenerator[AgentEvent, None] | None]

class StagedBackend(EmptyReviewBackend):
    """A provider that remembers nothing beyond the current request."""

    supports_review_instructions = True

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
    ('missing-grounds', 'evidence_incomplete'), ('truncated-tool-output', None),
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
                     lines: int = 1, docs_count: int = 4, feature_message: str = 'feature') -> InvestigationRun:
    before = {'api.py': "def hello():\n    return 'world'\n"}
    before.update({f'module_{i:02}.py': 'VALUE = 0\n' for i in range(count - 1)})
    before.update({f'00_guide_{i}.md': '# Guide\nExisting behavior\n' for i in range(docs_count)})
    after = {name: content + ''.join(f'# changed {i}\n' for i in range(lines)) for name, content in before.items()}
    repo = tmp_path / 'many_files'
    seed_feature_branch(repo, base=before, feature=after, feature_message=feature_message)
    return InvestigationRun(repo, tmp_path, patch)


@pytest.mark.parametrize('scope_id', ['python', 'generic', 'structure'])
async def test_useful_completed_reads_borrow_cumulative_capacity_and_allow_later_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope_id: str) -> None:
    review = many_file_review(tmp_path, monkeypatch, docs_count=9)
    stage_spend: list[int] = []

    @review.backend.script(scope_id)
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        initial = not stage_spend
        starts = stage['advisory_tool_call_target'] + 1 if initial else 1
        stage_spend.append(starts)
        assert stage['observed_tool_starts'] == sum(stage_spend[:-1])
        assert stage['remaining_tool_calls'] == 48 - sum(stage_spend[:-1])
        path = stage['assigned_files'][0]
        for index in range(starts):
            yield from read_events(review.repo, path, event_id=f"stage-{len(stage_spend)}-read-{index}")
        if initial:
            line = 3 if scope_id == 'generic' else 2
            finding = dict(record(line), file=path, evidence=f'{path}:{line}')
            output['candidates'] = [dict(candidate(disposition='confirmed', finding=finding),
                                         file=path, line=line, grounds=finding['evidence'])]
            if scope_id == 'structure':
                output['candidates'].append(dict(candidate(), file=path, line=line))
        elif stage['stage'] == 'triage':
            output['candidates'] = [dict(item, disposition='rejected') for item in stage['candidates']]

    await review.finish(scope_id, findings=('Grounded defect',))
    stages = [stage for stage in review.backend.stages if stage['scope_id'] == scope_id]
    metadata = [event['metadata'] for event in stage_ends(review, scope_id)]
    assert len(stages) > 1 and stage_spend[0] > stages[0]['advisory_tool_call_target']
    assert [item['observed_tool_starts'] for item in metadata] == [
        sum(stage_spend[:index + 1]) for index in range(len(stage_spend))]
    assert metadata[-1]['remaining_tool_calls'] == 48 - sum(stage_spend)
    assert all(item['admitted'] for item in metadata)


@pytest.mark.parametrize(('sandbox', 'long_grounds'), [(False, False), (True, False), (False, True)],
                         ids=['exact-path-inputs', 'inline-inputs', 'clipped-closed-conclusion'])
async def test_stage_builders_preserve_transport_and_bounded_semantic_handoffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, sandbox: bool, long_grounds: bool) -> None:
    ext_dir.write_module(
        "import json\n"
        "from daydream.deep.prompts import build_per_stack_prompt, build_structural_prompt\n"
        "def wrap(builder, kw):\n"
        "    state = kw['review_stage']\n"
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
        assert set(call['output_schema']['properties']) == {'targets', 'notes', 'candidates', 'contradictions'}
        assert 'with issues' not in prose
        assert 'Advisory' in prose or 'advisory' in prose
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
                rejected = dict(candidate(disposition='rejected'),
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
                assert stage['candidates'] == []
                instruction = review.backend.calls[-1]['review_instructions']
                assert 'The greeting is allowed by the updated consumer contract.' in instruction
                assert 'Host-retained receipt authority' not in instruction
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
    no_tool_turns = 0

    @backend.script()
    async def response(stage: dict[str, Any], output: dict[str, Any]) -> AsyncGenerator[AgentEvent, None]:
        nonlocal no_tool_turns
        prompt = backend.calls[-1]['prompt']
        if stop == 'deadline':
            allowance = re.search(r'Hard reviewer allowance: at most ([\d.e+-]+) seconds', prompt)
            assert allowance is not None
            dispatch_deadlines.append(asyncio.get_running_loop().time() + float(allowance[1]))
        yield RequestEvent(prompt=prompt, output_schema=stage['response_contract']['schema'],
                           config=PiRequestConfig(schema_emulated=False, no_tools=False))
        if not stage['progress']:
            for path in stage['assigned_files']:
                for event in read_events(review.repo, path, event_id=f'admitted-{path}'):
                    yield event
            assert "return 'world'" in (review.repo / 'api.py').read_text()
            output['candidates'] = [dict(candidate(disposition='confirmed', finding=record()),
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
                await asyncio.sleep(1)
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
    admitted, stopped = stage_ends(review, 'python')
    assert admitted['status'] == 'succeeded'
    assert admitted['metadata']['observed_tool_starts'] > 0
    assert stopped['metadata']['observed_tool_starts'] == (
        49 if stop == 'tool' else admitted['metadata']['observed_tool_starts']
        + (0 if stop in {'builder', 'deadline'} else 1))
    assert scopes(data)['python']['partial_evidence'] is True
    assert (stopped['status'], stopped.get('reason_code')) == (
        ('timed_out', 'timed_out') if stop == 'deadline' else ('failed', 'domain_failure'))
    assert [stage['stage'] for stage in backend.stages if stage['scope_id'] == 'python'] == (
        ['first_pass'] if stop == 'builder' else ['first_pass'] * 2)
    if stop == 'deadline':
        assert no_tool_turns > 1
        assert stopped['metadata']['native_output'] is True
        assert stopped['metadata']['submission_starts'] == 0
        assert stopped['metadata']['admitted'] is False
        assert len(dispatch_deadlines) == 2
        # The prompt allowance precedes provider entry and is therefore an
        # upper bound on the deadline. Variable dispatch overhead may lower
        # the second estimate, but must not extend the first absolute bound.
        assert dispatch_deadlines[1] <= dispatch_deadlines[0] + 0.5
        assert asyncio.get_running_loop().time() >= dispatch_deadlines[0] - 0.5


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
            output['candidates'] = [candidate(disposition='rejected'), candidate()]
            assert validator.is_valid(output)
            invented = dict(output, candidates=[dict(candidate(), candidate_id='invented-native-id')])
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
                output['contradictions'] = [closed_id]
            elif decision == 'unknown-contradiction':
                output['contradictions'] = ['foreign-candidate']
            elif decision == 'extra-target':
                output['targets'] = [{'target_id': 'api.py', 'status': 'reviewed', 'reason': ''}]
            if decision in {'rejected', 'extra-target'}:
                assert validator.is_valid(output) is (decision == 'rejected')

    await investigation.finish('python', reason=reason,
                               findings=('Grounded defect',) if decision == 'confirmed' else ())
    assert [s['stage'] for s in backend.stages if s['scope_id'] == 'python'] == ['first_pass', 'triage']
    assert len(contracts) == 2 and all(contract['schema'] and contract['skeleton'] for contract in contracts)
    events = stage_ends(investigation, 'python')
    assert [event['metadata']['attempt'] for event in events] == [1, 1]
    if decision in {'rejected', 'extra-target'}:
        assert events[-1]['metadata']['failure_class'] == (
            'assignment_mismatch' if decision == 'extra-target' else None)


@pytest.mark.parametrize(('confirmed', 'fault'), [
    (False, None), (True, None), (False, 'cutoff'), (True, 'cutoff'), (False, 'syntax')])
async def test_integration_retains_admitted_findings_and_cumulative_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed: bool, fault: str | None) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=41, lines=35)
    cutoff = fault == 'cutoff'
    backend = review.backend
    observations: list[ToolStartEvent] = []

    @backend.script('structure', terminal=False)
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['stage'] == 'integration':
            starts = 0 if fault == 'syntax' else 18
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
            assert stage['observed_tool_starts'] == 18
            assert stage['remaining_tool_calls'] == 78
            pending, = stage['candidates']
            assert pending['disposition'] == 'open'
            assert stage['assigned_candidate_ids'] == [pending['candidate_id']]
            output['candidates'] = [candidate(disposition='rejected', candidate_id=pending['candidate_id'])]
        for i in range(starts):
            name = ('api.py', 'module_00.py')[i] if stage['stage'] == 'integration' and i < 2 else 'api.py'
            event = ToolStartEvent(id=f'observed-{len(observations)}', name='Read', input={'file_path': name})
            observations.append(event)
            yield event
            if stage['stage'] == 'integration':
                yield ToolResultEvent(id=event.id, output=(review.repo / name).read_text(), is_error=False)
        if stage['stage'] == 'integration' and confirmed:
            output['candidates'].append(dict(candidate(disposition='confirmed', finding=record()),
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
        assert phases['alternatives']['status'] == 'failed'
        assert phases['alternatives']['reason_codes'] == ['malformed_output']
        event, = metadata
        assert event['attempt'] == 1 and event['observed_tool_starts'] == 0
        assert event['failure_class'] == 'syntax_failure'
        coverage = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
        assert coverage['diagnostics']['scopes']['structure'] in coverage['diagnostics']['phases']['alternatives']
    else:
        assert phases['alternatives']['status'] == ('failed' if cutoff else 'complete')
        assert len(observations) == (97 if cutoff else 18)
        assert metadata[-1]['observed_tool_starts'] == (97 if cutoff else 18)
        assert metadata[-1]['remaining_tool_calls'] == (0 if cutoff else 78)
        stages = [stage for stage in backend.stages if stage['scope_id'] == 'structure']
        assert [stage['stage'] for stage in stages] == ['integration', 'triage']
        assert all(stage['advisory_tool_call_target'] <= stage['remaining_tool_calls'] for stage in stages)


async def test_failed_retry_discards_poisoned_progress_but_charges_every_observed_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    await review.finish('python')
    assert attempts == 2
    assert stage_ends(review, 'python')[0]['metadata']['observed_tool_starts'] == 6
    assert stage_ends(review, 'python')[-1]['metadata']['observed_tool_starts'] == 6
    assert stage_ends(review, 'python')[-1]['metadata']['remaining_tool_calls'] == 90
    python_stages = [stage for stage in backend.stages if stage['scope_id'] == 'python']
    assert all(stage['stage'] == 'first_pass' for stage in python_stages)
    assert len(python_stages) > 2
    assert all(len(stage['assigned_files']) <= 4 for stage in python_stages)


async def test_runner_cancellation_closes_stream_and_does_not_admit_an_aborted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    review = many_file_review(tmp_path, monkeypatch)
    checkpoint_emitted = asyncio.Event()

    class CancellationBackend(StagedBackend):
        cancel_done = False

        async def cancel(self) -> None:
            await asyncio.sleep(0)
            self.cancel_done = True

    backend = CancellationBackend(review.repo)
    review.backend = backend
    closed = asyncio.Event()

    @backend.script()
    async def response(stage: dict[str, Any], output: dict[str, Any]) -> AsyncGenerator[AgentEvent, None]:
        if stage['progress']:
            try:
                for event in read_events(review.repo, 'api.py', event_id='cancelled-read'):
                    yield event
                output['candidates'] = [candidate(disposition='confirmed', finding=dict(
                    record(), description='Aborted checkpoint'))]
                yield ResultEvent(structured_output=output, continuation=None)
                checkpoint_emitted.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
            return
        for path in stage['assigned_files']:
            for event in read_events(review.repo, path, event_id=f'admitted-{path}'):
                yield event
        output['candidates'] = [candidate(disposition='confirmed', finding=record())]

    task = asyncio.create_task(review.run())
    try:
        await asyncio.wait_for(checkpoint_emitted.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert closed.is_set() and backend.cancel_done
    completed, interrupted = stage_ends(review, 'python')
    assert completed['metadata']['admitted'] is True
    assert interrupted['metadata']['admitted'] is False
    assert interrupted['metadata']['observed_tool_starts'] == 5
    assert not review.output.exists()
