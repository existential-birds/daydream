"""Bounded review stages through runner.run, real Git and the external Backend seam."""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, MaxTurnsError, ResultEvent, ToolResultEvent, ToolStartEvent
from tests.conftest import ExtDir
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.deep_orchestrator.test_review_completion import ReviewRun, record, scopes
from tests.harness.git_helpers import seed_feature_branch
from tests.harness.stub_backend import completed_stage_reads, review_stage_state, stage_result
from tests.test_deep_orchestrator import _profile_with_pipeline, _sanctioned_inputs


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
        self.default_source_reads = True

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        stage = review_stage_state(prompt)
        if stage is not None:
            self.stages.append(stage)
            self.calls.append({'prompt': prompt, 'output_schema': args[0] if args else kwargs.get('output_schema'),
                               **kwargs})
            if self.stage_delay is not None:
                await asyncio.sleep(self.stage_delay(stage))
            output = stage_result(stage)
            emitted = False
            if self.stage_response is not None:
                for event in self.stage_response(stage, output) or ():
                    emitted = True
                    yield event
            if not emitted and self.default_source_reads:
                for event in completed_stage_reads(cwd, stage):
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
    ('unmatched-output', 'evidence_incomplete'),
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
        elif fault == 'unmatched-output':
            yield ToolResultEvent(id='unmatched-source', is_error=False,
                                  output=(investigation.repo / 'api.py').read_text())

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
                     lines: int = 1, wired: bool = True, feature_message: str = 'feature') -> InvestigationRun:
    before = {'api.py': "def hello():\n    return 'world'\n",
              'a_flags.py': 'def parse_flags(args):\n    return {}\n',
              'z_request.py': "def build_request(options):\n    return {'legacy': False}\n"}
    before.update({f'module_{i:02}.py': 'VALUE = 0\n' for i in range(count - 3)})
    before.update({f'00_guide_{i}.md': '# Guide\nExisting behavior\n' for i in range(4)})
    after = {name: content + ''.join(f'# changed {i}\n' for i in range(lines)) for name, content in before.items()}
    after['a_flags.py'] = "def parse_flags(args):\n    return {'dry_run': '--dry-run' in args}\n"
    after['z_request.py'] = ('def build_request(options):\n'
                             + ("    return {'dry_run': options['dry_run']}\n" if wired else '    return {}\n'))
    repo = tmp_path / 'many_files'
    seed_feature_branch(repo, base=before, feature=after, feature_message=feature_message)
    return InvestigationRun(repo, tmp_path, patch)


@pytest.mark.parametrize('scope_id', ['python', 'generic', 'structure'])
async def test_useful_completed_reads_borrow_cumulative_capacity_and_allow_later_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope_id: str,
) -> None:
    api_body = (
        "from module_00 import greeting\n\ndef hello():\n"
        "    expected = greeting()\n"
        "    boundary = {\n"
        "        'encoding': 'utf8',\n"
        "        'source': 'cli',\n"
        "        'contract': 'world',\n"
        "    }\n"
        "    if boundary['encoding'] != 'utf8':\n"
        "        raise ValueError(boundary['source'])\n"
    )
    before = {'api.py': api_body + "    return expected\n"}
    for index in range(17):
        next_import = (f"from module_{index + 1:02} import greeting\n" if index < 16 else '')
        value = 'greeting()' if index < 16 else "'world'"
        before[f'module_{index:02}.py'] = (
            next_import + "\ndef greeting():\n    format_options = {\n"
            "        'template': '{}',\n"
            "        'encoding': 'utf8',\n"
            "        'error': 'unsupported encoding',\n"
            "    }\n"
            "    if format_options['encoding'] != 'utf8':\n"
            "        raise ValueError(format_options['error'])\n"
            f"    value = {value}\n"
            "    return format_options['template'].format(value)\n"
        )
    guide = (
        '# Greeting contract\nhello() returns world.\n\n'
        '## Invocation\nCall hello() without arguments.\n\n'
        '## Encoding\nThe greeting uses utf8.\n\n'
        '## Failure handling\nAn unsupported encoding raises ValueError.\n'
    )
    before.update({f'guide_{i:02}.md': guide for i in range(9)})
    after = {path: text + '# changed boundary\n' for path, text in before.items()}
    after['api.py'] = api_body + "    return 'universe'\n"
    repo = tmp_path / 'useful_work'
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != scope_id:
            return
        if stage['stage'] == 'triage':
            for path in stage['assigned_files']:
                source = (repo / path).read_text()
                yield ToolStartEvent(id=f'triage-{path}', name='Read',
                                    input={'file_path': path, 'offset': 1, 'limit': len(source.splitlines())})
                yield ToolResultEvent(id=f'triage-{path}', output=source, is_error=False)
                assert "return 'universe'" in source
            output['candidates'] = [dict(item, disposition='rejected',
                                        grounds='api.py:12 is the greeting-contract defect already confirmed; '
                                        'the alleged return at api.py:2 is absent.')
                                    for item in stage['candidates']]
            return
        if stage['progress']:
            output['targets'] = []
            for path in stage['assigned_files']:
                source = (repo / path).read_text()
                yield ToolStartEvent(id=f'later-{path}', name='Read',
                                    input={'file_path': path, 'offset': 1, 'limit': len(source.splitlines())})
                yield ToolResultEvent(id=f'later-{path}', output=source, is_error=False)
                assert ('def greeting():' in source if path.endswith('.py') else 'hello() returns world.' in source)
                output['targets'].append({'target_id': path, 'status': 'reviewed', 'reason': ''})
            return
        # Each assigned first-pass file gets complete source in disjoint bounded
        # segments. Extra reads follow the concrete hello/greeting boundary.
        assigned = stage['assigned_files']
        paths = (sorted(path for path in assigned if path.endswith('.py')) if scope_id == 'structure'
                 else assigned + (['api.py'] if scope_id == 'generic' else ['guide_00.md']))
        observed = 0
        captured: dict[str, str] = {}
        for path in paths:
            source = (repo / path).read_text()
            lines = source.splitlines(keepends=True)
            count = 1 if scope_id == 'structure' else 4 if path in assigned else 2
            segments = [(1 + index * len(lines) // count,
                         lines[index * len(lines) // count:(index + 1) * len(lines) // count])
                        for index in range(count)]
            for offset, segment in segments:
                yield ToolStartEvent(id=f'useful-{observed}', name='Read',
                                    input={'file_path': path, 'offset': offset, 'limit': len(segment)})
                excerpt = ''.join(segment)
                yield ToolResultEvent(id=f'useful-{observed}', output=excerpt, is_error=False)
                captured[path] = captured.get(path, '') + excerpt
                observed += 1
            assert captured[path] == source
        if scope_id == 'structure':
            yield ToolStartEvent(id='documented-contract', name='Read', input={'file_path': 'guide_00.md'})
            yield ToolResultEvent(id='documented-contract', output=(repo / 'guide_00.md').read_text(), is_error=False)
            observed += 1
        assert observed == (19 if scope_id == 'structure' else 18)
        assert "return 'universe'" in (repo / 'api.py').read_text()
        path = 'guide_00.md' if scope_id == 'generic' else 'api.py'
        line = 2 if scope_id == 'generic' else 12
        finding = dict(record(), file=path, line=line,
                       description='Greeting breaks the documented world contract',
                       evidence=f'{path}:{line}; api.py:12 returns universe; guide_00.md:2 promises world')
        output['candidates'] = [dict(candidate(disposition='confirmed', finding=finding),
                                     file=path, line=line, grounds=finding['evidence'])]
        if scope_id == 'structure':
            output['candidates'].append(candidate())

    review.backend.stage_response = response
    await review.finish(scope_id, findings=('Greeting breaks the documented world contract',))
    stages = [stage for stage in review.backend.stages if stage['scope_id'] == scope_id]
    spent = 19 if scope_id == 'structure' else 18
    assert stage_ends(review, scope_id)[0]['metadata']['observed_tool_starts'] == spent
    assert stage_ends(review, scope_id)[0]['metadata']['remaining_tool_calls'] == 48 - spent
    assert len(stages) > 1
    assert stages[1]['observed_tool_starts'] == spent
    completed_stages = stage_ends(review, scope_id)
    assert all(later['metadata']['observed_tool_starts'] > earlier['metadata']['observed_tool_starts']
               for earlier, later in zip(completed_stages, completed_stages[1:], strict=False))
    assert completed_stages[-1]['metadata']['observed_tool_starts'] == {
        'python': 32, 'generic': 23, 'structure': 20,
    }[scope_id]


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-path-inputs', 'inline-inputs'])
async def test_each_stage_builder_receives_current_assignment_and_bounded_triage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, sandbox: bool,
) -> None:
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
    review = many_file_review(tmp_path, monkeypatch,
                             feature_message='fix: SETTLED_UNRELATED_HISTORY\n\nDaydream-Run: previous-fixture')
    review.backend.sandbox = sandbox

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        call = review.backend.calls[-1]
        prose = call['prompt'].split('Host review stage:\n')[0]
        assert set(call['output_schema']['properties']) == {'targets', 'notes', 'candidates', 'contradictions'}
        assert 'with issues' not in prose
        assert 'Advisory' in prose or 'advisory' in prose
        assert 'Error Handling Semantics (QUAL-04)' in prose
        assert 'Confidence and Convention Rules' in prose
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
                assert set(pointers) == {label for label in stage['context_inputs']
                                         if not label.startswith('source-')}
                for label in ('diff', 'hunk-index'):
                    path = pointers[label]
                    yield ToolStartEvent(id=f'context-{label}', name='Read', input={'file_path': str(path)})
                    yield ToolResultEvent(id=f'context-{label}', output=path.read_text(), is_error=False)
            assigned_line = next(line for line in prose.splitlines() if 'Assigned files:' in line)
            assert assigned_line.split('Assigned files:', 1)[1].strip() == ', '.join(stage['assigned_files'])
            assert stage['candidates'] == []
            for index, path in enumerate(stage['assigned_files']):
                yield ToolStartEvent(id=f'bounded-{index}', name='Read', input={'file_path': path})
                yield ToolResultEvent(id=f'bounded-{index}', output=(review.repo / path).read_text(), is_error=False)
            if 'api.py' in stage['assigned_files']:
                output['candidates'] = [candidate(disposition='rejected')] + [candidate() for _ in range(9)]
        else:
            assert stage['stage'] == 'triage'
            assert 'SETTLED_UNRELATED_HISTORY' not in prose
            assert [item['candidate_id'] for item in stage['candidates']] == stage['assigned_candidate_ids']
            assert all(item['disposition'] == 'open' for item in stage['candidates'])
            assert 'Assigned files:' not in prose
            assert 'api.py' in stage['assigned_files']
            assert all(label.startswith('source-') for label in stage['context_inputs'])
            assert ('Sanctioned phase inputs (read only these exact files):' not in call['prompt']) is sandbox
            output['candidates'] = [dict(item, disposition='rejected') for item in stage['candidates']]

    review.backend.stage_response = response
    data = await review.finish('python')
    assert all(scope['status'] == 'complete' for scope in scopes(data).values())
    stages = [stage for stage in review.backend.stages if stage['scope_id'] == 'python']
    assert len([stage for stage in stages if stage['stage'] == 'first_pass']) > 1
    assert [stage['stage'] for stage in stages].count('triage') == 2
    assert [len(stage['assigned_candidate_ids']) for stage in stages if stage['stage'] == 'triage'] == [8, 1]


@pytest.mark.parametrize(('stop', 'reason'), [('tool', 'host_tool_budget_exhaustion'),
    ('model', 'model_budget_exhaustion'), ('backend', 'backend_failure'),
    ('builder', 'backend_failure'),
    ('deadline', 'host_pipeline_budget_exhaustion')])
async def test_unsuccessful_later_stage_retains_only_prior_admitted_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, stop: str, reason: str,
) -> None:
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

    if stop == 'deadline':
        def delay_later_provider(stage: dict[str, Any]) -> float:
            if stage['scope_id'] != 'python':
                return 0
            allowance = re.search(r'Hard reviewer allowance: at most ([\d.e+-]+) seconds',
                                  backend.calls[-1]['prompt'])
            assert allowance is not None
            dispatch_deadlines.append(asyncio.get_running_loop().time() + float(allowance[1]))
            # Leave real Git preparation and the first admission unstalled. The
            # later provider alone outlives the existing absolute deadline.
            return deadline_test_wall_s * 2 if stage['progress'] else 0

        backend.stage_delay = delay_later_provider

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] == 'python':
            if not stage['progress']:
                output['targets'] = []
                for path in stage['assigned_files']:
                    source = (review.repo / path).read_text()
                    yield ToolStartEvent(id=f'admitted-{path}', name='Read',
                                        input={'file_path': path, 'offset': 1, 'limit': len(source.splitlines())})
                    yield ToolResultEvent(id=f'admitted-{path}', output=source, is_error=False)
                    output['targets'].append({'target_id': path, 'status': 'reviewed', 'reason': ''})
                assert "return 'world'" in (review.repo / 'api.py').read_text()
                output['candidates'] = [dict(candidate(disposition='confirmed', finding=record()),
                                             grounds="api.py:1-2 defines hello() and returns 'world'.")]
                return
            yield ResultEvent(structured_output=stage_result(stage, candidates=[
                candidate(disposition='confirmed', finding=dict(record(), description='Unadmitted defect'))]),
                continuation=None)
            if stop == 'tool':
                for i in range(stage['remaining_tool_calls'] + 1):
                    yield ToolStartEvent(id=f'cutoff-{i}', name='Read', input={'file_path': 'api.py'})
            else:
                raise MaxTurnsError('model exhausted') if stop == 'model' else RuntimeError('backend unavailable')

    backend.stage_response = response
    overrides: dict[str, Any] = (
        {'review_profile': _profile_with_pipeline(review_wall_budget_s=deadline_test_wall_s)}
        if stop == 'deadline' else {}
    )
    data = await review.finish('python', reason=reason, findings=('Grounded defect',), **overrides)
    assert scopes(data)['structure']['status'] == 'complete'
    admitted, stopped = stage_ends(review, 'python')
    assert admitted['status'] == 'succeeded'
    assert admitted['metadata']['observed_tool_starts'] > 0
    assert stopped['metadata']['observed_tool_starts'] == (
        49 if stop == 'tool' else admitted['metadata']['observed_tool_starts'])
    assert scopes(data)['python']['partial_evidence'] is True
    assert (stopped['status'], stopped.get('reason_code')) == (
        ('timed_out', 'timed_out') if stop == 'deadline' else ('failed', 'domain_failure'))
    assert [stage['stage'] for stage in backend.stages if stage['scope_id'] == 'python'] == (
        ['first_pass'] if stop == 'builder' else ['first_pass'] * 2)
    if stop == 'deadline':
        assert len(dispatch_deadlines) == 2
        assert dispatch_deadlines[1] == pytest.approx(dispatch_deadlines[0], abs=0.5)
        assert asyncio.get_running_loop().time() >= dispatch_deadlines[0] - 0.5


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
                pending, = stage['candidates']
                closed_id, = stage['closed_candidate_ids']
                assert stage['progress'] == []
                assert closed_id != pending['candidate_id']
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
                    output['candidates'][0]['candidate_id'] = closed_id
                elif decision == 'duplicate':
                    output['candidates'] *= 2
                elif decision == 'omitted':
                    output['candidates'] = []
                elif decision == 'contradiction':
                    output['contradictions'] = [closed_id]
                elif decision == 'unknown-contradiction':
                    output['contradictions'] = ['foreign-candidate']

    backend.stage_response = response
    await investigation.finish('python', reason=reason,
                               findings=('Grounded defect',) if decision == 'confirmed' else ())
    assert [s['stage'] for s in backend.stages if s['scope_id'] == 'python'] == ['first_pass', 'triage']


@pytest.mark.parametrize('cutoff', [False, True])
@pytest.mark.parametrize('wired', [False, True])
async def test_cross_file_integration_preserves_cumulative_spend_and_flag_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cutoff: bool, wired: bool,
) -> None:
    review = many_file_review(tmp_path, monkeypatch, count=41, lines=35, wired=wired)
    backend = review.backend
    observations: list[ToolStartEvent] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] == 'structure':
            if stage['stage'] == 'integration':
                starts = 18
                assert stage['observed_tool_starts'] == 0
                assert stage['remaining_tool_calls'] == 96
                assert stage['assigned_target_ids'] == ['integration:structure']
                assert stage['progress'] == []
                assert {'a_flags.py', 'z_request.py'} <= set(stage['assigned_files'])
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
                name = ('a_flags.py', 'z_request.py')[i] if stage['stage'] == 'integration' and i < 2 else 'api.py'
                event = ToolStartEvent(id=f'observed-{len(observations)}', name='Read',
                                       input={'file_path': name})
                observations.append(event)
                yield event
                if stage['stage'] == 'integration':
                    yield ToolResultEvent(id=event.id, output=(review.repo / name).read_text(), is_error=False)
            if (stage['stage'] == 'integration'
                    and "options['dry_run']" not in (review.repo / 'z_request.py').read_text()):
                finding = dict(record(), file='z_request.py', line=2,
                               description='Parsed dry-run flag is dropped during request construction',
                               evidence='a_flags.py:2 parses dry_run; z_request.py:2 returns an empty request')
                output['candidates'].append(dict(candidate(disposition='confirmed', finding=finding),
                    file='z_request.py', trigger='Pass --dry-run through parse_flags and build_request',
                    consequence='Request omits dry_run and performs a live operation', grounds=finding['evidence']))

    backend.stage_response = response
    data = await review.finish('structure', reason='host_tool_budget_exhaustion' if cutoff else None,
                              findings=() if wired else ('Parsed dry-run flag is dropped during request construction',))
    assert set(scopes(data)) == {'python', 'generic', 'structure'}
    assert len(observations) == (97 if cutoff else 18)
    metadata = [event['metadata'] for event in stage_ends(review, 'structure')]
    assert metadata[-1]['observed_tool_starts'] == (97 if cutoff else 19)
    assert metadata[-1]['remaining_tool_calls'] == (0 if cutoff else 77)
    stages = [stage for stage in backend.stages if stage['scope_id'] == 'structure']
    assert [stage['stage'] for stage in stages] == ['integration', 'triage']
    assert all(stage['advisory_tool_call_target'] <= stage['remaining_tool_calls'] for stage in stages)


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
    data = await review.finish('python')
    assert attempts == 2
    assert stage_ends(review, 'python')[0]['metadata']['observed_tool_starts'] == 6
    assert stage_ends(review, 'python')[-1]['metadata']['observed_tool_starts'] == 19
    assert stage_ends(review, 'python')[-1]['metadata']['remaining_tool_calls'] == 77
    assert all(scope['status'] == 'complete' for scope in scopes(data).values())
    python_stages = [stage for stage in backend.stages if stage['scope_id'] == 'python']
    assert all(stage['stage'] == 'first_pass' for stage in python_stages)
    assert len(python_stages) > 2
    assert all(len(stage['assigned_files']) <= 4 for stage in python_stages)


async def test_runner_cancellation_closes_stream_and_does_not_admit_an_aborted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = many_file_review(tmp_path, monkeypatch)
    checkpoint_emitted = asyncio.Event()

    class CancellationBackend(StagedBackend):
        closed = False
        cancel_done = False

        async def cancel(self) -> None:
            await asyncio.sleep(0)
            self.cancel_done = True

        async def execute(self, cwd: Path, prompt: str, *args: Any,
                          **kwargs: Any) -> AsyncIterator[AgentEvent]:
            state = review_stage_state(prompt)
            if state is not None and state['scope_id'] == 'python' and state['progress']:
                try:
                    yield ToolStartEvent(id='cancelled-read', name='Read', input={'file_path': 'api.py'})
                    yield ToolResultEvent(id='cancelled-read', output=(review.repo / 'api.py').read_text(),
                                          is_error=False)
                    yield ResultEvent(structured_output=stage_result(state, candidates=[
                        candidate(disposition='confirmed', finding=dict(record(), description='Aborted checkpoint'))
                    ]), continuation=None)
                    checkpoint_emitted.set()
                    await asyncio.Event().wait()
                finally:
                    self.closed = True
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    backend = CancellationBackend(review.repo)

    def admitted(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] == 'python':
            for path in stage['assigned_files']:
                yield ToolStartEvent(id=f'admitted-{path}', name='Read', input={'file_path': path})
                yield ToolResultEvent(id=f'admitted-{path}', output=(review.repo / path).read_text(), is_error=False)
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]

    backend.stage_response = admitted
    review.backend = backend
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
    assert backend.closed and backend.cancel_done
    completed, interrupted = stage_ends(review, 'python')
    assert completed['metadata']['admitted'] is True
    assert interrupted['metadata']['admitted'] is False
    assert interrupted['metadata']['observed_tool_starts'] == 5
    assert not review.output.exists()
