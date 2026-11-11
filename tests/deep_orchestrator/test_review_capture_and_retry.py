"""Large staged-review transport and retry scenarios through the production runner."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from daydream.backends import (
    AgentEvent,
    ResultEvent,
    TextEvent,
    TurnEndEvent,
)
from daydream.backends.codex import CodexBackend
from daydream.config import REVIEWER_TOOL_CALL_BUDGET
from tests.conftest import ExtDir
from tests.deep_orchestrator.test_review_completion import record, scopes
from tests.deep_orchestrator.test_review_investigation import (
    InvestigationRun,
    assert_host_dispatch_measurements,
    candidate,
    read_events,
    stage_ends,
)
from tests.harness.git_helpers import seed_feature_branch
from tests.test_deep_orchestrator import _sanctioned_inputs


def staged_review(tmp: Path, patch: pytest.MonkeyPatch, *, files: int = 8) -> InvestigationRun:
    before = {f'module_{index:02}.py': 'VALUE = 0\n' for index in range(files - 1)}
    before['api.py'] = "def hello():\n    return 'world'\n"
    after = {name: body.replace('VALUE = 0', 'VALUE = 1') for name, body in before.items()}
    after['api.py'] = before['api.py'].replace("return 'world'", "return 'universe'")
    repo = tmp / 'capture_sources'
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp, patch)
    # Retry/admission journeys share one logical reviewer and cumulative stage allowance.
    review.config = replace(review.config, deep_shard_enabled=False)
    return review


def supporting_contents(prompt: str) -> dict[str, str]:
    """The scripted provider consumes whole bounded supporting sections on either transport."""

    captured = dict(re.findall(r'<sanctioned-input label="([^"]+)">\n(.*?)\n</sanctioned-input>', prompt, re.S))
    if 'Sanctioned phase inputs (read only these exact files):' in prompt:
        captured.update({label: path.read_text() for label, path in _sanctioned_inputs(prompt).items()})
    bundle = captured.get('review-assignment')
    if bundle is not None:
        sections = list(re.finditer(r'^### ([^\n]+) \(supporting\)\n', bundle, re.M))
        for index, section in enumerate(sections):
            body = (bundle[section.end():sections[index + 1].start()] if index + 1 < len(sections)
                    else bundle[section.end():])
            captured[section[1]] = body.removesuffix('\n\n') if index + 1 < len(sections) else body
    return captured


def capture(review: InvestigationRun, stage: dict[str, Any], *, bundle: bool = True) -> tuple[str, dict[str, Any]]:
    """Read the delivered assignment and bind its bounded index to the current files."""
    contents = supporting_contents(review.backend.calls[-1]['prompt'])
    index = json.loads(contents['hunk-index'])
    assert set(index) == set(stage['assigned_files'])
    if bundle and not review.backend.sandbox:
        assert len(Path(stage['supporting_bundle']['path']).read_bytes()) <= 24 * 1024
    return contents['diff'], index


def canonical_decisions(index: dict[str, Any], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve exact (null-range, supplied-diff) selectors to the references in the delivered hunk index."""
    delivered = {option['source_id']: reference for entry in index.values() for part in entry['assignments']
                 for option, reference in zip(part['source_options'], part['source_refs'], strict=True)}
    assert all(selector['source_range'] is None and selector['basis'] == 'supplied_diff'
               for decision in decisions for selector in decision['evidence_refs'])
    return [{**decision, 'evidence_refs': [delivered[selector['source_id']] for selector in decision['evidence_refs']]}
            for decision in decisions]


@pytest.mark.parametrize('invalid_selection', [False, True], ids=['retry', 'unpreparable-foreign-source'])
async def test_schema_rejection_retries_once_with_fresh_grounding_and_charged_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, invalid_selection: bool) -> None:
    """A schema-rejected stage retries once with fresh grounding; both attempts charge their tool starts.

    An unknown selection still blocks the retry when other fields leave the payload unpreparable.
    """
    ext_dir.write_module(
        'from daydream.deep.prompts import build_per_stack_prompt\n'
        'def staged(**kw):\n'
        '    return build_per_stack_prompt(**kw) + \"\\nCUSTOM_UTF8=é🙂 CUSTOM_ATTEMPT=\" '
        '+ str(kw[\"review_stage\"][\"attempt\"])\n'
        'def register(r): r.override_prompt(\"per-stack\", staged)\n',
    )
    review = staged_review(tmp_path, monkeypatch)
    attempts = 0
    initial_options: list[dict[str, Any]] = []

    @review.backend.script('python', terminal=False)
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal attempts
        initial_assignment = 'api.py' in stage['assigned_files']
        if initial_assignment:
            attempts += 1
            assert stage['attempt'] == attempts and stage['max_attempts'] == 2
            assert f'CUSTOM_ATTEMPT={attempts}' in review.backend.calls[-1]['prompt']
            options = [option for part in stage['assignment_parts'] for option in part['source_options']]
            if attempts == 1:
                initial_options.extend(copy.deepcopy(options))
            else:
                assert options == initial_options
        for path in stage['assigned_files']:
            yield from read_events(review.repo, path, event_id=f'attempt-{attempts}-{path}')
        if not initial_assignment:
            assert all('FAILED_ATTEMPT_MARKER' not in str(item) for item in stage['notes'])
        elif invalid_selection:
            del output['notes']
            output['targets'][0]['evidence_refs'] = [
                {'source_id': 'source:' + '0' * 64, 'basis': 'supplied_diff', 'source_range': None}]
        elif attempts == 1:
            output['targets'][0]['PRIVATE_UNKNOWN_PROPERTY_MARKER'] = None
            output['notes'] = 'FAILED_ATTEMPT_MARKER'
            output['candidates'] = [candidate(disposition='confirmed', finding=dict(
                record(), description='FAILED_ATTEMPT_MARKER'))]
        else:
            assert stage['progress'] == stage['candidates'] == []
            assert stage['observed_tool_starts'] == 4 and stage['remaining_tool_calls'] == REVIEWER_TOOL_CALL_BUDGET - 4
            prompt = review.backend.calls[-1]['prompt']
            assert 'PRIVATE_UNKNOWN_PROPERTY_MARKER' not in prompt and 'FAILED_ATTEMPT_MARKER' not in prompt
            rejection = stage['schema_rejection']
            assert rejection['error_count'] > 0 and rejection['candidate_count'] > 0
            assert set(rejection) == {'category', 'schema_path', 'error_count', 'candidate_count'}
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        yield TextEvent(text=json.dumps(output))
        yield TurnEndEvent()
        yield ResultEvent(structured_output=None, continuation=None)

    if invalid_selection:
        await review.finish('python', reason='malformed_output')
        assert attempts == 1
        assert [(event['metadata']['attempt'], event['metadata']['failure_class'])
                for event in stage_ends(review, 'python')] == [(1, 'source_attribution')]
        return
    await review.finish('python', findings=('Grounded defect',))
    assert attempts == 2
    metadata = [event['metadata'] for event in stage_ends(review, 'python')]
    assert metadata[0]['admitted'] is False and metadata[0]['attempt_tool_starts'] == 4
    assert metadata[0]['schema_rejection']['error_count'] > 0
    assert metadata[1]['attempt'] == 2 and metadata[1]['observed_tool_starts'] == 8
    assert metadata[1]['remaining_tool_calls'] == REVIEWER_TOOL_CALL_BUDGET - 8
    assert metadata[1]['admitted'] is True
    assert metadata[-1]['observed_tool_starts'] == 12
    assert metadata[-1]['remaining_tool_calls'] == REVIEWER_TOOL_CALL_BUDGET - 12
    measurements = assert_host_dispatch_measurements(review)
    assert all(item['attempt'] == 1 for item in measurements)
    retry_scopes = {item['stage_scope_id'] for item in measurements
                    if any(event['scope_id'] == item['stage_scope_id'] and event['metadata']['attempt'] == 2
                           for event in stage_ends(review, 'python'))}
    assert len(retry_scopes) == 1


@pytest.mark.parametrize('failing_open', [1, 3], ids=['shared-change', 'first-admitted-scope'])
async def test_failed_canonical_stage_input_preserves_successful_sibling_review(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    failing_open: int,
) -> None:
    """A real canonical read failure fails the scopes it reaches without hiding a sibling's evidence.

    The shared change captures and revalidates the index once; each admitted scope revalidates next.
    A failed shared capture records every planned scope as failed before any reviewer runs.
    """
    review = InvestigationRun(multi_stack_target, tmp_path, monkeypatch)
    real_read, real_open = Path.read_text, os.open
    canonical_opens = 0

    def open_input(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal canonical_opens
        if Path(path).name == 'hunk-index.json':
            canonical_opens += 1
            if canonical_opens == failing_open:
                raise OSError('PRIVATE_DIAGNOSTIC_PATH_SENTINEL')
        return real_open(path, *args, **kwargs)

    @review.backend.script('react')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        assert '<div>universe</div>' in real_read(review.repo / 'App.tsx')
        finding = dict(record(line=1), file='App.tsx', description='Surviving sibling defect',
                       evidence='App.tsx:1 renders universe instead of the greeting contract')
        output['candidates'] = [candidate(disposition='confirmed', finding=finding, file='App.tsx', line=1,
                                         grounds=finding['evidence'])]

    monkeypatch.setattr(os, 'open', open_input)
    if failing_open == 3:
        await review.finish('python', reason='malformed_artifact', statuses=('failed',),
                                   findings=('Surviving sibling defect',))
        assert canonical_opens > 3
    else:
        assert await review.run() == 0
        data = review.load()
        assert data['findings'] == [] and review.backend.stages == []
        assert {scope: (outcome['status'], outcome['reason_codes']) for scope, outcome in scopes(data).items()} == {
            scope: ('failed', ['malformed_artifact']) for scope in ('generic', 'python', 'react', 'structure')}
    assert 'PRIVATE_DIAGNOSTIC_PATH_SENTINEL' not in capsys.readouterr().out


async def test_old_only_hunk_and_later_continuations_keep_canonical_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Pi (Codex) reviewer selects sources; the host persists canonical old-only and continuation refs."""

    from tests.harness.git_helpers import git

    prefix = ''.join(f'# stable boundary {index:03}\n' for index in range(98))
    before = 'DELETED_FIRST_LINE = True\n' + prefix + 'VALUE = 0\n'
    added = ''.join(f'# LATER_REQUIRED_CHANGE_{index:04} ' + 'x' * 96 + '\n' for index in range(500))
    repo = tmp_path / 'mixed_hunk_ranges'
    seed_feature_branch(repo, base={'api.py': before}, feature={'api.py': prefix + 'VALUE = 1\n' + added})
    git(repo, 'config', 'diff.context', '0')
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend.sandbox = True
    native = CodexBackend(model=review.backend.model)
    monkeypatch.setattr(native, 'execute', review.backend.execute)
    monkeypatch.setattr('daydream.runner.create_backend', lambda *_a, **_k: native)
    assignments: list[dict[str, Any]] = []
    fragments: list[str] = []
    decisions: list[dict[str, Any]] = []
    fragment_stages: list[bool] = []
    whole_part_digests: set[str] = set()

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        diff, index = capture(review, stage)
        binding = json.loads(supporting_contents(review.backend.calls[-1]['prompt'])['input-binding'])
        assert Draft202012Validator(review.backend.calls[-1]['output_schema']).is_valid(output)
        delivered = index['api.py']['assignments']
        fragment_stages.append(all(reference['fragment'] is not None for part in delivered
                                   for reference in part['source_refs']))
        canonical = canonical_decisions(index, output['targets'])
        revision = binding['analyzed_revision']
        unmarked = re.sub(r'^\[Ordered continuation[^\n]*\n', '', diff, flags=re.M)
        delivered_part_digests = {hashlib.sha256(('diff --git ' + part).encode()).hexdigest()
                                  for part in re.split(r'^diff --git ', unmarked, flags=re.M)[1:]}
        revisions = {'old': revision['merge_base_sha'], 'new': revision.get('source_id', revision['head_sha'])}
        for reference in (reference for decision in canonical for reference in decision['evidence_refs']):
            # Digests and revisions come from delivered bytes, never from the model's selection.
            assert reference['artifact_sha256'] == binding['canonical_inputs']['canonical-diff']
            assert (reference['path'], reference['revision']) == ('api.py', revisions[reference['side']])
            if reference['fragment'] is None:
                whole_part_digests.add(reference['part_sha256'])
                assert reference['part_sha256'] in delivered_part_digests
        scoped = index['api.py']
        fragments.append(diff)
        decisions.extend(canonical)
        for part in stage['assignment_parts']:
            assignments.append(part)
            expected = (part['old_start'], part['old_start'] + part['old_count'] - 1,
                        part['new_start'], part['new_start'] + part['new_count'] - 1)
            assert any((hunk['old_start'], hunk['old_end'], hunk['new_start'], hunk['new_end']) == expected
                       for hunk in scoped['hunks'])
            if part['new_count'] != 0:
                assert (part['old_start'], part['old_count'], part['new_start'], part['new_count']) == (100, 1, 99, 501)

    # This parser contract deliberately retains a configured zero-context patch.
    # Captured review has host-owned diff formatting; the legacy loop still admits
    # this valid input and must preserve both old-only and later fragment ranges.
    assert await review.run(output_mode='loop', findings_out=None) == 0
    coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    assert all(scope['status'] == 'complete' for scope in coverage['stack_outcomes'])
    assert True in fragment_stages
    assert whole_part_digests, 'the old-only whole hunk must be checked against delivered part bytes'
    assert assignments[0]['old_start'] == 1 and assignments[0]['new_count'] == 0
    first_reference, = decisions[0]['evidence_refs']
    assert first_reference['side'] == 'old'
    assert first_reference['source_range'] == {'start': 1, 'end': 1}
    assert any(reference['side'] == 'new' for decision in decisions for reference in decision['evidence_refs'])
    expected_decisions = sorted(decisions, key=lambda item: item['target_id'])
    assert next(scope for scope in coverage['stack_outcomes'] if scope['scope_id'] == 'python')[
        'target_decisions'] == expected_decisions
    combined = '\n'.join(fragments)
    assert combined.count('-DELETED_FIRST_LINE = True') == 1
    for index in range(500):
        assert combined.count(f'LATER_REQUIRED_CHANGE_{index:04}') == 1


async def test_redaction_expansion_rejects_stage_without_losing_admitted_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret-heavy reason must not break publication of earlier accepted work."""
    review = staged_review(tmp_path, monkeypatch)
    admitted: list[dict[str, Any]] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if not stage['progress']:
            admitted.extend(canonical_decisions(capture(review, stage, bundle=False)[1], output['targets']))
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        else:
            output['targets'][0]['reason'] = 'TOKEN=x ' * 20000

    data = await review.finish('python', reason='evidence_incomplete', findings=('Grounded defect',))
    assert scopes(data)['python']['target_decisions'] == admitted
    events = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in events] == [True, False]
    # The canonical result-byte bound is an admission limit, not a source attribution fault.
    assert events[-1]['metadata']['failure_class'] == 'admission_failure'
    assert 'TOKEN=x' not in review.output.read_text()
    saved = json.loads((review.repo / '.daydream/deep/stack-python-records.json').read_text())
    assert [finding['description'] for finding in saved['issues']] == ['Grounded defect']
