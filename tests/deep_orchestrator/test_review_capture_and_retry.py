"""Large staged-review transport and retry scenarios through the production runner."""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import (
    AgentEvent,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.config_file import DaydreamFileConfig
from tests.conftest import ExtDir
from tests.deep_orchestrator.test_review_completion import record, scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, candidate, read_events, stage_ends
from tests.harness.git_helpers import seed_feature_branch
from tests.test_deep_orchestrator import _sanctioned_inputs


def staged_review(tmp: Path, patch: pytest.MonkeyPatch, *, files: int = 8) -> InvestigationRun:
    before = {f'module_{index:02}.py': 'VALUE = 0\n' for index in range(files - 1)}
    before['api.py'] = "def hello():\n    return 'world'\n"
    after = {name: body.replace('VALUE = 0', 'VALUE = 1') for name, body in before.items()}
    after['api.py'] = before['api.py'].replace("return 'world'", "return 'universe'")
    repo = tmp / 'capture_sources'
    seed_feature_branch(repo, base=before, feature=after)
    return InvestigationRun(repo, tmp, patch)


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


@pytest.mark.parametrize('outcome', ['success', 'second-rejection'])
async def test_schema_rejection_retries_once_with_fresh_grounding_and_charged_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, ext_dir: ExtDir) -> None:
    ext_dir.write_module(
        'from daydream.deep.prompts import build_per_stack_prompt\n'
        'def staged(**kw):\n'
        '    return build_per_stack_prompt(**kw) + \"\\nCUSTOM_ATTEMPT=\" + str(kw[\"review_stage\"][\"attempt\"])\n'
        'def register(r): r.override_prompt(\"per-stack\", staged)\n', api_version=8,
    )
    review = staged_review(tmp_path, monkeypatch)
    attempts = 0

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal attempts
        initial_assignment = 'api.py' in stage['assigned_files']
        if initial_assignment:
            attempts += 1
            assert stage['attempt'] == attempts and stage['max_attempts'] == 2
            assert f'CUSTOM_ATTEMPT={attempts}' in review.backend.calls[-1]['prompt']
        for path in stage['assigned_files']:
            yield from read_events(review.repo, path, event_id=f'attempt-{attempts}-{path}')
        if not initial_assignment:
            assert all('FAILED_ATTEMPT_MARKER' not in str(item) for item in stage['notes'])
            return
        if attempts == 1 or outcome == 'second-rejection':
            output['targets'][0]['PRIVATE_UNKNOWN_PROPERTY_MARKER'] = None
            output['notes'] = 'FAILED_ATTEMPT_MARKER'
            output['candidates'] = [candidate(disposition='confirmed', finding=dict(
                record(), description='FAILED_ATTEMPT_MARKER'))]
        else:
            assert stage['progress'] == stage['candidates'] == []
            assert stage['observed_tool_starts'] == 4 and stage['remaining_tool_calls'] == 44
            prompt = review.backend.calls[-1]['prompt']
            assert 'PRIVATE_UNKNOWN_PROPERTY_MARKER' not in prompt and 'FAILED_ATTEMPT_MARKER' not in prompt
            rejection = stage['schema_rejection']
            assert rejection['error_count'] > 0 and rejection['candidate_count'] > 0
            assert set(rejection) == {'category', 'schema_path', 'error_count', 'candidate_count'}
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]

    await review.finish('python', findings=('Grounded defect',) if outcome == 'success' else (),
                        reason=None if outcome == 'success' else 'malformed_output')
    assert attempts == 2
    metadata = [event['metadata'] for event in stage_ends(review, 'python')]
    assert metadata[0]['admitted'] is False and metadata[0]['attempt_tool_starts'] == 4
    assert metadata[0]['schema_rejection']['error_count'] > 0
    assert metadata[1]['attempt'] == 2 and metadata[1]['observed_tool_starts'] == 8
    assert metadata[1]['remaining_tool_calls'] == 40
    assert metadata[1]['admitted'] is (outcome == 'success')
    if outcome == 'success':
        assert metadata[-1]['observed_tool_starts'] == 12 and metadata[-1]['remaining_tool_calls'] == 36


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-paths', 'inline'])
@pytest.mark.parametrize(('payload', 'omit_final'), [('medium-file', False), ('oversized-hunk', False),
                                                      ('oversized-hunk', True), ('unicode-line', False)])
async def test_oversized_hunk_assignments_reassemble_every_canonical_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool, payload: str, omit_final: bool) -> None:
    from daydream import git_ops
    from daydream.prompt_budget import SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES

    unicode = payload == 'unicode-line'
    medium = payload == 'medium-file'
    required = ("VALUE = '" + '界' * 11_000 + "'\n" if unicode else 'VALUE = 1\n' + ''.join(
        f'# continuation payload {index:04}: ' + 'x' * 90 + '\n' for index in range(540)))
    before = 'VALUE = 0\n'
    if medium:
        before = "def hello():\n    return 'world'\n"
        required = "def hello():\n    return 'universe'\n" + ''.join(
            f'# changed payload {index:04}: ' + 'x' * 90 + '\n' for index in range(140))
    repo = tmp_path / payload
    seed_feature_branch(repo, base={'api.py': before}, feature={'api.py': required})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend.sandbox = sandbox
    canonical = git_ops.diff(repo, 'main')
    expected_body = canonical.split('\n@@', 1)[1].split('\n', 1)[1].encode()
    if medium:
        assert 12 * 1024 < len(canonical.encode()) < 24 * 1024
    fragments: list[bytes] = []
    assignments: list[dict[str, Any]] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        assert stage['stage'] == 'first_pass' and stage['assigned_files'] == ['api.py']
        part, = stage['assignment_parts']
        assert part['kind'] == ('file' if medium and not sandbox else 'continuation')
        if not medium:
            assert part['part_count'] > 1
            assert (part['old_start'], part['old_count'], part['new_start'], part['new_count']) == (
                1, 1, 1, 1 if unicode else 541)
        assignments.append(part)
        prompt = review.backend.calls[-1]['prompt']
        if sandbox:
            rendered = prompt[prompt.index('Sanctioned phase inputs (captured verbatim):'):]
            assert len(rendered.encode()) <= SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES
            assert str(repo / '.daydream') not in prompt
            assert 'Sanctioned phase inputs (read only these exact files):' not in prompt
        diff, index = capture(review, stage)
        assert list(index) == ['api.py']
        assert index['api.py']['assignments'][0]['target_id'] == part['target_id']
        body = diff.split('\n@@', 1)[1].split('\n', 1)[1].encode()
        if medium and not sandbox:
            assert diff == canonical
            assert '<sanctioned-input label="intent">' in prompt
            assert 'The PR updates greetings across stacks.' in prompt
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        else:
            assert part['fragment_offset'] == sum(map(len, fragments))
            assert part['fragment_bytes'] == len(body)
        fragments.append(body)
        if omit_final and part['part_index'] == part['part_count']:
            output['targets'][0].update(status='not_reviewed', reason='Final continuation remains unexamined.')

    data = await review.finish('python', reason='evidence_incomplete' if omit_final else None,
                               findings=('Grounded defect',) if medium and not sandbox else ())
    assert scopes(data)['python']['files'] == ['api.py']
    if medium:
        assert len(assignments) == (2 if sandbox else 1)
    else:
        assert len(assignments) > 1
    assert [part['part_index'] for part in assignments] == list(range(1, len(assignments) + 1))
    assert b''.join(fragments) == expected_body
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))
    if unicode:
        assert len(assignments) >= 3 and assignments[-1]['segment_count'] == len(assignments)
        assert any(part['fragment_line_offset'] > 0 for part in assignments)


@pytest.mark.parametrize('bundle_capable', [False, True], ids=['legacy-files', 'explicit-bundle'])
async def test_current_exact_builder_signatures_route_generic_and_rust_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, bundle_capable: bool) -> None:
    ext_dir.write_module(
        'from dataclasses import dataclass\n'
        'from daydream.deep.prompts import (build_generic_fallback_prompt, build_per_stack_prompt, '
        'build_structural_prompt)\n'
        'def generic(*, strategy, files, diff_path, intent_path, alternatives_path, output_path, cwd, '
        'exploration_dir=None, is_docs_only=False, prior_commits=None, inline_diff=None, '
        'intent_authoritative=False, include_alternatives=True, frontier_files=None, review_stage=None):\n'
        '    return build_generic_fallback_prompt(**locals()) + "\\nEXACT_GENERIC_OVERRIDE"\n'
        '@dataclass\n'
        'class LanguageBuilder:\n'
        f'    review_input_bundle: bool = {bundle_capable!r}\n'
        '    def __call__(self, *, strategy, stack_name, files, diff_path, intent_path, alternatives_path, '
        'output_path, cwd, exploration_dir=None, prior_commits=None, inline_diff=None, '
        'intent_authoritative=False, include_alternatives=True, frontier_files=None, review_stage=None):\n'
        '        args = dict(locals()); del args["self"]\n'
        '        return build_per_stack_prompt(**args) + "\\nEXACT_LANGUAGE_OVERRIDE"\n'
        'def structure(*, strategy, files, diff_path, intent_path, alternatives_path, output_path, cwd, '
        'exploration_dir=None, prior_commits=None, intent_authoritative=False, include_alternatives=True, '
        'review_stage=None):\n'
        '    return build_structural_prompt(**locals()) + "\\nEXACT_STRUCTURE_OVERRIDE"\n'
        'def register(r):\n'
        '    r.override_prompt("generic-fallback", generic)\n'
        '    r.override_prompt("per-stack", LanguageBuilder())\n'
        '    r.override_prompt("structural", structure)\n', api_version=8,
    )
    before = {f'guide_{index}.md': '# Guide\nold contract\n' for index in range(4)}
    before.update({f'component_{index}.rs': 'pub fn capacity() -> usize { 1 }\n' for index in range(4)})
    repo = tmp_path / 'extension_shards'
    seed_feature_branch(repo, base=before, feature={name: body + '// changed boundary\n'
                                                  for name, body in before.items()})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    language_invocations: list[list[str]] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        prompt = review.backend.calls[-1]['prompt']
        if stage['scope_id'].startswith('generic#'):
            assert 'EXACT_GENERIC_OVERRIDE' in prompt
            assert 'Documentation' in prompt or 'documentation' in prompt
        elif stage['scope_id'].startswith('rust#'):
            assert 'EXACT_LANGUAGE_OVERRIDE' in prompt
            assert 'Error Handling Semantics' in prompt and 'Nested serde defaults' in prompt
            assert 'Confidence and Convention Rules' in prompt
            diff, _ = capture(review, stage, bundle=bundle_capable)
            assert diff.startswith('diff --git ')
            assert ('review-assignment' in stage['context_inputs']) is bundle_capable
            language_invocations.append(stage['assigned_files'])
        else:
            assert stage['scope_id'] == 'structure' and 'EXACT_STRUCTURE_OVERRIDE' in prompt

    review.backend.stage_response = response
    data = await review.finish('structure', deep_shard_enabled=True, deep_shard_max_files=1)
    expected = {f'{stack}#{index}' for stack in ('generic', 'rust') for index in range(4)} | {'structure'}
    assert set(scopes(data)) == expected
    assert {stage['scope_id'] for stage in review.backend.stages} == expected
    assert len(language_invocations) == 4
    assert {path for files in language_invocations for path in files} == {f'component_{i}.rs' for i in range(4)}
    assert all(event['metadata']['admitted'] for scope in expected for event in stage_ends(review, scope))


async def test_realistic_diff_and_index_are_scoped_to_assigned_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:

    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for file_index in range(16):
        name = 'api.py' if file_index == 0 else f'module_{file_index:02}.py'
        lines = [f'# stable line {line:04}: ' + 's' * 90 + '\n' for line in range(240)]
        old = lines.copy()
        for hunk in range(20):
            offset = hunk * 12 + 5
            old[offset] = f'BOUNDARY_{hunk:02} = 0\n'
            lines[offset] = f'BOUNDARY_{hunk:02} = 1\n'
        before[name], after[name] = ''.join(old), ''.join(lines)
    repo = tmp_path / 'realistic_context'
    seed_feature_branch(repo, base=before, feature=after)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    exposed: list[str] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        diff, _ = capture(review, stage)
        assert set(path for path in after if f'diff --git a/{path} b/{path}' in diff) == set(stage['assigned_files'])
        exposed.append(diff)
        if 'api.py' in stage['assigned_files'] and not stage['progress']:
            assert 'BOUNDARY_00 = 1' in (repo / 'api.py').read_text()
            output['candidates'] = [candidate(disposition='confirmed', finding=record(line=6))]

    await review.finish('python', findings=('Grounded defect',))
    assert len((repo / '.daydream/diff.patch').read_bytes()) >= 156_000
    assert len((repo / '.daydream/hunk-index.json').read_bytes()) >= 32_000
    assert len(exposed) > 4
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))


@pytest.mark.parametrize('path', ['component_雪.py', 'nested/component_"quote".py', 'component_\t.py'])
async def test_quoted_git_paths_keep_complete_required_stage_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str) -> None:

    repo = tmp_path / 'quoted_paths'
    seed_feature_branch(repo, base={path: 'VALUE = 0\n'}, feature={path: "VALUE = 'QUOTED_REQUIRED_CHANGE'\n"})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    required: list[str] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        assert stage['assigned_files'] == [path]
        diff, _ = capture(review, stage)
        assert 'QUOTED_REQUIRED_CHANGE' in diff
        required.append(diff)

    data = await review.finish('python')
    assert scopes(data)['python']['files'] == [path]
    assert len(required) == 1


async def test_failed_canonical_stage_input_preserves_successful_sibling_review(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    review = InvestigationRun(multi_stack_target, tmp_path, monkeypatch)
    real_read = Path.read_text
    canonical_reads = 0

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal canonical_reads
        if path.name == 'hunk-index.json' and 'stage-inputs' not in path.parts:
            canonical_reads += 1
            if canonical_reads == 2:
                raise OSError('PRIVATE_DIAGNOSTIC_PATH_SENTINEL')
        return real_read(path, *args, **kwargs)

    @review.backend.script('react')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        assert '<div>universe</div>' in real_read(review.repo / 'App.tsx')
        finding = dict(record(line=1), file='App.tsx', description='Surviving sibling defect',
                       evidence='App.tsx:1 renders universe instead of the greeting contract')
        output['candidates'] = [dict(candidate(disposition='confirmed', finding=finding), file='App.tsx', line=1,
                                     grounds=finding['evidence'])]

    monkeypatch.setattr(Path, 'read_text', read)
    await review.finish('python', reason='malformed_artifact', statuses=('failed',),
                               findings=('Surviving sibling defect',))
    assert canonical_reads >= 2
    assert 'PRIVATE_DIAGNOSTIC_PATH_SENTINEL' not in capsys.readouterr().out


async def test_changed_prepared_input_identity_blocks_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:

    review = staged_review(tmp_path, monkeypatch, files=1)

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        pointers = _sanctioned_inputs(review.backend.calls[-1]['prompt'])
        diff = pointers['review-assignment'] if 'review-assignment' in pointers else pointers['diff']
        diff.write_text('Changed after the prepared exact-path input was read.\n')
        output['targets'][0]['unknown_duplicate_id'] = None

    await review.finish('python', reason='evidence_incomplete')
    event, = stage_ends(review, 'python')
    assert event['metadata']['failure_class'] == 'admission_failure'
    assert event['metadata']['admitted'] is False and event['metadata']['attempt'] == 1
    assert event['metadata']['observed_tool_starts'] == 0


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-whole-context', 'inline-context-unavailable'])
async def test_shared_intent_uses_actual_transport_allowance_after_required_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool) -> None:

    review = staged_review(tmp_path, monkeypatch, files=2)
    shared_intent = 'The greeting API promises world and this change must preserve callers.\n' * 600
    assert 40_000 < len(shared_intent.encode()) < 48_000

    review.backend.responder = lambda prompt: (
        [TextEvent(text=shared_intent), ResultEvent(structured_output=None, continuation=None)]
        if 'understand the intent of these changes' in prompt.lower() else None)
    review.backend.sandbox = sandbox

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        statuses = {item['label']: item['status'] for item in stage['context_statuses']}
        if sandbox:
            assert 'intent' not in stage['context_inputs'] and statuses['intent'] == 'unavailable'
        else:
            pointers = _sanctioned_inputs(review.backend.calls[-1]['prompt'])
            assert 'intent' in stage['context_inputs'] and statuses['intent'] == 'complete'
            assert pointers['intent'].read_text() == shared_intent
            yield ToolStartEvent(id='shared-context', name='Read', input={'file_path': str(pointers['intent'])})
            yield ToolResultEvent(id='shared-context', output=pointers['intent'].read_text(), is_error=False)
        output['candidates'] = [candidate(disposition='confirmed', finding=record())]

    data = await review.finish('python', findings=('Grounded defect',), expected_exit=1 if sandbox else 0,
                               file_config=DaydreamFileConfig(supervisor='off'))
    if sandbox:
        assert data['terminal_result']['pipeline_state'] == 'failed'
    event, = stage_ends(review, 'python')
    assert event['metadata']['observed_tool_starts'] == (0 if sandbox else 1)


@pytest.mark.parametrize(('case', 'reason', 'attempts'), [
    ('unparseable', 'malformed_output', 1),
    ('invalid-native-valid-text', 'malformed_output', 2),
    ('nested-invalid-root', 'malformed_output', 2),
    ('incomplete-nested-stage', 'malformed_output', 1),
    ('text-origin-nested', 'malformed_output', 2),
    ('empty-final-turn', 'missing_output', 1),
    ('native-prose', None, 1),
    ('pi-schema-retry', None, 2),
    ('missing-notes', None, 2),
    ('wrong-type-notes', None, 2),
    ('native-missing-notes', 'malformed_output', 1),
    ('compliant-json', None, 1),
    ('misplaced-closing-brace', 'malformed_output', 1),
])
async def test_strict_stage_selection_never_salvages_rejected_or_incomplete_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, reason: str | None, attempts: int) -> None:
    review = staged_review(tmp_path, monkeypatch, files=2)

    @review.backend.script(terminal=False)
    def stream(stage: dict[str, Any], valid: dict[str, Any]) -> Iterable[AgentEvent]:
        prompt = review.backend.calls[-1]['prompt']
        attempt = sum(item['scope_id'] == 'python' for item in review.backend.stages)
        assert stage['attempt'] == attempt
        valid['candidates'] = ([] if case in {'compliant-json', 'misplaced-closing-brace'} else
                               [candidate(disposition='confirmed', finding=record())])
        invalid = {'unknown_outer_key': valid}
        if case.endswith('notes'):
            if case == 'native-missing-notes':
                yield RequestEvent(prompt=prompt, output_schema=stage['response_contract']['schema'],
                                   config=PiRequestConfig(schema_emulated=False, no_tools=False))
            if attempt == 1:
                valid['candidates'][0]['finding']['description'] = 'FAILED_SCHEMA_ONLY_ATTEMPT'
                if case == 'wrong-type-notes':
                    valid['notes'] = 3
                else:
                    valid.pop('notes')
            else:
                assert stage['notes'] == stage['candidates'] == []
                assert stage['observed_tool_starts'] == 0 and stage['remaining_tool_calls'] == 48
                assert 'FAILED_SCHEMA_ONLY_ATTEMPT' not in prompt
            yield ResultEvent(structured_output=valid, continuation=None)
            return
        final_turn = case in {'text-origin-nested', 'empty-final-turn', 'native-prose', 'pi-schema-retry'}
        if final_turn:
            yield TextEvent(text='Planning the assigned source investigation.')
            yield TurnEndEvent()
        if case == 'pi-schema-retry':
            if attempt == 1:
                valid['targets'][0]['unknown_duplicate_target'] = None
                valid['candidates'][0]['finding']['description'] = 'FAILED_PI_ATTEMPT_MARKER'
            else:
                assert stage['candidates'] == stage['notes'] == stage['progress'] == []
                assert 'FAILED_PI_ATTEMPT_MARKER' not in prompt
        text = {
            'unparseable': '{"targets": [{"target_id": "api.py", "status": "reviewed", "reason": ""}], '
                          '"notes": "unfinished',
            'nested-invalid-root': json.dumps(invalid),
            'incomplete-nested-stage': '{"outer": ' + json.dumps(valid),
            'text-origin-nested': json.dumps({'invalid_outer': valid}),
            'native-prose': 'The investigation is finished.',
            'misplaced-closing-brace': json.dumps({'targets': valid['targets'], 'notes': 'Reviewed real source'})
                                     + ', "candidates": [], "contradictions": []}',
        }.get(case, json.dumps(valid))
        yield TextEvent(text=text)
        if final_turn:
            yield TurnEndEvent()
        if case == 'empty-final-turn':
            yield TextEvent(text='')
            yield TurnEndEvent()
        native = (invalid if case == 'invalid-native-valid-text' else valid if final_turn
                  and (case != 'pi-schema-retry' or attempt > 1) else None)
        yield ResultEvent(structured_output=native, continuation=None,
                          structured_output_origin='text' if case in {'text-origin-nested', 'empty-final-turn'}
                          else 'native')

    await review.finish('python', reason=reason,
                        findings=('Grounded defect',) if reason is None and case != 'compliant-json' else ())
    events = stage_ends(review, 'python')
    assert len(events) == attempts
    assert [event['metadata']['attempt'] for event in events] == list(range(1, attempts + 1))
    assert [event['metadata']['admitted'] for event in events] == (
        [False, True] if case in {'pi-schema-retry', 'missing-notes', 'wrong-type-notes'}
        else [reason is None] * attempts)
    if attempts == 2:
        assert events[0]['metadata']['schema_rejection']['error_count'] > 0
    elif case in {'unparseable', 'incomplete-nested-stage', 'empty-final-turn',
                  'compliant-json', 'misplaced-closing-brace'}:
        assert events[0]['metadata']['schema_rejection'] is None
    metadata = events[-1]['metadata']
    assert metadata['observed_tool_starts'] == 0 and metadata['remaining_tool_calls'] == 48
    if case in {'compliant-json', 'misplaced-closing-brace'}:
        assert metadata['failure_class'] == ('syntax_failure' if case == 'misplaced-closing-brace' else None)


async def test_opaque_assignment_handles_cannot_alias_real_changed_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    repo = tmp_path / 'assignment_identity'
    payload = ''.join(f'# REQUIRED_LARGE_CHANGE_{index:04} ' + 'x' * 90 + '\n' for index in range(600))
    seed_feature_branch(repo, base={'00_oversize.md': '# Before\n', 'part:000001': 'BEFORE = 0\n'},
                        feature={'00_oversize.md': '# After\n' + payload,
                                 'part:000001': 'REAL_FILE_REQUIRED_CHANGE = 1\n'})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    targets: list[str] = []
    diffs: list[str] = []

    @review.backend.script('generic')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        target_ids = stage['assigned_target_ids']
        assert len(target_ids) == len(set(target_ids)) and not set(targets).intersection(target_ids)
        targets.extend(target_ids)
        scoped_diff, _ = capture(review, stage)
        if '00_oversize.md' in stage['assigned_files']:
            assert 'REQUIRED_LARGE_CHANGE_' in scoped_diff
        if 'part:000001' in stage['assigned_files']:
            assert 'REAL_FILE_REQUIRED_CHANGE' in scoped_diff
        diffs.append(scoped_diff)
    data = await review.finish('generic')
    assert set(scopes(data)['generic']['files']) == {'00_oversize.md', 'part:000001'}
    assert 'part:000001' in targets and len(targets) > 2
    combined = '\n'.join(diffs)
    for index in range(600):
        assert combined.count(f'REQUIRED_LARGE_CHANGE_{index:04}') == 1


@pytest.mark.parametrize('fault', ['unknown-target', 'unknown-candidate', 'contradiction', 'grounds',
                                  'grounds-missing', 'grounds-null', 'grounds-number', 'handoff'])
async def test_mixed_schema_and_admission_rejection_never_retries_or_loses_prior_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    review = staged_review(tmp_path, monkeypatch)

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if not stage['progress']:
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            return
        if stage['attempt'] > 1:
            # A second valid answer demonstrates the erroneous recovery path;
            # production must never invoke it for a mixed semantic rejection.
            return
        output['unknown_extra'] = None
        output['notes'] = 'FAILED_MIXED_ATTEMPT_MARKER'
        if fault == 'unknown-target':
            output['targets'][0]['target_id'] = 'foreign-assignment'
        elif fault == 'contradiction':
            output['contradictions'] = [stage['closed_candidate_ids'][0]]
        elif fault == 'handoff':
            output['notes'] += 'x' * 70_000
        else:
            path = stage['assigned_files'][0]
            finding = dict(record(line=1), file=path, evidence=f'{path}:1 changes VALUE from 0 to 1')
            item = dict(candidate(disposition='confirmed', finding=finding), file=path, line=1,
                        grounds=finding['evidence'])
            if fault == 'unknown-candidate':
                item['candidate_id'] = 'foreign-candidate'
            elif fault == 'grounds-missing':
                item.pop('grounds')
            else:
                item['grounds'] = {'grounds-null': None, 'grounds-number': 7}.get(fault, '')
            output['candidates'] = [item]

    await review.finish('python', reason='malformed_output', findings=('Grounded defect',))
    events = stage_ends(review, 'python')
    assert len(events) == 2
    assert [event['metadata']['admitted'] for event in events] == [True, False]
    assert [event['metadata']['attempt'] for event in events] == [1, 1]
    assert events[-1]['metadata']['observed_tool_starts'] == 0
    assert events[-1]['metadata']['remaining_tool_calls'] == 48
    assert events[-1]['metadata']['schema_rejection']['error_count'] > 0
    assert all('FAILED_MIXED_ATTEMPT_MARKER' not in call['prompt'] for call in review.backend.calls)


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-paths', 'inline'])
async def test_old_only_hunk_and_later_continuations_keep_canonical_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool) -> None:

    from tests.harness.git_helpers import git

    prefix = ''.join(f'# stable boundary {index:03}\n' for index in range(98))
    before = 'DELETED_FIRST_LINE = True\n' + prefix + 'VALUE = 0\n'
    added = ''.join(f'# LATER_REQUIRED_CHANGE_{index:04} ' + 'x' * 96 + '\n' for index in range(500))
    repo = tmp_path / 'mixed_hunk_ranges'
    seed_feature_branch(repo, base={'api.py': before}, feature={'api.py': prefix + 'VALUE = 1\n' + added})
    git(repo, 'config', 'diff.context', '0')
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend.sandbox = sandbox
    assignments: list[dict[str, Any]] = []
    fragments: list[str] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        diff, index = capture(review, stage)
        scoped = index['api.py']
        fragments.append(diff)
        for part in stage['assignment_parts']:
            assignments.append(part)
            expected = (part['old_start'], part['old_start'] + part['old_count'] - 1,
                        part['new_start'], part['new_start'] + part['new_count'] - 1)
            assert any((hunk['old_start'], hunk['old_end'], hunk['new_start'], hunk['new_end']) == expected
                       for hunk in scoped['hunks'])
            if part['new_count'] != 0:
                assert (part['old_start'], part['old_count'], part['new_start'], part['new_count']) == (100, 1, 99, 501)

    assert await review.run(findings_out=None) == 0
    coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    assert all(scope['status'] == 'complete' for scope in coverage['stack_outcomes'])
    assert assignments[0]['old_start'] == 1 and assignments[0]['new_count'] == 0
    combined = '\n'.join(fragments)
    assert combined.count('-DELETED_FIRST_LINE = True') == 1
    for index in range(500):
        assert combined.count(f'LATER_REQUIRED_CHANGE_{index:04}') == 1
