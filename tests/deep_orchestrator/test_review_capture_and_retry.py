"""Large staged-review transport and retry scenarios through the production runner."""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import (
    AgentEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.config_file import DaydreamFileConfig
from tests.conftest import ExtDir
from tests.deep_orchestrator.test_review_completion import record, scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, StagedBackend, candidate, stage_ends
from tests.harness.git_helpers import seed_feature_branch
from tests.test_deep_orchestrator import _sanctioned_inputs


def staged_review(tmp: Path, patch: pytest.MonkeyPatch, *, source_bytes: int,
                  files: int = 8) -> InvestigationRun:
    before = {f'module_{index:02}.py': 'VALUE = 0\n' for index in range(files - 1)}
    before['api.py'] = "def hello():\n    return 'world'\n"
    padding = ''.join(f'# source context {index:04}: ' + 'x' * 72 + '\n'
                      for index in range(source_bytes // 96 + 1))
    if source_bytes > 1_000_000:
        padding = '\n' * 10 + '# complete source context: ' + 'x' * source_bytes + '\n'
    before = {name: body + padding for name, body in before.items()}
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


@pytest.mark.parametrize('fault', [
    'zero-match-search', 'failed-read', 'clipped-read', 'unmatched-result', 'oversized-read',
])
async def test_optional_tool_results_do_not_poison_a_valid_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    review = staged_review(tmp_path, monkeypatch, source_bytes=400, files=1)

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        tool_name = 'Bash' if fault == 'zero-match-search' else 'Read'
        arguments = (
            {'command': 'rg definitely_absent .'}
            if fault == 'zero-match-search' else {'file_path': 'optional.py'}
        )
        yield ToolStartEvent(id='optional', name=tool_name, input=arguments)
        yield ToolResultEvent(id='optional', output='x' * (2 * 1024 * 1024 + 1) if fault == 'oversized-read'
                              else 'optional context', is_error=fault in {'zero-match-search', 'failed-read'},
                              exit_code=1 if fault == 'zero-match-search' else None,
                              truncated=fault == 'clipped-read')
        if fault == 'unmatched-result':
            yield ToolResultEvent(id='unassociated', output='failed optional lookup', is_error=True)
        output['candidates'] = [candidate(disposition='confirmed', finding=record())]

    review.backend.stage_response = response
    data = await review.finish('python', findings=('Grounded defect',))
    assert scopes(data)['python']['status'] == 'complete'
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    metadata, = [item['metadata'] for item in stage_ends(review, 'python')]
    assert metadata['admitted'] is True
    assert metadata['observed_tool_starts'] == 1


@pytest.mark.parametrize('outcome', ['success', 'second-rejection'])
async def test_schema_rejection_retries_once_with_fresh_grounding_and_charged_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, ext_dir: ExtDir,
) -> None:
    ext_dir.write_module(
        'from daydream.deep.prompts import build_per_stack_prompt\n'
        'def staged(**kw):\n'
        '    return build_per_stack_prompt(**kw) + \"\\nCUSTOM_ATTEMPT=\" + str(kw[\"review_stage\"][\"attempt\"])\n'
        'def register(r): r.override_prompt(\"per-stack\", staged)\n', api_version=8,
    )
    review = staged_review(tmp_path, monkeypatch, source_bytes=400)
    attempts = 0

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal attempts
        if stage['scope_id'] != 'python':
            return
        initial_assignment = 'api.py' in stage['assigned_files']
        if initial_assignment:
            attempts += 1
            assert stage['attempt'] == attempts and stage['max_attempts'] == 2
            assert f'CUSTOM_ATTEMPT={attempts}' in review.backend.calls[-1]['prompt']
        for path in stage['assigned_files']:
            yield ToolStartEvent(id=f'attempt-{attempts}-{path}', name='Read', input={'file_path': path})
            yield ToolResultEvent(id=f'attempt-{attempts}-{path}', output=(review.repo / path).read_text(),
                                  is_error=False, truncated=False)
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

    review.backend.stage_response = response
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


@pytest.mark.parametrize(('sandbox', 'expected_stages'), [(False, 1), (True, 2)], ids=['exact-paths', 'inline'])
async def test_transport_cap_keeps_a_medium_changed_file_whole_without_enlarging_inline_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool, expected_stages: int,
) -> None:
    from daydream import git_ops
    from daydream.prompt_budget import SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES

    repo = tmp_path / 'medium_assignment'
    payload = ''.join(f'# changed payload {index:04}: ' + 'x' * 90 + '\n' for index in range(140))
    seed_feature_branch(repo, base={'api.py': "def hello():\n    return 'world'\n"},
                        feature={'api.py': "def hello():\n    return 'universe'\n" + payload})
    canonical = git_ops.diff(repo, 'main')
    assert 12 * 1024 < len(canonical.encode()) < 24 * 1024
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend.sandbox = sandbox
    diffs: list[str] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        prompt = review.backend.calls[-1]['prompt']
        captured = supporting_contents(prompt)
        diffs.append(captured['diff'])
        if sandbox:
            rendered = prompt[prompt.index('Sanctioned phase inputs (captured verbatim):'):]
            assert len(rendered.encode()) <= SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES
            assert stage['assignment_parts'][0]['kind'] == 'continuation'
        else:
            bundle = Path(stage['supporting_bundle']['path']).read_bytes()
            assert len(bundle) <= 24 * 1024
            assert stage['assignment_parts'][0]['kind'] == 'file'
            assert captured['diff'] == canonical
            assert '<sanctioned-input label="intent">' in prompt
            assert 'The PR updates greetings across stacks.' in prompt
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]

    review.backend.stage_response = response
    await review.finish('python', findings=() if sandbox else ('Grounded defect',))
    assert len(diffs) == expected_stages
    joined = ''.join(diffs)
    assert all(joined.count(f'# changed payload {index:04}:') == 1 for index in range(140))
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-paths', 'inline'])
@pytest.mark.parametrize('omit_final', [False, True], ids=['all-parts', 'missing-final-part'])
async def test_oversized_hunk_assignments_reassemble_every_canonical_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool, omit_final: bool,
) -> None:
    from daydream import git_ops
    from daydream.prompt_budget import SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES

    required = ''.join(f'# continuation payload {index:04}: ' + 'x' * 90 + '\n' for index in range(540))
    repo = tmp_path / 'oversized_hunk'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n'},
                        feature={'api.py': 'VALUE = 1\n' + required})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend.sandbox = sandbox
    canonical = git_ops.diff(repo, 'main')
    expected_body = canonical.split('\n@@', 1)[1].split('\n', 1)[1].encode()
    fragments: list[bytes] = []
    assignments: list[dict[str, Any]] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        assert stage['stage'] == 'first_pass' and stage['assigned_files'] == ['api.py']
        part, = stage['assignment_parts']
        assert part['kind'] == 'continuation' and part['part_count'] > 1
        assert (part['old_start'], part['old_count'], part['new_start'], part['new_count']) == (1, 1, 1, 541)
        assignments.append(part)
        prompt = review.backend.calls[-1]['prompt']
        if sandbox:
            header = 'Sanctioned phase inputs (captured verbatim):'
            rendered = prompt[prompt.index(header):]
            assert len(rendered.encode()) <= SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES
            assert str(repo / '.daydream') not in prompt
            assert 'Sanctioned phase inputs (read only these exact files):' not in prompt
            captured = dict(re.findall(r'<sanctioned-input label="([^"]+)">\n(.*?)\n</sanctioned-input>',
                                       rendered, flags=re.S))
        else:
            assert len(Path(stage['supporting_bundle']['path']).read_bytes()) <= 24 * 1024
        captured = supporting_contents(prompt)
        diff = captured['diff']
        index = json.loads(captured['hunk-index'])
        assert list(index) == ['api.py']
        assert index['api.py']['assignments'][0]['target_id'] == part['target_id']
        body = diff.split('\n@@', 1)[1].split('\n', 1)[1].encode()
        assert part['fragment_offset'] == sum(map(len, fragments))
        assert part['fragment_bytes'] == len(body)
        fragments.append(body)
        if omit_final and part['part_index'] == part['part_count']:
            output['targets'][0].update(status='not_reviewed', reason='Final continuation remains unexamined.')

    review.backend.stage_response = response
    data = await review.finish('python', reason='evidence_incomplete' if omit_final else None)
    assert scopes(data)['python']['files'] == ['api.py']
    assert len(assignments) > 1
    assert [part['part_index'] for part in assignments] == list(range(1, len(assignments) + 1))
    assert b''.join(fragments) == expected_body
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))


async def test_current_exact_builder_signatures_route_generic_and_rust_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir,
) -> None:
    ext_dir.write_module(
        'from daydream.deep.prompts import (build_generic_fallback_prompt, build_per_stack_prompt, '
        'build_structural_prompt)\n'
        'def generic(*, strategy, files, diff_path, intent_path, alternatives_path, output_path, cwd, '
        'exploration_dir=None, is_docs_only=False, prior_commits=None, inline_diff=None, '
        'intent_authoritative=False, include_alternatives=True, frontier_files=None, review_stage=None):\n'
        '    return build_generic_fallback_prompt(**locals()) + "\\nEXACT_GENERIC_OVERRIDE"\n'
        'def language(*, strategy, stack_name, files, diff_path, intent_path, alternatives_path, output_path, cwd, '
        'exploration_dir=None, prior_commits=None, inline_diff=None, intent_authoritative=False, '
        'include_alternatives=True, frontier_files=None, review_stage=None):\n'
        '    return build_per_stack_prompt(**locals()) + "\\nEXACT_LANGUAGE_OVERRIDE"\n'
        'def structure(*, strategy, files, diff_path, intent_path, alternatives_path, output_path, cwd, '
        'exploration_dir=None, prior_commits=None, intent_authoritative=False, include_alternatives=True, '
        'review_stage=None):\n'
        '    return build_structural_prompt(**locals()) + "\\nEXACT_STRUCTURE_OVERRIDE"\n'
        'def register(r):\n'
        '    r.override_prompt("generic-fallback", generic)\n'
        '    r.override_prompt("per-stack", language)\n'
        '    r.override_prompt("structural", structure)\n', api_version=8,
    )
    before = {f'guide_{index}.md': '# Guide\nold contract\n' for index in range(4)}
    before.update({f'component_{index}.rs': 'pub fn capacity() -> usize { 1 }\n' for index in range(4)})
    repo = tmp_path / 'extension_shards'
    seed_feature_branch(repo, base=before, feature={name: body + '// changed boundary\n'
                                                  for name, body in before.items()})
    review = InvestigationRun(repo, tmp_path, monkeypatch)

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        prompt = review.backend.calls[-1]['prompt']
        if stage['scope_id'].startswith('generic#'):
            assert 'EXACT_GENERIC_OVERRIDE' in prompt
            assert 'Documentation' in prompt or 'documentation' in prompt
        elif stage['scope_id'].startswith('rust#'):
            assert 'EXACT_LANGUAGE_OVERRIDE' in prompt
            assert 'Error Handling Semantics' in prompt and 'Nested serde defaults' in prompt
        else:
            assert stage['scope_id'] == 'structure' and 'EXACT_STRUCTURE_OVERRIDE' in prompt

    review.backend.stage_response = response
    assert await review.run(deep_shard_enabled=True, deep_shard_max_files=1) == 0
    data = review.load()
    expected = {f'{stack}#{index}' for stack in ('generic', 'rust') for index in range(4)} | {'structure'}
    assert set(scopes(data)) == expected
    assert all(scope['status'] == 'complete' for scope in scopes(data).values())
    assert {stage['scope_id'] for stage in review.backend.stages} == expected


async def test_triage_carries_only_relevant_admitted_semantic_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = staged_review(tmp_path, monkeypatch, source_bytes=16_000, files=4)

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        if stage['stage'] == 'first_pass':
            output['candidates'] = [candidate(), dict(candidate(disposition='rejected'), file='module_00.py')]
            return
        pending, = stage['candidates']
        assert stage['assigned_files'] == ['api.py']
        assert stage['assigned_candidate_ids'] == [pending['candidate_id']]
        assert pending['file'] == 'api.py'
        assert pending['grounds'] and pending['trigger'] and pending['consequence']
        output['candidates'] = [dict(pending, disposition='confirmed', finding=record())]

    review.backend.stage_response = response
    await review.finish('python', findings=('Grounded defect',))
    metadata = [event['metadata'] for event in stage_ends(review, 'python')]
    assert [item['logical_stage'] for item in metadata] == ['first_pass', 'triage']
    assert metadata[-1]['admitted'] is True


async def test_realistic_diff_and_index_are_scoped_to_assigned_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        captured = supporting_contents(review.backend.calls[-1]['prompt'])
        diff, index = captured['diff'], json.loads(captured['hunk-index'])
        assert set(index) == set(stage['assigned_files'])
        assert set(path for path in after if f'diff --git a/{path} b/{path}' in diff) == set(stage['assigned_files'])
        assert len(Path(stage['supporting_bundle']['path']).read_bytes()) <= 24 * 1024
        exposed.append(diff)
        if 'api.py' in stage['assigned_files'] and not stage['progress']:
            assert 'BOUNDARY_00 = 1' in (repo / 'api.py').read_text()
            output['candidates'] = [candidate(disposition='confirmed', finding=record(line=6))]

    review.backend.stage_response = response
    await review.finish('python', findings=('Grounded defect',))
    assert len((repo / '.daydream/diff.patch').read_bytes()) >= 156_000
    assert len((repo / '.daydream/hunk-index.json').read_bytes()) >= 32_000
    assert len(exposed) > 4
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))


@pytest.mark.parametrize('case', ['unparseable', 'invalid-native-valid-text', 'nested-invalid-root',
                                  'incomplete-nested-stage'])
async def test_strict_stage_selection_never_salvages_rejected_or_incomplete_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    from tests.harness.stub_backend import review_stage_state, stage_result

    review = staged_review(tmp_path, monkeypatch, source_bytes=400, files=2)

    class IncompleteBackend(StagedBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any,
                          **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            if stage is not None and stage['scope_id'] == 'python':
                self.stages.append(stage)
                self.calls.append({'prompt': prompt, **kwargs})
                valid = stage_result(stage)
                invalid = {'unknown_outer_key': valid}
                text = (json.dumps(valid) if case == 'invalid-native-valid-text' else json.dumps(invalid)
                        if case == 'nested-invalid-root' else '{\"outer\": ' + json.dumps(valid)
                        if case == 'incomplete-nested-stage' else
                        '{\"targets\": [{\"target_id\": \"api.py\", \"status\": \"reviewed\", \"reason\": \"\"}], '
                        '\"notes\": \"unfinished')
                yield TextEvent(text=text)
                yield ResultEvent(structured_output=invalid if case == 'invalid-native-valid-text' else None,
                                  continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    review.backend = IncompleteBackend(review.repo)
    await review.finish('python', reason='malformed_output')
    expected_attempts = 2 if case in {'invalid-native-valid-text', 'nested-invalid-root'} else 1
    assert len([stage for stage in review.backend.stages if stage['scope_id'] == 'python']) == expected_attempts
    events = stage_ends(review, 'python')
    assert all(event['metadata']['admitted'] is False for event in events)
    assert events[-1]['metadata']['attempt'] == expected_attempts
    if expected_attempts == 1:
        assert events[0]['metadata']['schema_rejection'] is None


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-paths', 'inline'])
async def test_long_unicode_line_continuations_reassemble_complete_diff_with_byte_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool,
) -> None:

    from daydream import git_ops

    repo = tmp_path / 'unicode_line'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n'},
                        feature={'api.py': "VALUE = '" + '界' * 11_000 + "'\n"})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend.sandbox = sandbox
    chunks: list[bytes] = []
    mappings: list[dict[str, Any]] = []
    canonical = git_ops.diff(repo, 'main')
    expected_body = canonical.split('\n@@', 1)[1].split('\n', 1)[1].encode()

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        part, = stage['assignment_parts']
        mappings.append(part)
        prompt = review.backend.calls[-1]['prompt']
        diff = supporting_contents(prompt)['diff']
        body = diff.split('\n@@', 1)[1].split('\n', 1)[1].encode()
        assert part['kind'] == 'continuation' and part['fragment_offset'] == sum(map(len, chunks))
        assert part['fragment_bytes'] == len(body)
        assert (part['old_start'], part['old_count'], part['new_start'], part['new_count']) == (1, 1, 1, 1)
        chunks.append(body)

    review.backend.stage_response = response
    await review.finish('python')
    assert b''.join(chunks) == expected_body
    assert len(mappings) >= 3 and mappings[-1]['segment_count'] == len(mappings)
    assert any(part['fragment_line_offset'] > 0 for part in mappings)




@pytest.mark.parametrize('path', ['component_雪.py', 'nested/component_"quote".py', 'component_\t.py'])
async def test_quoted_git_paths_keep_complete_required_stage_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:

    repo = tmp_path / 'quoted_paths'
    seed_feature_branch(repo, base={path: 'VALUE = 0\n'},
                        feature={path: "VALUE = 'QUOTED_REQUIRED_CHANGE'\n"})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    required: list[str] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        assert stage['assigned_files'] == [path]
        captured = supporting_contents(review.backend.calls[-1]['prompt'])
        diff = captured['diff']
        assert 'QUOTED_REQUIRED_CHANGE' in diff
        assert path in json.loads(captured['hunk-index'])
        required.append(diff)

    review.backend.stage_response = response
    data = await review.finish('python')
    assert scopes(data)['python']['files'] == [path]
    assert len(required) == 1


async def test_failed_canonical_stage_input_preserves_successful_sibling_review(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] == 'react':
            assert '<div>universe</div>' in real_read(review.repo / 'App.tsx')
            finding = dict(record(line=1), file='App.tsx', description='Surviving sibling defect',
                           evidence='App.tsx:1 renders universe instead of the greeting contract')
            output['candidates'] = [dict(candidate(disposition='confirmed', finding=finding), file='App.tsx', line=1,
                                         grounds=finding['evidence'])]

    monkeypatch.setattr(Path, 'read_text', read)
    review.backend.stage_response = response
    data = await review.finish('python', reason='malformed_artifact', statuses=('failed',),
                               findings=('Surviving sibling defect',))
    assert scopes(data)['react']['status'] == 'complete'
    assert canonical_reads >= 2
    assert 'PRIVATE_DIAGNOSTIC_PATH_SENTINEL' not in capsys.readouterr().out






async def test_changed_prepared_input_identity_blocks_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:

    review = staged_review(tmp_path, monkeypatch, source_bytes=400, files=1)

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        pointers = _sanctioned_inputs(review.backend.calls[-1]['prompt'])
        diff = pointers['review-assignment'] if 'review-assignment' in pointers else pointers['diff']
        diff.write_text('Changed after the prepared exact-path input was read.\n')
        output['targets'][0]['unknown_duplicate_id'] = None

    review.backend.stage_response = response
    await review.finish('python', reason='evidence_incomplete')
    event, = stage_ends(review, 'python')
    assert event['metadata']['failure_class'] == 'admission_failure'
    assert event['metadata']['admitted'] is False and event['metadata']['attempt'] == 1
    assert event['metadata']['observed_tool_starts'] == 0


async def test_candidate_triage_receives_only_relevant_admitted_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / 'mixed_source'
    before = {'api.py': "def hello():\n    return 'world'\n" + '# context\n' * 1_800,
              'module.py': 'VALUE = 0\n# OTHER_FILE_SENSITIVE_MARKER\n' + '# context\n' * 1_800}
    seed_feature_branch(repo, base=before,
                        feature={'api.py': before['api.py'].replace("'world'", "'universe'"),
                                 'module.py': before['module.py'].replace('VALUE = 0', 'VALUE = 1')})
    review = InvestigationRun(repo, tmp_path, monkeypatch)

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        if stage['stage'] == 'first_pass':
            assert set(stage['assigned_files']) == {'api.py', 'module.py'}
            output['candidates'] = [candidate()]
            return
        pending, = stage['candidates']
        assert stage['assigned_files'] == ['api.py']
        assert pending['file'] == 'api.py'
        assert 'OTHER_FILE_SENSITIVE_MARKER' not in pending['grounds']
        assert pending['grounds'] and pending['trigger'] and pending['consequence']
        assert "return 'universe'" in (repo / 'api.py').read_text()
        output['candidates'] = [dict(pending, disposition='confirmed', finding=record())]

    review.backend.stage_response = response
    await review.finish('python', findings=('Grounded defect',))
    assert stage_ends(review, 'python')[-1]['metadata']['admitted'] is True


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-whole-context', 'inline-context-unavailable'])
async def test_shared_intent_uses_actual_transport_allowance_after_required_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool,
) -> None:

    review = staged_review(tmp_path, monkeypatch, source_bytes=400, files=2)
    shared_intent = 'The greeting API promises world and this change must preserve callers.\n' * 600
    assert 40_000 < len(shared_intent.encode()) < 48_000

    class LargeIntentBackend(StagedBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any,
                          **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if 'understand the intent of these changes' in prompt.lower():
                self.calls.append({'prompt': prompt, **kwargs})
                yield TextEvent(text=shared_intent)
                yield ResultEvent(structured_output=None, continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    review.backend = LargeIntentBackend(review.repo)
    review.backend.sandbox = sandbox

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        from tests.test_deep_orchestrator import _sanctioned_inputs
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

    review.backend.stage_response = response
    if sandbox:
        assert await review.run(file_config=DaydreamFileConfig(supervisor='off')) == 1
        data = review.load()
        assert scopes(data)['python']['status'] == 'complete'
        assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
        assert data['terminal_result']['pipeline_state'] == 'failed'
    else:
        await review.finish('python', findings=('Grounded defect',), file_config=DaydreamFileConfig(supervisor='off'))
    event, = stage_ends(review, 'python')
    if not sandbox:
        assert event['metadata']['observed_tool_starts'] == 1
    else:
        assert event['metadata']['observed_tool_starts'] == 0


async def test_pi_style_final_turn_schema_retry_ignores_prior_planning_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.stub_backend import review_stage_state, stage_result

    review = staged_review(tmp_path, monkeypatch, source_bytes=400, files=2)

    class TurnBackend(StagedBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any,
                          **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            if stage is not None and stage['scope_id'] == 'python':
                self.calls.append({'prompt': prompt, **kwargs})
                self.stages.append(stage)
                attempt = len([item for item in self.stages if item['scope_id'] == 'python'])
                assert stage['attempt'] == attempt
                yield TextEvent(text='I will inspect the assigned symbols before making a judgment.')
                yield TurnEndEvent()
                finding = dict(record(), description='FAILED_PI_ATTEMPT_MARKER') if attempt == 1 else record()
                output = stage_result(stage, candidates=[candidate(disposition='confirmed', finding=finding)])
                if attempt == 1:
                    output['targets'][0]['unknown_duplicate_target'] = None
                else:
                    assert stage['candidates'] == stage['notes'] == stage['progress'] == []
                    assert 'FAILED_PI_ATTEMPT_MARKER' not in prompt
                yield TextEvent(text=json.dumps(output))
                yield TurnEndEvent()
                yield ResultEvent(structured_output=None if attempt == 1 else output, continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    review.backend = TurnBackend(review.repo)
    await review.finish('python', findings=('Grounded defect',))
    events = stage_ends(review, 'python')
    assert len(events) == 2
    assert [event['metadata']['attempt'] for event in events] == [1, 2]
    assert [event['metadata']['admitted'] for event in events] == [False, True]
    assert events[0]['metadata']['schema_rejection']['error_count'] > 0
    assert events[-1]['metadata']['observed_tool_starts'] == 0
    assert events[-1]['metadata']['remaining_tool_calls'] == 48


@pytest.mark.parametrize('selection', ['text-origin-nested', 'empty-final-turn', 'native-prose'])
async def test_final_turn_selection_distinguishes_inferred_text_from_native_structured_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str,
) -> None:
    from tests.harness.stub_backend import review_stage_state, stage_result

    review = staged_review(tmp_path, monkeypatch, source_bytes=400, files=2)

    class SelectionBackend(StagedBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any,
                          **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            if stage is not None and stage['scope_id'] == 'python':
                self.calls.append({'prompt': prompt, **kwargs})
                self.stages.append(stage)
                yield TextEvent(text='Planning the assigned source investigation.')
                yield TurnEndEvent()
                output = stage_result(stage, candidates=[candidate(disposition='confirmed', finding=record())])
                if selection == 'empty-final-turn':
                    yield TextEvent(text=json.dumps(output))
                    yield TurnEndEvent()
                    yield TextEvent(text='')
                    yield TurnEndEvent()
                else:
                    yield TextEvent(text=json.dumps({'invalid_outer': output})
                                    if selection == 'text-origin-nested' else 'The investigation is finished.')
                    yield TurnEndEvent()
                yield ResultEvent(structured_output=output, continuation=None,
                                  structured_output_origin='native' if selection == 'native-prose' else 'text')
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    review.backend = SelectionBackend(review.repo)
    reason = {'text-origin-nested': 'malformed_output', 'empty-final-turn': 'missing_output'}.get(selection)
    await review.finish('python', reason=reason, findings=('Grounded defect',) if selection == 'native-prose' else ())
    events = stage_ends(review, 'python')
    assert len(events) == (2 if selection == 'text-origin-nested' else 1)
    assert events[-1]['metadata']['admitted'] is (selection == 'native-prose')
    if selection == 'text-origin-nested':
        metadata = events[-1]['metadata']
        assert metadata['observed_tool_starts'] == 0 and metadata['remaining_tool_calls'] == 48
        assert metadata['schema_rejection']['error_count'] > 0


async def test_opaque_assignment_handles_cannot_alias_real_changed_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:

    repo = tmp_path / 'assignment_identity'
    payload = ''.join(f'# REQUIRED_LARGE_CHANGE_{index:04} ' + 'x' * 90 + '\n' for index in range(600))
    seed_feature_branch(repo, base={'00_oversize.md': '# Before\n', 'part:000001': 'BEFORE = 0\n'},
                        feature={'00_oversize.md': '# After\n' + payload,
                                 'part:000001': 'REAL_FILE_REQUIRED_CHANGE = 1\n'})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    targets: list[str] = []
    diffs: list[str] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'generic':
            return
        target_ids = stage['assigned_target_ids']
        assert len(target_ids) == len(set(target_ids)) and not set(targets).intersection(target_ids)
        targets.extend(target_ids)
        captured = supporting_contents(review.backend.calls[-1]['prompt'])
        scoped_diff = captured['diff']
        scoped_index = json.loads(captured['hunk-index'])
        assert set(scoped_index) == set(stage['assigned_files'])
        if '00_oversize.md' in stage['assigned_files']:
            assert 'REQUIRED_LARGE_CHANGE_' in scoped_diff
        if 'part:000001' in stage['assigned_files']:
            assert 'REAL_FILE_REQUIRED_CHANGE' in scoped_diff
        diffs.append(scoped_diff)
    review.backend.stage_response = response
    data = await review.finish('generic')
    assert set(scopes(data)['generic']['files']) == {'00_oversize.md', 'part:000001'}
    assert 'part:000001' in targets and len(targets) > 2
    combined = '\n'.join(diffs)
    for index in range(600):
        assert combined.count(f'REQUIRED_LARGE_CHANGE_{index:04}') == 1


@pytest.mark.parametrize('fault', ['unknown-target', 'unknown-candidate', 'contradiction', 'grounds',
                                  'grounds-missing', 'grounds-null', 'grounds-number', 'handoff'])
async def test_mixed_schema_and_admission_rejection_never_retries_or_loses_prior_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    review = staged_review(tmp_path, monkeypatch, source_bytes=400)
    rejected_assignments: list[tuple[str, ...]] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        if not stage['progress']:
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            return
        assignment = tuple(stage['assigned_target_ids'])
        rejected_assignments.append(assignment)
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

    review.backend.stage_response = response
    await review.finish('python', reason='malformed_output', findings=('Grounded defect',))
    assert len(rejected_assignments) == 1
    events = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in events] == [True, False]
    assert [event['metadata']['attempt'] for event in events] == [1, 1]
    assert events[-1]['metadata']['observed_tool_starts'] == 0
    assert events[-1]['metadata']['remaining_tool_calls'] == 48
    assert events[-1]['metadata']['schema_rejection']['error_count'] > 0
    assert all('FAILED_MIXED_ATTEMPT_MARKER' not in call['prompt'] for call in review.backend.calls)


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-paths', 'inline'])
async def test_old_only_hunk_and_later_continuations_keep_canonical_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool,
) -> None:

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

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        prompt = review.backend.calls[-1]['prompt']
        captured = supporting_contents(prompt)
        scoped = json.loads(captured['hunk-index'])['api.py']
        fragments.append(captured['diff'])
        for part in stage['assignment_parts']:
            assignments.append(part)
            expected = (part['old_start'], part['old_start'] + part['old_count'] - 1,
                        part['new_start'], part['new_start'] + part['new_count'] - 1)
            assert any((hunk['old_start'], hunk['old_end'], hunk['new_start'], hunk['new_end']) == expected
                       for hunk in scoped['hunks'])
            if part['new_count'] != 0:
                assert (part['old_start'], part['old_count'], part['new_start'], part['new_count']) == (100, 1, 99, 501)

    review.backend.stage_response = response
    assert await review.run(findings_out=None) == 0
    coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
    assert all(scope['status'] == 'complete' for scope in coverage['stack_outcomes'])
    assert assignments[0]['old_start'] == 1 and assignments[0]['new_count'] == 0
    combined = '\n'.join(fragments)
    assert combined.count('-DELETED_FIRST_LINE = True') == 1
    for index in range(500):
        assert combined.count(f'LATER_REQUIRED_CHANGE_{index:04}') == 1


@pytest.mark.parametrize('fault', ['missing-notes', 'wrong-type-notes', 'native-missing-notes'])
async def test_schema_only_field_shape_rejection_uses_backend_correction_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    review = staged_review(tmp_path, monkeypatch, source_bytes=400, files=2)
    attempts = 0

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        nonlocal attempts
        if stage['scope_id'] != 'python':
            return
        if fault == 'native-missing-notes':
            from daydream.backends import PiRequestConfig, RequestEvent
            yield RequestEvent(prompt=review.backend.calls[-1]['prompt'],
                               output_schema=stage['response_contract']['schema'],
                               config=PiRequestConfig(schema_emulated=False, no_tools=False))
        attempts += 1
        if attempts == 1:
            output['candidates'] = [candidate(disposition='confirmed', finding=dict(
                record(), description='FAILED_SCHEMA_ONLY_ATTEMPT'))]
            if fault in {'missing-notes', 'native-missing-notes'}:
                output.pop('notes')
            else:
                output['notes'] = 3
        else:
            assert stage['attempt'] == 2 and stage['observed_tool_starts'] == 0
            assert stage['remaining_tool_calls'] == 48
            assert stage['notes'] == stage['candidates'] == []
            assert 'FAILED_SCHEMA_ONLY_ATTEMPT' not in review.backend.calls[-1]['prompt']
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]

    review.backend.stage_response = response
    native = fault == 'native-missing-notes'
    await review.finish('python', reason='malformed_output' if native else None,
                        findings=() if native else ('Grounded defect',))
    assert attempts == (1 if native else 2)
    events = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in events] == ([False] if native else [False, True])


@pytest.mark.parametrize('bundle_capable', [False, True], ids=['legacy-files', 'explicit-bundle'])
async def test_current_unhashable_callable_override_runs_each_real_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, bundle_capable: bool,
) -> None:
    ext_dir.write_module(
        'from dataclasses import dataclass\n'
        'from daydream.deep.prompts import build_per_stack_prompt\n'
        '@dataclass\n'
        'class Builder:\n'
        f'    review_input_bundle: bool = {bundle_capable!r}\n'
        '    def __call__(self, *, strategy, stack_name, files, diff_path, intent_path, alternatives_path, '
        'output_path, cwd, exploration_dir=None, prior_commits=None, inline_diff=None, '
        'intent_authoritative=False, include_alternatives=True, frontier_files=None, review_stage=None):\n'
        '        args = dict(locals()); del args["self"]\n'
        '        return build_per_stack_prompt(**args) + "\\nUNHASHABLE_CURRENT_CONTRACT_OVERRIDE"\n'
        'def register(r): r.override_prompt("per-stack", Builder())\n', api_version=8,
    )
    review = staged_review(tmp_path, monkeypatch, source_bytes=400, files=5)
    invocations: list[list[str]] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        prompt = review.backend.calls[-1]['prompt']
        assert 'UNHASHABLE_CURRENT_CONTRACT_OVERRIDE' in prompt
        supporting = supporting_contents(prompt)
        assert supporting['diff'].startswith('diff --git ')
        assert isinstance(json.loads(supporting['hunk-index']), dict)
        assert ('review-assignment' in stage['context_inputs']) is bundle_capable
        invocations.append(stage['assigned_files'])
        for path in stage['assigned_files']:
            body = (review.repo / path).read_text()
            assert ("return 'universe'" in body) if path == 'api.py' else 'VALUE = 1' in body
        output['notes'] = 'The changed source declarations preserve their assigned value contract.'

    review.backend.stage_response = response
    await review.finish('python')
    assert len(invocations) == 2
    assert {path for files in invocations for path in files} == {'api.py', *(f'module_{i:02}.py' for i in range(4))}
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))
