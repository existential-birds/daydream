"""Frozen source access and decisions through the production review runner."""
from __future__ import annotations

import ast
import json
import re
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ResultEvent, ToolResultEvent, ToolStartEvent
from daydream.deep.records import record_uid
from tests.deep_orchestrator.test_review_completion import scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, StagedBackend, stage_ends
from tests.harness.git_helpers import git, seed_feature_branch, tracked_source_state
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.review_result import merge_result
from tests.harness.stub_backend import review_stage_state


class NativeSourceBackend(StagedBackend):
    supports_source_recipe = True

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if self.sandbox and 'cross-stack merge agent' in prompt.lower():
            self.calls.append({'prompt': prompt, **kwargs})
            items: list[dict[str, Any]] = []
            for match in re.finditer(r'<sanctioned-input label="stack-records-[^"]+">\n(.*?)\n</sanctioned-input>',
                                     prompt, flags=re.S):
                for record in json.loads(match[1])['issues']:
                    items.append({**{key: record[key] for key in (
                        'file', 'line', 'severity', 'description', 'confidence', 'rationale', 'evidence')},
                        'id': len(items) + 1, 'lens': 'per-stack', 'source_uids': [record_uid(record)]})
            yield ResultEvent(structured_output=merge_result(items), continuation=None)
            return
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


@pytest.mark.parametrize('defective', [False, True], ids=['unchanged-compatible', 'unchanged-contract-defect'])
@pytest.mark.parametrize('transport', ['read', 'full-sha-git-show'])
async def test_unchanged_tracked_dependency_has_independently_verified_source_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defective: bool, transport: str,
) -> None:
    repo = tmp_path / 'unchanged_dependency'
    dependency = 'def greeting():\n    return ' + ('"universe"\n' if defective else '"world"\n')
    api = 'from dependency import greeting\nEXPECTED = "world"\nassert greeting() == EXPECTED\n'
    seed_feature_branch(repo, base={'api.py': api, 'dependency.py': dependency},
                        feature={'api.py': api + '# Preserve the greeting contract.\n'})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    decisions: list[str] = []
    title = 'Unchanged dependency violates the retained greeting contract'

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        for path in ('api.py', 'dependency.py'):
            source = (repo / path).read_text()
            if path == 'dependency.py' and transport == 'full-sha-git-show':
                call = ToolStartEvent(id=f'actual-{path}', name='Bash',
                                      input={'command': f'git show {review.pr.head_sha}:{path}'})
                assert git(repo, 'show', f'{review.pr.head_sha}:{path}') == source.rstrip('\n')
            else:
                call = ToolStartEvent(id=f'actual-{path}', name='Read', input={'file_path': path})
            yield call
            yield ToolResultEvent(id=f'actual-{path}', output=source, is_error=False)
        tree = ast.parse((repo / 'dependency.py').read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        returned = fn.body[0]
        assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Constant)
        assert isinstance(returned.value.value, str)
        decisions.append(returned.value.value)
        assert 'EXPECTED = "world"' in (repo / 'api.py').read_text()
        if returned.value.value != 'world':
            finding = {'id': 1, 'file': 'dependency.py', 'line': 2, 'severity': 'medium', 'confidence': 'MEDIUM',
                       'description': title, 'rationale': 'The retained caller requires world from greeting.',
                       'evidence': 'dependency.py:2 returns universe; api.py:2-3 requires world.'}
            output['candidates'] = [{'candidate_id': '', 'file': 'dependency.py', 'line': 2,
                                     'trigger': 'Run the retained API assertion', 'consequence': 'AssertionError',
                                     'grounds': finding['evidence'], 'disposition': 'confirmed', 'finding': finding}]
        output['notes'] = 'The retained API expectation was compared to the unchanged tracked dependency return.'

    review.backend.stage_response = response
    await review.finish('python', findings=(title,) if defective else ())
    assert decisions == ['universe' if defective else 'world']
    metadata, = [event['metadata'] for event in stage_ends(review, 'python')]
    assert metadata['fresh_source_reads'] == 2 and metadata['admitted']


@pytest.mark.parametrize('fault', ['wrong-bytes', 'untracked', 'escape'])
async def test_dependency_read_faults_cannot_establish_source_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    repo = tmp_path / 'dependency_fault'
    dependency = 'def greeting():\n    return "universe"\n'
    api = 'from dependency import greeting\nassert greeting() == "world"\n'
    seed_feature_branch(repo, base={'api.py': api, 'dependency.py': dependency},
                        feature={'api.py': api + '# Preserve the caller contract.\n'})
    if fault == 'untracked':
        git(repo, 'rm', '--cached', 'dependency.py')
        git(repo, 'commit', '-m', 'stop tracking dependency')
        (repo / '.git/info/exclude').write_text('dependency.py\n')
    if fault == 'escape':
        (tmp_path / 'dependency.py').write_text(dependency)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    native_results: list[ToolResultEvent] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        source = (repo / 'api.py').read_text()
        yield ToolStartEvent(id='fresh-api', name='Read', input={'file_path': 'api.py'})
        yield ToolResultEvent(id='fresh-api', output=source, is_error=False)
        path = '../dependency.py' if fault == 'escape' else 'dependency.py'
        returned = dependency.replace('universe', 'world') if fault == 'wrong-bytes' else dependency
        yield ToolStartEvent(id='faulty-dependency', name='Read', input={'file_path': path})
        result = ToolResultEvent(id='faulty-dependency', output=returned, is_error=False)
        native_results.append(result)
        yield result
        finding = {'id': 1, 'file': 'dependency.py', 'line': 2, 'severity': 'medium', 'confidence': 'MEDIUM',
                   'description': 'Dependency greeting violates caller',
                   'rationale': 'A world caller receives universe.',
                   'evidence': 'dependency.py:2 returns universe; api.py:2 expects world.'}
        output['candidates'] = [{'candidate_id': '', 'file': 'dependency.py', 'line': 2,
                                 'trigger': 'Run API', 'consequence': 'AssertionError',
                                 'grounds': finding['evidence'], 'disposition': 'confirmed', 'finding': finding}]

    review.backend.stage_response = response
    await review.finish('python', reason='evidence_incomplete')
    metadata, = [event['metadata'] for event in stage_ends(review, 'python')]
    assert not metadata['admitted'] and metadata['attempt'] == 1
    assert metadata['fresh_source_reads'] == 1
    assert all(result.is_error is False and not result.truncated for result in native_results)


@pytest.mark.parametrize('required', [False, True], ids=['retired-clean', 'live-caller-defect'])
@pytest.mark.parametrize('inline', [False, True], ids=['exact-projection', 'inline-native-recipe'])
async def test_deleted_source_is_readable_without_git_and_supports_a_real_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: bool, inline: bool,
) -> None:
    repo = tmp_path / 'deleted_contract'
    old = '# A public greeting used by retained consumers.\ndef hello():\n    return "world"\n'
    caller = 'from retired import hello\nprint(hello())\n' if required else 'print("world")\n'
    seed_feature_branch(repo, base={'retired.py': old, 'consumer.py': caller},
                        feature={'consumer.py': caller + '# audited consumer\n'})
    git(repo, 'rm', 'retired.py')
    git(repo, 'commit', '-m', 'remove greeting module')
    before = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    review.backend.default_source_reads = False
    review.backend.sandbox = inline
    reads: list[dict[str, Any]] = []
    decided: list[bool] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            for window in stage.get('source_access', []):
                if window['read_required']:
                    access = window['access']
                    if inline:
                        recipe = review.backend.calls[-1]['source_recipe']
                        source = recipe.selector(**access['arguments'])
                        native = json.dumps({'source': source.metadata(), 'body': source.body})
                        name, inputs = 'read_source', access['arguments']
                    else:
                        path = Path(access['path'])
                        name, inputs, native = 'Read', {'file_path': str(path)}, path.read_text()
                    yield ToolStartEvent(id=f'structure-{window["file"]}', name=name, input=inputs)
                    yield ToolResultEvent(id=f'structure-{window["file"]}', output=native, is_error=False)
            return
        bodies: dict[str, str] = {}
        for window in stage.get('source_access', []):
            if not window['read_required']:
                continue
            access = window['access']
            if inline:
                recipe = review.backend.calls[-1]['source_recipe']
                source = recipe.selector(**access['arguments'])
                body = source.body
                native = json.dumps({'source': {key: value for key, value in window.items()
                                              if key not in {'access', 'read_required'}}, 'body': body})
                assert access['tool'] == 'read_source' and str(repo / '.daydream') not in json.dumps(window)
                tool_name, tool_input = access['tool'], access['arguments']
            else:
                path = Path(access['path'])
                body = path.read_text(encoding='utf-8')
                native = body
                tool_name, tool_input = 'Read', {'file_path': str(path)}
            reads.append(window)
            bodies[window['file']] = body
            call = f'frozen-{len(reads)}'
            yield ToolStartEvent(id=call, name=tool_name, input=tool_input)
            yield ToolResultEvent(id=call, output=native, is_error=False)
        if 'retired.py' not in bodies or 'consumer.py' not in bodies:
            return
        assert bodies['retired.py'] == old
        assert 'def hello():' in bodies['retired.py'] and not (repo / 'retired.py').exists()
        broken = 'from retired import hello' in bodies['consumer.py']
        decided.append(broken)
        output['notes'] = 'Deleted greeting remains imported.' if broken else 'Consumer no longer imports the module.'
        if broken:
            finding = {'id': 1, 'file': 'consumer.py', 'line': 1, 'severity': 'medium',
                       'description': 'Retained consumer imports a deleted greeting module',
                       'confidence': 'MEDIUM', 'rationale': 'The import cannot resolve after deleting retired.py.',
                       'evidence': 'consumer.py:1 imports retired.hello; retired.py:2 defined hello before deletion.'}
            output['candidates'] = [{'candidate_id': '', 'file': 'consumer.py', 'line': 1,
                                     'trigger': 'Run the retained consumer', 'consequence': 'ModuleNotFoundError',
                                     'grounds': finding['evidence'], 'disposition': 'confirmed', 'finding': finding}]

    review.backend.stage_response = response
    assert await review.run() == 0
    data = review.load()
    assert scopes(data)['python']['status'] == 'complete', 'deleted source must have an admissible read-only transport'
    assert all(call.get('read_only') is True for call in review.backend.calls
               if review_stage_state(call['prompt']) is not None)
    assert decided == [required]
    assert [finding['title'] for finding in data['findings']] == (
        ['Retained consumer imports a deleted greeting module'] if required else [])
    deleted = [window for window in reads if window['file'] == 'retired.py']
    assert len(deleted) == 1 and deleted[0]['side'] == 'before'
    assert deleted[0]['revision'] == review.pr.base_sha
    assert all(window['read_required'] for window in reads)
    assert stage_ends(review, 'python')[0]['metadata']['observed_tool_starts'] == 2
    assert tracked_source_state(repo) == before


async def test_clean_admitted_source_is_reused_for_later_units_and_new_review_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / 'clean_source_reuse'
    old = ''.join(f'# Previous contract section {index:03}: ' + 'x' * 110 + '\n' for index in range(600))
    current = ('def encode_payload(value):\n    return {"payload": value}\n'
               + ''.join(f'# New contract section {index:03}: ' + 'y' * 40 + '\n' for index in range(70))
               + 'def send_payload(value):\n    return encode_payload(value)["payload"]\n')
    seed_feature_branch(repo, base={'api.py': old}, feature={'api.py': current})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    review.backend.default_source_reads = False
    read_calls: list[str] = []
    decisions: list[str] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            for window in stage.get('source_access', []):
                if window['read_required']:
                    path = Path(window['access']['path'])
                    yield ToolStartEvent(id=f'structure-{window["file"]}', name='Read',
                                         input={'file_path': str(path)})
                    yield ToolResultEvent(id=f'structure-{window["file"]}', output=path.read_text(), is_error=False)
            return
        assigned, = stage['assigned_target_ids']
        decisions.append(assigned)
        source_windows = [window for window in stage.get('source_access', []) if window['side'] == 'after']
        if len(decisions) == 1:
            for window in stage['source_access']:
                path = Path(window['access']['path'])
                body = path.read_text()
                assert body == (current if window['side'] == 'after' else old)
                if window['side'] == 'after':
                    assert 'encode_payload(value)["payload"]' in body
                read_calls.append(assigned)
                call_id = f'complete-clean-source-{window["side"]}'
                yield ToolStartEvent(id=call_id, name='Read', input={'file_path': str(path)})
                yield ToolResultEvent(id=call_id, output=body, is_error=False)
            output['notes'] = ('ENCODER_RETURNS_PAYLOAD: encode_payload stores value at payload; '
                               'send_payload retrieves it.')
        else:
            assert source_windows and all(not window['read_required'] for window in source_windows)
            assert stage['admitted_source_windows']
            instruction = review.backend.calls[-1]['review_instructions']
            identities = json.JSONDecoder().raw_decode(instruction.split('Persistent assignment identities: ', 1)[1])[0]
            assert identities['admitted_source_windows'] == stage['admitted_source_windows']
            current_receipt, = [window for window in identities['admitted_source_windows']
                               if window['side'] == 'after']
            assert (current_receipt['revision'], current_receipt['start_byte'], current_receipt['end_byte']) == (
                review.pr.head_sha, 0, len(current.encode()))
            assert any('ENCODER_RETURNS_PAYLOAD' in note for note in stage['notes'])
            # Host receipts retain authority without carrying source bodies into
            # each fresh continuation. The admitted note preserves the decision.
            assert stage['evidence'] == []
            output['notes'] = 'The newly assigned continuation preserves the same payload producer/consumer contract.'

    review.backend.stage_response = response
    await review.finish('python', review_profile=independent_alternatives_profile())
    assert len(decisions) > 2 and len(set(decisions)) == len(decisions)
    assert len(read_calls) == 2
    phases = stage_ends(review, 'python')
    assert all(phase['metadata']['admitted'] for phase in phases)
    assert phases[-1]['metadata']['observed_tool_starts'] == 2


@pytest.mark.parametrize('defective', [False, True], ids=['compatible-addition', 'added-greeting-defect'])
async def test_added_source_requires_real_reads_even_when_the_complete_addition_is_in_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defective: bool,
) -> None:
    repo = tmp_path / 'readable_addition'
    body = 'def hello():\n    return ' + ("'universe'\n" if defective else "'world'\n")
    seed_feature_branch(repo, base={'consumer.py': 'EXPECTED_GREETING = "world"\n'},
                        feature={'api.py': body})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    decisions: list[str] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        window, = [window for window in stage['source_access'] if window['file'] == 'api.py']
        assert window['side'] == 'after' and window['read_required']
        path = Path(window['access']['path'])
        current = path.read_text()
        assert current == body
        yield ToolStartEvent(id='required-added-source', name='Read', input={'file_path': str(path)})
        yield ToolResultEvent(id='required-added-source', output=current, is_error=False)
        consumer = (repo / 'consumer.py').read_text()
        yield ToolStartEvent(id='existing-consumer-contract', name='Read', input={'file_path': 'consumer.py'})
        yield ToolResultEvent(id='existing-consumer-contract', output=consumer, is_error=False)
        assert 'EXPECTED_GREETING = "world"' in consumer
        decisions.append(current)
        if "return 'universe'" in current:
            finding = {'id': 1, 'file': 'api.py', 'line': 2, 'severity': 'medium', 'confidence': 'MEDIUM',
                       'description': 'Added greeting violates the existing consumer contract',
                       'rationale': 'The consumer expects world; the added greeting returns universe.',
                       'evidence': 'api.py:2 returns universe; consumer.py:1 expects world.'}
            output['candidates'] = [{'candidate_id': '', 'file': 'api.py', 'line': 2,
                                     'trigger': 'Call the added hello function', 'consequence': 'Wrong greeting',
                                     'grounds': finding['evidence'], 'disposition': 'confirmed', 'finding': finding}]

    review.backend.stage_response = response
    await review.finish('python', findings=('Added greeting violates the existing consumer contract',)
                        if defective else ())
    assert decisions == [body]
    metadata, = [phase['metadata'] for phase in stage_ends(review, 'python')]
    assert metadata['observed_tool_starts'] == 2 and metadata['admitted']


@pytest.mark.parametrize('fault', ['none', 'body', 'range', 'revision', 'digest', 'selector',
                                  'truncated', 'cancelled', 'opaque-range', 'impostor-path', 'symlink', 'changed-file',
                                  'supporting-only', 'body-then-valid', 'read-body-then-valid'])
async def test_source_admission_revalidates_bytes_ranges_identity_and_confinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    repo = tmp_path / 'source_faults'
    body = 'def hello():\n    return "世界"\n'
    seed_feature_branch(repo, base={'consumer.py': 'EXPECTED_GREETING = "世界"\n'}, feature={'api.py': body})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    inline = fault not in {'opaque-range', 'impostor-path', 'symlink', 'changed-file', 'supporting-only',
                           'read-body-then-valid'}
    review.backend.sandbox = inline
    emitted: list[str] = []
    native_results: list[ToolResultEvent] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        window, = [window for window in stage['source_access'] if window['file'] == 'api.py']
        if inline:
            inputs = dict(window['access']['arguments'])
            recipe = review.backend.calls[-1]['source_recipe']
            source = recipe.selector(**inputs)
            packet = {'source': source.metadata(), 'body': source.body}
            assert source.body == body and source.start_byte == 0
            if fault in {'body', 'body-then-valid'}:
                packet['body'] = body.replace('世界', 'forged')
            elif fault == 'range':
                packet['source']['start_byte'] = 1
            elif fault == 'revision':
                packet['source']['revision'] = review.pr.base_sha
            elif fault == 'digest':
                packet['source']['content_sha256'] = '0' * 64
            elif fault == 'selector':
                inputs['target_id'] = 'not-assigned'
            name, native = 'read_source', json.dumps(packet)
        else:
            path = Path(window['access']['path'])
            if fault == 'supporting-only':
                path = Path(stage['supporting_bundle']['path'])
            elif fault == 'impostor-path':
                path = tmp_path / 'source-impostor.txt'
                path.write_text(body)
            elif fault in {'symlink', 'changed-file'}:
                path.unlink()
                if fault == 'symlink':
                    outside = tmp_path / 'outside-source.txt'
                    outside.write_text(body)
                    path.symlink_to(outside)
                else:
                    path.write_text(body.replace('世界', 'forged'))
            inputs = {'file_path': str(path)}
            if fault == 'opaque-range':
                inputs['window'] = 'provider-opaque-range'
            name, native = 'Read', path.read_text()
            if fault == 'read-body-then-valid':
                native = native.replace('世界', 'forged')
        emitted.append(native)
        yield ToolStartEvent(id='native-source-fault', name=name, input=inputs)
        result = ToolResultEvent(id='native-source-fault', output=native, is_error=False,
                                 truncated=fault == 'truncated', cancelled=fault == 'cancelled')
        native_results.append(result)
        yield result
        if fault in {'body-then-valid', 'read-body-then-valid'}:
            valid = json.dumps({'source': source.metadata(), 'body': source.body}) if inline else body
            yield ToolStartEvent(id='fresh-after-invalid-source', name=name, input=inputs)
            valid_result = ToolResultEvent(id='fresh-after-invalid-source', output=valid, is_error=False)
            native_results.append(valid_result)
            yield valid_result
        output['notes'] = 'Greeting source is compatible with its consumer contract.'

    review.backend.stage_response = response
    if fault == 'symlink':
        assert await review.run() == 1
        assert len(emitted) == 1 and not review.output.exists()
        assert len([stage for stage in review.backend.stages if stage['scope_id'] == 'python']) == 1
        return
    await review.finish('python', reason=None if fault == 'none' else 'evidence_incomplete',
                        statuses=None if fault == 'none' else ('failed', 'incomplete'))
    metadata, = [phase['metadata'] for phase in stage_ends(review, 'python')]
    assert len(emitted) == 1 and metadata['attempt'] == 1
    assert native_results[0].is_error is False
    assert native_results[0].truncated is (fault == 'truncated')
    assert native_results[0].cancelled is (fault == 'cancelled')
    assert metadata['admitted'] is (fault == 'none')
    assert metadata['observed_tool_starts'] == (2 if fault in {'body-then-valid', 'read-body-then-valid'} else 1)
    if fault == 'supporting-only':
        assert metadata['fresh_source_reads'] == metadata['source_body_bytes'] == 0
    if fault != 'none':
        assert metadata['failure_class'] in {'source_access_failure', 'missing_source_receipt', 'capture_loss'}


@pytest.mark.parametrize('oversized', [False, True], ids=['before-only-normal', 'old-unicode-continuations'])
async def test_deleted_before_only_windows_cover_native_old_continuations_with_verified_byte_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oversized: bool,
) -> None:
    repo = tmp_path / 'old_continuations'
    old = '# Obsolete state ' + ('界' * 60_000 if oversized else 'retired') + '\nVALUE_1 = 1\n'
    raw = old.encode('utf-8')
    seed_feature_branch(repo, base={'retired.py': old, 'consumer.py': 'CURRENT_CONTRACT = "active"\n'},
                        feature={'consumer.py': 'CURRENT_CONTRACT = "active"\n'
                                                '# Removed the unused historical module.\n'})
    git(repo, 'rm', 'retired.py')
    git(repo, 'commit', '-m', 'remove unused historical source')
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    windows: list[dict[str, Any]] = []
    units: list[str] = []
    old_decisions: list[bool] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        for window in stage['source_access']:
            if window['file'] != 'retired.py':
                continue
            assert window['side'] == 'before' and window['source_path'] == 'retired.py'
            assert window['revision'] == review.pr.base_sha
            source = Path(window['access']['path']).read_text()
            assert source.encode() == raw[window['start_byte']:window['end_byte']]
            assert len(source.encode()) <= 128 * 1024
            windows.append(window)
            call = f'old-window-{len(windows)}'
            yield ToolStartEvent(id=call, name='Read', input={'file_path': window['access']['path']})
            yield ToolResultEvent(id=call, output=source, is_error=False)
            for name, value in re.findall(r'VALUE_(\d+) = (\d+)', source):
                assert name == value == '1'
                old_decisions.append(True)
        for path in stage['assigned_files']:
            if path != 'retired.py':
                body = (repo / path).read_text()
                assert 'CURRENT_CONTRACT = "active"' in body and 'import' not in body
                yield ToolStartEvent(id=f'consumer-{len(units)}', name='Read', input={'file_path': path})
                yield ToolResultEvent(id=f'consumer-{len(units)}', output=body, is_error=False)
        units.extend(stage['assigned_target_ids'])
        output['notes'] = 'The removed historical state is unused by the retained active consumer.'

    review.backend.stage_response = response
    await review.finish('python')
    assert windows and old_decisions and not (repo / 'retired.py').exists()
    assert len(units) == len(set(units)) and len(units) > (2 if oversized else 1)
    covered_end = 0
    for window in sorted(windows, key=lambda window: window['start_byte']):
        assert window['start_byte'] <= covered_end
        covered_end = max(covered_end, window['end_byte'])
    assert covered_end == len(raw)
    metadata = [event['metadata'] for event in stage_ends(review, 'python')]
    assert all(event['admitted'] for event in metadata)
    assert sum(event['base_only_source_reads'] for event in metadata) == len(windows)


@pytest.mark.parametrize('defective', [False, True], ids=['compatible-rename', 'renamed-source-defect'])
async def test_rename_sides_bind_original_paths_and_source_bytes_before_and_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defective: bool,
) -> None:
    repo = tmp_path / 'renamed_source'
    prefix = ''.join(f'# Stable API contract context {index:02}\n' for index in range(30))
    old = prefix + 'def hello():\n    return "world"\n'
    new = prefix + 'def hello():\n    return ' + ('"universe"\n' if defective else '"world"\n')
    seed_feature_branch(repo, base={'legacy.py': old, 'consumer.py': 'EXPECTED_GREETING = "world"\n'},
                        feature={'consumer.py': 'EXPECTED_GREETING = "world"\n'
                                                '# Existing public contract is retained.\n'})
    git(repo, 'mv', 'legacy.py', 'modern.py')
    (repo / 'modern.py').write_text(new)
    git(repo, 'add', 'modern.py')
    git(repo, 'commit', '-m', 'rename the greeting module')
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    observed: list[dict[str, Any]] = []
    decisions: list[str] = []
    title = 'Renamed greeting no longer satisfies the consumer contract'

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        for window in stage['source_access']:
            if window['file'] != 'modern.py':
                continue
            observed.append(window)
            source = Path(window['access']['path']).read_text()
            assert source == (old if window['side'] == 'before' else new)
            yield ToolStartEvent(id=f'rename-{window["side"]}', name='Read',
                                 input={'file_path': window['access']['path']})
            yield ToolResultEvent(id=f'rename-{window["side"]}', output=source, is_error=False)
        tree = ast.parse(new)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        returned = function.body[0]
        assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Constant)
        assert isinstance(returned.value.value, str)
        decisions.append(returned.value.value)
        consumer = (repo / 'consumer.py').read_text()
        assert 'EXPECTED_GREETING = "world"' in consumer
        yield ToolStartEvent(id='retained-consumer-contract', name='Read', input={'file_path': 'consumer.py'})
        yield ToolResultEvent(id='retained-consumer-contract', output=consumer, is_error=False)
        if returned.value.value != 'world':
            finding = {'id': 1, 'file': 'modern.py', 'line': 32, 'severity': 'medium', 'confidence': 'MEDIUM',
                       'description': title, 'rationale': 'The rename changes the public greeting unexpectedly.',
                       'evidence': 'legacy.py:32 returned world before; modern.py:32 returns universe after.'}
            output['candidates'] = [{'candidate_id': '', 'file': 'modern.py', 'line': 32,
                                     'trigger': 'Call the relocated hello function', 'consequence': 'Wrong greeting',
                                     'grounds': finding['evidence'], 'disposition': 'confirmed', 'finding': finding}]

    review.backend.stage_response = response
    await review.finish('python', findings=(title,) if defective else ())
    assert decisions == ['universe' if defective else 'world']
    assert {(window['side'], window['source_path']) for window in observed} == {
        ('before', 'legacy.py'), ('after', 'modern.py')}
    assert all(window['file'] == 'modern.py' for window in observed)
    assert not (repo / 'legacy.py').exists()


@pytest.mark.parametrize('opaque_kind', ['concatenated-shell', 'unknown-read-range'])
@pytest.mark.parametrize('fresh', [False, True], ids=['opaque-only-incomplete', 'fresh-verified-source'])
async def test_successful_opaque_source_is_supporting_and_needs_fresh_verified_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opaque_kind: str, fresh: bool,
) -> None:
    repo = tmp_path / 'opaque_source'
    api = 'from module import greeting\ndef hello():\n    return greeting()\n'
    module = 'def greeting():\n    return "world"\n'
    seed_feature_branch(repo, base={'api.py': api, 'module.py': module},
                        feature={'api.py': api + '# Retain the delegation contract.\n',
                                 'module.py': module + '# Retain the greeting contract.\n'})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    declared: list[bool] = []
    native_results: list[ToolResultEvent] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        if opaque_kind == 'concatenated-shell':
            call = ToolStartEvent(id='opaque-success', name='Bash', input={'command': 'cat api.py module.py'})
            body = (repo / 'api.py').read_text() + (repo / 'module.py').read_text()
        else:
            window = next(window for window in stage['source_access']
                          if window['file'] == 'api.py' and window['side'] == 'after')
            call = ToolStartEvent(id='opaque-success', name='Read',
                                 input={'file_path': window['access']['path'], 'provider_range': 'opaque'})
            body = Path(window['access']['path']).read_text()
        result = ToolResultEvent(id=call.id, output=body, is_error=False)
        native_results.append(result)
        yield call
        yield result
        if fresh:
            for window in stage['source_access']:
                if window['side'] != 'after':
                    continue
                source = Path(window['access']['path']).read_text()
                yield ToolStartEvent(id=f'fresh-{window["file"]}', name='Read',
                                     input={'file_path': window['access']['path']})
                yield ToolResultEvent(id=f'fresh-{window["file"]}', output=source, is_error=False)
                tree = ast.parse(source)
                fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
                returned = fn.body[0]
                assert isinstance(returned, ast.Return)
                if window['file'] == 'api.py':
                    assert isinstance(returned.value, ast.Call) and isinstance(returned.value.func, ast.Name)
                    assert returned.value.func.id == 'greeting'
                else:
                    assert isinstance(returned.value, ast.Constant) and returned.value.value == 'world'
        declared.append(fresh)
        output['notes'] = 'hello delegates to greeting, whose literal still returns the required world greeting.'

    review.backend.stage_response = response
    await review.finish('python', reason=None if fresh else 'evidence_incomplete')
    assert declared == [fresh] and native_results[0].is_error is False
    metadata, = [phase['metadata'] for phase in stage_ends(review, 'python')]
    assert metadata['admitted'] is fresh
    assert metadata['fresh_source_reads'] == (2 if fresh else 0)
    assert metadata['observed_tool_starts'] == (3 if fresh else 1)


@pytest.mark.parametrize('fault', ['none', 'forged-footer', 'truncated'],
                         ids=['genuine-bounded-reads', 'forged-footer-terminal', 'native-truncation-terminal'])
async def test_pi_bounded_read_representation_preserves_real_source_range_union_and_rejects_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    repo = tmp_path / 'native_range_union'
    body = 'VALUE = 1\ndef greeting():\n    return "world"\nEXPECTED = "world"\nassert greeting() == EXPECTED\n'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n'}, feature={'api.py': body})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    outputs: list[ToolResultEvent] = []
    judged: list[bool] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        source = (repo / 'api.py').read_text()
        assert source == body
        expected = [
            'VALUE = 1\ndef greeting():\n\n[4 more lines in file. Use offset=3 to continue.]',
            '    return "world"\nEXPECTED = "world"\nassert greeting() == EXPECTED'
            '\n\n[1 more lines in file. Use offset=6 to continue.]',
        ]
        for index, (offset, limit) in enumerate([(1, 2), (3, 3)]):
            native = expected[index]
            if index == 0 and fault == 'forged-footer':
                native = native.replace('4 more lines', '400 more lines').replace('offset=3', 'offset=777')
            yield ToolStartEvent(id=f'bounded-native-{index}', name='read',
                                 input={'path': 'api.py', 'offset': offset, 'limit': limit})
            result = ToolResultEvent(id=f'bounded-native-{index}', output=native, is_error=False,
                                     truncated=index == 0 and fault == 'truncated')
            outputs.append(result)
            yield result
        tree = ast.parse(source)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        returned = fn.body[0]
        assert isinstance(returned, ast.Return) and isinstance(returned.value, ast.Constant)
        judged.append(returned.value.value == 'world')
        assert judged[-1]
        output['notes'] = 'The changed value is one and greeting agrees with EXPECTED=world.'

    review.backend.stage_response = response
    await review.finish('python', reason=None if fault == 'none' else 'evidence_incomplete')
    assert judged == [True] and len(outputs) == 2
    assert all(result.is_error is False for result in outputs)
    metadata, = [event['metadata'] for event in stage_ends(review, 'python')]
    assert metadata['admitted'] is (fault == 'none')
    assert metadata['attempt'] == 1 and metadata['observed_tool_starts'] == 2
    if fault == 'none':
        assert metadata['fresh_source_reads'] == 2 and metadata['source_body_bytes'] == len(body.encode())
    else:
        assert metadata['failure_class'] in {'source_access_failure', 'capture_loss'}


@pytest.mark.parametrize('symlink', [False, True], ids=['regular-source', 'unavailable-nonregular-source'])
async def test_added_nonregular_source_has_typed_access_failure_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symlink: bool,
) -> None:
    repo = tmp_path / 'source_mode'
    body = 'def greeting():\n    return "world"\n'
    seed_feature_branch(repo, base={'target.py': body}, feature={'target.py': body + '# feature context\n'})
    if symlink:
        (repo / 'api.py').symlink_to('target.py')
    else:
        (repo / 'api.py').write_text(body)
    git(repo, 'add', 'api.py')
    git(repo, 'commit', '-m', 'add public source')
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    judged: list[str] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] != 'python':
            return
        tree = ast.parse((repo / 'api.py').read_text())
        fn, = tree.body
        assert isinstance(fn, ast.FunctionDef) and isinstance(fn.body[0], ast.Return)
        assert isinstance(fn.body[0].value, ast.Constant)
        assert isinstance(fn.body[0].value.value, str)
        judged.append(fn.body[0].value.value)
        output['notes'] = 'The added greeting preserves its world return contract.'

    review.backend.stage_response = response
    data = await review.finish('python', reason='evidence_incomplete' if symlink else None)
    phases = stage_ends(review, 'python')
    if symlink:
        assert judged == [] and not any(s['scope_id'] == 'python' for s in review.backend.stages)
        assert not phases
        coverage = json.loads((repo / '.daydream/deep/review-coverage.json').read_text())
        assert coverage['diagnostics']['scopes']['python'].startswith('Frozen source access unavailable or rejected:')
        assert scopes(data)['structure']['reason_codes'] == ['evidence_incomplete']
    else:
        assert len(phases) == 1
        metadata = phases[0]['metadata']
        assert judged == ['world'] and metadata['admitted'] and metadata['fresh_source_reads'] > 0


@pytest.mark.parametrize('defective', [False, True], ids=['unchanged-dependency-clean', 'open-dependency-confirmed'])
async def test_open_unchanged_dependency_reaches_source_grounded_empty_target_triage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defective: bool,
) -> None:
    repo = tmp_path / 'dependency_triage'
    dependency = 'def greeting():\n    return ' + ('"universe"\n' if defective else '"world"\n')
    api = 'from dependency import greeting\nEXPECTED = "world"\nassert greeting() == EXPECTED\n'
    seed_feature_branch(repo, base={'api.py': api, 'dependency.py': dependency},
                        feature={'api.py': api + '# Preserve the greeting contract.\n'})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    review.backend = NativeSourceBackend(repo)
    title = 'Unchanged greeting dependency violates the retained caller contract'
    decisions: list[tuple[str, str]] = []

    def response(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        if stage['scope_id'] != 'python':
            return
        for path in ('api.py', 'dependency.py'):
            body = (repo / path).read_text()
            yield ToolStartEvent(id=f'{stage["stage"]}-{path}', name='Read', input={'file_path': path})
            yield ToolResultEvent(id=f'{stage["stage"]}-{path}', output=body, is_error=False)
        tree = ast.parse(dependency)
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef) and isinstance(fn.body[0], ast.Return)
        assert isinstance(fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str)
        actual = fn.body[0].value.value
        decisions.append((stage['stage'], actual))
        assert 'EXPECTED = "world"' in (repo / 'api.py').read_text()
        if actual == 'world':
            output['notes'] = 'The unchanged dependency preserves the retained world contract.'
            return
        finding = {'id': 1, 'file': 'dependency.py', 'line': 2, 'severity': 'medium', 'confidence': 'MEDIUM',
                   'description': title, 'rationale': 'The retained API requires world from greeting.',
                   'evidence': 'dependency.py:2 returns universe; api.py:2-3 requires world.'}
        candidate_id = '' if stage['stage'] == 'first_pass' else stage['assigned_candidate_ids'][0]
        if stage['stage'] == 'triage':
            assert stage['assigned_target_ids'] == output['targets'] == []
            assert stage['assigned_files'] == ['dependency.py']
        output['candidates'] = [{'candidate_id': candidate_id, 'file': 'dependency.py', 'line': 2,
                                 'trigger': 'Run the retained caller assertion', 'consequence': 'AssertionError',
                                 'grounds': finding['evidence'],
                                 'disposition': 'open' if stage['stage'] == 'first_pass' else 'confirmed',
                                 'finding': None if stage['stage'] == 'first_pass' else finding}]
        output['notes'] = 'The actual dependency return contradicts the retained caller expectation.'

    review.backend.stage_response = response
    await review.finish('python', findings=(title,) if defective else ())
    assert decisions == ([('first_pass', 'universe'), ('triage', 'universe')] if defective
                         else [('first_pass', 'world')])
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))
