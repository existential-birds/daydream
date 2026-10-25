"""Large staged-review transport and retry scenarios through the production runner."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
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
from tests.deep_orchestrator.test_review_investigation import (
    InvestigationRun,
    assert_host_dispatch_measurements,
    candidate,
    read_events,
    stage_ends,
)
from tests.deep_orchestrator.test_review_stage_reservations import NativeStageBackend
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


def part_identity(parts: list[dict[str, Any]]) -> str:
    """Pin every original assignment field while normalizing freshly created Git revisions."""
    normalized = copy.deepcopy(parts)
    for part in normalized:
        for reference in part['source_refs']:
            reference['revision'] = reference['side']
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


@pytest.mark.parametrize('limit', ['files', 'bytes', 'all-omitted', 'navigation'])
async def test_structure_keeps_bounded_interaction_review_and_patch_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str, ext_dir: ExtDir,
) -> None:
    if limit == 'files':
        before = {f'module_{index:04}.py': 'VALUE = 0\n' for index in range(520)}
        after = {path: 'VALUE = 1\n' for path in before}
    elif limit == 'bytes':
        before = {'api.py': 'VALUE = 0\n'}
        after = {'api.py': 'VALUE = 1\n' + ('# changed contract ' + 'x' * 2100 + '\n') * 4500}
    elif limit == 'all-omitted':
        before = {f'retired_{index:04}.py': 'VALUE = 0\n' for index in range(520)}
        before['api.py'] = 'VALUE = 0\n'
        after = {'api.py': 'VALUE = 1\n'}
    else:
        ext_dir.write_module(
            'def structural(**kw): return "CUSTOM_CATALOG_BUILDER"\n'
            'structural.review_input_bundle = True\n'
            'def register(r): r.override_prompt("structural", structural)\n', api_version=8,
        )
        before = {f'module_{index:04}.py': 'VALUE = 0\n' for index in range(70)}
        before['api.py'] = 'VALUE = 0\n'
        before['retired.py'] = 'REMOVED_CONTEXT = "frozen before"\n'
        before['legacy.py'] = ''.join(f'# unchanged contract {index:02}\n' for index in range(40)) + 'VALUE = 0\n'
        after = {path: 'VALUE = 1\n' for path in before if path.startswith('module_')}
        after['api.py'] = 'VALUE = 1\n' + ''.join(
            f'# MULTIPART_CONTEXT_{index:04} ' + 'x' * 96 + '\n' for index in range(500))
        after['modern.py'] = before['legacy.py'].replace('VALUE = 0', 'VALUE = 1')
        after['zzz_late.py'] = 'LATE_CONTEXT = "frozen after"\n'
    repo = tmp_path / 'structure_limits'
    seed_feature_branch(repo, base=before, feature=after)
    if limit == 'all-omitted':
        from tests.harness.git_helpers import git
        git(repo, 'rm', *(path for path in before if path != 'api.py'))
        git(repo, 'commit', '-m', 'remove retired sources')
    elif limit == 'navigation':
        from tests.harness.git_helpers import git
        git(repo, 'rm', 'retired.py', 'legacy.py')
        git(repo, 'commit', '-m', 'retire and rename context sources')
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    if limit != 'files':
        review.backend = NativeStageBackend(repo)
    if limit == 'all-omitted':
        @review.backend.script('python')
        def unfinished_language(stage: dict[str, Any], output: dict[str, Any]) -> None:
            # This fixture owns Structure projections, not recovery of the 131
            # unrelated language batches. Ordinary invocation failures continue.
            raise RuntimeError('Language reviewer not configured in this Structure-only fixture.')
    catalog_seen = False

    @review.backend.script('structure')
    def respond(stage: dict[str, Any], output: dict[str, Any]) -> None:
        nonlocal catalog_seen
        if limit == 'all-omitted':
            assert 'supporting_catalog' not in stage
            assert stage['context_availability'] == {
                'supporting_parts': 0, 'supporting_catalogs': 0, 'omitted_supporting_parts': 521}
            assert any(status['label'] == 'supporting-parts' and status['status'] == 'unavailable'
                       for status in stage['context_statuses'])
            catalog_seen = True
            return
        catalog = stage['supporting_catalog']
        assert catalog['status'] == ('complete' if limit == 'navigation' else 'partial')
        assert (catalog['omitted_part_count'] > 0) is (limit != 'navigation')
        # A compacted provider retains its system instructions, not the original user state.
        instructions = review.backend.calls[-1]['review_instructions']
        guide_line = next(line for line in instructions.splitlines()
                          if line.startswith('Persistent exact access guide (supporting metadata): '))
        guide = json.loads(guide_line.split(': ', 1)[1])
        assert guide['supporting_catalog'] == catalog
        guidance = catalog['guidance']
        assert 'displayed integration diff is an inventory' in guidance
        assert 'before/after patch context' in guidance and 'even without a Git execution tool' in guidance
        assert 'do not guarantee complete old source files' in guidance
        assert 'read only parts relevant to concrete concerns' in guidance
        pending = [Path(guide['supporting_catalog']['path'])]
        paths = set(_sanctioned_inputs(review.backend.calls[-1]['prompt']).values())
        parts: list[dict[str, Any]] = []
        while pending:
            path = pending.pop()
            assert path not in paths
            paths.add(path)
            node = json.loads(path.read_text())
            pending.extend(Path(child['path']) for child in reversed(node.get('catalogs', [])))
            parts.extend(node.get('parts', []))
        paths.update(Path(part['path']) for part in parts)
        assert len(parts) == catalog['part_count'] > 0
        assert len(paths) <= 512
        assert sum(path.stat().st_size for path in paths) <= 8 * 1024 * 1024
        assert all(path.stat().st_size <= 24 * 1024 for path in paths)
        if limit == 'navigation':
            assert stage['context_availability']['supporting_catalogs'] > 1
            by_file: dict[str, list[str]] = {}
            for part in parts:
                by_file.setdefault(part['file'], []).append(Path(part['path']).read_text())
            assert '-REMOVED_CONTEXT = "frozen before"' in ''.join(by_file['retired.py'])
            renamed = ''.join(by_file['modern.py'])
            assert 'rename from legacy.py' in renamed and 'rename to modern.py' in renamed
            assert '-VALUE = 0\n+VALUE = 1\n' in renamed
            assert '+LATE_CONTEXT = "frozen after"' in ''.join(by_file['zzz_late.py'])
            assert len(by_file['api.py']) > 1
            body = ''.join(part.split('\n@@', 1)[1].split('\n', 1)[1] for part in by_file['api.py'])
            assert body == '-VALUE = 0\n+' + after['api.py'].replace('\n', '\n+').removesuffix('+')
            prompt = review.backend.calls[-1]['prompt']
            assert 'CUSTOM_CATALOG_BUILDER' in prompt
            assert 'MULTIPART_CONTEXT_' not in prompt and 'REMOVED_CONTEXT' not in prompt
            assert all(str(Path(part['path'])) not in prompt for part in parts)
        if limit == 'files':
            assert len(parts) + catalog['omitted_part_count'] == 520
        assert stage['context_availability']['omitted_supporting_parts'] == catalog['omitted_part_count']
        if limit != 'navigation':
            assert any(status['label'] == 'supporting-parts' and status['status'] == 'partial'
                       for status in stage['context_statuses'])
        catalog_seen = True

    assert await review.run(deep_shard_enabled=False) == 0
    data = review.load()
    assert scopes(data)['structure']['status'] == 'complete'
    assert scopes(data)['structure']['reason_codes'] == []
    assert catalog_seen
    assert data['findings'] == []


@pytest.mark.parametrize('invalid_source', [False, True], ids=['later-part', 'invalid-later-part'])
@pytest.mark.parametrize('sandbox', [False, True], ids=['catalog', 'inline'])
@pytest.mark.parametrize('native', [False, True], ids=['canonical-backend', 'native-pi'])
async def test_structure_can_attribute_a_later_assigned_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_source: bool, sandbox: bool, native: bool,
) -> None:
    """Native callbacks must resolve later work beyond the bounded prompt examples."""
    review = staged_review(tmp_path, monkeypatch)
    if native:
        review.backend = NativeStageBackend(review.repo)
    review.backend.sandbox = sandbox
    canonical_digest = hashlib.sha256(git_ops.diff(review.repo, 'main').encode()).hexdigest()
    submitted: list[dict[str, Any]] = []

    @review.backend.script('structure')
    def respond(stage: dict[str, Any], output: dict[str, Any]) -> Iterable[AgentEvent]:
        offered = stage['source_refs']['integration:structure']
        assert len(offered) <= 4
        assert all(ref['path'] != 'module_06.py' for ref in offered)
        if sandbox:
            contents = supporting_contents(review.backend.calls[-1]['prompt'])
            descriptors = [(label.removesuffix('-sources'), json.loads(value))
                           for label, value in contents.items() if label.endswith('-sources')]
            label, part = next((label, part) for label, part in descriptors if part['target_id'] == 'module_06.py')
            source = contents[label].encode()
        else:
            pending = [Path(stage['supporting_catalog']['path'])]
            parts: list[dict[str, Any]] = []
            while pending:
                node = json.loads(pending.pop().read_text())
                pending.extend(Path(child['path']) for child in node.get('catalogs', []))
                parts.extend(node.get('parts', []))
            part = next(part for part in parts if part['file'] == 'module_06.py')
            source = Path(part['path']).read_bytes()
        reference = {
            'artifact_sha256': canonical_digest, 'part_sha256': hashlib.sha256(source).hexdigest(),
            'revision': review.pr.head_sha, 'path': 'module_06.py', 'side': 'new',
            'source_range': {'start': 1, 'end': 1}, 'fragment': None, 'basis': 'supplied_diff', 'role': 'assigned',
        }
        assert next(ref for ref in part['source_refs'] if ref['side'] == 'new') == reference
        assert len(part['source_options']) == len(part['source_refs'])
        option = next(option for option in part['source_options'] if option['side'] == 'new')
        assert option['basis'] == 'supplied_diff'
        expected_targets = [{**target, 'evidence_refs': [reference]} for target in output['targets']]
        submitted.extend(copy.deepcopy(expected_targets))
        if native:
            # The provider sends only a source selector. Production admission
            # must resolve it from the complete assignment inventory itself.
            selector = {'source_id': option['source_id'], 'basis': 'supplied_diff', 'source_range': None}
            output['targets'][0]['evidence_refs'] = [selector]
            binding = review.backend.calls[-1]['host_tools']
            if invalid_source:
                rejected = copy.deepcopy(output)
                rejected['targets'][0]['evidence_refs'][0]['source_id'] = 'source:' + '0' * 64
                rejected['notes'] = 'REJECTED_LATER_PART_NOTE'
                yield ToolStartEvent(id='foreign-later-part', name='structured_output', input=rejected)
                yield ToolResultEvent(id='foreign-later-part', output='Receipt.', is_error=False)
                assert binding.accepted is False and binding.possibly_accepted is False
            yield ToolStartEvent(id='later-part', name='structured_output', input=output)
            yield ToolResultEvent(id='later-part', output='Receipt.', is_error=False)
            assert binding.accepted_result['targets'] == expected_targets
        else:
            output['targets'][0]['evidence_refs'] = [copy.deepcopy(reference)]
            if invalid_source:
                output['targets'][0]['evidence_refs'][0]['part_sha256'] = '0' * 64

    rejected = invalid_source and not native
    data = await review.finish('structure', reason='evidence_incomplete' if rejected else None)
    scope = scopes(data)['structure']
    assert scope['target_decisions'] == ([] if rejected else submitted)
    event, = stage_ends(review, 'structure')
    assert event['metadata']['admitted'] is not rejected
    assert event['metadata']['failure_class'] == ('source_attribution' if rejected else None)
    if native:
        assert event['metadata']['successful_submissions'] == 1
        assert event['metadata']['failed_submissions'] == int(invalid_source)
        assert event['metadata']['submission_starts'] == 1 + int(invalid_source)
        assert 'REJECTED_LATER_PART_NOTE' not in str(data)
        assert all('source_id' not in str(decision) for decision in scope['target_decisions'])


@pytest.mark.parametrize('outcome', ['success', 'second-rejection'])
async def test_schema_rejection_retries_once_with_fresh_grounding_and_charged_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, ext_dir: ExtDir) -> None:
    ext_dir.write_module(
        'from daydream.deep.prompts import build_per_stack_prompt\n'
        'def staged(**kw):\n'
        '    return build_per_stack_prompt(**kw) + \"\\nCUSTOM_UTF8=é🙂 CUSTOM_ATTEMPT=\" '
        '+ str(kw[\"review_stage\"][\"attempt\"])\n'
        'def register(r): r.override_prompt(\"per-stack\", staged)\n', api_version=8,
    )
    review = staged_review(tmp_path, monkeypatch)
    attempts = 0

    @review.backend.script('python', terminal=False)
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
        elif attempts == 1 or outcome == 'second-rejection':
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
        yield TextEvent(text=json.dumps(output))
        yield TurnEndEvent()
        yield ResultEvent(structured_output=None, continuation=None)

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
    measurements = assert_host_dispatch_measurements(review)
    assert all(item['attempt'] == 1 for item in measurements)
    retry_scopes = {item['stage_scope_id'] for item in measurements
                    if any(event['scope_id'] == item['stage_scope_id'] and event['metadata']['attempt'] == 2
                           for event in stage_ends(review, 'python'))}
    assert len(retry_scopes) == 1


@pytest.mark.parametrize(
    ('sandbox', 'payload', 'omit_final', 'native'),
    [(False, 'medium-file', False, False), (True, 'medium-file', False, False),
    (False, 'oversized-hunk', False, False), (True, 'oversized-hunk', True, False),
    (False, 'unicode-line', False, False), (True, 'unicode-separators', False, False),
    (False, 'unicode-line', False, True), (True, 'unicode-line', False, True)])
async def test_oversized_hunk_assignments_reassemble_every_canonical_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool, payload: str, omit_final: bool, native: bool,
) -> None:
    unicode = payload in {'unicode-line', 'unicode-separators'}
    medium = payload == 'medium-file'
    required = ("VALUE = '" + '界' * 11_000 + "'\n" if unicode else 'VALUE = 1\n' + ''.join(
        f'# continuation payload {index:04}: ' + 'x' * 90 + '\n' for index in range(540)))
    if payload == 'unicode-separators':
        # Git counts LF lines. Unicode separators and source text resembling a
        # hunk header remain bytes within this one changed source line.
        required = "VALUE = '" + ('界\u2028@@ -900,1 +900,1 @@\u0085' * 1200) + "'\n"
    before = 'VALUE = 0\n'
    if medium:
        before = "def hello():\n    return 'world'\n"
        required = "def hello():\n    return 'universe'\n" + ''.join(
            f'# changed payload {index:04}: ' + 'x' * 90 + '\n' for index in range(140))
    repo = tmp_path / payload
    seed_feature_branch(repo, base={'api.py': before}, feature={'api.py': required})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    if native:
        review.backend = NativeStageBackend(repo)
    review.backend.sandbox = sandbox
    canonical = git_ops.diff(repo, 'main')
    expected_body = canonical.split('\n@@', 1)[1].split('\n', 1)[1].encode()
    if medium:
        assert 12 * 1024 < len(canonical.encode()) < 24 * 1024
    fragments: list[bytes] = []
    assignments: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        assert stage['stage'] == 'first_pass' and stage['assigned_files'] == ['api.py']
        prompt = review.backend.calls[-1]['prompt']
        if sandbox:
            rendered = prompt[prompt.index('Sanctioned phase inputs (captured verbatim):'):]
            assert len(rendered.encode()) <= 64 * 1024
            assert str(repo / '.daydream') not in prompt
            assert 'Sanctioned phase inputs (read only these exact files):' not in prompt
        diff, index = capture(review, stage)
        assert list(index) == ['api.py']
        assert index['api.py']['assignments'] == stage['assignment_parts']
        notice = '[Ordered continuation: this is a partial hunk; see assignment mapping.]\n'
        delivered = ([notice + piece for piece in diff.split(notice)[1:]]
                     if stage['assignment_parts'][0]['kind'] == 'continuation' else [diff])
        for part, part_diff, target in zip(stage['assignment_parts'], delivered, output['targets'], strict=True):
            assert part['kind'] == ('file' if medium and not sandbox else 'continuation')
            if not medium:
                assert part['part_count'] > 1
                assert (part['old_start'], part['old_count'], part['new_start'], part['new_count']) == (
                    1, 1, 1, 1 if unicode else 541)
            assignments.append(part)
            assert target['target_id'] == part['target_id']
            body = part_diff.split('\n@@', 1)[1].split('\n', 1)[1].encode()
            first = not fragments
            source_side = 'old' if first else 'new'
            reference = {
                'artifact_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
                'part_sha256': hashlib.sha256(part_diff.encode()).hexdigest(),
                'revision': review.pr.base_sha if first else review.pr.head_sha,
                'path': 'api.py', 'side': source_side,
                'source_range': {'start': 1, 'end': before.count('\n')} if part['kind'] == 'file' else None,
                'fragment': None if part['kind'] == 'file' else {
                    'offset': sum(map(len, fragments)), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
                },
                'basis': 'supplied_diff', 'role': 'assigned',
            }
            assert part['source_refs'][:1] == [reference]
            if native:
                # Native stage output contains no canonical metadata. The callback
                # must preserve this exact UTF-8 fragment and its null LF range.
                assert target['evidence_refs'] == [{
                    'source_id': part['source_options'][0]['source_id'],
                    'basis': 'supplied_diff', 'source_range': None,
                }]
            else:
                assert target['evidence_refs'] == [reference]
            if medium and not sandbox:
                assert part_diff == canonical
                assert '<sanctioned-input label="intent">' in prompt
                assert 'The PR updates greetings across stacks.' in prompt
                output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            else:
                assert part['fragment_offset'] == sum(map(len, fragments))
                assert part['fragment_bytes'] == len(body)
                assert reference['source_range'] is None
                assert reference['fragment'] == {'offset': part['fragment_offset'], 'bytes': len(body),
                                                 'sha256': hashlib.sha256(body).hexdigest()}
                assert 'never a source line number' in part['fragment_offset_basis']
            fragments.append(body)
            if omit_final and part['part_index'] == part['part_count']:
                target.update(status='not_reviewed', reason='Final continuation remains unexamined.')
            decisions.append({**copy.deepcopy(target), 'evidence_refs': [reference]})

    data = await review.finish('python', reason='evidence_incomplete' if omit_final else None,
                               findings=('Grounded defect',) if medium and not sandbox else ())
    assert scopes(data)['python']['files'] == ['api.py']
    assert scopes(data)['python']['target_decisions'] == sorted(decisions, key=lambda item: item['target_id'])
    if medium:
        assert len(assignments) == (2 if sandbox else 1)
    else:
        assert len(assignments) > 1
    assert [part['part_index'] for part in assignments] == list(range(1, len(assignments) + 1))
    assert b''.join(fragments) == expected_body
    identities = {
        ('medium-file', False): 'c3cb237aa7d541ca2ba90f11a24b431b59aa9e206a0b3604ac7f700bf622d0ef',
        ('medium-file', True): '8c08383148792020a99f0cca9bb8864f17dfa03d92709732d3594e54370801c1',
        ('oversized-hunk', False): 'ca39014e9198a8b6f4cfc4988b5e40262181d89c91aa470bb77b70e1b88f9796',
        ('oversized-hunk', True): 'ce7a626e36f3ab5b13b1f536d3e8b82c4b4833715a6063c83a23fe731dcedfbf',
        ('unicode-line', False): '059e5e7459c7cf1dd2402609c78678ea16bb5a3a0fe6eaa4752efa6a5f0b6c1d',
        ('unicode-separators', True): '3fc0493b93126e84667c65ef6fa19a12d250094cd7ff53ed18ea10b43b84f902',
    }
    assert part_identity(assignments) == identities[payload, sandbox]
    if sandbox:
        old_calls = {'medium-file': 2, 'oversized-hunk': 8, 'unicode-separators': 4}
        assert len(stage_ends(review, 'python')) < old_calls[payload]
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))
    if unicode:
        assert len(assignments) >= 3 and assignments[-1]['segment_count'] == len(assignments)
        assert any(part['fragment_line_offset'] > 0 for part in assignments)
        new_summaries = [part['source_summary'][0]['new'] for part in assignments
                         if part['source_summary'][0]['new'] is not None]
        assert len(new_summaries) >= 2
        assert all(summary['start'] == summary['end'] == 1 for summary in new_summaries)
        assert new_summaries[0]['start_byte'] == 0 and not new_summaries[0]['partial_start']
        assert new_summaries[-1]['end_byte'] == len(required.encode()) and not new_summaries[-1]['partial_end']
        assert all(left['end_byte'] == right['start_byte'] and left['partial_end'] and right['partial_start']
                   for left, right in zip(new_summaries, new_summaries[1:]))
        assert assignments[-1]['source_summary'][0]['old'] is None
        assert any(summary['partial_start'] for summary in new_summaries)
        full_index = json.loads((repo / '.daydream/hunk-index.json').read_text())['api.py']
        assert full_index['added_total'] == full_index['removed_total'] == 1
        assert len(full_index['hunks']) == 1


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
            assert 'supporting_catalog' not in stage
            assert 'supporting_catalog' not in json.loads(stage['access_guide'])
            assert {'diff', 'hunk-index'} <= _sanctioned_inputs(prompt).keys()

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

    # Exercise multiple bounded invocations within the same logical reviewer.
    await review.finish('python', findings=('Grounded defect',), deep_shard_enabled=False)
    assert len((repo / '.daydream/diff.patch').read_bytes()) >= 156_000
    assert len((repo / '.daydream/hunk-index.json').read_bytes()) >= 32_000
    assert len(exposed) > 4
    assert all(event['metadata']['admitted'] for event in stage_ends(review, 'python'))


@pytest.mark.parametrize('path', ['component_雪.py', 'nested/component_"quote".py', 'component_\t.py', 'api[1].py'])
async def test_quoted_git_paths_keep_complete_required_stage_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str) -> None:

    repo = tmp_path / 'quoted_paths'
    seed_feature_branch(repo, base={path: '# Stable source context\nVALUE = 0\n'},
                        feature={path: "# Stable source context\nVALUE = 'QUOTED_REQUIRED_CHANGE'\n"})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    required: list[str] = []
    submitted: list[dict[str, Any]] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        assert stage['assigned_files'] == [path]
        diff, _ = capture(review, stage)
        assert 'QUOTED_REQUIRED_CHANGE' in diff
        binding = json.loads(supporting_contents(review.backend.calls[-1]['prompt'])['input-binding'])
        digest = hashlib.sha256(diff.encode()).hexdigest()
        assert digest == binding['canonical_inputs']['canonical-diff']
        # Construct provider citations independently of the new offered-ref
        # metadata so this same request also exercises the pre-change owner.
        references = [{
            'artifact_sha256': digest, 'part_sha256': digest,
            'revision': binding['analyzed_revision'][revision_key], 'path': path, 'side': side,
            'source_range': {'start': 2, 'end': 2}, 'fragment': None,
            'basis': 'supplied_diff', 'role': 'assigned',
        } for side, revision_key in [('old', 'merge_base_sha'), ('new', 'head_sha')]]
        output['targets'][0]['evidence_refs'] = references
        submitted.extend(copy.deepcopy(references))
        required.append(diff)

    data = await review.finish('python')
    assert scopes(data)['python']['files'] == [path]
    decision, = scopes(data)['python']['target_decisions']
    assert decision['target_id'] == path and decision['status'] == 'reviewed'
    assert decision['evidence_refs'] == submitted
    assert len(required) == 1


async def test_failed_canonical_stage_input_preserves_successful_sibling_review(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A real canonical read failure must isolate its scope without hiding the successful sibling's evidence."""
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
        output['candidates'] = [candidate(disposition='confirmed', finding=finding, file='App.tsx', line=1,
                                         grounds=finding['evidence'])]

    monkeypatch.setattr(Path, 'read_text', read)
    await review.finish('python', reason='malformed_artifact', statuses=('failed',),
                               findings=('Surviving sibling defect',))
    assert canonical_reads >= 2
    assert 'PRIVATE_DIAGNOSTIC_PATH_SENTINEL' not in capsys.readouterr().out


async def test_changed_prepared_input_identity_blocks_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed payload coverage missed an otherwise valid native submission after its prepared file changed."""
    review = staged_review(tmp_path, monkeypatch, files=1)
    review.backend = NativeStageBackend(review.repo)
    bindings: list[Any] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        pointers = _sanctioned_inputs(review.backend.calls[-1]['prompt'])
        diff = pointers['review-assignment'] if 'review-assignment' in pointers else pointers['diff']
        diff.write_text('Changed after the prepared exact-path input was read.\n')
        bindings.append(review.backend.calls[-1]['host_tools'])

    await review.finish('python', reason='evidence_incomplete', statuses=('failed',))
    assert len(bindings) == 1
    fatal = bindings[0].fatal_error
    assert fatal is not None and fatal.code == 'input_unavailable' and fatal.preacceptance
    assert type(fatal.__cause__).__name__ == 'SanctionedInputUnavailable'
    assert 'review-assignment' in str(fatal) and 'Changing the submission cannot resolve' in str(fatal)
    event, = stage_ends(review, 'python')
    assert event['metadata']['failure_class'] == 'admission_failure'
    assert event['metadata']['admitted'] is False and event['metadata']['attempt'] == 1
    assert event['metadata']['observed_tool_starts'] == 1


@pytest.mark.parametrize('sandbox', [False, True], ids=['exact-whole-context', 'inline-context-unavailable'])
async def test_shared_intent_uses_actual_transport_allowance_after_required_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool) -> None:

    review = staged_review(tmp_path, monkeypatch, files=2)
    shared_intent = 'The greeting API promises world and this change must preserve callers.\n' * 1000
    assert 64 * 1024 < len(shared_intent.encode()) < 72_000

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


async def test_mixed_output_population_keeps_native_authority_final_turns_and_scope_local_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent output streams cannot borrow a valid package or retry state from siblings."""
    cases = {
        "unparseable": ("malformed_output", 1),
        "invalid-native-valid-text": ("malformed_output", 1),
        "nested-invalid-root": (None, 1),
        "incomplete-nested-stage": ("malformed_output", 1),
        "text-origin-nested": (None, 1),
        "empty-final-turn": ("missing_output", 1),
        "native-prose": (None, 1),
        "pi-schema-retry": (None, 2),
        "missing-notes": (None, 2),
        "empty-object": (None, 2),
        "scalar-only-object": (None, 2),
        "wrong-type-notes": (None, 2),
        "native-missing-notes": ("malformed_output", 1),
        "compliant-json": (None, 1),
        "prose-json": (None, 1),
        "fenced-json": (None, 1),
        "incidental-json": (None, 1),
        "ambiguous-json": ("malformed_output", 1),
        "wrapper-siblings": ("malformed_output", 1),
        "result-list": ("malformed_output", 1),
        "recursive-wrapper": ("malformed_output", 1),
        "partial-second-object": ("malformed_output", 1),
        "partial-second-array": ("malformed_output", 1),
        "malformed-tail": ("malformed_output", 1),
        "unfinished-fence": ("malformed_output", 1),
        "orphan-close": ("malformed_output", 1),
        "unmatched-prose": ("malformed_output", 1),
        "oversized-final": ("malformed_output", 1),
        "output-shaped-invalid": ("malformed_output", 1),
        "output-shaped-before-valid": ("malformed_output", 1),
        "misplaced-closing-brace": ("malformed_output", 1),
    }
    repo = tmp_path / "output_population"
    before = {f"{case}/api.py": "def hello():\n    return 'world'\n" for case in cases}
    seed_feature_branch(
        repo, base=before, feature={path: body.replace("world", "universe") for path, body in before.items()}
    )
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    scope_cases: dict[str, str] = {}

    def stream(stage: dict[str, Any], valid: dict[str, Any]) -> Iterable[AgentEvent]:
        (path,) = stage["assigned_files"]
        case = path.split("/")[0]
        scope_id = stage["scope_id"]
        scope_cases[scope_id] = case
        prompt = review.backend.calls[-1]['prompt']
        attempt = sum(item["scope_id"] == scope_id for item in review.backend.stages)
        assert stage['attempt'] == attempt
        valid["candidates"] = (
            []
            if case in {"compliant-json", "misplaced-closing-brace"}
            else [
                candidate(
                    disposition="confirmed",
                    file=path,
                    line=2,
                    finding=dict(record(), file=path, description="Accepted " + case, evidence=f"{path}:2"),
                )
            ]
        )
        invalid = {'unknown_outer_key': valid}
        if case in {'empty-object', 'scalar-only-object'}:
            assert stage['notes'] == stage['candidates'] == []
            text = json.dumps(({} if case == 'empty-object' else {'incidental': 1}) if attempt == 1 else valid)
            yield TextEvent(text=text)
            yield ResultEvent(structured_output=None, continuation=None)
            return
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
            if case != "native-missing-notes":
                yield TextEvent(text=json.dumps(valid))
            yield ResultEvent(structured_output=valid if case == "native-missing-notes" else None, continuation=None)
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
            'fenced-json': 'Done.\n```json\n' + json.dumps(valid) + '\n```\nFinished.',
            'incidental-json': 'Inspected metadata["sender"]["login"] and ' + json.dumps(['large' * 100] * 10)
                              + ' then ' + json.dumps(valid),
            'ambiguous-json': json.dumps(valid) + '\nExample: ' + json.dumps({'result': valid}),
            'wrapper-siblings': json.dumps({'result': valid, 'comment': 'finished'}),
            'result-list': json.dumps([valid]),
            'recursive-wrapper': json.dumps({'outer': {'inner': valid}}),
            'partial-second-object': json.dumps(valid) + '\n{',
            'partial-second-array': json.dumps(valid) + '\n[',
            'malformed-tail': json.dumps(valid) + '\n{broken}',
            'unfinished-fence': '```json\n' + json.dumps(valid),
            'orphan-close': json.dumps(valid) + '\n]',
            'unmatched-prose': 'Here is { the result ' + json.dumps(valid),
            'oversized-final': json.dumps(valid) + 'x' * (128 * 1024),
            'output-shaped-invalid': json.dumps({'targets': 'invalid', 'result': valid}),
            'output-shaped-before-valid': json.dumps({'targets': 'invalid', 'result': valid}) + json.dumps(valid),
            'prose-json': 'Here is the result:\n' + json.dumps(valid) + '\nInvestigation finished.',
            'misplaced-closing-brace': json.dumps({'targets': valid['targets'], 'notes': 'Reviewed real source'})
                                     + ', "candidates": [], "contradictions": []}',
        }.get(case, json.dumps(valid))
        yield TextEvent(text=text)
        if final_turn:
            yield TurnEndEvent()
        if case == 'empty-final-turn':
            yield TextEvent(text='')
            yield TurnEndEvent()
        native = invalid if case == 'invalid-native-valid-text' else valid if case == 'native-prose' else None
        yield ResultEvent(structured_output=native, continuation=None)

    review.backend.scripts = {f"python#{index}": (stream, False) for index in range(len(cases))}
    assert (
        await review.run(
            deep_shard_enabled=True, deep_shard_max_files=1, deep_shard_max_bytes=10**9, deep_shard_fanout_cap=64
        )
        == 0
    )
    data = review.load()
    assert len(scope_cases) == len(cases) and set(scope_cases.values()) == set(cases)
    expected_titles = {
        "Accepted " + case for case, (reason, _) in cases.items() if reason is None and case != "compliant-json"
    }
    assert len(data["findings"]) == len(expected_titles)
    assert {finding["title"] for finding in data["findings"]} == expected_titles
    outcomes = scopes(data)
    assert outcomes["structure"]["status"] == "complete"
    for scope_id, case in scope_cases.items():
        reason, attempts = cases[case]
        outcome = outcomes[scope_id]
        assert outcome["files"] == [f"{case}/api.py"]
        assert outcome["status"] == ("incomplete" if reason else "complete")
        assert outcome["reason_codes"] == ([reason] if reason else [])
        assert len(outcome["target_decisions"]) == (0 if reason else 1)
        if not reason:
            (decision,) = outcome["target_decisions"]
            assert decision["target_id"] == f"{case}/api.py" and decision["status"] == "reviewed"
            assert decision["evidence_refs"]
        events = stage_ends(review, scope_id)
        assert len(events) == attempts
        assert [event["metadata"]["attempt"] for event in events] == list(range(1, attempts + 1))
        assert [event["metadata"]["admitted"] for event in events] == (
            [False, True] if attempts == 2 else [reason is None]
        )
        metadata = events[-1]["metadata"]
        assert metadata["observed_tool_starts"] == 0 and metadata["remaining_tool_calls"] == 48
        if attempts == 2:
            assert events[0]["metadata"]["schema_rejection"]["error_count"] > 0
        elif case in {
            "fenced-json",
            "incidental-json",
            "ambiguous-json",
            "wrapper-siblings",
            "result-list",
            "recursive-wrapper",
            "partial-second-object",
            "partial-second-array",
            "malformed-tail",
            "unfinished-fence",
            "orphan-close",
            "unmatched-prose",
            "oversized-final",
            "unparseable",
            "incomplete-nested-stage",
            "empty-final-turn",
            "compliant-json",
            "misplaced-closing-brace",
        }:
            assert metadata["schema_rejection"] is None
        if case in {"compliant-json", "misplaced-closing-brace"}:
            assert metadata["failure_class"] == ("syntax_failure" if reason else None)
        saved = json.loads((repo / f".daydream/deep/stack-{scope_id}-records.json").read_text())
        assert bool(saved.get("incomplete")) is bool(reason)
        assert [item["description"] for item in saved["issues"]] == (
            ["Accepted " + case] if reason is None and case != "compliant-json" else []
        )


@pytest.mark.parametrize('packaging', ['prose', 'wrapper', 'native', 'domain-invalid'])
async def test_staged_results_present_only_after_admission_and_preserve_raw_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], packaging: str,
) -> None:
    review = staged_review(tmp_path, monkeypatch, files=2)
    marker = 'HOST_PRESENTATION_MARKER'
    sentinel = 'ghp_' + 'x' * 16
    raw_text: list[str] = []

    @review.backend.script(terminal=False)
    def stream(stage: dict[str, Any], valid: dict[str, Any]) -> Iterable[AgentEvent]:
        valid['notes'] = marker + ' ' + sentinel
        valid['candidates'] = [candidate(disposition='confirmed', finding=record())]
        if packaging == 'domain-invalid':
            valid['candidates'][0]['grounds'] = ''
        if packaging == 'native':
            yield ResultEvent(structured_output=valid, continuation=None)
        else:
            text = (json.dumps({'result': valid}) if packaging == 'wrapper' else
                    'Here is the completed investigation:\n' + json.dumps(valid))
            raw_text.append(text)
            yield TextEvent(text=text)
            yield TurnEndEvent()
            yield ResultEvent(structured_output=None, continuation=None)

    rejected = packaging == 'domain-invalid'
    await review.finish('python', reason='evidence_incomplete' if rejected else None,
                        findings=() if rejected else ('Grounded defect',), log_mode=True)
    output = capsys.readouterr().out
    assert sentinel not in output
    lines = [line for line in output.splitlines() if line.startswith('[result]')]
    assert sum(marker in line for line in lines) == (0 if rejected else 1)
    if not rejected:
        assert any(marker in line and '[REDACTED_API_KEY]' in line for line in lines)
    if raw_text:
        trajectories = [json.loads(path.read_text()) for path in (review.repo / '.daydream').rglob('*.json')
                        if path.name == 'trajectory.json' or '--' in path.name]
        messages = [step.get('message') for document in trajectories for step in document.get('steps', [])]
        assert raw_text[0].replace(sentinel, '[REDACTED_API_KEY]') in messages


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
    # Assignment handles must remain unique across batches of one logical scope.
    data = await review.finish('generic', deep_shard_enabled=False)
    assert set(scopes(data)['generic']['files']) == {'00_oversize.md', 'part:000001'}
    assert 'part:000001' in targets and len(targets) > 2
    combined = '\n'.join(diffs)
    for index in range(600):
        assert combined.count(f'REQUIRED_LARGE_CHANGE_{index:04}') == 1


async def test_sharded_admission_rejects_each_bad_sibling_atomically_and_retains_earlier_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent rejected scopes cannot erase each other's frozen, previously admitted work."""
    source_faults = (
        "revision",
        "artifact-digest",
        "part-digest",
        "path",
        "side",
        "range",
        "fragment",
        "supporting-only",
        "missing-assigned",
        "reference-count",
        "supporting-digest",
        "supporting-missing-path",
        "supporting-range",
        "supporting-reversed-range",
    )
    mixed_faults = (
        "unknown-target",
        "unknown-candidate",
        "contradiction",
        "grounds",
        "grounds-missing",
        "grounds-null",
        "grounds-number",
        "handoff",
    )
    faults = (*source_faults, *(f"mixed-{fault}" for fault in mixed_faults), "healthy")
    before = {f"{fault}/module_{index:02}.py": "VALUE = 0\n" for fault in faults for index in range(8)}
    repo = tmp_path / "independent_admission"
    seed_feature_branch(repo, base=before, feature={name: "VALUE = 1\n" for name in before})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    admitted: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, list[str]] = {}
    scope_faults: dict[str, str] = {}

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        scope_id = stage["scope_id"]
        if scope_id == "structure":
            return
        fault = stage["assigned_files"][0].split("/", 1)[0]
        assert all(path.startswith(fault + "/") for path in stage["assigned_files"])
        scope_faults[scope_id] = fault
        if not stage["progress"]:
            admitted[scope_id] = copy.deepcopy(output["targets"])
            path = stage['assigned_files'][0]
            finding = dict(
                record(line=1),
                file=path,
                description="defect_" + fault.replace("-", "_"),
                evidence=f"{path}:1 changes VALUE from 0 to 1",
            )
            output["candidates"] = [
                candidate(disposition="confirmed", finding=finding, file=path, line=1, grounds=finding["evidence"])
            ]
            return
        if fault == "healthy":
            return
        rejected[scope_id] = stage["assigned_target_ids"]
        output["notes"] = "REJECTED_STAGE_POISON"
        if fault.startswith("mixed-"):
            fault = fault.removeprefix("mixed-")
            output["unknown_extra"] = None
            if fault == "unknown-target":
                output["targets"][0]["target_id"] = "foreign-assignment"
            elif fault == "contradiction":
                output["contradictions"] = [stage["closed_candidate_ids"][0]]
            elif fault == "handoff":
                output["notes"] += "x" * 70_000
            else:
                path = stage["assigned_files"][0]
                finding = dict(record(line=1), file=path, evidence=f"{path}:1 changes VALUE from 0 to 1")
                item = candidate(
                    disposition="confirmed", finding=finding, file=path, line=1, grounds=finding["evidence"]
                )
                if fault == "unknown-candidate":
                    item["candidate_id"] = "foreign-candidate"
                elif fault == "grounds-missing":
                    item.pop("grounds")
                else:
                    item["grounds"] = {"grounds-null": None, "grounds-number": 7}.get(fault, "")
                output["candidates"] = [item]
            return
        reference = output['targets'][0]['evidence_refs'][0]
        if fault in {"revision", "artifact-digest", "part-digest"}:
            key = {"revision": "revision", "artifact-digest": "artifact_sha256", "part-digest": "part_sha256"}[fault]
            reference[key] = "0" * (40 if fault == "revision" else 64)
        elif fault == "path":
            reference['path'] = 'unassigned.py'
        elif fault == "side":
            reference['side'] = 'diff'
        elif fault == "range":
            reference['source_range'] = {'start': 999999, 'end': 999999}
        elif fault == "fragment":
            reference['fragment'] = {'offset': 999999, 'bytes': 1, 'sha256': '0' * 64}
        elif fault == "supporting-only":
            reference['role'] = 'supporting'
        elif fault == "reference-count":
            output['targets'][0]['evidence_refs'] *= 5
        elif fault.startswith("supporting-"):
            path = f"{fault}/module_06.py"
            supporting = {
                "artifact_sha256": hashlib.sha256((repo / path).read_bytes()).hexdigest(),
                "part_sha256": None,
                "revision": stage["analyzed_revision"]["head_sha"],
                "path": path,
                "side": "new",
                "source_range": {"start": 1, "end": 1},
                "fragment": None,
                "basis": "source_inspection",
                "role": "supporting",
            }
            if fault == "supporting-digest":
                supporting['artifact_sha256'] = '0' * 64
            elif fault == "supporting-missing-path":
                supporting['path'] = 'missing-frozen.py'
            else:
                supporting["source_range"] = {"start": 2, "end": 1 if fault == "supporting-reversed-range" else 2}
            output['targets'][0]['evidence_refs'].append(supporting)
        else:
            output['targets'][0]['evidence_refs'] = []

    review.backend.stage_response = response
    assert (
        await review.run(
            deep_shard_enabled=True, deep_shard_max_files=8, deep_shard_max_bytes=10**9, deep_shard_fanout_cap=32
        )
        == 0
    )
    data = review.load()
    assert set(scope_faults.values()) == set(faults) and len(scope_faults) == len(faults)
    assert len(data["findings"]) == len(faults)
    assert {finding["title"] for finding in data["findings"]} == {
        "defect_" + fault.replace("-", "_") for fault in faults
    }
    outcomes = scopes(data)
    assert outcomes["structure"]["status"] == "complete"
    for scope_id, fault in scope_faults.items():
        outcome = outcomes[scope_id]
        assert outcome["files"] == [f"{fault}/module_{index:02}.py" for index in range(8)]
        events = stage_ends(review, scope_id)
        assert len(events) == 2 and [event["metadata"]["attempt"] for event in events] == [1, 1]
        if fault == "healthy":
            assert outcome["status"] == "complete" and outcome["reason_codes"] == []
            assert len(outcome["target_decisions"]) == 8
            assert all(event["metadata"]["admitted"] for event in events)
            continue
        malformed = fault.startswith("mixed-") or fault == "reference-count"
        assert outcome["status"] == "incomplete"
        assert outcome["reason_codes"] == ["malformed_output" if malformed else "evidence_incomplete"]
        assert outcome["target_decisions"] == sorted(admitted[scope_id], key=lambda item: item["target_id"])
        assert not set(rejected[scope_id]).intersection(item["target_id"] for item in outcome["target_decisions"])
        assert [event["metadata"]["admitted"] for event in events] == [True, False]
        last = events[-1]["metadata"]
        # The full captured patch crosses the run-wide scale threshold, so
        # both stages keep the doubled 96-call allowance without tool starts.
        assert events[0]["metadata"]["remaining_tool_calls"] == last["remaining_tool_calls"] == 96
        assert last["observed_tool_starts"] == 0
        assert bool(last.get("schema_rejection")) is malformed
        if not fault.startswith("mixed-"):
            assert last["failure_class"] == "source_attribution"
    saved = json.loads((repo / ".daydream/deep/review-coverage.json").read_text())
    assert saved["stack_outcomes"] == data["terminal_result"]["stack_outcomes"]
    assert all("REJECTED_STAGE_POISON" not in call["prompt"] for call in review.backend.calls)


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
    decisions: list[dict[str, Any]] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        diff, index = capture(review, stage)
        scoped = index['api.py']
        fragments.append(diff)
        decisions.extend(copy.deepcopy(output['targets']))
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


@pytest.mark.parametrize('fault', ['reordered-fragment', 'byte-offset-as-line'])
async def test_continuation_references_cannot_alias_another_fragment_or_a_source_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Byte offsets identify a specific delivered fragment, even when it splits one source line."""
    repo = tmp_path / 'continuation_attribution'
    # Oversized before and after lines leave a real admitted stage before the
    # second fragment, even when nearby small continuations share a batch.
    seed_feature_branch(repo, base={'api.py': "VALUE = '" + '古' * 11000 + "'\n"},
                        feature={'api.py': "VALUE = '" + '界' * 11000 + "'\n"})
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    admitted: list[dict[str, Any]] = []
    injected: list[str] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        capture(review, stage)
        if not admitted:
            assert [part['part_index'] for part in stage['assignment_parts']] == [1]
            admitted.extend(copy.deepcopy(output['targets']))
            output['candidates'] = [candidate(disposition='confirmed', finding=record(line=1))]
            return
        target = next(target for part, target in zip(stage['assignment_parts'], output['targets'], strict=True)
                      if part['part_index'] == 2)
        reference, = target['evidence_refs']
        assert reference['source_range'] is None
        injected.append(target['target_id'])
        assert reference['fragment']['offset'] > 1
        assert reference['part_sha256'] != admitted[0]['evidence_refs'][0]['part_sha256']
        if fault == 'reordered-fragment':
            target['evidence_refs'] = copy.deepcopy(admitted[0]['evidence_refs'])
        else:
            reference['source_range'] = {'start': reference['fragment']['offset'],
                                         'end': reference['fragment']['offset']}
            reference['fragment'] = None

    data = await review.finish('python', reason='evidence_incomplete', findings=('Grounded defect',))
    assert len(injected) == 1 and injected[0] not in {target['target_id'] for target in admitted}
    assert scopes(data)['python']['target_decisions'] == admitted
    events = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in events] == [True, False]
    assert events[-1]['metadata']['failure_class'] == 'source_attribution'


async def test_redaction_expansion_rejects_stage_without_losing_admitted_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret-heavy reason must not break publication of earlier accepted work."""
    review = staged_review(tmp_path, monkeypatch)
    admitted: list[dict[str, Any]] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if not stage['progress']:
            admitted.extend(copy.deepcopy(output['targets']))
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        else:
            output['targets'][0]['reason'] = 'TOKEN=x ' * 20000

    data = await review.finish('python', reason='evidence_incomplete', findings=('Grounded defect',))
    assert scopes(data)['python']['target_decisions'] == admitted
    events = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in events] == [True, False]
    assert events[-1]['metadata']['failure_class'] == 'source_attribution'
    assert 'TOKEN=x' not in review.output.read_text()
    saved = json.loads((review.repo / '.daydream/deep/stack-python-records.json').read_text())
    assert [finding['description'] for finding in saved['issues']] == ['Grounded defect']


async def test_parallel_scope_attribution_overflow_retains_findings_and_admitted_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Individually valid scopes cannot crowd accepted findings out of the bounded handoff."""
    from daydream.findings import MAX_ARTIFACT_BYTES

    monkeypatch.setattr('daydream.review_result.RUN_TARGET_DECISIONS_MAX_BYTES', 256 * 1024)
    review = staged_review(tmp_path, monkeypatch, files=20)
    review.config = replace(review.config, deep_shard_enabled=True, deep_shard_max_files=1,
                            deep_shard_max_bytes=10**9, deep_shard_fanout_cap=32)
    submitted: dict[str, list[dict[str, Any]]] = {}

    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if stage['scope_id'] == 'structure':
            return
        if 'api.py' in stage['assigned_files']:
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
        else:
            output['targets'][0]['reason'] = 'r' * 60000
        submitted[stage['scope_id']] = copy.deepcopy(output['targets'])

    review.backend.stage_response = response
    assert await review.run() == 0
    data = review.load()
    assert review.output.stat().st_size <= MAX_ARTIFACT_BYTES
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    assert data['terminal_result']['analysis_state'] == 'incomplete'
    outcomes = scopes(data)
    assert len(submitted) == 20, 'the fixture must exercise independent concurrent reviewer scopes'
    assert any(scope['reason_codes'] == ['evidence_incomplete'] for scope in outcomes.values())
    for scope_id, decisions in submitted.items():
        outcome = outcomes[scope_id]
        assert outcome['target_decisions'] == (decisions if outcome['status'] == 'complete' else [])
    coverage = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    assert coverage['stack_outcomes'] == data['terminal_result']['stack_outcomes']
