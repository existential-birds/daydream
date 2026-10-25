"""Large staged-review transport and retry scenarios through the production runner."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import replace
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
from daydream.backends.pi import PiBackend
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
from tests.harness.git_helpers import seed_feature_branch
from tests.harness.table_cases import isolated_case
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


async def _case_structure_keeps_bounded_interaction_review_and_patch_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str, ext_dir: ExtDir
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
        native = PiBackend(model=review.backend.model)

        async def execute(self: PiBackend, cwd: Path, prompt: str, *args: Any,
                          **kwargs: Any) -> AsyncIterator[AgentEvent]:
            kwargs.pop('wall_budget_s', None)
            kwargs.pop('tool_call_budget', None)
            async for event in review.backend.execute(cwd, prompt, *args, **kwargs):
                yield event

        monkeypatch.setattr(PiBackend, 'execute', execute)
        monkeypatch.setattr('daydream.runner.create_backend', lambda *_a, **_k: native)
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


async def test_structure_keeps_bounded_interaction_review_and_patch_navigation(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[str] = ["files", "bytes", "all-omitted", "navigation"]
    case_index = 0
    for limit in cases_0:
        with subtests.test(limit=limit), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            await _case_structure_keeps_bounded_interaction_review_and_patch_navigation(
                tmp_path=scenario.tmp_path,
                monkeypatch=scenario.monkeypatch,
                limit=limit,
                ext_dir=ExtDir(scenario.tmp_path, scenario.monkeypatch),
            )


async def _case_structure_can_attribute_a_later_assigned_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_digest: bool, sandbox: bool
) -> None:
    """Bounded examples must not exclude genuine later Structure work from admission."""
    review = staged_review(tmp_path, monkeypatch)
    review.backend.sandbox = sandbox
    submitted: list[dict[str, Any]] = []

    @review.backend.script('structure')
    def respond(stage: dict[str, Any], output: dict[str, Any]) -> None:
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
        reference = next(copy.deepcopy(ref) for ref in part['source_refs'] if ref['side'] == 'new')
        assert reference['path'] == 'module_06.py'
        assert reference['revision'] == stage['analyzed_revision']['head_sha']
        assert reference['part_sha256'] == hashlib.sha256(source).hexdigest()
        assert reference['source_range'] == {'start': 1, 'end': 1} and reference['fragment'] is None
        if invalid_digest:
            reference['part_sha256'] = '0' * 64
        output['targets'][0]['evidence_refs'] = [reference]
        submitted.extend(copy.deepcopy(output['targets']))

    data = await review.finish('structure', reason='evidence_incomplete' if invalid_digest else None)
    scope = scopes(data)['structure']
    assert scope['target_decisions'] == ([] if invalid_digest else submitted)
    event, = stage_ends(review, 'structure')
    assert event['metadata']['admitted'] is not invalid_digest
    assert event['metadata']['failure_class'] == ('source_attribution' if invalid_digest else None)


async def test_structure_can_attribute_a_later_assigned_part(tmp_path: Path, subtests: pytest.Subtests) -> None:
    cases_0: Sequence[bool] = [False, True]
    cases_1: Sequence[bool] = [False, True]
    case_index = 0
    for invalid_digest in cases_0:
        for sandbox in cases_1:
            with (
                subtests.test(invalid_digest=invalid_digest, sandbox=sandbox),
                isolated_case(tmp_path, str(case_index)) as scenario,
            ):
                case_index += 1
                await _case_structure_can_attribute_a_later_assigned_part(
                    tmp_path=scenario.tmp_path,
                    monkeypatch=scenario.monkeypatch,
                    invalid_digest=invalid_digest,
                    sandbox=sandbox,
                )


async def _case_schema_rejection_retries_once_with_fresh_grounding_and_charged_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, ext_dir: ExtDir
) -> None:
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


async def test_schema_rejection_retries_once_with_fresh_grounding_and_charged_starts(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[str] = ["success", "second-rejection"]
    case_index = 0
    for outcome in cases_0:
        with subtests.test(outcome=outcome), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            await _case_schema_rejection_retries_once_with_fresh_grounding_and_charged_starts(
                tmp_path=scenario.tmp_path,
                monkeypatch=scenario.monkeypatch,
                outcome=outcome,
                ext_dir=ExtDir(scenario.tmp_path, scenario.monkeypatch),
            )


async def _case_oversized_hunk_assignments_reassemble_every_canonical_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool, payload: str, omit_final: bool
) -> None:
    from daydream import git_ops
    from daydream.prompt_budget import SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES

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
            assert len(rendered.encode()) <= SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES
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
            assert target['evidence_refs'] == part['source_refs'][:1]
            body = part_diff.split('\n@@', 1)[1].split('\n', 1)[1].encode()
            if medium and not sandbox:
                assert part_diff == canonical
                assert '<sanctioned-input label="intent">' in prompt
                assert 'The PR updates greetings across stacks.' in prompt
                output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            else:
                assert part['fragment_offset'] == sum(map(len, fragments))
                assert part['fragment_bytes'] == len(body)
                reference, = target['evidence_refs']
                assert reference['source_range'] is None
                assert reference['fragment'] == {'offset': part['fragment_offset'], 'bytes': len(body),
                                                 'sha256': hashlib.sha256(body).hexdigest()}
                assert 'never a source line number' in part['fragment_offset_basis']
            fragments.append(body)
            if omit_final and part['part_index'] == part['part_count']:
                target.update(status='not_reviewed', reason='Final continuation remains unexamined.')
        decisions.extend(copy.deepcopy(output['targets']))

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


async def test_oversized_hunk_assignments_reassemble_every_canonical_byte(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[bool] = [False, True]
    cases_1: Sequence[tuple[str, bool]] = [
        ("medium-file", False),
        ("oversized-hunk", False),
        ("oversized-hunk", True),
        ("unicode-line", False),
        ("unicode-separators", False),
    ]
    case_index = 0
    for sandbox in cases_0:
        for payload, omit_final in cases_1:
            with (
                subtests.test(sandbox=sandbox, payload=payload, omit_final=omit_final),
                isolated_case(tmp_path, str(case_index)) as scenario,
            ):
                case_index += 1
                await _case_oversized_hunk_assignments_reassemble_every_canonical_byte(
                    tmp_path=scenario.tmp_path,
                    monkeypatch=scenario.monkeypatch,
                    sandbox=sandbox,
                    payload=payload,
                    omit_final=omit_final,
                )


async def _case_current_exact_builder_signatures_route_generic_and_rust_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: ExtDir, bundle_capable: bool
) -> None:
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


async def test_current_exact_builder_signatures_route_generic_and_rust_shards(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[bool] = [False, True]
    case_index = 0
    for bundle_capable in cases_0:
        with subtests.test(bundle_capable=bundle_capable), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            await _case_current_exact_builder_signatures_route_generic_and_rust_shards(
                tmp_path=scenario.tmp_path,
                monkeypatch=scenario.monkeypatch,
                ext_dir=ExtDir(scenario.tmp_path, scenario.monkeypatch),
                bundle_capable=bundle_capable,
            )


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


async def _case_quoted_git_paths_keep_complete_required_stage_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:

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


async def test_quoted_git_paths_keep_complete_required_stage_inputs(tmp_path: Path, subtests: pytest.Subtests) -> None:
    cases_0: Sequence[str] = [
        "api.py",
        "component_雪.py",
        'nested/component_"quote".py',
        "component_\t.py",
        "api[1].py",
    ]
    case_index = 0
    for path in cases_0:
        with subtests.test(path=path), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            await _case_quoted_git_paths_keep_complete_required_stage_inputs(
                tmp_path=scenario.tmp_path, monkeypatch=scenario.monkeypatch, path=path
            )


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
        output['candidates'] = [candidate(disposition='confirmed', finding=finding, file='App.tsx', line=1,
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


async def _case_shared_intent_uses_actual_transport_allowance_after_required_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool
) -> None:

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


async def test_shared_intent_uses_actual_transport_allowance_after_required_assignment(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[bool] = [False, True]
    case_index = 0
    for sandbox in cases_0:
        with subtests.test(sandbox=sandbox), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            await _case_shared_intent_uses_actual_transport_allowance_after_required_assignment(
                tmp_path=scenario.tmp_path, monkeypatch=scenario.monkeypatch, sandbox=sandbox
            )


async def _case_strict_stage_selection_never_salvages_rejected_or_incomplete_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, reason: str | None, attempts: int
) -> None:
    review = staged_review(tmp_path, monkeypatch, files=2)

    @review.backend.script(terminal=False)
    def stream(stage: dict[str, Any], valid: dict[str, Any]) -> Iterable[AgentEvent]:
        prompt = review.backend.calls[-1]['prompt']
        attempt = sum(item['scope_id'] == 'python' for item in review.backend.stages)
        assert stage['attempt'] == attempt
        valid['candidates'] = ([] if case in {'compliant-json', 'misplaced-closing-brace'} else
                               [candidate(disposition='confirmed', finding=record())])
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

    await review.finish('python', reason=reason,
                        findings=('Grounded defect',) if reason is None and case != 'compliant-json' else ())
    events = stage_ends(review, 'python')
    assert len(events) == attempts
    assert [event['metadata']['attempt'] for event in events] == list(range(1, attempts + 1))
    assert [event['metadata']['admitted'] for event in events] == (
        [False, True] if case in {'pi-schema-retry', 'missing-notes', 'wrong-type-notes',
                                 'empty-object', 'scalar-only-object'}
        else [reason is None] * attempts)
    if attempts == 2:
        assert events[0]['metadata']['schema_rejection']['error_count'] > 0
    elif case in {'fenced-json', 'incidental-json', 'ambiguous-json', 'wrapper-siblings', 'result-list',
                  'recursive-wrapper', 'partial-second-object', 'partial-second-array', 'malformed-tail',
                  'unfinished-fence', 'orphan-close', 'unmatched-prose', 'oversized-final', 'unparseable',
                  'incomplete-nested-stage', 'empty-final-turn', 'compliant-json', 'misplaced-closing-brace'}:
        assert events[0]['metadata']['schema_rejection'] is None
    metadata = events[-1]['metadata']
    assert metadata['observed_tool_starts'] == 0 and metadata['remaining_tool_calls'] == 48
    if case in {'compliant-json', 'misplaced-closing-brace'}:
        assert metadata['failure_class'] == ('syntax_failure' if case == 'misplaced-closing-brace' else None)


async def test_strict_stage_selection_never_salvages_rejected_or_incomplete_output(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[tuple[str, str | None, int]] = [
        ("unparseable", "malformed_output", 1),
        ("invalid-native-valid-text", "malformed_output", 1),
        ("nested-invalid-root", None, 1),
        ("incomplete-nested-stage", "malformed_output", 1),
        ("text-origin-nested", None, 1),
        ("empty-final-turn", "missing_output", 1),
        ("native-prose", None, 1),
        ("pi-schema-retry", None, 2),
        ("missing-notes", None, 2),
        ("empty-object", None, 2),
        ("scalar-only-object", None, 2),
        ("wrong-type-notes", None, 2),
        ("native-missing-notes", "malformed_output", 1),
        ("compliant-json", None, 1),
        ("prose-json", None, 1),
        ("fenced-json", None, 1),
        ("incidental-json", None, 1),
        ("ambiguous-json", "malformed_output", 1),
        ("wrapper-siblings", "malformed_output", 1),
        ("result-list", "malformed_output", 1),
        ("recursive-wrapper", "malformed_output", 1),
        ("partial-second-object", "malformed_output", 1),
        ("partial-second-array", "malformed_output", 1),
        ("malformed-tail", "malformed_output", 1),
        ("unfinished-fence", "malformed_output", 1),
        ("orphan-close", "malformed_output", 1),
        ("unmatched-prose", "malformed_output", 1),
        ("oversized-final", "malformed_output", 1),
        ("output-shaped-invalid", "malformed_output", 1),
        ("output-shaped-before-valid", "malformed_output", 1),
        ("misplaced-closing-brace", "malformed_output", 1),
    ]
    case_index = 0
    for case, reason, attempts in cases_0:
        with (
            subtests.test(case=case, reason=reason, attempts=attempts),
            isolated_case(tmp_path, str(case_index)) as scenario,
        ):
            case_index += 1
            await _case_strict_stage_selection_never_salvages_rejected_or_incomplete_output(
                tmp_path=scenario.tmp_path,
                monkeypatch=scenario.monkeypatch,
                case=case,
                reason=reason,
                attempts=attempts,
            )


async def _case_staged_results_present_only_after_admission_and_preserve_raw_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], packaging: str
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


async def test_staged_results_present_only_after_admission_and_preserve_raw_text(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[str] = ["prose", "wrapper", "native", "domain-invalid"]
    case_index = 0
    for packaging in cases_0:
        with subtests.test(packaging=packaging), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            capsys.readouterr()
            await _case_staged_results_present_only_after_admission_and_preserve_raw_text(
                tmp_path=scenario.tmp_path, monkeypatch=scenario.monkeypatch, capsys=capsys, packaging=packaging
            )


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


async def _case_mixed_schema_and_admission_rejection_never_retries_or_loses_prior_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    review = staged_review(tmp_path, monkeypatch)

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if not stage['progress']:
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            return
        if stage['attempt'] > 1:
            # A second valid answer would wrongly recover from mixed semantic rejection; production must not invoke it.
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
            item = candidate(disposition='confirmed', finding=finding, file=path, line=1, grounds=finding['evidence'])
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


async def test_mixed_schema_and_admission_rejection_never_retries_or_loses_prior_findings(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[str] = [
        "unknown-target",
        "unknown-candidate",
        "contradiction",
        "grounds",
        "grounds-missing",
        "grounds-null",
        "grounds-number",
        "handoff",
    ]
    case_index = 0
    for fault in cases_0:
        with subtests.test(fault=fault), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            await _case_mixed_schema_and_admission_rejection_never_retries_or_loses_prior_findings(
                tmp_path=scenario.tmp_path, monkeypatch=scenario.monkeypatch, fault=fault
            )


async def _case_invalid_source_reference_preserves_prior_admitted_target_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """A later well-shaped citation cannot alter admitted work unless frozen identity matches."""
    review = staged_review(tmp_path, monkeypatch)
    admitted: list[dict[str, Any]] = []
    rejected_ids: list[str] = []

    @review.backend.script('python')
    def response(stage: dict[str, Any], output: dict[str, Any]) -> None:
        if not stage['progress']:
            admitted.extend(copy.deepcopy(output['targets']))
            output['candidates'] = [candidate(disposition='confirmed', finding=record())]
            return
        rejected_ids.extend(stage['assigned_target_ids'])
        output['notes'] = 'INVALID_ATTRIBUTION_MUST_NOT_SURVIVE'
        reference = output['targets'][0]['evidence_refs'][0]
        if fault == 'revision':
            reference['revision'] = '0' * 40
        elif fault == 'artifact-digest':
            reference['artifact_sha256'] = '0' * 64
        elif fault == 'part-digest':
            reference['part_sha256'] = '0' * 64
        elif fault == 'path':
            reference['path'] = 'unassigned.py'
        elif fault == 'side':
            reference['side'] = 'diff'
        elif fault == 'range':
            reference['source_range'] = {'start': 999999, 'end': 999999}
        elif fault == 'fragment':
            reference['fragment'] = {'offset': 999999, 'bytes': 1, 'sha256': '0' * 64}
        elif fault == 'supporting-only':
            reference['role'] = 'supporting'
        elif fault == 'reference-count':
            output['targets'][0]['evidence_refs'] *= 5
        elif fault.startswith('supporting-'):
            supporting = {
                'artifact_sha256': hashlib.sha256((review.repo / 'module_06.py').read_bytes()).hexdigest(),
                'part_sha256': None, 'revision': stage['analyzed_revision']['head_sha'],
                'path': 'module_06.py', 'side': 'new', 'source_range': {'start': 1, 'end': 1},
                'fragment': None, 'basis': 'source_inspection', 'role': 'supporting',
            }
            if fault == 'supporting-digest':
                supporting['artifact_sha256'] = '0' * 64
            elif fault == 'supporting-missing-path':
                supporting['path'] = 'missing-frozen.py'
            else:
                supporting['source_range'] = ({'start': 2, 'end': 1} if fault == 'supporting-reversed-range'
                                               else {'start': 2, 'end': 2})
            output['targets'][0]['evidence_refs'].append(supporting)
        else:
            output['targets'][0]['evidence_refs'] = []

    reason = 'malformed_output' if fault == 'reference-count' else 'evidence_incomplete'
    data = await review.finish('python', reason=reason, findings=('Grounded defect',))
    events = stage_ends(review, 'python')
    assert [event['metadata']['admitted'] for event in events] == [True, False]
    assert all(event['metadata']['attempt'] == 1 for event in events)
    assert events[-1]['metadata']['failure_class'] == 'source_attribution'
    assert bool(events[-1]['metadata'].get('schema_rejection')) is (fault == 'reference-count')
    decisions = scopes(data)['python']['target_decisions']
    assert decisions == sorted(admitted, key=lambda item: item['target_id'])
    assert not set(rejected_ids).intersection(item['target_id'] for item in decisions)
    coverage = json.loads((review.repo / '.daydream/deep/review-coverage.json').read_text())
    assert next(scope for scope in coverage['stack_outcomes'] if scope['scope_id'] == 'python')[
        'target_decisions'] == decisions
    assert all('INVALID_ATTRIBUTION_MUST_NOT_SURVIVE' not in call['prompt'] for call in review.backend.calls)


async def test_invalid_source_reference_preserves_prior_admitted_target_decisions(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[str] = [
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
    ]
    case_index = 0
    for fault in cases_0:
        with subtests.test(fault=fault), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            await _case_invalid_source_reference_preserves_prior_admitted_target_decisions(
                tmp_path=scenario.tmp_path, monkeypatch=scenario.monkeypatch, fault=fault
            )


async def _case_old_only_hunk_and_later_continuations_keep_canonical_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: bool
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


async def test_old_only_hunk_and_later_continuations_keep_canonical_ranges(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[bool] = [False, True]
    case_index = 0
    for sandbox in cases_0:
        with subtests.test(sandbox=sandbox), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            await _case_old_only_hunk_and_later_continuations_keep_canonical_ranges(
                tmp_path=scenario.tmp_path, monkeypatch=scenario.monkeypatch, sandbox=sandbox
            )


async def _case_continuation_references_cannot_alias_another_fragment_or_a_source_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
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


async def test_continuation_references_cannot_alias_another_fragment_or_a_source_line(
    tmp_path: Path, subtests: pytest.Subtests
) -> None:
    cases_0: Sequence[str] = ["reordered-fragment", "byte-offset-as-line"]
    case_index = 0
    for fault in cases_0:
        with subtests.test(fault=fault), isolated_case(tmp_path, str(case_index)) as scenario:
            case_index += 1
            await _case_continuation_references_cannot_alias_another_fragment_or_a_source_line(
                tmp_path=scenario.tmp_path, monkeypatch=scenario.monkeypatch, fault=fault
            )


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
