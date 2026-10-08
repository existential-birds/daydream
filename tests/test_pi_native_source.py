"""The installed native Pi CLI loads the shipped bounded frozen-source tool."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from importlib.resources import as_file, files
from pathlib import Path

import pytest

from daydream.backends import RequestEvent, ResultEvent, ToolResultEvent, ToolStartEvent
from daydream.backends.pi import PiBackend
from daydream.git_ops.source import frozen_source
from daydream.review_source import SourceRecipe, SourceWindow
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.protocol_cli import install_protocol_cli


@pytest.mark.parametrize('selector', ['valid', 'unknown', 'arbitrary-path', 'wrong-side'])
async def test_installed_pi_reads_only_its_frozen_packet_and_preserves_native_call_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str,
) -> None:
    native_pi = shutil.which('pi')
    if native_pi is None or shutil.which('node') is None:
        pytest.skip('The installed official Pi CLI and Node are required for native tool loading.')
    repo = tmp_path / 'native_deleted_source'
    body = 'def greeting():\n    return "世界"\n'
    seed_feature_branch(repo, base={'retired.py': body, 'keep.py': 'VALUE = 1\n'},
                        feature={'keep.py': 'VALUE = 2\n'})
    revision = git(repo, 'rev-parse', 'main')
    git(repo, 'rm', 'retired.py')
    git(repo, 'commit', '-m', 'delete greeting')
    oid, raw = frozen_source(repo, revision, 'retired.py')
    recipe = SourceRecipe((SourceWindow(
        target_ids=('retired.py',), file='retired.py', source_path='retired.py', side='before', revision=revision,
        content_sha256=hashlib.sha256(raw).hexdigest(), blob_oid=oid, start_line=1, end_line=2,
        start_byte=0, end_byte=len(raw), body=raw.decode('utf-8'),
    ),), repo)
    provider = Path(__file__).parent / 'fixtures/pi_native/source_provider.ts'
    bin_dir = tmp_path / 'provider-boundary'
    bin_dir.mkdir()
    shim = bin_dir / 'pi'
    shim.write_text(f'#!{sys.executable}\nimport os, sys\n'
                    f'os.execv({native_pi!r}, [{native_pi!r}, "--extension", {str(provider)!r}, *sys.argv[1:]])\n')
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('PI_PROVIDER', 'source-fixture')
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(tmp_path / 'isolated-pi-config'))
    for name in ('PI_API_KEY', 'NOUS_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    arguments = {'target_id': 'unknown.py' if selector == 'unknown' else 'retired.py',
                 'side': 'after' if selector == 'wrong-side' else 'before'}
    if selector == 'arbitrary-path':
        arguments['path'] = str(tmp_path / 'outside-packet.txt')
        (tmp_path / 'outside-packet.txt').write_text('OUTSIDE_PACKET_MUST_NOT_BE_READ\n')
    monkeypatch.setenv('DAYDREAM_TEST_SOURCE_SELECTOR', json.dumps(arguments))
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': {'observed_body': {'type': 'string'}, 'source_error': {'type': 'boolean'},
                             'active_tools': {'type': 'array', 'items': {'type': 'string'}}},
              'required': ['observed_body', 'source_error', 'active_tools']}
    backend = PiBackend(model='source-model', reasoning_effort='high')
    events = [event async for event in backend.execute(
        repo, 'Read the supplied deleted source window.', output_schema=schema,
        read_only=True, persist_session=False, source_recipe=recipe,
    )]
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    final = next(event for event in reversed(events) if isinstance(event, ResultEvent)).structured_output
    assert len(starts) == len(results) == 1
    assert starts[0].id == results[0].id == 'native-frozen-source-001'
    assert starts[0].name == 'read_source' and starts[0].input == arguments
    assert results[0].is_error is (selector != 'valid')
    assert not results[0].truncated and not results[0].cancelled
    assert final == {'observed_body': body if selector == 'valid' else '', 'source_error': selector != 'valid',
                     'active_tools': ['find', 'grep', 'ls', 'read', 'read_source']}
    assert not (repo / 'retired.py').exists()
    if selector == 'valid':
        assert recipe.native_result(starts[0].input, results[0].output) == recipe.windows[0]
    with as_file(files('daydream.backends').joinpath('pi_read_source.ts')) as resource:
        assert resource.is_file()


@pytest.mark.parametrize('contract', ['matching', 'mismatched', 'plain'])
async def test_pi_request_has_one_matching_invocation_schema_and_keeps_custom_output_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract: str,
) -> None:
    fixture = install_protocol_cli(tmp_path / 'external-provider', 'pi')
    monkeypatch.setenv('PATH', str(fixture.bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('PI_PROVIDER', 'nous')
    monkeypatch.delenv('PI_API_KEY', raising=False)
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': {'status': {'type': 'string'}}, 'required': ['status']}
    other = {'type': 'object', 'additionalProperties': False,
             'properties': {'obsolete': {'type': 'string'}}, 'required': ['obsolete']}
    prompt = 'Review the assigned source.'
    if contract != 'plain':
        prompt += '\nHost review stage:\n' + json.dumps(
            {'response_contract': {'schema': schema if contract == 'matching' else other}})
    events = [event async for event in PiBackend(model='fixture-model').execute(
        tmp_path, prompt, output_schema=schema, read_only=True, require_complete_root=True,
    )]
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert request.prompt.count('"additionalProperties"') == (2 if contract == 'mismatched' else 1)
    assert '"status"' in request.prompt
    assert ('"obsolete"' in request.prompt) is (contract == 'mismatched')


async def test_installed_pi_bounded_read_footer_is_verified_against_actual_source_lines_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_pi = shutil.which('pi')
    if native_pi is None or shutil.which('node') is None:
        pytest.skip('The installed official Pi CLI and Node are required for native tool loading.')
    repo = tmp_path / 'native_bounded_read'
    body = 'VALUE = 1\ndef greeting():\n    return "world"\nEXPECTED = "world"\nassert greeting() == EXPECTED\n'
    seed_feature_branch(repo, base={'api.py': 'VALUE = 0\n'}, feature={'api.py': body})
    revision = git(repo, 'rev-parse', 'HEAD')
    oid, raw = frozen_source(repo, revision, 'api.py')
    recipe = SourceRecipe((SourceWindow(
        target_ids=('api.py',), file='api.py', source_path='api.py', side='after', revision=revision,
        content_sha256=hashlib.sha256(raw).hexdigest(), blob_oid=oid, start_line=1, end_line=5,
        start_byte=0, end_byte=len(raw), body=body,
    ),), repo)
    provider = Path(__file__).parent / 'fixtures/pi_native/source_provider.ts'
    bin_dir = tmp_path / 'provider-boundary'
    bin_dir.mkdir()
    shim = bin_dir / 'pi'
    shim.write_text(f'#!{sys.executable}\nimport os, sys\n'
                    f'os.execv({native_pi!r}, [{native_pi!r}, "--extension", {str(provider)!r}, *sys.argv[1:]])\n')
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('PI_PROVIDER', 'source-fixture')
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(tmp_path / 'isolated-pi-config'))
    monkeypatch.setenv('DAYDREAM_TEST_SOURCE_TOOL', 'read')
    monkeypatch.setenv('DAYDREAM_TEST_SOURCE_SELECTOR', json.dumps({'path': 'api.py', 'offset': 1, 'limit': 2}))
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': {'observed_body': {'type': 'string'}, 'source_error': {'type': 'boolean'},
                             'active_tools': {'type': 'array', 'items': {'type': 'string'}}},
              'required': ['observed_body', 'source_error', 'active_tools']}
    events = [event async for event in PiBackend(model='source-model').execute(
        repo, 'Read the first two source lines.', output_schema=schema, read_only=True,
        persist_session=False, source_recipe=recipe,
    )]
    start = next(event for event in events if isinstance(event, ToolStartEvent))
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert start.id == result.id == 'native-frozen-source-001' and start.name == 'read'
    assert result.output == 'VALUE = 1\ndef greeting():\n\n[4 more lines in file. Use offset=3 to continue.]'
    assert result.is_error is False and result.truncated is False
    matched = recipe.match_read(start.input, result.output, repo)
    assert matched is not None, 'genuine bounded Pi read must yield independently verified source evidence'
    assert matched.body == 'VALUE = 1\ndef greeting():\n'
    assert (matched.start_line, matched.end_line, matched.start_byte, matched.end_byte) == (1, 2, 0, 26)
