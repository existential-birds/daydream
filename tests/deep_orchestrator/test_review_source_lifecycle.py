"""Runner-enforced review stops join owned source readers and retain admitted evidence."""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from daydream.backends import (
    AgentEvent,
    ClaudeRequestConfig,
    RequestEvent,
    ResultEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.source_reader import SourceReader, source_tool_output
from daydream.config_file import DaydreamFileConfig
from daydream.review_source import SourceRecipe
from tests.deep_orchestrator.test_review_completion import record, scopes
from tests.deep_orchestrator.test_review_investigation import InvestigationRun, candidate, stage_ends
from tests.deep_orchestrator.test_review_native_sources import NativeSourceBackend
from tests.harness.git_helpers import seed_feature_branch, tracked_source_state
from tests.harness.stub_backend import review_stage_state, stage_result
from tests.test_deep_orchestrator import _profile_with_pipeline


async def _read_source(reader: SourceReader, arguments: dict[str, Any]) -> str:
    # Complete the external provider's RPC before yielding its result. MCP
    # client task groups do not cross the backend generator's yield boundary.
    result = None
    async with httpx.AsyncClient(headers={'Authorization': f'Bearer {reader.token}'}, timeout=None) as client:
        async with streamable_http_client(reader.url, http_client=client) as (receive, send, _):
            async with ClientSession(receive, send) as session:
                await session.initialize()
                result = await session.call_tool('read_source', arguments)
    if result is None:
        raise asyncio.CancelledError
    assert not result.isError
    return source_tool_output([block.model_dump(exclude_none=True) for block in result.content])


async def test_runner_tool_budget_closes_owned_readers_and_retains_prior_admitted_source_and_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / 'owned_reader_budget'
    body = "def hello():\n    return 'universe'\n" + ''.join(
        f'# changed source {index:04}: ' + 'x' * 90 + '\n' for index in range(500))
    seed_feature_branch(repo, base={'api.py': "def hello():\n    return 'world'\n", 'guide.md': '# Greeting\n'},
                        feature={'api.py': body, 'guide.md': '# Greeting\nRetain the greeting contract.\n'})
    original = tracked_source_state(repo)
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    urls: list[str] = []
    closed: list[str] = []
    reused: list[dict[str, Any]] = []

    class ReaderBackend(NativeSourceBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            if stage is None or stage['scope_id'] not in {'python', 'structure'}:
                async for event in super().execute(cwd, prompt, *args, **kwargs):
                    yield event
                return
            self.stages.append(stage)
            self.calls.append({'cwd': cwd, 'prompt': prompt, **kwargs})
            recipe = kwargs.get('source_recipe')
            assert isinstance(recipe, SourceRecipe)
            successful = stage['scope_id'] == 'python' and not stage['progress']
            if stage['scope_id'] == 'python' and not successful:
                window, = stage['admitted_source_windows']
                assert window['file'] == 'api.py' and window['revision'] == review.pr.head_sha
                assert (window['start_byte'], window['end_byte']) == (0, len(body.encode()))
                reused.append(window)
                assert stage['source_access'] and all(not access['read_required']
                                                      for access in stage['source_access'])
                decision, = stage['closed_decisions']
                assert decision['candidate_id'] == 'python:candidate:1' and decision['disposition'] == 'confirmed'
                assert decision['evidence_references']
            yield RequestEvent(prompt=prompt, config=ClaudeRequestConfig(source_tool_enabled=True, read_only=True))
            reader = SourceReader(recipe)
            try:
                async with reader:
                    urls.append(reader.url)
                    arguments = next(access['access']['arguments'] for access in stage['source_access']
                                     if access['file'] == 'api.py' and access['side'] == 'after')
                    yield ToolStartEvent(id='grounded-source', name='read_source', input=arguments)
                    captured = await _read_source(reader, arguments)
                    assert json.loads(captured)['body'] == body
                    yield ToolResultEvent(id='grounded-source', output=captured, is_error=False)
                    if successful:
                        output = stage_result(stage, candidates=[candidate(disposition='confirmed', finding=record())])
                        yield ResultEvent(structured_output=output, continuation=None)
                    else:
                        # The provider has not submitted when the unchanged hard
                        # allowance stops it. The final received start is charged.
                        for index in range(60):
                            call_id = f'boundary-probe-{index}'
                            yield ToolStartEvent(id=call_id, name='Grep', input={'pattern': 'hello', 'path': 'api.py'})
                            yield ToolResultEvent(id=call_id, output='api.py:1:def hello():', is_error=False)
                        pytest.fail('The runner did not enforce its cumulative tool allowance')
            finally:
                if reader.url:
                    closed.append(reader.url)

    review.backend = ReaderBackend(repo)
    assert await review.run(deep_shard_enabled=False, file_config=DaydreamFileConfig(supervisor='off')) == 0
    data = review.load()
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    assert data['terminal_result']['analysis_state'] == 'incomplete'
    inventory = scopes(data)
    assert inventory['generic']['status'] == 'complete'
    for scope in ('python', 'structure'):
        assert inventory[scope]['status'] == 'incomplete'
        assert inventory[scope]['reason_codes'] == ['host_tool_budget_exhaustion']
        terminal = stage_ends(review, scope)[-1]['metadata']
        assert terminal['hard_tool_call_allowance'] == (47 if scope == 'python' else 48)
        assert terminal['observed_tool_starts'] == 49
        assert terminal['attempt_tool_starts'] == terminal['hard_tool_call_allowance'] + 1
        assert terminal['remaining_tool_calls'] == 0 and terminal['attempt'] == 1
        assert terminal['failure_class'] == 'quantitative_exhaustion' and not terminal['admitted']
    stages = stage_ends(review, 'python')
    assert len(stages) == 2 and stages[0]['metadata']['admitted']
    assert stages[0]['metadata']['fresh_source_reads'] == 1
    assert stages[1]['metadata']['reused_source_windows'] == len(reused) == 1
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    assert phases['alternatives']['status'] == 'failed'
    assert phases['alternatives']['reason_codes'] == ['host_tool_budget_exhaustion']
    assert len(urls) == 3 and sorted(closed) == sorted(urls)
    async with httpx.AsyncClient() as client:
        for url in urls:
            with pytest.raises(httpx.ConnectError):
                await client.post(url)
    retained = json.loads((repo / '.daydream/deep/stack-python-records.json').read_text())
    assert [finding['description'] for finding in retained['issues']] == ['Grounded defect']
    assert tracked_source_state(repo) == original


async def test_runner_wall_cancellation_joins_real_git_source_workers_before_reader_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / 'owned_reader_wall'
    body = "def hello():\n    return 'universe'\n" + ''.join(
        f'# changed source {index:04}: ' + 'x' * 90 + '\n' for index in range(500))
    dependencies = {f'dependency_{scope}.py': 'CONTRACT = "world"\n' for scope in ('python', 'structure')}
    seed_feature_branch(repo, base={'api.py': "def hello():\n    return 'world'\n", 'guide.md': '# Greeting\n',
                                  **dependencies},
                        feature={'api.py': body, 'guide.md': '# Greeting\nUpdated contract.\n'})
    original = tracked_source_state(repo)
    real_git = shutil.which('git')
    assert real_git is not None
    gates = tmp_path / 'git_gates'
    gates.mkdir()
    binary = tmp_path / 'gated_tools'
    binary.mkdir()
    wrapper = binary / 'git'
    wrapper.write_text(
        '#!/bin/sh\n'
        f'root={shlex.quote(str(gates))}\n'
        f'real_git={shlex.quote(real_git)}\n'
        'target=\nfor target do :; done\n'
        'scope=\n'
        'case "$target" in\n'
        '  dependency_python.py) scope=python ;;\n'
        '  dependency_structure.py) scope=structure ;;\n'
        'esac\n'
        'if [ "$1" = ls-tree ] && [ -n "$scope" ] && [ -f "$root/active" ] '
        '&& [ ! -f "$root/release-$scope" ]; then\n'
        '  identity="$scope-$$"\n'
        '  printf "%s\\n" "$target" > "$root/started-$identity"\n'
        '  while [ ! -f "$root/release-$scope" ]; do /bin/sleep 0.01; done\n'
        '  "$real_git" "$@"\n'
        '  status=$?\n'
        '  printf "git completed\\n" > "$root/done-$identity"\n'
        '  exit "$status"\n'
        'fi\n'
        'exec "$real_git" "$@"\n',
    )
    wrapper.chmod(0o755)
    original_path = os.environ['PATH']
    review = InvestigationRun(repo, tmp_path, monkeypatch)
    urls: list[str] = []
    releases: list[asyncio.Task[None]] = []
    exits: dict[str, list[tuple[int, bool, bool]]] = {}
    cancelled: set[str] = set()
    reused: list[dict[str, Any]] = []

    def running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    class WallBackend(NativeSourceBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            stage = review_stage_state(prompt)
            if stage is None or stage['scope_id'] not in {'python', 'structure'}:
                async for event in super().execute(cwd, prompt, *args, **kwargs):
                    yield event
                return
            self.stages.append(stage)
            self.calls.append({'cwd': cwd, 'prompt': prompt, **kwargs})
            recipe = kwargs.get('source_recipe')
            assert isinstance(recipe, SourceRecipe)
            scope = stage['scope_id']
            successful = scope == 'python' and not stage['progress']
            if not successful:
                # Planning and fixture setup use the ordinary Git executable.
                # The wrapper delegates all non-gated reads directly to it.
                monkeypatch.setenv('PATH', str(binary) + os.pathsep + original_path)
            if scope == 'python' and not successful:
                window, = stage['admitted_source_windows']
                assert window['file'] == 'api.py' and window['revision'] == review.pr.head_sha
                assert window['end_byte'] == len(body.encode())
                reused.append(window)
            yield RequestEvent(prompt=prompt, config=ClaudeRequestConfig(source_tool_enabled=True, read_only=True))
            reader = SourceReader(recipe)
            try:
                async with reader:
                    urls.append(reader.url)
                    if successful:
                        arguments = next(access['access']['arguments'] for access in stage['source_access']
                                         if access['file'] == 'api.py' and access['side'] == 'after')
                        yield ToolStartEvent(id='admitted-source', name='read_source', input=arguments)
                        captured = await _read_source(reader, arguments)
                        assert json.loads(captured)['body'] == body
                        yield ToolResultEvent(id='admitted-source', output=captured, is_error=False)
                        output = stage_result(stage, candidates=[candidate(disposition='confirmed', finding=record())])
                        yield ResultEvent(structured_output=output, continuation=None)
                        return
                    # Only this actual frozen-source operation blocks; recipe
                    # preparation and its initial revalidation have completed.
                    (gates / 'active').touch()
                    arguments = {'target_id': f'dependency_{scope}.py', 'side': 'after'}
                    yield ToolStartEvent(id='pending-dependency', name='read_source', input=arguments)
                    request = asyncio.create_task(_read_source(reader, arguments))
                    try:
                        while not list(gates.glob(f'started-{scope}-*')):
                            if request.done():
                                await request
                                pytest.fail('Frozen source read returned before the real Git gate started')
                            await asyncio.sleep(0.01)
                        await asyncio.shield(request)
                        pytest.fail('The gated source read returned before the runner wall deadline')
                    except BaseException:
                        cancelled.add(scope)
                        async def release() -> None:
                            await asyncio.sleep(0.2)
                            (gates / f'release-{scope}').touch()
                        releases.append(asyncio.create_task(release()))
                        raise
                    finally:
                        if not request.done():
                            request.cancel()
                            with anyio.CancelScope(shield=True), suppress(asyncio.CancelledError):
                                await request
            finally:
                if not successful:
                    # Take this snapshot synchronously at reader exit. Later
                    # synthesis work cannot hide a source worker that escaped.
                    exits[scope] = [(int(marker.name.rsplit('-', 1)[1]),
                                     (gates / marker.name.replace('started-', 'done-', 1)).exists(),
                                     running(int(marker.name.rsplit('-', 1)[1])))
                                    for marker in gates.glob(f'started-{scope}-*')]

    review.backend = WallBackend(repo)
    try:
        assert await review.run(deep_shard_enabled=False, file_config=DaydreamFileConfig(supervisor='off'),
                                review_profile=_profile_with_pipeline(review_wall_budget_s=10)) == 0
    finally:
        for scope in ('python', 'structure'):
            (gates / f'release-{scope}').touch()
        await asyncio.gather(*releases, return_exceptions=True)
        # A red control must also leave the test process clean.
        async with asyncio.timeout(5):
            while any(running(int(marker.name.rsplit('-', 1)[1])) for marker in gates.glob('started-*')):
                await asyncio.sleep(0.01)
    assert cancelled == set(exits) == {'python', 'structure'}
    for snapshots in exits.values():
        assert snapshots and all(done and not alive for _, done, alive in snapshots)
    assert len(reused) == 1 and len(urls) == 3
    async with httpx.AsyncClient() as client:
        for url in urls:
            with pytest.raises(httpx.ConnectError):
                await client.post(url)
    data = review.load()
    assert [finding['title'] for finding in data['findings']] == ['Grounded defect']
    assert data['terminal_result']['analysis_state'] == 'incomplete'
    for scope in ('python', 'structure'):
        assert scopes(data)[scope]['status'] == ('incomplete' if scope == 'python' else 'uncovered')
        assert scopes(data)[scope]['reason_codes'] == ['host_pipeline_budget_exhaustion']
        terminal = stage_ends(review, scope)[-1]['metadata']
        assert terminal['failure_class'] == 'quantitative_exhaustion' and not terminal['admitted']
    stages = stage_ends(review, 'python')
    assert len(stages) == 2 and stages[0]['metadata']['admitted']
    assert stages[1]['metadata']['reused_source_windows'] == 1
    phases = {phase['phase']: phase for phase in data['terminal_result']['phase_outcomes']}
    assert phases['alternatives']['status'] == 'failed'
    assert phases['alternatives']['reason_codes'] == ['host_pipeline_budget_exhaustion']
    assert tracked_source_state(repo) == original
