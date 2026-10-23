"""Real host -> reviewer process -> native CLI interruption contract."""
from __future__ import annotations

import asyncio
import errno
import json
import os
import shlex
import signal
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.benchmark.reviewbench.harbor_agent import DaydreamPublicReviewAgent
from tests.harness.codex_lifecycle_cli import install_codex_lifecycle_cli
from tests.test_reviewbench_harbor_agent import public_input


class LocalReviewerEnvironment:
    """Buffered exec with Harbor's cancellation boundary: children outlive awaits."""

    def __init__(self, environment: dict[str, str], repository: Path, *,
                 publication_fault: str | None = None, close_stdout: bool = False, foreign_stderr: bytes = b'',
                 entrypoint_bootstrap: str | None = None) -> None:
        self.environment = environment
        self.repository = repository
        self.publication_fault = publication_fault
        self.close_stdout = close_stdout
        self.foreign_stderr = foreign_stderr
        self.entrypoint_bootstrap = entrypoint_bootstrap
        self.results: list[SimpleNamespace] = []
        self.children: list[asyncio.subprocess.Process] = []
        self.collectors: list[asyncio.Task[tuple[bytes, bytes]]] = []

    async def exec(self, command: str, **kwargs: Any) -> Any:
        arguments = shlex.split(command)
        assert arguments[0] == 'python'
        arguments[0] = sys.executable
        inject = self.publication_fault is not None or self.foreign_stderr or self.entrypoint_bootstrap
        if inject and arguments[1:4] == [
            '-I', '-m', 'daydream.benchmark.reviewbench.harbor_process',
        ]:
            # Patch the actual filesystem install only in the supervisor. The
            # production entrypoint/native CLI remain separate, unpatched children.
            bootstrap = ('import errno,os,runpy,sys\nfrom pathlib import Path\n' +
                f'os.write(2,{self.foreign_stderr!r})\n' +
                'original_link = os.link\n'
                'def fail_link(source,destination,**kwargs):\n'
                f'    if Path(destination).name == {self.publication_fault!r}:\n'
                '        raise OSError(errno.ENOSPC,"publication-secret-canary")\n'
                '    return original_link(source,destination,**kwargs)\n'
                'os.link = fail_link\n'
                + ('import asyncio\noriginal_spawn=asyncio.create_subprocess_exec\n'
                   'async def spawn(*args,**kwargs):\n'
                   '    if args[1:4] == ("-I","-m","daydream.benchmark.reviewbench.harbor_entrypoint"):\n'
                   f'        args=(args[0],"-I","-c",{self.entrypoint_bootstrap!r})\n'
                   '    return await original_spawn(*args,**kwargs)\n'
                   'asyncio.create_subprocess_exec=spawn\n' if self.entrypoint_bootstrap else '') +
                'sys.argv = ["daydream.benchmark.reviewbench.harbor_process", *sys.argv[1:]]\n'
                'runpy.run_module(sys.argv[0],run_name="__main__",alter_sys=True)\n')
            arguments = [sys.executable, '-I', '-c', bootstrap, *arguments[4:]]
        writer: int | None = None
        if self.close_stdout:
            reader, writer = os.pipe()
            os.close(reader)  # A real collector pipe with no remaining reader.
        try:
            child = await asyncio.create_subprocess_exec(*arguments, cwd=self.repository, env=self.environment,
                stdout=writer if writer is not None else asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=True)
        finally:
            if writer is not None:
                os.close(writer)
        self.children.append(child)
        collector = asyncio.create_task(child.communicate())
        self.collectors.append(collector)
        stdout, stderr = await asyncio.shield(collector)
        result = SimpleNamespace(return_code=child.returncode, stdout=(stdout or b'').decode(), stderr=stderr.decode())
        self.results.append(result)
        return result

    async def close(self) -> None:
        for child in self.children:
            if child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        await asyncio.wait_for(asyncio.gather(*self.collectors, return_exceptions=True), timeout=5)


@pytest.mark.parametrize('interruption', [
    'timeout', 'cancel', 'repeated_cancel', 'force', 'control_failure', 'late_child', 'slow_process_listing',
    'collector_timeout', 'collector_transport', 'collector_repeated_cancel',
    'collector_control_failure', 'collector_marker_failure',
])
async def test_outer_timeout_stops_owned_reviewer_and_retains_process_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str,
) -> None:
    """An interrupted host exec must stop its child and retain joined evidence.

    test_cancelled_public_reviewer_exports_joined_native_evidence_and_propagates_cancellation
    cancels inside the reviewer loop and cannot detect a detached child after a
    cancelled host collector. This harness enters production host and executable
    modules, real Git and Codex transport; only the external CLI is scripted.
    Collector-error rows additionally cover an already-failed await; cancellation
    rows alone cannot detect cleanup returning before that child finalizes.
    """
    if interruption == 'late_child' and sys.platform != 'linux':
        pytest.skip('late orphan containment requires Linux PR_SET_CHILD_SUBREAPER')
    if interruption == 'slow_process_listing' and sys.platform != 'darwin':
        pytest.skip('macOS process-listing transport regression')
    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', structure_mode='preterminal_silence',
        spawn_on_term=interruption == 'late_child')
    if interruption == 'slow_process_listing':
        # A slow external process listing must not destroy reviewer evidence.
        # The original timeout row cannot deterministically reproduce scheduler
        # starvation of the old 1s ps collector. Keep actual subprocess behavior.
        listing = cli.bin_dir / 'ps'
        listing.write_text(f"#!{sys.executable}\n" +
            'import json,os,sys,time\nfrom pathlib import Path\n' +
            f'root = Path({str(cli.root)!r})\n' +
            "if any(json.loads(p.read_text())['mode']=='preterminal_silence' " +
            "for p in (root/'observations').glob('*.json')): time.sleep(2)\n" +
            "os.execv('/bin/ps', ['/bin/ps', *sys.argv[1:]])\n")
        listing.chmod(0o700)
    child_environment = {**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))}
    collector_failure = (TimeoutError('collector failure secret') if interruption == 'collector_timeout'
                         else ConnectionError('collector failure secret'))
    collector_release = asyncio.Event()
    detached: list[asyncio.Task[Any]] = []
    if interruption == 'collector_marker_failure':
        link = os.link

        def fail_marker(source: Any, destination: Any, **kwargs: Any) -> None:
            if Path(destination).name.startswith('reviewer-cancel-'):
                raise OSError(errno.ENOSPC, 'marker failure secret')
            link(source, destination, **kwargs)

        monkeypatch.setattr(os, 'link', fail_marker)

    class ControlFailureEnvironment(LocalReviewerEnvironment):
        async def exec(self, command: str, **kwargs: Any) -> Any:
            if interruption in {'control_failure', 'collector_control_failure'} and '--cancel' in command:
                return SimpleNamespace(return_code=1, stdout='', stderr='control exec unavailable')
            if interruption.startswith('collector_') and '--cancel' not in command:
                # A transport failure loses the Docker client's await without
                # killing the separate in-container supervisor/native process.
                detached.append(asyncio.create_task(super().exec(command, **kwargs)))
                await collector_release.wait()
                raise collector_failure
            return await super().exec(command, **kwargs)

    local = ControlFailureEnvironment(child_environment, Path(environment['DAYDREAM_RB_REPOSITORY']))
    logs = Path(environment['DAYDREAM_RB_EVIDENCE']).parent
    agent = DaydreamPublicReviewAgent(logs_dir=logs)
    running = asyncio.create_task(agent.run('frozen public task', local, SimpleNamespace()))
    try:
        deadline = asyncio.get_running_loop().time() + 15
        while not any(row['mode'] == 'preterminal_silence' for row in cli.observations()):
            if running.done():
                await running
            assert asyncio.get_running_loop().time() < deadline, 'scripted native CLI did not reach readiness'
            await asyncio.sleep(0.01)
        evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
        if interruption == 'force':
            checkpoints = evidence.with_name(evidence.name + '-checkpoints')
            checkpoint_deadline = asyncio.get_running_loop().time() + 10
            while not list(checkpoints.glob('*/checkpoint.json')):
                assert asyncio.get_running_loop().time() < checkpoint_deadline, 'no active checkpoint retained'
                await asyncio.sleep(0.01)
            blocked = next(row for row in cli.observations() if row['mode'] == 'preterminal_silence')
            os.kill(blocked['ppid'], signal.SIGSTOP)
        if interruption.startswith('collector_'):
            collector_release.set()
            if interruption == 'collector_repeated_cancel':
                while not list(logs.glob('reviewer-cancel-*.json')) and not running.done():
                    await asyncio.sleep(0.01)
                running.cancel()
                await asyncio.sleep(0.01)
                running.cancel()
            with pytest.raises(type(collector_failure)) as raised:
                await asyncio.wait_for(running, timeout=30)
            assert raised.value is collector_failure
            await asyncio.wait_for(asyncio.gather(*detached), timeout=5)
        elif interruption in {'timeout', 'slow_process_listing'}:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(running, timeout=0.05)
        else:
            running.cancel()
            if interruption == 'repeated_cancel':
                await asyncio.sleep(0.01)
                running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, timeout=30)
        receipt = json.loads((logs / 'reviewer-process.json').read_text())
        assert receipt['interruption'] == ('host_execution_failed'
            if interruption.startswith('collector_') else 'host_cancellation')
        assert receipt['outer_deadline_seconds'] is None
        if interruption == 'late_child':
            escaped = json.loads((cli.root / 'escaped.json').read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(escaped['pid'], 0)
            assert receipt['descendant_containment'] == 'linux_subreaper'
            assert receipt['owned_descendants_cleanup'] == 'joined'
        assert receipt['cleanup_outcome'] == ('forced_termination' if interruption == 'force' else 'graceful_exit')
        host_receipt = json.loads((logs / 'reviewer-host.json').read_text())
        assert host_receipt['interruption'] == ('host_execution_failed'
            if interruption.startswith('collector_') else 'host_task_cancelled')
        if interruption.startswith('collector_'):
            assert host_receipt['output_collection'] == 'unavailable'
            assert host_receipt['execution_error'] == type(collector_failure).__name__
            assert host_receipt['process_completion'] == 'observed'
            assert 'collector failure secret' not in (logs / 'reviewer-host.json').read_text()
        if interruption in {'control_failure', 'collector_control_failure'}:
            assert host_receipt['cancellation_channels'] == {'shared_marker': 'published', 'control_exec': 'failed'}
        if interruption == 'collector_marker_failure':
            assert host_receipt['cancellation_channels'] == {
                'shared_marker': 'unavailable:OSError', 'control_exec': 'returned_zero'}
        if interruption == 'force':
            assert not (evidence / 'execution.json').exists()
            assert list(checkpoints.glob('*/checkpoint.json'))
            assert receipt['terminal_evidence_available'] is False
            assert all(child.returncode is not None for child in local.children)
            assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
            return
        assert (evidence / 'execution.json').is_file(), 'outer timeout lost native execution evidence'
        execution = json.loads((evidence / 'execution.json').read_text())
        assert execution['status'] == 'cancelled' and execution['terminal_result'] is None
        root = json.loads((evidence / 'trajectory.json').read_text())
        siblings = [row for row in root['extra']['subtrajectories'] if 'sibling_trajectory_ref' in row]
        assert siblings
        assert any('Synthetic initial commentary' in (evidence / row['sibling_trajectory_ref']).read_text()
                   for row in siblings)
        assert (logs / 'reviewer-entrypoint.log').is_file()
        assert all(child.returncode is not None for child in local.children)
        for row in cli.observations():
            with pytest.raises(ProcessLookupError):
                os.kill(row['pid'], 0)
        for path in logs.rglob('*'):
            if path.is_file():
                assert b'private-refresh-canary' not in path.read_bytes()
                assert b'forbidden-api-canary' not in path.read_bytes()
        assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
    finally:
        if not running.done():
            running.cancel()
        escaped_file = cli.root / 'escaped.json'
        if escaped_file.exists():
            escaped = json.loads(escaped_file.read_text())
            try:
                os.killpg(escaped['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
        for row in cli.observations():
            try:
                os.killpg(row['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
        await local.close()
        await asyncio.gather(*detached, return_exceptions=True)
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.parametrize('credential_type', ['api_key', 'oauth'])
async def test_codex_tool_output_of_admitted_pi_auth_is_redacted_in_interrupted_bundle(
    tmp_path: Path, credential_type: str,
) -> None:
    """Native tool evidence cannot retain schema-admitted opaque Pi credentials.

    Existing supervisor privacy cases emit familiar credential labels directly
    to pipes. This reads the actual validator-admitted auth object in a scripted
    Codex process and emits genuine public tool events through the real runner,
    active checkpoint and joined root/sibling exports. Reviewer API-key and judge
    OAuth stores retain their original files; this invokes no Pi provider.
    """
    from daydream.benchmark.reviewbench.diagnostics import read_checkpoint
    from daydream.benchmark.reviewbench.harbor_auth import validate_judge_home, validate_pi_reviewer_home
    from daydream.redaction import redact_structured_text

    environment = public_input(tmp_path)
    pi_home = tmp_path / 'synthetic-pi-auth'
    pi_home.mkdir(mode=0o700)
    secret = 'opaque-reviewbench-pi-auth-secret-canary'
    credential = pi_home / 'auth.json'
    auth = ({'nous': {'type': 'api_key', 'key': secret}} if credential_type == 'api_key' else
            {'openai-codex': {'type': 'oauth', 'access': secret + '-access', 'refresh': secret + '-refresh',
                              'accountId': 'synthetic-public-account', 'expires': 9000000000000}})
    original_auth = json.dumps(auth).encode()
    credential.write_bytes(original_auth)
    if credential_type == 'api_key':
        validate_pi_reviewer_home(pi_home, 'nous')
    else:
        validate_judge_home(pi_home)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', structure_mode='preterminal_silence',
        empty_reviews=True, tool_output_path=credential)
    local = LocalReviewerEnvironment({**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))},
        Path(environment['DAYDREAM_RB_REPOSITORY']))
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    checkpoints = evidence.with_name(evidence.name + '-checkpoints')
    running = asyncio.create_task(DaydreamPublicReviewAgent(logs_dir=evidence.parent).run(
        'frozen synthetic task', local, SimpleNamespace()))
    try:
        deadline = asyncio.get_running_loop().time() + 15
        while not any(row['mode'] == 'preterminal_silence' for row in cli.observations()):
            if running.done():
                await running
            assert asyncio.get_running_loop().time() < deadline, 'native tool process did not reach readiness'
            await asyncio.sleep(0.01)
        while not (manifests := list(checkpoints.glob('*/checkpoint.json'))):
            assert asyncio.get_running_loop().time() < deadline, 'active tool checkpoint was not retained'
            await asyncio.sleep(0.01)
        manifest, documents = read_checkpoint(sorted(manifests)[-1].parent)
        assert manifest['kind'] == 'checkpoint' and len(documents) >= 3
        checkpoint_text = json.dumps(documents)
        assert 'ordinary-public-key' in checkpoint_text
        assert secret not in checkpoint_text, 'active checkpoint retained an admitted Pi credential'
        tool_payloads = [json.loads(result['content']) for document in documents for step in document['steps']
                         for result in step.get('observation', {}).get('results', [])
                         if result['source_call_id'] == 'fixture-input']
        assert tool_payloads
        for payload in tool_payloads:
            assert payload['public'] == {'type': 'record', 'key': 'ordinary-public-key'}
            assert payload['usage'] == {'input': 0, 'output': None}
            admitted = payload['data']['nous' if credential_type == 'api_key' else 'openai-codex']
            assert admitted['type'] == credential_type
            if credential_type == 'oauth':
                assert admitted['accountId'] == 'synthetic-public-account' and admitted['expires'] == 9000000000000
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=35)
        execution = json.loads((evidence / 'execution.json').read_text())
        assert execution['status'] == 'cancelled' and execution['candidate'] is None
        assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
        root = json.loads((evidence / 'trajectory.json').read_text())
        siblings = [row['sibling_trajectory_ref'] for row in root['extra']['subtrajectories']
                    if 'sibling_trajectory_ref' in row]
        assert siblings and all((evidence / reference).is_file() for reference in siblings)
        retained = {path: path.read_bytes() for path in evidence.parent.rglob('*') if path.is_file()}
        assert any(b'ordinary-public-key' in body for body in retained.values())
        assert not [str(path.relative_to(evidence.parent))
                    for path, body in retained.items() if secret.encode() in body]
        for body in retained.values():
            once = redact_structured_text(body.decode('utf-8'))
            assert redact_structured_text(once) == once
        collected = ''.join((result.stdout or '') + (result.stderr or '') for result in local.results)
        assert secret not in collected
        assert credential.read_bytes() == original_auth
    finally:
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await local.close()


@pytest.mark.parametrize('receipt_conflict', [False, True])
async def test_cancelled_host_before_reviewer_start_cannot_launch_native_inference(
    tmp_path: Path, receipt_conflict: bool,
) -> None:
    """A delayed exec may start after host cancellation; a tombstone must stop it.

    The running-review regression reaches an existing process. This independently
    covers the startup race using real control/executable commands and filesystem.
    """
    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='preterminal_silence')
    child_environment = {**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))}
    launch, entered, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class DelayedEnvironment(LocalReviewerEnvironment):
        async def exec(self, command: str, **kwargs: Any) -> Any:
            if '--cancel' not in command:
                entered.set()
                await launch.wait()
            result = await super().exec(command, **kwargs)
            if '--cancel' in command:
                stopped.set()
            return result

    local = DelayedEnvironment(child_environment, Path(environment['DAYDREAM_RB_REPOSITORY']))
    logs = Path(environment['DAYDREAM_RB_EVIDENCE']).parent
    if receipt_conflict:
        logs.mkdir(parents=True)
        (logs / 'reviewer-host.json').write_text('prior competing host receipt')
    agent = DaydreamPublicReviewAgent(logs_dir=logs)
    running = asyncio.create_task(agent.run('frozen public task', local, SimpleNamespace()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        running.cancel()
        await asyncio.wait_for(stopped.wait(), timeout=5)
        launch.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=5)
        receipt = json.loads((logs / 'reviewer-process.json').read_text())
        assert receipt['cleanup_outcome'] == 'cancelled_before_start'
        assert not cli.observations()
        if receipt_conflict:
            assert (logs / 'reviewer-host.json').read_text() == 'prior competing host receipt'
        assert not Path(environment['DAYDREAM_RB_EVIDENCE']).exists()
        assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
    finally:
        launch.set()
        if not running.done():
            running.cancel()
        await local.close()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.parametrize('diagnostic_failure,foreign_stderr,entrypoint_payload', [
    pytest.param('none', b'', b'', id='none'),
    pytest.param('log_conflict', b'', b'', id='log_conflict'),
    pytest.param('receipt_conflict', b'', b'', id='receipt_conflict'),
    pytest.param('log_symlink', b'', b'', id='log_symlink'),
    pytest.param('receipt_symlink', b'', b'', id='receipt_symlink'),
    pytest.param('supervisor_enospc', b'', b'', id='supervisor_enospc'),
    pytest.param('host_fallback_enospc', b'', b'', id='host_fallback_enospc'),
    pytest.param('closed_collector_pipe', b'', b'', id='closed_collector_pipe'),
    pytest.param('log_conflict', b'External runtime warning: ' + b'x' * 2048 + b'\n', b'', id='foreign_stderr'),
    pytest.param('none', b'', b'{"message":"\\ud800"}\n', id='surrogate_json'),
])
async def test_completed_process_review_keeps_native_result_and_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str], diagnostic_failure: str,
    foreign_stderr: bytes, entrypoint_payload: bytes,
) -> None:
    """Process supervision must preserve the ordinary scorable successful path.

    Existing terminal tests bypass executable startup and supervisor buffering;
    this runs a scripted successful native review through the production host.
    Optional diagnostic install failures must preserve the actual successful
    native result; finalized native evidence/findings remain authoritative.
    A real executable bootstrap emits harmless foreign stderr in one row: an
    aggregate output-size assertion otherwise mistakes runtime warnings for
    oversized Daydream diagnostics, which quiet process fixtures cannot expose.
    A child JSON surrogate must also survive supervisor redaction/UTF-8 log
    publication; ordinary ASCII streams cannot expose that encoding boundary.
    """
    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', empty_reviews=True)
    declared_task = tmp_path / 'declared-task'
    declared_task.mkdir()
    (declared_task / 'task.toml').write_text('[agent]\ntimeout_sec = 900\n')
    (tmp_path / 'config.json').write_text(json.dumps({'task': {'path': str(declared_task)}, 'agent': {}}))
    local = LocalReviewerEnvironment({**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))},
        Path(environment['DAYDREAM_RB_REPOSITORY']),
        publication_fault='reviewer-process.json' if diagnostic_failure == 'supervisor_enospc' else None,
        close_stdout=diagnostic_failure == 'closed_collector_pipe', foreign_stderr=foreign_stderr,
        entrypoint_bootstrap=('import asyncio,os\n'
            f'os.write(1,{entrypoint_payload!r})\n'
            'os.dup2(2,1)\n'  # Keep stdout purely JSON; retain ordinary runner output on captured stderr.
            'from daydream.benchmark.reviewbench.harbor_entrypoint import _executable_main\n'
            'raise SystemExit(asyncio.run(_executable_main()))\n') if entrypoint_payload else None)
    logs = Path(environment['DAYDREAM_RB_EVIDENCE']).parent
    protected: Path | None = None
    destination: Path | None = None
    if diagnostic_failure.endswith(('conflict', 'symlink')):
        logs.mkdir(parents=True)
        protected = tmp_path / 'competing-diagnostic'
        protected.write_text('prior competing diagnostic bytes')
        destination = logs / ('reviewer-entrypoint.log' if diagnostic_failure.startswith('log')
                              else 'reviewer-process.json')
        if diagnostic_failure.endswith('symlink'):
            destination.symlink_to(protected)
        else:
            destination.write_bytes(protected.read_bytes())
    host_logs = tmp_path / 'host-logs' if diagnostic_failure == 'host_fallback_enospc' else logs
    if diagnostic_failure == 'host_fallback_enospc':
        link = os.link

        def fail_host_log(source: Any, destination: Any, **kwargs: Any) -> None:
            if Path(destination).name == 'reviewer-entrypoint.log':
                raise OSError(errno.ENOSPC, 'publication-secret-canary')
            link(source, destination, **kwargs)

        monkeypatch.setattr(os, 'link', fail_host_log)
    try:
        await asyncio.wait_for(DaydreamPublicReviewAgent(logs_dir=host_logs).run(
            'frozen public task', local, SimpleNamespace()), timeout=30)
        record = json.loads((Path(environment['DAYDREAM_RB_EVIDENCE']) / 'execution.json').read_text())
        assert record['status'] == 'completed_empty'
        assert record['configuration']['phases']['per_stack_review']['backend'] == 'codex'
        assert record['configuration']['phases']['per_stack_review']['model'] == 'gpt-6.1-sol'
        assert record['configuration']['phases']['per_stack_review']['reasoning_effort'] == 'high'
        assert json.loads(Path(environment['DAYDREAM_RB_OUTPUT']).read_text())['findings'] == []
        assert all(child.returncode == 0 for child in local.children)
        if diagnostic_failure in {'none', 'host_fallback_enospc'}:
            receipt = json.loads((logs / 'reviewer-process.json').read_text())
            assert receipt['cleanup_outcome'] == 'not_requested' and receipt['child_returncode'] == 0
            assert receipt['outer_deadline_seconds'] == 900
            assert receipt['outer_deadline_availability'] == 'declared trial config'
            assert receipt['terminal_evidence_available'] is True
        if protected is not None and destination is not None:
            assert protected.read_text() == 'prior competing diagnostic bytes'
            assert destination.read_text() == 'prior competing diagnostic bytes'
        captured = capfd.readouterr()
        all_output = captured.out + captured.err + ''.join(
            (result.stdout or '') + (result.stderr or '') for result in local.results)
        diagnostics = [line for line in all_output.splitlines() if line.startswith('ReviewBench ')]
        if diagnostic_failure != 'none':
            surface, error, code = {
                'host_fallback_enospc': ('host log', 'OSError', errno.ENOSPC),
                'closed_collector_pipe': ('process output', 'BrokenPipeError', errno.EPIPE),
                'supervisor_enospc': ('process', 'OSError', errno.ENOSPC),
            }.get(diagnostic_failure, ('process', 'FileExistsError', errno.EEXIST))
            assert diagnostics == [f'ReviewBench {surface} diagnostics unavailable ({error}; errno={code}).']
            assert sum(len(line.encode()) + 1 for line in diagnostics) < 1024
        else:
            assert diagnostics == []
        if foreign_stderr:
            assert foreign_stderr.decode() in all_output
        if entrypoint_payload:
            line = (logs / 'reviewer-entrypoint.log').read_text(encoding='utf-8').splitlines()[0]
            assert json.loads(line) == {'message': '\ud800'}
        for canary in ('publication-secret-canary', 'private-refresh-canary', 'private-id-canary',
                       'forbidden-api-canary', 'forbidden-github-canary'):
            assert canary not in all_output
    finally:
        await local.close()


@pytest.mark.parametrize('payload', [
    b'-----BEGIN PRIVATE KEY-----\nCOMPLETE_KEY_CANARY\n-----END PRIVATE KEY-----\n'
    b'-----BEGIN PRIVATE KEY-----\nINCOMPLETE_KEY_CANARY\n',
    b'{"refresh_token": "STRUCTURED_SECRET_CANARY"}\n',
    b'{"refresh_token": "INCOMPLETE_LINE_CANARY',
    b'Authorization: Bearer ' + b'A' * 65536 + b'BOUNDARY_SECRET_CANARY\n',
], ids=['complete_then_incomplete_pem', 'structured_secret', 'incomplete_line', 'oversized_credential'])
async def test_supervisor_captures_real_child_output_without_partial_credential_disclosure(payload: bytes) -> None:
    """Bounded capture cannot sever redaction context or leak an unfinished key.

    Native trajectory redaction tests do not own supervisor pipe buffering. A
    complete PEM before an unfinished PEM, chunk boundaries and a stream limit
    can expose credentials if raw bytes are truncated before redaction.
    """
    from daydream.benchmark.reviewbench.harbor_process import OUTPUT_LIMIT_BYTES, _capture, _safe_output

    child = await asyncio.create_subprocess_exec(sys.executable, '-I', '-c',
        'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); sys.stdout.buffer.flush()',
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
    assert child.stdout is not None and child.stdin is not None
    retained = bytearray()
    flags: dict[str, bool] = {}
    capture = asyncio.create_task(_capture(child.stdout, retained, flags))
    child.stdin.write(payload)
    child.stdin.close()
    await asyncio.wait_for(capture, timeout=5)
    await child.wait()
    safe = _safe_output(retained, flags)
    assert 'CANARY' not in safe
    assert len(safe.encode()) <= OUTPUT_LIMIT_BYTES
    assert 'REDACTED' in safe or flags.get('discarded') or flags.get('incomplete_line_discarded')


@pytest.mark.parametrize('failure', ['competing_log', 'log_symlink', 'link_failure', 'parent_symlink'])
def test_supervisor_publication_preserves_competing_bytes_on_real_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Diagnostic publication must not overwrite another invocation's evidence.

    Existing native export persistence tests do not exercise the supervisor's
    independent stdout/cleanup receipt. Inject at the actual hard-link operation.
    """
    from daydream.benchmark.reviewbench.harbor_process import _publish_process_evidence

    evidence = tmp_path / 'logs/daydream'
    evidence.parent.mkdir()
    log = evidence.parent / 'reviewer-entrypoint.log'
    protected = tmp_path / 'protected'
    protected.write_text('competing diagnostic bytes')
    if failure == 'competing_log':
        log.write_bytes(protected.read_bytes())
    elif failure == 'log_symlink':
        log.symlink_to(protected)
    elif failure == 'parent_symlink':
        evidence.parent.rmdir()
        protected_directory = tmp_path / 'protected-logs'
        protected_directory.mkdir()
        evidence.parent.symlink_to(protected_directory, target_is_directory=True)
    else:
        link = os.link

        def fail_link(source: Any, destination: Any, **kwargs: Any) -> None:
            if Path(destination).name == log.name:
                raise OSError(errno.ENOSPC, 'injected full destination filesystem')
            link(source, destination, **kwargs)

        monkeypatch.setattr(os, 'link', fail_link)
    with pytest.raises((OSError, ValueError)):
        _publish_process_evidence({'DAYDREAM_RB_EVIDENCE': str(evidence)}, b'new diagnostic', {'schema': 1})
    assert protected.read_text() == 'competing diagnostic bytes'
    if failure in {'competing_log', 'log_symlink'}:
        assert log.read_text() == 'competing diagnostic bytes'
    assert not (evidence.parent / 'reviewer-process.json').exists()
    assert not list(evidence.parent.glob('*.tmp'))


@pytest.mark.parametrize('unsafe', ['root_symlink', 'insecure_root', 'child_symlink'])
def test_invocation_admission_rejects_unsafe_storage_before_creating_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe: str,
) -> None:
    """Root validation must precede writes beneath redirected or nonprivate storage.

    Log/checkpoint symlink tests cover publication destinations, not invocation
    control-directory admission. Creating the child before validating the root
    writes outside the admitted owner even when the subsequent check rejects it.
    """
    from daydream.benchmark.reviewbench.harbor_process import invocation_directory

    external = tmp_path / 'external-owner'
    external.mkdir()
    root = tmp_path / 'daydream-reviewbench-invocations'
    invocation = str(uuid.uuid4())
    if unsafe == 'root_symlink':
        root.symlink_to(external, target_is_directory=True)
    else:
        root.mkdir(mode=0o700)
        if unsafe == 'insecure_root':
            root.chmod(0o755)
        else:
            (root / invocation).symlink_to(external, target_is_directory=True)
    monkeypatch.setattr('daydream.benchmark.reviewbench.harbor_process.tempfile.gettempdir', lambda: str(tmp_path))
    with pytest.raises(ValueError, match='not private'):
        invocation_directory(invocation)
    assert not list(external.iterdir())
    if unsafe == 'insecure_root':
        assert not list(root.iterdir())


async def test_optional_host_log_failure_preserves_actual_native_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str],
) -> None:
    """An optional fallback log must not replace the reviewer's actual failure.

    Successful-publication parity does not cover competing failures. Use real
    credential admission failure in the reviewer child and fault only the host's
    optional diagnostic hard link; native result/output publication stays real.
    """
    environment = public_input(tmp_path)
    (Path(environment['DAYDREAM_RB_CODEX_HOME']) / 'auth.json').unlink()
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', empty_reviews=True)
    local = LocalReviewerEnvironment({**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))},
        Path(environment['DAYDREAM_RB_REPOSITORY']))
    link = os.link

    def fail_host_log(source: Any, destination: Any, **kwargs: Any) -> None:
        if Path(destination).name == 'reviewer-entrypoint.log':
            raise OSError(errno.ENOSPC, 'publication-secret-canary')
        link(source, destination, **kwargs)

    monkeypatch.setattr(os, 'link', fail_host_log)
    try:
        with pytest.raises(RuntimeError, match='native reviewer failed'):
            await asyncio.wait_for(DaydreamPublicReviewAgent(logs_dir=tmp_path / 'host-logs').run(
                'frozen public task', local, SimpleNamespace()), timeout=10)
        assert local.children[0].returncode == 1
        assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
        diagnostic = capfd.readouterr().err
        assert 'diagnostics unavailable' in diagnostic and 'errno=28' in diagnostic
        assert 'publication-secret-canary' not in diagnostic
    finally:
        await local.close()


@pytest.mark.parametrize('failure', ['mkdir_enospc', 'metadata_enospc', 'reused', 'symlink'])
async def test_checkpoint_initialization_is_optional_only_for_operational_io_failure(
    tmp_path: Path, failure: str,
) -> None:
    """Optional checkpoint admission must not veto an otherwise healthy review.

    Successful-process parity faults only optional final supervisor files. It
    misses observer construction before runner admission. Fault actual mkdir or
    metadata hard-link operations in the executable, keeping runner, native CLI,
    Git and canonical publications real; conflicting paths remain strict.
    """
    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', empty_reviews=True)
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    checkpoints = evidence.with_name(evidence.name + '-checkpoints')
    protected = tmp_path / 'other-owner'
    if failure in {'reused', 'symlink'}:
        checkpoints.parent.mkdir(parents=True)
        protected.mkdir()
        (protected / 'preserve').write_text('other owner bytes')
        if failure == 'reused':
            checkpoints.mkdir()
            (checkpoints / 'preserve').write_text('prior checkpoint bytes')
        else:
            checkpoints.symlink_to(protected, target_is_directory=True)
    bootstrap = ('import errno,os,runpy\nfrom pathlib import Path\n'
        'original_mkdir,original_link=os.mkdir,os.link\n'
        'def mkdir(path,*args,**kwargs):\n'
        f'    if {failure == "mkdir_enospc"!r} and Path(path).name=="daydream-checkpoints":\n'
        '        raise OSError(errno.ENOSPC,"checkpoint failure secret")\n'
        '    return original_mkdir(path,*args,**kwargs)\n'
        'def link(source,destination,**kwargs):\n'
        f'    if {failure == "metadata_enospc"!r} and Path(destination).name=="metadata.json":\n'
        '        raise OSError(errno.ENOSPC,"checkpoint failure secret")\n'
        '    return original_link(source,destination,**kwargs)\n'
        'os.mkdir,os.link=mkdir,link\n'
        'runpy.run_module("daydream.benchmark.reviewbench.harbor_entrypoint",run_name="__main__")\n')
    child = await asyncio.create_subprocess_exec(sys.executable, '-I', '-c', bootstrap,
        cwd=environment['DAYDREAM_RB_REPOSITORY'],
        env={**os.environ, **environment, 'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=30)
        if failure in {'reused', 'symlink'}:
            assert child.returncode == 1
            assert not cli.observations(), 'unsafe checkpoint path reached native inference'
            assert not evidence.exists() and not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
            assert (protected / 'preserve').read_text() == 'other owner bytes'
            if failure == 'reused':
                assert (checkpoints / 'preserve').read_text() == 'prior checkpoint bytes'
            return
        assert child.returncode == 0, stderr.decode()
        record = json.loads((evidence / 'execution.json').read_text())
        assert record['status'] == 'completed_empty' and cli.observations()
        assert record['checkpoint_error'] == 'checkpoint initialization unavailable: OSError errno=28'
        assert json.loads(Path(environment['DAYDREAM_RB_OUTPUT']).read_text())['findings'] == []
        assert 'diagnostics unavailable' in stderr.decode() and 'errno=28' in stderr.decode()
        assert 'checkpoint failure secret' not in (stdout + stderr).decode()
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


@pytest.mark.parametrize('shutdown', ['graceful', 'repeated_signal', 'force'])
async def test_sigterm_joins_real_finalization_writer_before_export_or_uses_checkpoint(
    tmp_path: Path, shutdown: str,
) -> None:
    """SIGTERM must respect the runner's shield around an actual filesystem writer.

    The existing interrupted-review test cancels native inference before the
    finalization thread exists. Block an actual atomic install in that thread,
    then drive the production host/supervisor/signal path; no runner or finalizer
    result is replaced. A stuck writer must still meet the supervisor force bound.
    """
    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', empty_reviews=True)
    ready, signalled, exported, joined = [tmp_path / name for name in
        ('writer-ready.json', 'signal-received', 'export-observation.json', 'writer-joined')]
    release = tmp_path / 'release-writer'
    os.mkfifo(release, mode=0o600)
    bootstrap = f'''import asyncio,json,os,threading
from pathlib import Path
from daydream import runner
from daydream.benchmark.reviewbench import harbor_entrypoint as entry
ready,signalled,exported,joined,release=map(Path,{[str(p) for p in (ready, signalled, exported, joined, release)]!r})
replace=os.replace
blocked=False
def install(source,destination,*args,**kwargs):
 global blocked
 if (not blocked and threading.current_thread().name.startswith('AnyIO')
     and Path(destination).name=='publish-manifest.json'):
  blocked=True
  ready.write_text(json.dumps({{'pid':os.getpid()}}))
  with release.open('rb') as stream: assert stream.read(1)==b'R'
 return replace(source,destination,*args,**kwargs)
os.replace=install
finalize=runner._finalize_run_artifacts
def observe_finalize(*args,**kwargs):
 try: return finalize(*args,**kwargs)
 finally: joined.write_text('writer returned')
runner._finalize_run_artifacts=observe_finalize
publish=entry._publish_native_evidence
def observe_export(*args,**kwargs):
 exported.write_text(json.dumps({{'writer_joined':joined.exists()}}))
 return publish(*args,**kwargs)
entry._publish_native_evidence=observe_export
async def drive():
 loop=asyncio.get_running_loop()
 register=loop.add_signal_handler
 def observe_signal(number,callback,*args):
  def received():
   signalled.write_text('signal observed')
   callback(*args)
  register(number,received)
 loop.add_signal_handler=observe_signal
 return await entry._executable_main()
raise SystemExit(asyncio.run(drive()))
'''
    local = LocalReviewerEnvironment({**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))},
        Path(environment['DAYDREAM_RB_REPOSITORY']), entrypoint_bootstrap=bootstrap)
    logs = Path(environment['DAYDREAM_RB_EVIDENCE']).parent
    running = asyncio.create_task(DaydreamPublicReviewAgent(logs_dir=logs).run('frozen task', local, SimpleNamespace()))
    try:
        deadline = asyncio.get_running_loop().time() + 15
        while not ready.exists():
            if running.done():
                await running
            assert asyncio.get_running_loop().time() < deadline, 'actual finalization install never reached readiness'
            await asyncio.sleep(0.01)
        running.cancel()
        while not signalled.exists():
            assert asyncio.get_running_loop().time() < deadline, 'reviewer executable never received SIGTERM'
            await asyncio.sleep(0.01)
        if shutdown == 'repeated_signal':
            os.kill(json.loads(ready.read_text())['pid'], signal.SIGTERM)
        if shutdown != 'force':
            # Give an incorrectly abandoned writer a chance to reach export;
            # the final assertion also observes the complete writer returning.
            observe_until = asyncio.get_running_loop().time() + 0.2
            while not exported.exists() and asyncio.get_running_loop().time() < observe_until:
                await asyncio.sleep(0.01)
            assert not exported.exists(), 'native export began before its finalization writer joined'
            with release.open('wb') as stream:
                stream.write(b'R')
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, timeout=35)
        process = json.loads((logs / 'reviewer-process.json').read_text())
        evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
        assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
        if shutdown == 'force':
            assert process['cleanup_outcome'] == 'forced_termination'
            assert process['terminal_evidence_available'] is False
            assert not exported.exists() and not joined.exists()
            from daydream.benchmark.reviewbench.diagnostics import interrupted_readout

            retained = interrupted_readout(evidence)
            assert isinstance(retained['checkpoint'], dict)
            assert retained['checkpoint']['kind'] == 'checkpoint'
            assert len(retained['checkpoint']['documents']) >= 3
            assert retained['terminal_evidence'] == 'unavailable'
            assert retained['unavailable_checkpoints'] == []
            return
        assert process['cleanup_outcome'] == 'graceful_exit' and process['child_returncode'] == 130
        assert json.loads(exported.read_text()) == {'writer_joined': True}
        root = json.loads((evidence / 'trajectory.json').read_text())
        assert root['extra']['subtrajectories']
        for document in root['extra']['subtrajectories']:
            if reference := document.get('sibling_trajectory_ref'):
                assert (evidence / reference).is_file()
        execution = json.loads((evidence / 'execution.json').read_text())
        assert execution['interruption'] == 'reviewer_task_cancelled' and execution['status'] == 'cancelled'
        assert execution['candidate'] is None
    finally:
        try:
            descriptor = os.open(release, os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            pass
        else:
            os.write(descriptor, b'R')
            os.close(descriptor)
        if not running.done():
            running.cancel()
        await local.close()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.parametrize('surface', ['receipt', 'private_stop', 'shared_stop'])
@pytest.mark.parametrize('storage', [
    'regular', 'wrong_invocation', 'fifo', 'oversize', 'leaf_symlink', 'parent_symlink',
])
async def test_process_control_observation_requires_bounded_owned_regular_files(
    tmp_path: Path, surface: str, storage: str,
) -> None:
    """A diagnostic/control read cannot follow another owner or block shutdown.

    Real interruption tests publish healthy regular receipts/markers and cannot
    detect a FIFO poll hanging the supervisor, redirected parent traversal, or a
    huge receipt falsely completing failed-collector cleanup. Run the actual
    readers in a bounded subprocess so a regressed blocking read fails safely.
    """
    invocation = str(uuid.uuid4())
    real = tmp_path / 'real'
    directory = real / 'owner'
    directory.mkdir(parents=True)
    if storage == 'parent_symlink':
        alias = tmp_path / 'alias'
        alias.symlink_to(real, target_is_directory=True)
        directory = alias / 'owner'
    marker_id = str(uuid.uuid4()) if storage == 'wrong_invocation' else invocation
    body = {'schema': 1, 'invocation_id': marker_id, 'child_returncode': 130}
    if surface != 'receipt':
        body = {'invocation_id': marker_id, 'interruption': 'host_execution_failed'}
    name = ('reviewer-process.json' if surface == 'receipt' else 'cancel' if surface == 'private_stop'
            else 'reviewer-cancel-' + invocation + '.json')
    path = directory / name
    if storage == 'fifo':
        os.mkfifo(path, mode=0o600)
    elif storage == 'leaf_symlink':
        target = tmp_path / 'other-owner.json'
        target.write_text(json.dumps(body))
        path.symlink_to(target)
    else:
        path.write_text(json.dumps(body) + (' ' * 65537 if storage == 'oversize' else ''))
    if surface == 'receipt':
        program = ('from daydream.benchmark.reviewbench.harbor_agent import _process_completion_observed\n'
            f'value = _process_completion_observed(Path({str(directory)!r}), {invocation!r})\n')
    else:
        private = directory if surface == 'private_stop' else tmp_path / 'unused-private'
        evidence = directory / 'daydream' if surface == 'shared_stop' else tmp_path / 'unused-shared/daydream'
        program = ('from daydream.benchmark.reviewbench.harbor_process import _requested_interruption\n'
            f'value = _requested_interruption(Path({str(private)!r}), '
            f'{{"DAYDREAM_RB_EVIDENCE": {str(evidence)!r}}}, {invocation!r})\n')
    expected = storage == 'regular'
    child = await asyncio.create_subprocess_exec(sys.executable, '-I', '-c',
        'from pathlib import Path\n' + program + f'assert bool(value) is {expected!r}, value\n',
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        _, stderr = await asyncio.wait_for(child.communicate(), timeout=3)
        assert child.returncode == 0, stderr.decode()
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


def test_owned_stop_tombstone_preserves_first_reason_and_rejects_competing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Competing control bytes cannot be accepted through the other stop channel.

    Process cancellation tests have a fresh invocation. Admission must preserve
    its first valid reason without overwriting bytes, while rejecting a different
    owner's invalid private tombstone.
    """
    from daydream.benchmark.reviewbench.harbor_process import invocation_directory, request_cancellation

    monkeypatch.setattr('daydream.benchmark.reviewbench.harbor_process.tempfile.gettempdir', lambda: str(tmp_path))
    invocation = str(uuid.uuid4())
    request_cancellation(invocation, 'host_execution_failed')
    tombstone = invocation_directory(invocation) / 'cancel'
    first = tombstone.read_bytes()
    request_cancellation(invocation)
    assert tombstone.read_bytes() == first
    tombstone.write_bytes(b'competing owner bytes')
    with pytest.raises(ValueError, match='marker conflicts'):
        request_cancellation(invocation)
    assert tombstone.read_bytes() == b'competing owner bytes'
