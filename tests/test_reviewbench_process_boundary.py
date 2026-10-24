"""Real host -> reviewer process -> native CLI interruption contract."""
from __future__ import annotations

import asyncio
import errno
import json
import os
import signal
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.benchmark.reviewbench.harbor_agent import DaydreamPublicReviewAgent
from tests.harness.codex_lifecycle_cli import install_codex_lifecycle_cli
from tests.harness.reviewbench import LocalReviewerEnvironment, public_input


@pytest.mark.parametrize('interruption', [
    'timeout', 'cancel', 'repeated_cancel', 'force', 'late_child', 'slow_process_listing',
    'collector_timeout', 'collector_transport', 'collector_repeated_cancel',
])
async def test_outer_timeout_stops_owned_reviewer_and_retains_process_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str,
) -> None:
    if interruption == 'late_child' and sys.platform != 'linux':
        pytest.skip('late orphan containment requires Linux PR_SET_CHILD_SUBREAPER')
    if interruption == 'slow_process_listing' and sys.platform != 'darwin':
        pytest.skip('macOS process-listing transport regression')
    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', structure_mode='preterminal_silence',
        spawn_on_term=interruption == 'late_child')
    # Exercise rotation in the actual native subprocess, before it reaches the
    # cancellation rendezvous, rather than supplying a pre-finalized auth result.
    native_home = cli.root / 'native-home'
    executable = cli.bin_dir / 'codex'
    header, body = executable.read_text().split('\n', 1)
    executable.write_text(header + '\n' +
        'import json,os\nfrom pathlib import Path\n'
        'home=Path(os.environ["CODEX_HOME"])\n'
        'auth=json.loads((home/"auth.json").read_text())\n'
        'auth["tokens"]["refresh_token"]="private-cancelled-rotation"\n'
        '(home/"auth.json").write_text(json.dumps(auth))\n' +
        f'Path({str(native_home)!r}).write_text(str(home))\n' + body)
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
    class CollectorFailureEnvironment(LocalReviewerEnvironment):
        async def exec(self, command: str, **kwargs: Any) -> Any:
            if interruption.startswith('collector_'):
                # A transport failure loses the Docker client's await without
                # killing the separate in-container supervisor/native process.
                detached.append(asyncio.create_task(super().exec(command, **kwargs)))
                await collector_release.wait()
                raise collector_failure
            return await super().exec(command, **kwargs)

    local = CollectorFailureEnvironment(child_environment, Path(environment['DAYDREAM_RB_REPOSITORY']))
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
        assert host_receipt['cancellation_channels'] == {'shared_marker': 'published'}
        if interruption == 'force':
            assert not (evidence / 'execution.json').exists()
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
        assert root['extra']['partial'] is True
        assert any(event['event'] == 'phase_end' and event.get('status') == 'cancelled'
                   for event in root['extra']['phase_events'])
        documents = [json.loads((evidence / row['sibling_trajectory_ref']).read_bytes()) for row in siblings]
        assert all(document['session_id'] == root['session_id'] for document in documents)
        assert any(step['source'] == 'agent' and step.get('extra', {}).get('stop_reason') == 'cancelled'
                   for document in documents for step in document['steps'])
        assert (logs / 'trajectory.json').read_bytes() == (evidence / 'trajectory.json').read_bytes()
        assert not Path(native_home.read_text()).exists()
        from daydream.benchmark.reviewbench.harbor_auth import reviewer_credentials

        canonical = Path(environment['DAYDREAM_RB_CODEX_HOME'])
        with reviewer_credentials(canonical, tmp_path / 'after-cancel') as home:
            assert json.loads((home / 'auth.json').read_text())['tokens']['refresh_token'] == (
                'private-cancelled-rotation')
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


@pytest.mark.parametrize('credential_type,tool_output_form', [
    ('api_key', 'mapping'), ('oauth', 'mapping'), ('api_key', 'quoted_log'), ('oauth', 'quoted_log'),
])
async def test_codex_tool_output_of_admitted_pi_auth_is_redacted_in_interrupted_bundle(
    tmp_path: Path, credential_type: str, tool_output_form: str,
) -> None:
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
    tool_input = credential
    if tool_output_form == 'quoted_log':
        tool_input = tmp_path / 'quoted-tool-output.json'
        tool_input.write_text(json.dumps('Read ' + json.dumps('data: ' + json.dumps(auth))))
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', structure_mode='preterminal_silence',
        empty_reviews=True, tool_output_path=tool_input)
    local = LocalReviewerEnvironment({**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))},
        Path(environment['DAYDREAM_RB_REPOSITORY']))
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    running = asyncio.create_task(DaydreamPublicReviewAgent(logs_dir=evidence.parent).run(
        'frozen synthetic task', local, SimpleNamespace()))
    try:
        # Cold reviewer imports and snapshot setup are outside the cancellation
        # contract below; allow parallel full-suite startup before timing it.
        deadline = asyncio.get_running_loop().time() + 60
        while not any(row['mode'] == 'preterminal_silence' for row in cli.observations()):
            if running.done():
                await running
            assert asyncio.get_running_loop().time() < deadline, 'native tool process did not reach readiness'
            await asyncio.sleep(0.01)
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
        documents = [json.loads((evidence / reference).read_bytes()) for reference in siblings]
        tool_payloads = [json.loads(result['content'])
                         for document in documents for step in document['steps']
                         for result in step.get('observation', {}).get('results', [])
                         if result['source_call_id'] == 'fixture-input']
        assert tool_payloads
        for payload in tool_payloads:
            assert payload['public'] == {'type': 'record', 'key': 'ordinary-public-key'}
            assert payload['usage'] == {'input': 0, 'output': None}
            data = payload['data']
            if tool_output_form == 'quoted_log':
                assert isinstance(data, str) and data.startswith('Read ')
                data = json.loads(data.removeprefix('Read '))
                assert isinstance(data, str) and data.startswith('data: ')
                data = json.loads(data.removeprefix('data: '))
            admitted = data['nous' if credential_type == 'api_key' else 'openai-codex']
            assert admitted['type'] == credential_type
            if credential_type == 'oauth':
                assert admitted['accountId'] == 'synthetic-public-account' and admitted['expires'] == 9000000000000
        # The supervisor signals the reviewer and native CLI independently.
        # Native exit may be recorded as a backend error before cancellation
        # reaches the reviewer; retain that truthful attempt outcome.
        interrupted = next(document for document in documents
            if any(result['source_call_id'] == 'fixture-input' for step in document['steps']
                   for result in step.get('observation', {}).get('results', []))
            and any(step.get('extra', {}).get('stop_reason') == 'cancelled'
                    or step.get('extra', {}).get('error_subtype') == 'CodexError' for step in document['steps']))
        final_invocation = interrupted['extra']['subtrajectories'][-1]
        final_lifecycle = final_invocation['backend_lifecycle']
        assert final_invocation['ended_at'] is not None and final_lifecycle['teardown_completed_s'] is not None
        assert final_lifecycle['process_returncode'] is not None
        assert final_lifecycle['spawn_completed_s'] is not None and final_lifecycle['last_native_event_s'] is not None
        assert final_lifecycle['last_native_event_kind'] == 'item_completed'
        if final_invocation.get('host_dispatch'):
            assert final_invocation['host_dispatch']['stage_scope_id']
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


@pytest.mark.skipif(sys.platform != 'linux', reason='late orphan containment requires Linux PR_SET_CHILD_SUBREAPER')
@pytest.mark.parametrize('process_table_failure', [False, True])
async def test_forced_reviewer_exit_reaps_late_descendant_holding_output(
    tmp_path: Path, process_table_failure: bool,
) -> None:
    """A fork after the final ownership snapshot must not block reap on inherited pipes."""
    entrypoint = '''import os,signal,time
from pathlib import Path
root=Path(os.environ['CONTROL_ROOT'])
def fork_late(signum,frame):
 pid=os.fork()
 if pid==0:
  os.setsid()
  signal.signal(signal.SIGUSR1,signal.SIG_IGN)
  (root/'orphan').write_text(str(os.getpid()))
  while True: time.sleep(1)
 while not (root/'orphan').exists(): time.sleep(0.001)
signal.signal(signal.SIGTERM,signal.SIG_IGN)
signal.signal(signal.SIGUSR1,fork_late)
(root/'ready').write_text(str(os.getpid()))
while True: time.sleep(1)
'''
    bootstrap = f'''import asyncio,errno,json,os,signal,sys,time,uuid
from pathlib import Path
from daydream.benchmark.reviewbench import harbor_process as owner
root=Path({str(tmp_path)!r})
async def drive():
 invocation=str(uuid.uuid4())
 environment={{**os.environ,'DAYDREAM_RB_EVIDENCE':str(root/'logs/daydream'),'CONTROL_ROOT':str(root)}}
 original_spawn=asyncio.create_subprocess_exec
 original_listdir=os.listdir
 fail_listing=False
 def listdir(path='.'):
  nonlocal fail_listing
  if path==b'/proc' and fail_listing:
   fail_listing=False
   raise OSError(errno.EIO,'scripted process table read failure')
  return original_listdir(path)
 os.listdir=listdir
 async def spawn(*args,**kwargs):
  child=await original_spawn(sys.executable,'-I','-c',{entrypoint!r},**kwargs)
  original_kill=child.kill
  injected=False
  def kill():
   nonlocal injected,fail_listing
   if not injected:
    injected=True
    os.kill(child.pid,signal.SIGUSR1)
    deadline=time.monotonic()+5
    while not (root/'orphan').exists():
     if time.monotonic()>=deadline: raise RuntimeError('late fork never became ready')
     time.sleep(0.001)
    original_kill()
    fail_listing={process_table_failure!r}
   else: original_kill()
  child.kill=kill
  return child
 asyncio.create_subprocess_exec=spawn
 owner.SHUTDOWN_GRACE_SECONDS=0.1
 async def cancel_when_ready():
  while not (root/'ready').exists(): await asyncio.sleep(0.001)
  owner.publish_diagnostic_bytes(owner.cancellation_marker(root/'logs',invocation),json.dumps({{
   'invocation_id':invocation,'interruption':'host_task_cancelled'}}).encode())
 stopping=asyncio.create_task(cancel_when_ready())
 try:
  await owner.supervise(invocation,environment)
  await stopping
  orphan=int((root/'orphan').read_text())
  try:
   os.kill(orphan,0)
   orphan_alive=True
  except ProcessLookupError: orphan_alive=False
  print(json.dumps({{'orphan_alive':orphan_alive}}))
 finally:
  if (root/'orphan').exists():
   orphan=int((root/'orphan').read_text())
   try:
    os.kill(orphan,signal.SIGKILL)
    os.waitpid(orphan,0)
   except ProcessLookupError: pass
  stopping.cancel()
  await asyncio.gather(stopping,return_exceptions=True)
asyncio.run(drive())
'''
    child = await asyncio.create_subprocess_exec(sys.executable, '-I', '-c', bootstrap,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=15)
        assert child.returncode == 0, stderr.decode()
        assert json.loads(stdout)['orphan_alive'] is False
        receipt = json.loads((tmp_path / 'logs/reviewer-process.json').read_text())
        assert receipt['cleanup_outcome'] == ('supervisor_failed' if process_table_failure else 'forced_termination')
        assert receipt['owned_descendants_cleanup'] == 'joined'
        assert receipt['child_returncode'] == -signal.SIGKILL
        assert receipt['terminal_evidence_available'] is False
    finally:
        for name in ('orphan', 'ready'):
            marker = tmp_path / name
            if marker.exists():
                try:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if child.returncode is None:
            os.killpg(child.pid, signal.SIGKILL)
            await child.wait()


@pytest.mark.parametrize('receipt_conflict', [False, True])
async def test_cancelled_host_before_reviewer_start_cannot_launch_native_inference(
    tmp_path: Path, receipt_conflict: bool,
) -> None:
    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='preterminal_silence')
    child_environment = {**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))}
    launch, entered = asyncio.Event(), asyncio.Event()

    class DelayedEnvironment(LocalReviewerEnvironment):
        async def exec(self, command: str, **kwargs: Any) -> Any:
            entered.set()
            await launch.wait()
            return await super().exec(command, **kwargs)

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
        deadline = asyncio.get_running_loop().time() + 5
        while not list(logs.glob('reviewer-cancel-*.json')):
            assert asyncio.get_running_loop().time() < deadline, 'host did not publish startup cancellation'
            await asyncio.sleep(0.01)
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


@pytest.mark.parametrize('diagnostic_failure,foreign_stderr,entrypoint_payload,declared_timeout,transport_timeout', [
    pytest.param('none', b'', b'', 900, 932, id='none'),
    pytest.param('none', b'', b'', 3601.25, 3634, id='long_outer_timeout'),
    pytest.param('none', b'', b'', None, None, id='unknown_outer_timeout'),
    pytest.param('log_conflict', b'', b'', 900, 932, id='log_conflict'),
    pytest.param('receipt_conflict', b'', b'', 900, 932, id='receipt_conflict'),
    pytest.param('log_symlink', b'', b'', 900, 932, id='log_symlink'),
    pytest.param('receipt_symlink', b'', b'', 900, 932, id='receipt_symlink'),
    pytest.param('supervisor_enospc', b'', b'', 900, 932, id='supervisor_enospc'),
    pytest.param('log_enospc', b'', b'', 900, 932, id='log_enospc'),
    pytest.param('log_conflict', b'External runtime warning: ' + b'x' * 2048 + b'\n', b'', 900, 932,
                 id='foreign_stderr'),
    pytest.param('none', b'', b'{"message":"\\ud800"}\n', 900, 932, id='surrogate_json'),
    pytest.param('none', b'', b'-----BEGIN PRIVATE KEY-----\nCOMPLETE_KEY_CANARY\n-----END PRIVATE KEY-----\n'
                 b'-----BEGIN PRIVATE KEY-----\nINCOMPLETE_KEY_CANARY\n', 900, 932, id='complete_then_incomplete_pem'),
    pytest.param('none', b'', b'{"refresh_token": "STRUCTURED_SECRET_CANARY"}\n', 900, 932, id='structured_secret'),
    *[pytest.param('none', b'', ('Read ' + json.dumps('data: ' + json.dumps(credential)) + ' trailing prose\n')
                   .encode(), 900, 932, id='quoted_' + kind)
      for kind, credential in [
          ('api_key', {'type': 'api_key', 'key': 'QUOTED_KEY_CANARY'}),
          ('oauth', {'type': 'oauth', 'access': 'QUOTED_ACCESS_CANARY', 'refresh': 'QUOTED_REFRESH_CANARY'}),
      ]],
    pytest.param('none', b'', b'{"refresh_token": "INCOMPLETE_LINE_CANARY', 900, 932, id='incomplete_line'),
    pytest.param('none', b'', b'Authorization: Bearer ' + b'A' * 65536 + b'BOUNDARY_SECRET_CANARY\n', 900, 932,
                 id='oversized_credential'),
])
async def test_completed_process_review_keeps_native_result_despite_diagnostic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str], diagnostic_failure: str,
    foreign_stderr: bytes, entrypoint_payload: bytes, declared_timeout: float | None, transport_timeout: int | None,
) -> None:
    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', empty_reviews=True)
    if declared_timeout is not None:
        declared_task = tmp_path / 'declared-task'
        declared_task.mkdir()
        (declared_task / 'task.toml').write_text(f'[agent]\ntimeout_sec = {declared_timeout}\n')
        (tmp_path / 'config.json').write_text(json.dumps({'task': {'path': str(declared_task)}, 'agent': {}}))
    received_timeouts: list[int | None] = []

    class DeadlineEnvironment(LocalReviewerEnvironment):
        async def exec(self, command: str, **kwargs: Any) -> Any:
            received_timeouts.append(kwargs.get('timeout_sec'))
            return await super().exec(command, **kwargs)

    local = DeadlineEnvironment({**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))},
        Path(environment['DAYDREAM_RB_REPOSITORY']),
        publication_fault={'supervisor_enospc': 'reviewer-process.json',
                           'log_enospc': 'reviewer-entrypoint.log'}.get(diagnostic_failure),
        foreign_stderr=foreign_stderr,
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
    try:
        await asyncio.wait_for(DaydreamPublicReviewAgent(logs_dir=logs).run(
            'frozen public task', local, SimpleNamespace()), timeout=30)
        record = json.loads((Path(environment['DAYDREAM_RB_EVIDENCE']) / 'execution.json').read_text())
        assert record['status'] == 'completed_empty'
        assert received_timeouts == [transport_timeout]
        assert json.loads(Path(environment['DAYDREAM_RB_OUTPUT']).read_text())['findings'] == []
        assert all(child.returncode == 0 for child in local.children)
        assert not list(logs.rglob('*checkpoint*')), 'fresh completed run created interruption checkpoints'
        if diagnostic_failure == 'none':
            receipt = json.loads((logs / 'reviewer-process.json').read_text())
            assert receipt['cleanup_outcome'] == 'not_requested' and receipt['child_returncode'] == 0
            assert receipt['outer_deadline_seconds'] == declared_timeout
            assert receipt['outer_deadline_availability'] == (
                'declared trial config' if declared_timeout is not None else 'unavailable')
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
                'supervisor_enospc': ('process', 'OSError', errno.ENOSPC),
                'log_enospc': ('process', 'OSError', errno.ENOSPC),
            }.get(diagnostic_failure, ('process', 'FileExistsError', errno.EEXIST))
            assert diagnostics == [f'ReviewBench {surface} diagnostics unavailable ({error}; errno={code}).']
            assert sum(len(line.encode()) + 1 for line in diagnostics) < 1024
        else:
            assert diagnostics == []
        if foreign_stderr:
            assert foreign_stderr.decode() in all_output
        if entrypoint_payload:
            from daydream.benchmark.reviewbench.harbor_process import OUTPUT_LIMIT_BYTES

            retained = (logs / 'reviewer-entrypoint.log').read_bytes()
            assert len(retained) <= 2 * OUTPUT_LIMIT_BYTES
            if b'ud800' in entrypoint_payload:
                assert json.loads(retained.decode().splitlines()[0]) == {'message': '\ud800'}
            elif b'INCOMPLETE_LINE_CANARY' in entrypoint_payload:
                assert receipt['stdout_capture']['incomplete_line_discarded']
            elif len(entrypoint_payload) > OUTPUT_LIMIT_BYTES:
                assert receipt['stdout_capture']['discarded'] and receipt['output_truncated']
            else:
                assert b'REDACTED' in retained
            assert b'CANARY' not in retained
            assert 'CANARY' not in all_output
        for canary in ('publication-secret-canary', 'private-refresh-canary', 'private-id-canary',
                       'forbidden-api-canary', 'forbidden-github-canary'):
            assert canary not in all_output
    finally:
        await local.close()


def test_supervisor_publication_refuses_redirected_log_parent(tmp_path: Path) -> None:
    from daydream.benchmark.reviewbench.harbor_process import _publish_process_evidence

    protected = tmp_path / 'other-owner'
    protected.mkdir()
    (protected / 'preserve').write_text('competing diagnostic bytes')
    logs = tmp_path / 'logs'
    logs.symlink_to(protected, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        _publish_process_evidence({'DAYDREAM_RB_EVIDENCE': str(logs / 'daydream')}, b'new diagnostic', {'schema': 1})
    assert [p.name for p in protected.iterdir()] == ['preserve']
    assert (protected / 'preserve').read_text() == 'competing diagnostic bytes'


async def test_optional_process_log_failure_preserves_actual_native_failure(tmp_path: Path) -> None:
    environment = public_input(tmp_path)
    (Path(environment['DAYDREAM_RB_CODEX_HOME']) / 'auth.json').unlink()
    local = LocalReviewerEnvironment({**os.environ, **environment},
        Path(environment['DAYDREAM_RB_REPOSITORY']), publication_fault='reviewer-entrypoint.log')
    try:
        with pytest.raises(RuntimeError, match='native reviewer failed'):
            await asyncio.wait_for(DaydreamPublicReviewAgent(
                logs_dir=Path(environment['DAYDREAM_RB_EVIDENCE']).parent).run(
                'frozen public task', local, SimpleNamespace()), timeout=10)
        assert local.children[0].returncode == 1
        assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
        diagnostic = ''.join(result.stderr for result in local.results)
        assert 'diagnostics unavailable' in diagnostic and 'errno=28' in diagnostic
        assert 'publication-secret-canary' not in diagnostic
    finally:
        await local.close()


@pytest.mark.parametrize('shutdown', ['graceful', 'repeated_signal', 'force'])
async def test_sigterm_joins_real_finalization_writer_before_export_or_reports_unavailable_evidence(
    tmp_path: Path, shutdown: str,
) -> None:
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
            assert not (evidence / 'execution.json').exists()
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


@pytest.mark.parametrize('surface', ['receipt', 'shared_stop'])
@pytest.mark.parametrize('storage', [
    'regular', 'wrong_invocation', 'fifo', 'oversize', 'leaf_symlink', 'parent_symlink',
])
async def test_process_control_observation_requires_bounded_owned_regular_files(
    tmp_path: Path, surface: str, storage: str,
) -> None:
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
    name = 'reviewer-process.json' if surface == 'receipt' else 'reviewer-cancel-' + invocation + '.json'
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
        imports = 'from daydream.benchmark.reviewbench.harbor_agent import _process_completion_observed\n'
        observation = f'value = _process_completion_observed(Path({str(directory)!r}), {invocation!r})\n'
    else:
        evidence = directory / 'daydream'
        imports = 'from daydream.benchmark.reviewbench.harbor_process import _requested_interruption\n'
        observation = (f'value = _requested_interruption('
            f'{{"DAYDREAM_RB_EVIDENCE": {str(evidence)!r}}}, {invocation!r})\n')
    expected = storage == 'regular'
    child = await asyncio.create_subprocess_exec(sys.executable, '-I', '-c',
        'from pathlib import Path\n' + imports + 'print("ready", flush=True)\ninput()\n' + observation
        + f'assert bool(value) is {expected!r}, value\n',
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        # Bound diagnostic I/O separately from cold imports under parallel suite load.
        assert child.stdout is not None
        assert await asyncio.wait_for(child.stdout.readline(), timeout=15) == b'ready\n'
        _, stderr = await asyncio.wait_for(child.communicate(b'\n'), timeout=3)
        assert child.returncode == 0, stderr.decode()
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
