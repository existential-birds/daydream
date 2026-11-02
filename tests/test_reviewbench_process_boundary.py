"""Real host -> reviewer process -> native CLI interruption contract."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.benchmark.reviewbench.harbor_agent import DaydreamPublicReviewAgent
from tests.harness.codex_lifecycle_cli import install_codex_lifecycle_cli
from tests.harness.reviewbench import LocalReviewerEnvironment, public_input


async def test_collector_failure_and_repeated_cancel_stop_owned_reviewer_and_retain_process_evidence(
    tmp_path: Path,
) -> None:
    """A lost Docker exec await plus repeated host cancellation still joins the detached reviewer.

    The native process rotates its credential before the cancellation rendezvous; the
    rotation survives, every owned process is gone, and only cancelled evidence remains.
    """
    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', structure_mode='preterminal_silence')
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
    logs = Path(environment['DAYDREAM_RB_EVIDENCE']).parent
    child_environment = {**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))}
    collector_failure = ConnectionError('collector failure secret')
    collector_release = asyncio.Event()
    detached: list[asyncio.Task[Any]] = []

    class CollectorFailureEnvironment(LocalReviewerEnvironment):
        async def exec(self, command: str, **kwargs: Any) -> Any:
            # A transport failure loses the Docker client's await without
            # killing the separate in-container supervisor/native process.
            detached.append(asyncio.create_task(super().exec(command, **kwargs)))
            await collector_release.wait()
            raise collector_failure

    local = CollectorFailureEnvironment(child_environment, Path(environment['DAYDREAM_RB_REPOSITORY']))
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
        collector_release.set()
        while not list(logs.glob('reviewer-cancel-*.json')) and not running.done():
            await asyncio.sleep(0.01)
        running.cancel()
        await asyncio.sleep(0.01)
        running.cancel()
        with pytest.raises(ConnectionError) as raised:
            await asyncio.wait_for(running, timeout=30)
        assert raised.value is collector_failure
        await asyncio.wait_for(asyncio.gather(*detached), timeout=5)
        receipt = json.loads((logs / 'reviewer-process.json').read_text())
        assert receipt['interruption'] == 'host_execution_failed'
        assert receipt['outer_deadline_seconds'] is None
        assert receipt['cleanup_outcome'] == 'graceful_exit'
        host_receipt = json.loads((logs / 'reviewer-host.json').read_text())
        assert host_receipt['interruption'] == 'host_execution_failed'
        assert host_receipt['output_collection'] == 'unavailable'
        assert host_receipt['execution_error'] == 'ConnectionError'
        assert host_receipt['process_completion'] == 'observed'
        assert 'collector failure secret' not in (logs / 'reviewer-host.json').read_text()
        assert host_receipt['cancellation_channels'] == {'shared_marker': 'published'}
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
        for row in cli.observations():
            try:
                os.killpg(row['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
        await local.close()
        await asyncio.gather(*detached, return_exceptions=True)
        await asyncio.gather(running, return_exceptions=True)


async def test_host_cancellation_redacts_admitted_pi_auth_from_interrupted_bundle(tmp_path: Path) -> None:
    """Harbor cancelling the agent (its trial timeout) records host cancellation and exports a redacted bundle.

    An admitted credential echoed through real native tool output never reaches retained
    evidence or collected output, while public payload fields survive.
    """
    from daydream.benchmark.reviewbench.harbor_auth import validate_pi_reviewer_home

    environment = public_input(tmp_path)
    pi_home = tmp_path / 'synthetic-pi-auth'
    pi_home.mkdir(mode=0o700)
    secret = 'opaque-reviewbench-pi-auth-secret-canary'
    credential = pi_home / 'auth.json'
    auth = {'nous': {'type': 'api_key', 'key': secret}}
    original_auth = json.dumps(auth).encode()
    credential.write_bytes(original_auth)
    validate_pi_reviewer_home(pi_home, 'nous')
    # Format combinations belong to test_redaction; this case proves the real
    # native tool -> interrupted trajectory -> exported bundle wiring.
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
        receipt = json.loads((evidence.parent / 'reviewer-process.json').read_text())
        assert receipt['interruption'] == 'host_cancellation' and receipt['cleanup_outcome'] == 'graceful_exit'
        host_receipt = json.loads((evidence.parent / 'reviewer-host.json').read_text())
        assert host_receipt['interruption'] == 'host_task_cancelled'
        assert host_receipt['cancellation_channels'] == {'shared_marker': 'published'}
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
            assert isinstance(data, str) and data.startswith('Read ')
            data = json.loads(data.removeprefix('Read '))
            assert isinstance(data, str) and data.startswith('data: ')
            data = json.loads(data.removeprefix('data: '))
            assert data['nous']['type'] == 'api_key'
        retained = {path: path.read_bytes() for path in evidence.parent.rglob('*') if path.is_file()}
        assert any(b'ordinary-public-key' in body for body in retained.values())
        assert not [str(path.relative_to(evidence.parent))
                    for path, body in retained.items() if secret.encode() in body]
        collected = ''.join((result.stdout or '') + (result.stderr or '') for result in local.results)
        assert secret not in collected
        assert credential.read_bytes() == original_auth
    finally:
        if not running.done():
            running.cancel()
        await local.close()
        await asyncio.gather(running, return_exceptions=True)
        await local.close()


async def test_cancelled_host_before_reviewer_start_cannot_launch_native_inference(tmp_path: Path) -> None:
    """Cancellation before launch publishes the stop marker, starts no native CLI, keeps a competing receipt."""
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
        assert (logs / 'reviewer-host.json').read_text() == 'prior competing host receipt'
        assert not Path(environment['DAYDREAM_RB_EVIDENCE']).exists()
        assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
    finally:
        launch.set()
        if not running.done():
            running.cancel()
        await local.close()
        await asyncio.gather(running, return_exceptions=True)


async def test_completed_review_bounds_oversized_entrypoint_output_and_keeps_native_result(
    tmp_path: Path, capfd: pytest.CaptureFixture[str],
) -> None:
    """A completed review keeps its result while an oversized credential-bearing stdout line is discarded.

    The declared trial timeout becomes the outer deadline plus the collection allowance.
    """
    from daydream.benchmark.reviewbench.harbor_process import OUTPUT_LIMIT_BYTES

    environment = public_input(tmp_path)
    cli = install_codex_lifecycle_cli(tmp_path / 'cli', mode='healthy', empty_reviews=True)
    declared_task = tmp_path / 'declared-task'
    declared_task.mkdir()
    (declared_task / 'task.toml').write_text('[agent]\ntimeout_sec = 900\n')
    (tmp_path / 'config.json').write_text(json.dumps({'task': {'path': str(declared_task)}, 'agent': {}}))
    payload = b'Authorization: Bearer ' + b'A' * OUTPUT_LIMIT_BYTES + b'BOUNDARY_SECRET_CANARY\n'
    received_timeouts: list[int | None] = []

    class DeadlineEnvironment(LocalReviewerEnvironment):
        async def exec(self, command: str, **kwargs: Any) -> Any:
            received_timeouts.append(kwargs.get('timeout_sec'))
            return await super().exec(command, **kwargs)

    local = DeadlineEnvironment({**os.environ, **environment,
        'PATH': os.pathsep.join((str(cli.bin_dir), os.environ['PATH']))},
        Path(environment['DAYDREAM_RB_REPOSITORY']),
        entrypoint_bootstrap=('import asyncio,os\n'
            f'os.write(1,{payload!r})\n'
            'os.dup2(2,1)\n'  # Keep stdout purely JSON; retain ordinary runner output on captured stderr.
            'from daydream.benchmark.reviewbench.harbor_entrypoint import _executable_main\n'
            'raise SystemExit(asyncio.run(_executable_main()))\n'))
    logs = Path(environment['DAYDREAM_RB_EVIDENCE']).parent
    try:
        await asyncio.wait_for(DaydreamPublicReviewAgent(logs_dir=logs).run(
            'frozen public task', local, SimpleNamespace()), timeout=30)
        record = json.loads((Path(environment['DAYDREAM_RB_EVIDENCE']) / 'execution.json').read_text())
        assert record['status'] == 'completed_empty'
        assert received_timeouts == [932]
        assert json.loads(Path(environment['DAYDREAM_RB_OUTPUT']).read_text())['findings'] == []
        assert all(child.returncode == 0 for child in local.children)
        assert not list(logs.rglob('*checkpoint*')), 'fresh completed run created interruption checkpoints'
        receipt = json.loads((logs / 'reviewer-process.json').read_text())
        assert receipt['cleanup_outcome'] == 'not_requested' and receipt['child_returncode'] == 0
        assert receipt['outer_deadline_seconds'] == 900
        assert receipt['outer_deadline_availability'] == 'declared trial config'
        assert receipt['terminal_evidence_available'] is True
        assert receipt['stdout_capture']['discarded'] and receipt['output_truncated']
        captured = capfd.readouterr()
        all_output = captured.out + captured.err + ''.join(
            (result.stdout or '') + (result.stderr or '') for result in local.results)
        assert not [line for line in all_output.splitlines() if line.startswith('ReviewBench ')]
        retained = (logs / 'reviewer-entrypoint.log').read_bytes()
        assert len(retained) <= 2 * OUTPUT_LIMIT_BYTES
        assert b'CANARY' not in retained and 'CANARY' not in all_output
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


async def test_sigterm_joins_real_finalization_writer_before_export(tmp_path: Path) -> None:
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


@pytest.mark.skipif(sys.platform != 'linux', reason='late orphan containment requires Linux PR_SET_CHILD_SUBREAPER')
@pytest.mark.parametrize('process_table_failure', [False, True])
async def test_forced_reviewer_exit_reaps_late_descendant_holding_output(
    tmp_path: Path, process_table_failure: bool,
) -> None:
    """A fork after the final ownership snapshot must not block reap, even when the process table read fails."""
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


async def test_process_control_observation_requires_bounded_owned_regular_files(tmp_path: Path) -> None:
    """Only a bounded regular file owned by this invocation is a receipt or shared stop request."""
    from daydream.benchmark.reviewbench.harbor_agent import _process_completion_observed
    from daydream.benchmark.reviewbench.harbor_process import _requested_interruption

    invocation = str(uuid.uuid4())
    for surface in ('receipt', 'shared_stop'):
        for storage in ('regular', 'wrong_invocation', 'fifo', 'oversize', 'leaf_symlink', 'parent_symlink'):
            root = tmp_path / f'{surface}-{storage}'
            real = root / 'real'
            directory = real / 'owner'
            directory.mkdir(parents=True)
            if storage == 'parent_symlink':
                alias = root / 'alias'
                alias.symlink_to(real, target_is_directory=True)
                directory = alias / 'owner'
            marker_id = str(uuid.uuid4()) if storage == 'wrong_invocation' else invocation
            body: dict[str, Any] = (
                {'schema': 1, 'invocation_id': marker_id, 'child_returncode': 130} if surface == 'receipt'
                else {'invocation_id': marker_id, 'interruption': 'host_execution_failed'})
            name = 'reviewer-process.json' if surface == 'receipt' else 'reviewer-cancel-' + invocation + '.json'
            path = directory / name
            if storage == 'fifo':
                os.mkfifo(path, mode=0o600)
            elif storage == 'leaf_symlink':
                target = root / 'other-owner.json'
                target.write_text(json.dumps(body))
                path.symlink_to(target)
            else:
                path.write_text(json.dumps(body) + (' ' * 65537 if storage == 'oversize' else ''))
            observed: list[object] = []

            def observe(surface: str = surface, directory: Path = directory, observed: list[object] = observed) -> None:
                observed.append(_process_completion_observed(directory, invocation) if surface == 'receipt'
                    else _requested_interruption({'DAYDREAM_RB_EVIDENCE': str(directory / 'daydream')}, invocation))

            # A daemon thread bounds a FIFO open that would otherwise block forever.
            reader = threading.Thread(target=observe, daemon=True)
            reader.start()
            reader.join(timeout=3)
            assert observed, f'{surface}/{storage} observation blocked'
            assert bool(observed[0]) is (storage == 'regular'), (surface, storage, observed)
