"""Real host -> reviewer process -> native CLI interruption contract."""
from __future__ import annotations

import asyncio
import errno
import json
import os
import shlex
import signal
import sys
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
                 publication_fault: str | None = None, close_stdout: bool = False) -> None:
        self.environment = environment
        self.repository = repository
        self.publication_fault = publication_fault
        self.close_stdout = close_stdout
        self.results: list[SimpleNamespace] = []
        self.children: list[asyncio.subprocess.Process] = []
        self.collectors: list[asyncio.Task[tuple[bytes, bytes]]] = []

    async def exec(self, command: str, **kwargs: Any) -> Any:
        arguments = shlex.split(command)
        assert arguments[0] == 'python'
        arguments[0] = sys.executable
        if self.publication_fault is not None and arguments[1:4] == [
            '-I', '-m', 'daydream.benchmark.reviewbench.harbor_process',
        ]:
            # Patch the actual filesystem install only in the supervisor. The
            # production entrypoint/native CLI remain separate, unpatched children.
            bootstrap = ('import errno,os,runpy,sys\nfrom pathlib import Path\n'
                'original_link = os.link\n'
                'def fail_link(source,destination,**kwargs):\n'
                f'    if Path(destination).name == {self.publication_fault!r}:\n'
                '        raise OSError(errno.ENOSPC,"publication-secret-canary")\n'
                '    return original_link(source,destination,**kwargs)\n'
                'os.link = fail_link\n'
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
])
async def test_outer_timeout_stops_owned_reviewer_and_retains_process_evidence(
    tmp_path: Path, interruption: str,
) -> None:
    """An outer cancellation must cross exec and retain joined native evidence.

    test_cancelled_public_reviewer_exports_joined_native_evidence_and_propagates_cancellation
    cancels inside the reviewer loop and cannot detect a detached child after a
    cancelled host collector. This harness enters production host and executable
    modules, real Git and Codex transport; only the external CLI is scripted.
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
    class ControlFailureEnvironment(LocalReviewerEnvironment):
        async def exec(self, command: str, **kwargs: Any) -> Any:
            if interruption == 'control_failure' and '--cancel' in command:
                return SimpleNamespace(return_code=1, stdout='', stderr='control exec unavailable')
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
        if interruption in {'timeout', 'slow_process_listing'}:
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
        assert receipt['interruption'] == 'host_cancellation'
        assert receipt['outer_deadline_seconds'] is None
        if interruption == 'late_child':
            escaped = json.loads((cli.root / 'escaped.json').read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(escaped['pid'], 0)
            assert receipt['descendant_containment'] == 'linux_subreaper'
            assert receipt['owned_descendants_cleanup'] == 'joined'
        assert receipt['cleanup_outcome'] == ('forced_termination' if interruption == 'force' else 'graceful_exit')
        host_receipt = json.loads((logs / 'reviewer-host.json').read_text())
        assert host_receipt['interruption'] == 'host_task_cancelled'
        if interruption == 'control_failure':
            assert host_receipt['cancellation_channels'] == {'shared_marker': 'published', 'control_exec': 'failed'}
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
        await asyncio.gather(running, return_exceptions=True)


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


@pytest.mark.parametrize('diagnostic_failure', [
    'none', 'log_conflict', 'receipt_conflict', 'log_symlink', 'receipt_symlink',
    'supervisor_enospc', 'host_fallback_enospc', 'closed_collector_pipe',
])
async def test_completed_process_review_keeps_native_result_and_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str], diagnostic_failure: str,
) -> None:
    """Process supervision must preserve the ordinary scorable successful path.

    Existing terminal tests bypass executable startup and supervisor buffering;
    this runs a scripted successful native review through the production host.
    Optional diagnostic install failures must preserve the actual successful
    native result; finalized native evidence/findings remain authoritative.
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
        close_stdout=diagnostic_failure == 'closed_collector_pipe')
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
        if diagnostic_failure != 'none':
            captured = capfd.readouterr()
            diagnostics = captured.err + ''.join(result.stderr for result in local.results)
            assert 'diagnostics unavailable' in diagnostics
            assert 'errno=' in diagnostics and len(diagnostics) < 1024
            assert 'publication-secret-canary' not in diagnostics
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
    import uuid

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
