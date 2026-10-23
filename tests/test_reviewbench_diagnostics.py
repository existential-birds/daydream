"""Checkpoint publication tests keep the actual recorder and filesystem boundary."""
from __future__ import annotations

import errno
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import TextEvent
from daydream.benchmark.reviewbench.diagnostics import (
    InterruptedReviewDiagnostics,
    interrupted_readout,
    read_checkpoint,
)
from daydream.trajectory import DaydreamPhase, RunWriteSnapshot, run_directory
from tests.harness.trajectory import make_recorder


@pytest.mark.parametrize('fault', ['sibling', 'commit', 'symlink', 'parent_symlink', 'ancestor_swap'])
async def test_checkpoint_fault_preserves_previous_complete_document_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """A torn generation cannot replace the last usable root/fork checkpoint.

    Existing recorder common-cutoff tests stop at the immutable callback. This
    independent export boundary must survive actual link failures and competing
    symlinks while preserving exact source bytes and original references. Leaf
    symlink checks miss a parent inserted at mkdir or replaced during publication.
    """
    evidence = tmp_path / 'logs' / 'daydream'
    sink = InterruptedReviewDiagnostics(evidence, {'task_id': 7, 'api_key': 'private-key-canary'})
    snapshots: list[RunWriteSnapshot] = []

    def retain(_recorder: Any, snapshot: RunWriteSnapshot) -> None:
        snapshots.append(snapshot)
        sink.retain(snapshot)

    session = str(uuid.uuid4())
    root = make_recorder(tmp_path, session_id=session, on_write=retain,
                         path=run_directory(tmp_path / '.daydream', session) / 'trajectory.json')
    async with root:
        async with root.fork('deep-python') as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                invocation.observe(TextEvent('public activity before interruption'))
                root.write_partial()
                previous = sink.checkpoint_directory / '00000001'
                manifest, documents = read_checkpoint(previous)
                saved = {path: (previous / path).read_bytes() for path in manifest['documents']}
                assert len(documents) == 2
                assert {document['extra']['snapshot_at'] for document in documents} == {snapshots[-1].cutoff_at}
                for document in snapshots[-1].documents:
                    assert document.json_bytes in saved.values()

                link, mkdir = os.link, os.mkdir
                victim = tmp_path / 'competing-private-file'
                victim.write_bytes(b'competing bytes')
                outside = tmp_path / 'outside-checkpoints'
                outside.mkdir()
                generation = sink.checkpoint_directory / '00000002'
                swapped = False

                def target_path(destination: Any, descriptor: int | None) -> Path:
                    path = Path(destination)
                    if descriptor is None:
                        return path
                    # Observe only the real directory identity at the fault
                    # boundary, supporting both path- and dirfd-based publishers.
                    metadata = os.fstat(descriptor)
                    candidates = [generation, generation / 'runs' / session,
                                  generation / 'runs' / session / 'trajectories']
                    for candidate in candidates:
                        if candidate.exists():
                            current = candidate.stat()
                            if (current.st_dev, current.st_ino) == (metadata.st_dev, metadata.st_ino):
                                return candidate / path
                    return path

                def substitute_parent(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
                    destination = target_path(path, dir_fd)
                    if fault == 'parent_symlink' and destination == generation / 'runs':
                        os.symlink(outside, path, dir_fd=dir_fd)
                        return
                    mkdir(path, mode, dir_fd=dir_fd)

                def fail_install(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
                    nonlocal swapped
                    path = target_path(destination, kwargs.get('dst_dir_fd'))
                    if '00000002' in path.parts and (
                        (fault in {'sibling', 'symlink'} and path.parent.name == 'trajectories')
                        or (fault == 'commit' and path.name == 'checkpoint.json')
                    ):
                        if fault == 'symlink':
                            os.symlink(victim, destination, dir_fd=kwargs.get('dst_dir_fd'))
                        else:
                            raise OSError(errno.ENOSPC, 'injected full destination filesystem')
                    if fault == 'ancestor_swap' and path == generation / 'runs' / session / 'trajectory.json':
                        (generation / 'runs').rename(generation / 'runs-held')
                        (generation / 'runs').symlink_to(outside, target_is_directory=True)
                        swapped = True
                    link(source, destination, *args, **kwargs)

                with monkeypatch.context() as patch:
                    patch.setattr(os, 'link', fail_install)
                    patch.setattr(os, 'mkdir', substitute_parent)
                    invocation.observe(TextEvent('later public activity'))
                    root.write_partial()
                    assert sink.last_error is not None
                    readout = interrupted_readout(evidence)
                    assert readout['checkpoint']['generation'] == '00000001'
                    assert readout['terminal_evidence'] == 'unavailable'
                    assert readout['publication_failures']
                    assert len(readout['unavailable_checkpoints']) == 1
                    assert {path: (previous / path).read_bytes() for path in saved} == saved
                    assert victim.read_bytes() == b'competing bytes'
                    assert list(outside.iterdir()) == []
                    assert swapped == (fault == 'ancestor_swap')
                    assert not evidence.exists()
    assert b'private-key-canary' not in (sink.checkpoint_directory / 'metadata.json').read_bytes()


@pytest.mark.parametrize('competition', ['bytes', 'ancestor_symlink'])
async def test_checkpoint_retention_and_readout_preserve_missing_usage_and_competing_bytes(
    tmp_path: Path, competition: str,
) -> None:
    """Bounded retention preserves competing bytes and doesn't manufacture usage.

    The terminal export tests cannot cover pruning old checkpoint generations
    or selecting the newest committed generation after a damaged newest tree.
    A same-content target behind a substituted parent detects pruning that
    follows an ancestor symlink and deletes another owner's diagnostic bytes.
    """
    evidence = tmp_path / 'logs/daydream'
    sink = InterruptedReviewDiagnostics(evidence, {'outer_deadline_seconds': 900, 'native_allowance': 1200})
    session = str(uuid.uuid4())
    recorder = make_recorder(tmp_path, session_id=session, on_write=lambda _rec, snapshot: sink.retain(snapshot))
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.INTENT) as invocation:
            invocation.observe(TextEvent('first observation'))
            recorder.write_partial()
            first = sink.checkpoint_directory / '00000001'
            first_root = first / 'runs' / session / 'trajectory.json'
            expected = b'competing root bytes'
            if competition == 'bytes':
                first_root.write_bytes(expected)
            else:
                expected = first_root.read_bytes()
                outside = tmp_path / 'competing-owner'
                (outside / session).mkdir(parents=True)
                (outside / session / 'trajectory.json').write_bytes(expected)
                (first / 'runs').rename(first / 'runs-held')
                (first / 'runs').symlink_to(outside, target_is_directory=True)
            for index in range(3):
                invocation.observe(TextEvent(f'observation {index}'))
                recorder.write_partial()
            assert first_root.read_bytes() == expected
            assert not (sink.checkpoint_directory / '00000002').exists()
            report = interrupted_readout(evidence)
            assert report['checkpoint']['generation'] == '00000004'
            assert report['identity_and_policy']['outer_deadline_seconds'] == 900
            assert 'terminal_result' not in report
            manifest, documents = read_checkpoint(sink.checkpoint_directory / '00000004')
            assert documents[0]['final_metrics'].get('total_prompt_tokens') is None
            assert documents[0]['extra']['partial'] is True
            (sink.checkpoint_directory / '00000004' / manifest['root']).write_bytes(b'torn data')
            assert interrupted_readout(evidence)['checkpoint']['generation'] == '00000003'
    with pytest.raises(FileExistsError):
        InterruptedReviewDiagnostics(evidence, {})


@pytest.mark.parametrize('location', ['generation', 'ancestor'])
def test_checkpoint_readout_rejects_symlinked_generation(tmp_path: Path, location: str) -> None:
    """An operator-supplied bundle must not resolve a symlink outside its root."""
    actual = tmp_path / 'actual'
    actual.mkdir()
    generation = actual if location == 'generation' else actual / 'generation'
    generation.mkdir(exist_ok=True)
    (generation / 'checkpoint.json').write_text(json.dumps({'schema': 1, 'kind': 'checkpoint'}))
    link = tmp_path / 'link'
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        read_checkpoint(link if location == 'generation' else link / 'generation')


@pytest.mark.parametrize('operation', ['create', 'readout'])
def test_checkpoint_boundary_rejects_symlinked_evidence_parent(tmp_path: Path, operation: str) -> None:
    """A safe final name cannot authorize writes or reads through a redirected parent.

    Existing generation and leaf checks start below the checkpoint root, missing
    operator-supplied ancestry and the constructor's initial metadata publication.
    """
    outside = tmp_path / 'another-owner'
    outside.mkdir()
    link = tmp_path / 'logs'
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        if operation == 'create':
            InterruptedReviewDiagnostics(link / 'daydream', {})
        else:
            interrupted_readout(link / 'daydream')
    assert list(outside.iterdir()) == []


def test_interrupted_readout_retains_host_receipt_when_reviewer_never_published(tmp_path: Path) -> None:
    """Host-only startup failures remain visible without inventing reviewer evidence.

    Joined-export and checkpoint readers miss cancellation before the reviewer
    starts, where the adapter's independent receipt is the only available input.
    This fixture supplies the external host artifact to the offline read boundary.
    """
    logs = tmp_path / 'logs'
    logs.mkdir()
    receipt = {'schema': 1, 'invocation_id': 'host-invocation', 'interruption': 'host_task_cancelled',
               'cancellation_channels': {'control_exec': 'unavailable:TimeoutError'},
               'cleanup_task': 'detached_at_absolute_bound', 'output_collection': 'unavailable'}
    (logs / 'reviewer-host.json').write_text(json.dumps(receipt))
    report = interrupted_readout(logs / 'daydream')
    assert report['host'] == receipt
    assert report['terminal_evidence'] == report['checkpoint'] == 'unavailable'
    assert 'execution' not in report and 'activity' not in report and 'process' not in report


@pytest.mark.parametrize('receipt', ['reviewer-host.json', 'result.json'])
@pytest.mark.parametrize('damage', ['corrupt', 'symlink'])
async def test_optional_receipt_damage_cannot_hide_valid_native_checkpoint(
    tmp_path: Path, receipt: str, damage: str,
) -> None:
    """Independent optional inputs cannot suppress a valid interrupted-run readout.

    Healthy bundle tests miss torn Harbor metadata and redirected host receipts.
    Keep the actual recorder-produced checkpoint; fault only the external reader
    input, then require its safe availability reason alongside retained activity.
    """
    evidence = tmp_path / 'logs/daydream'
    sink = InterruptedReviewDiagnostics(evidence, {'task_id': 7})
    session = str(uuid.uuid4())
    recorder = make_recorder(tmp_path, session_id=session, on_write=lambda _rec, snapshot: sink.retain(snapshot))
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.INTENT) as invocation:
            invocation.observe(TextEvent('observed native activity'))
            recorder.write_partial()
            path = evidence.parent / receipt if receipt == 'reviewer-host.json' else tmp_path / receipt
            outside = tmp_path / 'private-receipt'
            outside.write_text('{"private": "outside-receipt-canary"}')
            if damage == 'corrupt':
                path.write_bytes(b'{torn')
            else:
                path.symlink_to(outside)
            report = interrupted_readout(evidence)
            assert report['checkpoint']['generation'] == '00000001'
            assert report['activity'][0]['trajectory_id'] == session
            assert report['input_availability'][receipt] == (
                'unavailable:JSONDecodeError' if damage == 'corrupt' else 'unavailable:ValueError')
            assert report['terminal_evidence'] == 'unavailable'
            assert 'outside-receipt-canary' not in json.dumps(report)
            assert outside.read_text() == '{"private": "outside-receipt-canary"}'
