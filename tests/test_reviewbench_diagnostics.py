"""Checkpoint storage owns atomicity, bounded retention, and offline recovery.

Recorder graph/privacy contracts live in test_trajectory; grammar in test_redaction.
"""
from __future__ import annotations

import errno
import hashlib
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
from daydream.trajectory import DaydreamPhase, RunWriteSnapshot, TrajectoryRecorder, run_directory
from tests.harness.trajectory import make_recorder


def _checkpoint_run(tmp_path: Path) -> tuple[InterruptedReviewDiagnostics, TrajectoryRecorder, list[RunWriteSnapshot]]:
    sink = InterruptedReviewDiagnostics(tmp_path / 'logs/daydream', {'api_key': 'opaque-metadata-canary'})
    snapshots: list[RunWriteSnapshot] = []

    def retain(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot) -> None:
        snapshot.validate(recorder.session_id)
        snapshots.append(snapshot)
        sink.retain(snapshot)

    session = str(uuid.uuid4())
    recorder = make_recorder(tmp_path, session_id=session, on_write=retain,
                             path=run_directory(tmp_path / '.daydream', session) / 'trajectory.json')
    return sink, recorder, snapshots


@pytest.mark.parametrize('fault', ['disk_full', 'competing_symlink'])
async def test_checkpoint_fault_preserves_previous_complete_document_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """One failed atomic install cannot expose half a root/fork snapshot."""
    sink, root, snapshots = _checkpoint_run(tmp_path)
    victim = tmp_path / 'competing-private-file'
    victim.write_bytes(b'competing bytes')
    async with root:
        async with root.fork('{"type":"api_key","key":"opaque-fork-canary"}') as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                invocation.observe(TextEvent('public activity before interruption'))
                root.write_partial()
                previous = sink.checkpoint_directory / 'checkpoint-00000001.json'
                saved = previous.read_bytes()
                manifest, documents = read_checkpoint(previous)
                assert len(documents) == 2
                assert {doc['extra']['snapshot_at'] for doc in documents} == {snapshots[-1].cutoff_at}
                bundle = json.loads(saved)
                assert {entry['json'].encode() for entry in bundle['documents'].values()} == {
                    document.json_bytes for document in snapshots[-1].documents}
                link = os.link

                def fail_install(source: Any, destination: Any, **kwargs: Any) -> None:
                    if Path(destination).name == 'checkpoint-00000002.json':
                        if fault == 'disk_full':
                            raise OSError(errno.ENOSPC, 'injected full filesystem')
                        os.symlink(victim, destination, dir_fd=kwargs.get('dst_dir_fd'))
                    link(source, destination, **kwargs)

                with monkeypatch.context() as patch:
                    patch.setattr(os, 'link', fail_install)
                    invocation.observe(TextEvent('later activity'))
                    root.write_partial()
                report = interrupted_readout(tmp_path / 'logs/daydream')
                assert sink.last_error and report['publication_failures']
                assert report['checkpoint']['generation'] == '00000001'
                assert report['terminal_evidence'] == 'unavailable'
                assert previous.read_bytes() == saved and victim.read_bytes() == b'competing bytes'
                assert not (tmp_path / 'logs/daydream').exists()
                assert '-canary' not in saved.decode() + (sink.checkpoint_directory / 'metadata.json').read_text()
                assert manifest['root'] in manifest['documents']


@pytest.mark.parametrize('competition', ['bytes', 'symlink'])
async def test_checkpoint_retention_preserves_competing_bytes_and_falls_back_from_corruption(
    tmp_path: Path, competition: str,
) -> None:
    """Keep two owned checkpoints; preserve substituted output and unknown usage."""
    sink, recorder, _ = _checkpoint_run(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.INTENT) as invocation:
            invocation.observe(TextEvent('first'))
            recorder.write_partial()
            first = sink.checkpoint_directory / 'checkpoint-00000001.json'
            victim = tmp_path / 'other-owner'
            victim.write_bytes(b'competing bytes')
            if competition == 'bytes':
                first.write_bytes(victim.read_bytes())
            else:
                first.unlink()
                first.symlink_to(victim)
            for index in range(3):
                invocation.observe(TextEvent(f'observation {index}'))
                recorder.write_partial()
            assert first.read_bytes() == victim.read_bytes() == b'competing bytes'
            assert not (sink.checkpoint_directory / 'checkpoint-00000002.json').exists()
            newest = sink.checkpoint_directory / 'checkpoint-00000004.json'
            _, documents = read_checkpoint(newest)
            assert documents[0]['final_metrics'].get('total_prompt_tokens') is None
            assert documents[0]['extra']['partial'] is True
            assert interrupted_readout(tmp_path / 'logs/daydream')['checkpoint']['generation'] == '00000004'
            newest.write_bytes(b'torn data')
            assert interrupted_readout(tmp_path / 'logs/daydream')['checkpoint']['generation'] == '00000003'
    with pytest.raises(FileExistsError):
        InterruptedReviewDiagnostics(tmp_path / 'logs/daydream', {})


@pytest.mark.parametrize('operation', ['create', 'readout', 'read_checkpoint'])
def test_checkpoint_boundary_rejects_symlinked_ancestry(tmp_path: Path, operation: str) -> None:
    """An operator path cannot redirect evidence reads/writes into another owner."""
    outside = tmp_path / 'another-owner'
    outside.mkdir()
    link = tmp_path / 'logs'
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        if operation == 'create':
            InterruptedReviewDiagnostics(link / 'daydream', {})
        elif operation == 'readout':
            interrupted_readout(link / 'daydream')
        else:
            read_checkpoint(link / 'checkpoint.json')
    assert list(outside.iterdir()) == []


def test_interrupted_readout_retains_host_only_failure(tmp_path: Path) -> None:
    logs = tmp_path / 'logs'
    logs.mkdir()
    receipt = {'interruption': 'host_task_cancelled', 'output_collection': 'unavailable'}
    (logs / 'reviewer-host.json').write_text(json.dumps(receipt))
    report = interrupted_readout(logs / 'daydream')
    assert report['host'] == receipt and report['terminal_evidence'] == report['checkpoint'] == 'unavailable'
    assert 'execution' not in report and 'activity' not in report and 'process' not in report


@pytest.mark.parametrize(('receipt', 'damage'), [
    (receipt, damage) for receipt in ('reviewer-host.json', 'result.json')
    for damage in ('corrupt', 'symlink', 'wrong_container')
] + [('result.json', 'exception_container')])
async def test_optional_receipt_damage_cannot_hide_checkpoint(tmp_path: Path, receipt: str, damage: str) -> None:
    sink, recorder, _ = _checkpoint_run(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.INTENT) as invocation:
            invocation.observe(TextEvent('observed activity'))
            recorder.write_partial()
            path = sink.checkpoint_directory.parent / receipt if receipt == 'reviewer-host.json' else tmp_path / receipt
            outside = tmp_path / 'private-receipt'
            outside.write_text('{"private":"outside-canary"}')
            if damage == 'symlink':
                path.symlink_to(outside)
            else:
                path.write_text({'corrupt': '{torn', 'wrong_container': '[]',
                                 'exception_container': '{"exception_info":[]}'}[damage])
            report = interrupted_readout(tmp_path / 'logs/daydream')
            assert report['checkpoint']['generation'] == '00000001'
            assert report['activity'][0]['trajectory_id'] == recorder.session_id
            assert report['input_availability'][receipt].startswith('unavailable:')
            assert report['terminal_evidence'] == 'unavailable' and 'outside-canary' not in json.dumps(report)
            assert outside.read_text() == '{"private":"outside-canary"}'


@pytest.mark.parametrize(('field', 'value'), [
    ('schema', True), ('kind', []), ('documents', []), ('documents', None), ('root', 'absent'),
    ('cutoff_at', None), ('session_id', []), ('recorder_status', {}), ('digest', []), ('document', []),
    ('extra', []), ('subtrajectories', [False]), ('phase_events', [False]), ('active_dispatches', [False]),
])
async def test_checkpoint_schema_corruption_selects_previous_generation(tmp_path: Path, field: str, value: Any) -> None:
    """Reject malformed committed containers and intact-looking changed document bytes."""
    sink, recorder, _ = _checkpoint_run(tmp_path)
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.INTENT) as invocation:
            for message in ('first', 'second'):
                invocation.observe(TextEvent(message))
                recorder.write_partial()
            newest = sink.checkpoint_directory / 'checkpoint-00000002.json'
            bundle = json.loads(newest.read_bytes())
            if field in ('document', 'extra', 'subtrajectories', 'phase_events', 'active_dispatches'):
                entry = bundle['documents'][bundle['root']]
                document = json.loads(entry['json'])
                if field == 'document':
                    document = value
                elif field == 'extra':
                    document['extra'] = value
                else:
                    document['extra'][field] = value
                entry['json'] = json.dumps(document)
                entry['sha256'] = hashlib.sha256(entry['json'].encode()).hexdigest()
            elif field == 'digest':
                bundle['documents'][bundle['root']]['sha256'] = value
            else:
                bundle[field] = value
            newest.write_text(json.dumps(bundle))
            with pytest.raises(ValueError):
                read_checkpoint(newest)
            report = interrupted_readout(tmp_path / 'logs/daydream')
            assert report['checkpoint']['generation'] == '00000001'
            assert report['unavailable_checkpoints'] == [{'generation': '00000002', 'reason': 'ValueError'}]
