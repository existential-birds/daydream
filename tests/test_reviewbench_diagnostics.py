from __future__ import annotations

import errno
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import TextEvent
from daydream.benchmark.reviewbench.diagnostics import InterruptedReviewDiagnostics
from daydream.trajectory import DaydreamPhase, RunWriteSnapshot, TrajectoryRecorder, run_directory
from tests.harness.reviewbench import checkpoint_documents
from tests.harness.trajectory import make_recorder


def _checkpoint_run(tmp_path: Path) -> tuple[InterruptedReviewDiagnostics, TrajectoryRecorder, list[RunWriteSnapshot]]:
    sink = InterruptedReviewDiagnostics(tmp_path / 'logs/daydream', {'api_key': 'opaque-metadata-canary'})
    snapshots: list[RunWriteSnapshot] = []

    def retain(_recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot) -> None:
        snapshots.append(snapshot)
        sink.retain(snapshot)

    recorder = make_recorder(tmp_path, on_write=retain,
                             path=run_directory(tmp_path / '.daydream', 'test') / 'trajectory.json')
    return sink, recorder, snapshots


@pytest.mark.parametrize('fault', ['disk_full', 'competing_symlink'])
async def test_checkpoint_fault_preserves_previous_complete_document_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
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
                manifest, documents = checkpoint_documents(previous)
                assert len(documents) == 2 and manifest['root'] in manifest['documents']
                assert {doc['extra']['snapshot_at'] for doc in documents} == {snapshots[-1].cutoff_at}
                assert {entry['json'].encode() for entry in manifest['documents'].values()} == {
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
                assert sink.last_error
                failure = json.loads((sink.checkpoint_directory / 'failure-00000002.json').read_bytes())
                assert failure['reason'] == sink.last_error and failure['cutoff_at'] == snapshots[-1].cutoff_at
                assert previous.read_bytes() == saved and victim.read_bytes() == b'competing bytes'
                assert not (tmp_path / 'logs/daydream').exists()
                assert '-canary' not in saved.decode() + (sink.checkpoint_directory / 'metadata.json').read_text()


@pytest.mark.parametrize('competition', ['bytes', 'symlink'])
async def test_checkpoint_retention_preserves_competing_bytes(tmp_path: Path, competition: str) -> None:
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
            assert (sink.checkpoint_directory / 'checkpoint-00000003.json').is_file()
            _, documents = checkpoint_documents(sink.checkpoint_directory / 'checkpoint-00000004.json')
            assert documents[0]['final_metrics'].get('total_prompt_tokens') is None
            assert documents[0]['extra']['partial'] is True
    with pytest.raises(FileExistsError):
        InterruptedReviewDiagnostics(tmp_path / 'logs/daydream', {})


def test_checkpoint_writer_rejects_symlinked_ancestry(tmp_path: Path) -> None:
    outside = tmp_path / 'another-owner'
    outside.mkdir()
    link = tmp_path / 'logs'
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        InterruptedReviewDiagnostics(link / 'daydream', {})
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize('fault', ['missing_reference', 'wrong_identity', 'activity'])
async def test_checkpoint_rejects_invalid_capture_before_publication(tmp_path: Path, fault: str) -> None:
    """Valid graph/IO tests cannot detect admission of a malformed frozen capture."""
    sink, root, snapshots = _checkpoint_run(tmp_path)
    async with root:
        async with root.fork('child') as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                invocation.observe(TextEvent('retained activity'))
                root.write_partial()
                original = snapshots[-1]
                previous = sink.checkpoint_directory / 'checkpoint-00000001.json'
                saved = previous.read_bytes()
                payload = json.loads(original.documents[0].json_bytes)
                if fault == 'activity':
                    payload['extra']['phase_events'] = [False]
                else:
                    entry = payload['extra']['subtrajectories'][0]
                    entry['sibling_trajectory_ref' if fault == 'missing_reference' else 'trajectory_id'] = 'foreign'
                document = replace(original.documents[0], json_bytes=json.dumps(payload).encode())
                sink.retain(replace(original, documents=(document, *original.documents[1:])))
                assert sink.last_error and previous.read_bytes() == saved
                assert not (sink.checkpoint_directory / 'checkpoint-00000002.json').exists()
