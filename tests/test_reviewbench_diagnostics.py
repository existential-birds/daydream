"""Checkpoint publication tests keep the actual recorder and filesystem boundary."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent, TurnEndEvent
from daydream.benchmark.reviewbench.diagnostics import (
    InterruptedReviewDiagnostics,
    interrupted_readout,
    read_checkpoint,
)
from daydream.benchmark.reviewbench.harbor_auth import validate_judge_home, validate_pi_reviewer_home
from daydream.trajectory import DaydreamPhase, RunWriteSnapshot, dispatch_scope, run_directory
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


@pytest.mark.parametrize(('receipt', 'damage'), [
    (receipt, damage) for receipt in ('reviewer-host.json', 'result.json')
    for damage in ('corrupt', 'symlink', 'wrong_container')
] + [('result.json', 'exception_container')])
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
            elif damage == 'wrong_container':
                path.write_bytes(b'[]')
            elif damage == 'exception_container':
                path.write_text(json.dumps({'exception_info': ['malformed']}))
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


@pytest.mark.parametrize('label_kind', [
    'api_key', 'marker_json', 'scheme_token', 'scheme_suffix', 'marker_after_bare_value',
    'marker_after_quoted_value', 'marker_prefixed_value', 'repeated_marker_prefixed_value',
    'sensitive_container', 'duplicate_sensitive_container', 'native_credential', 'duplicate_native_credential',
])
@pytest.mark.parametrize('identified_dispatch', [False, True], ids=['legacy-fork', 'identified-dispatch'])
async def test_private_fork_labels_are_admitted_before_identity_and_checkpoint_publication(
    tmp_path: Path, identified_dispatch: bool, label_kind: str,
) -> None:
    """Host labels must remain private without breaking original document refs.

    Message/request privacy coverage does not exercise a caller-supplied fork
    descriptor or a queued scope. Keep the real recorder, dispatch registration,
    immutable callback and no-clobber filesystem publication, including two
    different private labels that must not collapse onto one legacy identity.
    """
    evidence = tmp_path / 'logs/daydream'
    sink = InterruptedReviewDiagnostics(evidence, {'task_id': 7})
    snapshots: list[RunWriteSnapshot] = []

    def retain(_recorder: Any, snapshot: RunWriteSnapshot) -> None:
        snapshots.append(snapshot)
        sink.retain(snapshot)

    session = str(uuid.uuid4())
    root = make_recorder(tmp_path, session_id=session, on_write=retain,
                         path=run_directory(tmp_path / '.daydream', session) / 'trajectory.json')
    canaries = ['sk-activeforkcanary123', 'sk-otherforkcanary456', 'sk-queuedforkcanary789']
    labels = canaries
    if label_kind == 'marker_json':
        canaries = ['opaque-descriptor-one', 'opaque-descriptor-two', 'opaque-descriptor-queued']
        labels = [json.dumps({'api_key': '[REDACTED] ' + canary}) for canary in canaries]
    elif label_kind in ('native_credential', 'duplicate_native_credential'):
        canaries = ['opaque-native-label-one', 'opaque-native-label-two', 'opaque-native-label-queued']
        labels = ['{"type":"api_key","key":"' + canary + '"'
                  + (',"key":"[REDACTED]"}' if label_kind == 'duplicate_native_credential' else '}')
                  for canary in canaries]
    elif label_kind != 'api_key':
        canaries = ['opaque-descriptor-one', 'opaque-descriptor-two', 'opaque-descriptor-queued']
        if label_kind == 'sensitive_container':
            labels = [json.dumps({'api_key': {'v': canary}}) for canary in canaries]
        elif label_kind == 'duplicate_sensitive_container':
            labels = ['{"outer":{"api_key":{"v":"' + canary + '"}},"outer":{}}' for canary in canaries]
        elif label_kind == 'scheme_token':
            labels = ['auth=Bearer [REDACTED]' + canary for canary in canaries]
        elif label_kind == 'scheme_suffix':
            labels = ['auth=Bearer ' + canary + '[REDACTED]' for canary in canaries]
        elif label_kind == 'marker_prefixed_value':
            labels = ['refresh_token=[REDACTED]' + canary for canary in canaries]
        elif label_kind == 'repeated_marker_prefixed_value':
            labels = ['refresh_token=[REDACTED][REDACTED_API_KEY]' + canary for canary in canaries]
        elif label_kind == 'marker_after_bare_value':
            labels = ['refresh_token=' + canary + ' [REDACTED]' for canary in canaries]
        else:
            labels = ['refresh_token="' + canary + '" [REDACTED]' for canary in canaries]
    async with root:
        async with dispatch_scope(root if identified_dispatch else None, phase=DaydreamPhase.DEEP,
                                  descriptors=labels) as dispatch:
            async with root.fork(labels[0], dispatch=dispatch) as first:
                async with first.invocation(phase=DaydreamPhase.DEEP) as invocation:
                    invocation.observe(TextEvent('first public activity'))
                    async with root.fork(labels[1], dispatch=dispatch) as second:
                        async with second.invocation(phase=DaydreamPhase.DEEP) as other:
                            other.observe(TextEvent('second public activity'))
                            root.write_partial()
                            snapshot = snapshots[-1]
                            joined = b''.join(document.json_bytes for document in snapshot.documents)
                            for canary in canaries:
                                assert canary.encode() not in joined
                                assert canary not in str(first.path) + str(second.path)
                            assert first.descriptor != second.descriptor
                            assert first.trajectory_id != second.trajectory_id and first.path != second.path
                            assert sink.last_error is None
                            generation = sink.checkpoint_directory / '00000001'
                            manifest, documents = read_checkpoint(generation)
                            by_id = {document['trajectory_id']: document for document in documents}
                            references = by_id[session]['extra']['subtrajectories']
                            for child in (first, second):
                                reference = next(row for row in references
                                                 if row.get('trajectory_id') == child.trajectory_id)
                                expected = child.path.relative_to(tmp_path / '.daydream').as_posix()
                                assert reference['sibling_trajectory_ref'] == expected
                                retained = json.loads((generation / expected).read_bytes())
                                assert retained['trajectory_id'] == child.trajectory_id
                                assert retained['session_id'] == session
                            saved = [(generation / name).read_bytes() for name in manifest['documents']]
                            assert all(document.json_bytes in saved for document in snapshot.documents)
                            if dispatch is not None:
                                plan = by_id[session]['extra']['active_dispatches'][0]
                                assert plan['planned_descriptors'][:2] == [first.descriptor, second.descriptor]
                                assert plan['queued_descriptors'] == plan['planned_descriptors'][2:]
                                assert plan['attempted_count'] == 2 and plan['completed_count'] == 0
                                assert [fork['descriptor'] for fork in plan['forks']] == [
                                    first.descriptor, second.descriptor]


async def test_checkpoint_before_first_child_event_retains_host_only_fork_document(tmp_path: Path) -> None:
    """An admitted invocation exists before its first user or native event.

    Active-fork tests already observe text before flushing and miss this startup
    window. A root with prior evidence must not publish a dangling child ref or
    force the sink back to an older checkpoint while the child starts.
    """
    evidence = tmp_path / 'logs/daydream'
    sink = InterruptedReviewDiagnostics(evidence, {'task_id': 7})
    snapshots: list[RunWriteSnapshot] = []

    def retain(_recorder: Any, snapshot: RunWriteSnapshot) -> None:
        snapshots.append(snapshot)
        sink.retain(snapshot)

    session = str(uuid.uuid4())
    root = make_recorder(tmp_path, session_id=session, on_write=retain,
                         path=run_directory(tmp_path / '.daydream', session) / 'trajectory.json')
    async with root:
        async with root.invocation(phase=DaydreamPhase.INTENT) as prior:
            prior.observe(TextEvent('prior public activity'))
            prior.observe(ResultEvent(structured_output=None, continuation=None))
        root.write_partial()
        first_bytes = snapshots[-1].documents[0].json_bytes
        async with root.fork('starting-child') as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW):
                root.write_partial()
                assert sink.last_error is None
                snapshot = snapshots[-1]
                manifest, documents = read_checkpoint(sink.checkpoint_directory / '00000002')
                assert manifest['cutoff_at'] == snapshot.cutoff_at
                assert {document['trajectory_id'] for document in documents} == {session, child.trajectory_id}
                assert {document['extra']['snapshot_at'] for document in documents} == {snapshot.cutoff_at}
                host = next(document for document in documents if document['trajectory_id'] == child.trajectory_id)
                assert len(host['steps']) == 1 and host['steps'][0]['source'] == 'system'
                assert host['steps'][0]['extra']['host_event'] == 'partial_snapshot'
                assert host['extra']['subtrajectories'][0]['ended_at'] is None
                assert host['extra']['subtrajectories'][0]['step_ids'] == []
                assert host['final_metrics'].get('total_prompt_tokens') is None
                assert snapshots[0].documents[0].json_bytes == first_bytes


@pytest.mark.parametrize('damage', [
    'manifest_list', 'documents_list', 'documents_null', 'missing_root', 'null_cutoff', 'session_list',
    'status_list', 'status_object', 'kind_list',
    'digest_list', 'document_list', 'extra_list', 'subtrajectories_list_item', 'phase_events_list_item',
])
async def test_checkpoint_schema_corruption_selects_previous_immutable_generation(tmp_path: Path, damage: str) -> None:
    """Malformed committed containers must not hide the previous valid checkpoint.

    Byte-corruption cases reach the digest guard, while a list manifest or
    malformed document inventory previously escaped the reader with AttributeError.
    Mutate actual files only after two valid generations were recorder-produced.
    """
    evidence = tmp_path / 'logs/daydream'
    sink = InterruptedReviewDiagnostics(evidence, {'task_id': 7})
    session = str(uuid.uuid4())
    recorder = make_recorder(tmp_path, session_id=session, on_write=lambda _rec, snapshot: sink.retain(snapshot))
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.INTENT) as invocation:
            invocation.observe(TextEvent('first public activity'))
            recorder.write_partial()
            previous = sink.checkpoint_directory / '00000001'
            saved_manifest = (previous / 'checkpoint.json').read_bytes()
            invocation.observe(TextEvent('second public activity'))
            recorder.write_partial()
            newest = sink.checkpoint_directory / '00000002'
            marker = newest / 'checkpoint.json'
            manifest = json.loads(marker.read_bytes())
            corrupted: Any = dict(manifest)
            if damage == 'manifest_list':
                corrupted = []
            elif damage == 'documents_list':
                corrupted['documents'] = []
            elif damage == 'documents_null':
                corrupted['documents'] = None
            elif damage == 'missing_root':
                del corrupted['root']
            elif damage == 'null_cutoff':
                corrupted['cutoff_at'] = None
            elif damage == 'session_list':
                corrupted['session_id'] = []
            elif damage == 'digest_list':
                corrupted['documents'] = {manifest['root']: []}
            elif damage == 'status_list':
                corrupted['recorder_status'] = []
            elif damage == 'status_object':
                corrupted['recorder_status'] = {}
            elif damage == 'kind_list':
                corrupted['kind'] = []
            else:
                source = newest / manifest['root']
                document = json.loads(source.read_bytes())
                if damage == 'document_list':
                    document = []
                elif damage == 'extra_list':
                    document['extra'] = []
                elif damage == 'subtrajectories_list_item':
                    document['extra']['subtrajectories'] = [False]
                else:
                    document['extra']['phase_events'] = [False]
                body = json.dumps(document).encode()
                source.write_bytes(body)
                corrupted['documents'][manifest['root']] = hashlib.sha256(body).hexdigest()
            marker.write_text(json.dumps(corrupted))
            with pytest.raises(ValueError):
                read_checkpoint(newest)
            report = interrupted_readout(evidence)
            assert report['checkpoint']['generation'] == '00000001'
            assert report['unavailable_checkpoints'] == [{'generation': '00000002', 'reason': 'ValueError'}]
            assert (previous / 'checkpoint.json').read_bytes() == saved_manifest
            read_checkpoint(previous)


@pytest.mark.parametrize('kind', ['api_key', 'oauth'])
@pytest.mark.parametrize('surface', ['observation', 'message', 'arguments'])
async def test_admitted_native_auth_is_private_in_immutable_checkpoint_and_final_evidence(
    tmp_path: Path, kind: str, surface: str,
) -> None:
    """A validator-admitted input store must not leak through generic auth fields.

    Sensitive-name fixtures miss native type/api_key/key and type/oauth/access/
    refresh. Read an actual admitted fake input file, then preserve real events,
    snapshot validation, no-clobber checkpoint publication and joined final bytes.
    """
    home = tmp_path / ('reviewer' if kind == 'api_key' else 'judge')
    home.mkdir()
    credential = ({'type': kind, 'key': 'opaque-admitted-key-canary'} if kind == 'api_key' else
                  {'type': kind, 'access': 'opaque-admitted-access-canary', 'refresh': 'opaque-admitted-refresh-canary',
                   'accountId': 'offline-account', 'expires': 9000000000})
    auth = {'nous' if kind == 'api_key' else 'openai-codex': credential}
    source = home / 'auth.json'
    source.write_text(json.dumps(auth))
    source.chmod(0o600)
    if kind == 'api_key':
        validate_pi_reviewer_home(home, 'nous')
    else:
        validate_judge_home(home)
    source_bytes = source.read_bytes()
    public = {'type': 'record', 'key': 'ordinary-public-key', 'access': 'ordinary-public-access',
              'refresh': 'ordinary-public-refresh'}
    payload = {'data': json.loads(source_bytes), 'public': public, 'usage': {'input': 0, 'output': None}}
    evidence = tmp_path / 'logs/daydream'
    sink = InterruptedReviewDiagnostics(evidence, {'task_id': 7})
    snapshots: list[RunWriteSnapshot] = []

    def retain(_recorder: Any, snapshot: RunWriteSnapshot) -> None:
        snapshot.validate(_recorder.session_id)
        snapshots.append(snapshot)
        sink.retain(snapshot)

    root = make_recorder(tmp_path, session_id=str(uuid.uuid4()), on_write=retain)
    async with root:
        async with root.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            if surface == 'arguments':
                invocation.observe(ToolStartEvent(id='read-auth', name='Inspect', input=payload))
            elif surface == 'observation':
                invocation.observe(ToolStartEvent(id='read-auth', name='Inspect', input={}))
                invocation.observe(ToolResultEvent(id='read-auth', output=json.dumps(payload), is_error=False))
            else:
                invocation.observe(TextEvent(json.dumps(payload)))
            root.write_partial()
            first = snapshots[-1]
            frozen = b''.join(document.json_bytes for document in first.documents)
            assert b'opaque-admitted-' not in frozen
            _, documents = read_checkpoint(sink.checkpoint_directory / '00000001')
            assert b'opaque-admitted-' not in json.dumps(documents).encode()
            step = next(step for step in documents[0]['steps'] if step['source'] == 'agent')
            retained = (step['tool_calls'][0]['arguments'] if surface == 'arguments' else
                        json.loads(step['observation']['results'][0]['content']
                                   if surface == 'observation' else step['message']))
            assert retained['public'] == public and retained['usage'] == {'input': 0, 'output': None}
            if kind == 'oauth':
                assert retained['data']['openai-codex']['accountId'] == 'offline-account'
                assert retained['data']['openai-codex']['expires'] == 9000000000
            invocation.observe(TurnEndEvent(message_id='auth-read-step'))
            invocation.observe(TextEvent('public activity after checkpoint'))
            root.write_partial()
            assert b''.join(document.json_bytes for document in first.documents) == frozen
            assert 'opaque-admitted-' not in json.dumps(interrupted_readout(evidence))
    assert b'opaque-admitted-' not in root.path.read_bytes()
    for path in sink.checkpoint_directory.rglob('*'):
        if path.is_file():
            assert b'opaque-admitted-' not in path.read_bytes()
    assert source.read_bytes() == source_bytes
