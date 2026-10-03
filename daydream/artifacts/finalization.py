"""Freeze joined run evidence and publish or restore its artifact transaction."""
from __future__ import annotations

import os
import secrets
import stat
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from daydream.artifacts import external, filesystem, ledger, publication, transactions
from daydream.artifacts.models import (
    _DAYDREAM,
    _REVIEW_OUTPUT,
    ArtifactDisposition,
    ArtifactTreeSnapshot,
    ArtifactVisibilityError,
    DestinationDelivery,
    _DestinationRecord,
    _SessionState,
    _TerminalState,
    _Transition,
)
from daydream.json_utils import _fsync_directory

if TYPE_CHECKING:
    from daydream.artifacts.session import ArtifactSession
    from daydream.trajectory import RunWriteSnapshot


def freeze(session: ArtifactSession, run_snapshot: RunWriteSnapshot) -> ArtifactTreeSnapshot:
    session._require_active()
    try:
        run_snapshot.validate(session.layout.session_id)
    except (ValueError, UnicodeError) as exc:
        raise ArtifactVisibilityError(f"run snapshot {exc}") from exc
    seen_paths: set[Path] = set()
    for document in run_snapshot.documents:
        path = document.path
        route = session._trajectory_route
        if route is not None and document.trajectory_id == session.layout.session_id:
            selected = route.full if run_snapshot.status == "complete" else route.partial
            if path not in (selected.requested, selected.frozen_path):
                raise ArtifactVisibilityError("run snapshot root path does not match its paired route")
            if selected.frozen_path is None:
                raise ArtifactVisibilityError("run snapshot root has no frozen destination")
            path = selected.frozen_path
        elif route is not None:
            try:
                path.relative_to(route.run_dir)
            except ValueError as exc:
                raise ArtifactVisibilityError("run snapshot child path is outside its private run") from exc
        if path in seen_paths:
            raise ArtifactVisibilityError("run snapshot contains duplicate document identity")
        seen_paths.add(path)
        try:
            relative = path.relative_to(session.layout.live_root)
        except ValueError as exc:
            raise ArtifactVisibilityError("run snapshot document path is outside live artifacts") from exc
        filesystem._validate_relative_name(relative.as_posix())
        cursor = session.layout.live_root
        for part in relative.parts[:-1]:
            cursor /= part
            if cursor.exists() or cursor.is_symlink():
                metadata = cursor.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise ArtifactVisibilityError("run snapshot path has unsafe ancestry")
        filesystem._atomic_bytes(session.layout.live_root / relative, document.json_bytes)
    entries = filesystem.manifest_tree(session.layout.live_root)
    frozen_root = session.layout.live_root.parent / "frozen"
    if frozen_root.exists() or frozen_root.is_symlink():
        raise ArtifactVisibilityError("frozen artifact tree already exists")
    filesystem._copy_tree(session.layout.live_root, frozen_root, entries)
    filesystem._atomic_json(frozen_root.parent / "frozen-manifest.json", filesystem._manifest_payload(entries))
    result = ArtifactTreeSnapshot(
        session_id=session.layout.session_id,
        workspace_key=session.layout.workspace_key,
        root=frozen_root,
        manifest=entries,
        destinations=tuple(session._destinations),
    )
    session._frozen_snapshot = result
    session._state = _SessionState.FROZEN
    return result


def finalize_frozen(
    session: ArtifactSession, snapshot: ArtifactTreeSnapshot, *, disposition: ArtifactDisposition
) -> None:
    if session._state is not _SessionState.FROZEN:
        raise ArtifactVisibilityError("artifact session is not ready for frozen publication")
    if snapshot is not session._frozen_snapshot:
        raise ArtifactVisibilityError("frozen artifact snapshot identity mismatch")
    if not isinstance(disposition, ArtifactDisposition):
        raise ArtifactVisibilityError("artifact disposition is unsupported")
    if len(snapshot.destinations) != len(session._destinations) or any(
        actual is not expected for actual, expected in zip(snapshot.destinations, session._destinations, strict=True)
    ):
        raise ArtifactVisibilityError("frozen artifact destination identity mismatch")
    if filesystem.manifest_tree(snapshot.root) != snapshot.manifest:
        raise ArtifactVisibilityError("frozen artifact snapshot changed before publication")
    if disposition is ArtifactDisposition.ROLLBACK:
        session._restore_prior()
        session._retire_late_paths()
        session._state = _SessionState.PUBLISHED
        return
    public_entries = tuple(
        entry
        for entry in snapshot.manifest
        if entry.path == _DAYDREAM or entry.path.startswith(f"{_DAYDREAM}/") or entry.path == _REVIEW_OUTPUT
    )
    transaction_id = f"publish-{session.layout.session_id}-{secrets.token_hex(8)}"
    transaction = session.layout.state_root / "transactions" / transaction_id
    transaction.mkdir(parents=True)
    transactions._write_transaction_owner(
        transaction,
        workspace_key=session.layout.workspace_key,
        session_id=session.layout.session_id,
        kind="publish",
    )
    mark = partial(
        transactions._write_transition,
        transaction / "journal.json",
        transaction_id=transaction_id,
        session_id=session.layout.session_id,
    )
    publication_stage: Path | None = None
    try:
        filesystem._atomic_json(transaction / "publish-manifest.json", filesystem._manifest_payload(public_entries))
        publication_stage = publication._create_source_stage(
            session.layout.source,
            transaction_id,
            "publish",
            workspace_key=session.layout.workspace_key,
        )
        public_stage = publication_stage / "public"
        filesystem._copy_tree(snapshot.root, public_stage, public_entries)
        canonical_stage = transaction / "canonical-stage"
        filesystem._copy_tree(snapshot.root, canonical_stage, public_entries)
        publish_records: list[_DestinationRecord] = []
        for index, item in enumerate(session._routed):
            destination, baseline_record = item.route, item.record
            baseline_root = session._detach_transaction / f"destination-{index:04d}-baseline"
            publish_baseline = transaction / f"destination-{index:04d}-baseline"
            filesystem._copy_tree(baseline_root, publish_baseline, baseline_record.baseline)
            projection = publication_stage / f"destination-{index:04d}"
            filesystem._copy_tree(baseline_root, projection, baseline_record.baseline)
            if destination.delivery is DestinationDelivery.LIVE_EXTERNAL:
                actual = filesystem.manifest_tree(Path(baseline_record.base), (baseline_record.relative,))
                root_entry = next((entry for entry in actual if entry.path == baseline_record.relative), None)
                installed_matches = (
                    root_entry is not None
                    and root_entry.kind == "file"
                    and root_entry.sha256 == baseline_record.installed_sha256
                )
                if installed_matches:
                    metadata = Path(baseline_record.requested).lstat()
                    installed_matches = (
                        not stat.S_ISLNK(metadata.st_mode)
                        and stat.S_ISREG(metadata.st_mode)
                        and (metadata.st_dev, metadata.st_ino)
                        == (baseline_record.expected_dev, baseline_record.expected_ino)
                    )
                if actual != baseline_record.baseline and not installed_matches:
                    raise ArtifactVisibilityError("external artifact destination changed before finalization")
                published = actual
            elif destination.delivery is DestinationDelivery.DEFERRED:
                if destination.frozen_path is None:
                    raise ArtifactVisibilityError("deferred destination has no frozen path")
                write_relative = destination.frozen_path.relative_to(session.layout.live_root).as_posix()
                published = publication._overlay_destination(
                    snapshot.root, write_relative, projection, baseline_record
                )
            elif item.late is None:
                published = baseline_record.baseline
            else:
                published = publication._overlay_destination(
                    item.late.parent, item.late.name, projection, baseline_record
                )
            publish_records.append(replace(baseline_record, published=published))
        session._retire_late_paths()
        ledger._write_destination_records(
            transaction, publish_records, include_published=True, include_baseline=True
        )
        external._inherit_external_capability_proofs(session._detach_transaction, transaction)
        mark(state=_Transition.PUBLISH_STAGED)
    except BaseException:
        if publication_stage is not None:
            publication._retire_source_stage(
                session.layout.source, transaction_id, "publish", workspace_key=session.layout.workspace_key
            )
        transactions._retire_transaction(
            session.layout.state_root, transaction, terminal_state=_TerminalState.PUBLISH_RECONCILED
        )
        raise
    session._state = _SessionState.PUBLISHING
    (transaction / "public-backup").mkdir()
    _fsync_directory(transaction)
    mark(state=_Transition.PUBLISH_BACKED_UP)
    publication._replace_public_from_tree(session.layout.source, public_stage, public_entries)
    for index, record in enumerate(publish_records):
        if record.delivery is DestinationDelivery.LIVE_EXTERNAL:
            continue
        publication._replace_destination_from_tree(
            publication_stage / f"destination-{index:04d}",
            record,
            allowed=(record.baseline,),
            desired=record.published,
            transaction=transaction,
            workspace_key=session.layout.workspace_key,
            source=session.layout.source,
        )
    mark(state=_Transition.PUBLISH_INSTALLED)
    canonical = session.layout.state_root / "canonical"
    os.replace(canonical, transaction / "old-canonical")
    os.replace(canonical_stage, canonical)
    _fsync_directory(session.layout.state_root)
    filesystem._atomic_json(
        session.layout.state_root / "canonical-manifest.json", filesystem._manifest_payload(public_entries)
    )
    mark(state=_Transition.PUBLISH_VERIFIED)
    session._canonical_entries = public_entries
    transactions._retire_transaction(
        session.layout.state_root,
        session._detach_transaction,
        terminal_state=_TerminalState.DETACHED_RECONCILED,
    )
    publication._retire_source_stage(
        session.layout.source, transaction_id, "publish", workspace_key=session.layout.workspace_key
    )
    transactions._retire_transaction(
        session.layout.state_root, transaction, terminal_state=_TerminalState.PUBLISH_VERIFIED
    )
    session._state = _SessionState.PUBLISHED


def restore_prior(session: ArtifactSession) -> None:
    state_root = session.layout.state_root
    transaction = session._detach_transaction
    if not transaction.exists():
        # Already retired or already taken out of the replay path by an
        # earlier attempt (``finalize_frozen`` and session close both call
        # this). There is nothing left to restore from.
        return
    canonical = state_root / "canonical"
    publication._reset_source_stage(
        session.layout.source, transaction.name, "restore", workspace_key=session.layout.workspace_key
    )
    source_stage = publication._create_source_stage(
        session.layout.source,
        transaction.name,
        "restore",
        workspace_key=session.layout.workspace_key,
    )
    # Destination failures and public failures need opposite treatment, so
    # they are collected apart rather than into one list.
    destination_errors: list[Exception] = []
    public_errors: list[Exception] = []
    try:
        publication._restore_destination_records(session.layout.source, transaction, session._records(), source_stage)
    except Exception as exc:
        destination_errors.append(exc)
    try:
        projection = source_stage / "public"
        filesystem._copy_tree(canonical, projection, session._canonical_entries)
        publication._replace_public_from_tree(session.layout.source, projection, session._canonical_entries)
    except Exception as exc:
        public_errors.append(exc)
    try:
        publication._retire_source_stage(
            session.layout.source, transaction.name, "restore", workspace_key=session.layout.workspace_key
        )
    except Exception as exc:
        destination_errors.append(exc)
    try:
        external._cleanup_external_directories(transaction, session._created_external_parents)
        session._created_external_parents.clear()
    except Exception as exc:
        destination_errors.append(exc)
    if public_errors:
        # The journal has to stay: the next session open replays it and
        # reinstalls the public tree from canonical before anything can
        # adopt the missing tree as a new baseline. Name it instead.
        raise transactions._transaction_context(public_errors[0], state_root, transaction)
    try:
        session._retire_late_paths()
    except Exception as exc:
        destination_errors.append(exc)
    if destination_errors:
        raise transactions._clear_failed_restore(state_root, session.layout.source, transaction, destination_errors[0])
    transactions._retire_transaction(state_root, transaction, terminal_state=_TerminalState.DETACHED_RECONCILED)
