"""Public artifact validation, staged installation, and baseline restoration."""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from daydream.artifacts import external, filesystem, transfer
from daydream.artifacts.models import (
    _DAYDREAM,
    _OPERATIONAL_NAMES,
    _PUBLIC_ANCHORS,
    _PUBLIC_DIRECTORY_ANCHORS,
    _REVIEW_OUTPUT,
    _SCHEMA_VERSION,
    ArtifactManifestEntry,
    ArtifactVisibilityError,
    DestinationDelivery,
    OutputLabel,
    _DestinationRecord,
    _ExternalEntryLifecycle,
    _ExternalEntryPurpose,
)
from daydream.artifacts.ownership import SourceOwnership
from daydream.json_utils import _fsync_directory


def _validated_canonical_entries(
    state_root: Path,
) -> tuple[ArtifactManifestEntry, ...] | None:
    """Return an attested prior public tree, if this workspace has one."""
    canonical = state_root / "canonical"
    if not canonical.exists() and not canonical.is_symlink():
        return None
    entries = filesystem._parse_manifest(state_root / "canonical-manifest.json")
    if filesystem.manifest_tree(canonical) != entries:
        raise ArtifactVisibilityError("canonical artifact recovery copy is corrupt")
    return entries


def _validate_public_tree(
    source: Path,
    *,
    canonical_entries: tuple[ArtifactManifestEntry, ...] | None,
) -> tuple[ArtifactManifestEntry, ...]:
    """Admit current public artifact roots, rejecting operational namespaces and unsafe entries."""
    daydream = source / _DAYDREAM
    if daydream.exists() or daydream.is_symlink():
        try:
            metadata = daydream.lstat()
        except OSError as exc:
            raise ArtifactVisibilityError("artifact root could not be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactVisibilityError("artifact roots accept only regular files and directories")
        children = {child.name for child in daydream.iterdir()}
        if children & _OPERATIONAL_NAMES:
            raise ArtifactVisibilityError("unsupported operational workspace blocks artifact detach")
        anchor_children = children
        for name in sorted(anchor_children & _PUBLIC_ANCHORS):
            anchor = daydream / name
            try:
                anchor_metadata = anchor.lstat()
            except OSError as exc:
                raise ArtifactVisibilityError("public artifact anchor could not be inspected") from exc
            expected_directory = name in _PUBLIC_DIRECTORY_ANCHORS
            if stat.S_ISLNK(anchor_metadata.st_mode) or (
                expected_directory != stat.S_ISDIR(anchor_metadata.st_mode)
                or not (stat.S_ISDIR(anchor_metadata.st_mode) or stat.S_ISREG(anchor_metadata.st_mode))
            ):
                raise ArtifactVisibilityError("public artifact anchor has the wrong filesystem type")
    review = source / _REVIEW_OUTPUT
    if review.exists() or review.is_symlink():
        metadata = review.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArtifactVisibilityError("artifact roots accept only regular files and directories")
    public_entries = filesystem.manifest_tree(source, (_DAYDREAM, _REVIEW_OUTPUT))
    daydream_entry = filesystem._entry_at(public_entries, _DAYDREAM)
    if daydream_entry is not None and daydream_entry.kind != "directory":
        raise ArtifactVisibilityError("artifact roots accept only regular files and directories")
    review_entry = filesystem._entry_at(public_entries, _REVIEW_OUTPUT)
    if review_entry is not None and review_entry.kind != "file":
        raise ArtifactVisibilityError("artifact roots accept only regular files and directories")
    daydream_prefix = f"{_DAYDREAM}/"
    manifested_children = {
        entry.path.removeprefix(daydream_prefix): entry
        for entry in public_entries
        if entry.path.startswith(daydream_prefix) and "/" not in entry.path.removeprefix(daydream_prefix)
    }
    if manifested_children.keys() & _OPERATIONAL_NAMES:
        raise ArtifactVisibilityError("unsupported operational workspace blocks artifact detach")
    anchor_children = set(manifested_children)
    for name in sorted(anchor_children & _PUBLIC_ANCHORS):
        expected_kind = "directory" if name in _PUBLIC_DIRECTORY_ANCHORS else "file"
        if manifested_children[name].kind != expected_kind:
            raise ArtifactVisibilityError("public artifact anchor has the wrong filesystem type")
    nonstatic_children = anchor_children - _PUBLIC_ANCHORS
    for name in sorted(nonstatic_children):
        relative = f"{_DAYDREAM}/{name}"
        prefix = f"{relative}/"
        actual = tuple(entry for entry in public_entries if entry.path == relative or entry.path.startswith(prefix))
        expected = tuple(
            entry for entry in canonical_entries or () if entry.path == relative or entry.path.startswith(prefix)
        )
        if not expected or actual != expected:
            raise ArtifactVisibilityError("public .daydream tree contains an unregistered artifact anchor")
    return public_entries


def _replace_public_from_tree(
    source: SourceOwnership, tree: Path, entries: tuple[ArtifactManifestEntry, ...],
) -> None:
    """Install a staged public tree onto a source whose own tree is already detached."""
    with source.operation():
        if filesystem.manifest_tree(source.path, (_DAYDREAM, _REVIEW_OUTPUT)):
            raise ArtifactVisibilityError("public artifacts changed before publication")
        for name in (_DAYDREAM, _REVIEW_OUTPUT):
            staged = tree / name
            if staged.exists() or staged.is_symlink():
                _install_staged_path(staged, source.path / name, source=source)
        if filesystem.manifest_tree(source.path, (_DAYDREAM, _REVIEW_OUTPUT)) != entries:
            raise ArtifactVisibilityError("public artifact publication verification failed")


def _install_staged_path(staged: Path, target: Path, *, source: SourceOwnership) -> None:
    metadata = staged.lstat()
    try:
        with source.parent_descriptor(target) as parent_fd:
            if stat.S_ISREG(metadata.st_mode):
                os.link(staged, target.name, dst_dir_fd=parent_fd, follow_symlinks=False)
                _fsync_installed_file(parent_fd, target.name)
                os.fsync(parent_fd)
            elif stat.S_ISDIR(metadata.st_mode):
                os.replace(staged, target.name, dst_dir_fd=parent_fd)
                os.fsync(parent_fd)
            else:
                raise ArtifactVisibilityError("artifact source stage has an unsafe entry")
        if stat.S_ISREG(metadata.st_mode):
            staged.unlink()
            _fsync_directory(staged.parent)
    except OSError as exc:
        raise ArtifactVisibilityError("artifact destination changed during atomic install") from exc


def _fsync_installed_file(parent_fd: int, name: str) -> None:
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_regular_noclobber(
    staged_source: Path, target: Path, entry: ArtifactManifestEntry, *, source: SourceOwnership,
) -> None:
    # Manifested parent directories are installed first. Creating parents here
    # could recreate a checkout that disappeared during recovery.
    staged = target.parent / f".{target.name}.{secrets.token_hex(8)}.install"
    with source.parent_descriptor(target) as parent_fd:
        try:
            filesystem._copy_regular_entry(
                staged_source, staged, entry, changed_message="artifact recovery source changed", create_parents=False,
                destination_fd=parent_fd,
            )
            try:
                with source.operation():
                    os.link(staged.name, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
            except FileExistsError as exc:
                raise ArtifactVisibilityError("artifact destination changed during no-clobber install") from exc
            _fsync_installed_file(parent_fd, target.name)
            os.fsync(parent_fd)
        finally:
            with suppress(OSError):
                os.unlink(staged.name, dir_fd=parent_fd)
                os.fsync(parent_fd)


def _restore_manifest_noclobber(
    source: SourceOwnership,
    canonical: Path,
    entries: tuple[ArtifactManifestEntry, ...],
) -> tuple[str, ...]:
    with source.operation():
        conflicts: list[str] = []
        for entry in entries:
            if entry.kind != "directory":
                continue
            target = source.path / entry.path
            if target.exists() or target.is_symlink():
                metadata = target.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    conflicts.append(entry.path)
                continue
            source.mkdir(target, mode=entry.mode)
            with source.parent_descriptor(target) as parent_fd:
                descriptor = os.open(
                    target.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                try:
                    os.fchmod(descriptor, entry.mode)
                finally:
                    os.close(descriptor)
        for entry in entries:
            if entry.kind != "file":
                continue
            target = source.path / entry.path
            if target.exists() or target.is_symlink():
                try:
                    actual = filesystem.manifest_tree(source.path, (entry.path,))
                except ArtifactVisibilityError:
                    conflicts.append(entry.path)
                    continue
                if actual != (entry,):
                    conflicts.append(entry.path)
                continue
            _install_regular_noclobber(canonical / entry.path, target, entry, source=source)
        return tuple(conflicts)


def _source_stage_path(source: SourceOwnership, transaction_id: str, purpose: str) -> Path:
    if (
        not transaction_id
        or any(character in transaction_id for character in ("/", "\\", "\0"))
        or purpose not in ("publish", "restore")
    ):
        raise ArtifactVisibilityError("artifact source-stage identity is invalid")
    return source.path.parent / f".{source.path.name}.daydream-{purpose}-{transaction_id}"


def _source_stage_owner(
    source: SourceOwnership, transaction_id: str, purpose: str, workspace_key: str,
) -> dict[str, object]:
    """Derive the owner attestation a source stage carries, re-stat'ing its parent."""
    parent = source.path.parent.lstat()
    return {
        "schema_version": _SCHEMA_VERSION,
        "workspace_key": workspace_key,
        "transaction_id": transaction_id,
        "purpose": purpose,
        "parent_dev": parent.st_dev,
        "parent_ino": parent.st_ino,
    }


def _create_source_stage(source: SourceOwnership, transaction_id: str, purpose: str, *, workspace_key: str) -> Path:
    stage = _source_stage_path(source, transaction_id, purpose)
    if stage.exists() or stage.is_symlink():
        raise ArtifactVisibilityError("artifact source-stage collision")
    stage.mkdir(mode=0o700)
    filesystem._atomic_json(
        stage / "stage-owner.json", _source_stage_owner(source, transaction_id, purpose, workspace_key)
    )
    _fsync_directory(source.path.parent)
    return stage


def _retire_source_stage(source: SourceOwnership, transaction_id: str, purpose: str, *, workspace_key: str) -> None:
    """Re-attest this identity's source stage, then remove the tree it owns."""
    stage = _source_stage_path(source, transaction_id, purpose)
    expected = _source_stage_owner(source, transaction_id, purpose, workspace_key)
    if filesystem._load_json(stage / "stage-owner.json") != expected:
        raise ArtifactVisibilityError("artifact source-stage ownership is malformed")
    filesystem._remove_owned_tree(stage, source.path.parent)


def _reset_source_stage(source: SourceOwnership, transaction_id: str, purpose: str, *, workspace_key: str) -> None:
    """Reclaim only a stage attesting this exact workspace, transaction, purpose, and
    parent. Its contents are regenerable from durable baselines; foreign, tampered, or
    unattested residue remains a collision.
    """
    stage = _source_stage_path(source, transaction_id, purpose)
    if not stage.exists() and not stage.is_symlink():
        return
    if stage.is_symlink() or not stage.is_dir():
        return
    owner = stage / "stage-owner.json"
    if not owner.is_file():
        return
    if filesystem._load_json(owner) != _source_stage_owner(source, transaction_id, purpose, workspace_key):
        return
    filesystem._remove_owned_tree(stage, source.path.parent)


def _overlay_destination(
    frozen_root: Path,
    write_relative: str,
    projection: Path,
    record: _DestinationRecord,
) -> tuple[ArtifactManifestEntry, ...]:
    generated = filesystem.manifest_tree(frozen_root, (write_relative,))
    if not generated:
        return record.baseline
    root_entry = next((entry for entry in generated if entry.path == write_relative), None)
    expects_directory = record.label is OutputLabel.DUMP_DIRECTORY
    if root_entry is None or (root_entry.kind == "directory") is not expects_directory:
        raise ArtifactVisibilityError("generated artifact destination has the wrong filesystem type")
    source_prefix = Path(write_relative)
    target_prefix = Path(record.relative)
    for entry in generated:
        suffix = Path(entry.path).relative_to(source_prefix)
        target_relative = target_prefix / suffix
        target = projection / target_relative
        source = frozen_root / entry.path
        if entry.kind == "directory":
            if target.exists() or target.is_symlink():
                metadata = target.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise ArtifactVisibilityError("dump output conflicts with a baseline file")
            else:
                target.mkdir(parents=True, mode=entry.mode)
            os.chmod(target, entry.mode)
            continue
        if target.exists() or target.is_symlink():
            metadata = target.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ArtifactVisibilityError("generated output conflicts with a baseline directory")
            target.unlink()
        filesystem._copy_regular_entry(source, target, entry)
    for directory in sorted(
        (path for path in projection.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(projection)
    return filesystem.manifest_tree(projection, (record.relative,))


def _replace_destination_from_tree(
    tree: Path,
    record: _DestinationRecord,
    *,
    allowed: Sequence[tuple[ArtifactManifestEntry, ...]],
    desired: tuple[ArtifactManifestEntry, ...],
    transaction: Path,
    workspace_key: str,
    source: SourceOwnership,
) -> None:
    with source.operation():
        root = Path(record.base)
        if record.expected_kind == "directory":
            _merge_directory_from_tree(
                tree,
                record,
                desired=desired,
                transaction=transaction,
                workspace_key=workspace_key,
                source=source,
            )
            return
        actual = filesystem.manifest_tree(root, (record.relative,))
        if not any(filesystem._manifest_is_subset(actual, candidate) for candidate in allowed):
            raise ArtifactVisibilityError("explicit artifact destination changed during publication")
        if actual:
            transfer._remove_manifested(
                root,
                actual,
                (record.relative,),
                transaction=transaction,
                workspace_key=workspace_key,
                purpose="replace-destination",
                record_id=record.record_id,
                source=source,
            )
        staged = tree / record.relative
        if desired:
            for parent_relative in record.missing_parents:
                parent = root / parent_relative
                source.mkdir(parent, mode=0o700)
            _install_staged_path(staged, root / record.relative, source=source)
        else:
            for parent_relative in reversed(record.missing_parents):
                with suppress(OSError):
                    source.rmdir(root / parent_relative)
        if filesystem.manifest_tree(root, (record.relative,)) != desired:
            raise ArtifactVisibilityError("explicit artifact destination verification failed")


def _remove_directory_destination(
    record: _DestinationRecord,
    *,
    transaction: Path,
    workspace_key: str,
    source: SourceOwnership,
) -> None:
    """Remove the validated live destination and its owned parents, restoring an originally
    absent directory.
    """
    root = Path(record.base)
    actual = filesystem.manifest_tree(root, (record.relative,))
    if actual:
        transfer._remove_manifested(
            root,
            actual,
            (record.relative,),
            transaction=transaction,
            workspace_key=workspace_key,
            purpose="remove-dump-destination",
            record_id=record.record_id,
            changed_message="dump destination changed during removal",
            source=source,
        )
    for parent_relative in reversed(record.missing_parents):
        parent = root / parent_relative
        with suppress(OSError):
            source.rmdir(parent)
    if filesystem.manifest_tree(root, (record.relative,)):
        raise ArtifactVisibilityError("dump destination removal verification failed")


def _merge_directory_from_tree(
    tree: Path,
    record: _DestinationRecord,
    *,
    desired: tuple[ArtifactManifestEntry, ...],
    transaction: Path,
    workspace_key: str,
    source: SourceOwnership,
) -> None:
    root = Path(record.base)
    target_root = root / record.relative
    baseline = {entry.path: entry for entry in record.baseline}
    desired_by_path = {entry.path: entry for entry in desired}
    if not desired:
        # An absent baseline restores to absence; dropping a real baseline
        # from the projection is a malformed restore.
        if record.baseline:
            raise ArtifactVisibilityError("dump destination projection dropped a baseline file")
        _remove_directory_destination(
            record, transaction=transaction, workspace_key=workspace_key, source=source
        )
        return
    desired_root = desired_by_path.get(record.relative)
    if desired_root is None or desired_root.kind != "directory":
        raise ArtifactVisibilityError("dump destination projection is malformed")
    if target_root.exists() or target_root.is_symlink():
        metadata = target_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactVisibilityError("dump destination shell changed")
    else:
        for parent_relative in record.missing_parents:
            source.mkdir(root / parent_relative, mode=0o700, exist_ok=True)
        source.mkdir(target_root, mode=desired_root.mode)

    for entry in desired:
        if entry.kind != "directory" or entry.path == record.relative:
            continue
        target = root / entry.path
        if target.exists() or target.is_symlink():
            metadata = target.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactVisibilityError("dump destination directory changed")
        else:
            source.mkdir(target, mode=entry.mode)

    requested = Path(record.requested)
    baseline_was_detached = requested == source.path or source.path in requested.parents
    for entry in desired:
        if entry.kind != "file":
            continue
        target = root / entry.path
        actual = filesystem.manifest_tree(root, (entry.path,))
        actual_entry = next((candidate for candidate in actual if candidate.path == entry.path), None)
        expected = baseline.get(entry.path)
        if actual_entry == entry:
            continue
        if actual_entry is not None:
            if expected is None or actual_entry != expected:
                raise ArtifactVisibilityError("dump destination changed during merge")
            transfer._remove_manifested(
                root,
                actual,
                (entry.path,),
                transaction=transaction,
                workspace_key=workspace_key,
                purpose="replace-dump-file",
                record_id=record.record_id,
                source=source,
            )
        elif expected is not None and not baseline_was_detached:
            # External baselines remain live; their disappearance is a conflict.
            # In-source baselines were deliberately transferred before dispatch.
            raise ArtifactVisibilityError("dump destination baseline disappeared")
        _install_regular_noclobber(tree / entry.path, target, entry, source=source)
    for baseline_entry in record.baseline:
        if baseline_entry.kind == "file" and baseline_entry.path not in desired_by_path:
            raise ArtifactVisibilityError("dump destination projection dropped a baseline file")


def _wrote_recorded_session(actual: Sequence[ArtifactManifestEntry], record: _DestinationRecord) -> bool:
    """Whether the live entry at ``record`` still holds bytes this session wrote."""
    root_entry = filesystem._entry_at(actual, record.relative, kind="file")
    return root_entry is not None and root_entry.sha256 in (record.prepared_sha256, record.installed_sha256)


def _reinstall_external_baseline(
    transaction: Path,
    record: _DestinationRecord,
    baseline_root: Path,
    baseline_file: ArtifactManifestEntry,
) -> None:
    """Reinstall one external destination's baseline bytes through the live path."""
    baseline_path = baseline_root / record.relative
    external._publish_live_external(
        transaction,
        record,
        filesystem._read_regular(baseline_path, baseline_path.lstat()),
        exchange=external._name_exchange_factory(),
        capability=external._recorded_external_capability(transaction, Path(record.requested).parent),
        content_mode=baseline_file.mode,
    )


def _restore_destination_records(
    source: SourceOwnership,
    transaction: Path,
    records: Sequence[_DestinationRecord],
    source_stage: Path,
) -> None:
    with source.operation():
        workspace_key = transaction.parent.parent.name
        for record in records:
            try:
                index = int(record.record_id.removeprefix("destination-"))
            except ValueError as exc:
                raise ArtifactVisibilityError("artifact destination record identity is malformed") from exc
            baseline_root = transaction / f"destination-{index:04d}-baseline"
            install_stage = source_stage / f"destination-{index:04d}"
            filesystem._copy_tree(baseline_root, install_stage, record.baseline)
            root = Path(record.base)
            actual = filesystem.manifest_tree(root, (record.relative,))
            if record.delivery is DestinationDelivery.LIVE_EXTERNAL:
                if actual == record.baseline:
                    continue
                baseline_file = filesystem._entry_at(record.baseline, record.relative, kind="file")
                if not actual and baseline_file is not None:
                    _reinstall_external_baseline(
                        transaction,
                        replace(
                            record,
                            baseline_state="absent",
                            expected_dev=None,
                            expected_ino=None,
                            prepared_sha256=None,
                            installed_sha256=None,
                        ),
                        baseline_root,
                        baseline_file,
                    )
                    continue
                if not _wrote_recorded_session(actual, record):
                    raise ArtifactVisibilityError("external trajectory conflict during recovery")
                expected: tuple[int | None, int | None] = (record.expected_dev, record.expected_ino)
                if None in expected:
                    expected = external._external_target_identity_from_ledger(transaction, record)
                requested_metadata = Path(record.requested).lstat()
                if (requested_metadata.st_dev, requested_metadata.st_ino) != expected:
                    raise ArtifactVisibilityError("external trajectory identity changed during recovery")
                if baseline_file is None:
                    transfer._remove_manifested(
                        root,
                        actual,
                        (record.relative,),
                        transaction=transaction,
                        workspace_key=workspace_key,
                        purpose="restore-live-external",
                        record_id=record.record_id,
                        source=source,
                    )
                    continue
                _reinstall_external_baseline(transaction, record, baseline_root, baseline_file)
                continue
            allowed = [record.baseline]
            if record.published:
                allowed.append(record.published)
            recorded_session = _wrote_recorded_session(actual, record)
            if not recorded_session and not any(
                filesystem._manifest_is_subset(actual, candidate) for candidate in allowed
            ):
                raise ArtifactVisibilityError("explicit artifact destination changed during recovery")
            if recorded_session:
                allowed.append(actual)
            _replace_destination_from_tree(
                install_stage,
                record,
                allowed=allowed,
                desired=record.baseline,
                transaction=transaction,
                workspace_key=workspace_key,
                source=source,
            )
        parent_indexes = [
            index
            for index, external in enumerate(external._external_records(transaction))
            if external["purpose"] == _ExternalEntryPurpose.MISSING_PARENT.value
            and external["lifecycle"] == _ExternalEntryLifecycle.ATTESTED.value
        ]
        external._cleanup_external_directories(transaction, parent_indexes)
