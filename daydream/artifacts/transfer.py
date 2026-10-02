"""Attested ownership transfers with durable intent and conflict records."""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from daydream.artifacts import filesystem
from daydream.artifacts.models import (
    _DAYDREAM,
    _REVIEW_OUTPUT,
    _SCHEMA_VERSION,
    ArtifactManifestEntry,
    ArtifactVisibilityError,
)
from daydream.json_utils import _fsync_directory

_transfer_observer: Any | None = None
_transfer_post_observer: Any | None = None


def _stage_registry(transaction: Path) -> list[dict[str, object]]:
    result = filesystem._load_ledger(
        transaction / "stages.json",
        items_key="stages",
        message="artifact transfer-stage registry is malformed",
        identity=(("transaction_id", transaction.name),),
    )
    for index, entry in enumerate(result):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"stage_id", "path", "purpose", "record_id", "parent_dev", "parent_ino"}
            or entry.get("stage_id") != f"stage-{index:04d}"
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("purpose"), str)
            or not isinstance(entry.get("record_id"), str)
            or type(entry.get("parent_dev")) is not int
            or type(entry.get("parent_ino")) is not int
        ):
            raise ArtifactVisibilityError("artifact transfer-stage registry is malformed")
    return result


def _transfer_stage_owner(entry: dict[str, object], *, transaction: Path, workspace_key: str) -> dict[str, object]:
    """Derive the owner attestation a transfer stage carries, from its registry row."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "workspace_key": workspace_key,
        "transaction_id": transaction.name,
        "stage_id": entry["stage_id"],
        "purpose": entry["purpose"],
        "record_id": entry["record_id"],
        "parent_dev": entry["parent_dev"],
        "parent_ino": entry["parent_ino"],
    }


def _create_transfer_stage(
    parent: Path,
    *,
    transaction: Path,
    workspace_key: str,
    purpose: str,
    record_id: str,
) -> Path:
    parent_metadata = parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ArtifactVisibilityError("artifact transfer-stage parent is unsafe")
    registry = _stage_registry(transaction)
    stage_id = f"stage-{len(registry):04d}"
    stage = parent / f".daydream-transfer-{transaction.name}-{stage_id}"
    entry: dict[str, object] = {
        "stage_id": stage_id,
        "path": str(stage),
        "purpose": purpose,
        "record_id": record_id,
        "parent_dev": parent_metadata.st_dev,
        "parent_ino": parent_metadata.st_ino,
    }
    filesystem._atomic_json(
        transaction / "stages.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "transaction_id": transaction.name,
            "stages": [*registry, entry],
        },
    )
    if stage.exists() or stage.is_symlink():
        raise ArtifactVisibilityError("artifact transfer-stage collision")
    stage.mkdir(mode=0o700)
    filesystem._atomic_json(
        stage / "stage-owner.json",
        _transfer_stage_owner(entry, transaction=transaction, workspace_key=workspace_key),
    )
    _fsync_directory(parent)
    return stage


def _validate_transfer_stage(stage: Path, *, transaction: Path, workspace_key: str) -> None:
    registry = _stage_registry(transaction)
    matching = next((entry for entry in registry if entry["path"] == str(stage)), None)
    if matching is None or stage.parent != Path(str(matching["path"])).parent:
        raise ArtifactVisibilityError("artifact transfer-stage ownership is missing")
    parent_metadata = stage.parent.lstat()
    if (parent_metadata.st_dev, parent_metadata.st_ino) != (matching["parent_dev"], matching["parent_ino"]):
        raise ArtifactVisibilityError("artifact transfer-stage parent identity changed")
    expected = _transfer_stage_owner(matching, transaction=transaction, workspace_key=workspace_key)
    if filesystem._load_json(stage / "stage-owner.json") != expected:
        raise ArtifactVisibilityError("artifact transfer-stage owner is malformed")


def _cleanup_registered_stages(transaction: Path, *, workspace_key: str) -> None:
    for entry in _stage_registry(transaction):
        stage = Path(cast(str, entry["path"]))
        if not stage.exists() and not stage.is_symlink():
            continue
        _validate_transfer_stage(stage, transaction=transaction, workspace_key=workspace_key)
        for intent in _load_transfer_intents(stage):
            relative = cast(str, intent["relative"])
            expected = filesystem._manifest_entry_from_payload(intent["expected"])
            moved_entries = filesystem.manifest_tree(stage / "entries", (relative,))
            moved = next((entry for entry in moved_entries if entry.path == relative), None)
            if moved is not None and moved != expected:
                _record_transfer_conflict(
                    transaction,
                    record_id=cast(str, intent["record_id"]),
                    expected=expected,
                    observed=moved,
                    stage=stage,
                )
                raise ArtifactVisibilityError("artifact recovery retained an unexpected transferred entry")
        filesystem._remove_owned_tree(stage, stage.parent)


def _load_transfer_intents(stage: Path, *, owner: dict[str, Any] | None = None) -> list[dict[str, object]]:
    path = stage / "intents.json"
    if not path.exists() and not path.is_symlink():
        return []
    if owner is None:
        owner = filesystem._load_json(stage / "stage-owner.json")
    result = filesystem._load_ledger(
        path,
        items_key="intents",
        message="artifact transfer intent is malformed",
        identity=(("stage_id", owner.get("stage_id")),),
        required=True,
    )
    for index, intent in enumerate(result):
        if (
            not isinstance(intent, dict)
            or set(intent) != {"index", "record_id", "source", "relative", "expected"}
            or intent.get("index") != index
            or intent.get("record_id") != owner.get("record_id")
            or not isinstance(intent.get("source"), str)
            or not isinstance(intent.get("relative"), str)
        ):
            raise ArtifactVisibilityError("artifact transfer intent is malformed")
        relative = cast(str, intent["relative"])
        source = Path(cast(str, intent["source"]))
        filesystem._validate_relative_name(relative)
        expected = filesystem._manifest_entry_from_payload(intent["expected"])
        if expected.path != relative or not source.is_absolute() or filesystem._absolute_lexical(source) != source:
            raise ArtifactVisibilityError("artifact transfer intent is malformed")
    return result


def _write_transfer_intents(
    stage: Path,
    entries: Sequence[tuple[Path, str, ArtifactManifestEntry]],
    *,
    append: bool = False,
) -> None:
    """Durably record every intended move before transferring any entry."""
    owner = filesystem._load_json(stage / "stage-owner.json")
    intents = _load_transfer_intents(stage, owner=owner) if append else []
    intents.extend(
        {
            "index": index,
            "record_id": owner["record_id"],
            "source": str(path),
            "relative": relative,
            "expected": asdict(expected),
        }
        for index, (path, relative, expected) in enumerate(entries, start=len(intents))
    )
    filesystem._atomic_json(
        stage / "intents.json",
        {"schema_version": _SCHEMA_VERSION, "stage_id": owner["stage_id"], "intents": intents},
    )


def _append_conflict(
    transaction: Path,
    *,
    record_id: str,
    reason: str,
    expected_sha256: str | None,
    observed_sha256: str | None,
    expected_kind: str,
    observed_kind: str,
    stage_id: str | None,
) -> None:
    """Append one adjudication record to the transaction's conflict registry."""
    path = transaction / "conflicts.json"
    conflicts = filesystem._load_ledger(
        path,
        items_key="conflicts",
        message="artifact conflict registry is malformed",
        identity=(("transaction_id", transaction.name),),
    )
    conflicts.append(
        {
            "record_id": record_id,
            "reason": reason,
            "expected_sha256": expected_sha256,
            "observed_sha256": observed_sha256,
            "expected_kind": expected_kind,
            "observed_kind": observed_kind,
            "stage_id": stage_id,
        }
    )
    filesystem._atomic_json(
        path,
        {
            "schema_version": _SCHEMA_VERSION,
            "transaction_id": transaction.name,
            "conflicts": conflicts,
        },
    )


def _record_transfer_conflict(
    transaction: Path,
    *,
    record_id: str,
    expected: ArtifactManifestEntry,
    observed: ArtifactManifestEntry,
    stage: Path,
) -> None:
    _append_conflict(
        transaction,
        record_id=record_id,
        reason="unexpected_replacement",
        expected_sha256=expected.sha256,
        observed_sha256=observed.sha256,
        expected_kind=expected.kind,
        observed_kind=observed.kind,
        stage_id=cast(str, filesystem._load_json(stage / "stage-owner.json")["stage_id"]),
    )


def _transfer_entry(
    path: Path,
    *,
    stage: Path,
    relative: str,
    expected: ArtifactManifestEntry,
    transaction: Path,
    record_intent: bool = True,
) -> None:
    moved = stage / "entries" / relative
    moved.parent.mkdir(parents=True, exist_ok=True)
    if record_intent:
        # Solo-transfer path: one fsynced intent immediately before the move.
        # Batched callers (_remove_manifested) write every intent durably up
        # front instead, so the intent ledger is never a per-entry rewrite.
        _write_transfer_intents(stage, [(path, relative, expected)], append=True)
    observer = _transfer_observer
    if observer is not None:
        observer(path, moved)
    try:
        os.replace(path, moved)
        _fsync_directory(path.parent)
        _fsync_directory(moved.parent)
    except OSError as exc:
        raise ArtifactVisibilityError("artifact entry changed during ownership transfer") from exc
    post_observer = _transfer_post_observer
    if post_observer is not None:
        post_observer(path, moved)
    observed_entries = filesystem.manifest_tree(stage / "entries", (relative,))
    observed = next((entry for entry in observed_entries if entry.path == relative), None)
    if observed != expected:
        if observed is None:
            raise ArtifactVisibilityError("artifact ownership transfer produced no entry")
        _record_transfer_conflict(
            transaction,
            record_id=str(filesystem._load_json(stage / "stage-owner.json")["record_id"]),
            expected=expected,
            observed=observed,
            stage=stage,
        )
        try:
            os.link(moved, path, follow_symlinks=False)
            _fsync_directory(path.parent)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ArtifactVisibilityError("artifact entry conflict could not be restored without clobbering") from exc
        raise ArtifactVisibilityError("artifact entry changed during ownership transfer conflict")


def _remove_manifested(
    root: Path,
    entries: tuple[ArtifactManifestEntry, ...],
    names: Sequence[str] = (_DAYDREAM, _REVIEW_OUTPUT),
    *,
    transaction: Path,
    workspace_key: str,
    purpose: str,
    stage_parent: Path,
    record_id: str = "public",
    remove_directories: bool = True,
    changed_message: str = "public artifacts changed during detach",
) -> None:
    """Stage every manifested file out of ``root``, then drop its directories."""
    if filesystem.manifest_tree(root, names) != entries:
        raise ArtifactVisibilityError(changed_message)
    stage = _create_transfer_stage(
        stage_parent,
        transaction=transaction,
        workspace_key=workspace_key,
        purpose=purpose,
        record_id=record_id,
    )
    file_entries = [entry for entry in entries if entry.kind == "file"]
    if file_entries:
        # The complete intent set is durable before the first move, including crash recovery.
        _write_transfer_intents(stage, [(root / entry.path, entry.path, entry) for entry in file_entries])
    for entry in file_entries:
        _transfer_entry(
            root / entry.path,
            stage=stage,
            relative=entry.path,
            expected=entry,
            transaction=transaction,
            record_intent=False,
        )
    if remove_directories:
        for entry in filesystem._deepest_first_directories(entries):
            target = root / entry.path
            try:
                metadata = target.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise ArtifactVisibilityError("public artifact changed during removal")
                target.rmdir()
            except ArtifactVisibilityError:
                raise
            except OSError as exc:
                raise ArtifactVisibilityError("public artifact directory changed during detach") from exc
            _fsync_directory(target.parent)
    _validate_transfer_stage(stage, transaction=transaction, workspace_key=workspace_key)
    filesystem._remove_owned_tree(stage, stage_parent)
