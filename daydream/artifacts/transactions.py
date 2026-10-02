"""Durable artifact transactions, recovery, and retirement."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from daydream.artifacts import external, filesystem, ledger, publication, transfer
from daydream.artifacts.models import (
    _DAYDREAM,
    _REVIEW_OUTPUT,
    _SCHEMA_VERSION,
    ArtifactManifestEntry,
    ArtifactVisibilityError,
    _ExternalEntryLifecycle,
    _TerminalState,
    _Transition,
)
from daydream.json_utils import _fsync_directory

_cleanup_observer: Any | None = None


def _cleanup_root(state_root: Path) -> Path:
    root = state_root / "cleanup"
    filesystem._create_private_directory(root)
    return root


def _load_cleanup_ticket(path: Path, state_root: Path) -> dict[str, object]:
    payload = filesystem._load_json(path)
    if (
        set(payload) != {"schema_version", "workspace_key", "transaction_id", "terminal_state", "stage_ids"}
        or payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("workspace_key") != state_root.name
        or not isinstance(payload.get("transaction_id"), str)
        or payload.get("terminal_state") not in tuple(_TerminalState)
        or not isinstance(payload.get("stage_ids"), list)
        or not all(isinstance(value, str) for value in cast(list[object], payload["stage_ids"]))
    ):
        raise ArtifactVisibilityError("artifact cleanup ticket is malformed")
    transaction_id = cast(str, payload["transaction_id"])
    if path.name != f"{transaction_id}.json" or any(character in transaction_id for character in ("/", "\\", "\0")):
        raise ArtifactVisibilityError("artifact cleanup ticket identity is malformed")
    return payload


def _notify_cleanup(state: str, transaction_id: str) -> None:
    observer = _cleanup_observer
    if observer is not None:
        observer(state, transaction_id)


def _remove_cleanup_transaction(path: Path, cleanup: Path) -> None:
    if path.parent != cleanup or path.is_symlink() or not path.is_dir():
        raise ArtifactVisibilityError("artifact cleanup transaction is unsafe")
    journal = path / "journal.json"
    if journal.is_file() and not journal.is_symlink():
        journal.unlink()
        _fsync_directory(path)
        _notify_cleanup("JOURNAL_REMOVED", path.name)
    filesystem._remove_owned_tree(path, cleanup)
    _notify_cleanup("CLEANUP_DIRECTORY_REMOVED", path.name)


def _finish_cleanup_ticket(state_root: Path, ticket: Path) -> None:
    cleanup = _cleanup_root(state_root)
    payload = _load_cleanup_ticket(ticket, state_root)
    transaction_id = cast(str, payload["transaction_id"])
    active = state_root / "transactions" / transaction_id
    retired = cleanup / transaction_id
    if active.exists() or active.is_symlink():
        if active.is_symlink() or not active.is_dir() or retired.exists() or retired.is_symlink():
            raise ArtifactVisibilityError("artifact cleanup ownership is ambiguous")
        if (active / "conflicts.json").exists() or (active / "conflicts.json").is_symlink():
            raise ArtifactVisibilityError("artifact conflict transaction is not cleanup eligible")
        stage_ids = [str(entry["stage_id"]) for entry in transfer._stage_registry(active)]
        if stage_ids != payload["stage_ids"]:
            raise ArtifactVisibilityError("artifact cleanup stage identity is malformed")
        os.replace(active, retired)
        _fsync_directory(active.parent)
        _fsync_directory(cleanup)
        _notify_cleanup("TRANSACTION_RENAMED", transaction_id)
    if retired.exists() or retired.is_symlink():
        if retired.is_symlink() or not retired.is_dir():
            raise ArtifactVisibilityError("artifact cleanup transaction is unsafe")
        _remove_cleanup_transaction(retired, cleanup)
    ticket.unlink()
    _fsync_directory(cleanup)
    _notify_cleanup("SIDECAR_REMOVED", transaction_id)


def _retire_transaction(state_root: Path, transaction: Path, *, terminal_state: _TerminalState) -> None:
    if (transaction / "conflicts.json").exists() or (transaction / "conflicts.json").is_symlink():
        raise ArtifactVisibilityError("artifact conflict transaction is not cleanup eligible")
    registry = transaction / "destinations.json"
    if registry.exists() or registry.is_symlink():
        registry_payload = filesystem._load_json(registry)
        include_published = registry_payload.get("includes_published")
        if type(include_published) is not bool:
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        destinations = ledger._load_destination_records(transaction, include_published=include_published)
    else:
        destinations = ()
    external._reconcile_external_entries(transaction, destinations)
    transfer._cleanup_registered_stages(transaction, workspace_key=state_root.name)
    _notify_cleanup("STAGES_REMOVED", transaction.name)
    cleanup = _cleanup_root(state_root)
    ticket = cleanup / f"{transaction.name}.json"
    if ticket.exists() or ticket.is_symlink():
        raise ArtifactVisibilityError("artifact cleanup ticket collision")
    filesystem._atomic_json(
        ticket,
        {
            "schema_version": _SCHEMA_VERSION,
            "workspace_key": state_root.name,
            "transaction_id": transaction.name,
            "terminal_state": terminal_state.value,
            "stage_ids": [str(entry["stage_id"]) for entry in transfer._stage_registry(transaction)],
        },
    )
    _notify_cleanup("TICKET_FSYNCED", transaction.name)
    _finish_cleanup_ticket(state_root, ticket)


def _recover_cleanup(state_root: Path) -> None:
    cleanup = state_root / "cleanup"
    if not cleanup.exists() and not cleanup.is_symlink():
        return
    if cleanup.is_symlink() or not cleanup.is_dir():
        raise ArtifactVisibilityError("artifact cleanup root is unsafe")
    tickets: dict[str, Path] = {}
    retired: set[str] = set()
    for entry in cleanup.iterdir():
        if entry.is_symlink():
            raise ArtifactVisibilityError("artifact cleanup residue is unsafe")
        if entry.is_file() and entry.suffix == ".json":
            payload = _load_cleanup_ticket(entry, state_root)
            tickets[cast(str, payload["transaction_id"])] = entry
        elif entry.is_dir():
            retired.add(entry.name)
        else:
            raise ArtifactVisibilityError("artifact cleanup residue is unsafe")
    if retired - tickets.keys():
        raise ArtifactVisibilityError("artifact cleanup transaction has no sidecar")
    for transaction_id, ticket in sorted(tickets.items()):
        if transaction_id in retired or (state_root / "transactions" / transaction_id).exists():
            _finish_cleanup_ticket(state_root, ticket)
        else:
            ticket.unlink()
            _fsync_directory(cleanup)


def _write_transition(journal: Path, *, transaction_id: str, session_id: str, state: _Transition) -> None:
    filesystem._atomic_json(
        journal,
        {
            "schema_version": _SCHEMA_VERSION,
            "transaction_id": transaction_id,
            "session_id": session_id,
            "state": state.value,
        },
    )
    observer = _transition_observer
    if observer is not None:
        observer(state.value)


_transition_observer: Any | None = None


def _load_transition(journal: Path, transaction_id: str) -> tuple[_Transition, str]:
    payload = filesystem._load_json(journal)
    if set(payload) != {"schema_version", "transaction_id", "session_id", "state"}:
        raise ArtifactVisibilityError("artifact journal schema is malformed")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != _SCHEMA_VERSION
        or payload["transaction_id"] != transaction_id
    ):
        raise ArtifactVisibilityError("artifact journal identity is malformed")
    session_id = payload["session_id"]
    try:
        state = _Transition(payload["state"])
    except ValueError as exc:
        raise ArtifactVisibilityError("artifact journal state is malformed") from exc
    if not isinstance(session_id, str):
        raise ArtifactVisibilityError("artifact journal state is malformed")
    return state, session_id


def _write_transaction_owner(
    transaction: Path,
    *,
    workspace_key: str,
    session_id: str,
    kind: Literal["detach", "publish"],
) -> None:
    filesystem._atomic_json(
        transaction / "transaction-owner.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "workspace_key": workspace_key,
            "transaction_id": transaction.name,
            "session_id": session_id,
            "kind": kind,
        },
    )


def _load_transaction_owner(transaction: Path, state_root: Path) -> dict[str, object]:
    payload = filesystem._load_json(transaction / "transaction-owner.json")
    if (
        set(payload) != {"schema_version", "workspace_key", "transaction_id", "session_id", "kind"}
        or payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("workspace_key") != state_root.name
        or payload.get("transaction_id") != transaction.name
        or not isinstance(payload.get("session_id"), str)
        or payload.get("kind") not in ("detach", "publish")
    ):
        raise ArtifactVisibilityError("artifact transaction owner is malformed")
    return payload


def _quarantine_transaction(state_root: Path, transaction: Path) -> Path:
    """Move failed recovery out of automatic replay while retaining every baseline byte for
    the operator. Deleting it would lose detached destination content; keeping it in
    transactions would repeat the failure on every session open.
    """
    root = state_root / "unreconciled"
    filesystem._create_private_directory(root)
    preserved = root / transaction.name
    if preserved.exists() or preserved.is_symlink():
        raise ArtifactVisibilityError("artifact unreconciled transaction collision")
    os.replace(transaction, preserved)
    _fsync_directory(transaction.parent)
    _fsync_directory(root)
    return preserved


class _NamedTransactionError(ArtifactVisibilityError):
    """A storage error whose message already names its transaction and workspace."""


def _transaction_context(error: Exception, state_root: Path, transaction: Path, *, detail: str = "") -> Exception:
    """Name the failed transaction and workspace, preserving the original message
    prefix for callers that match it. Already-contextual errors are unchanged.
    """
    if isinstance(error, _NamedTransactionError):
        return error
    suffix = f" (artifact transaction {transaction.name} in workspace {state_root}{detail})"
    contextual = _NamedTransactionError(f"{error}{suffix}")
    contextual.__cause__ = error
    return contextual


def _transaction_holds_conflict(transaction: Path) -> bool:
    """Both conflict registries stop retirement and recovery: preserve the evidence
    until an operator adjudicates the divergence.
    """
    registry = transaction / "conflicts.json"
    if registry.exists() or registry.is_symlink():
        return True
    return any(
        record["lifecycle"] == _ExternalEntryLifecycle.CONFLICT.value
        for record in external._external_records(transaction)
    )


def _clear_failed_restore(state_root: Path, source: Path, transaction: Path, error: Exception) -> Exception:
    """Quarantine a failed destination restore only after the public tree has been
    reinstalled. Public-restore failures retain their journal because replay is still
    required to restore that tree.
    """
    if _transaction_holds_conflict(transaction):
        return _transaction_context(error, state_root, transaction)
    publication._reset_source_stage(source, transaction.name, "restore", workspace_key=state_root.name)
    preserved = _quarantine_transaction(state_root, transaction)
    return _transaction_context(
        error,
        state_root,
        transaction,
        detail=f"; it could not restore every explicit destination and is preserved at {preserved}",
    )


def _reconcile_failed_detach(state_root: Path, source: Path, transaction: Path) -> None:
    canonical = state_root / "canonical"
    canonical_manifest = state_root / "canonical-manifest.json"
    if not canonical.exists() or not canonical_manifest.exists():
        return
    entries = filesystem._parse_manifest(canonical_manifest)
    if filesystem.manifest_tree(canonical) != entries:
        raise ArtifactVisibilityError("canonical artifact recovery copy is corrupt")
    conflicts = publication._restore_manifest_noclobber(source, canonical, entries)
    conflict_registry = transaction / "conflicts.json"
    if conflicts or conflict_registry.exists() or conflict_registry.is_symlink():
        raise ArtifactVisibilityError("artifact recovery retained a concurrent replacement conflict")
    _retire_transaction(state_root, transaction, terminal_state=_TerminalState.DETACHED_RECONCILED)


def _recover_transactions(state_root: Path, source: Path) -> None:
    _recover_cleanup(state_root)
    transactions = state_root / "transactions"
    if not transactions.exists():
        return
    if transactions.is_symlink() or not transactions.is_dir():
        raise ArtifactVisibilityError("artifact transaction root is unsafe")
    records: list[tuple[Path, _Transition, str]] = []
    for transaction in sorted(transactions.iterdir(), key=lambda path: path.name):
        if transaction.is_symlink() or not transaction.is_dir():
            raise ArtifactVisibilityError("artifact transaction residue is unsafe")
        journal = transaction / "journal.json"
        if not journal.is_file():
            publishing = _load_transaction_owner(transaction, state_root)["kind"] == "publish"
            source_stage = publication._source_stage_path(source, transaction.name, "publish") if publishing else None
            if source_stage is not None and (source_stage.exists() or source_stage.is_symlink()):
                publication._retire_source_stage(source, transaction.name, "publish", workspace_key=state_root.name)
            _retire_transaction(
                state_root,
                transaction,
                terminal_state=(
                    _TerminalState.PUBLISH_RECONCILED if publishing else _TerminalState.DETACHED_RECONCILED
                ),
            )
            continue
        state, session_id = _load_transition(journal, transaction.name)
        records.append((transaction, state, session_id))

    completed_sessions: set[str] = set()
    records.sort(key=lambda item: (not item[1].is_publish, item[0].name))
    for transaction, state, session_id in records:
        try:
            _recover_transaction(state_root, source, transaction, state, session_id, completed_sessions)
        except Exception as exc:
            raise _transaction_context(exc, state_root, transaction) from exc


def _recover_transaction(
    state_root: Path,
    source: Path,
    transaction: Path,
    state: _Transition,
    session_id: str,
    completed_sessions: set[str],
) -> None:
    """Reconcile one journalled transaction, retiring it when it is settled."""
    if (transaction / "conflicts.json").exists() or (transaction / "conflicts.json").is_symlink():
        raise ArtifactVisibilityError("artifact recovery retained a closed conflict")
    if session_id in completed_sessions and not state.is_publish:
        _retire_transaction(state_root, transaction, terminal_state=_TerminalState.DETACHED_RECONCILED)
        return
    canonical = state_root / "canonical"
    canonical_manifest = state_root / "canonical-manifest.json"
    transaction_manifest = transaction / "manifest.json"
    published_entries: tuple[ArtifactManifestEntry, ...] | None = None
    destination_records = ledger._load_destination_records(transaction, include_published=state.is_publish)
    external._reconcile_external_entries(transaction, destination_records)
    publication_stage = (
        publication._source_stage_path(source, transaction.name, "publish") if state.is_publish else None
    )
    restore_stage: Path | None = None
    if state.is_publish:
        published_entries = filesystem._parse_manifest(transaction / "publish-manifest.json")
    if state is _Transition.DETACH_STAGED:
        entries = filesystem._parse_manifest(transaction_manifest)
        stage = transaction / "detach-stage"
        if not canonical.exists():
            if filesystem.manifest_tree(stage) != entries:
                raise ArtifactVisibilityError("staged detach is incomplete")
            os.replace(stage, canonical)
            _fsync_directory(state_root)
        else:
            if filesystem.manifest_tree(canonical) != entries:
                raise ArtifactVisibilityError("staged detach ownership is ambiguous")
            if stage.exists() and filesystem.manifest_tree(stage) != entries:
                raise ArtifactVisibilityError("staged detach ownership is ambiguous")
        if not canonical_manifest.exists():
            filesystem._atomic_json(canonical_manifest, filesystem._manifest_payload(entries))
    if state is _Transition.PUBLISH_INSTALLED and not canonical.exists():
        replacement = transaction / "canonical-stage"
        if published_entries is None or filesystem.manifest_tree(replacement) != published_entries:
            raise ArtifactVisibilityError("staged canonical publication copy is corrupt")
        os.replace(replacement, canonical)
        _fsync_directory(state_root)
    if not canonical.exists() or not canonical_manifest.exists():
        raise ArtifactVisibilityError("canonical artifact recovery copy is missing")
    canonical_entries = filesystem._parse_manifest(canonical_manifest)
    canonical_actual = filesystem.manifest_tree(canonical)
    if canonical_actual != canonical_entries and not (
        state is _Transition.PUBLISH_INSTALLED and canonical_actual == published_entries
    ):
        raise ArtifactVisibilityError("canonical artifact recovery copy is corrupt")

    if state is _Transition.PUBLISH_INSTALLED:
        assert published_entries is not None
        if filesystem.manifest_tree(source, (_DAYDREAM, _REVIEW_OUTPUT)) != published_entries:
            raise ArtifactVisibilityError("installed public artifact recovery copy is corrupt")
        replacement = transaction / "canonical-stage"
        if replacement.exists():
            if filesystem.manifest_tree(replacement) != published_entries:
                raise ArtifactVisibilityError("staged canonical publication copy is corrupt")
            backup = transaction / "old-canonical"
            os.replace(canonical, backup)
            os.replace(replacement, canonical)
            _fsync_directory(state_root)
        elif filesystem.manifest_tree(canonical) != published_entries:
            raise ArtifactVisibilityError("installed canonical publication copy is missing")
        filesystem._atomic_json(canonical_manifest, filesystem._manifest_payload(published_entries))
        canonical_entries = published_entries
    elif state is _Transition.PUBLISH_VERIFIED:
        assert published_entries is not None
        if canonical_entries != published_entries:
            raise ArtifactVisibilityError("verified canonical publication identity is corrupt")
        if filesystem.manifest_tree(source, (_DAYDREAM, _REVIEW_OUTPUT)) != canonical_entries:
            raise ArtifactVisibilityError("verified public artifact recovery copy is corrupt")
    else:
        public_entries = filesystem.manifest_tree(source, (_DAYDREAM, _REVIEW_OUTPUT))
        allowed = [canonical_entries]
        if published_entries is not None:
            allowed.append(published_entries)
        if not any(filesystem._manifest_is_subset(public_entries, candidate) for candidate in allowed):
            transfer._append_conflict(
                transaction,
                record_id="public",
                reason="unexpected_replacement",
                expected_sha256=None,
                observed_sha256=None,
                expected_kind="directory",
                observed_kind="unknown",
                stage_id=None,
            )
            raise ArtifactVisibilityError("public artifacts changed during recovery")
        if public_entries:
            transfer._remove_manifested(
                source,
                public_entries,
                transaction=transaction,
                workspace_key=state_root.name,
                purpose="recover-public",
                stage_parent=source.parent,
            )
        publication._reset_source_stage(source, transaction.name, "restore", workspace_key=state_root.name)
        restore_stage = publication._create_source_stage(
            source, transaction.name, "restore", workspace_key=state_root.name
        )
        public_stage = restore_stage / "public"
        filesystem._copy_tree(canonical, public_stage, canonical_entries)
        publication._replace_public_from_tree(source, public_stage, canonical_entries)
    if state in (_Transition.PUBLISH_INSTALLED, _Transition.PUBLISH_VERIFIED):
        for record in destination_records:
            if filesystem.manifest_tree(Path(record.base), (record.relative,)) != record.published:
                raise ArtifactVisibilityError("installed explicit artifact recovery copy is corrupt")
    else:
        assert restore_stage is not None
        if destination_records:
            try:
                publication._restore_destination_records(source, transaction, destination_records, restore_stage)
            except Exception as exc:
                raise _clear_failed_restore(state_root, source, transaction, exc) from exc
        publication._retire_source_stage(source, transaction.name, "restore", workspace_key=state_root.name)
    if state.is_publish:
        completed_sessions.add(session_id)
        if publication_stage is None:
            raise ArtifactVisibilityError("artifact publication stage is missing")
        if publication_stage.exists() or publication_stage.is_symlink():
            publication._retire_source_stage(source, transaction.name, "publish", workspace_key=state_root.name)
        elif state is not _Transition.PUBLISH_VERIFIED:
            raise ArtifactVisibilityError("artifact publication stage is missing")
    _retire_transaction(
        state_root,
        transaction,
        terminal_state=(
            _TerminalState.PUBLISH_VERIFIED
            if state is _Transition.PUBLISH_VERIFIED
            else _TerminalState.PUBLISH_RECONCILED
            if state.is_publish
            else _TerminalState.DETACHED_RECONCILED
        ),
    )
