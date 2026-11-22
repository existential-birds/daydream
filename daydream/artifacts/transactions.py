"""Durable artifact transactions, recovery, and retirement."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from daydream.artifacts import external, filesystem, ledger, publication, transfer
from daydream.artifacts.models import (
    _EARLIER_FORMAT_STATES,
    _SCHEMA_VERSION,
    ArtifactSourceUnavailableError,
    ArtifactVisibilityError,
    _DestinationRecord,
    _ExternalEntryLifecycle,
    _TerminalState,
    _Transition,
)
from daydream.artifacts.ownership import SourceOwnership
from daydream.json_utils import _fsync_directory


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


def _remove_cleanup_transaction(path: Path, cleanup: Path) -> None:
    if path.parent != cleanup or path.is_symlink() or not path.is_dir():
        raise ArtifactVisibilityError("artifact cleanup transaction is unsafe")
    journal = path / "journal.json"
    if journal.is_file() and not journal.is_symlink():
        journal.unlink()
        _fsync_directory(path)
    filesystem._remove_owned_tree(path, cleanup)


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
    if retired.exists() or retired.is_symlink():
        if retired.is_symlink() or not retired.is_dir():
            raise ArtifactVisibilityError("artifact cleanup transaction is unsafe")
        _remove_cleanup_transaction(retired, cleanup)
    ticket.unlink()
    _fsync_directory(cleanup)


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
    if isinstance(payload["state"], str) and payload["state"] in _EARLIER_FORMAT_STATES:
        state_root = journal.parents[2]
        raise ArtifactVisibilityError(
            f"artifact transaction {transaction_id} in workspace {state_root} was written by an earlier Daydream, "
            "which kept a canonical recovery copy; this build does not recover it. Run that earlier release once in "
            f"this checkout to finish it, or restore .daydream/ and .review-output.md from {state_root / 'canonical'} "
            f"by hand and move {journal.parent} out of {journal.parent.parent}"
        )
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
    source: SourceOwnership,
) -> None:
    source.validate()
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
    source.write_proof(transaction)


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
    if isinstance(error, ArtifactSourceUnavailableError):
        error.add_note(suffix.strip())
        return error
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


def _clear_failed_restore(state_root: Path, source: SourceOwnership, transaction: Path, error: Exception) -> Exception:
    """Quarantine a failed destination restore only after the public tree has been
    reinstalled. Public-restore failures retain their journal because replay is still
    required to restore that tree.
    """
    if _transaction_holds_conflict(transaction):
        return _transaction_context(error, state_root, transaction)
    source.validate_proof(transaction)
    publication._reset_source_stage(source, transaction.name, "restore", workspace_key=state_root.name)
    preserved = _quarantine_transaction(state_root, transaction)
    return _transaction_context(
        error,
        state_root,
        transaction,
        detail=f"; it could not restore every explicit destination and is preserved at {preserved}",
    )


def _recover_transactions(state_root: Path, source: SourceOwnership) -> None:
    """Replay every pending transaction, publications first, after its source-identity proof."""
    _recover_cleanup(state_root)
    transactions = state_root / "transactions"
    if not transactions.exists():
        return
    if transactions.is_symlink() or not transactions.is_dir():
        raise ArtifactVisibilityError("artifact transaction root is unsafe")
    pending: list[tuple[Path, str, _Transition | None, str]] = []
    for transaction in sorted(transactions.iterdir(), key=lambda path: path.name):
        if transaction.is_symlink() or not transaction.is_dir():
            raise ArtifactVisibilityError("artifact transaction residue is unsafe")
        journal = transaction / "journal.json"
        recorded = _load_transition(journal, transaction.name) if journal.is_file() else None
        owner = _load_transaction_owner(transaction, state_root)
        state, session_id = recorded if recorded is not None else (None, cast(str, owner["session_id"]))
        pending.append((transaction, cast(str, owner["kind"]), state, session_id))
    sessions = {session_id: transaction for transaction, kind, _state, session_id in pending if kind == "detach"}
    for transaction, kind, state, session_id in sorted(pending, key=lambda item: (item[1] != "publish", item[0].name)):
        if not transaction.exists():
            continue  # A rolled-forward publication already settled its session.
        try:
            if kind == "publish":
                _recover_publication(state_root, source, transaction, state, session_id, sessions.get(session_id))
            else:
                _recover_session(state_root, source, transaction, completed=False)
        except Exception as exc:
            contextual = _transaction_context(exc, state_root, transaction)
            if contextual is exc:
                raise
            raise contextual from exc


def _restore_from_records(
    state_root: Path, source: SourceOwnership, transaction: Path, records: tuple[_DestinationRecord, ...],
) -> None:
    """Restore explicit destinations to their baselines, quarantining a transaction that cannot."""
    try:
        source.validate_proof(transaction)
        publication._restore_destinations(source, transaction, records, workspace_key=state_root.name)
    except Exception as exc:
        raise _clear_failed_restore(state_root, source, transaction, exc) from exc


def _recover_session(state_root: Path, source: SourceOwnership, transaction: Path, *, completed: bool) -> None:
    """Return a session's parked tree and explicit destinations, or discard what its publication replaced."""
    if _transaction_holds_conflict(transaction):
        raise ArtifactVisibilityError("artifact recovery retained a closed conflict")
    source.validate_proof(transaction)
    if completed:
        publication._discard_parked(source, transaction, workspace_key=state_root.name)
    else:
        records = ledger._load_destination_records(transaction, include_published=False)
        external._reconcile_external_entries(transaction, records)
        publication._unpark(source, transaction, workspace_key=state_root.name)
        _restore_from_records(state_root, source, transaction, records)
    source.validate()
    _retire_transaction(state_root, transaction, terminal_state=_TerminalState.DETACHED_RECONCILED)


def _recover_publication(
    state_root: Path, source: SourceOwnership, transaction: Path, state: _Transition | None, session_id: str,
    session_transaction: Path | None,
) -> None:
    """Roll an unverified publication back, or an installed one forward; earlier evidence is never read."""
    if _transaction_holds_conflict(transaction):
        raise ArtifactVisibilityError("artifact recovery retained a closed conflict")
    source.validate_proof(transaction)
    publication_stage = publication._source_stage_path(source, transaction.name, "publish")
    terminal = _TerminalState.PUBLISH_RECONCILED
    if state is not None:
        records = ledger._load_destination_records(transaction, include_published=True)
        external._reconcile_external_entries(transaction, records)
        manifest = transaction / "publish-manifest.json"
        expected = filesystem._parse_manifest(manifest) if manifest.exists() or manifest.is_symlink() else None
        if state in (_Transition.PUBLIC_STAGED, _Transition.PUBLIC_CARRIED):
            if session_transaction is None:
                raise ArtifactVisibilityError("artifact publication lost its session transaction")
            if expected is not None:
                publication._roll_back_public(
                    source, transaction, session_transaction, expected=expected, session_id=session_id,
                    workspace_key=state_root.name,
                )
                # Durably mark the rollback done: a rerun may meet the unparked tree and no stage.
                manifest.unlink()
                _fsync_directory(transaction)
            publication._unpark(source, session_transaction, workspace_key=state_root.name)
            _restore_from_records(state_root, source, transaction, records)
        else:
            if expected is not None and publication._public_view(source.path, session_id) != expected:
                raise ArtifactVisibilityError("installed public artifact recovery copy is corrupt")
            for record in records:
                if filesystem.manifest_tree(Path(record.base), (record.relative,)) != record.published:
                    raise ArtifactVisibilityError("installed explicit artifact recovery copy is corrupt")
            if state is _Transition.PUBLIC_VERIFIED:
                terminal = _TerminalState.PUBLISH_VERIFIED
            # Settle the session before this transaction retires; it alone proves the session published.
            if session_transaction is not None and session_transaction.exists():
                _recover_session(state_root, source, session_transaction, completed=True)
    if publication_stage.exists() or publication_stage.is_symlink():
        publication._retire_source_stage(source, transaction.name, "publish", workspace_key=state_root.name)
    source.validate()
    _retire_transaction(state_root, transaction, terminal_state=terminal)
