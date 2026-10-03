"""Atomic publication and crash reconciliation for live external files."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import secrets
import stat
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from daydream.artifacts import filesystem, ledger, transfer
from daydream.artifacts.models import (
    _EXTERNAL_LIFECYCLE_VALUES,
    _EXTERNAL_PURPOSE_VALUES,
    _SCHEMA_VERSION,
    ArtifactVisibilityError,
    _DestinationRecord,
    _ExternalEntryLifecycle,
    _ExternalEntryPurpose,
)
from daydream.json_utils import _fsync_directory


@dataclass(frozen=True)
class _NameExchangeResult:
    result: int
    error_number: int | None


@dataclass(frozen=True)
class _ExternalEntryIssue:
    kind: Literal["directory", "fifo", "socket", "symlink", "special", "read_error"]


class _AtomicNameExchange:
    """Narrow platform name-exchange binding with an explicit result."""

    def __init__(self) -> None:
        self._library = ctypes.CDLL(None, use_errno=True)
        if sys.platform.startswith("linux"):
            symbol = "renameat2"
            self._flags = 0x2
        elif sys.platform == "darwin":
            symbol = "renameatx_np"
            self._flags = 0x12
        else:
            raise ArtifactVisibilityError("live external atomic exchange is unsupported")
        try:
            function = getattr(self._library, symbol)
        except AttributeError as exc:
            raise ArtifactVisibilityError("live external atomic exchange is unsupported") from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        self._function = function

    def call(self, parent_fd: int, staged_name: str, target_name: str) -> _NameExchangeResult:
        for name in (staged_name, target_name):
            _validate_exchange_name(name, message="atomic exchange name is invalid")
        ctypes.set_errno(0)
        result = self._function(parent_fd, os.fsencode(staged_name), parent_fd, os.fsencode(target_name), self._flags)
        if result == 0:
            return _NameExchangeResult(0, None)
        if result == -1:
            return _NameExchangeResult(-1, ctypes.get_errno())
        return _NameExchangeResult(result, None)


_external_entry_observer: Any | None = None
_name_exchange_factory: Any = _AtomicNameExchange
_external_link: Any = os.link


def _notify_external(state: str, purpose: _ExternalEntryPurpose, path: Path) -> None:
    observer = _external_entry_observer
    if observer is not None:
        observer(state, purpose.value, path)


_EXTERNAL_KEYS = frozenset(
    "record_id purpose parent name parent_dev parent_ino expected_kind expected_mode "
    "expected_sha256 lifecycle entry_dev entry_ino failure_reason destination_record_id".split()
)
_EXTERNAL_FAILURE_REASONS = (
    None,
    "unsupported",
    "conflict",
    "identity_changed",
    "operational_refusal",
    "ambiguous",
    "abi_result",
)


def _external_records(transaction: Path) -> list[dict[str, object]]:
    result = filesystem._load_ledger(
        transaction / "external-entries.json",
        items_key="entries",
        message="external entry ledger is malformed",
        identity=(("transaction_id", transaction.name),),
    )
    for index, entry in enumerate(result):
        if (
            not isinstance(entry, dict)
            or set(entry) != _EXTERNAL_KEYS
            or entry.get("record_id") != f"external-{index:04d}"
            or entry.get("purpose") not in _EXTERNAL_PURPOSE_VALUES
            or entry.get("lifecycle") not in _EXTERNAL_LIFECYCLE_VALUES
            or not isinstance(entry.get("parent"), str)
            or not isinstance(entry.get("name"), str)
            or type(entry.get("parent_dev")) is not int
            or type(entry.get("parent_ino")) is not int
            or entry.get("expected_kind") not in ("file", "directory")
            or type(entry.get("expected_mode")) is not int
            or not all(
                value is None or type(value) is int for value in (entry.get("entry_dev"), entry.get("entry_ino"))
            )
            or entry.get("failure_reason") not in _EXTERNAL_FAILURE_REASONS
            or not (entry.get("destination_record_id") is None or isinstance(entry.get("destination_record_id"), str))
        ):
            raise ArtifactVisibilityError("external entry ledger is malformed")
        parent = Path(cast(str, entry["parent"]))
        name = cast(str, entry["name"])
        if not parent.is_absolute() or filesystem._absolute_lexical(parent) != parent:
            raise ArtifactVisibilityError("external entry ledger is malformed")
        _validate_exchange_name(name)
        digest = entry["expected_sha256"]
        if digest is not None and not filesystem._is_sha256(digest):
            raise ArtifactVisibilityError("external entry ledger is malformed")
    return result


def _write_external_records(transaction: Path, records: list[dict[str, object]]) -> None:
    filesystem._atomic_json(
        transaction / "external-entries.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "transaction_id": transaction.name,
            "entries": records,
        },
    )


def _validate_exchange_name(name: str, *, message: str = "external entry name is invalid") -> None:
    if (
        not name
        or name in (".", "..")
        or any(character in name for character in ("/", "\\", "\0"))
        or Path(name).name != name
    ):
        raise ArtifactVisibilityError(message)


def _new_external_record(
    transaction: Path,
    *,
    purpose: _ExternalEntryPurpose,
    parent: Path,
    name: str,
    expected_kind: Literal["file", "directory"],
    expected_mode: int,
    expected_sha256: str | None,
    destination_record_id: str | None = None,
) -> int:
    _validate_exchange_name(name)
    parent_metadata = parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ArtifactVisibilityError("external entry parent is unsafe")
    records = _external_records(transaction)
    index = len(records)
    records.append(
        {
            "record_id": f"external-{index:04d}",
            "purpose": purpose.value,
            "parent": str(parent),
            "name": name,
            "parent_dev": parent_metadata.st_dev,
            "parent_ino": parent_metadata.st_ino,
            "expected_kind": expected_kind,
            "expected_mode": expected_mode,
            "expected_sha256": expected_sha256,
            "lifecycle": _ExternalEntryLifecycle.CREATION_INTENT.value,
            "entry_dev": None,
            "entry_ino": None,
            "failure_reason": None,
            "destination_record_id": destination_record_id,
        }
    )
    _write_external_records(transaction, records)
    _notify_external("CREATION_INTENT", purpose, parent / name)
    return index


def _update_external_record(
    transaction: Path,
    index: int,
    *,
    lifecycle: _ExternalEntryLifecycle,
    entry_dev: int | None = None,
    entry_ino: int | None = None,
    expected_mode: int | None = None,
    expected_sha256: str | None = None,
    identity: tuple[int, int, int, str] | None = None,
    failure_reason: str | None = None,
) -> dict[str, object]:
    """Persist one lifecycle transition and return the record as written.

    ``identity`` is the whole observed (dev, ino, mode, digest) tuple at once.
    """
    if identity is not None:
        entry_dev, entry_ino, expected_mode, expected_sha256 = identity
    records = _external_records(transaction)
    if index >= len(records):
        raise ArtifactVisibilityError("external entry record identity is malformed")
    current = dict(records[index])
    current["lifecycle"] = lifecycle.value
    updates = {"entry_dev": entry_dev, "entry_ino": entry_ino,
               "expected_mode": expected_mode, "expected_sha256": expected_sha256}
    current.update((key, value) for key, value in updates.items() if value is not None)
    current["failure_reason"] = failure_reason
    records[index] = current
    _write_external_records(transaction, records)
    observer = _external_entry_observer
    if observer is not None:
        observer(
            lifecycle.value.upper(),
            cast(str, current["purpose"]),
            Path(cast(str, current["parent"])) / cast(str, current["name"]),
        )
    return current


def _open_parent_fd(parent: Path) -> tuple[int, os.stat_result]:
    canonical = filesystem._declared_directory(parent, label="external output parent")
    fd = os.open(
        canonical,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    metadata = os.fstat(fd)
    return fd, metadata


def _external_identity(parent_fd: int, name: str) -> tuple[int, int, int, str] | _ExternalEntryIssue | None:
    _validate_exchange_name(name)
    try:
        fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        return _ExternalEntryIssue("symlink" if exc.errno == errno.ELOOP else "read_error")
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            kind: Literal["directory", "fifo", "socket", "symlink", "special", "read_error"]
            if stat.S_ISDIR(metadata.st_mode):
                kind = "directory"
            elif stat.S_ISFIFO(metadata.st_mode):
                kind = "fifo"
            elif stat.S_ISSOCK(metadata.st_mode):
                kind = "socket"
            else:
                kind = "special"
            return _ExternalEntryIssue(kind)
        digest = hashlib.sha256()
        try:
            while chunk := os.read(fd, 64 * 1024):
                digest.update(chunk)
        except OSError:
            return _ExternalEntryIssue("read_error")
        return metadata.st_dev, metadata.st_ino, stat.S_IMODE(metadata.st_mode), digest.hexdigest()
    finally:
        os.close(fd)


def _ledger_identity(record: dict[str, object]) -> tuple[object, object, object, object]:
    """Return the (dev, ino, mode, digest) identity one external ledger row attests."""
    return record["entry_dev"], record["entry_ino"], record["expected_mode"], record["expected_sha256"]


def _issue(*observations: object) -> _ExternalEntryIssue | None:
    """Return the first observation that is a nonregular/unreadable entry, if any."""
    return next((value for value in observations if isinstance(value, _ExternalEntryIssue)), None)


def _create_attested_external_file(
    transaction: Path,
    *,
    purpose: _ExternalEntryPurpose,
    parent_fd: int,
    parent: Path,
    name: str,
    content: bytes,
    destination_record_id: str | None = None,
    mode: int = 0o600,
) -> int:
    digest = hashlib.sha256(content).hexdigest()
    index = _new_external_record(
        transaction,
        purpose=purpose,
        parent=parent,
        name=name,
        expected_kind="file",
        expected_mode=mode,
        expected_sha256=digest,
        destination_record_id=destination_record_id,
    )
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode, dir_fd=parent_fd)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(fd)
        metadata = os.fstat(fd)
    finally:
        os.close(fd)
    os.fsync(parent_fd)
    _notify_external("ENTRY_CREATED", purpose, parent / name)
    identity = _external_identity(parent_fd, name)
    if identity != (metadata.st_dev, metadata.st_ino, mode, digest):
        _mark_external_conflict(transaction, index, reason="identity_changed", observed=_issue(identity))
        raise ArtifactVisibilityError("external output attestation failed")
    _update_external_record(
        transaction,
        index,
        lifecycle=_ExternalEntryLifecycle.ATTESTED,
        entry_dev=metadata.st_dev,
        entry_ino=metadata.st_ino,
    )
    return index


def _cleanup_external_name(transaction: Path, index: int, parent_fd: int) -> None:
    record = _external_records(transaction)[index]
    name = cast(str, record["name"])
    expected = _ledger_identity(record)
    _update_external_record(transaction, index, lifecycle=_ExternalEntryLifecycle.CLEANUP_PREPARED)
    observed = _external_identity(parent_fd, name)
    if observed != expected:
        _mark_external_conflict(transaction, index, reason="identity_changed", observed=_issue(observed))
        raise ArtifactVisibilityError("external output cleanup identity changed")
    os.unlink(name, dir_fd=parent_fd)
    os.fsync(parent_fd)
    _notify_external(
        "ENTRY_REMOVED",
        _ExternalEntryPurpose(cast(str, record["purpose"])),
        Path(cast(str, record["parent"])) / name,
    )
    _update_external_record(transaction, index, lifecycle=_ExternalEntryLifecycle.RETIRED)


def _external_failure_reason(result: _NameExchangeResult, *, probe: bool) -> str:
    if result.result not in (0, -1):
        return "abi_result"
    if result.error_number in (errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL) and probe:
        return "unsupported"
    if result.error_number in (errno.ENOENT, errno.EXDEV, errno.ELOOP):
        return "identity_changed"
    if result.error_number in (errno.EIO, errno.EINTR):
        return "ambiguous"
    return "operational_refusal"


def _expected_external_identity(record: _DestinationRecord) -> tuple[int | None, int | None, int, str | None]:
    """Return the (dev, ino, mode, digest) identity a live external entry must show."""
    digest = record.installed_sha256
    mode = 0o600
    if digest is None:
        baseline = filesystem._entry_at(record.baseline, record.relative, kind="file")
        if baseline is not None:
            digest = baseline.sha256
            mode = baseline.mode
    return record.expected_dev, record.expected_ino, mode, digest


def _mark_external_conflict(
    transaction: Path,
    index: int,
    *,
    reason: str,
    observed: _ExternalEntryIssue | None = None,
) -> None:
    record = _update_external_record(
        transaction,
        index,
        lifecycle=_ExternalEntryLifecycle.CONFLICT,
        failure_reason=reason,
    )
    transfer._append_conflict(
        transaction,
        record_id=f"external-{index:04d}",
        reason=reason,
        expected_sha256=cast("str | None", record["expected_sha256"]),
        observed_sha256=None,
        expected_kind=cast(str, record["expected_kind"]),
        observed_kind="unknown" if observed is None else observed.kind,
        stage_id=f"external-{index:04d}",
    )


def _ensure_external_parent(transaction: Path, parent: Path) -> list[int]:
    missing: list[Path] = []
    cursor = parent
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        cursor = cursor.parent
    filesystem._declared_directory(cursor, label="external output ancestor")
    created: list[int] = []
    for directory in reversed(missing):
        parent_path = directory.parent
        parent_fd, parent_metadata = _open_parent_fd(parent_path)
        try:
            index = _new_external_record(
                transaction,
                purpose=_ExternalEntryPurpose.MISSING_PARENT,
                parent=parent_path,
                name=directory.name,
                expected_kind="directory",
                expected_mode=0o700,
                expected_sha256=None,
            )
            os.mkdir(directory.name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            _notify_external("ENTRY_CREATED", _ExternalEntryPurpose.MISSING_PARENT, directory)
            fd = os.open(
                directory.name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            try:
                metadata = os.fstat(fd)
            finally:
                os.close(fd)
            if (parent_metadata.st_dev, parent_metadata.st_ino) != (
                os.fstat(parent_fd).st_dev,
                os.fstat(parent_fd).st_ino,
            ):
                raise ArtifactVisibilityError("external output parent identity changed")
            _update_external_record(
                transaction,
                index,
                lifecycle=_ExternalEntryLifecycle.ATTESTED,
                entry_dev=metadata.st_dev,
                entry_ino=metadata.st_ino,
            )
            created.append(index)
        finally:
            os.close(parent_fd)
    return created


def _probe_external_parent(transaction: Path, parent: Path, exchange: _AtomicNameExchange) -> tuple[int, int]:
    parent_fd, parent_metadata = _open_parent_fd(parent)
    token = secrets.token_hex(12)
    name_a = f".daydream-probe-{token}-a"
    name_b = f".daydream-probe-{token}-b"
    name_link = f".daydream-probe-{token}-link"
    a_index: int | None = None
    b_index: int | None = None
    link_index: int | None = None
    try:
        a_index = _create_attested_external_file(
            transaction,
            purpose=_ExternalEntryPurpose.PROBE_EXCHANGE_A,
            parent_fd=parent_fd,
            parent=parent,
            name=name_a,
            content=b"daydream exchange probe a",
        )
        b_index = _create_attested_external_file(
            transaction,
            purpose=_ExternalEntryPurpose.PROBE_EXCHANGE_B,
            parent_fd=parent_fd,
            parent=parent,
            name=name_b,
            content=b"daydream exchange probe b",
        )
        a_before = _external_identity(parent_fd, name_a)
        b_before = _external_identity(parent_fd, name_b)
        if not isinstance(a_before, tuple) or not isinstance(b_before, tuple):
            _mark_external_conflict(
                transaction, a_index, reason="identity_changed", observed=_issue(a_before, b_before)
            )
            raise ArtifactVisibilityError("live external atomic exchange probe entry changed")
        _update_external_record(transaction, a_index, lifecycle=_ExternalEntryLifecycle.OPERATION_PREPARED)
        result = exchange.call(parent_fd, name_a, name_b)
        _notify_external("PRIMARY_CALLED", _ExternalEntryPurpose.PROBE_EXCHANGE_A, parent / name_a)
        os.fsync(parent_fd)
        a_after = _external_identity(parent_fd, name_a)
        b_after = _external_identity(parent_fd, name_b)
        if result.result != 0 and a_after == a_before and b_after == b_before:
            _cleanup_external_name(transaction, a_index, parent_fd)
            _cleanup_external_name(transaction, b_index, parent_fd)
            _update_external_record(
                transaction,
                a_index,
                lifecycle=_ExternalEntryLifecycle.RETIRED,
                failure_reason=_external_failure_reason(result, probe=True),
            )
            raise ArtifactVisibilityError("live external atomic exchange probe was refused")
        if result.result != 0 or a_after != b_before or b_after != a_before:
            _mark_external_conflict(
                transaction,
                a_index,
                reason=_external_failure_reason(result, probe=True),
                observed=_issue(a_after, b_after),
            )
            raise ArtifactVisibilityError("live external atomic exchange probe failed")
        _update_external_record(transaction, a_index, lifecycle=_ExternalEntryLifecycle.REVERSAL_ATTEMPTED)
        reverse = exchange.call(parent_fd, name_a, name_b)
        _notify_external("REVERSAL_CALLED", _ExternalEntryPurpose.PROBE_EXCHANGE_A, parent / name_a)
        os.fsync(parent_fd)
        reverse_a = _external_identity(parent_fd, name_a)
        reverse_b = _external_identity(parent_fd, name_b)
        if reverse.result != 0 or reverse_a != a_before or reverse_b != b_before:
            _mark_external_conflict(transaction, a_index, reason="ambiguous", observed=_issue(reverse_a, reverse_b))
            raise ArtifactVisibilityError("live external atomic exchange reversal failed")
        _update_external_record(transaction, a_index, lifecycle=_ExternalEntryLifecycle.ATTESTED)
        link_index = _new_external_record(
            transaction,
            purpose=_ExternalEntryPurpose.PROBE_LINK_TARGET,
            parent=parent,
            name=name_link,
            expected_kind="file",
            expected_mode=0o600,
            expected_sha256=a_before[3],
        )
        try:
            _external_link(name_a, name_link, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
            _notify_external("LINK_CREATED", _ExternalEntryPurpose.PROBE_LINK_TARGET, parent / name_link)
        except OSError as exc:
            link_observation = _external_identity(parent_fd, name_link)
            if link_observation is None:
                _update_external_record(
                    transaction,
                    link_index,
                    lifecycle=_ExternalEntryLifecycle.RETIRED,
                    failure_reason="operational_refusal",
                )
            else:
                _mark_external_conflict(transaction, link_index, reason="conflict", observed=_issue(link_observation))
            raise ArtifactVisibilityError("live external no-clobber link probe failed") from exc
        os.fsync(parent_fd)
        linked = _external_identity(parent_fd, name_link)
        if linked != a_before:
            _mark_external_conflict(transaction, link_index, reason="identity_changed", observed=_issue(linked))
            raise ArtifactVisibilityError("live external no-clobber link attestation failed")
        assert isinstance(linked, tuple)
        _update_external_record(
            transaction,
            link_index,
            lifecycle=_ExternalEntryLifecycle.ATTESTED,
            entry_dev=linked[0],
            entry_ino=linked[1],
        )
        _cleanup_external_name(transaction, link_index, parent_fd)
        _cleanup_external_name(transaction, a_index, parent_fd)
        _cleanup_external_name(transaction, b_index, parent_fd)
        return parent_metadata.st_dev, parent_metadata.st_ino
    except BaseException:
        # Exact attested probe entries may be retired; ambiguous/conflicted entries stay.
        for index in (link_index, a_index, b_index):
            if index is None:
                continue
            record = _external_records(transaction)[index]
            if record["lifecycle"] in (
                _ExternalEntryLifecycle.ATTESTED.value,
                _ExternalEntryLifecycle.CLEANUP_PREPARED.value,
            ):
                with suppress(ArtifactVisibilityError):
                    _cleanup_external_name(transaction, index, parent_fd)
        raise
    finally:
        os.close(parent_fd)


def _cleanup_external_directories(transaction: Path, indexes: Sequence[int]) -> None:
    for index in reversed(indexes):
        record = _external_records(transaction)[index]
        if record["purpose"] != _ExternalEntryPurpose.MISSING_PARENT.value:
            raise ArtifactVisibilityError("external parent record identity is malformed")
        directory = Path(cast(str, record["parent"])) / cast(str, record["name"])
        if not directory.exists() and not directory.is_symlink():
            _update_external_record(transaction, index, lifecycle=_ExternalEntryLifecycle.RETIRED)
            continue
        metadata = directory.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (record["entry_dev"], record["entry_ino"])
        ):
            _mark_external_conflict(transaction, index, reason="identity_changed")
            raise ArtifactVisibilityError("external parent identity changed during cleanup")
        try:
            directory.rmdir()
        except OSError as exc:
            _mark_external_conflict(transaction, index, reason="conflict")
            raise ArtifactVisibilityError("external parent is not empty during cleanup") from exc
        _fsync_directory(directory.parent)
        _update_external_record(transaction, index, lifecycle=_ExternalEntryLifecycle.RETIRED)


def _reconcile_external_entries(transaction: Path, destinations: Sequence[_DestinationRecord]) -> None:
    destination_by_id = {record.record_id: record for record in destinations}
    exchange: _AtomicNameExchange | None = None
    for index, record in enumerate(_external_records(transaction)):
        lifecycle = _ExternalEntryLifecycle(cast(str, record["lifecycle"]))
        purpose = _ExternalEntryPurpose(cast(str, record["purpose"]))
        parent = Path(cast(str, record["parent"]))
        name = cast(str, record["name"])
        if lifecycle is _ExternalEntryLifecycle.RETIRED:
            continue
        if lifecycle is _ExternalEntryLifecycle.CONFLICT:
            raise ArtifactVisibilityError("external entry recovery retained a closed conflict")
        if lifecycle is _ExternalEntryLifecycle.CREATION_INTENT:
            intent_parent_fd, _ = _open_parent_fd(parent)
            try:
                intent_observation = _external_identity(intent_parent_fd, name)
            finally:
                os.close(intent_parent_fd)
            if intent_observation is not None:
                _mark_external_conflict(transaction, index, reason="conflict", observed=_issue(intent_observation))
                raise ArtifactVisibilityError("external unattested entry is retained as conflict")
            _update_external_record(transaction, index, lifecycle=_ExternalEntryLifecycle.RETIRED)
            continue
        if purpose is _ExternalEntryPurpose.MISSING_PARENT:
            if lifecycle is _ExternalEntryLifecycle.ATTESTED:
                continue
            raise ArtifactVisibilityError("external parent recovery state is invalid")
        if purpose is _ExternalEntryPurpose.PUBLICATION_LINK_TARGET and lifecycle in (
            _ExternalEntryLifecycle.ATTESTED,
            _ExternalEntryLifecycle.INSTALLED,
        ):
            # Destination reconciliation owns the installed target name.
            continue
        parent_fd, parent_metadata = _open_parent_fd(parent)
        try:
            if (parent_metadata.st_dev, parent_metadata.st_ino) != (record["parent_dev"], record["parent_ino"]):
                _mark_external_conflict(transaction, index, reason="identity_changed")
                raise ArtifactVisibilityError("external entry parent changed during recovery")
            expected_stage = _ledger_identity(record)
            observed_stage = _external_identity(parent_fd, name)
            if lifecycle in (_ExternalEntryLifecycle.ATTESTED, _ExternalEntryLifecycle.CLEANUP_PREPARED):
                if observed_stage is None:
                    _update_external_record(transaction, index, lifecycle=_ExternalEntryLifecycle.RETIRED)
                elif observed_stage == expected_stage:
                    _cleanup_external_name(transaction, index, parent_fd)
                else:
                    _mark_external_conflict(
                        transaction,
                        index,
                        reason="identity_changed",
                        observed=_issue(observed_stage),
                    )
                    raise ArtifactVisibilityError("external attested entry changed during recovery")
                continue
            if lifecycle is _ExternalEntryLifecycle.REVERSAL_ATTEMPTED:
                _mark_external_conflict(transaction, index, reason="ambiguous")
                raise ArtifactVisibilityError("external reversal boundary retained both entries")
            if lifecycle is not _ExternalEntryLifecycle.OPERATION_PREPARED:
                raise ArtifactVisibilityError("external entry recovery state is invalid")
            destination_id = record["destination_record_id"]
            destination = destination_by_id.get(cast(str, destination_id))
            if purpose is _ExternalEntryPurpose.PROBE_EXCHANGE_A:
                probe_b = next(
                    (
                        candidate
                        for candidate in _external_records(transaction)
                        if candidate["purpose"] == _ExternalEntryPurpose.PROBE_EXCHANGE_B.value
                        and candidate["parent"] == str(parent)
                    ),
                    None,
                )
                if probe_b is None:
                    _mark_external_conflict(transaction, index, reason="ambiguous")
                    raise ArtifactVisibilityError("external probe pair is incomplete")
                target_name = cast(str, probe_b["name"])
                expected_target = _ledger_identity(probe_b)
            elif destination is None:
                _mark_external_conflict(transaction, index, reason="ambiguous")
                raise ArtifactVisibilityError("external prepared entry has no destination identity")
            else:
                target_name = Path(destination.requested).name
                expected_target = _expected_external_identity(destination)
            observed_target = _external_identity(parent_fd, target_name)
            if observed_stage == expected_stage and observed_target == expected_target:
                _cleanup_external_name(transaction, index, parent_fd)
                continue
            if observed_target == expected_stage and observed_stage == expected_target:
                assert isinstance(observed_stage, tuple)
                _update_external_record(
                    transaction,
                    index,
                    lifecycle=_ExternalEntryLifecycle.REVERSAL_ATTEMPTED,
                    identity=observed_stage,
                    failure_reason="ambiguous",
                )
                if exchange is None:
                    exchange = _name_exchange_factory()
                result = exchange.call(parent_fd, name, target_name)
                os.fsync(parent_fd)
                target_after = _external_identity(parent_fd, target_name)
                stage_after = _external_identity(parent_fd, name)
                if result.result == 0 and target_after == expected_target and stage_after == expected_stage:
                    assert isinstance(stage_after, tuple)
                    _update_external_record(
                        transaction, index, lifecycle=_ExternalEntryLifecycle.ATTESTED, identity=stage_after
                    )
                    _cleanup_external_name(transaction, index, parent_fd)
                    continue
                reverse_issue = _issue(target_after, stage_after)
                _mark_external_conflict(
                    transaction,
                    index,
                    reason="identity_changed" if reverse_issue is not None else "conflict",
                    observed=reverse_issue,
                )
                raise ArtifactVisibilityError("external prepared exchange was recovered as conflict")
            observed_issue = _issue(observed_stage, observed_target)
            _mark_external_conflict(
                transaction,
                index,
                reason="identity_changed" if observed_issue is not None else "conflict",
                observed=observed_issue,
            )
            raise ArtifactVisibilityError("external prepared entry arrangement is ambiguous")
        finally:
            os.close(parent_fd)


def _finish_live_install(
    transaction: Path,
    record: _DestinationRecord,
    installed: tuple[int, int, int, str],
    digest: str,
    *,
    stage_index: int,
    parent_fd: int,
) -> _DestinationRecord:
    """Persist and retire the stage after one live-external install succeeded."""
    updated = replace(
        record,
        expected_dev=installed[0],
        expected_ino=installed[1],
        prepared_sha256=digest,
        installed_sha256=digest,
    )
    ledger._persist_live_destination_record(transaction, updated)
    _cleanup_external_name(transaction, stage_index, parent_fd)
    return updated


def _publish_live_external(
    transaction: Path,
    record: _DestinationRecord,
    content: bytes,
    *,
    exchange: _AtomicNameExchange,
    capability: tuple[int, int],
    content_mode: int = 0o600,
) -> _DestinationRecord:
    requested = Path(record.requested)
    parent = requested.parent
    stage_name = f".daydream-output-{secrets.token_hex(16)}"
    parent_fd, parent_metadata = _open_parent_fd(parent)
    try:
        if (parent_metadata.st_dev, parent_metadata.st_ino) != capability:
            raise ArtifactVisibilityError("live external output parent identity changed")
        stage_index = _create_attested_external_file(
            transaction,
            purpose=_ExternalEntryPurpose.PUBLICATION_STAGE,
            parent_fd=parent_fd,
            parent=parent,
            name=stage_name,
            content=content,
            destination_record_id=record.record_id,
            mode=content_mode,
        )
        stage_identity = _external_identity(parent_fd, stage_name)
        if not isinstance(stage_identity, tuple):
            _mark_external_conflict(
                transaction, stage_index, reason="identity_changed", observed=_issue(stage_identity)
            )
            raise ArtifactVisibilityError("live external publication stage changed")
        digest = stage_identity[3]
        if record.installed_sha256 is None and record.baseline_state != "file":
            target_index = _new_external_record(
                transaction,
                purpose=_ExternalEntryPurpose.PUBLICATION_LINK_TARGET,
                parent=parent,
                name=requested.name,
                expected_kind="file",
                expected_mode=stage_identity[2],
                expected_sha256=digest,
                destination_record_id=record.record_id,
            )
            try:
                _external_link(
                    stage_name,
                    requested.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                _notify_external("LINK_CREATED", _ExternalEntryPurpose.PUBLICATION_LINK_TARGET, requested)
            except OSError as exc:
                _mark_external_conflict(transaction, target_index, reason="conflict")
                _mark_external_conflict(transaction, stage_index, reason="conflict")
                raise ArtifactVisibilityError(
                    "live external absent destination changed before no-clobber install"
                ) from exc
            os.fsync(parent_fd)
            target_identity = _external_identity(parent_fd, requested.name)
            if target_identity != stage_identity:
                observed_issue = _issue(target_identity)
                _mark_external_conflict(transaction, target_index, reason="identity_changed", observed=observed_issue)
                _mark_external_conflict(transaction, stage_index, reason="identity_changed", observed=observed_issue)
                raise ArtifactVisibilityError("live external link installation identity changed")
            assert isinstance(target_identity, tuple)
            _update_external_record(
                transaction,
                target_index,
                lifecycle=_ExternalEntryLifecycle.ATTESTED,
                entry_dev=target_identity[0],
                entry_ino=target_identity[1],
            )
            _update_external_record(transaction, target_index, lifecycle=_ExternalEntryLifecycle.INSTALLED)
            return _finish_live_install(
                transaction, record, target_identity, digest, stage_index=stage_index, parent_fd=parent_fd
            )

        expected_identity = _expected_external_identity(record)
        _update_external_record(transaction, stage_index, lifecycle=_ExternalEntryLifecycle.OPERATION_PREPARED)
        result = exchange.call(parent_fd, stage_name, requested.name)
        _notify_external("PRIMARY_CALLED", _ExternalEntryPurpose.PUBLICATION_STAGE, parent / stage_name)
        os.fsync(parent_fd)
        target_after = _external_identity(parent_fd, requested.name)
        stage_after = _external_identity(parent_fd, stage_name)
        exact_pre = target_after == expected_identity and stage_after == stage_identity
        exact_swapped = target_after == stage_identity and stage_after == expected_identity
        if result.result == 0 and exact_swapped:
            assert isinstance(target_after, tuple) and isinstance(stage_after, tuple)
            _update_external_record(
                transaction, stage_index, lifecycle=_ExternalEntryLifecycle.INSTALLED, identity=stage_after
            )
            return _finish_live_install(
                transaction, record, target_after, digest, stage_index=stage_index, parent_fd=parent_fd
            )
        if result.result != 0 and exact_pre:
            _cleanup_external_name(transaction, stage_index, parent_fd)
            _update_external_record(
                transaction,
                stage_index,
                lifecycle=_ExternalEntryLifecycle.RETIRED,
                failure_reason=_external_failure_reason(result, probe=False),
            )
            raise ArtifactVisibilityError("live external atomic exchange was refused")
        if target_after == stage_identity and stage_after is not None:
            if isinstance(stage_after, _ExternalEntryIssue):
                _mark_external_conflict(transaction, stage_index, reason="identity_changed", observed=stage_after)
                raise ArtifactVisibilityError("live external atomic exchange displaced a nonregular entry")
            _update_external_record(
                transaction,
                stage_index,
                lifecycle=_ExternalEntryLifecycle.REVERSAL_ATTEMPTED,
                identity=stage_after,
                failure_reason=("conflict" if result.result == 0 else _external_failure_reason(result, probe=False)),
            )
            reverse = exchange.call(parent_fd, stage_name, requested.name)
            _notify_external("REVERSAL_CALLED", _ExternalEntryPurpose.PUBLICATION_STAGE, parent / stage_name)
            os.fsync(parent_fd)
            target_reversed = _external_identity(parent_fd, requested.name)
            stage_reversed = _external_identity(parent_fd, stage_name)
            if reverse.result == 0 and target_reversed == stage_after and stage_reversed == stage_identity:
                _mark_external_conflict(transaction, stage_index, reason="ambiguous")
            else:
                reverse_issue = _issue(target_reversed, stage_reversed)
                _mark_external_conflict(
                    transaction,
                    stage_index,
                    reason="identity_changed" if reverse_issue is not None else "conflict",
                    observed=reverse_issue,
                )
            raise ArtifactVisibilityError("live external atomic exchange failed after mutation")
        observed_issue = _issue(target_after, stage_after)
        _mark_external_conflict(
            transaction,
            stage_index,
            reason=(
                "identity_changed" if observed_issue is not None else _external_failure_reason(result, probe=False)
            ),
            observed=observed_issue,
        )
        raise ArtifactVisibilityError("live external atomic exchange result is ambiguous")
    finally:
        os.close(parent_fd)


def _recorded_external_capability(transaction: Path, parent: Path) -> tuple[int, int]:
    for record in _external_records(transaction):
        if (
            record["purpose"] == _ExternalEntryPurpose.PROBE_EXCHANGE_A.value
            and Path(cast(str, record["parent"])) == parent
            and record["entry_dev"] is not None
            and record["entry_ino"] is not None
        ):
            return cast(int, record["parent_dev"]), cast(int, record["parent_ino"])
    raise ArtifactVisibilityError("live external output has no durable capability proof")


def _inherit_external_capability_proofs(origin: Path, destination: Path) -> None:
    """Carry completed parent capability probes into the publication rollback journal.

    Only retired successful probes are copied: active names and their recovery
    ownership stay with the originating detach transaction. The parent identity
    is checked again by live-external publication when a baseline is restored.
    """
    proofs = [record for record in _external_records(origin)
              if record["purpose"] == _ExternalEntryPurpose.PROBE_EXCHANGE_A.value
              and record["lifecycle"] == _ExternalEntryLifecycle.RETIRED.value
              and record["failure_reason"] is None
              and record["entry_dev"] is not None and record["entry_ino"] is not None]
    if proofs:
        _write_external_records(destination, [
            {**record, "record_id": f"external-{index:04d}"} for index, record in enumerate(proofs)
        ])


def _external_target_identity_from_ledger(transaction: Path, destination: _DestinationRecord) -> tuple[int, int]:
    for record in reversed(_external_records(transaction)):
        if (
            record["destination_record_id"] == destination.record_id
            and record["purpose"] == _ExternalEntryPurpose.PUBLICATION_LINK_TARGET.value
            and record["lifecycle"] in (_ExternalEntryLifecycle.ATTESTED.value, _ExternalEntryLifecycle.INSTALLED.value)
            and type(record["entry_dev"]) is int
            and type(record["entry_ino"]) is int
        ):
            return record["entry_dev"], record["entry_ino"]
    raise ArtifactVisibilityError("external target identity is not durably attested")
