"""Private benchmark storage: strict loaders, atomic writes, locking, and recovery. Reject
duplicate keys, unsafe tags, and malformed documents with WorkspaceCorrupt naming the
file. Files are 0600 and directories 0700.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from daydream.json_utils import _fsync_directory, _fsync_file, atomic_write_bytes


class WorkspaceError(Exception):
    """Base error for the private benchmark workspace subsystem."""


class WorkspaceCorrupt(WorkspaceError):
    """A workspace file violates a schema/checksum/path/orphan invariant."""


class LockContentionError(WorkspaceError):
    """Another process holds the workspace lock (explicit non-blocking probe)."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A :class:`yaml.SafeLoader` that rejects documents with duplicate keys."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise WorkspaceCorrupt(
                f"duplicate key {key!r} in YAML mapping at line {key_node.start_mark.line}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_yaml_strict(path: Path) -> dict[str, Any]:
    """Load a mapping or raise ``WorkspaceCorrupt`` for unreadable, invalid, or empty YAML."""
    try:
        data = yaml.load(Path(path).read_bytes(), Loader=_UniqueKeyLoader)
    except WorkspaceCorrupt:
        raise
    except yaml.YAMLError as exc:
        raise WorkspaceCorrupt(f"{path}: invalid YAML: {exc}") from exc
    except OSError as exc:
        raise WorkspaceCorrupt(f"{path}: unreadable: {exc}") from exc
    if data is None:
        raise WorkspaceCorrupt(f"{path}: empty YAML document")
    if not isinstance(data, dict):
        raise WorkspaceCorrupt(f"{path}: YAML root is not a mapping")
    return data


def load_json_strict(path: Path) -> dict[str, Any]:
    """Load a JSON file as a strict dict, rejecting parse errors / non-dict roots."""
    try:
        data = json.loads(Path(path).read_bytes())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise WorkspaceCorrupt(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkspaceCorrupt(f"{path}: JSON root is not an object")
    return data


def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
    """Atomically write ``content`` to ``path`` via the shared primitive."""
    ensure_private_dir(path.parent)
    atomic_write_bytes(path, content, mode=mode, fsync=True)


def atomic_write_json(path: Path, data: Any, *, mode: int = 0o600) -> None:
    """Atomically write ``data`` as JSON to ``path`` with a strict mode."""
    payload = json.dumps(data, indent=2).encode("utf-8")
    _atomic_write(path, payload, mode=mode)


def atomic_write_yaml(path: Path, data: Any, *, mode: int = 0o600) -> None:
    """Atomically write ``data`` as YAML to ``path`` with a strict mode."""
    payload = yaml.safe_dump(data, sort_keys=False).encode("utf-8")
    _atomic_write(path, payload, mode=mode)


def ensure_private_dir(path: Path, mode: int = 0o700) -> None:
    """Create ``path`` (and any missing parents) with private ``0700`` modes."""
    missing: list[Path] = []
    cursor: Path | None = path
    while cursor is not None and not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        os.chmod(created, mode)
    if path.exists():
        os.chmod(path, mode)


def sha256_file(path: Path) -> str:
    """Return the lowercase 64-hex sha256 digest of ``path``'s bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _HeldLock:
    """A workspace lock currently held by this process (fd + reuse depth)."""

    fd: int
    depth: int


class WorkspaceLock:
    """Exclusive flock on the retained .benchmark.lock file, reentrant per process/root.
    Nested acquisitions share a descriptor and depth counter. Hold the lock through
    read, validation, mutation, and commit to serialize curators. The lock file remains
    after release to avoid removal races.
    """

    _held: dict[Path, _HeldLock] = {}

    def __init__(self, root: Path, *, blocking: bool = True) -> None:
        self._root = Path(root)
        self._blocking = blocking

    def __enter__(self) -> "WorkspaceLock":
        held = WorkspaceLock._held.get(self._root)
        if held is not None:
            # Already holding the lock for this root in this process — reentrant
            # on the same open file description. Bump the depth and reuse the fd.
            held.depth += 1
            return self

        lock_path = self._root / ".benchmark.lock"
        self._root.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        flags = fcntl.LOCK_EX | (0 if self._blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError:
            os.close(fd)
            raise LockContentionError(
                f"workspace is locked by another process: {self._root}"
            ) from None
        WorkspaceLock._held[self._root] = _HeldLock(fd=fd, depth=1)
        return self

    def __exit__(self, *_exc: object) -> Literal[False]:
        held = WorkspaceLock._held.get(self._root)
        if held is None:
            return False
        held.depth -= 1
        if held.depth <= 0:
            fcntl.flock(held.fd, fcntl.LOCK_UN)
            os.close(held.fd)
            WorkspaceLock._held.pop(self._root, None)
        return False


# Same-filesystem journal: replacements are ordered, with benchmark.yaml last.
# Recovery discards prepared transactions, restores committing transactions in reverse,
# and verifies complete transactions. Fsync at each boundary preserves whole before/
# after states; recovered files remain 0600 and retained scaffold directories 0700.


@dataclass
class _TargetState:
    rel: str
    operation: Literal["replace", "retire"]
    stage_path: Path | None
    backup_path: Path | None
    original_existed: bool
    before_digest: str | None
    after_digest: str | None


class Transaction:
    """Same-filesystem journal with explicit stage/prepare/begin_commit/commit steps.
    Context exit does not commit or erase partial state; startup recovery requires that
    evidence to resolve an interrupted operation.
    """

    def __init__(self, root: Path, *, op_id: str, kind: str) -> None:
        self._root = Path(root)
        self._op_id = str(op_id)
        self._kind = str(kind)
        self._dir = self._root / "transactions" / self._op_id
        ensure_private_dir(self._dir)
        self._states: dict[str, _TargetState] = {}
        self._applied_count = 0
        self._state: str = "open"
        self._created_dirs: list[str] = []

    # -- journal helpers ----------------------------------------------------

    def _journal_path(self) -> Path:
        return self._dir / "journal.json"

    def _build_document(self) -> dict[str, Any]:
        targets = []
        for rel, st in self._states.items():
            targets.append(
                {
                    "rel": rel,
                    "operation": st.operation,
                    "stage": st.stage_path.name if st.stage_path else None,
                    "backup": st.backup_path.name if st.backup_path else None,
                    "original_existed": st.original_existed,
                    "before_digest": st.before_digest,
                    "after_digest": st.after_digest,
                }
            )
        return {
            "op_id": self._op_id,
            "kind": self._kind,
            "state": self._state,
            "replacement_order": self._replacement_order(),
            "applied_count": self._applied_count,
            "created_dirs": self._created_dirs,
            "targets": targets,
        }

    def _write_journal(self) -> None:
        doc = self._build_document()
        atomic_write_json(self._journal_path(), doc, mode=0o600)

    # pipeline

    def create_dir(self, target_rel: str | Path) -> None:
        """Journal ownership of a new 0700 directory. Recovery removes only empty owned
        directories after an interrupted transaction.
        """
        rel = _resolve_target(self._root, target_rel)
        ensure_private_dir(self._root / rel)
        if rel not in self._created_dirs:
            self._created_dirs.append(rel)

    def stage(self, target_rel: str | Path, content: bytes) -> None:
        """Stage ``content`` for an atomic replace of ``target_rel``."""
        rel = _resolve_target(self._root, target_rel)
        if rel in self._states:
            raise WorkspaceCorrupt(f"{self._root}: duplicate staged target {rel!r}")
        target = self._root / rel
        ensure_private_dir(target.parent)
        index = len(self._states)
        stage_path = self._dir / f"stage-{index:04d}.bin"
        _atomic_write(stage_path, content, mode=0o600)
        after_digest = sha256_file(stage_path)
        if target.exists():
            backup_path = self._dir / f"backup-{index:04d}.bin"
            shutil.copyfile(target, backup_path)
            _fsync_file(backup_path)
            original_existed = True
            before_digest = sha256_file(target)
        else:
            backup_path = None
            original_existed = False
            before_digest = None
        self._states[rel] = _TargetState(
            rel=rel,
            operation="replace",
            stage_path=stage_path,
            backup_path=backup_path,
            original_existed=original_existed,
            before_digest=before_digest,
            after_digest=after_digest,
        )

    def retire(self, target_rel: str | Path, *, expected_sha256: str) -> None:
        """Stage exact-digest file retirement with a backup. Interrupted commits restore
        it; complete commits verify absence.
        """
        rel = _resolve_target(self._root, target_rel)
        if rel in self._states:
            raise WorkspaceCorrupt(f"{self._root}: duplicate staged target {rel!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise WorkspaceCorrupt(f"{self._root}: retirement digest is not lowercase sha256")
        target = self._root / rel
        if not target.is_file():
            raise WorkspaceCorrupt(f"{self._root}: retirement target {rel!r} is missing")
        actual = sha256_file(target)
        if actual != expected_sha256:
            raise WorkspaceCorrupt(
                f"{self._root}: retirement target {rel!r} digest mismatch "
                f"(expected {expected_sha256}, got {actual})"
            )
        index = len(self._states)
        backup_path = self._dir / f"backup-{index:04d}.bin"
        shutil.copyfile(target, backup_path)
        _fsync_file(backup_path)
        self._states[rel] = _TargetState(
            rel=rel,
            operation="retire",
            stage_path=None,
            backup_path=backup_path,
            original_existed=True,
            before_digest=actual,
            after_digest=None,
        )

    def _replacement_order(self) -> list[str]:
        """Derive stable journal order from the target owner; publish its manifest last."""
        return sorted(self._states, key=lambda rel: rel == "benchmark.yaml")

    def prepare(self) -> None:
        """Persist the ``prepared`` journal (fsync'd) for startup recovery."""
        _fsync_directory(self._dir)
        self._state = "prepared"
        self._write_journal()
        _fsync_file(self._journal_path())

    def _begin_committing(self) -> None:
        """Transition the journal to ``committing`` with nothing applied."""
        self._state = "committing"
        self._applied_count = 0
        self._write_journal()
        _fsync_file(self._journal_path())

    def begin_commit(self) -> None:
        """Mark committing, then apply targets in order. Persist applied_count before each
        replacement so recovery includes a target even if the process dies between
        replacement and the next journal write.
        """
        self._begin_committing()
        self._apply_replacements()
        _fsync_directory(self._root)

    def _apply_replacements(self) -> None:
        """Replace in declared order, fsyncing each file and parent before applying the
        next target.
        """
        for rel in self._replacement_order():
            self._apply_replacement(rel)

    def _apply_replacement(self, rel: str) -> None:
        """Durably apply one already-journaled target in replacement order."""
        st = self._states[rel]
        rel = _resolve_target(self._root, rel)
        target = self._root / rel
        ensure_private_dir(target.parent)
        self._applied_count += 1
        self._write_journal()
        _fsync_file(self._journal_path())
        if st.operation == "retire":
            target.unlink()
        else:
            if st.stage_path is None:
                raise WorkspaceCorrupt(
                    f"{self._root}: replacement target {rel!r} has no staged file"
                )
            os.replace(st.stage_path, target)
            _fsync_file(target)
            os.chmod(target, 0o600)
        _fsync_directory(target.parent)

    def _complete_commit(self) -> None:
        """Publish and verify the durable complete state before cleanup."""
        self._state = "complete"
        self._write_journal()
        _fsync_file(self._journal_path())
        _verify_complete(self._root, self._dir, self._build_document(), retire=False)

    def commit(self) -> None:
        """Run the full pipeline; normal and recovered completion share verification."""
        self.prepare()
        self.begin_commit()
        self._complete_commit()
        shutil.rmtree(self._dir, ignore_errors=True)

    def __enter__(self) -> "Transaction":
        return self

    # The journal state machine is driven explicitly above; leaving the with
    # block after prepare()/begin_commit() must NOT clean up so
    # the persisted crash-state remains for recover_startup to heal.
    def __exit__(self, *_exc: object) -> Literal[False]:
        return False


def _resolve_target(root: Path, target: str | Path) -> str:
    """Resolve current containment; mutable symlink topology must never be cached.

    Absolute paths inside root are allowed. Parent symlinks cannot escape it.
    """
    try:
        root_r = Path(root).resolve()
    except (OSError, RuntimeError) as exc:
        raise WorkspaceCorrupt(f"{root}: cannot resolve the workspace root: {exc}") from exc
    p = Path(target)
    candidate = p if p.is_absolute() else root_r / p
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise WorkspaceCorrupt(f"{root_r}: cannot resolve target {str(target)!r}: {exc}") from exc
    try:
        rel = resolved.relative_to(root_r)
    except ValueError:
        raise WorkspaceCorrupt(
            f"{root_r}: target resolves outside the workspace root: {str(target)!r}"
        ) from None
    return rel.as_posix()


def resolve_authoring_path(root: Path, rel: str | Path) -> Path:
    """Resolve a contained authoring path, rejecting absolute inputs even when they point
    inside root.
    """
    if Path(rel).is_absolute():
        raise WorkspaceCorrupt(f"{root}: authoring path must be relative: {rel!r}")
    return root / _resolve_target(root, rel)


# startup recovery + orphan rules


def recover_startup(
    root: Path,
    *,
    indexed: set[str] | None = None,
    on_disk: set[Path] | None = None,
) -> None:
    """Validate every journal and cross-transaction claim before any recovery mutation.
    Prepared journals roll back; committing journals restore backups in reverse;
    complete journals verify after-state digests before clearing. With no journal,
    indexed/on-disk mismatches are corruption. Delete only positively identified
    pre-journal residue; unknown transaction entries fail closed.
    """
    root = Path(root)
    txn_root = root / "transactions"
    journal_files: list[Path] = []
    residue_dirs: list[Path] = []
    if txn_root.exists():
        for op_dir in txn_root.iterdir():
            if op_dir.is_symlink():
                # Never follow a symlink under transactions/ — it is
                # unidentifiable residue and fails closed.
                raise WorkspaceCorrupt(
                    f"{root}: unidentifiable symlink under transactions: {op_dir.name}"
                )
            if not op_dir.is_dir():
                # A plain file under transactions/ is not our residue -> fail closed.
                raise WorkspaceCorrupt(
                    f"{root}: unidentifiable file under transactions: {op_dir.name}"
                )
            jf = op_dir / "journal.json"
            if jf.exists():
                journal_files.append(jf)
            elif _is_transaction_residue(op_dir):
                residue_dirs.append(op_dir)
            else:
                # A dir with unknown contents is not positively-identified
                # residue — fail closed and leave it untouched.
                raise WorkspaceCorrupt(
                    f"{root}: unidentifiable residue under transactions: {op_dir.name}"
                )

    if journal_files:
        # Phase 1: validate every journal + collect claimed target rels,
        # rejecting any cross-transaction conflict before mutation.
        journaled: list[tuple[Path, dict[str, Any]]] = []
        claims: dict[str, str] = {}
        for jf in journal_files:
            try:
                doc = load_json_strict(jf)
            except WorkspaceCorrupt as exc:
                raise WorkspaceCorrupt(
                    f"{root}: unreadable transaction journal: {exc}"
                ) from exc
            op_dir = jf.parent
            _validate_journal(root, op_dir, doc)
            journaled.append((op_dir, doc))
            for t in _targets_from_doc(doc):
                rel = _resolve_target(root, t["rel"])
                if rel in claims:
                    raise WorkspaceCorrupt(
                        f"{root}: cross-transaction target conflict on {rel!r}"
                    )
                claims[rel] = op_dir.name
        # Phase 2: dispatch each validated journal (no conflict possible now).
        for op_dir, doc in journaled:
            state = doc.get("state")
            if state == "prepared":
                _rollback_prepared(root, op_dir, doc)
            elif state == "committing":
                _rollback_committing(root, op_dir, doc)
            else:  # complete
                _verify_complete(root, op_dir, doc)
        _empty_transactions(root)
        return

    # No journal present: remove only positively-identified pre-journal
    # residue, then apply the orphan rule if indexed/on_disk were supplied.
    for op_dir in residue_dirs:
        shutil.rmtree(op_dir)
    if indexed is None or on_disk is None:
        return

    _apply_orphan_rule(root, indexed, on_disk)


_TRANSACTION_RESIDUE_RE = re.compile(r"^(?:stage|backup)-\d{4}\.bin$")


def _is_transaction_residue(op_dir: Path) -> bool:
    """Recognize real directories containing only stage-NNNN.bin/backup-NNNN.bin regular
    files. Journals, symlinks, subdirectories, and foreign files prevent deletion.
    """
    if not op_dir.is_dir() or op_dir.is_symlink():
        return False
    try:
        entries = list(op_dir.iterdir())
    except OSError:
        return False
    if not entries:
        return False
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            return False
        if not _TRANSACTION_RESIDUE_RE.match(entry.name):
            return False
    return True


def _empty_transactions(root: Path) -> None:
    """Remove only positively identified residue directories; retain the root and propagate
    failures.
    """
    txn_root = root / "transactions"
    if not txn_root.exists():
        return
    for op_dir in txn_root.iterdir():
        if _is_transaction_residue(op_dir):
            shutil.rmtree(op_dir)


def _validate_journal(root: Path, op_dir: Path, doc: dict[str, Any]) -> None:
    """Validate the full journal structure and resolved containment without filesystem
    mutation.
    """
    state = doc.get("state")
    if state not in ("prepared", "committing", "complete"):
        raise WorkspaceCorrupt(f"{root}: invalid journal state {state!r}")
    if doc.get("op_id") != op_dir.name:
        raise WorkspaceCorrupt(
            f"{root}: journal op_id {doc.get('op_id')!r} does not match dir {op_dir.name!r}"
        )
    targets = doc.get("targets")
    if not isinstance(targets, list):
        raise WorkspaceCorrupt(f"{root}: journal targets is not a list")
    rels: set[str] = set()
    for t in targets:
        if not isinstance(t, dict) or not isinstance(t.get("rel"), str):
            raise WorkspaceCorrupt(f"{root}: malformed journal target entry")
        rel = _resolve_target(root, t["rel"])
        if rel in rels:
            raise WorkspaceCorrupt(f"{root}: duplicate target rel in journal: {rel!r}")
        rels.add(rel)
        operation = t.get("operation", "replace")
        if operation not in ("replace", "retire"):
            raise WorkspaceCorrupt(
                f"{root}: journal target {rel!r} has invalid operation {operation!r}"
            )
        for field in ("stage", "backup"):
            val = t.get(field)
            if val is None and (
                field == "backup" or (field == "stage" and operation == "retire")
            ):
                continue
            # Reject empty strings too: os.path.basename("") == "", so the bare
            # name check alone would accept "", letting ``op_dir / "" == op_dir``
            # make _rollback_committing rename the whole op dir as the target.
            if not isinstance(val, str) or not val or os.path.basename(val) != val:
                raise WorkspaceCorrupt(f"{root}: journal target {rel!r} {field} is not a bare filename")
        if operation == "retire":
            if t.get("stage") is not None or t.get("backup") is None:
                raise WorkspaceCorrupt(
                    f"{root}: retired journal target {rel!r} has invalid stage/backup"
                )
            before_digest = t.get("before_digest")
            if (
                t.get("original_existed") is not True
                or not isinstance(before_digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", before_digest)
                or t.get("after_digest") is not None
            ):
                raise WorkspaceCorrupt(
                    f"{root}: retired journal target {rel!r} has invalid digest state"
                )
            if state in ("prepared", "committing"):
                backup_path = op_dir / t["backup"]
                if not backup_path.is_file() or sha256_file(backup_path) != before_digest:
                    raise WorkspaceCorrupt(
                        f"{root}: retired journal target {rel!r} backup digest mismatch"
                    )
    order = doc.get("replacement_order")
    if not isinstance(order, list):
        raise WorkspaceCorrupt(f"{root}: journal replacement_order is not a list")
    for item in order:
        if not isinstance(item, str):
            raise WorkspaceCorrupt(f"{root}: journal replacement_order entry is not a string")
        if item not in rels:
            raise WorkspaceCorrupt(
                f"{root}: journal replacement_order names unknown target {item!r}"
            )
    applied = doc.get("applied_count")
    if not isinstance(applied, int) or not (0 <= applied <= len(order)):
        raise WorkspaceCorrupt(f"{root}: journal applied_count is out of bounds")
    created = doc.get("created_dirs")
    if created is not None:
        if not isinstance(created, list):
            raise WorkspaceCorrupt(f"{root}: journal created_dirs is not a list")
        for rel in created:
            if not isinstance(rel, str):
                raise WorkspaceCorrupt(f"{root}: malformed journal created_dirs entry")
            _resolve_target(root, rel)


def _targets_from_doc(doc: dict[str, Any]) -> list[dict[str, Any]]:
    targets = doc.get("targets", [])
    if not isinstance(targets, list):
        raise WorkspaceCorrupt("op-doc targets is not a list")
    return targets


def _rollback_prepared(root: Path, op_dir: Path, doc: dict[str, Any]) -> None:
    # Staged files only — no real target was replaced, so nothing to restore.
    if op_dir.exists():
        shutil.rmtree(op_dir, ignore_errors=True)
    _remove_created_dirs(root, doc)


def _rollback_committing(root: Path, op_dir: Path, doc: dict[str, Any]) -> None:
    order = doc.get("replacement_order") or []
    applied = int(doc.get("applied_count") or 0)
    # Key by the canonical rel the validator computed (replacement_order holds
    # canonical rels), so a crafted non-canonical target rel can't silently
    # evade recovery via a key mismatch (fail-closed all-or-nothing).
    targets = {_resolve_target(root, t["rel"]): t for t in _targets_from_doc(doc)}
    # Reverse order so benchmark.yaml (last) is restored first.
    prefix = order[: applied if applied else len(order)]
    for rel in reversed(prefix):
        t = targets.get(rel)
        if t is None:
            continue
        target = root / rel
        backup = t.get("backup")
        if backup is not None:
            os.replace(op_dir / backup, target)
            os.chmod(target, 0o600)
            _fsync_directory(target.parent)
        else:
            with suppress(OSError):
                target.unlink()
            _fsync_directory(target.parent)
    if op_dir.exists():
        shutil.rmtree(op_dir, ignore_errors=True)
    _remove_created_dirs(root, doc)


def _remove_created_dirs(root: Path, doc: dict[str, Any]) -> None:
    """Remove scaffold subdirs created by an interrupted transaction."""
    created = doc.get("created_dirs") or []
    for rel in sorted(created, key=len, reverse=True):
        rel = _resolve_target(root, rel)
        with suppress(OSError):
            (root / rel).rmdir()


def _verify_complete(root: Path, op_dir: Path, doc: dict[str, Any], *, retire: bool = True) -> None:
    for t in _targets_from_doc(doc):
        rel = _resolve_target(root, t["rel"])
        target = root / rel
        if t.get("operation", "replace") == "retire":
            if target.exists():
                raise WorkspaceCorrupt(
                    f"{root}: complete journal retired target {rel} still exists"
                )
            continue
        if not target.exists():
            raise WorkspaceCorrupt(f"{root}: complete journal {rel} missing on disk")
        actual = sha256_file(target)
        if actual != t["after_digest"]:
            raise WorkspaceCorrupt(
                f"{root}: complete journal {rel} digest mismatch (expected {t['after_digest']}, got {actual})"
            )
        # Recovery never widens a private target's mode, even if it drifted.
        os.chmod(target, 0o600)
    if retire and op_dir.exists():
        shutil.rmtree(op_dir, ignore_errors=True)


def _apply_orphan_rule(root: Path, indexed: set[str], on_disk: set[Path]) -> None:
    indexed_norm = {p.replace(os.sep, "/") for p in indexed}
    disk_rel: set[str] = set()
    for item in on_disk:
        p = Path(item)
        if p.is_absolute():
            rel = os.path.relpath(p, root).replace(os.sep, "/")
        else:
            rel = p.as_posix()
        disk_rel.add(rel)
    for rel in sorted(disk_rel):
        if rel not in indexed_norm:
            raise WorkspaceCorrupt(f"{root}: orphan on-disk file not in manifest index: {rel}")
    for rel in sorted(indexed_norm):
        if not (root / rel).exists():
            raise WorkspaceCorrupt(f"{root}: manifest-indexed file missing on disk: {rel}")
