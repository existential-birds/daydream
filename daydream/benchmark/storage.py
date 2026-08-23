"""Strict, mode-safe filesystem primitives for the private benchmark workspace.

This module owns the on-disk safety layer for ``daydream benchmark``: strict
YAML/JSON loaders that reject duplicate keys and unsafe tags, mode-``0600``
atomic writes, mode-``0700`` private directory creation, sha256 checksums,
and (in later tasks) the workspace lock and transaction journal.

Every parse failure raises :class:`WorkspaceCorrupt` naming the offending file
— a corrupt file is an error, never silently defaulted.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml


class WorkspaceError(Exception):
    """Base error for the private benchmark workspace subsystem."""


class WorkspaceCorrupt(WorkspaceError):
    """A workspace file violates a schema/checksum/path/orphan invariant."""


class LockContentionError(WorkspaceError):
    """Another process holds the workspace lock (explicit non-blocking probe)."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A :class:`yaml.SafeLoader` that rejects documents with duplicate keys.

    Duplicate mapping keys in a manifest or case file are almost always a
    mistake (or a smuggling attempt) — they are treated as corruption.
    """


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
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
    """Load a YAML file as a dict, rejecting duplicates, non-dict roots, and unsafe tags.

    Raises:
        WorkspaceCorrupt: if the file is not valid YAML, is not a mapping, is
            empty, or contains duplicate keys / unsafe constructor tags.
    """
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
    """Atomically write ``content`` to ``path`` via same-dir temp + ``os.replace``.

    Mirrors :func:`daydream.json_utils.atomic_write_json` (temp + replace +
    fsync) but enforces a strict file mode on the final path and creates parent
    directories as private ``0700`` chains.
    """
    ensure_private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        os.chmod(path, mode)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


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
    """An ``fcntl``-backed exclusive lock scoped to a workspace root directory.

    Holds ``LOCK_EX`` (blocking) or ``LOCK_EX | LOCK_NB`` on a sibling
    ``<root>/.benchmark.lock`` file. The lock is process-reentrant per root:
    nested acquisitions within the same process share the same open fd and
    only bump a depth counter, so a command that holds the lock across its own
    journal writes cannot deadlock against itself. The curation service
    (``daydream/benchmark/curation.py``) relies on this blocking,
    process-reentrant semantics for every mutation: each locked mutation runs
    its whole read -> validate -> mutate -> commit sequence under one
    acquisition, so concurrent curators serialize instead of losing updates.

    The ``.benchmark.lock`` file itself is left on disk after release (removal
    races are unsafe), and mutual-exclusion among separate OS processes is
    provided by the kernel ``fcntl.flock`` byte-range lock.
    """

    _held: dict[Path, _HeldLock] = {}

    def __init__(self, root: Path, *, blocking: bool = True) -> None:
        self._root = Path(root)
        self._blocking = blocking
        self._acquired = False

    def __enter__(self) -> "WorkspaceLock":
        held = WorkspaceLock._held.get(self._root)
        if held is not None:
            # Already holding the lock for this root in this process — reentrant
            # on the same open file description. Bump the depth and reuse the fd.
            held.depth += 1
            self._acquired = False
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
        self._acquired = True
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


# ---------------------------------------------------------------------------
# Transaction journal (prepared | committing | complete) + startup recovery
# ---------------------------------------------------------------------------
#
# A transaction persists a same-filesystem journal under
# ``<root>/transactions/<op_id>/journal.json`` describing the exact ordered
# replacement of a set of workspace files. ``benchmark.yaml`` is always
# replaced last. On startup, ``recover_startup`` rolls a ``prepared`` journal
# forward-away (targets were never touched), rolls a ``committing`` journal
# back in reverse from backups/absent-markers, and verifies + cleans a
# ``complete`` journal. Because the journal lives on the same filesystem as
# the targets and every phase is fsynced before the next begins, a crash at
# any boundary restores either the whole before-state or the whole after-state
# — never a checksum-drifted partial. Recovery is mode-safe: verified targets
# keep ``0600``, and scaffold dirs kept by ``_remove_created_dirs`` stay ``0700``.


@dataclass
class _TargetState:
    rel: str
    stage_path: Path
    backup_path: Path | None
    original_existed: bool
    before_digest: str | None
    after_digest: str
    applied: bool = False


class Transaction:
    """A same-filesystem multi-file mutation journal with startup recovery.

    The context manager *stores* a transaction; it does not auto-*commit*.
    Callers drive ``stage``/``prepare``/``begin_commit``/``commit`` explicitly
    so a simulated crash (``inject_crash``) leaves the journal in the exact
    mid-state ``recover_startup`` is meant to heal. ``__exit__`` is therefore a
    no-op — partial state is deliberately left for recovery to adjudicate.
    """

    def __init__(self, root: Path, *, op_id: str, kind: str) -> None:
        self._root = Path(root)
        self._op_id = str(op_id)
        self._kind = str(kind)
        self._dir = self._root / "transactions" / self._op_id
        ensure_private_dir(self._dir)
        self._states: dict[str, _TargetState] = {}
        self._order: list[str] = []
        self._replacement_order: list[str] = []
        self._applied_count = 0
        self._state: str = "open"
        self._created_dirs: list[str] = []
        self._journal_document: dict[str, Any] | None = None

    # -- journal helpers ----------------------------------------------------

    def _journal_path(self) -> Path:
        return self._dir / "journal.json"

    def _build_document(self) -> dict[str, Any]:
        targets = []
        for rel in self._order:
            st = self._states[rel]
            targets.append(
                {
                    "rel": rel,
                    "stage": st.stage_path.name,
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
            "replacement_order": self._replacement_order,
            "applied_count": self._applied_count,
            "created_dirs": self._created_dirs,
            "targets": targets,
        }

    def _write_journal(self) -> None:
        doc = self._build_document()
        self._journal_document = doc
        atomic_write_json(self._journal_path(), doc, mode=0o600)

    # -----------------------------------------------------------------------
    # pipeline
    # -----------------------------------------------------------------------

    def create_dir(self, target_rel: str | Path) -> None:
        """Create a ``0700`` directory that this journal atomically owns.

        Directory creation is recorded in the journal so an interrupted
        transaction (``prepared`` / ``committing``) is rolled back by
        ``recover_startup`` (only empty directories are removed). This lets
        ``init_workspace`` build the private scaffold subdirs through the same
        crash-consistent journal instead of leaving them outside it.
        """
        rel = _resolve_target(self._root, target_rel)
        ensure_private_dir(self._root / rel)
        if rel not in self._created_dirs:
            self._created_dirs.append(rel)

    def stage(self, target_rel: str | Path, content: bytes) -> None:
        """Stage ``content`` for an atomic replace of ``target_rel``.

        Writes a staged file + a backup of any prior target under
        ``transactions/<op_id>/`` and records before/after digests. The real
        target is not touched here.
        """
        rel = _resolve_target(self._root, target_rel)
        if rel in self._states:
            raise WorkspaceCorrupt(f"{self._root}: duplicate staged target {rel!r}")
        target = self._root / rel
        ensure_private_dir(target.parent)
        index = len(self._order)
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
            stage_path=stage_path,
            backup_path=backup_path,
            original_existed=original_existed,
            before_digest=before_digest,
            after_digest=after_digest,
        )
        self._order.append(rel)
        # benchmark.yaml is always last in the ordered replacement list.
        if rel == "benchmark.yaml":
            self._replacement_order = self._order.copy()
        else:
            self._replacement_order = [r for r in self._replacement_order if r != "benchmark.yaml"] + [rel]

    def prepare(self) -> None:
        """Persist the ``prepared`` journal (fsync'd) for startup recovery."""
        _fsync_dir(self._dir)
        self._state = "prepared"
        self._write_journal()
        _fsync_file(self._journal_path())

    def _begin_committing(self) -> None:
        """Transition the journal to ``committing`` with nothing applied.

        A transaction enters ``committing`` by rewriting the journal with
        ``state == committing`` and ``applied_count == 0`` before any target
        is replaced. Both ``begin_commit`` and the per-target crash-injection
        branch collapse to this single sequence, so the ``target-<n>``
        mid-loop crash state stays byte-identical to a real halt reached via
        ``begin_commit``.
        """
        self._state = "committing"
        self._applied_count = 0
        self._write_journal()
        _fsync_file(self._journal_path())

    def begin_commit(self) -> None:
        """Rewrite the journal ``committing``, then apply targets in ordered list.

        Each target's ``applied_count`` is recorded in the journal *before* the
        target is replaced, so a crash between the replace and the next journal
        write still lets recovery roll that target back from its backup (the
        journal already accounts for it). Journaling-after-apply would leave the
        last-replaced file unrecoverable — a checksum-drifted mixed state the
        journal's never-drift contract forbids.
        """
        self._begin_committing()
        self._apply_replacements()
        _fsync_dir(self._root)

    def _apply_replacements(self, *, stop_after: int | None = None) -> None:
        """Apply targets in ``replacement_order``, fsyncing each rename.

        Each target's rename is made durable (file + *parent directory* fsync)
        before the next target is applied, so a crash at any per-target
        boundary still restores the whole before- or after-state, never a
        checksum-drifted mix. ``stop_after`` (a test-only hook) applies only
        the first ``N`` targets and leaves the journal ``committing`` with
        ``applied_count == N``, byte-identical to a real mid-loop crash.
        """
        for i, rel in enumerate(self._replacement_order):
            if stop_after is not None and i >= stop_after:
                break
            st = self._states[rel]
            rel = _resolve_target(self._root, rel)
            target = self._root / rel
            ensure_private_dir(target.parent)
            st.applied = True
            self._applied_count += 1
            self._write_journal()
            _fsync_file(self._journal_path())
            os.replace(st.stage_path, target)
            _fsync_file(target)
            os.chmod(target, 0o600)
            _fsync_dir(target.parent)

    def commit(self) -> None:
        """Run the full pipeline to ``complete`` and remove the journal."""
        self.prepare()
        self.begin_commit()
        self._state = "complete"
        self._write_journal()
        _fsync_file(self._journal_path())
        for st in self._states.values():
            actual = sha256_file(self._root / st.rel)
            if actual != st.after_digest:
                raise WorkspaceCorrupt(
                    f"{self._root}: commit verify {st.rel} expected {st.after_digest} got {actual}"
                )
        self._cleanup()

    def inject_crash(self, boundary: str | None = None) -> None:
        """Simulate a crash at a named pipeline boundary (test-only).

        With ``boundary is None`` this simply halts at the transaction's
        current state — recovery must adjudicate whatever phase is already
        persisted. With a named ``boundary``
        (``staged|backup|journal|data|manifest``) it first advances to that
        boundary (so a caller may drive an arbitrary mid-state) and then
        halts. A ``target-<n>`` boundary applies exactly the first ``n``
        targets of ``replacement_order`` under ``committing`` and halts,
        leaving a crash state byte-identical to a real mid-loop halt.
        """
        if boundary is None or boundary in ("staged", "backup"):
            # Staging (and any backup file) is written but no journal is
            # persisted yet — recovery finds nothing and leaves the target
            # perfectly ``before``.
            return
        if boundary == "journal":
            # Persist the ``prepared`` journal; recovery rolls the staged set
            # back (no target was ever replaced).
            self.prepare()
            return
        if boundary == "data":
            # Begin the commit: targets are applied under ``committing`` and
            # recovery rolls them back from backups.
            self.prepare()
            self.begin_commit()
            return
        if boundary == "manifest":
            # Run the full commit to ``complete`` so recovery verifies the
            # whole after-state.
            self.prepare()
            self.begin_commit()
            self.force_state("complete")
            return
        if boundary is not None and boundary.startswith("target-"):
            # Per-target boundary ``target-<n>``: prepare, then apply exactly
            # the first ``n`` targets and halt before the parent-dir fsync of
            # the next rename / the final root fsync. Recovery rolls all of
            # them back, restoring the all-old state. ``target-0`` is exactly
            # the ``journal`` boundary (prepared, nothing applied).
            suffix = boundary[len("target-"):]
            try:
                n = int(suffix)
            except ValueError:
                raise ValueError(f"invalid target boundary {boundary!r}") from None
            if not (0 <= n <= len(self._replacement_order)):
                raise ValueError(f"target boundary {n!r} out of range")
            self.prepare()
            if n == 0:
                return
            self._begin_committing()
            self._apply_replacements(stop_after=n)
            return
        raise ValueError(f"unknown crash boundary {boundary!r}")

    def force_state(self, state: str) -> None:
        """Force the journal into ``state`` (test-only hook)."""
        if state not in ("prepared", "committing", "complete"):
            raise ValueError(f"invalid journal state {state!r}")
        self._state = state
        self._write_journal()
        _fsync_file(self._journal_path())

    def _cleanup(self) -> None:
        """Remove the journal + staging dir, leaving an empty ``transactions/`` root."""
        if self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)

    def __enter__(self) -> "Transaction":
        return self

    # The journal state machine is driven explicitly above; leaving the with
    # block after prepare()/begin_commit()/inject_crash() must NOT clean up so
    # the persisted crash-state remains for recover_startup to heal.
    def __exit__(self, *_exc: object) -> Literal[False]:
        return False


# ---------------------------------------------------------------------------
# fsync helpers
# ---------------------------------------------------------------------------


def _fsync_file(path: Path) -> None:
    with open(path, "rb", buffering=0) as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@lru_cache(maxsize=None)
def _resolve_cached(root_r: str, target: str) -> str:
    """Resolve ``target`` against the already-resolved absolute ``root_r``.

    Every non-partial resolution/containment failure fails closed with
    :class:`WorkspaceCorrupt`, mirroring the sibling strict loaders. The
    result is memoized on ``(root_r, target)`` so recovery consumers that
    re-resolve the same validated rels after :func:`_validate_journal` reuse
    the canonical rel instead of repeating ``Path.resolve()`` syscalls.
    """
    p = Path(target)
    candidate = p if p.is_absolute() else Path(root_r) / p
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise WorkspaceCorrupt(
            f"{root_r}: cannot resolve target {target!r}: {exc}"
        ) from exc
    try:
        rel = resolved.relative_to(Path(root_r))
    except ValueError:
        raise WorkspaceCorrupt(
            f"{root_r}: target resolves outside the workspace root: {target!r}"
        ) from None
    return rel.as_posix()


def _resolve_target(root: Path, target: str | Path) -> str:
    """Resolve ``target`` to a canonical POSIX rel, forcing it beneath ``root``.

    The containment invariant is enforced at the *resolved* path, not the
    literal string: a ``../x`` relative input, an absolute path, and a path
    whose parent is a symlink to an outside directory all collapse to
    "resolves outside root" and are rejected uniformly. Absolute paths that
    resolve inside the root are accepted (the canonical ``rel`` is returned).
    """
    try:
        root_r = Path(root).resolve()
    except (OSError, RuntimeError) as exc:
        raise WorkspaceCorrupt(f"{root}: cannot resolve the workspace root: {exc}") from exc
    return _resolve_cached(str(root_r), str(target))


# ---------------------------------------------------------------------------
# startup recovery + orphan rules
# ---------------------------------------------------------------------------


def recover_startup(
    root: Path,
    *,
    indexed: set[str] | None = None,
    on_disk: set[Path] | None = None,
) -> None:
    """Recover an interrupted journal before reading workspace state.

    Recovery is fail-closed and two-phase. Phase 1 loads and validates the
    *entire* journal set (via :func:`_validate_journal`) and builds a
    ``{rel: op_dir}`` claim map, rejecting any cross-transaction target
    conflict before any filesystem mutation. Phase 2 then dispatches each
    validated journal: a ``prepared`` journal rolls back (targets were never
    applied), a ``committing`` journal rolls back in reverse from backups/
    absent markers, and a ``complete`` journal is verified against after-state
    digests then cleared. With ``indexed``/``on_disk`` supplied and no journal
    present, the orphan rule applies: an on-disk import/case/bundle not in
    ``indexed``, or an indexed file missing from disk, is corruption. When no
    journal is present, recovery still runs: it removes only positively-
    identified pre-journal ``stage-*.bin``/``backup-*.bin`` residue and treats
    any unidentifiable entry under ``transactions/`` as corruption.
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
    """True iff ``op_dir`` is positively-identified pre-journal residue.

    A residue dir is a real directory (not a symlink) whose *only* contents
    are regular files matching the exact ``stage-NNNN.bin`` / ``backup-NNNN.bin``
    staging patterns. A dir containing ``journal.json``, a subdirectory, a
    foreign file, or any symlink is -- by definition -- not residue, so
    recovery never guesses and never deletes anything it cannot identify.
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
    """Clean leftover per-op dirs, keeping the empty ``transactions/`` root.

    Only dirs that pass :func:`_is_transaction_residue` (positive
    identification) are removed. Symlinks, foreign files, and unknown subdirs
    are never touched here — ``recover_startup`` already raised on them before
    this runs — and a failure to remove a positively-identified dir propagates
    rather than being swallowed.
    """
    txn_root = root / "transactions"
    if not txn_root.exists():
        return
    for op_dir in txn_root.iterdir():
        if _is_transaction_residue(op_dir):
            shutil.rmtree(op_dir)


def _validate_journal(root: Path, op_dir: Path, doc: dict[str, Any]) -> None:
    """Strictly validate a journal document, failing closed with no mutation.

    Every check here is a hard requirement: a journal that violates any rule
    is corruption, never something to silently skip or partially trust. The
    validator performs no filesystem mutation — it only proves the doc is
    structurally sound and that every path it names resolves within ``root``.
    Recovery readers then trust the validated ``rel`` strings.
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
        for field in ("stage", "backup"):
            val = t.get(field)
            if val is None and field == "backup":
                continue
            # Reject empty strings too: os.path.basename("") == "", so the bare
            # name check alone would accept "", letting ``op_dir / "" == op_dir``
            # make _rollback_committing rename the whole op dir as the target.
            if not isinstance(val, str) or not val or os.path.basename(val) != val:
                raise WorkspaceCorrupt(f"{root}: journal target {rel!r} {field} is not a bare filename")
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
    return doc.get("targets", [])


def _rollback_prepared(root: Path, op_dir: Path, doc: dict[str, Any]) -> None:
    # Staged files only — no real target was replaced, so nothing to restore.
    for t in _targets_from_doc(doc):
        stage = op_dir / t["stage"]
        with suppress(OSError):
            stage.unlink()
        backup = t.get("backup")
        if backup:
            with suppress(OSError):
                (op_dir / backup).unlink()
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
            _fsync_dir(target.parent)
        else:
            with suppress(OSError):
                target.unlink()
            _fsync_dir(target.parent)
    if op_dir.exists():
        shutil.rmtree(op_dir, ignore_errors=True)
    _remove_created_dirs(root, doc)



def _remove_created_dirs(root: Path, doc: dict[str, Any]) -> None:
    """Remove scaffold subdirs created by an interrupted transaction.

    Only empty directories are removed, deepest first — a subdir that already
    holds real user content is preserved for the caller to adjudicate.
    """
    created = doc.get("created_dirs") or []
    for rel in sorted(created, key=len, reverse=True):
        rel = _resolve_target(root, rel)
        with suppress(OSError):
            (root / rel).rmdir()


def _verify_complete(root: Path, op_dir: Path, doc: dict[str, Any]) -> None:
    for t in _targets_from_doc(doc):
        rel = _resolve_target(root, t["rel"])
        target = root / rel
        if not target.exists():
            raise WorkspaceCorrupt(f"{root}: complete journal {rel} missing on disk")
        actual = sha256_file(target)
        if actual != t["after_digest"]:
            raise WorkspaceCorrupt(
                f"{root}: complete journal {rel} digest mismatch (expected {t['after_digest']}, got {actual})"
            )
        # Recovery never widens a private target's mode, even if it drifted.
        os.chmod(target, 0o600)
    if op_dir.exists():
        shutil.rmtree(op_dir, ignore_errors=True)


def _apply_orphan_rule(root: Path, indexed: set[str], on_disk: set[Path]) -> None:
    indexed_norm = {p.replace(os.sep, "/").lstrip("/") for p in indexed}
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
