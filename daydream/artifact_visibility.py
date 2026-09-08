"""Run-scoped storage boundary for model-invisible Daydream artifacts.

The source checkout exposes compatibility paths only between runs.  While a
session is bound, generated state lives under an owner-validated host runtime
root and callers may address it only through the exact workspace identity.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import stat
import sys
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal, cast

import anyio

from daydream import git_ops

if TYPE_CHECKING:
    from daydream.trajectory import RunWriteSnapshot, TrajectoryDocumentSnapshot
    from daydream.workspace import WorkContext


_SCHEMA_VERSION = 1
_DAYDREAM = ".daydream"
_REVIEW_OUTPUT = ".review-output.md"
_LEGACY_ANCHORS = frozenset(("runs", "deep", "exploration", "partial-fixes", "diff.patch", "hunk-index.json"))
_OPERATIONAL_NAMES = frozenset(("worktrees", "audit"))


class ArtifactVisibilityError(RuntimeError):
    """A live artifact namespace could not be opened or used safely."""


class _Transition(str, Enum):
    """Journalled in-flight transaction state; ``PUBLISH_*`` members publish."""

    DETACH_STAGED = "DETACH_STAGED"
    DETACH_CANONICAL = "DETACH_CANONICAL"
    DETACH_REMOVING = "DETACH_REMOVING"
    DETACHED = "DETACHED"
    PUBLISH_STAGED = "PUBLISH_STAGED"
    PUBLISH_BACKED_UP = "PUBLISH_BACKED_UP"
    PUBLISH_INSTALLED = "PUBLISH_INSTALLED"
    PUBLISH_VERIFIED = "PUBLISH_VERIFIED"

    @property
    def is_publish(self) -> bool:
        return self.name.startswith("PUBLISH_")


class _TerminalState(str, Enum):
    """Journalled outcome a retired transaction's cleanup ticket records."""

    DETACHED_RECONCILED = "DETACHED_RECONCILED"
    PUBLISH_RECONCILED = "PUBLISH_RECONCILED"
    PUBLISH_VERIFIED = "PUBLISH_VERIFIED"


class OutputLabel(str, Enum):
    """Closed classes of output registered before model execution."""

    PUBLIC_DAYDREAM = "public_daydream"
    PUBLIC_REVIEW_OUTPUT = "public_review_output"
    EXPLICIT_TRAJECTORY = "explicit_trajectory"
    EXPLICIT_TRAJECTORY_PARTIAL = "explicit_trajectory_partial"
    FINDINGS_OUTPUT = "findings_output"
    DUMP_DIRECTORY = "dump_directory"


class DestinationDelivery(str, Enum):
    """Closed host delivery policy for an operator-requested output."""

    DEFERRED = "deferred"
    LIVE_EXTERNAL = "live_external"
    FINALIZATION_MERGE = "finalization_merge"


class ArtifactDisposition(str, Enum):
    """Host outcome applied after immutable evidence finalization."""

    COMPLETE = "complete"
    PARTIAL_EVIDENCE = "partial_evidence"
    ROLLBACK = "rollback"


class _ExternalEntryPurpose(str, Enum):
    PROBE_EXCHANGE_A = "probe_exchange_a"
    PROBE_EXCHANGE_B = "probe_exchange_b"
    PROBE_LINK_TARGET = "probe_link_target"
    PUBLICATION_STAGE = "publication_stage"
    PUBLICATION_LINK_TARGET = "publication_link_target"
    MISSING_PARENT = "missing_parent"


class _ExternalEntryLifecycle(str, Enum):
    CREATION_INTENT = "creation_intent"
    ATTESTED = "attested"
    OPERATION_PREPARED = "operation_prepared"
    INSTALLED = "installed"
    REVERSAL_ATTEMPTED = "reversal_attempted"
    CLEANUP_PREPARED = "cleanup_prepared"
    RETIRED = "retired"
    CONFLICT = "conflict"


_EXTERNAL_PURPOSE_VALUES = frozenset(value.value for value in _ExternalEntryPurpose)
_EXTERNAL_LIFECYCLE_VALUES = frozenset(value.value for value in _ExternalEntryLifecycle)


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


@dataclass(frozen=True)
class ArtifactManifestEntry:
    """One no-follow filesystem entry in an immutable artifact tree."""

    path: str
    kind: Literal["directory", "file"]
    size: int
    mode: int
    sha256: str | None


@dataclass(frozen=True)
class ArtifactLayout:
    """One run's private namespace; every derived path is a property."""

    repo: Path
    source: Path
    git_common_dir: Path
    source_git_dir: Path
    repo_git_dir: Path
    operational_workspaces_root: Path
    session_id: str
    state_root: Path

    @property
    def artifact_runtime_root(self) -> Path:
        return self.state_root.parent

    @property
    def workspace_key(self) -> str:
        return self.state_root.name

    @property
    def live_root(self) -> Path:
        return self.state_root / "runs" / self.session_id / "live"

    @property
    def daydream_dir(self) -> Path:
        return self.live_root / _DAYDREAM

    @property
    def review_output(self) -> Path:
        return self.live_root / _REVIEW_OUTPUT

    @property
    def public_daydream_dir(self) -> Path:
        return self.source / _DAYDREAM

    @property
    def public_review_output(self) -> Path:
        return self.source / _REVIEW_OUTPUT


@dataclass(frozen=True)
class ArtifactWorkspaceIdentity:
    """Canonical source ownership and private host namespace."""

    repo: Path
    source: Path
    git_common_dir: Path
    source_git_dir: Path
    repo_git_dir: Path
    operational_state_root: Path
    state_root: Path
    workspace_key: str


@dataclass(frozen=True)
class PrivateRootLocations:
    """Sibling private roots selected once by runner composition."""

    artifact_runtime: Path
    operational_workspaces: Path


@dataclass(frozen=True)
class PrivateWorkspaceOwner:
    """Validated source/Git ownership shared by artifacts and worktrees."""

    source: Path
    git_common_dir: Path
    workspace_key: str
    artifact_state_root: Path
    operational_state_root: Path


@dataclass(frozen=True)
class RoutedDestination:
    label: OutputLabel
    requested: Path
    write_path: Path | None
    frozen_path: Path | None
    delivery: DestinationDelivery


@dataclass(frozen=True)
class TrajectoryOutputRoute:
    run_dir: Path
    full: RoutedDestination
    partial: RoutedDestination


@dataclass(frozen=True)
class _DestinationRecord:
    record_id: str
    requested: str
    base: str
    relative: str
    label: OutputLabel
    delivery: DestinationDelivery
    expected_kind: Literal["file", "directory"]
    baseline_state: Literal["absent", "file", "directory"]
    baseline: tuple[ArtifactManifestEntry, ...]
    missing_parents: tuple[str, ...]
    expected_dev: int | None = None
    expected_ino: int | None = None
    prepared_sha256: str | None = None
    installed_sha256: str | None = None
    published: tuple[ArtifactManifestEntry, ...] = ()


@dataclass(frozen=True)
class ArtifactTreeSnapshot:
    """Frozen run tree consumed by later archive/evaluation/publication."""

    session_id: str
    workspace_key: str
    root: Path
    manifest: tuple[ArtifactManifestEntry, ...]
    destinations: tuple[RoutedDestination, ...]


@dataclass
class _RoutedRecord:
    """One registered route paired with the destination ledger row it owns."""

    route: RoutedDestination
    record: _DestinationRecord
    late: Path | None = None


@dataclass(frozen=True)
class ArtifactEvidenceProvenance:
    """Where one run's evidence lived, as paths the consumer must not rebuild."""

    workspace_key: str
    session_id: str
    public_source: Path
    live_root: Path

    @property
    def public_daydream_dir(self) -> Path:
        return self.public_source / _DAYDREAM

    @property
    def public_review_output(self) -> Path:
        return self.public_source / _REVIEW_OUTPUT


_PUBLIC_LABELS = (OutputLabel.PUBLIC_DAYDREAM, OutputLabel.PUBLIC_REVIEW_OUTPUT)
_TRAJECTORY_LABELS = (OutputLabel.EXPLICIT_TRAJECTORY, OutputLabel.EXPLICIT_TRAJECTORY_PARTIAL)
_SESSION: ContextVar[ArtifactSession | None] = ContextVar("daydream_artifact_session", default=None)


def _default_private_base() -> Path:
    """Return the default private base; tests patch only this provider."""
    return Path.home() / ".daydream"


def private_root_locations(*, base: Path | None = None) -> PrivateRootLocations:
    """Return disjoint sibling artifact and operational root declarations."""
    selected = _default_private_base() if base is None else base
    if not selected.is_absolute() or _absolute_lexical(selected) != selected:
        raise ArtifactVisibilityError("private storage base must be an absolute lexical path")
    return PrivateRootLocations(artifact_runtime=selected / "runtime", operational_workspaces=selected / "workspaces")


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _declared_directory_metadata(path: Path, *, label: str) -> tuple[Path, os.stat_result]:
    """Resolve a declared directory, returning it with its own no-follow metadata."""
    declared = _absolute_lexical(path)
    declared_metadata: os.stat_result | None = None
    for index, component in enumerate((declared, *declared.parents)):
        try:
            metadata = component.lstat()
        except OSError as exc:
            raise ArtifactVisibilityError(f"{label} is not an accessible directory") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactVisibilityError(f"{label} ancestry must not contain a symlink")
        if index == 0:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactVisibilityError(f"{label} must be a real directory, not a symlink")
            declared_metadata = metadata
    assert declared_metadata is not None
    try:
        return declared.resolve(strict=True), declared_metadata
    except OSError as exc:
        raise ArtifactVisibilityError(f"{label} is not an accessible directory") from exc


def _declared_directory(path: Path, *, label: str) -> Path:
    return _declared_directory_metadata(path, label=label)[0]


def _open_directory_descriptor(path: Path, *, label: str) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ArtifactVisibilityError(f"{label} is not an accessible directory") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ArtifactVisibilityError(f"{label} is not an accessible directory")
    return descriptor


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _workspace_key(source: Path, common_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"daydream-artifacts-v1\0")
    digest.update(os.fsencode(source))
    digest.update(b"\0")
    digest.update(os.fsencode(common_dir))
    return digest.hexdigest()


def _private_owner_payload(source: Path, common_dir: Path, workspace_key: str) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "workspace_key": workspace_key,
        "source": str(source),
        "git_common_dir": str(common_dir),
    }


def _validate_private_root_declaration(path: Path, *, label: str) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or _absolute_lexical(path) != path:
        raise ArtifactVisibilityError(f"{label} must be an absolute lexical path")
    for component in (path, *path.parents):
        if not component.exists() and not component.is_symlink():
            continue
        metadata = component.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactVisibilityError(f"{label} ancestry contains a symlink")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactVisibilityError(f"{label} ancestry is not a directory")


def validate_private_directory(path: Path, *, label: str, allow_absent: bool = False) -> None:
    """Refuse anything but a real, mode-0700 directory at ``path``.

    ``allow_absent`` accepts a path that does not exist yet, which discovery roots need.
    """
    if allow_absent and not path.exists() and not path.is_symlink():
        return
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArtifactVisibilityError(f"{label} is not an accessible directory") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactVisibilityError(f"{label} must be a real directory")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ArtifactVisibilityError(f"{label} must have mode 0700")


def _create_private_directory(path: Path) -> None:
    """Create ``path`` and every missing ancestor as a mode-0700 real directory."""
    missing: list[Path] = []
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    for existing in (cursor, *cursor.parents):
        if existing == path:
            # The leaf's own kind and mode are the storage root's, not its
            # ancestry's; validate_private_directory below reports it as such.
            continue
        try:
            metadata = existing.lstat()
        except OSError as exc:
            raise ArtifactVisibilityError("artifact runtime ancestry is not accessible") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactVisibilityError("artifact runtime ancestry contains a symlink")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
        _fsync_directory(directory.parent)
    validate_private_directory(path, label="private storage root")


def _preflight_owner_root(path: Path, expected: dict[str, object], *, label: str) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    validate_private_directory(path, label=label)
    owner_path = path / "owner.json"
    if not owner_path.exists() and not owner_path.is_symlink():
        if any(path.iterdir()):
            raise ArtifactVisibilityError(f"{label} is nonempty without owner metadata")
        return False
    metadata = owner_path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ArtifactVisibilityError(f"{label} owner metadata is unsafe")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ArtifactVisibilityError(f"{label} owner metadata must have mode 0600")
    _validate_owner(owner_path, expected)
    return True


def _source_git_identity(source: Path, *, action: str) -> tuple[Path, Path]:
    """Resolve one canonical source directory together with its Git common dir."""
    canonical = _declared_directory(source, label="private workspace source")
    try:
        return canonical, git_ops.git_common_dir(canonical)
    except git_ops.GitError as exc:
        raise ArtifactVisibilityError(f"could not {action} private workspace Git ownership") from exc


def resolve_private_workspace_owner(source: Path, *, locations: PrivateRootLocations) -> PrivateWorkspaceOwner:
    """Resolve and durably validate the source-owned sibling namespaces."""
    canonical_source, common_dir = _source_git_identity(source, action="resolve")
    _validate_private_root_declaration(locations.artifact_runtime, label="artifact runtime")
    _validate_private_root_declaration(locations.operational_workspaces, label="operational workspace root")
    declared_artifacts, declared_operations = locations.artifact_runtime, locations.operational_workspaces
    if _overlaps(declared_artifacts, declared_operations):
        raise ArtifactVisibilityError("private storage roots overlap")
    for private_root in (declared_artifacts, declared_operations):
        if _overlaps(private_root, canonical_source) or _overlaps(private_root, common_dir):
            raise ArtifactVisibilityError("private storage root overlaps source Git ownership")
    workspace_key = _workspace_key(canonical_source, common_dir)
    artifact_state = declared_artifacts / workspace_key
    operational_state = declared_operations / workspace_key
    expected = _private_owner_payload(canonical_source, common_dir, workspace_key)

    artifact_owned = _preflight_owner_root(artifact_state, expected, label="artifact state root")
    operational_owned = _preflight_owner_root(operational_state, expected, label="operational state root")
    for root in (declared_artifacts, declared_operations, artifact_state, operational_state):
        _create_private_directory(root)
    if not artifact_owned:
        _atomic_json(artifact_state / "owner.json", expected)
    if not operational_owned:
        _atomic_json(operational_state / "owner.json", expected)
    _preflight_owner_root(artifact_state, expected, label="artifact state root")
    _preflight_owner_root(operational_state, expected, label="operational state root")
    if (artifact_state / "owner.json").read_bytes() != (operational_state / "owner.json").read_bytes():
        raise ArtifactVisibilityError("private peer owner metadata is not byte-identical")
    return PrivateWorkspaceOwner(
        source=canonical_source,
        git_common_dir=common_dir,
        workspace_key=workspace_key,
        artifact_state_root=artifact_state,
        operational_state_root=operational_state,
    )


def validate_private_workspace_owner(owner: PrivateWorkspaceOwner, *, source: Path, repo: Path | None = None) -> None:
    """Reattest one supplied owner at a consuming boundary."""
    canonical_source, common_dir = _source_git_identity(source, action="validate")
    expected_key = _workspace_key(canonical_source, common_dir)
    if (
        not isinstance(owner.source, Path)
        or not isinstance(owner.git_common_dir, Path)
        or not isinstance(owner.artifact_state_root, Path)
        or not isinstance(owner.operational_state_root, Path)
        or type(owner.workspace_key) is not str
        or owner.source != canonical_source
        or owner.git_common_dir != common_dir
        or owner.workspace_key != expected_key
    ):
        raise ArtifactVisibilityError("private workspace owner identity mismatch")
    artifact_parent = owner.artifact_state_root.parent
    operational_parent = owner.operational_state_root.parent
    if (
        owner.artifact_state_root.name != expected_key
        or owner.operational_state_root.name != expected_key
        or _overlaps(artifact_parent, operational_parent)
    ):
        raise ArtifactVisibilityError("private workspace owner path mismatch")
    for path, label in (
        (artifact_parent, "artifact runtime"),
        (operational_parent, "operational workspace root"),
        (owner.artifact_state_root, "artifact state root"),
        (owner.operational_state_root, "operational state root"),
    ):
        _validate_private_root_declaration(path, label=label)
        validate_private_directory(path, label=label)
    for private_root in (artifact_parent, operational_parent):
        if _overlaps(private_root, canonical_source) or _overlaps(private_root, common_dir):
            raise ArtifactVisibilityError("private workspace owner overlaps source Git ownership")
    expected = _private_owner_payload(canonical_source, common_dir, expected_key)
    _preflight_owner_root(owner.artifact_state_root, expected, label="artifact state root")
    _preflight_owner_root(owner.operational_state_root, expected, label="operational state root")
    if (owner.artifact_state_root / "owner.json").read_bytes() != (
        owner.operational_state_root / "owner.json"
    ).read_bytes():
        raise ArtifactVisibilityError("private peer owner metadata is not byte-identical")
    if repo is not None:
        canonical_repo = _declared_directory(repo, label="artifact repo")
        try:
            repo_common = git_ops.git_common_dir(canonical_repo)
        except git_ops.GitError as exc:
            raise ArtifactVisibilityError("could not validate artifact repo Git ownership") from exc
        if repo_common != common_dir:
            raise ArtifactVisibilityError("artifact repo does not share the source Git identity")
        if _overlaps(artifact_parent, canonical_repo):
            raise ArtifactVisibilityError("artifact runtime and repository overlap")


def operational_worktree_path(owner: PrivateWorkspaceOwner) -> Path:
    """Return the source-owned operational worktree directory without creating it."""
    return owner.operational_state_root / "operational"


def operational_worktree_root(owner: PrivateWorkspaceOwner) -> Path:
    """Create and validate the source-owned operational worktree directory."""
    root = operational_worktree_path(owner)
    _create_private_directory(root)
    return root


def derive_workspace_identity(work: WorkContext, *, owner: PrivateWorkspaceOwner) -> ArtifactWorkspaceIdentity:
    """Validate a WorkContext against one pre-resolved private owner."""
    validate_private_workspace_owner(owner, source=work.source, repo=work.repo)
    repo = _declared_directory(work.repo, label="artifact repo")
    try:
        source_git_dir = git_ops.git_dir(owner.source)
        repo_git_dir = git_ops.git_dir(repo)
    except git_ops.GitError as exc:
        raise ArtifactVisibilityError("could not resolve artifact Git directory ownership") from exc
    return ArtifactWorkspaceIdentity(
        repo=repo,
        source=owner.source,
        git_common_dir=owner.git_common_dir,
        source_git_dir=source_git_dir,
        repo_git_dir=repo_git_dir,
        operational_state_root=owner.operational_state_root,
        state_root=owner.artifact_state_root,
        workspace_key=owner.workspace_key,
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        _fsync_file(path)
        _fsync_directory(path.parent)
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise
    finally:
        os.close(fd)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArtifactVisibilityError("artifact metadata is not a regular file")
        value = json.loads(_read_regular(path, metadata))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactVisibilityError("artifact metadata is malformed") from exc
    if not isinstance(value, dict):
        raise ArtifactVisibilityError("artifact metadata is malformed")
    return cast(dict[str, Any], value)


def _load_ledger(
    path: Path,
    *,
    items_key: str,
    message: str,
    identity: Sequence[tuple[str, object]] = (),
    required: bool = False,
) -> list[dict[str, object]]:
    """Read one versioned ledger envelope, returning its still-unvalidated items.

    An absent ledger reads as empty unless ``required``; per-entry validation
    stays with the caller that knows the entry shape.
    """
    if not required and not path.exists() and not path.is_symlink():
        return []
    payload = _load_json(path)
    if (
        set(payload) != {"schema_version", items_key, *(key for key, _ in identity)}
        or payload.get("schema_version") != _SCHEMA_VERSION
        or any(payload.get(key) != value for key, value in identity)
        or not isinstance(payload.get(items_key), list)
    ):
        raise ArtifactVisibilityError(message)
    return cast(list[dict[str, object]], payload[items_key])


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_relative_name(name: str) -> None:
    raw_parts = name.split("/")
    if (
        not name
        or "\0" in name
        or "\\" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in raw_parts)
    ):
        raise ArtifactVisibilityError("artifact manifest contains an unsafe path")


def _read_regular(path: Path, metadata: os.stat_result) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ArtifactVisibilityError("artifact file changed during inspection")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise ArtifactVisibilityError("artifact file could not be read safely") from exc
    finally:
        os.close(fd)


def _walk(
    root: Path,
    path: Path,
    entries: list[ArtifactManifestEntry],
    inodes: set[tuple[int, int]],
    *,
    digest: bool,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ArtifactVisibilityError("artifact entry could not be inspected") from exc
    relative = path.relative_to(root).as_posix()
    _validate_relative_name(relative)
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactVisibilityError("artifact roots accept only regular files and directories")
    if stat.S_ISDIR(metadata.st_mode):
        entries.append(ArtifactManifestEntry(relative, "directory", 0, mode, None))
        try:
            children = sorted(path.iterdir(), key=lambda child: os.fsencode(child.name))
        except OSError as exc:
            raise ArtifactVisibilityError("artifact directory could not be inspected") from exc
        normalized: set[str] = set()
        for child in children:
            key = unicodedata.normalize("NFC", child.name)
            if key in normalized:
                raise ArtifactVisibilityError("artifact tree contains duplicate normalized names")
            normalized.add(key)
            _walk(root, child, entries, inodes, digest=digest)
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactVisibilityError("artifact roots accept only regular files and directories")
    inode = (metadata.st_dev, metadata.st_ino)
    if inode in inodes:
        raise ArtifactVisibilityError("artifact tree contains duplicate filesystem aliases")
    inodes.add(inode)
    if not digest:
        entries.append(ArtifactManifestEntry(relative, "file", metadata.st_size, mode, None))
        return
    content = _read_regular(path, metadata)
    entries.append(ArtifactManifestEntry(relative, "file", len(content), mode, hashlib.sha256(content).hexdigest()))


def _manifest(
    root: Path,
    names: Sequence[str] | None = None,
    *,
    digest: bool = True,
) -> tuple[ArtifactManifestEntry, ...]:
    """Enumerate a tree, hashing every file unless ``digest`` is disabled.

    A digest-free listing carries no ``sha256`` and is therefore only valid for
    enumerating a tree in order to delete it, never for attestation or storage.
    """
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise ArtifactVisibilityError("artifact manifest root is inaccessible") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ArtifactVisibilityError("artifact manifest root must be a real directory")
    entries: list[ArtifactManifestEntry] = []
    inodes: set[tuple[int, int]] = set()
    selected = sorted(names if names is not None else (child.name for child in root.iterdir()), key=os.fsencode)
    normalized: set[str] = set()
    for name in selected:
        _validate_relative_name(name)
        key = unicodedata.normalize("NFC", name)
        if key in normalized:
            raise ArtifactVisibilityError("artifact tree contains duplicate normalized names")
        normalized.add(key)
        path = root / name
        if path.exists() or path.is_symlink():
            _walk(root, path, entries, inodes, digest=digest)
    return tuple(entries)


def _manifest_payload(entries: tuple[ArtifactManifestEntry, ...]) -> dict[str, object]:
    return {"schema_version": _SCHEMA_VERSION, "entries": [asdict(entry) for entry in entries]}


def _parse_manifest(path: Path) -> tuple[ArtifactManifestEntry, ...]:
    raw_entries = _load_ledger(
        path,
        items_key="entries",
        message="artifact manifest schema is unsupported",
        required=True,
    )
    result: list[ArtifactManifestEntry] = []
    seen: set[str] = set()
    normalized: set[str] = set()
    for raw in raw_entries:
        entry = _manifest_entry_from_payload(raw, message="artifact manifest is malformed")
        if entry.path in seen:
            raise ArtifactVisibilityError("artifact manifest contains duplicate paths")
        normalized_path = unicodedata.normalize("NFC", entry.path)
        if normalized_path in normalized:
            raise ArtifactVisibilityError("artifact manifest contains duplicate normalized paths")
        seen.add(entry.path)
        normalized.add(normalized_path)
        result.append(entry)
    if [entry.path for entry in result] != sorted((entry.path for entry in result), key=os.fsencode):
        raise ArtifactVisibilityError("artifact manifest entries are not sorted")
    return tuple(result)


def _entries_belong_to(relative: str, entries: tuple[ArtifactManifestEntry, ...]) -> bool:
    prefix = f"{relative}/"
    return all(entry.path == relative or entry.path.startswith(prefix) for entry in entries)


def _entry_at(
    entries: Sequence[ArtifactManifestEntry],
    relative: str,
    *,
    kind: str | None = None,
) -> ArtifactManifestEntry | None:
    """Return the manifest entry at ``relative``, optionally of one kind only."""
    return next(
        (
            entry
            for entry in entries
            if entry.path == relative and (kind is None or entry.kind == kind)
        ),
        None,
    )


def _write_baseline_manifest(transaction: Path, index: int, record: _DestinationRecord) -> None:
    """Persist one record's baseline manifest, which never changes after capture."""
    _atomic_json(transaction / f"destination-{index:04d}-baseline-manifest.json", _manifest_payload(record.baseline))


def _write_destination_records(
    transaction: Path,
    records: Sequence[_DestinationRecord],
    *,
    include_published: bool,
    include_baseline: bool = False,
) -> None:
    """Rewrite the destination ledger.

    Baseline manifests are immutable once captured, so they are only written
    when this transaction has not seen them yet (``include_baseline``).
    """
    registry: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if include_baseline:
            _write_baseline_manifest(transaction, index, record)
        if include_published:
            _atomic_json(
                transaction / f"destination-{index:04d}-published-manifest.json",
                _manifest_payload(record.published),
            )
        registry.append(
            {
                "index": index,
                "record_id": record.record_id,
                "requested": record.requested,
                "base": record.base,
                "relative": record.relative,
                "label": record.label.value,
                "delivery": record.delivery.value,
                "expected_kind": record.expected_kind,
                "baseline_state": record.baseline_state,
                "missing_parents": list(record.missing_parents),
                "expected_dev": record.expected_dev,
                "expected_ino": record.expected_ino,
                "prepared_sha256": record.prepared_sha256,
                "installed_sha256": record.installed_sha256,
            }
        )
    _atomic_json(
        transaction / "destinations.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "includes_published": include_published,
            "destinations": registry,
        },
    )


_DESTINATION_KEYS = frozenset(
    "index record_id requested base relative label delivery expected_kind baseline_state "
    "missing_parents expected_dev expected_ino prepared_sha256 installed_sha256".split()
)


def _load_destination_records(transaction: Path, *, include_published: bool) -> tuple[_DestinationRecord, ...]:
    raw_destinations = _load_ledger(
        transaction / "destinations.json",
        items_key="destinations",
        message="artifact destination registry is malformed",
        identity=(("includes_published", include_published),),
    )
    result: list[_DestinationRecord] = []
    for expected_index, raw in enumerate(raw_destinations):
        if not isinstance(raw, dict) or set(raw) != _DESTINATION_KEYS:
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        index = raw["index"]
        record_id = raw["record_id"]
        requested = raw["requested"]
        base = raw["base"]
        relative = raw["relative"]
        label_value = raw["label"]
        delivery_value = raw["delivery"]
        expected_kind = raw["expected_kind"]
        baseline_state = raw["baseline_state"]
        missing_raw = raw["missing_parents"]
        expected_dev = raw["expected_dev"]
        expected_ino = raw["expected_ino"]
        prepared_sha256 = raw["prepared_sha256"]
        installed_sha256 = raw["installed_sha256"]
        if (
            type(index) is not int
            or index != expected_index
            or not isinstance(record_id, str)
            or record_id != f"destination-{index:04d}"
            or not isinstance(requested, str)
            or not isinstance(base, str)
            or not isinstance(relative, str)
            or not isinstance(missing_raw, list)
            or not all(isinstance(value, str) for value in missing_raw)
            or expected_kind not in ("file", "directory")
            or baseline_state not in ("absent", "file", "directory")
            or not all(value is None or type(value) is int for value in (expected_dev, expected_ino))
            or not all(value is None or _is_sha256(value) for value in (prepared_sha256, installed_sha256))
        ):
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        _validate_relative_name(relative)
        requested_path = Path(requested)
        base_path = Path(base)
        if (
            not requested_path.is_absolute()
            or _absolute_lexical(requested_path) != requested_path
            or not base_path.is_absolute()
            or _absolute_lexical(base_path) != base_path
            or base_path / relative != requested_path
        ):
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        try:
            label = OutputLabel(label_value)
            delivery = DestinationDelivery(delivery_value)
        except (TypeError, ValueError) as exc:
            raise ArtifactVisibilityError("artifact destination registry is malformed") from exc
        if label in _PUBLIC_LABELS:
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        baseline = _parse_manifest(transaction / f"destination-{index:04d}-baseline-manifest.json")
        published = (
            _parse_manifest(transaction / f"destination-{index:04d}-published-manifest.json")
            if include_published
            else ()
        )
        if not _entries_belong_to(relative, baseline) or not _entries_belong_to(relative, published):
            raise ArtifactVisibilityError("artifact destination manifest identity is malformed")
        if baseline_state == "absent" and baseline:
            raise ArtifactVisibilityError("artifact destination baseline identity is malformed")
        root_entry = _entry_at(baseline, relative)
        if baseline_state != "absent" and (root_entry is None or root_entry.kind != baseline_state):
            raise ArtifactVisibilityError("artifact destination baseline identity is malformed")
        missing_parents = tuple(cast(list[str], missing_raw))
        for missing in missing_parents:
            _validate_relative_name(missing)
            if Path(missing) not in Path(relative).parents:
                raise ArtifactVisibilityError("artifact destination parent identity is malformed")
        if any(_overlaps(requested_path, Path(existing.requested)) for existing in result):
            raise ArtifactVisibilityError("artifact destination registry contains overlapping paths")
        result.append(
            _DestinationRecord(
                record_id,
                requested,
                base,
                relative,
                label,
                delivery,
                expected_kind,
                baseline_state,
                baseline,
                missing_parents,
                cast(int | None, expected_dev),
                cast(int | None, expected_ino),
                cast(str | None, prepared_sha256),
                cast(str | None, installed_sha256),
                published,
            )
        )
    return tuple(result)


def _copy_tree(source: Path, destination: Path, entries: tuple[ArtifactManifestEntry, ...]) -> None:
    destination.mkdir(parents=True, mode=0o700)
    for entry in entries:
        target = destination / entry.path
        if entry.kind == "directory":
            target.mkdir()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        original = source / entry.path
        source_metadata = original.lstat()
        content = _read_regular(original, source_metadata)
        if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
            raise ArtifactVisibilityError("artifact file changed during copy")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, entry.mode)
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fchmod(fd, entry.mode)
                os.utime(fd, ns=(source_metadata.st_atime_ns, source_metadata.st_mtime_ns))
                os.fsync(handle.fileno())
        finally:
            os.close(fd)
    for entry in reversed(entries):
        if entry.kind == "directory":
            directory = destination / entry.path
            os.chmod(directory, entry.mode)
            _fsync_directory(directory)
    _fsync_directory(destination)
    if _manifest(destination) != entries:
        raise ArtifactVisibilityError("artifact copy verification failed")


_transfer_observer: Any | None = None
_transfer_post_observer: Any | None = None


def _stage_registry(transaction: Path) -> list[dict[str, object]]:
    result = _load_ledger(
        transaction / "stages.json",
        items_key="stages",
        message="artifact transfer-stage registry is malformed",
        identity=(("transaction_id", transaction.name),),
    )
    for index, entry in enumerate(result):
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {"stage_id", "path", "purpose", "record_id", "parent_dev", "parent_ino"}
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
    _atomic_json(
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
    _atomic_json(
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
    if _load_json(stage / "stage-owner.json") != expected:
        raise ArtifactVisibilityError("artifact transfer-stage owner is malformed")


def _cleanup_registered_stages(transaction: Path, *, workspace_key: str) -> None:
    for entry in _stage_registry(transaction):
        stage = Path(cast(str, entry["path"]))
        if not stage.exists() and not stage.is_symlink():
            continue
        _validate_transfer_stage(stage, transaction=transaction, workspace_key=workspace_key)
        for intent in _load_transfer_intents(stage):
            relative = cast(str, intent["relative"])
            expected = _manifest_entry_from_payload(intent["expected"])
            moved_entries = _manifest(stage / "entries", (relative,))
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
        _remove_owned_tree(stage, stage.parent)


def _manifest_entry_from_payload(
    value: object,
    *,
    message: str = "artifact transfer intent is malformed",
) -> ArtifactManifestEntry:
    """Validate one persisted manifest entry payload into its dataclass."""
    if not isinstance(value, dict) or set(value) != {"path", "kind", "size", "mode", "sha256"}:
        raise ArtifactVisibilityError(message)
    relative = value["path"]
    kind = value["kind"]
    size = value["size"]
    mode = value["mode"]
    digest = value["sha256"]
    if (
        not isinstance(relative, str)
        or kind not in ("directory", "file")
        or type(size) is not int
        or type(mode) is not int
        or size < 0
        or not 0 <= mode <= 0o7777
        or (kind == "directory" and (size != 0 or digest is not None))
        or (kind == "file" and not _is_sha256(digest))
    ):
        raise ArtifactVisibilityError(message)
    _validate_relative_name(relative)
    return ArtifactManifestEntry(
        relative,
        cast(Literal["directory", "file"], kind),
        size,
        mode,
        cast(str | None, digest),
    )


def _load_transfer_intents(stage: Path, *, owner: dict[str, Any] | None = None) -> list[dict[str, object]]:
    path = stage / "intents.json"
    if not path.exists() and not path.is_symlink():
        return []
    if owner is None:
        owner = _load_json(stage / "stage-owner.json")
    result = _load_ledger(
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
        _validate_relative_name(relative)
        expected = _manifest_entry_from_payload(intent["expected"])
        if expected.path != relative or not source.is_absolute() or _absolute_lexical(source) != source:
            raise ArtifactVisibilityError("artifact transfer intent is malformed")
    return result


def _record_transfer_intent(stage: Path, *, path: Path, relative: str, expected: ArtifactManifestEntry) -> None:
    owner = _load_json(stage / "stage-owner.json")
    intents = _load_transfer_intents(stage, owner=owner)
    intents.append(
        {
            "index": len(intents),
            "record_id": owner["record_id"],
            "source": str(path),
            "relative": relative,
            "expected": asdict(expected),
        }
    )
    _atomic_json(
        stage / "intents.json",
        {
            "schema_version": _SCHEMA_VERSION,
            "stage_id": owner["stage_id"],
            "intents": intents,
        },
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
    conflicts = _load_ledger(
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
    _atomic_json(
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
        stage_id=cast(str, _load_json(stage / "stage-owner.json")["stage_id"]),
    )


def _transfer_entry(
    path: Path,
    *,
    stage: Path,
    relative: str,
    expected: ArtifactManifestEntry,
    transaction: Path,
) -> None:
    moved = stage / "entries" / relative
    moved.parent.mkdir(parents=True, exist_ok=True)
    _record_transfer_intent(stage, path=path, relative=relative, expected=expected)
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
    observed_entries = _manifest(stage / "entries", (relative,))
    observed = next((entry for entry in observed_entries if entry.path == relative), None)
    if observed != expected:
        if observed is None:
            raise ArtifactVisibilityError("artifact ownership transfer produced no entry")
        _record_transfer_conflict(
            transaction,
            record_id=str(_load_json(stage / "stage-owner.json")["record_id"]),
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


def _deepest_first_directories(entries: Sequence[ArtifactManifestEntry]) -> list[ArtifactManifestEntry]:
    """Return the directory entries ordered so children always precede parents."""
    return sorted(
        (entry for entry in entries if entry.kind == "directory"),
        key=lambda item: item.path.count("/"),
        reverse=True,
    )


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
    if _manifest(root, names) != entries:
        raise ArtifactVisibilityError(changed_message)
    stage = _create_transfer_stage(
        stage_parent,
        transaction=transaction,
        workspace_key=workspace_key,
        purpose=purpose,
        record_id=record_id,
    )
    for entry in entries:
        if entry.kind == "file":
            _transfer_entry(
                root / entry.path,
                stage=stage,
                relative=entry.path,
                expected=entry,
                transaction=transaction,
            )
    if remove_directories:
        for entry in _deepest_first_directories(entries):
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
    _remove_owned_tree(stage, stage_parent)


def _remove_owned_tree(path: Path, owner: Path) -> None:
    if path.parent != owner or path.is_symlink():
        raise ArtifactVisibilityError("refusing unsafe transaction cleanup")
    if not path.exists():
        return
    entries = _manifest(path, digest=False)
    for entry in entries:
        if entry.kind == "file":
            (path / entry.path).unlink()
    for entry in _deepest_first_directories(entries):
        (path / entry.path).rmdir()
    path.rmdir()
    _fsync_directory(owner)


_cleanup_observer: Any | None = None
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
    result = _load_ledger(
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
                value is None or type(value) is int
                for value in (entry.get("entry_dev"), entry.get("entry_ino"))
            )
            or entry.get("failure_reason") not in _EXTERNAL_FAILURE_REASONS
            or not (entry.get("destination_record_id") is None or isinstance(entry.get("destination_record_id"), str))
        ):
            raise ArtifactVisibilityError("external entry ledger is malformed")
        parent = Path(cast(str, entry["parent"]))
        name = cast(str, entry["name"])
        if not parent.is_absolute() or _absolute_lexical(parent) != parent:
            raise ArtifactVisibilityError("external entry ledger is malformed")
        _validate_exchange_name(name)
        digest = entry["expected_sha256"]
        if digest is not None and not _is_sha256(digest):
            raise ArtifactVisibilityError("external entry ledger is malformed")
    return result


def _write_external_records(transaction: Path, records: list[dict[str, object]]) -> None:
    _atomic_json(
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
    if entry_dev is not None:
        current["entry_dev"] = entry_dev
    if entry_ino is not None:
        current["entry_ino"] = entry_ino
    if expected_mode is not None:
        current["expected_mode"] = expected_mode
    if expected_sha256 is not None:
        current["expected_sha256"] = expected_sha256
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
    canonical = _declared_directory(parent, label="external output parent")
    fd = os.open(
        canonical,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    metadata = os.fstat(fd)
    return fd, metadata


def _external_identity(parent_fd: int, name: str) -> tuple[int, int, int, str] | _ExternalEntryIssue | None:
    _validate_exchange_name(name)
    try:
        fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
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
        baseline = _entry_at(record.baseline, record.relative, kind="file")
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
    _append_conflict(
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
    _declared_directory(cursor, label="external output ancestor")
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
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
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
            or (metadata.st_dev, metadata.st_ino)
            != (record["entry_dev"], record["entry_ino"])
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
    _persist_live_destination_record(transaction, updated)
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
                "identity_changed"
                if observed_issue is not None
                else _external_failure_reason(result, probe=False)
            ),
            observed=observed_issue,
        )
        raise ArtifactVisibilityError("live external atomic exchange result is ambiguous")
    finally:
        os.close(parent_fd)


def _cleanup_root(state_root: Path) -> Path:
    root = state_root / "cleanup"
    _create_private_directory(root)
    return root


def _load_cleanup_ticket(path: Path, state_root: Path) -> dict[str, object]:
    payload = _load_json(path)
    if (
        set(payload)
        != {"schema_version", "workspace_key", "transaction_id", "terminal_state", "stage_ids"}
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
    _remove_owned_tree(path, cleanup)
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
        stage_ids = [str(entry["stage_id"]) for entry in _stage_registry(active)]
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
        registry_payload = _load_json(registry)
        include_published = registry_payload.get("includes_published")
        if type(include_published) is not bool:
            raise ArtifactVisibilityError("artifact destination registry is malformed")
        destinations = _load_destination_records(transaction, include_published=include_published)
    else:
        destinations = ()
    _reconcile_external_entries(transaction, destinations)
    _cleanup_registered_stages(transaction, workspace_key=state_root.name)
    _notify_cleanup("STAGES_REMOVED", transaction.name)
    cleanup = _cleanup_root(state_root)
    ticket = cleanup / f"{transaction.name}.json"
    if ticket.exists() or ticket.is_symlink():
        raise ArtifactVisibilityError("artifact cleanup ticket collision")
    _atomic_json(
        ticket,
        {
            "schema_version": _SCHEMA_VERSION,
            "workspace_key": state_root.name,
            "transaction_id": transaction.name,
            "terminal_state": terminal_state.value,
            "stage_ids": [str(entry["stage_id"]) for entry in _stage_registry(transaction)],
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
    _atomic_json(
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
    payload = _load_json(journal)
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
    _atomic_json(
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
    payload = _load_json(transaction / "transaction-owner.json")
    if (
        set(payload)
        != {"schema_version", "workspace_key", "transaction_id", "session_id", "kind"}
        or payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("workspace_key") != state_root.name
        or payload.get("transaction_id") != transaction.name
        or not isinstance(payload.get("session_id"), str)
        or payload.get("kind") not in ("detach", "publish")
    ):
        raise ArtifactVisibilityError("artifact transaction owner is malformed")
    return payload


def _validate_owner(path: Path, expected: dict[str, object]) -> None:
    owner = _load_json(path)
    if (
        set(owner) != {"schema_version", "workspace_key", "source", "git_common_dir"}
        or type(owner["schema_version"]) is not int
        or owner["schema_version"] != _SCHEMA_VERSION
        or not isinstance(owner["workspace_key"], str)
        or not isinstance(owner["source"], str)
        or not isinstance(owner["git_common_dir"], str)
        or owner != expected
    ):
        raise ArtifactVisibilityError("artifact owner metadata does not match the source")


def _validate_legacy_public(source: Path) -> None:
    daydream = source / _DAYDREAM
    if daydream.exists() or daydream.is_symlink():
        try:
            metadata = daydream.lstat()
        except OSError as exc:
            raise ArtifactVisibilityError("artifact root could not be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactVisibilityError("artifact roots accept only regular files and directories")
        children = {child.name for child in daydream.iterdir()}
        anchor_children = set(children)
        for name in sorted(children & _OPERATIONAL_NAMES):
            # An emptied operational root is inert residue (the workspace
            # retirement pass removes it); refusing it here would wedge every
            # future run on the repository. Only a root that still holds
            # entries blocks the detach — and residue roots are also excluded
            # from the anchor check below so an emptied namespace cannot
            # resurface as "no recognized artifact anchor".
            operational = daydream / name
            try:
                operational_metadata = operational.lstat()
                if not stat.S_ISLNK(operational_metadata.st_mode) and stat.S_ISDIR(
                    operational_metadata.st_mode
                ):
                    occupied = any(True for _ in operational.iterdir())
                else:
                    occupied = True
            except OSError as exc:
                raise ArtifactVisibilityError("artifact root could not be inspected") from exc
            if occupied:
                raise ArtifactVisibilityError("legacy operational workspace blocks artifact detach")
            anchor_children.discard(name)
        if anchor_children and not anchor_children.intersection(_LEGACY_ANCHORS):
            raise ArtifactVisibilityError("legacy .daydream tree has no recognized artifact anchor")
    review = source / _REVIEW_OUTPUT
    if review.exists() or review.is_symlink():
        metadata = review.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArtifactVisibilityError("artifact roots accept only regular files and directories")


def _manifest_is_subset(
    actual: tuple[ArtifactManifestEntry, ...],
    expected: tuple[ArtifactManifestEntry, ...],
) -> bool:
    expected_by_path = {entry.path: entry for entry in expected}
    return all(expected_by_path.get(entry.path) == entry for entry in actual)


def _replace_public_from_tree(source: Path, tree: Path, entries: tuple[ArtifactManifestEntry, ...]) -> None:
    """Install a staged public tree onto a source whose own tree is already detached."""
    if _manifest(source, (_DAYDREAM, _REVIEW_OUTPUT)):
        raise ArtifactVisibilityError("public artifacts changed before publication")
    for name in (_DAYDREAM, _REVIEW_OUTPUT):
        staged = tree / name
        if staged.exists() or staged.is_symlink():
            _install_staged_path(staged, source / name)
    if _manifest(source, (_DAYDREAM, _REVIEW_OUTPUT)) != entries:
        raise ArtifactVisibilityError("public artifact publication verification failed")


def _install_staged_path(staged: Path, target: Path) -> None:
    metadata = staged.lstat()
    try:
        if stat.S_ISREG(metadata.st_mode):
            os.link(staged, target, follow_symlinks=False)
            _fsync_file(target)
            _fsync_directory(target.parent)
            staged.unlink()
            _fsync_directory(staged.parent)
        elif stat.S_ISDIR(metadata.st_mode):
            os.replace(staged, target)
            _fsync_directory(target.parent)
        else:
            raise ArtifactVisibilityError("artifact source stage has an unsafe entry")
    except OSError as exc:
        raise ArtifactVisibilityError("artifact destination changed during atomic install") from exc


def _install_regular_noclobber(source: Path, target: Path, entry: ArtifactManifestEntry) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.parent / f".{target.name}.{secrets.token_hex(8)}.install"
    try:
        _copy_regular_entry(source, staged, entry, changed_message="artifact recovery source changed")
        try:
            os.link(staged, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise ArtifactVisibilityError("artifact destination changed during no-clobber install") from exc
        os.chmod(target, entry.mode)
        _fsync_file(target)
        _fsync_directory(target.parent)
    finally:
        with suppress(OSError):
            staged.unlink()
            _fsync_directory(staged.parent)


def _restore_manifest_noclobber(
    root: Path,
    canonical: Path,
    entries: tuple[ArtifactManifestEntry, ...],
) -> tuple[str, ...]:
    conflicts: list[str] = []
    for entry in entries:
        if entry.kind != "directory":
            continue
        target = root / entry.path
        if target.exists() or target.is_symlink():
            metadata = target.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                conflicts.append(entry.path)
            continue
        target.mkdir(mode=entry.mode)
        os.chmod(target, entry.mode)
        _fsync_directory(target.parent)
    for entry in entries:
        if entry.kind != "file":
            continue
        target = root / entry.path
        if target.exists() or target.is_symlink():
            try:
                actual = _manifest(root, (entry.path,))
            except ArtifactVisibilityError:
                conflicts.append(entry.path)
                continue
            if actual != (entry,):
                conflicts.append(entry.path)
            continue
        _install_regular_noclobber(canonical / entry.path, target, entry)
    return tuple(conflicts)


def _source_stage_path(source: Path, transaction_id: str, purpose: str) -> Path:
    if (
        not transaction_id
        or any(character in transaction_id for character in ("/", "\\", "\0"))
        or purpose not in ("publish", "restore")
    ):
        raise ArtifactVisibilityError("artifact source-stage identity is invalid")
    return source.parent / f".{source.name}.daydream-{purpose}-{transaction_id}"


def _source_stage_owner(source: Path, transaction_id: str, purpose: str, workspace_key: str) -> dict[str, object]:
    """Derive the owner attestation a source stage carries, re-stat'ing its parent."""
    parent = source.parent.lstat()
    return {
        "schema_version": _SCHEMA_VERSION,
        "workspace_key": workspace_key,
        "transaction_id": transaction_id,
        "purpose": purpose,
        "parent_dev": parent.st_dev,
        "parent_ino": parent.st_ino,
    }


def _create_source_stage(source: Path, transaction_id: str, purpose: str, *, workspace_key: str) -> Path:
    stage = _source_stage_path(source, transaction_id, purpose)
    if stage.exists() or stage.is_symlink():
        raise ArtifactVisibilityError("artifact source-stage collision")
    stage.mkdir(mode=0o700)
    _atomic_json(stage / "stage-owner.json", _source_stage_owner(source, transaction_id, purpose, workspace_key))
    _fsync_directory(source.parent)
    return stage


def _retire_source_stage(source: Path, transaction_id: str, purpose: str, *, workspace_key: str) -> None:
    """Re-attest this identity's source stage, then remove the tree it owns."""
    stage = _source_stage_path(source, transaction_id, purpose)
    expected = _source_stage_owner(source, transaction_id, purpose, workspace_key)
    if _load_json(stage / "stage-owner.json") != expected:
        raise ArtifactVisibilityError("artifact source-stage ownership is malformed")
    _remove_owned_tree(stage, source.parent)


def _copy_regular_entry(
    source: Path,
    target: Path,
    entry: ArtifactManifestEntry,
    *,
    changed_message: str = "artifact file changed during destination staging",
) -> None:
    """Copy one manifested regular file, refusing a source that no longer matches."""
    content = _read_regular(source, source.lstat())
    if len(content) != entry.size or hashlib.sha256(content).hexdigest() != entry.sha256:
        raise ArtifactVisibilityError(changed_message)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, entry.mode)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)
    os.chmod(target, entry.mode)


def _overlay_destination(
    frozen_root: Path,
    write_relative: str,
    projection: Path,
    record: _DestinationRecord,
) -> tuple[ArtifactManifestEntry, ...]:
    generated = _manifest(frozen_root, (write_relative,))
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
        _copy_regular_entry(source, target, entry)
    for directory in sorted(
        (path for path in projection.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(projection)
    return _manifest(projection, (record.relative,))


def _replace_destination_from_tree(
    tree: Path,
    record: _DestinationRecord,
    *,
    allowed: Sequence[tuple[ArtifactManifestEntry, ...]],
    desired: tuple[ArtifactManifestEntry, ...],
    transaction: Path,
    workspace_key: str,
    source: Path,
) -> None:
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
    actual = _manifest(root, (record.relative,))
    if not any(_manifest_is_subset(actual, candidate) for candidate in allowed):
        raise ArtifactVisibilityError("explicit artifact destination changed during publication")
    if actual:
        _remove_manifested(
            root,
            actual,
            (record.relative,),
            transaction=transaction,
            workspace_key=workspace_key,
            purpose="replace-destination",
            stage_parent=root,
            record_id=record.record_id,
        )
    staged = tree / record.relative
    if desired:
        for parent_relative in record.missing_parents:
            parent = root / parent_relative
            parent.mkdir(mode=0o700)
            _fsync_directory(parent.parent)
        _install_staged_path(staged, root / record.relative)
    else:
        for parent_relative in reversed(record.missing_parents):
            with suppress(OSError):
                (root / parent_relative).rmdir()
                _fsync_directory((root / parent_relative).parent)
    if _manifest(root, (record.relative,)) != desired:
        raise ArtifactVisibilityError("explicit artifact destination verification failed")


def _merge_directory_from_tree(
    tree: Path,
    record: _DestinationRecord,
    *,
    desired: tuple[ArtifactManifestEntry, ...],
    transaction: Path,
    workspace_key: str,
    source: Path,
) -> None:
    root = Path(record.base)
    target_root = root / record.relative
    baseline = {entry.path: entry for entry in record.baseline}
    desired_by_path = {entry.path: entry for entry in desired}
    desired_root = desired_by_path.get(record.relative)
    if desired_root is None or desired_root.kind != "directory":
        raise ArtifactVisibilityError("dump destination projection is malformed")
    if target_root.exists() or target_root.is_symlink():
        metadata = target_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactVisibilityError("dump destination shell changed")
    else:
        target_root.mkdir(parents=True, mode=desired_root.mode)
        _fsync_directory(target_root.parent)

    for entry in desired:
        if entry.kind != "directory" or entry.path == record.relative:
            continue
        target = root / entry.path
        if target.exists() or target.is_symlink():
            metadata = target.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactVisibilityError("dump destination directory changed")
        else:
            target.mkdir(mode=entry.mode)
            _fsync_directory(target.parent)

    requested = Path(record.requested)
    baseline_was_detached = requested == source or source in requested.parents
    for entry in desired:
        if entry.kind != "file":
            continue
        target = root / entry.path
        actual = _manifest(root, (entry.path,))
        actual_entry = next((candidate for candidate in actual if candidate.path == entry.path), None)
        expected = baseline.get(entry.path)
        if actual_entry == entry:
            continue
        if actual_entry is not None:
            if expected is None or actual_entry != expected:
                raise ArtifactVisibilityError("dump destination changed during merge")
            _remove_manifested(
                root,
                actual,
                (entry.path,),
                transaction=transaction,
                workspace_key=workspace_key,
                purpose="replace-dump-file",
                stage_parent=root,
                record_id=record.record_id,
            )
        elif expected is not None and not baseline_was_detached:
            # External baselines remain live; their disappearance is a conflict.
            # In-source baselines were deliberately transferred before dispatch.
            raise ArtifactVisibilityError("dump destination baseline disappeared")
        _install_regular_noclobber(tree / entry.path, target, entry)
    for baseline_entry in record.baseline:
        if baseline_entry.kind == "file" and baseline_entry.path not in desired_by_path:
            raise ArtifactVisibilityError("dump destination projection dropped a baseline file")


def _wrote_recorded_session(actual: Sequence[ArtifactManifestEntry], record: _DestinationRecord) -> bool:
    """Whether the live entry at ``record`` still holds bytes this session wrote."""
    root_entry = _entry_at(actual, record.relative, kind="file")
    return root_entry is not None and root_entry.sha256 in (record.prepared_sha256, record.installed_sha256)


def _reinstall_external_baseline(
    transaction: Path,
    record: _DestinationRecord,
    baseline_root: Path,
    baseline_file: ArtifactManifestEntry,
) -> None:
    """Reinstall one external destination's baseline bytes through the live path."""
    baseline_path = baseline_root / record.relative
    _publish_live_external(
        transaction,
        record,
        _read_regular(baseline_path, baseline_path.lstat()),
        exchange=_name_exchange_factory(),
        capability=_recorded_external_capability(transaction, Path(record.requested).parent),
        content_mode=baseline_file.mode,
    )


def _restore_destination_records(
    source: Path,
    transaction: Path,
    records: Sequence[_DestinationRecord],
    source_stage: Path,
) -> None:
    workspace_key = transaction.parent.parent.name
    for record in records:
        try:
            index = int(record.record_id.removeprefix("destination-"))
        except ValueError as exc:
            raise ArtifactVisibilityError("artifact destination record identity is malformed") from exc
        baseline_root = transaction / f"destination-{index:04d}-baseline"
        install_stage = source_stage / f"destination-{index:04d}"
        _copy_tree(baseline_root, install_stage, record.baseline)
        root = Path(record.base)
        actual = _manifest(root, (record.relative,))
        if record.delivery is DestinationDelivery.LIVE_EXTERNAL:
            if actual == record.baseline:
                continue
            baseline_file = _entry_at(record.baseline, record.relative, kind="file")
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
                expected = _external_target_identity_from_ledger(transaction, record)
            requested_metadata = Path(record.requested).lstat()
            if (requested_metadata.st_dev, requested_metadata.st_ino) != expected:
                raise ArtifactVisibilityError("external trajectory identity changed during recovery")
            if baseline_file is None:
                _remove_manifested(
                    root,
                    actual,
                    (record.relative,),
                    transaction=transaction,
                    workspace_key=workspace_key,
                    purpose="restore-live-external",
                    stage_parent=root,
                    record_id=record.record_id,
                )
                continue
            _reinstall_external_baseline(transaction, record, baseline_root, baseline_file)
            continue
        allowed = [record.baseline]
        if record.published:
            allowed.append(record.published)
        recorded_session = _wrote_recorded_session(actual, record)
        if not any(_manifest_is_subset(actual, candidate) for candidate in allowed) and not recorded_session:
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
        for index, external in enumerate(_external_records(transaction))
        if external["purpose"] == _ExternalEntryPurpose.MISSING_PARENT.value
        and external["lifecycle"] == _ExternalEntryLifecycle.ATTESTED.value
    ]
    _cleanup_external_directories(transaction, parent_indexes)


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


def _external_target_identity_from_ledger(transaction: Path, destination: _DestinationRecord) -> tuple[int, int]:
    for record in reversed(_external_records(transaction)):
        if (
            record["destination_record_id"] == destination.record_id
            and record["purpose"] == _ExternalEntryPurpose.PUBLICATION_LINK_TARGET.value
            and record["lifecycle"]
            in (_ExternalEntryLifecycle.ATTESTED.value, _ExternalEntryLifecycle.INSTALLED.value)
            and type(record["entry_dev"]) is int
            and type(record["entry_ino"]) is int
        ):
            return record["entry_dev"], record["entry_ino"]
    raise ArtifactVisibilityError("external target identity is not durably attested")


def _persist_live_destination_record(transaction: Path, updated: _DestinationRecord) -> None:
    records = list(_load_destination_records(transaction, include_published=False))
    for index, record in enumerate(records):
        if record.record_id == updated.record_id:
            records[index] = updated
            _write_destination_records(transaction, records, include_published=False)
            return
    raise ArtifactVisibilityError("live external destination ledger identity is missing")


def _reconcile_failed_detach(state_root: Path, source: Path, transaction: Path) -> None:
    canonical = state_root / "canonical"
    canonical_manifest = state_root / "canonical-manifest.json"
    if not canonical.exists() or not canonical_manifest.exists():
        return
    entries = _parse_manifest(canonical_manifest)
    if _manifest(canonical) != entries:
        raise ArtifactVisibilityError("canonical artifact recovery copy is corrupt")
    conflicts = _restore_manifest_noclobber(source, canonical, entries)
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
            source_stage = _source_stage_path(source, transaction.name, "publish") if publishing else None
            if source_stage is not None and (source_stage.exists() or source_stage.is_symlink()):
                _retire_source_stage(source, transaction.name, "publish", workspace_key=state_root.name)
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
        if (transaction / "conflicts.json").exists() or (transaction / "conflicts.json").is_symlink():
            raise ArtifactVisibilityError("artifact recovery retained a closed conflict")
        if session_id in completed_sessions and not state.is_publish:
            _retire_transaction(state_root, transaction, terminal_state=_TerminalState.DETACHED_RECONCILED)
            continue
        canonical = state_root / "canonical"
        canonical_manifest = state_root / "canonical-manifest.json"
        transaction_manifest = transaction / "manifest.json"
        published_entries: tuple[ArtifactManifestEntry, ...] | None = None
        destination_records = _load_destination_records(transaction, include_published=state.is_publish)
        _reconcile_external_entries(transaction, destination_records)
        publication_stage = (_source_stage_path(source, transaction.name, "publish") if state.is_publish else None)
        restore_stage: Path | None = None
        if state.is_publish:
            published_entries = _parse_manifest(transaction / "publish-manifest.json")
        if state is _Transition.DETACH_STAGED:
            entries = _parse_manifest(transaction_manifest)
            stage = transaction / "detach-stage"
            if not canonical.exists():
                if _manifest(stage) != entries:
                    raise ArtifactVisibilityError("staged detach is incomplete")
                os.replace(stage, canonical)
                _fsync_directory(state_root)
            else:
                if _manifest(canonical) != entries:
                    raise ArtifactVisibilityError("staged detach ownership is ambiguous")
                if stage.exists() and _manifest(stage) != entries:
                    raise ArtifactVisibilityError("staged detach ownership is ambiguous")
            if not canonical_manifest.exists():
                _atomic_json(canonical_manifest, _manifest_payload(entries))
        if state is _Transition.PUBLISH_INSTALLED and not canonical.exists():
            replacement = transaction / "canonical-stage"
            if published_entries is None or _manifest(replacement) != published_entries:
                raise ArtifactVisibilityError("staged canonical publication copy is corrupt")
            os.replace(replacement, canonical)
            _fsync_directory(state_root)
        if not canonical.exists() or not canonical_manifest.exists():
            raise ArtifactVisibilityError("canonical artifact recovery copy is missing")
        canonical_entries = _parse_manifest(canonical_manifest)
        canonical_actual = _manifest(canonical)
        if canonical_actual != canonical_entries and not (
            state is _Transition.PUBLISH_INSTALLED and canonical_actual == published_entries
        ):
            raise ArtifactVisibilityError("canonical artifact recovery copy is corrupt")

        if state is _Transition.PUBLISH_INSTALLED:
            assert published_entries is not None
            if _manifest(source, (_DAYDREAM, _REVIEW_OUTPUT)) != published_entries:
                raise ArtifactVisibilityError("installed public artifact recovery copy is corrupt")
            replacement = transaction / "canonical-stage"
            if replacement.exists():
                if _manifest(replacement) != published_entries:
                    raise ArtifactVisibilityError("staged canonical publication copy is corrupt")
                backup = transaction / "old-canonical"
                os.replace(canonical, backup)
                os.replace(replacement, canonical)
                _fsync_directory(state_root)
            elif _manifest(canonical) != published_entries:
                raise ArtifactVisibilityError("installed canonical publication copy is missing")
            _atomic_json(canonical_manifest, _manifest_payload(published_entries))
            canonical_entries = published_entries
        elif state is _Transition.PUBLISH_VERIFIED:
            assert published_entries is not None
            if canonical_entries != published_entries:
                raise ArtifactVisibilityError("verified canonical publication identity is corrupt")
            if _manifest(source, (_DAYDREAM, _REVIEW_OUTPUT)) != canonical_entries:
                raise ArtifactVisibilityError("verified public artifact recovery copy is corrupt")
        else:
            public_entries = _manifest(source, (_DAYDREAM, _REVIEW_OUTPUT))
            allowed = [canonical_entries]
            if published_entries is not None:
                allowed.append(published_entries)
            if not any(_manifest_is_subset(public_entries, candidate) for candidate in allowed):
                _append_conflict(
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
                _remove_manifested(
                    source,
                    public_entries,
                    transaction=transaction,
                    workspace_key=state_root.name,
                    purpose="recover-public",
                    stage_parent=source.parent,
                )
            restore_stage = _create_source_stage(source, transaction.name, "restore", workspace_key=state_root.name)
            public_stage = restore_stage / "public"
            _copy_tree(canonical, public_stage, canonical_entries)
            _replace_public_from_tree(source, public_stage, canonical_entries)
        if state in (_Transition.PUBLISH_INSTALLED, _Transition.PUBLISH_VERIFIED):
            for record in destination_records:
                if _manifest(Path(record.base), (record.relative,)) != record.published:
                    raise ArtifactVisibilityError("installed explicit artifact recovery copy is corrupt")
        else:
            if destination_records:
                assert restore_stage is not None
                _restore_destination_records(source, transaction, destination_records, restore_stage)
            assert restore_stage is not None
            _retire_source_stage(source, transaction.name, "restore", workspace_key=state_root.name)
        if state.is_publish:
            completed_sessions.add(session_id)
            if publication_stage is None:
                raise ArtifactVisibilityError("artifact publication stage is missing")
            if publication_stage.exists() or publication_stage.is_symlink():
                _retire_source_stage(source, transaction.name, "publish", workspace_key=state_root.name)
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


class ArtifactSession:
    """Held workspace lease and strict live-path router for one run."""

    def __init__(
        self,
        layout: ArtifactLayout,
        *,
        lock_fd: int,
        repo_fd: int,
        canonical_entries: tuple[ArtifactManifestEntry, ...],
        detach_transaction: Path,
    ) -> None:
        self.layout = layout
        self._lock_fd = lock_fd
        self._repo_fd = repo_fd
        self._canonical_entries = canonical_entries
        self._detach_transaction = detach_transaction
        self._state: Literal["active", "frozen", "publishing", "published", "closed"] = "active"
        self._destinations: list[RoutedDestination] = []
        self._routed: list[_RoutedRecord] = []
        self._trajectory_route: TrajectoryOutputRoute | None = None
        self._frozen_snapshot: ArtifactTreeSnapshot | None = None
        self._name_exchange: _AtomicNameExchange | None = None
        self._external_capabilities: dict[Path, tuple[int, int]] = {}
        self._created_external_parents: list[int] = []

    @property
    def daydream_dir(self) -> Path:
        return self.layout.daydream_dir

    @property
    def review_output(self) -> Path:
        return self.layout.review_output

    @property
    def provenance(self) -> ArtifactEvidenceProvenance:
        return ArtifactEvidenceProvenance(
            workspace_key=self.layout.workspace_key,
            session_id=self.layout.session_id,
            public_source=self.layout.source,
            live_root=self.layout.live_root,
        )

    def _require_active(self) -> None:
        if self._state != "active":
            raise ArtifactVisibilityError("artifact session is frozen and no longer writable")

    def _route_repo(self, repo: Path) -> None:
        self._require_active()
        declared = _absolute_lexical(repo)
        try:
            canonical, metadata = _declared_directory_metadata(declared, label="artifact repo")
            held = os.fstat(self._repo_fd)
        except (OSError, ArtifactVisibilityError) as exc:
            raise ArtifactVisibilityError("requested repo does not match active artifact session") from exc
        if (
            declared != self.layout.repo
            or canonical != self.layout.repo
            or (metadata.st_dev, metadata.st_ino) != (held.st_dev, held.st_ino)
        ):
            raise ArtifactVisibilityError("requested repo does not match active artifact session")

    def _validate_destination(
        self,
        requested: Path,
        *,
        label: OutputLabel,
        additional: Sequence[Path] = (),
        public_subtree_owner: RoutedDestination | None = None,
    ) -> tuple[Path, Path, bool]:
        self._require_active()
        if not isinstance(label, OutputLabel):
            raise ArtifactVisibilityError("artifact destination label is unsupported")
        if public_subtree_owner is not None and not (
            any(public_subtree_owner is destination for destination in self._destinations)
            and self._is_public_daydream_owner(public_subtree_owner)
        ):
            raise ArtifactVisibilityError("public trajectory owner identity mismatch")
        if not requested.is_absolute():
            raise ArtifactVisibilityError("artifact destination must be absolute")
        if "\0" in os.fspath(requested):
            raise ArtifactVisibilityError("artifact destination contains an unsafe path")
        declared = _absolute_lexical(requested)
        declared_inside_source = self.layout.source in declared.parents
        if declared_inside_source and label not in _PUBLIC_LABELS:
            relative = declared.relative_to(self.layout.source).as_posix()
            try:
                tracked = git_ops.tracked_path_collisions(self.layout.source, relative)
            except git_ops.GitError as exc:
                raise ArtifactVisibilityError("could not validate artifact destination ownership") from exc
            if tracked:
                raise ArtifactVisibilityError("tracked artifact destination collision")
        for index, parent in enumerate((declared, *declared.parents)):
            if not parent.exists() and not parent.is_symlink():
                continue
            metadata = parent.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ArtifactVisibilityError("artifact destination ancestry contains a symlink")
            expected_directory = label in (OutputLabel.PUBLIC_DAYDREAM, OutputLabel.DUMP_DIRECTORY)
            if index == 0 and (
                expected_directory != stat.S_ISDIR(metadata.st_mode)
                or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
            ):
                raise ArtifactVisibilityError("artifact destination has the wrong filesystem type")
            if index > 0 and not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactVisibilityError("artifact destination ancestry is not a directory")
            if parent == self.layout.source.parent:
                break
        try:
            canonical = declared.resolve(strict=False)
        except OSError as exc:
            raise ArtifactVisibilityError("artifact destination could not be resolved") from exc
        protected = (
            self.layout.artifact_runtime_root,
            self.layout.operational_workspaces_root,
            self.layout.source / ".git",
            self.layout.repo / ".git",
            self.layout.source_git_dir,
            self.layout.repo_git_dir,
            self.layout.git_common_dir,
        )
        if canonical == self.layout.source:
            raise ArtifactVisibilityError("artifact destination may not replace the source root")
        if any(_overlaps(canonical, path) for path in protected):
            raise ArtifactVisibilityError("artifact destination overlaps private or Git storage")
        if self.layout.repo != self.layout.source and _overlaps(canonical, self.layout.repo):
            raise ArtifactVisibilityError("artifact destination overlaps the active model repository")
        owned_public_subtree = (
            public_subtree_owner is not None
            and label in _TRAJECTORY_LABELS
            and self.layout.public_daydream_dir in canonical.parents
        )
        if label not in _PUBLIC_LABELS and (
            _overlaps(canonical, self.layout.public_daydream_dir)
            or canonical == self.layout.public_review_output
        ) and not owned_public_subtree:
            raise ArtifactVisibilityError("artifact destination overlaps a public compatibility root")
        prior_paths = [
            prior.requested.resolve(strict=False)
            for prior in self._destinations
            if not (owned_public_subtree and prior is public_subtree_owner)
        ]
        prior_paths.extend(additional)
        for prior_path in prior_paths:
            if canonical == prior_path:
                raise ArtifactVisibilityError("artifact destination collision")
            if _overlaps(canonical, prior_path):
                raise ArtifactVisibilityError("artifact destination overlap")
        inside_source = canonical == self.layout.source or self.layout.source in canonical.parents
        return declared, canonical, inside_source

    def _is_public_daydream_owner(self, destination: RoutedDestination) -> bool:
        """Return whether one route is this session's public ``.daydream`` owner."""
        return (
            destination.label is OutputLabel.PUBLIC_DAYDREAM
            and destination.requested == self.layout.public_daydream_dir
            and destination.write_path == self.layout.daydream_dir
            and destination.frozen_path == self.layout.daydream_dir
            and destination.delivery is DestinationDelivery.DEFERRED
        )

    def _public_daydream_owner(self) -> RoutedDestination | None:
        return next(
            (
                destination
                for destination in self._destinations
                if self._is_public_daydream_owner(destination)
            ),
            None,
        )

    def _private_public_trajectory_path(self, canonical: Path, *, owner: RoutedDestination) -> Path:
        if not any(owner is destination for destination in self._destinations):
            raise ArtifactVisibilityError("public trajectory owner identity mismatch")
        try:
            relative = canonical.relative_to(self.layout.public_daydream_dir)
        except ValueError as exc:
            raise ArtifactVisibilityError("public trajectory path is outside its owner") from exc
        _validate_relative_name(relative.as_posix())
        private = self.layout.daydream_dir / relative
        ancestry = [self.layout.daydream_dir]
        for part in relative.parts:
            ancestry.append(ancestry[-1] / part)
        for cursor in ancestry:
            try:
                metadata = cursor.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ArtifactVisibilityError("private trajectory path could not be inspected") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ArtifactVisibilityError("private trajectory ancestry contains a symlink")
            is_leaf = cursor == private
            if is_leaf and not stat.S_ISREG(metadata.st_mode):
                raise ArtifactVisibilityError("private trajectory has the wrong filesystem type")
            if not is_leaf and not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactVisibilityError("private trajectory ancestry is not a directory")
        return private

    def _capture_destination_record(self, route: RoutedDestination, *, canonical: Path, inside_source: bool) -> None:
        dump = route.label is OutputLabel.DUMP_DIRECTORY
        if inside_source:
            base = self.layout.source
        else:
            base = canonical.parent
            while not base.exists() and not base.is_symlink():
                base = base.parent
        relative = canonical.relative_to(base).as_posix()
        _validate_relative_name(relative)
        baseline = _manifest(base, (relative,))
        root_entry = next((entry for entry in baseline if entry.path == relative), None)
        baseline_state: Literal["absent", "file", "directory"] = ("absent" if root_entry is None else root_entry.kind)
        expected_dev: int | None = None
        expected_ino: int | None = None
        if not inside_source and root_entry is not None and root_entry.kind == "file":
            metadata = canonical.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ArtifactVisibilityError("external destination baseline identity changed")
            expected_dev = metadata.st_dev
            expected_ino = metadata.st_ino
        missing: list[str] = []
        cursor = canonical.parent
        while cursor != base and not cursor.exists() and not cursor.is_symlink():
            missing.append(cursor.relative_to(base).as_posix())
            cursor = cursor.parent
        missing.reverse()
        index = len(self._routed)
        record = _DestinationRecord(
            record_id=f"destination-{index:04d}",
            requested=str(route.requested),
            base=str(base),
            relative=relative,
            label=route.label,
            delivery=route.delivery,
            expected_kind="directory" if dump else "file",
            baseline_state=baseline_state,
            baseline=baseline,
            missing_parents=tuple(missing),
            expected_dev=expected_dev,
            expected_ino=expected_ino,
        )
        baseline_root = self._detach_transaction / f"destination-{index:04d}-baseline"
        _copy_tree(base, baseline_root, baseline)
        self._routed.append(_RoutedRecord(route, record))
        _write_baseline_manifest(self._detach_transaction, index, record)
        self._persist_records()
        if inside_source and baseline:
            _remove_manifested(
                base,
                baseline,
                (relative,),
                transaction=self._detach_transaction,
                workspace_key=self.layout.workspace_key,
                purpose="detach-dump" if dump else "detach-destination",
                stage_parent=base,
                record_id=record.record_id,
                remove_directories=not dump,
                changed_message="explicit artifact destination changed during detach",
            )

    def _records(self) -> list[_DestinationRecord]:
        return [item.record for item in self._routed]

    def _persist_records(self) -> None:
        _write_destination_records(self._detach_transaction, self._records(), include_published=False)

    def _routed_for(self, route: RoutedDestination) -> _RoutedRecord:
        item = next((entry for entry in self._routed if entry.route is route), None)
        if item is None:
            raise ArtifactVisibilityError("artifact destination ledger identity mismatch")
        return item

    def register_destination(self, requested: Path, *, label: OutputLabel) -> RoutedDestination:
        if label in _TRAJECTORY_LABELS:
            raise ArtifactVisibilityError("paired trajectory registration is required")
        declared, canonical, inside_source = self._validate_destination(requested, label=label)
        if label is OutputLabel.PUBLIC_DAYDREAM:
            if canonical != self.layout.public_daydream_dir:
                raise ArtifactVisibilityError("public Daydream output has an invalid destination")
            write_path: Path | None = self.layout.daydream_dir
            delivery = DestinationDelivery.DEFERRED
        elif label is OutputLabel.PUBLIC_REVIEW_OUTPUT:
            if canonical != self.layout.public_review_output:
                raise ArtifactVisibilityError("public review output has an invalid destination")
            write_path = self.layout.review_output
            delivery = DestinationDelivery.DEFERRED
        elif _overlaps(canonical, self.layout.public_daydream_dir) or canonical == self.layout.public_review_output:
            raise ArtifactVisibilityError("artifact destination overlaps a public compatibility root")
        elif label is OutputLabel.DUMP_DIRECTORY:
            write_path = None
            delivery = DestinationDelivery.FINALIZATION_MERGE
        else:
            write_path = self.layout.live_root / ".explicit" / f"{len(self._destinations):04d}" / declared.name
            delivery = DestinationDelivery.DEFERRED
        routed = RoutedDestination(label, declared, write_path, write_path, delivery)
        self._destinations.append(routed)
        if label not in _PUBLIC_LABELS:
            self._capture_destination_record(routed, canonical=canonical, inside_source=inside_source)
        return routed

    @staticmethod
    def _paired_trajectory(
        full_requested: Path,
        partial_requested: Path,
        private_full: Path,
        private_partial: Path,
        delivery: DestinationDelivery,
    ) -> tuple[RoutedDestination, RoutedDestination]:
        """Build the full/partial pair; only a live-external pair writes in place."""
        external = delivery is DestinationDelivery.LIVE_EXTERNAL
        return (
            RoutedDestination(
                OutputLabel.EXPLICIT_TRAJECTORY,
                full_requested,
                full_requested if external else private_full,
                private_full,
                delivery,
            ),
            RoutedDestination(
                OutputLabel.EXPLICIT_TRAJECTORY_PARTIAL,
                partial_requested,
                partial_requested if external else private_partial,
                private_partial,
                delivery,
            ),
        )

    def register_trajectory_output(self, requested: Path | None) -> TrajectoryOutputRoute:
        """Register the root trajectory and its P07 partial as one atomic route."""
        self._require_active()
        if self._trajectory_route is not None:
            raise ArtifactVisibilityError("paired trajectory output is already registered")
        run_dir = self.daydream_dir / "runs" / self.layout.session_id
        default_requested = self.layout.public_daydream_dir / "runs" / self.layout.session_id / "trajectory.json"
        declared_requested = default_requested if requested is None else _absolute_lexical(requested)
        try:
            canonical_requested = declared_requested.resolve(strict=False)
            canonical_default = default_requested.resolve(strict=False)
        except OSError as exc:
            raise ArtifactVisibilityError("artifact destination could not be resolved") from exc
        partial_requested = declared_requested.with_suffix(declared_requested.suffix + ".partial")
        private_full = run_dir / "trajectory.json"
        private_partial = run_dir / "trajectory.json.partial"
        public_root = self.layout.public_daydream_dir
        if canonical_requested == canonical_default:
            full, partial = self._paired_trajectory(
                declared_requested,
                partial_requested,
                private_full,
                private_partial,
                DestinationDelivery.DEFERRED,
            )
        elif public_root in canonical_requested.parents:
            owner = self._public_daydream_owner()
            if owner is None:
                self._validate_destination(declared_requested, label=OutputLabel.EXPLICIT_TRAJECTORY)
                raise AssertionError("unreachable public trajectory validation")
            full_declared, full_canonical, _full_inside = self._validate_destination(
                declared_requested,
                label=OutputLabel.EXPLICIT_TRAJECTORY,
                public_subtree_owner=owner,
            )
            partial_declared, partial_canonical, _partial_inside = self._validate_destination(
                partial_requested,
                label=OutputLabel.EXPLICIT_TRAJECTORY_PARTIAL,
                additional=(full_canonical,),
                public_subtree_owner=owner,
            )
            if not (public_root in full_canonical.parents and public_root in partial_canonical.parents):
                raise ArtifactVisibilityError("paired trajectory destinations cross routing boundaries")
            full, partial = self._paired_trajectory(
                full_declared,
                partial_declared,
                self._private_public_trajectory_path(full_canonical, owner=owner),
                self._private_public_trajectory_path(partial_canonical, owner=owner),
                DestinationDelivery.DEFERRED,
            )
        else:
            full_declared, full_canonical, full_inside = self._validate_destination(
                declared_requested,
                label=OutputLabel.EXPLICIT_TRAJECTORY,
            )
            partial_declared, partial_canonical, partial_inside = self._validate_destination(
                partial_requested,
                label=OutputLabel.EXPLICIT_TRAJECTORY_PARTIAL,
                additional=(full_canonical,),
            )
            if full_inside is not partial_inside:
                raise ArtifactVisibilityError("paired trajectory destinations cross routing boundaries")
            delivery = (DestinationDelivery.DEFERRED if full_inside else DestinationDelivery.LIVE_EXTERNAL)
            full, partial = self._paired_trajectory(
                full_declared, partial_declared, private_full, private_partial, delivery
            )
            self._destinations.extend((full, partial))
            initial = len(self._routed)
            try:
                self._capture_destination_record(full, canonical=full_canonical, inside_source=full_inside)
                self._capture_destination_record(partial, canonical=partial_canonical, inside_source=partial_inside)
                if delivery is DestinationDelivery.LIVE_EXTERNAL:
                    if self._name_exchange is None:
                        self._name_exchange = _name_exchange_factory()
                    parent = full_declared.parent
                    self._created_external_parents.extend(_ensure_external_parent(self._detach_transaction, parent))
                    self._external_capabilities[parent] = _probe_external_parent(
                        self._detach_transaction,
                        parent,
                        self._name_exchange,
                    )
            except BaseException as primary:
                try:
                    self._rollback_paired_capture(initial)
                except Exception as recovery_error:
                    primary.add_note(
                        "paired trajectory registration retained a closed recovery conflict "
                        f"({type(recovery_error).__name__})"
                    )
                self._destinations = [
                    destination
                    for destination in self._destinations
                    if destination is not full and destination is not partial
                ]
                raise
        route = TrajectoryOutputRoute(run_dir, full, partial)
        self._trajectory_route = route
        return route

    def _rollback_paired_capture(self, initial: int) -> None:
        """Undo the two destination captures a failed paired registration made."""
        captured = self._records()[initial:]
        if captured:
            restore_stage = _create_source_stage(
                self.layout.source,
                self._detach_transaction.name,
                "restore",
                workspace_key=self.layout.workspace_key,
            )
            try:
                _restore_destination_records(self.layout.source, self._detach_transaction, captured, restore_stage)
            finally:
                _retire_source_stage(
                    self.layout.source,
                    self._detach_transaction.name,
                    "restore",
                    workspace_key=self.layout.workspace_key,
                )
        for index in range(initial, initial + 2):
            baseline_root = self._detach_transaction / f"destination-{index:04d}-baseline"
            if baseline_root.exists() or baseline_root.is_symlink():
                _remove_owned_tree(baseline_root, self._detach_transaction)
            manifest = self._detach_transaction / f"destination-{index:04d}-baseline-manifest.json"
            if manifest.exists() and not manifest.is_symlink():
                manifest.unlink()
        del self._routed[initial:]
        self._persist_records()
        _cleanup_external_directories(self._detach_transaction, self._created_external_parents)
        self._created_external_parents.clear()

    def write_trajectory_document(
        self,
        route: TrajectoryOutputRoute,
        document: TrajectoryDocumentSnapshot,
        status: Literal["complete", "partial"],
    ) -> None:
        """Write exact P07 bytes to private evidence and the authorized live root."""
        self._require_active()
        if route is not self._trajectory_route or status not in ("complete", "partial"):
            raise ArtifactVisibilityError("trajectory output route identity mismatch")
        selected = route.full if status == "complete" else route.partial
        if document.trajectory_id == self.layout.session_id:
            allowed = (selected.requested, selected.frozen_path)
            if document.path not in allowed:
                raise ArtifactVisibilityError("trajectory document path does not match its paired route")
            private_path = selected.frozen_path
        else:
            try:
                relative = document.path.relative_to(route.run_dir)
            except ValueError as exc:
                raise ArtifactVisibilityError("child trajectory path is outside the private run") from exc
            _validate_relative_name(relative.as_posix())
            private_path = document.path
        if private_path is None or type(document.json_bytes) is not bytes:
            raise ArtifactVisibilityError("trajectory document bytes are malformed")
        _atomic_bytes(private_path, document.json_bytes)
        if document.trajectory_id != self.layout.session_id:
            return
        if selected.delivery is not DestinationDelivery.LIVE_EXTERNAL:
            return
        item = self._routed_for(selected)
        digest = hashlib.sha256(document.json_bytes).hexdigest()
        item.record = replace(item.record, prepared_sha256=digest)
        self._persist_records()
        if self._name_exchange is None:
            raise ArtifactVisibilityError("live external atomic exchange was not initialized")
        capability = self._external_capabilities.get(selected.requested.parent)
        if capability is None:
            raise ArtifactVisibilityError("live external output parent was not probed")
        item.record = _publish_live_external(
            self._detach_transaction,
            item.record,
            document.json_bytes,
            exchange=self._name_exchange,
            capability=capability,
        )
        self._persist_records()

    def freeze(self, run_snapshot: RunWriteSnapshot) -> ArtifactTreeSnapshot:
        self._require_active()
        if run_snapshot.status not in ("complete", "partial") or not isinstance(
            run_snapshot.cutoff_at, str
        ) or not run_snapshot.cutoff_at:
            raise ArtifactVisibilityError("run snapshot metadata is malformed")
        if run_snapshot.root_trajectory_id != self.layout.session_id:
            raise ArtifactVisibilityError("run snapshot root does not match artifact session")
        seen_ids: set[str] = set()
        seen_paths: set[Path] = set()
        root_seen = False
        for document in run_snapshot.documents:
            if document.trajectory_id in seen_ids:
                raise ArtifactVisibilityError("run snapshot contains duplicate document identity")
            seen_ids.add(document.trajectory_id)
            path = document.path
            route = self._trajectory_route
            if route is not None and document.trajectory_id == self.layout.session_id:
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
                relative = path.relative_to(self.layout.live_root)
            except ValueError as exc:
                raise ArtifactVisibilityError("run snapshot document path is outside live artifacts") from exc
            _validate_relative_name(relative.as_posix())
            cursor = self.layout.live_root
            for part in relative.parts[:-1]:
                cursor /= part
                if cursor.exists() or cursor.is_symlink():
                    metadata = cursor.lstat()
                    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                        raise ArtifactVisibilityError("run snapshot path has unsafe ancestry")
            if type(document.json_bytes) is not bytes:
                raise ArtifactVisibilityError("run snapshot document bytes are malformed")
            try:
                payload = json.loads(document.json_bytes)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ArtifactVisibilityError("run snapshot document JSON is malformed") from exc
            if not isinstance(payload, dict):
                raise ArtifactVisibilityError("run snapshot document JSON is malformed")
            if (
                payload.get("trajectory_id") != document.trajectory_id
                or payload.get("session_id") != self.layout.session_id
            ):
                raise ArtifactVisibilityError("run snapshot document identity is malformed")
            root_seen = root_seen or document.trajectory_id == self.layout.session_id
            _atomic_bytes(self.layout.live_root / relative, document.json_bytes)
        if not root_seen:
            raise ArtifactVisibilityError("run snapshot is missing its root document")
        entries = _manifest(self.layout.live_root)
        frozen_root = self.layout.live_root.parent / "frozen"
        if frozen_root.exists() or frozen_root.is_symlink():
            raise ArtifactVisibilityError("frozen artifact tree already exists")
        _copy_tree(self.layout.live_root, frozen_root, entries)
        _atomic_json(frozen_root.parent / "frozen-manifest.json", _manifest_payload(entries))
        result = ArtifactTreeSnapshot(
            session_id=self.layout.session_id,
            workspace_key=self.layout.workspace_key,
            root=frozen_root,
            manifest=entries,
            destinations=tuple(self._destinations),
        )
        self._frozen_snapshot = result
        self._state = "frozen"
        return result

    def finalization_merge_path(self, route: RoutedDestination, *, snapshot: ArtifactTreeSnapshot) -> Path:
        if self._state != "frozen" or snapshot is not self._frozen_snapshot:
            raise ArtifactVisibilityError("artifact session is not ready for frozen finalization")
        if not any(route is registered for registered in self._destinations):
            raise ArtifactVisibilityError("artifact destination route identity mismatch")
        if route.delivery is not DestinationDelivery.FINALIZATION_MERGE:
            raise ArtifactVisibilityError("artifact destination is not a finalization merge")
        item = self._routed_for(route)
        if item.late is not None:
            return item.late
        late = self.layout.live_root.parent / "late" / item.record.record_id
        if late.exists() or late.is_symlink():
            raise ArtifactVisibilityError("artifact finalization stage already exists")
        late.mkdir(parents=True, mode=0o700)
        _fsync_directory(late.parent)
        item.late = late
        return late

    def _retire_late_paths(self) -> None:
        late_parent = self.layout.live_root.parent / "late"
        for late in (item.late for item in self._routed if item.late is not None):
            if late.exists() or late.is_symlink():
                if late.parent != late_parent or late.is_symlink() or not late.is_dir():
                    raise ArtifactVisibilityError("artifact finalization stage ownership changed")
                _remove_owned_tree(late, late_parent)
        if late_parent.exists() and not late_parent.is_symlink():
            with suppress(OSError):
                late_parent.rmdir()
                _fsync_directory(late_parent.parent)

    def finalize_frozen(self, snapshot: ArtifactTreeSnapshot, *, disposition: ArtifactDisposition) -> None:
        if self._state != "frozen":
            raise ArtifactVisibilityError("artifact session is not ready for frozen publication")
        if snapshot is not self._frozen_snapshot:
            raise ArtifactVisibilityError("frozen artifact snapshot identity mismatch")
        if not isinstance(disposition, ArtifactDisposition):
            raise ArtifactVisibilityError("artifact disposition is unsupported")
        if len(snapshot.destinations) != len(self._destinations) or any(
            actual is not expected
            for actual, expected in zip(snapshot.destinations, self._destinations, strict=True)
        ):
            raise ArtifactVisibilityError("frozen artifact destination identity mismatch")
        if _manifest(snapshot.root) != snapshot.manifest:
            raise ArtifactVisibilityError("frozen artifact snapshot changed before publication")
        if disposition is ArtifactDisposition.ROLLBACK:
            self._restore_prior()
            self._retire_late_paths()
            self._state = "published"
            return
        public_entries = tuple(
            entry
            for entry in snapshot.manifest
            if entry.path == _DAYDREAM
            or entry.path.startswith(f"{_DAYDREAM}/")
            or entry.path == _REVIEW_OUTPUT
        )
        transaction_id = f"publish-{self.layout.session_id}-{secrets.token_hex(8)}"
        transaction = self.layout.state_root / "transactions" / transaction_id
        transaction.mkdir(parents=True)
        _write_transaction_owner(
            transaction,
            workspace_key=self.layout.workspace_key,
            session_id=self.layout.session_id,
            kind="publish",
        )
        mark = partial(
            _write_transition,
            transaction / "journal.json",
            transaction_id=transaction_id,
            session_id=self.layout.session_id,
        )
        publication_stage: Path | None = None
        try:
            _atomic_json(transaction / "publish-manifest.json", _manifest_payload(public_entries))
            publication_stage = _create_source_stage(
                self.layout.source,
                transaction_id,
                "publish",
                workspace_key=self.layout.workspace_key,
            )
            public_stage = publication_stage / "public"
            _copy_tree(snapshot.root, public_stage, public_entries)
            canonical_stage = transaction / "canonical-stage"
            _copy_tree(snapshot.root, canonical_stage, public_entries)
            publish_records: list[_DestinationRecord] = []
            for index, item in enumerate(self._routed):
                destination, baseline_record = item.route, item.record
                baseline_root = self._detach_transaction / f"destination-{index:04d}-baseline"
                publish_baseline = transaction / f"destination-{index:04d}-baseline"
                _copy_tree(baseline_root, publish_baseline, baseline_record.baseline)
                projection = publication_stage / f"destination-{index:04d}"
                _copy_tree(baseline_root, projection, baseline_record.baseline)
                if destination.delivery is DestinationDelivery.LIVE_EXTERNAL:
                    actual = _manifest(Path(baseline_record.base), (baseline_record.relative,))
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
                    write_relative = destination.frozen_path.relative_to(self.layout.live_root).as_posix()
                    published = _overlay_destination(snapshot.root, write_relative, projection, baseline_record)
                elif item.late is None:
                    published = baseline_record.baseline
                else:
                    published = _overlay_destination(item.late.parent, item.late.name, projection, baseline_record)
                publish_records.append(replace(baseline_record, published=published))
            self._retire_late_paths()
            _write_destination_records(transaction, publish_records, include_published=True, include_baseline=True)
            mark(state=_Transition.PUBLISH_STAGED)
        except BaseException:
            if publication_stage is not None:
                _retire_source_stage(
                    self.layout.source, transaction_id, "publish", workspace_key=self.layout.workspace_key
                )
            _retire_transaction(self.layout.state_root, transaction, terminal_state=_TerminalState.PUBLISH_RECONCILED)
            raise
        self._state = "publishing"
        (transaction / "public-backup").mkdir()
        _fsync_directory(transaction)
        mark(state=_Transition.PUBLISH_BACKED_UP)
        _replace_public_from_tree(self.layout.source, public_stage, public_entries)
        for index, record in enumerate(publish_records):
            if record.delivery is DestinationDelivery.LIVE_EXTERNAL:
                continue
            _replace_destination_from_tree(
                publication_stage / f"destination-{index:04d}",
                record,
                allowed=(record.baseline,),
                desired=record.published,
                transaction=transaction,
                workspace_key=self.layout.workspace_key,
                source=self.layout.source,
            )
        mark(state=_Transition.PUBLISH_INSTALLED)
        canonical = self.layout.state_root / "canonical"
        os.replace(canonical, transaction / "old-canonical")
        os.replace(canonical_stage, canonical)
        _fsync_directory(self.layout.state_root)
        _atomic_json(self.layout.state_root / "canonical-manifest.json", _manifest_payload(public_entries))
        mark(state=_Transition.PUBLISH_VERIFIED)
        self._canonical_entries = public_entries
        _retire_transaction(
            self.layout.state_root,
            self._detach_transaction,
            terminal_state=_TerminalState.DETACHED_RECONCILED,
        )
        _retire_source_stage(self.layout.source, transaction_id, "publish", workspace_key=self.layout.workspace_key)
        _retire_transaction(self.layout.state_root, transaction, terminal_state=_TerminalState.PUBLISH_VERIFIED)
        self._state = "published"

    def _restore_prior(self) -> None:
        canonical = self.layout.state_root / "canonical"
        source_stage = _create_source_stage(
            self.layout.source,
            self._detach_transaction.name,
            "restore",
            workspace_key=self.layout.workspace_key,
        )
        errors: list[Exception] = []
        try:
            _restore_destination_records(self.layout.source, self._detach_transaction, self._records(), source_stage)
        except Exception as exc:
            errors.append(exc)
        try:
            projection = source_stage / "public"
            _copy_tree(canonical, projection, self._canonical_entries)
            _replace_public_from_tree(self.layout.source, projection, self._canonical_entries)
        except Exception as exc:
            errors.append(exc)
        _retire_source_stage(
            self.layout.source,
            self._detach_transaction.name,
            "restore",
            workspace_key=self.layout.workspace_key,
        )
        try:
            _cleanup_external_directories(self._detach_transaction, self._created_external_parents)
            self._created_external_parents.clear()
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise errors[0]
        self._retire_late_paths()
        _retire_transaction(
            self.layout.state_root,
            self._detach_transaction,
            terminal_state=_TerminalState.DETACHED_RECONCILED,
        )

    def _close(self) -> None:
        self._state = "closed"
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        os.close(self._repo_fd)


def artifact_dir_for(repo: Path) -> Path:
    session = _SESSION.get()
    if session is None:
        return repo / _DAYDREAM
    session._route_repo(repo)
    return session.daydream_dir


def artifact_session_active() -> bool:
    """Whether the current task is bound to one live artifact session."""
    return _SESSION.get() is not None


def review_output_path_for(repo: Path) -> Path:
    session = _SESSION.get()
    if session is None:
        return repo / _REVIEW_OUTPUT
    session._route_repo(repo)
    return session.review_output


def assert_model_cwd_clean(cwd: Path) -> None:
    declared = _declared_directory(cwd, label="model cwd")
    session = _SESSION.get()
    if session is not None:
        for destination in session._destinations:
            if destination.delivery is DestinationDelivery.LIVE_EXTERNAL:
                requested = destination.requested.resolve(strict=False)
                if _overlaps(declared, requested):
                    raise ArtifactVisibilityError("model cwd overlaps a live external artifact destination")
    for name in (_DAYDREAM, _REVIEW_OUTPUT):
        candidate = declared / name
        if candidate.exists() or candidate.is_symlink():
            raise ArtifactVisibilityError("model cwd contains generated Daydream artifacts")


@contextmanager
def bind_artifact_session(session: ArtifactSession) -> Iterator[ArtifactSession]:
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)


def _rebaseline_canonical_from_public(
    state_root: Path,
    source: Path,
    public_entries: tuple[ArtifactManifestEntry, ...],
    *,
    transaction: Path,
) -> None:
    """Adopt the observed public tree as the new canonical recovery baseline.

    Between two runs the published artifacts exist in two synchronized
    copies: the public tree and the canonical recovery copy under the private
    state root. Any benign external change to the public tree between runs
    (a deleted ``.review-output.md``, ``git clean -fdx``, a stray external
    write) previously bricked the checkout: session open compared the two
    copies strictly and failed with no in-band recovery path. Session open
    is the one safe place to reconcile: no artifact transaction is in
    flight, the session lock excludes concurrent sessions, so the observed
    public state is adopted as the new baseline with an actionable warning.
    Divergence detected mid-run (during detach or publication) still fails
    closed with the conflict machinery.
    """
    canonical = state_root / "canonical"
    canonical_manifest = state_root / "canonical-manifest.json"
    backup = transaction / "rebaseline-old-canonical"
    stage = transaction / "rebaseline-stage"
    if canonical.exists():
        os.replace(canonical, backup)
    try:
        _copy_tree(source, stage, public_entries)
        if canonical.exists() or canonical.is_symlink():
            _remove_owned_tree(canonical, state_root)
        os.replace(stage, canonical)
        _fsync_directory(state_root)
        _atomic_json(canonical_manifest, _manifest_payload(public_entries))
    except BaseException:
        with suppress(OSError):
            shutil.rmtree(stage, ignore_errors=True)
        if not canonical.exists() and backup.exists():
            os.replace(backup, canonical)
            _fsync_directory(state_root)
        raise
    with suppress(OSError):
        shutil.rmtree(backup, ignore_errors=True)
    _print_rebaseline_warning(source, canonical)


def _print_rebaseline_warning(source: Path, canonical: Path) -> None:
    from daydream.agent import console
    from daydream.ui import print_warning

    print_warning(
        console,
        "Public artifacts changed between runs; adopting the observed public "
        f"state at {source / _DAYDREAM} (and {source / _REVIEW_OUTPUT}) as the "
        f"new baseline in the canonical recovery copy at {canonical}. "
        "Mid-run divergence still fails closed.",
    )


def _open_layout(work: WorkContext, session_id: str, owner: PrivateWorkspaceOwner) -> ArtifactSession:
    """Detach the public tree and return the held session, on a worker thread."""
    if not session_id or "\0" in session_id or "/" in session_id or "\\" in session_id or session_id in (".", ".."):
        raise ArtifactVisibilityError("artifact session id is invalid")
    identity = derive_workspace_identity(work, owner=owner)
    source = identity.source
    workspace_key = identity.workspace_key
    state_root = identity.state_root
    with ExitStack() as held:
        repo_fd = _open_directory_descriptor(identity.repo, label="artifact repo")
        held.callback(os.close, repo_fd)
        try:
            collisions = git_ops.tracked_artifact_collisions(source)
        except git_ops.GitError as exc:
            raise ArtifactVisibilityError("could not validate artifact Git ownership") from exc
        if collisions:
            raise ArtifactVisibilityError("tracked artifact collision blocks the run")
        lock_path = state_root / ".artifact.lock"
        try:
            if lock_path.exists() or lock_path.is_symlink():
                metadata = lock_path.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise ArtifactVisibilityError("artifact workspace lock is unsafe")
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except (OSError, ArtifactVisibilityError) as exc:
            raise ArtifactVisibilityError("artifact workspace lock is unsafe") from exc
        held.callback(os.close, lock_fd)
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise ArtifactVisibilityError("artifact workspace lock is unsafe")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArtifactVisibilityError("artifact workspace is locked by another process") from exc
        held.callback(fcntl.flock, lock_fd, fcntl.LOCK_UN)
        transaction: Path | None = None
        try:
            _recover_transactions(state_root, source)
            _validate_legacy_public(source)
            runs = state_root / "runs"
            transactions = state_root / "transactions"
            _create_private_directory(runs)
            _create_private_directory(transactions)
            run_root = runs / session_id
            if run_root.exists() or run_root.is_symlink():
                raise ArtifactVisibilityError("artifact session id already exists")
            run_root.mkdir(mode=0o700)

            public_entries = _manifest(source, (_DAYDREAM, _REVIEW_OUTPUT))
            transaction_id = f"detach-{session_id}-{secrets.token_hex(8)}"
            transaction = transactions / transaction_id
            transaction.mkdir(mode=0o700)
            _write_transaction_owner(transaction, workspace_key=workspace_key, session_id=session_id, kind="detach")
            mark = partial(
                _write_transition,
                transaction / "journal.json",
                transaction_id=transaction_id,
                session_id=session_id,
            )
            detach_stage = transaction / "detach-stage"
            canonical = state_root / "canonical"
            canonical_manifest = state_root / "canonical-manifest.json"
            # Canonical is itself the durable byte-identical copy the DETACH_REMOVING
            # ordering needs, so stage one only when this workspace has none yet.
            # Recovery tolerates the absent stage for exactly this reason. Session
            # open is the one safe reconciliation point for a benign between-run
            # public drift: the workspace lock excludes concurrent sessions and no
            # artifact transaction is in flight, so the drift is adopted as the new
            # canonical baseline BEFORE the detach transaction stages anything.
            # Doing this after staging would leave the stale stage the recovery
            # path validates against on a crash.
            canonical_present = canonical.exists()
            if canonical_present:
                canonical_entries = _parse_manifest(canonical_manifest)
                if public_entries != canonical_entries:
                    _rebaseline_canonical_from_public(
                        state_root, source, public_entries, transaction=transaction
                    )
                    canonical_entries = public_entries
            else:
                _copy_tree(source, detach_stage, public_entries)
            _atomic_json(transaction / "manifest.json", _manifest_payload(public_entries))
            mark(state=_Transition.DETACH_STAGED)
            if canonical_present:
                # Re-baselining already made canonical match the observed public
                # state; keep the equality the recovery path relies on.
                canonical_entries = _parse_manifest(canonical_manifest)
                if public_entries != canonical_entries:
                    raise ArtifactVisibilityError("canonical artifact recovery copy is inconsistent")
            else:
                os.replace(detach_stage, canonical)
                _fsync_directory(state_root)
                _atomic_json(canonical_manifest, _manifest_payload(public_entries))
                canonical_entries = public_entries
            mark(state=_Transition.DETACH_CANONICAL)
            mark(state=_Transition.DETACH_REMOVING)
            if public_entries:
                _remove_manifested(
                    source,
                    public_entries,
                    transaction=transaction,
                    workspace_key=workspace_key,
                    purpose="detach-public",
                    stage_parent=source.parent,
                )
            mark(state=_Transition.DETACHED)
            layout = ArtifactLayout(
                repo=identity.repo,
                source=source,
                git_common_dir=identity.git_common_dir,
                source_git_dir=identity.source_git_dir,
                repo_git_dir=identity.repo_git_dir,
                operational_workspaces_root=identity.operational_state_root.parent,
                session_id=session_id,
                state_root=state_root,
            )
            _copy_tree(canonical, layout.live_root, canonical_entries)
        except BaseException as primary:
            if transaction is not None and transaction.exists() and not transaction.is_symlink():
                try:
                    _reconcile_failed_detach(state_root, source, transaction)
                except Exception:
                    primary.add_note("artifact recovery retained a closed conflict")
            raise
        held.pop_all()
    return ArtifactSession(
        layout,
        lock_fd=lock_fd,
        repo_fd=repo_fd,
        canonical_entries=canonical_entries,
        detach_transaction=transaction,
    )


def _close_artifact_session(session: ArtifactSession) -> None:
    """Reconcile and close one acquired session on a blocking worker thread."""
    primary: BaseException | None = None
    try:
        if session._state == "publishing":
            _recover_transactions(session.layout.state_root, session.layout.source)
        elif session._state not in ("published", "closed"):
            session._restore_prior()
    except BaseException as exc:
        primary = exc
    try:
        session._close()
    except BaseException as close_error:
        if primary is None:
            raise
        primary.add_note(f"artifact session close failed ({type(close_error).__name__})")
    if primary is not None:
        raise primary


@asynccontextmanager
async def open_artifact_session(
    work: WorkContext,
    *,
    session_id: str,
    owner: PrivateWorkspaceOwner,
) -> AsyncIterator[ArtifactSession]:
    with anyio.CancelScope(shield=True):
        session = await anyio.to_thread.run_sync(partial(_open_layout, work, session_id, owner))
    token = _SESSION.set(session)
    primary: BaseException | None = None
    try:
        yield session
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(_close_artifact_session, session)
        except BaseException as recovery_error:
            if primary is None:
                raise
            primary.add_note(f"artifact recovery retained a closed conflict ({type(recovery_error).__name__})")
        finally:
            _SESSION.reset(token)
