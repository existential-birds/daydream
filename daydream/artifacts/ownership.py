"""Validated source ownership and disjoint private workspace roots."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Self

from daydream import git_ops

if TYPE_CHECKING:
    from daydream.git_ops.captured_source import CapturedSource
    from daydream.workspace import WorkContext


from daydream.artifacts import filesystem
from daydream.artifacts.models import (
    _SCHEMA_VERSION,
    ArtifactSourceUnavailableError,
    ArtifactVisibilityError,
    ArtifactWorkspaceIdentity,
    PrivateRootLocations,
    PrivateWorkspaceOwner,
)


def _default_private_base() -> Path:
    """Return the default private base; tests patch only this provider."""
    return Path.home() / ".daydream"


def private_root_locations(*, base: Path | None = None) -> PrivateRootLocations:
    """Return disjoint sibling artifact and operational root declarations."""
    selected = _default_private_base() if base is None else base
    if not selected.is_absolute() or filesystem._absolute_lexical(selected) != selected:
        raise ArtifactVisibilityError("private storage base must be an absolute lexical path")
    return PrivateRootLocations(artifact_runtime=selected / "runtime", operational_workspaces=selected / "workspaces")


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


def _preflight_owner_root(path: Path, expected: dict[str, object], *, label: str) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    filesystem.validate_private_directory(path, label=label)
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
    filesystem._validate_owner(owner_path, expected)
    return True


def _source_git_identity(source: Path, *, action: str) -> tuple[Path, Path]:
    """Resolve one canonical source directory together with its Git common dir."""
    canonical = filesystem._declared_directory(source, label="private workspace source")
    try:
        return canonical, git_ops.git_common_dir(canonical)
    except git_ops.GitError as exc:
        raise ArtifactVisibilityError(f"could not {action} private workspace Git ownership") from exc


def resolve_private_workspace_owner(source: Path, *, locations: PrivateRootLocations) -> PrivateWorkspaceOwner:
    """Resolve and durably validate the source-owned sibling namespaces."""
    canonical_source, common_dir = _source_git_identity(source, action="resolve")
    filesystem._validate_private_root_declaration(locations.artifact_runtime, label="artifact runtime")
    filesystem._validate_private_root_declaration(locations.operational_workspaces, label="operational workspace root")
    declared_artifacts, declared_operations = locations.artifact_runtime, locations.operational_workspaces
    if filesystem._overlaps(declared_artifacts, declared_operations):
        raise ArtifactVisibilityError("private storage roots overlap")
    for private_root in (declared_artifacts, declared_operations):
        if filesystem._overlaps(private_root, canonical_source) or filesystem._overlaps(private_root, common_dir):
            raise ArtifactVisibilityError("private storage root overlaps source Git ownership")
    workspace_key = _workspace_key(canonical_source, common_dir)
    artifact_state = declared_artifacts / workspace_key
    operational_state = declared_operations / workspace_key
    expected = _private_owner_payload(canonical_source, common_dir, workspace_key)

    artifact_owned = _preflight_owner_root(artifact_state, expected, label="artifact state root")
    operational_owned = _preflight_owner_root(operational_state, expected, label="operational state root")
    for root in (declared_artifacts, declared_operations, artifact_state, operational_state):
        filesystem._create_private_directory(root)
    if not artifact_owned:
        filesystem._atomic_json(artifact_state / "owner.json", expected)
    if not operational_owned:
        filesystem._atomic_json(operational_state / "owner.json", expected)
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
        or filesystem._overlaps(artifact_parent, operational_parent)
    ):
        raise ArtifactVisibilityError("private workspace owner path mismatch")
    for path, label in (
        (artifact_parent, "artifact runtime"),
        (operational_parent, "operational workspace root"),
        (owner.artifact_state_root, "artifact state root"),
        (owner.operational_state_root, "operational state root"),
    ):
        filesystem._validate_private_root_declaration(path, label=label)
        filesystem.validate_private_directory(path, label=label)
    for private_root in (artifact_parent, operational_parent):
        if filesystem._overlaps(private_root, canonical_source) or filesystem._overlaps(private_root, common_dir):
            raise ArtifactVisibilityError("private workspace owner overlaps source Git ownership")
    expected = _private_owner_payload(canonical_source, common_dir, expected_key)
    _preflight_owner_root(owner.artifact_state_root, expected, label="artifact state root")
    _preflight_owner_root(owner.operational_state_root, expected, label="operational state root")
    if (owner.artifact_state_root / "owner.json").read_bytes() != (
        owner.operational_state_root / "owner.json"
    ).read_bytes():
        raise ArtifactVisibilityError("private peer owner metadata is not byte-identical")
    if repo is not None:
        canonical_repo = filesystem._declared_directory(repo, label="artifact repo")
        try:
            repo_common = git_ops.git_common_dir(canonical_repo)
        except git_ops.GitError as exc:
            raise ArtifactVisibilityError("could not validate artifact repo Git ownership") from exc
        if repo_common != common_dir:
            raise ArtifactVisibilityError("artifact repo does not share the source Git identity")
        if filesystem._overlaps(artifact_parent, canonical_repo):
            raise ArtifactVisibilityError("artifact runtime and repository overlap")


def operational_worktree_path(owner: PrivateWorkspaceOwner) -> Path:
    """Return the source-owned operational worktree directory without creating it."""
    return owner.operational_state_root / "operational"


def operational_worktree_root(owner: PrivateWorkspaceOwner) -> Path:
    """Create and validate the source-owned operational worktree directory."""
    root = operational_worktree_path(owner)
    filesystem._create_private_directory(root)
    return root


def derive_workspace_identity(work: WorkContext, *, owner: PrivateWorkspaceOwner) -> ArtifactWorkspaceIdentity:
    """Validate a WorkContext against one pre-resolved private owner."""
    validate_private_workspace_owner(owner, source=work.source,
                                     repo=work.repo if work.captured_source is None else None)
    if work.captured_source is not None:
        _validate_captured_workspace(work.repo, work.captured_source, owner)
    repo = filesystem._declared_directory(work.repo, label="artifact repo")
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


def _validate_captured_workspace(repo: Path, capture: CapturedSource, owner: PrivateWorkspaceOwner) -> None:
    """Bind independent review storage explicitly to the existing private run owner."""
    if repo != capture.root or not repo.is_relative_to(operational_worktree_path(owner)):
        raise ArtifactVisibilityError('captured review source is outside its private workspace owner')
    capture.validate()


class SourceOwnership:
    """A pinned checkout capability; durable proof and mutations share its identity.

    Only this owner acquires or compares checkout identities. A recovered capability
    must match the journal's proof before any operation can use the current path.
    """

    def __init__(self, path: Path, *, label: str = "artifact source") -> None:
        self.path, expected = filesystem._available_directory_metadata(path, label=label)
        self._label = label
        try:
            self._descriptor = filesystem._open_directory_descriptor(self.path, label=label)
        except ArtifactVisibilityError as exc:
            filesystem._validate_directory_identity(
                self.path, device=expected.st_dev, inode=expected.st_ino, label=label,
            )
            if isinstance(exc.__cause__, (FileNotFoundError, PermissionError)):
                raise ArtifactSourceUnavailableError(f"{label} is unavailable") from exc.__cause__
            raise
        try:
            self._metadata = os.fstat(self._descriptor)
            if (expected.st_dev, expected.st_ino) != (self._metadata.st_dev, self._metadata.st_ino):
                raise ArtifactVisibilityError(f"{label} directory identity changed during acquisition")
            self.validate()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def validate(self) -> None:
        if self._descriptor < 0:
            raise ArtifactVisibilityError("checkout ownership is closed")
        filesystem._validate_directory_identity(
            self.path, device=self._metadata.st_dev, inode=self._metadata.st_ino, label=self._label,
        )

    def _proof(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION, "source": str(self.path),
            "device": self._metadata.st_dev, "inode": self._metadata.st_ino,
        }

    def write_proof(self, transaction: Path) -> None:
        self.validate()
        filesystem._atomic_json(transaction / "source-identity.json", self._proof())

    def validate_proof(self, transaction: Path) -> None:
        try:
            payload = filesystem._load_json(transaction / "source-identity.json")
        except ArtifactVisibilityError as exc:
            raise ArtifactVisibilityError("artifact transaction source identity manifest is unavailable") from exc
        if (
            set(payload) != {"schema_version", "source", "device", "inode"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != _SCHEMA_VERSION
            or payload["source"] != str(self.path)
            or type(payload["device"]) is not int
            or type(payload["inode"]) is not int
        ):
            raise ArtifactVisibilityError("artifact transaction source identity manifest is malformed")
        if payload != self._proof():
            raise ArtifactVisibilityError("artifact source directory identity changed")
        self.validate()

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Reattest successful effects and classify filesystem loss without replacing primary failures."""
        self.validate()
        try:
            yield
        except OSError:
            self.validate()
            raise
        else:
            self.validate()

    @contextmanager
    def parent_descriptor(self, target: Path) -> Iterator[int]:
        """Anchor mutations to the original directory, never its replacement."""
        with self.operation():
            inside_source = self.path in target.parents
            descriptor = (
                os.dup(self._descriptor) if inside_source else filesystem._open_directory_descriptor(
                    filesystem._declared_directory(target.parent, label="artifact output parent"),
                    label="artifact output parent",
                )
            )
            try:
                if inside_source:
                    for component in target.parent.relative_to(self.path).parts:
                        child = os.open(
                            component, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=descriptor,
                        )
                        os.close(descriptor)
                        descriptor = child
                self.validate()
                yield descriptor
            finally:
                os.close(descriptor)

    def mkdir(self, target: Path, *, mode: int, exist_ok: bool = False) -> None:
        with self.parent_descriptor(target) as descriptor:
            try:
                os.mkdir(target.name, mode=mode, dir_fd=descriptor)
            except FileExistsError:
                metadata = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
                if not exist_ok or not stat.S_ISDIR(metadata.st_mode):
                    raise
            os.fsync(descriptor)

    def rmdir(self, target: Path) -> None:
        with self.parent_descriptor(target) as descriptor:
            os.rmdir(target.name, dir_fd=descriptor)
            os.fsync(descriptor)


class CodeWorkspace:
    """Guard live code operations with one captured checkout and Git routing."""

    def __init__(
        self, checkout: SourceOwnership, *, source: SourceOwnership,
        git_directories: tuple[SourceOwnership, ...],
    ) -> None:
        self._checkout = checkout
        self._source = source
        self._git_directories = git_directories
        self.path = checkout.path
        self._marker = self._find_git_marker()

    def _find_git_marker(self) -> Path | None:
        """Retain nested worktrees' discovery boundary before querying Git."""
        for ancestor in (self.path, *self.path.parents):
            marker = ancestor / ".git"
            try:
                marker.lstat()
            except FileNotFoundError:
                continue
            except PermissionError as exc:
                raise ArtifactSourceUnavailableError("artifact code workspace Git marker is unavailable") from exc
            return marker
        return None

    @classmethod
    @contextmanager
    def standalone(cls, path: Path) -> Iterator[CodeWorkspace]:
        """Hold the same authority for standalone Git or plain-directory analysis."""
        with ExitStack() as held:
            checkout = held.enter_context(SourceOwnership(path, label="quality code workspace"))
            code = cls(checkout, source=checkout, git_directories=())
            if code.run(lambda: git_ops.is_git_workspace(checkout.path)):
                paths = code.run(lambda: git_ops.git_dirs(checkout.path))
                try:
                    directories = tuple(
                        held.enter_context(SourceOwnership(p, label="artifact code workspace Git directory"))
                        for p in paths
                    )
                except (ArtifactVisibilityError, OSError):
                    code.validate()
                    raise
                code = cls(checkout, source=checkout, git_directories=directories)
                code.validate()
            yield code

    @property
    def is_git_workspace(self) -> bool:
        """Whether captured Git ownership, rather than recursive directory scope, owns reads."""
        return bool(self._git_directories)

    def validate_checkout(self) -> None:
        self._checkout.validate()

    def _validate_paths(self) -> None:
        self.validate_checkout()
        if any(d.path == self._source.path or self._source.path in d.path.parents for d in self._git_directories):
            self._source.validate()
        for directory in self._git_directories:
            directory.validate()
        if self._marker is not None:
            try:
                self._marker.lstat()
            except (FileNotFoundError, PermissionError) as exc:
                raise ArtifactSourceUnavailableError("artifact code workspace Git marker is unavailable") from exc
            filesystem._validate_projection_ancestry(self._marker, expected_kind="either")

    def validate(self) -> None:
        self._validate_paths()
        if self._git_directories:
            try:
                current = git_ops.git_dirs(self.path)
            except (git_ops.GitError, OSError):
                self._validate_paths()
                raise
            self._validate_paths()
            if current != tuple(d.path for d in self._git_directories):
                raise ArtifactVisibilityError("artifact code workspace Git routing changed")

    def run[T](self, operation: Callable[[], T]) -> T:
        """Enforce operation-time checks, including loss during a failed Git read."""
        self.validate()
        try:
            result = operation()
        except (git_ops.GitError, OSError):
            self.validate()
            raise
        self.validate()
        return result


class WorkspaceOwnership:
    """One descriptor lifetime from workspace setup through guarded Git cleanup."""

    def __init__(self, owner: PrivateWorkspaceOwner) -> None:
        self.owner = owner
        self._held = ExitStack()
        self._directories: dict[Path, SourceOwnership] = {}
        self._code: dict[Path, CodeWorkspace] = {}
        try:
            self.source = self._hold(owner.source, label="artifact source")
            self.source_code = self.bind_code(owner.source)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._held.close()

    def _hold(self, path: Path, *, label: str) -> SourceOwnership:
        if path not in self._directories:
            self._directories[path] = self._held.enter_context(SourceOwnership(path, label=label))
        return self._directories[path]

    def bind_code(self, repo: Path, *, captured_source: CapturedSource | None = None) -> CodeWorkspace:
        if captured_source is not None:
            _validate_captured_workspace(repo, captured_source, self.owner)
        if repo in self._code:
            self._code[repo].validate()
            return self._code[repo]
        checkout = self._hold(repo, label="artifact repo")
        code = CodeWorkspace(checkout, source=self.source, git_directories=())
        directories = code.run(lambda: git_ops.git_dirs(checkout.path))
        if captured_source is None and directories[1] != self.owner.git_common_dir:
            raise ArtifactVisibilityError("artifact repo does not share the source Git identity")
        try:
            git_owners = tuple(self._hold(p, label="artifact code workspace Git directory") for p in directories)
        except (ArtifactVisibilityError, OSError):
            code.validate()
            raise
        code = CodeWorkspace(checkout, source=self.source, git_directories=git_owners)
        code.validate()
        self._code[repo] = code
        return code

    def add_worktree(self, path: Path, ref: str) -> None:
        self.source_code.run(lambda: git_ops.worktree_add(self.source.path, path, ref, detach=True))
        self.bind_code(path)

    def remove_worktree(self, path: Path) -> None:
        self.source_code.validate()
        if path in self._code:
            self._code[path].validate()
        common = self._directories[self.owner.git_common_dir]
        git_ops.worktree_remove(self.source.path, path, force=True, git_directory_fd=common._descriptor)
