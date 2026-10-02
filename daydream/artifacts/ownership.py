"""Validated source ownership and disjoint private workspace roots."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from daydream import git_ops

if TYPE_CHECKING:
    from daydream.workspace import WorkContext


from daydream.artifacts import filesystem
from daydream.artifacts.models import (
    _SCHEMA_VERSION,
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
    validate_private_workspace_owner(owner, source=work.source, repo=work.repo)
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
