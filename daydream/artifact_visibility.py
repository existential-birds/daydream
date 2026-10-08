"""Run-scoped storage boundary for model-invisible Daydream artifacts.

The source checkout exposes compatibility paths only between runs.  While a
session is bound, generated state lives under an owner-validated host runtime
root and callers may address it only through the exact workspace identity."""

from __future__ import annotations

import fcntl
import hashlib
import os
import secrets
import shutil
import stat
from collections.abc import Sequence
from contextlib import ExitStack, asynccontextmanager, suppress
from contextvars import ContextVar
from dataclasses import replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Literal

import anyio

from daydream import git_ops
from daydream.json_utils import _fsync_directory

if TYPE_CHECKING:
    from daydream.trajectory import RunWriteSnapshot, TrajectoryDocumentSnapshot
    from daydream.workspace import WorkContext


from daydream.artifacts import external, filesystem, ledger, ownership, publication, transactions, transfer
from daydream.artifacts.external import _AtomicNameExchange
from daydream.artifacts.filesystem import (
    manifest_tree as manifest_tree,
    validate_private_directory as validate_private_directory,
)
from daydream.artifacts.models import (
    _DAYDREAM,
    _PUBLIC_LABELS,
    _REVIEW_OUTPUT,
    _TRAJECTORY_LABELS,
    ArtifactDisposition as ArtifactDisposition,
    ArtifactEvidenceProvenance as ArtifactEvidenceProvenance,
    ArtifactLayout as ArtifactLayout,
    ArtifactManifestEntry as ArtifactManifestEntry,
    ArtifactTreeSnapshot as ArtifactTreeSnapshot,
    ArtifactVisibilityError as ArtifactVisibilityError,
    ArtifactWorkspaceIdentity as ArtifactWorkspaceIdentity,
    DestinationDelivery as DestinationDelivery,
    OutputLabel as OutputLabel,
    PrivateRootLocations as PrivateRootLocations,
    PrivateWorkspaceOwner as PrivateWorkspaceOwner,
    RoutedDestination as RoutedDestination,
    TrajectoryOutputRoute as TrajectoryOutputRoute,
    _DestinationRecord,
    _RoutedRecord,
    _TerminalState,
    _Transition,
)
from daydream.artifacts.ownership import (
    derive_workspace_identity as derive_workspace_identity,
    operational_worktree_path as operational_worktree_path,
    operational_worktree_root as operational_worktree_root,
    private_root_locations as private_root_locations,
    resolve_private_workspace_owner as resolve_private_workspace_owner,
    validate_private_workspace_owner as validate_private_workspace_owner,
)

_SESSION: ContextVar[ArtifactSession | None] = ContextVar("daydream_artifact_session", default=None)


class _SessionState(str, Enum):
    """Lifecycle of an :class:`ArtifactSession`.

    Exactly the state strings the session always cycled through; ``str``-based
    so serialized diagnostics and comparisons keep their prior textual form.
    """

    ACTIVE = "active"
    FROZEN = "frozen"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    CLOSED = "closed"


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
        self._state = _SessionState.ACTIVE
        self._destinations: list[RoutedDestination] = []
        self._routed: list[_RoutedRecord] = []
        self._trajectory_route: TrajectoryOutputRoute | None = None
        self._completed_sibling_trajectories: dict[str, TrajectoryDocumentSnapshot] = {}
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

    @staticmethod
    def _projection_kind(route: RoutedDestination) -> Literal["file", "directory"]:
        return "directory" if route.label in (OutputLabel.PUBLIC_DAYDREAM, OutputLabel.DUMP_DIRECTORY) else "file"

    def _projection_routes(self) -> tuple[RoutedDestination, ...]:
        trajectory = self._trajectory_route
        paired = () if trajectory is None else (trajectory.full, trajectory.partial)
        return (*paired, *self._destinations)

    def _project_explicit_route(self, declared: Path, from_attr: str, to_attr: str) -> Path | None:
        for route in self._projection_routes():
            if route.label is OutputLabel.PUBLIC_DAYDREAM:
                continue
            from_value = getattr(route, from_attr)
            if from_value is None:
                continue
            if declared == from_value:
                kind = self._projection_kind(route)
                filesystem._validate_projection_ancestry(declared, expected_kind=kind)
                to_value: Path | None = getattr(route, to_attr)
                if to_value is None:
                    raise ArtifactVisibilityError("registered artifact destination has no writable path")
                filesystem._validate_projection_ancestry(to_value, expected_kind=kind)
                return to_value
        return None

    def _project_public_subtree(self, declared: Path, from_attr: str, to_attr: str) -> Path | None:
        for route in self._destinations:
            if route.label is not OutputLabel.PUBLIC_DAYDREAM:
                continue
            from_root: Path | None = getattr(route, from_attr)
            to_root: Path | None = getattr(route, to_attr)
            if from_root is None or to_root is None:
                raise ArtifactVisibilityError("registered artifact destination has no writable path")
            if declared == from_root or from_root in declared.parents:
                relative = declared.relative_to(from_root)
                projected = to_root / relative
                leaf_kind: Literal["directory", "either"] = "directory" if not relative.parts else "either"
                filesystem._validate_projection_ancestry(declared, expected_kind=leaf_kind)
                filesystem._validate_projection_ancestry(projected, expected_kind=leaf_kind)
                return projected
        return None

    def _projected_path(self, path: Path, *, repo: Path, source: str, target: str) -> Path:
        self._route_repo(repo)
        declared = filesystem._projection_path(path)
        projected = self._project_explicit_route(declared, source, target)
        if projected is None:
            projected = self._project_public_subtree(declared, source, target)
        if projected is None:
            raise ArtifactVisibilityError("path is not owned by a registered artifact destination")
        return projected

    def durable_path_for(self, path: Path, *, repo: Path) -> Path:
        """Project one registered live write path to its durable destination."""
        return self._projected_path(path, repo=repo, source="write_path", target="requested")

    def live_path_for(self, path: Path, *, repo: Path) -> Path:
        """Project one registered durable destination to its live write path."""
        return self._projected_path(path, repo=repo, source="requested", target="write_path")

    def _require_active(self) -> None:
        if self._state is not _SessionState.ACTIVE:
            raise ArtifactVisibilityError("artifact session is frozen and no longer writable")

    def write_review_input(self, repo: Path, relative: Path, text: str) -> Path:
        """Atomically generate a host-named review input inside the owning session."""
        self._route_repo(repo)
        if relative.is_absolute() or not relative.parts or any(part in {".", ".."} for part in relative.parts):
            raise ArtifactVisibilityError("review input name is unsafe")
        path = self.daydream_dir / "deep" / "stage-inputs" / relative
        filesystem._validate_projection_ancestry(path, expected_kind="either")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        filesystem._atomic_bytes(path, text.encode("utf-8"))
        return path

    def _route_repo(self, repo: Path) -> None:
        self._require_active()
        declared = filesystem._absolute_lexical(repo)
        try:
            canonical, metadata = filesystem._declared_directory_metadata(declared, label="artifact repo")
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
        declared = filesystem._absolute_lexical(requested)
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
        if any(filesystem._overlaps(canonical, path) for path in protected):
            raise ArtifactVisibilityError("artifact destination overlaps private or Git storage")
        if self.layout.repo != self.layout.source and filesystem._overlaps(canonical, self.layout.repo):
            raise ArtifactVisibilityError("artifact destination overlaps the active model repository")
        owned_public_subtree = (
            public_subtree_owner is not None
            and label in _TRAJECTORY_LABELS
            and self.layout.public_daydream_dir in canonical.parents
        )
        if (
            label not in _PUBLIC_LABELS
            and (
                filesystem._overlaps(canonical, self.layout.public_daydream_dir)
                or canonical == self.layout.public_review_output
            )
            and not owned_public_subtree
        ):
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
            if filesystem._overlaps(canonical, prior_path):
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
            (destination for destination in self._destinations if self._is_public_daydream_owner(destination)),
            None,
        )

    def _private_public_trajectory_path(self, canonical: Path, *, owner: RoutedDestination) -> Path:
        if not any(owner is destination for destination in self._destinations):
            raise ArtifactVisibilityError("public trajectory owner identity mismatch")
        try:
            relative = canonical.relative_to(self.layout.public_daydream_dir)
        except ValueError as exc:
            raise ArtifactVisibilityError("public trajectory path is outside its owner") from exc
        filesystem._validate_relative_name(relative.as_posix())
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
        filesystem._validate_relative_name(relative)
        baseline = filesystem.manifest_tree(base, (relative,))
        root_entry = next((entry for entry in baseline if entry.path == relative), None)
        baseline_state: Literal["absent", "file", "directory"] = "absent" if root_entry is None else root_entry.kind
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
        filesystem._copy_tree(base, baseline_root, baseline)
        self._routed.append(_RoutedRecord(route, record))
        ledger._write_baseline_manifest(self._detach_transaction, index, record)
        self._persist_records()
        if inside_source and baseline:
            transfer._remove_manifested(
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
        ledger._write_destination_records(self._detach_transaction, self._records(), include_published=False)

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
        from daydream.trajectory import partial_document_path, run_directory, run_document_path

        self._require_active()
        if self._trajectory_route is not None:
            raise ArtifactVisibilityError("paired trajectory output is already registered")
        run_dir = run_directory(self.daydream_dir, self.layout.session_id)
        default_requested = run_document_path(run_directory(self.layout.public_daydream_dir, self.layout.session_id))
        declared_requested = default_requested if requested is None else filesystem._absolute_lexical(requested)
        try:
            canonical_requested = declared_requested.resolve(strict=False)
            canonical_default = default_requested.resolve(strict=False)
        except OSError as exc:
            raise ArtifactVisibilityError("artifact destination could not be resolved") from exc
        partial_requested = partial_document_path(declared_requested)
        private_full = run_document_path(run_dir)
        private_partial = partial_document_path(run_document_path(run_dir))
        public_root = self.layout.public_daydream_dir
        if canonical_requested == canonical_default:
            full, partial = self._paired_trajectory(
                declared_requested,
                partial_requested,
                private_full,
                private_partial,
                DestinationDelivery.DEFERRED,
            )
        else:
            owner = self._public_daydream_owner() if public_root in canonical_requested.parents else None
            if public_root in canonical_requested.parents and owner is None:
                self._validate_destination(declared_requested, label=OutputLabel.EXPLICIT_TRAJECTORY)
                raise AssertionError("unreachable public trajectory validation")
            full_declared, full_canonical, full_inside = self._validate_destination(
                declared_requested, label=OutputLabel.EXPLICIT_TRAJECTORY, public_subtree_owner=owner,
            )
            partial_declared, partial_canonical, partial_inside = self._validate_destination(
                partial_requested, label=OutputLabel.EXPLICIT_TRAJECTORY_PARTIAL,
                additional=(full_canonical,), public_subtree_owner=owner,
            )
            if full_inside is not partial_inside or owner is not None and not (
                public_root in full_canonical.parents and public_root in partial_canonical.parents
            ):
                raise ArtifactVisibilityError("paired trajectory destinations cross routing boundaries")
            if owner is not None:
                private_full = self._private_public_trajectory_path(full_canonical, owner=owner)
                private_partial = self._private_public_trajectory_path(partial_canonical, owner=owner)
            delivery = DestinationDelivery.DEFERRED if full_inside else DestinationDelivery.LIVE_EXTERNAL
            full, partial = self._paired_trajectory(
                full_declared, partial_declared, private_full, private_partial, delivery
            )
            if owner is None:
                self._destinations.extend((full, partial))
                initial = len(self._routed)
                try:
                    self._capture_destination_record(full, canonical=full_canonical, inside_source=full_inside)
                    self._capture_destination_record(partial, canonical=partial_canonical, inside_source=partial_inside)
                    if delivery is DestinationDelivery.LIVE_EXTERNAL:
                        if self._name_exchange is None:
                            self._name_exchange = external._name_exchange_factory()
                        parent = full_declared.parent
                        self._created_external_parents.extend(
                            external._ensure_external_parent(self._detach_transaction, parent)
                        )
                        self._external_capabilities[parent] = external._probe_external_parent(
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
            restore_stage = publication._create_source_stage(
                self.layout.source,
                self._detach_transaction.name,
                "restore",
                workspace_key=self.layout.workspace_key,
            )
            try:
                publication._restore_destination_records(
                    self.layout.source, self._detach_transaction, captured, restore_stage
                )
            finally:
                publication._retire_source_stage(
                    self.layout.source,
                    self._detach_transaction.name,
                    "restore",
                    workspace_key=self.layout.workspace_key,
                )
        for index in range(initial, initial + 2):
            baseline_root = self._detach_transaction / f"destination-{index:04d}-baseline"
            if baseline_root.exists() or baseline_root.is_symlink():
                filesystem._remove_owned_tree(baseline_root, self._detach_transaction)
            manifest = self._detach_transaction / f"destination-{index:04d}-baseline-manifest.json"
            if manifest.exists() and not manifest.is_symlink():
                manifest.unlink()
        del self._routed[initial:]
        self._persist_records()
        external._cleanup_external_directories(self._detach_transaction, self._created_external_parents)
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
            filesystem._validate_relative_name(relative.as_posix())
            private_path = document.path
        if private_path is None or type(document.json_bytes) is not bytes:
            raise ArtifactVisibilityError("trajectory document bytes are malformed")
        filesystem._atomic_bytes(private_path, document.json_bytes)
        if document.trajectory_id != self.layout.session_id:
            if status == "complete":
                self._completed_sibling_trajectories[document.trajectory_id] = document
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
        item.record = external._publish_live_external(
            self._detach_transaction,
            item.record,
            document.json_bytes,
            exchange=self._name_exchange,
            capability=capability,
        )
        self._persist_records()

    def snapshot_completed_sibling_trajectories(self, *, session_id: str) -> tuple[TrajectoryDocumentSnapshot, ...]:
        """Return retained complete direct-child trajectories for the active session."""
        self._require_active()
        if session_id != self.layout.session_id:
            raise ArtifactVisibilityError("trajectory session does not match active artifact session")
        route = self._trajectory_route
        if route is None:
            raise ArtifactVisibilityError("trajectory output is not registered")

        from daydream.trajectory import TrajectoryDocumentSnapshot, siblings_directory

        trajectories_dir = siblings_directory(route.run_dir)
        snapshots: list[TrajectoryDocumentSnapshot] = []
        for trajectory_id, document in self._completed_sibling_trajectories.items():
            if (
                type(trajectory_id) is not str
                or not trajectory_id
                or not isinstance(document, TrajectoryDocumentSnapshot)
                or document.trajectory_id != trajectory_id
                or trajectory_id == self.layout.session_id
                or not isinstance(document.path, Path)
                or type(document.json_bytes) is not bytes
            ):
                raise ArtifactVisibilityError("retained sibling trajectory is malformed")
            try:
                relative = document.path.relative_to(trajectories_dir)
            except ValueError as exc:
                raise ArtifactVisibilityError("retained sibling trajectory path is unsafe") from exc
            if len(relative.parts) != 1 or relative.suffix != ".json":
                raise ArtifactVisibilityError("retained sibling trajectory path is unsafe")
            filesystem._validate_relative_name(relative.as_posix())
            snapshots.append(document)
        return tuple(sorted(snapshots, key=lambda document: document.trajectory_id))

    def freeze(self, run_snapshot: RunWriteSnapshot) -> ArtifactTreeSnapshot:
        self._require_active()
        try:
            run_snapshot.validate(self.layout.session_id)
        except (ValueError, UnicodeError) as exc:
            raise ArtifactVisibilityError(f"run snapshot {exc}") from exc
        seen_paths: set[Path] = set()
        for document in run_snapshot.documents:
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
            filesystem._validate_relative_name(relative.as_posix())
            cursor = self.layout.live_root
            for part in relative.parts[:-1]:
                cursor /= part
                if cursor.exists() or cursor.is_symlink():
                    metadata = cursor.lstat()
                    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                        raise ArtifactVisibilityError("run snapshot path has unsafe ancestry")
            filesystem._atomic_bytes(self.layout.live_root / relative, document.json_bytes)
        return self._freeze_tree()

    def freeze_unproduced(self) -> ArtifactTreeSnapshot:
        """Freeze a supported run that exited before producing a recorder document."""
        self._require_active()
        if self._trajectory_route is None:
            raise ArtifactVisibilityError("unproduced run has no registered trajectory route")
        return self._freeze_tree()

    def _freeze_tree(self) -> ArtifactTreeSnapshot:
        entries = filesystem.manifest_tree(self.layout.live_root)
        frozen_root = self.layout.live_root.parent / "frozen"
        if frozen_root.exists() or frozen_root.is_symlink():
            raise ArtifactVisibilityError("frozen artifact tree already exists")
        filesystem._copy_tree(self.layout.live_root, frozen_root, entries)
        filesystem._atomic_json(frozen_root.parent / "frozen-manifest.json", filesystem._manifest_payload(entries))
        result = ArtifactTreeSnapshot(
            session_id=self.layout.session_id,
            workspace_key=self.layout.workspace_key,
            root=frozen_root,
            manifest=entries,
            destinations=tuple(self._destinations),
        )
        self._frozen_snapshot = result
        self._state = _SessionState.FROZEN
        return result

    def finalization_merge_path(self, route: RoutedDestination, *, snapshot: ArtifactTreeSnapshot) -> Path:
        if self._state is not _SessionState.FROZEN or snapshot is not self._frozen_snapshot:
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
                filesystem._remove_owned_tree(late, late_parent)
        if late_parent.exists() and not late_parent.is_symlink():
            with suppress(OSError):
                late_parent.rmdir()
                _fsync_directory(late_parent.parent)

    def finalize_frozen(self, snapshot: ArtifactTreeSnapshot, *, disposition: ArtifactDisposition) -> None:
        if self._state is not _SessionState.FROZEN:
            raise ArtifactVisibilityError("artifact session is not ready for frozen publication")
        if snapshot is not self._frozen_snapshot:
            raise ArtifactVisibilityError("frozen artifact snapshot identity mismatch")
        if not isinstance(disposition, ArtifactDisposition):
            raise ArtifactVisibilityError("artifact disposition is unsupported")
        if len(snapshot.destinations) != len(self._destinations) or any(
            actual is not expected for actual, expected in zip(snapshot.destinations, self._destinations, strict=True)
        ):
            raise ArtifactVisibilityError("frozen artifact destination identity mismatch")
        if filesystem.manifest_tree(snapshot.root) != snapshot.manifest:
            raise ArtifactVisibilityError("frozen artifact snapshot changed before publication")
        if disposition is ArtifactDisposition.ROLLBACK:
            self._restore_prior()
            self._retire_late_paths()
            self._state = _SessionState.PUBLISHED
            return
        public_entries = tuple(
            entry
            for entry in snapshot.manifest
            if entry.path == _DAYDREAM or entry.path.startswith(f"{_DAYDREAM}/") or entry.path == _REVIEW_OUTPUT
        )
        transaction_id = f"publish-{self.layout.session_id}-{secrets.token_hex(8)}"
        transaction = self.layout.state_root / "transactions" / transaction_id
        transaction.mkdir(parents=True)
        transactions._write_transaction_owner(
            transaction,
            workspace_key=self.layout.workspace_key,
            session_id=self.layout.session_id,
            kind="publish",
        )
        mark = partial(
            transactions._write_transition,
            transaction / "journal.json",
            transaction_id=transaction_id,
            session_id=self.layout.session_id,
        )
        publication_stage: Path | None = None
        try:
            filesystem._atomic_json(transaction / "publish-manifest.json", filesystem._manifest_payload(public_entries))
            publication_stage = publication._create_source_stage(
                self.layout.source,
                transaction_id,
                "publish",
                workspace_key=self.layout.workspace_key,
            )
            public_stage = publication_stage / "public"
            filesystem._copy_tree(snapshot.root, public_stage, public_entries)
            canonical_stage = transaction / "canonical-stage"
            filesystem._copy_tree(snapshot.root, canonical_stage, public_entries)
            publish_records: list[_DestinationRecord] = []
            for index, item in enumerate(self._routed):
                destination, baseline_record = item.route, item.record
                baseline_root = self._detach_transaction / f"destination-{index:04d}-baseline"
                publish_baseline = transaction / f"destination-{index:04d}-baseline"
                filesystem._copy_tree(baseline_root, publish_baseline, baseline_record.baseline)
                projection = publication_stage / f"destination-{index:04d}"
                filesystem._copy_tree(baseline_root, projection, baseline_record.baseline)
                if destination.delivery is DestinationDelivery.LIVE_EXTERNAL:
                    actual = filesystem.manifest_tree(Path(baseline_record.base), (baseline_record.relative,))
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
                    published = publication._overlay_destination(
                        snapshot.root, write_relative, projection, baseline_record
                    )
                elif item.late is None:
                    published = baseline_record.baseline
                else:
                    published = publication._overlay_destination(
                        item.late.parent, item.late.name, projection, baseline_record
                    )
                publish_records.append(replace(baseline_record, published=published))
            self._retire_late_paths()
            ledger._write_destination_records(
                transaction, publish_records, include_published=True, include_baseline=True
            )
            external._inherit_external_capability_proofs(self._detach_transaction, transaction)
            mark(state=_Transition.PUBLISH_STAGED)
        except BaseException:
            if publication_stage is not None:
                publication._retire_source_stage(
                    self.layout.source, transaction_id, "publish", workspace_key=self.layout.workspace_key
                )
            transactions._retire_transaction(
                self.layout.state_root, transaction, terminal_state=_TerminalState.PUBLISH_RECONCILED
            )
            raise
        self._state = _SessionState.PUBLISHING
        (transaction / "public-backup").mkdir()
        _fsync_directory(transaction)
        mark(state=_Transition.PUBLISH_BACKED_UP)
        publication._replace_public_from_tree(self.layout.source, public_stage, public_entries)
        for index, record in enumerate(publish_records):
            if record.delivery is DestinationDelivery.LIVE_EXTERNAL:
                continue
            publication._replace_destination_from_tree(
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
        filesystem._atomic_json(
            self.layout.state_root / "canonical-manifest.json", filesystem._manifest_payload(public_entries)
        )
        mark(state=_Transition.PUBLISH_VERIFIED)
        self._canonical_entries = public_entries
        transactions._retire_transaction(
            self.layout.state_root,
            self._detach_transaction,
            terminal_state=_TerminalState.DETACHED_RECONCILED,
        )
        publication._retire_source_stage(
            self.layout.source, transaction_id, "publish", workspace_key=self.layout.workspace_key
        )
        transactions._retire_transaction(
            self.layout.state_root, transaction, terminal_state=_TerminalState.PUBLISH_VERIFIED
        )
        self._state = _SessionState.PUBLISHED

    def _restore_prior(self) -> None:
        state_root = self.layout.state_root
        transaction = self._detach_transaction
        if not transaction.exists():
            # Already retired or already taken out of the replay path by an
            # earlier attempt (``finalize_frozen`` and session close both call
            # this). There is nothing left to restore from.
            return
        canonical = state_root / "canonical"
        publication._reset_source_stage(
            self.layout.source, transaction.name, "restore", workspace_key=self.layout.workspace_key
        )
        source_stage = publication._create_source_stage(
            self.layout.source,
            transaction.name,
            "restore",
            workspace_key=self.layout.workspace_key,
        )
        # Destination failures and public failures need opposite treatment, so
        # they are collected apart rather than into one list.
        destination_errors: list[Exception] = []
        public_errors: list[Exception] = []
        try:
            publication._restore_destination_records(self.layout.source, transaction, self._records(), source_stage)
        except Exception as exc:
            destination_errors.append(exc)
        try:
            projection = source_stage / "public"
            filesystem._copy_tree(canonical, projection, self._canonical_entries)
            publication._replace_public_from_tree(self.layout.source, projection, self._canonical_entries)
        except Exception as exc:
            public_errors.append(exc)
        try:
            publication._retire_source_stage(
                self.layout.source, transaction.name, "restore", workspace_key=self.layout.workspace_key
            )
        except Exception as exc:
            destination_errors.append(exc)
        try:
            external._cleanup_external_directories(transaction, self._created_external_parents)
            self._created_external_parents.clear()
        except Exception as exc:
            destination_errors.append(exc)
        if public_errors:
            # The journal has to stay: the next session open replays it and
            # reinstalls the public tree from canonical before anything can
            # adopt the missing tree as a new baseline. Name it instead.
            raise transactions._transaction_context(public_errors[0], state_root, transaction)
        try:
            self._retire_late_paths()
        except Exception as exc:
            destination_errors.append(exc)
        if destination_errors:
            raise transactions._clear_failed_restore(state_root, self.layout.source, transaction, destination_errors[0])
        transactions._retire_transaction(state_root, transaction, terminal_state=_TerminalState.DETACHED_RECONCILED)

    def _close(self) -> None:
        self._state = _SessionState.CLOSED
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        os.close(self._repo_fd)


def _routing_session(
    repo: Path,
    session: ArtifactSession | None,
    *,
    allow_standalone: bool,
) -> ArtifactSession | None:
    """Admit one session under the shared strict or standalone routing policy."""
    if session is None:
        if not allow_standalone:
            raise ArtifactVisibilityError("an explicit artifact session is required for strict artifact routing")
        session = _SESSION.get()
    if session is not None:
        session._route_repo(repo)
    return session


def artifact_dir_for(
    repo: Path,
    *,
    session: ArtifactSession | None = None,
    allow_standalone: bool = False,
) -> Path:
    """Routed ``.daydream`` path for *repo*.

    Production callers pass an explicit session. Intentional standalone and
    legacy extension callers must affirmatively allow compatibility routing;
    only that path may consult the bound session before falling back to the
    public ``repo/.daydream`` location.
    """
    session = _routing_session(repo, session, allow_standalone=allow_standalone)
    if session is not None:
        return session.daydream_dir
    return repo / _DAYDREAM


def artifact_session_active() -> bool:
    """Whether the current task is bound to one live artifact session."""
    return _SESSION.get() is not None


def review_output_path_for(
    repo: Path,
    *,
    session: ArtifactSession | None = None,
    allow_standalone: bool = False,
) -> Path:
    """Routed ``.review-output.md`` path for *repo* (see :func:`artifact_dir_for`)."""
    session = _routing_session(repo, session, allow_standalone=allow_standalone)
    if session is not None:
        return session.review_output
    return repo / _REVIEW_OUTPUT


def assert_model_cwd_clean(cwd: Path) -> None:
    declared = filesystem._declared_directory(cwd, label="model cwd")
    session = _SESSION.get()
    if session is not None:
        for destination in session._destinations:
            if destination.delivery is DestinationDelivery.LIVE_EXTERNAL:
                requested = destination.requested.resolve(strict=False)
                if filesystem._overlaps(declared, requested):
                    raise ArtifactVisibilityError("model cwd overlaps a live external artifact destination")
    for name in (_DAYDREAM, _REVIEW_OUTPUT):
        candidate = declared / name
        if candidate.exists() or candidate.is_symlink():
            raise ArtifactVisibilityError("model cwd contains generated Daydream artifacts")


def _rebaseline_canonical_from_public(
    state_root: Path,
    source: Path,
    public_entries: tuple[ArtifactManifestEntry, ...],
    *,
    transaction: Path,
) -> None:
    """Adopt between-run public changes as the canonical recovery baseline and warn.
    Session opening holds the exclusive lock with no transaction in flight; mid-run
    divergence still fails through conflict recovery.
    """
    canonical = state_root / "canonical"
    canonical_manifest = state_root / "canonical-manifest.json"
    backup = transaction / "rebaseline-old-canonical"
    stage = transaction / "rebaseline-stage"
    if canonical.exists():
        os.replace(canonical, backup)
    try:
        filesystem._copy_tree(source, stage, public_entries)
        if canonical.exists() or canonical.is_symlink():
            filesystem._remove_owned_tree(canonical, state_root)
        os.replace(stage, canonical)
        _fsync_directory(state_root)
        filesystem._atomic_json(canonical_manifest, filesystem._manifest_payload(public_entries))
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
    from daydream.trajectory import RUNS_DIRNAME

    if not session_id or "\0" in session_id or "/" in session_id or "\\" in session_id or session_id in (".", ".."):
        raise ArtifactVisibilityError("artifact session id is invalid")
    identity = ownership.derive_workspace_identity(work, owner=owner)
    source = identity.source
    workspace_key = identity.workspace_key
    state_root = identity.state_root
    with ExitStack() as held:
        repo_fd = filesystem._open_directory_descriptor(identity.repo, label="artifact repo")
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
            transactions._recover_transactions(state_root, source)
            validated_canonical_entries = publication._validated_canonical_entries(state_root)
            validated_public_entries = publication._validate_public_tree(
                source,
                canonical_entries=validated_canonical_entries,
            )
            runs = state_root / RUNS_DIRNAME
            transactions_root = state_root / "transactions"
            filesystem._create_private_directory(runs)
            filesystem._create_private_directory(transactions_root)
            run_root = runs / session_id
            if run_root.exists() or run_root.is_symlink():
                raise ArtifactVisibilityError("artifact session id already exists")
            run_root.mkdir(mode=0o700)

            public_entries = filesystem.manifest_tree(source, (_DAYDREAM, _REVIEW_OUTPUT))
            if public_entries != validated_public_entries:
                raise ArtifactVisibilityError("public artifacts changed during session open")
            transaction_id = f"detach-{session_id}-{secrets.token_hex(8)}"
            transaction = transactions_root / transaction_id
            transaction.mkdir(mode=0o700)
            transactions._write_transaction_owner(
                transaction, workspace_key=workspace_key, session_id=session_id, kind="detach"
            )
            mark = partial(
                transactions._write_transition,
                transaction / "journal.json",
                transaction_id=transaction_id,
                session_id=session_id,
            )
            detach_stage = transaction / "detach-stage"
            canonical = state_root / "canonical"
            canonical_manifest = state_root / "canonical-manifest.json"
            # Canonical is itself the durable copy the DETACH_REMOVING ordering
            # needs, so stage one only when this workspace has none yet. Session
            # open is the one safe reconciliation point for benign between-run
            # public drift: the lock excludes concurrent sessions, so adopt it as
            # the canonical baseline BEFORE staging (doing so after would leave a
            # stale stage on a crash).
            canonical_present = canonical.exists()
            if canonical_present:
                canonical_entries = filesystem._parse_manifest(canonical_manifest)
                if filesystem.manifest_tree(canonical) != canonical_entries:
                    raise ArtifactVisibilityError("canonical artifact recovery copy is corrupt")
                if public_entries != canonical_entries:
                    _rebaseline_canonical_from_public(state_root, source, public_entries, transaction=transaction)
                    canonical_entries = public_entries
            else:
                filesystem._copy_tree(source, detach_stage, public_entries)
            filesystem._atomic_json(transaction / "manifest.json", filesystem._manifest_payload(public_entries))
            mark(state=_Transition.DETACH_STAGED)
            if canonical_present:
                # Re-baselining already made canonical match the observed public
                # state; keep the equality the recovery path relies on.
                canonical_entries = filesystem._parse_manifest(canonical_manifest)
                if public_entries != canonical_entries:
                    raise ArtifactVisibilityError("canonical artifact recovery copy is inconsistent")
            else:
                os.replace(detach_stage, canonical)
                _fsync_directory(state_root)
                filesystem._atomic_json(canonical_manifest, filesystem._manifest_payload(public_entries))
                canonical_entries = public_entries
            mark(state=_Transition.DETACH_CANONICAL)
            mark(state=_Transition.DETACH_REMOVING)
            if public_entries:
                transfer._remove_manifested(
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
            filesystem._copy_tree(canonical, layout.live_root, canonical_entries)
        except BaseException as primary:
            if transaction is not None and transaction.exists() and not transaction.is_symlink():
                try:
                    transactions._reconcile_failed_detach(state_root, source, transaction)
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
            transactions._recover_transactions(session.layout.state_root, session.layout.source)
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
