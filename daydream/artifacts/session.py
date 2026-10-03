"""Live artifact routing under one held workspace lease."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from daydream import git_ops
from daydream.json_utils import _fsync_directory

if TYPE_CHECKING:
    from daydream.trajectory import RunWriteSnapshot, TrajectoryDocumentSnapshot


from daydream.artifacts import (
    external,
    filesystem,
    finalization,
    ledger,
    publication,
    transfer,
)
from daydream.artifacts.external import _AtomicNameExchange
from daydream.artifacts.models import (
    _PUBLIC_LABELS,
    _TRAJECTORY_LABELS,
    ArtifactDisposition,
    ArtifactEvidenceProvenance,
    ArtifactLayout,
    ArtifactManifestEntry,
    ArtifactTreeSnapshot,
    ArtifactVisibilityError,
    DestinationDelivery,
    OutputLabel,
    RoutedDestination,
    TrajectoryOutputRoute,
    _DestinationRecord,
    _RoutedRecord,
    _SessionState,
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
        return finalization.freeze(self, run_snapshot)

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
        finalization.finalize_frozen(self, snapshot, disposition=disposition)

    def _restore_prior(self) -> None:
        finalization.restore_prior(self)

    def _close(self) -> None:
        self._state = _SessionState.CLOSED
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        os.close(self._repo_fd)
