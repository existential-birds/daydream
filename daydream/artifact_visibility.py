"""Run-scoped storage boundary for model-invisible Daydream artifacts.

The source checkout exposes compatibility paths only between runs.  While a
session is bound, generated state lives under an owner-validated host runtime
root and callers may address it only through the exact workspace identity."""

from __future__ import annotations

import fcntl
import os
import secrets
import shutil
import stat
from contextlib import ExitStack, asynccontextmanager, suppress
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import anyio

from daydream import git_ops
from daydream.json_utils import _fsync_directory

if TYPE_CHECKING:
    from daydream.workspace import WorkContext


from daydream.artifacts import (
    filesystem,
    ownership,
    publication,
    transactions,
    transfer,
)
from daydream.artifacts.filesystem import (
    manifest_tree as manifest_tree,
    validate_private_directory as validate_private_directory,
)
from daydream.artifacts.models import (
    _DAYDREAM,
    _REVIEW_OUTPUT,
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
    _SessionState as _SessionState,
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
from daydream.artifacts.session import ArtifactSession as ArtifactSession

_SESSION: ContextVar[ArtifactSession | None] = ContextVar("daydream_artifact_session", default=None)




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
