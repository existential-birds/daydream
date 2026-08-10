"""Fail-closed service-mode review worker.

Runs one immutable :class:`~daydream.service.models.ReviewJobV1` against a
detached checkout through a strictly read-only backend turn and returns a
strictly-passive :class:`~daydream.service.artifact.WorkerArtifactV1`. This is
the leaf that turns *process* outcome into *artifact* outcome; it owns no
executor identity, no lease, and no controller state (those live in other
Plan-008 leaves).

Fail-closed contract (every invariant is a hard gate, never a soft warning):

1. **Pre-flight**: the checkout must be detached at the job's exact candidate
   SHA, carry the exact candidate tree digest, be pristine (no
   staged/unstaged/untracked drift), and its full ``base_sha..HEAD`` diff must
   digest-match ``job.target.full_diff_digest``. Any mismatch is
   ``git_preflight_failed``.
2. **Lens inventory**: every required lens must be present in the inventory
   BEFORE dispatch; a gap is ``lens_unavailable`` and the backend never runs.
3. **Read-only run**: the service-mode phase in :mod:`daydream.phases` runs
   ``run_agent(read_only=True)``. A budget/supervisor abort is ``infra_error``
   (``budget_exhausted`` / ``tool_vetoed``), never ``clean``.
4. **Every lens must complete**: only lenses the agent declared completed with
   validated output count; any missing after dispatch is ``incomplete_lenses``
   (``infra_error``) unless the turn was cancelled.
5. **Mutation check**: the git head/tree/index/tracked/untracked state must be
   byte-identical before and after the turn; any change is
   ``mutation_detected`` — the worker-side proof the read-only run stayed
   read-only.
6. **Findings**: blocking findings (high/medium severity) keep
   ``terminal="findings"`` even when the process exited 0; a missing-free run
   with no blocking findings is ``terminal="clean"``.
7. **Process/UI/parse loss**: any mid-stream death (backend raise, unparseable
   output) is ``infra_error``, never ``clean``.

Git snapshot note: :mod:`daydream.git_ops` already exposes ``head_sha``,
``current_branch``, and ``status_porcelain``; it does NOT expose a
tree/index/untracked-set snapshot primitive, so this module shells ``git``
directly for those three digests (single point of contact exception noted for
the integrator) and reuses git_ops for everything else.

The agent call is deliberately NOT made here: it lives in the service-mode
phase entry (:func:`daydream.phases.phase_service_review`), so this module
never imports ``daydream.agent`` directly (the phase already holds that seam).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.backends import Backend
from daydream.phases import phase_service_review
from daydream.service.artifact import MAX_FINDINGS, WorkerArtifactV1
from daydream.service.models import ReviewJobV1

_logger = logging.getLogger(__name__)


class ServiceGitError(Exception):
    """A local ``git`` command required by the service worker failed."""


@dataclass(frozen=True)
class GitSnapshot:
    """Five git-identity surfaces of a checkout, captured atomically enough for
    change detection: HEAD SHA, tree digest, index digest, tracked listing
    digest, untracked listing digest."""

    head_sha: str
    tree_digest: str
    index_digest: str
    tracked_digest: str
    untracked_digest: str

    def digest(self) -> str:
        """A stable content hash of the whole snapshot."""
        payload = "\n".join(
            (
                self.head_sha,
                self.tree_digest,
                self.index_digest,
                self.tracked_digest,
                self.untracked_digest,
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def _git_capture(repo: Path, args: list[str]) -> str:
    """Run a read-only ``git`` command and return stdout, raising on failure.

    Local helper for the tree/index/untracked digests ``git_ops`` does not
    expose (see module docstring). Never user-controlled arguments.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - args are module-local constants
            ["git", *args],  # noqa: S607 - git is a trusted command
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise ServiceGitError(
            f"git {' '.join(args)} failed in {repo}: {type(exc).__name__}: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise ServiceGitError(f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout


def capture_git_state(repo: Path) -> GitSnapshot:
    """Snapshot the five git-identity surfaces of *repo*.

    * ``head_sha``: ``HEAD`` (via :func:`daydream.git_ops.head_sha`).
    * ``tree_digest``: ``HEAD^{tree}`` (the committed tree).
    * ``index_digest``: SHA-256 of the sorted ``git ls-files -s`` listing
      (staging state: modes + object names + paths).
    * ``tracked_digest``: SHA-256 of ``git diff HEAD`` (every tracked working
      tree change, staged or unstaged, relative to ``HEAD``).
    * ``untracked_digest``: SHA-256 of the ``git ls-files --others
      --exclude-standard`` listing (the untracked file set).

    Raises:
        ServiceGitError: If any git command fails (including *repo* not being a
            repository).
    """
    try:
        head = git_ops.head_sha(repo)
        tree = _git_capture(repo, ["rev-parse", "HEAD^{tree}"]).strip()
        index = hashlib.sha256(_git_capture(repo, ["ls-files", "-s"]).encode()).hexdigest()
        tracked = hashlib.sha256(_git_capture(repo, ["diff", "HEAD"]).encode()).hexdigest()
        untracked = hashlib.sha256(
            _git_capture(repo, ["ls-files", "--others", "--exclude-standard"]).encode()
        ).hexdigest()
    except (git_ops.GitError, ServiceGitError) as exc:
        raise ServiceGitError(
            f"could not snapshot git state in {repo}: {type(exc).__name__}: {exc}"
        ) from exc
    return GitSnapshot(
        head_sha=head,
        tree_digest=tree,
        index_digest=index,
        tracked_digest=tracked,
        untracked_digest=untracked,
    )


def assert_unchanged(before: GitSnapshot, after: GitSnapshot) -> bool:
    """Return True iff all five snapshot surfaces are identical."""
    return before == after


def changed_surfaces(before: GitSnapshot, after: GitSnapshot) -> tuple[str, ...]:
    """Return the names of the snapshot surfaces that differ (for diagnostics)."""
    changed = [
        name
        for name, attr in (
            ("head", "head_sha"),
            ("tree", "tree_digest"),
            ("index", "index_digest"),
            ("tracked", "tracked_digest"),
            ("untracked", "untracked_digest"),
        )
        if getattr(before, attr) != getattr(after, attr)
    ]
    return tuple(changed)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _completed_from(required: Sequence[str], declared: Sequence[str]) -> tuple[str, ...]:
    declared_set = set(declared)
    return tuple(name for name in required if name in declared_set)


def _missing_from_completed(required: Sequence[str], declared: Sequence[str]) -> tuple[str, ...]:
    declared_set = set(declared)
    return tuple(name for name in required if name not in declared_set)


def _verify_preflight(repo: Path, job: ReviewJobV1) -> str | None:
    """Return a fail-closed process outcome when the checkout does not match the
    job, or ``None`` when every pre-flight invariant holds.

    Checks in order: detached at the exact candidate SHA, exact candidate tree
    digest, pristine working tree, then an exact non-empty full-diff digest.
    """
    try:
        branch = git_ops.current_branch(repo)
        if branch is not None:
            _logger.warning("service pre-flight: checkout is on branch %r, not detached", branch)
            return "git_preflight_failed"
        head = git_ops.head_sha(repo)
        if head != job.target.candidate_sha:
            _logger.warning(
                "service pre-flight: HEAD %s != candidate %s", head, job.target.candidate_sha
            )
            return "git_preflight_failed"
        tree = _git_capture(repo, ["rev-parse", "HEAD^{tree}"]).strip()
        if tree != job.target.candidate_tree_digest:
            _logger.warning(
                "service pre-flight: tree %s != candidate tree %s",
                tree,
                job.target.candidate_tree_digest,
            )
            return "git_preflight_failed"
        if git_ops.status_porcelain(repo).strip():
            _logger.warning("service pre-flight: working tree is not pristine")
            return "git_preflight_failed"
        diff_text = _git_capture(repo, ["diff", "--no-ext-diff", f"{job.target.base_sha}..HEAD"])
        if not diff_text.strip():
            _logger.warning("service pre-flight: full diff %s..HEAD is empty", job.target.base_sha)
            return "git_preflight_failed"
        digest = hashlib.sha256(diff_text.encode()).hexdigest()
        if digest != job.target.full_diff_digest:
            _logger.warning("service pre-flight: full-diff digest mismatch")
            return "git_preflight_failed"
    except (ServiceGitError, git_ops.GitError) as exc:
        _logger.warning("service pre-flight: git verification failed: %s", exc)
        return "git_preflight_failed"
    return None


def _infra(
    job: ReviewJobV1,
    *,
    process_outcome: str,
    completed_lenses: tuple[str, ...],
    missing_lenses: tuple[str, ...],
    findings: tuple[dict[str, Any], ...] = (),
    hashes: dict[str, str] | None = None,
    timestamps: dict[str, str] | None = None,
) -> WorkerArtifactV1:
    return WorkerArtifactV1.infra_error(
        job,
        process_outcome=process_outcome,
        completed_lenses=completed_lenses,
        missing_lenses=missing_lenses,
        findings=findings,
        hashes=hashes,
        timestamps=timestamps,
    )


async def run_service_review(
    target_repo: Path,
    job: ReviewJobV1,
    backend: Backend,
    *,
    lens_inventory: Sequence[str],
) -> WorkerArtifactV1:
    """Run the fail-closed service review for *job* and return the passive artifact.

    Args:
        target_repo: Detached checkout at the job's exact candidate SHA (the
            worker re-verifies this; see the fail-closed contract).
        job: The immutable job to run.
        backend: The backend that will execute the read-only review turn.
        lens_inventory: Every lens that can be dispatched. Required lenses not
            present here fail before any dispatch.

    Returns:
        A :class:`WorkerArtifactV1`; ``terminal`` is ``"clean"``/``"findings"``
        for complete turns and ``"infra_error"``/``"cancelled"`` otherwise.
    """
    started = _now_iso()

    def finished() -> dict[str, str]:
        return {"started_at": started, "finished_at": _now_iso()}

    preflight = _verify_preflight(target_repo, job)
    if preflight is not None:
        return _infra(
            job,
            process_outcome=preflight,
            completed_lenses=(),
            missing_lenses=job.required_lenses,
            timestamps=finished(),
        )

    try:
        before = capture_git_state(target_repo)
    except (ServiceGitError, git_ops.GitError) as exc:
        _logger.warning("service: pre-turn git snapshot failed: %s", exc)
        return _infra(
            job,
            process_outcome="state_capture_failed",
            completed_lenses=(),
            missing_lenses=job.required_lenses,
            timestamps=finished(),
        )

    try:
        outcome = await phase_service_review(
            backend, target_repo, job, lens_inventory=lens_inventory
        )
    except asyncio.CancelledError:
        return WorkerArtifactV1.cancelled(
            job,
            completed_lenses=(),
            missing_lenses=job.required_lenses,
            hashes={"before_state": before.digest()},
            timestamps=finished(),
        )
    except Exception as exc:  # noqa: BLE001 - any worker/phase machinery failure is process_loss
        _logger.warning("service: review phase raised %s: %s", type(exc).__name__, exc)
        return _infra(
            job,
            process_outcome="process_loss",
            completed_lenses=(),
            missing_lenses=job.required_lenses,
            timestamps=finished(),
        )

    if outcome.inventory_missing:
        return _infra(
            job,
            process_outcome="lens_unavailable",
            completed_lenses=(),
            missing_lenses=outcome.inventory_missing,
            timestamps=finished(),
        )
    if outcome.aborted_reason is not None:
        return _infra(
            job,
            process_outcome=outcome.abort_process_outcome or "process_loss",
            completed_lenses=_completed_from(job.required_lenses, outcome.completed_lenses),
            missing_lenses=_missing_from_completed(job.required_lenses, outcome.completed_lenses),
            timestamps=finished(),
        )
    if outcome.process_error is not None:
        _, category = outcome.process_error
        process_outcome = "exited_nonzero" if category == "PROCESS_EXIT" else "process_loss"
        return _infra(
            job,
            process_outcome=process_outcome,
            completed_lenses=(),
            missing_lenses=job.required_lenses,
            timestamps=finished(),
        )
    if not outcome.parse_ok:
        _logger.warning("service: review output failed to parse: %r", outcome.raw_output)
        return _infra(
            job,
            process_outcome="parse_loss",
            completed_lenses=(),
            missing_lenses=job.required_lenses,
            timestamps=finished(),
        )

    completed = _completed_from(job.required_lenses, outcome.completed_lenses)
    missing = _missing_from_completed(job.required_lenses, outcome.completed_lenses)
    if missing:
        return _infra(
            job,
            process_outcome="incomplete_lenses",
            completed_lenses=completed,
            missing_lenses=missing,
            timestamps=finished(),
        )

    try:
        after = capture_git_state(target_repo)
    except (ServiceGitError, git_ops.GitError) as exc:
        _logger.warning("service: post-turn git snapshot failed: %s", exc)
        return _infra(
            job,
            process_outcome="state_capture_failed",
            completed_lenses=completed,
            missing_lenses=missing,
            timestamps=finished(),
        )
    if not assert_unchanged(before, after):
        _logger.warning(
            "service: mutation detected on surfaces: %s", changed_surfaces(before, after)
        )
        return _infra(
            job,
            process_outcome="mutation_detected",
            completed_lenses=completed,
            missing_lenses=missing,
            hashes={"before_state": before.digest(), "after_state": after.digest()},
            timestamps=finished(),
        )

    if len(outcome.issues) > MAX_FINDINGS:
        return _infra(
            job,
            process_outcome="findings_overflow",
            completed_lenses=completed,
            missing_lenses=missing,
            timestamps=finished(),
        )

    return WorkerArtifactV1.complete(
        job,
        completed_lenses=completed,
        findings=outcome.issues,
        hashes={
            "full_diff": job.target.full_diff_digest,
            "candidate_tree": job.target.candidate_tree_digest,
        },
        timestamps=finished(),
    )


def terminal_exit_code(artifact: WorkerArtifactV1) -> int:
    """Map an artifact terminal onto a process exit code.

    ``clean``/``findings`` are successful review outcomes (0); ``infra_error``
    is a hard failure (1); ``cancelled`` is a distinct non-zero code (2) so a
    controller can tell "cancelled" from "broken".
    """
    if artifact.terminal in ("clean", "findings"):
        return 0
    if artifact.terminal == "cancelled":
        return 2
    return 1
