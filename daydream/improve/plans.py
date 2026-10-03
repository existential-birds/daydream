"""Incremental plan landing, deterministic reservations, and HEAD-drift recovery."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.artifact_visibility import (
    PrivateWorkspaceOwner,
    operational_worktree_root,
    private_root_locations,
    resolve_private_workspace_owner,
    validate_private_workspace_owner,
)
from daydream.improve import plan_index
from daydream.improve.assemble import AdmittedPlan
from daydream.improve.plan_diagnostics import (
    _attempt_diagnostic,
    _validation_stage,
)
from daydream.improve.reanchor import (
    _REANCHOR_DIR_SUFFIX,
    _SAFE_DIRNAME,
)
from daydream.improve.redaction import redact_model_value
from daydream.improve.render import (
    plan_slug,
    render_plan,
)
from daydream.workspace import reject_public_operational_storage


@dataclass(frozen=True)
class PlanReservation:
    """A number reserved in selection order, independent of writer completion.

    Already planned or rejected findings consume no number.
    """

    index: int
    fingerprint: str
    number: int | None
    existing: plan_index.PlanIndexEntry | None = None
    existing_path: str | None = None


@dataclass(frozen=True)
class PlanOutcome:
    """What a single :meth:`PlanWriteSession.commit` did on disk."""

    status: str
    number: int | None
    path: str | None
    title: str


class PlanWriteSession:
    """Land each completed plan and reconcile its durable index immediately.

    Reserve numbers in selection order, reusing fileless host-blocked attempts.
    The index module recovers durable state and operator-edited README statuses.
    Synchronous commits have no await point, so concurrent async writers need
    no lock around this session's entries.
    """

    def __init__(
        self,
        plans_dir: Path,
        *,
        planned_at: str,
        non_interactive_default: bool = False,
        run_session_id: str | None = None,
        private_workspace_owner: PrivateWorkspaceOwner | None = None,
    ) -> None:
        self._plans_dir = plans_dir
        self._repo = plans_dir.parent
        owner = private_workspace_owner or resolve_private_workspace_owner(
            self._repo, locations=private_root_locations(),
        )
        validate_private_workspace_owner(owner, source=owner.source, repo=self._repo)
        reject_public_operational_storage(owner.source)
        self._worktrees_root = operational_worktree_root(owner)
        self._planned_at = planned_at
        self._planned_on = date.today()
        self._run_session_id = run_session_id
        plans_dir.mkdir(parents=True, exist_ok=True)
        self._non_interactive_default = (
            non_interactive_default or "non-interactive default" in plan_index._index_text(plans_dir).lower()
        )
        self._entries = plan_index._merged_index(plans_dir)
        self._rejected = plan_index.load_rejections(plans_dir)
        self._next_number = plan_index._highest_plan_number(plans_dir, self._entries.values()) + 1
        self._reserved_count = 0
        self._written: list[tuple[int, dict[str, Any]]] = []
        self._skipped: list[tuple[int, dict[str, Any]]] = []
        self._failed: list[tuple[int, dict[str, Any]]] = []
        self._diagnostics: list[tuple[int, dict[str, Any]]] = []
        self._reanchored: dict[int, plan_index.PlanIndexEntry] = {}
        self._reanchor_worktree: Path | None = None
        self._planned_at_errors: tuple[str, ...] = ()
        try:
            if not git_ops.commit_exists(self._repo, planned_at):
                self._planned_at_errors = ("PLANNED_AT_INVALID",)
            elif not git_ops.is_ancestor(self._repo, planned_at, "HEAD"):
                self._planned_at_errors = ("PLANNED_AT_NOT_ANCESTOR",)
        except git_ops.GitError:
            self._planned_at_errors = ("PLANNED_AT_CHECK_FAILED",)

    def reserve(
        self, findings: Sequence[dict[str, Any] | None]
    ) -> list[PlanReservation]:
        """Claim one plan number per finding, in the order given."""
        reservations: list[PlanReservation] = []
        for finding in findings:
            index = self._reserved_count
            self._reserved_count += 1
            if not isinstance(finding, dict):
                reservations.append(PlanReservation(index=index, fingerprint="", number=None))
                continue
            fingerprint = plan_index._finding_package_fingerprint(finding)
            identities = plan_index._finding_member_identities(finding)
            candidates: list[plan_index.PlanIndexEntry] = []
            reserved_numbers: list[int] = []
            durable_coverage = set(self._rejected)
            for entry in self._entries.values():
                coverage = plan_index._entry_member_coverage(entry)
                covered = plan_index._fully_covered(identities, coverage)
                if plan_index._is_retryable(self._plans_dir, entry):
                    if covered:
                        reserved_numbers.append(entry.number)
                else:
                    durable_coverage.update(coverage)
                    if covered:
                        candidates.append(entry)
            if len(candidates) == 1:
                existing = candidates[0]
                reservations.append(
                    PlanReservation(
                        index=index,
                        fingerprint=fingerprint,
                        number=None,
                        existing=existing,
                        existing_path=(
                            existing.path
                            if existing.path is not None and plan_index._has_plan_file(self._plans_dir, existing)
                            else None
                        ),
                    )
                )
                continue
            if plan_index._fully_covered(identities, durable_coverage):
                reservations.append(PlanReservation(index=index, fingerprint=fingerprint, number=None))
                continue
            for reserved in reserved_numbers:
                del self._entries[reserved]
            if reserved_numbers:
                number = min(reserved_numbers)
            else:
                number = self._next_number
                self._next_number += 1
            reservations.append(PlanReservation(index=index, fingerprint=fingerprint, number=number))
        return reservations

    def commit(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
    ) -> PlanOutcome:
        """Land one plan-writer result, writing its file when it is complete."""
        safe = redact_model_value(selection)
        if not isinstance(safe, dict):
            return PlanOutcome("ignored", None, None, "")
        finding = safe.get("finding")
        if not isinstance(finding, dict):
            return PlanOutcome("ignored", None, None, "")
        title = str(finding.get("title") or "Selected finding")
        if reservation.number is None:
            skipped: dict[str, Any] = {"finding": finding}
            existing = reservation.existing
            if existing is not None:
                skipped.update(
                    number=existing.number,
                    package_fingerprint=existing.package_fingerprint,
                    member_fingerprints=list(existing.member_fingerprints),
                    member_aliases=list(existing.member_aliases),
                )
                if reservation.existing_path is not None:
                    skipped["path"] = reservation.existing_path
            self._skipped.append((reservation.index, skipped))
            attempt = self._attempt_of(safe)
            if attempt is not None:
                self._diagnostics.append(
                    (
                        reservation.index,
                        _attempt_diagnostic(
                            finding=finding,
                            attempt=attempt,
                            received=_plan_payload(safe),
                            disposition="skipped",
                            stage="reconciliation",
                            errors=("ALREADY_PLANNED_OR_REJECTED",),
                        ),
                    )
                )
            return PlanOutcome(
                "skipped",
                existing.number if existing is not None else None,
                reservation.existing_path,
                title,
            )
        return self._land(reservation, safe)

    def finish(self) -> dict[str, list[dict[str, Any]]]:
        """Reconcile the index and return what this session landed."""
        self._write_index()
        self._release_reanchor_worktree()
        return {
            "written": _by_reservation(self._written),
            "skipped": _by_reservation(self._skipped),
            "failed": _by_reservation(self._failed),
            "diagnostics": _by_reservation(self._diagnostics),
        }

    def _release_reanchor_worktree(self) -> None:
        """Best-effort unlock, clearing the reference even if Git fails.

        Keep successful worktrees until pruning; failed re-anchors remove them so
        a later finding can reuse the path.
        """
        if self._reanchor_worktree is None:
            return
        try:
            git_ops.worktree_unlock(self._repo, self._reanchor_worktree)
        except git_ops.GitError:
            pass
        self._reanchor_worktree = None

    @staticmethod
    def _attempt_of(selection: dict[str, Any]) -> dict[str, Any] | None:
        attempt = selection.get("_attempt")
        return attempt if isinstance(attempt, dict) else None

    def _block(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
        *,
        stage: str,
        errors: Sequence[str],
        received: Any,
    ) -> PlanOutcome:
        assert reservation.number is not None
        number = reservation.number
        finding = selection["finding"]
        codes = [error.partition("@")[0] for error in errors]
        writer_failed = stage == "transport"
        failure = "PLAN_WRITER_FAILED" if writer_failed else "PLAN_VALIDATION_FAILED"
        detail = codes[0] if writer_failed else ",".join(codes)
        self._entries[number] = plan_index._blocked_entry(
            number=number,
            fingerprint=reservation.fingerprint,
            finding=finding,
            status=f"BLOCKED ({failure}: {detail})",
            planned_at=self._planned_at,
        )
        self._failed.append((reservation.index, finding))
        self._diagnostics.append(
            (
                reservation.index,
                _attempt_diagnostic(
                    finding=finding,
                    attempt=self._attempt_of(selection),
                    received=received,
                    disposition="blocked",
                    stage=stage,
                    errors=errors,
                ),
            )
        )
        self._write_index()
        return PlanOutcome(
            "blocked",
            number,
            None,
            str(finding.get("title") or "Selected finding"),
        )

    def _record_written(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
        *,
        path: str,
        artifact: dict[str, Any],
    ) -> PlanOutcome:
        """Record a successfully landed plan and return its outcome."""
        number = reservation.number
        finding = selection["finding"]
        self._written.append(
            (
                reservation.index,
                {
                    "finding": finding,
                    "number": number,
                    "path": path,
                    "title": selection["plan"].authored["title"],
                },
            )
        )
        self._diagnostics.append(
            (
                reservation.index,
                _attempt_diagnostic(
                    finding=finding,
                    attempt=self._attempt_of(selection),
                    received=_plan_payload(selection),
                    disposition="success",
                    stage="success",
                    artifact=artifact,
                ),
            )
        )
        return PlanOutcome("written", number, path, str(finding.get("title") or "Selected finding"))

    def _land(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
    ) -> PlanOutcome:
        finding = selection["finding"]
        assert reservation.number is not None  # commit() gates on the number
        number = reservation.number
        title = str(finding.get("title") or "Selected finding")
        attempt = self._attempt_of(selection)
        plan = selection.get("plan")
        plan_title = plan.authored["title"] if isinstance(plan, AdmittedPlan) else None
        slug = plan_slug(plan_title)
        if selection.get("error"):
            raw_errors = attempt.get("errors") if attempt is not None else None
            if not isinstance(raw_errors, (list, tuple)) and attempt is not None:
                legacy_code = attempt.get("transport_error_code")
                raw_errors = (legacy_code,) if isinstance(legacy_code, str) else ()
            error_entries = tuple(
                entry
                for entry in (
                    raw_errors if isinstance(raw_errors, (list, tuple)) else ()
                )
                if isinstance(entry, str)
                and re.fullmatch(
                    r"[A-Z][A-Z0-9_]{1,63}", entry.partition("@")[0]
                )
            )
            if not error_entries:
                error_entries = ("UNKNOWN",)
            stage = (
                _validation_stage(error_entries)
                if attempt is not None and attempt.get("validation")
                else "transport"
            )
            return self._block(
                reservation,
                selection,
                stage=stage,
                errors=error_entries,
                received=(
                    attempt.get("received_result")
                    if attempt is not None
                    else None
                ),
            )

        plan_result = _plan_payload(selection)
        if self._planned_at_errors:
            return self._block(
                reservation,
                selection,
                stage=_validation_stage(self._planned_at_errors),
                errors=self._planned_at_errors,
                received=plan_result,
            )

        try:
            current_head = git_ops.head_sha(self._repo)
        except git_ops.GitError:
            current_head = None
        reanchored = current_head != self._planned_at
        planned_at = self._planned_at
        plans_dir = self._plans_dir
        filename = f"{number:03d}-{slug}.md"

        def _reanchor_failed() -> PlanOutcome:
            # Failed worktrees must remain prunable and reusable by later findings.
            failed_wt = self._reanchor_worktree
            self._release_reanchor_worktree()
            if failed_wt is not None:
                try:
                    git_ops.worktree_remove_unlocked(self._repo, failed_wt)
                except git_ops.GitError:
                    pass
            return self._block(
                reservation,
                selection,
                stage="transport",
                errors=("PLAN_REANCHOR_FAILED",),
                received=plan_result,
            )

        if reanchored:
            try:
                planned_at = git_ops.head_sha(self._repo)
            except git_ops.GitError:
                return _reanchor_failed()
        try:
            if reanchored:
                worktree = self._reanchor_worktree
                if worktree is None:
                    run_id = self._run_session_id or f"run-{self._planned_at[:12]}"
                    if _SAFE_DIRNAME.fullmatch(run_id) is None:
                        run_id = f"run-{self._planned_at[:12]}"
                    worktree = self._worktrees_root / f"{run_id}{_REANCHOR_DIR_SUFFIX}"
                    # Add already locked so concurrent pruning has no live-worktree race.
                    git_ops.worktree_add(
                        self._repo,
                        worktree,
                        planned_at,
                        detach=True,
                        lock_reason=run_id,
                    )
                    self._reanchor_worktree = worktree
                plans_dir = worktree / "daydream_plans"
            try:
                assert plan_result is not None
                text = render_plan(
                    finding,
                    plan=plan_result,
                    planned_at=planned_at,
                    number=number,
                    planned_on=self._planned_on,
                    run_session_id=self._run_session_id,
                )
            except Exception:  # noqa: BLE001 - retain distinct render and re-anchor diagnostics
                if reanchored:
                    raise
                return self._block(
                    reservation,
                    selection,
                    stage="render",
                    errors=("RENDER_FAILED",),
                    received=plan_result,
                )
            if reanchored:
                plans_dir.mkdir(parents=True, exist_ok=True)
            (plans_dir / filename).write_text(text, encoding="utf-8")
            if reanchored:
                # Keep a durable main copy that survives worktree pruning.
                (self._plans_dir / filename).write_text(text, encoding="utf-8")
            entry = plan_index._index_entry(
                number=number,
                slug=slug,
                title=plan_title or title,
                fingerprint=reservation.fingerprint,
                finding=finding,
                planned_at=planned_at,
                status="TODO",
            )
            if reanchored:
                self._reanchored[number] = entry
                entries = dict(self._entries)
                entries.update(self._reanchored)
                self._write_index_files(
                    plans_dir,
                    [entries[index] for index in sorted(entries)],
                    check_links=True,
                )
                landed_rel = (
                    (self._plans_dir / filename).relative_to(self._repo).as_posix()
                )
                entry = replace(
                    entry,
                    status=f"{plan_index.REANCHORED_STATUS_PREFIX} (landed at {landed_rel})",
                )
            self._entries[number] = entry
            if not reanchored:
                outcome = self._record_written(
                    reservation,
                    selection,
                    path=filename,
                    artifact={"path": filename, "status": "TODO"},
                )
            # Index before returning so interruption cannot cause silent re-planning.
            self._write_index()
        except Exception:  # noqa: BLE001 - persist a safe re-anchor disposition
            if not reanchored:
                raise
            return _reanchor_failed()
        if not reanchored:
            return outcome
        landed_path = (plans_dir / filename).as_posix()
        return self._record_written(
            reservation,
            selection,
            path=landed_path,
            artifact={
                "path": landed_path,
                "status": "TODO",
                "reanchored": True,
                "planned_at": planned_at,
            },
        )

    def _write_index(self) -> None:
        """Write the durable sidecar before README after every landing, including interrupted runs."""
        entries = [self._entries[number] for number in sorted(self._entries)]
        self._write_index_files(self._plans_dir, entries)

    def _write_index_files(
        self,
        plans_dir: Path,
        entries: Sequence[plan_index.PlanIndexEntry],
        *,
        check_links: bool = False,
    ) -> None:
        """Write the sidecar and its rendered index for *entries* into *plans_dir*."""
        (plans_dir / plan_index.PLAN_INDEX_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": plan_index.PLAN_INDEX_SCHEMA_VERSION,
                    "artifact_type": "daydream.plan-index",
                    "plans": [asdict(entry) for entry in entries],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (plans_dir / "README.md").write_text(
            plan_index._render_index(
                [
                    plan_index._index_row(
                        entry,
                        plans_dir=plans_dir if check_links else None,
                    )
                    for entry in entries
                ],
                plans_dir=plans_dir,
                planned_on=self._planned_on,
                non_interactive_default=self._non_interactive_default,
                run_session_id=self._run_session_id,
            ),
            encoding="utf-8",
        )


def _plan_payload(selection: dict[str, Any]) -> AdmittedPlan | None:
    plan = selection.get("plan")
    return plan if isinstance(plan, AdmittedPlan) else None


def _by_reservation(
    entries: Sequence[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [entry for _, entry in sorted(entries, key=lambda item: item[0])]
