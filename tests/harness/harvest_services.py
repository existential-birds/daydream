"""Explicit per-run harvest service doubles for training tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rich.console import Console

from daydream.training.harvest import AnnotationPayload, HarvestServices
from daydream.training.harvest_types import BaseShaStatus, HarvestRow
from daydream.training.labeler_signals import FixAppliedSignal, LocalCommitAppliedSignal
from daydream.training.reward import ScoringInputs


class HarvestTestServices:
    """Forward a concrete per-run provider while explicitly replacing test seams.

    The double deliberately implements the public protocol itself. It neither
    subclasses nor reaches into the production provider, so each test owns its
    GitHub response, timing, and any acquisition override while retaining real
    SQLite/cache behavior through the supplied delegate.
    """

    def __init__(
        self,
        delegate: HarvestServices,
        *,
        rows: list[Mapping[str, Any]] | None = None,
        completed: set[str] | None = None,
        completed_sessions: Callable[[], set[str]] | None = None,
        github: Callable[..., Any] | None = None,
        resolve_repo: Callable[..., Path | None] | None = None,
        reviewer_prior: Callable[..., tuple[float | None, int]] | None = None,
        local_commit_applied: Callable[..., LocalCommitAppliedSignal] | None = None,
        append_annotation: Callable[..., bool] | None = None,
    ) -> None:
        self._delegate = delegate
        self._rows = rows
        self._completed = completed
        self._completed_sessions = completed_sessions
        self._github = github
        self._resolve_repo = resolve_repo
        self._reviewer_prior = reviewer_prior
        self._local_commit_applied = local_commit_applied
        self._append_annotation = append_annotation

    @property
    def archive_dir(self) -> Path:
        return self._delegate.archive_dir

    @property
    def progress_path(self) -> Path | None:
        return self._delegate.progress_path

    def query_rows(self, session_filter: str | None) -> list[Mapping[str, Any]]:
        return list(self._rows) if self._rows is not None else list(self._delegate.query_rows(session_filter))

    def completed_sessions(self) -> set[str]:
        if self._completed_sessions is not None:
            return self._completed_sessions()
        return set(self._completed) if self._completed is not None else self._delegate.completed_sessions()

    def resolve_repo(self, row: HarvestRow, *, console: Console) -> Path | None:
        if self._resolve_repo is not None:
            return self._resolve_repo(row, console=console)
        return self._delegate.resolve_repo(row, console=console)

    def materialize_base_sha(
        self, row: HarvestRow, repo_clone: Path | None, *, console: Console
    ) -> BaseShaStatus:
        return self._delegate.materialize_base_sha(row, repo_clone, console=console)

    def github(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        if self._github is not None:
            return self._github(repo, endpoint, **kwargs)
        return self._delegate.github(repo, endpoint, **kwargs)

    def reviewer_prior(
        self,
        logins: tuple[str, ...],
        *,
        before_valid_at: str,
        exclude_session: str,
        repo_slug: str | None,
    ) -> tuple[float | None, int]:
        if self._reviewer_prior is not None:
            return self._reviewer_prior(
                logins,
                before_valid_at=before_valid_at,
                exclude_session=exclude_session,
                repo_slug=repo_slug,
            )
        return self._delegate.reviewer_prior(
            logins,
            before_valid_at=before_valid_at,
            exclude_session=exclude_session,
            repo_slug=repo_slug,
        )

    def set_pr_link(self, row: HarvestRow, number: int, repo: str) -> None:
        self._delegate.set_pr_link(row, number, repo)

    def read_scoring_inputs(self, row: HarvestRow) -> ScoringInputs:
        return self._delegate.read_scoring_inputs(row)

    def read_recorded_fingerprints(self, row: HarvestRow) -> tuple[str, ...]:
        return self._delegate.read_recorded_fingerprints(row)

    def fix_applied(
        self, row: HarvestRow, *, changed_files: tuple[str, ...], repo_clone: Path
    ) -> FixAppliedSignal:
        return self._delegate.fix_applied(row, changed_files=changed_files, repo_clone=repo_clone)

    def local_commit_applied(self, row: HarvestRow, *, repo_clone: Path) -> LocalCommitAppliedSignal:
        if self._local_commit_applied is not None:
            return self._local_commit_applied(row, repo_clone=repo_clone)
        return self._delegate.local_commit_applied(row, repo_clone=repo_clone)

    def append_annotation(self, row: HarvestRow, payload: AnnotationPayload) -> bool:
        if self._append_annotation is not None:
            return self._append_annotation(row, payload)
        return self._delegate.append_annotation(row, payload)

    def mark_session_done(self, session_id: str) -> None:
        self._delegate.mark_session_done(session_id)

    def now_iso(self) -> str:
        return self._delegate.now_iso()

    def backoff_sleep(self, seconds: float) -> None:
        self._delegate.backoff_sleep(seconds)

    async def sleep_between_rows(self, seconds: float) -> None:
        """Skip real inter-row delay in provider-driven tests."""
