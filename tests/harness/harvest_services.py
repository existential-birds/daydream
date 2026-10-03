"""Explicit per-run harvest service doubles for training tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rich.console import Console

from daydream.training.harvest import AnnotationPayload, HarvestPass
from daydream.training.harvest_types import HarvestRow
from daydream.training.labeler_signals import LocalCommitAppliedSignal


class HarvestTestServices(HarvestPass):
    """Forward a concrete per-run provider while explicitly replacing test seams.

    The real pass retains acquisition order, SQLite/cache behavior, and policy.
    Overrides replace only the explicit external inputs, timing, or write edge.
    """

    def __init__(self, delegate: HarvestPass, *, rows: list[Mapping[str, Any]] | None = None,
        completed: set[str] | None = None, completed_sessions: Callable[[], set[str]] | None = None,
        github: Callable[..., Any] | None = None, resolve_repo: Callable[..., Path | None] | None = None,
        reviewer_prior: Callable[..., tuple[float | None, int]] | None = None,
        local_commit_applied: Callable[..., LocalCommitAppliedSignal] | None = None,
        append_annotation: Callable[..., bool] | None = None,
    ) -> None:
        super().__init__(delegate._config, github_auth=delegate._github_auth)
        self._rows = rows
        self._completed = completed
        self._completed_sessions = completed_sessions
        self._github = github
        self._resolve_repo = resolve_repo
        self._reviewer_prior = reviewer_prior
        self._local_commit_applied = local_commit_applied
        self._append_annotation = append_annotation

    def query_rows(self, session_filter: str | None) -> list[Mapping[str, Any]]:
        return list(self._rows) if self._rows is not None else list(super().query_rows(session_filter))

    def completed_sessions(self) -> set[str]:
        if self._completed_sessions is not None:
            return self._completed_sessions()
        return set(self._completed) if self._completed is not None else super().completed_sessions()

    def resolve_repo(self, row: HarvestRow, *, console: Console) -> Path | None:
        if self._resolve_repo is not None:
            return self._resolve_repo(row, console=console)
        return super().resolve_repo(row, console=console)


    def github(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        if self._github is not None:
            return self._github(repo, endpoint, **kwargs)
        return super().github(repo, endpoint, **kwargs)

    def reviewer_prior(
        self, logins: tuple[str, ...], *, before_valid_at: str, exclude_session: str, repo_slug: str | None,
    ) -> tuple[float | None, int]:
        if self._reviewer_prior is not None:
            return self._reviewer_prior(
                logins, before_valid_at=before_valid_at, exclude_session=exclude_session, repo_slug=repo_slug,
            )
        return super().reviewer_prior(
            logins, before_valid_at=before_valid_at, exclude_session=exclude_session, repo_slug=repo_slug,
        )





    def local_commit_applied(self, row: HarvestRow, *, repo_clone: Path) -> LocalCommitAppliedSignal:
        if self._local_commit_applied is not None:
            return self._local_commit_applied(row, repo_clone=repo_clone)
        return super().local_commit_applied(row, repo_clone=repo_clone)

    def append_annotation(self, row: HarvestRow, payload: AnnotationPayload) -> bool:
        if self._append_annotation is not None:
            return self._append_annotation(row, payload)
        return super().append_annotation(row, payload)




    @staticmethod
    async def sleep_between_rows(seconds: float) -> None:
        """Skip real inter-row delay in provider-driven tests."""
