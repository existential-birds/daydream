"""Archive, Git, GitHub, cache, and clock adapters for harvest passes."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from rich.console import Console

from daydream import git_ops
from daydream.archive.index import (
    append_label_observation,
    query_runs,
    readonly_connection,
    reviewer_set_penalty_prior,
    set_run_pr_link,
)
from daydream.git_ops import GitError, GitHubAuth
from daydream.training import labeler_versions
from daydream.training.backfill_cache import BackfillCache
from daydream.training.harvest import (
    AnnotationPayload,
    HarvestConfig,
    _github_with_retry,
    _materialize_base_sha_if_missing,
    _resolve_repo_for_row,
    assemble_scoring_inputs,
)
from daydream.training.harvest_types import BaseShaStatus, HarvestRow
from daydream.training.labeler_signals import (
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    fix_applied_signal,
    local_commit_applied_signal,
)
from daydream.training.reward import ScoringInputs


class ProductionHarvestServices:
    """Per-run adapters for archive, repository, GitHub, cache, and clocks."""

    def __init__(self, config: HarvestConfig, github_auth: GitHubAuth) -> None:
        self._config = config
        self._github_auth = github_auth
        self._cache: BackfillCache | None = None
        self._fetched_repos: set[Path] = set()

    @property
    def archive_dir(self) -> Path:
        return self._config.archive_dir

    @property
    def progress_path(self) -> Path | None:
        return self._config.cache_dir / "progress.jsonl" if self._config.cache_dir is not None else None

    def _uncached_github(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        return _github_with_retry(
            repo,
            endpoint,
            auth=self._github_auth,
            backoff_sleep=self.backoff_sleep,
            **kwargs,
        )

    def _cache_instance(self) -> BackfillCache | None:
        if self._config.cache_dir is None or self._config.dry_run:
            return None
        if self._cache is None:
            self._cache = BackfillCache(
                cache_dir=self._config.cache_dir,
                inner=self._uncached_github,
            )
        return self._cache

    def query_rows(self, session_filter: str | None) -> Sequence[Mapping[str, Any]]:
        if not self.archive_dir.exists():
            raise FileNotFoundError(f"archive_dir does not exist: {self.archive_dir}")
        if self._config.dry_run:
            with closing(readonly_connection(self.archive_dir)) as conn:
                return [
                    dict(row)
                    for row in conn.execute(
                        "SELECT * FROM runs WHERE session_id LIKE ? || '%'",
                        (session_filter or "",),
                    )
                ]
        if session_filter:
            return query_runs(
                self.archive_dir,
                "session_id LIKE ? || '%'",
                (session_filter,),
            )
        return query_runs(self.archive_dir)

    def completed_sessions(self) -> set[str]:
        cache = self._cache_instance()
        return cache.completed_sessions() if cache is not None else set()

    def resolve_repo(self, row: HarvestRow, *, console: Console) -> Path | None:
        clone_cache = self._config.repo_clone_root or (
            self._config.cache_dir / "repos" if self._config.cache_dir else None
        )
        if self._config.dry_run:
            candidates = [row.source_path]
            if clone_cache is not None and row.repo_slug:
                candidates.append(clone_cache / row.repo_slug)
            return next((p for p in candidates if p is not None and (p / ".git").exists()), None)
        return _resolve_repo_for_row(
            row,
            clone_cache,
            fetched_repos=self._fetched_repos,
            console=console,
        )

    def materialize_base_sha(
        self,
        row: HarvestRow,
        repo_clone: Path | None,
        *,
        console: Console,
    ) -> BaseShaStatus:
        if self._config.dry_run:
            return "available" if row.base_sha else "unavailable"
        return _materialize_base_sha_if_missing(row, repo_clone, console=console)

    def github(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        cache = self._cache_instance()
        if cache is not None:
            return cache(repo, endpoint, **kwargs)
        return self._uncached_github(repo, endpoint, **kwargs)

    def reviewer_prior(
        self,
        logins: tuple[str, ...],
        *,
        before_valid_at: str,
        exclude_session: str,
        repo_slug: str | None,
    ) -> tuple[float | None, int]:
        return reviewer_set_penalty_prior(
            self.archive_dir,
            list(logins),
            before_valid_at=before_valid_at,
            exclude_session=exclude_session,
            repo_slug=repo_slug,
            readonly=self._config.dry_run,
        )

    def set_pr_link(self, row: HarvestRow, number: int, repo: str) -> None:
        set_run_pr_link(self.archive_dir, row.session_id, number, repo)

    def read_scoring_inputs(self, row: HarvestRow) -> ScoringInputs:
        return assemble_scoring_inputs(row.archive_path)

    def read_recorded_fingerprints(self, row: HarvestRow) -> tuple[str, ...]:
        if row.findings_fingerprints is not None:
            return row.findings_fingerprints
        try:
            data = json.loads((row.archive_path / "findings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ()
        findings = data.get("findings") if isinstance(data, dict) else None
        if not isinstance(findings, list):
            return ()
        return tuple(
            str(finding["fingerprint"])
            for finding in findings
            if isinstance(finding, dict) and "fingerprint" in finding
        )

    @staticmethod
    def _file_at(repo: Path, path: str, sha: str) -> str:
        try:
            return git_ops.show(repo, sha, path).decode("utf-8", errors="replace")
        except GitError:
            return ""

    def fix_applied(
        self,
        row: HarvestRow,
        *,
        changed_files: tuple[str, ...],
        repo_clone: Path,
    ) -> FixAppliedSignal:
        return fix_applied_signal(
            row.as_signal_row(),
            changed_files=list(changed_files),
            repo_clone=repo_clone,
            diff_fetcher=git_ops.diff_name_only,
            commits_in_window_fetcher=lambda repo, head, base: list(reversed(git_ops.log_shas_since(repo, head, base))),
            file_at_fetcher=self._file_at,
        )

    def local_commit_applied(
        self,
        row: HarvestRow,
        *,
        repo_clone: Path,
    ) -> LocalCommitAppliedSignal:
        return local_commit_applied_signal(
            row.as_signal_row(),
            repo_clone=repo_clone,
            commits_since_fetcher=lambda repo, branch, since: git_ops.log_shas(
                repo,
                branch,
                since=since,
            ),
            file_at_fetcher=self._file_at,
        )

    def append_annotation(self, row: HarvestRow, payload: AnnotationPayload) -> bool:
        return append_label_observation(
            self.archive_dir,
            row.session_id,
            labeler_version=labeler_versions.LABELER_POLICY_VERSION,
            **asdict(payload),
        )

    def mark_session_done(self, session_id: str) -> None:
        cache = self._cache_instance()
        if cache is not None:
            cache.mark_session_done(session_id)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def backoff_sleep(seconds: float) -> None:
        time.sleep(seconds)

    @staticmethod
    async def sleep_between_rows(seconds: float) -> None:
        await anyio.sleep(seconds)
