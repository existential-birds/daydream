"""In-memory Hub double for offline hydration tests (issue #982).

Implements the :class:`~daydream.archive.hydrate.HubClient` protocol against a
plain ``dict[str, bytes]`` file tree. Kept in-package so integration tests and
future tooling share one fake; production modules must never import it.
"""

from __future__ import annotations

import copy
from pathlib import Path

from daydream.archive.hydrate import HubDownloadError, RepoInfo


class FakeHub:
    """Minimal in-memory Hub serving a checked-in snapshot fixture."""

    def __init__(
        self,
        *,
        repo_id: str,
        private: bool = True,
        files: dict[str, bytes] | None = None,
        head_sha: str = "head",
    ) -> None:
        self.repo_id = repo_id
        self.private = private
        self.files: dict[str, bytes] = dict(files or {})
        self._head = head_sha
        # Revision name -> immutable snapshot of the file tree at that revision.
        self._revisions: dict[str, dict[str, bytes]] = {head_sha: dict(self.files)}
        # Symbolic refs (branches) -> sha they currently point at.
        self._refs: dict[str, str] = {}

    def __repr__(self) -> str:
        return f"FakeHub(repo_id={self.repo_id!r})"

    def commit_revision(self, sha: str, *, ref: str | None = None) -> None:
        """Pin the current file tree under ``sha``; optionally point ``ref`` at it."""
        self._head = sha
        self._revisions[sha] = dict(self.files)
        if ref is not None:
            self._refs[ref] = sha

    def repo_info(self, revision: str | None = None) -> RepoInfo:
        return RepoInfo(sha=self._resolve_revision(revision), private=self.private)

    def list_revisions(self) -> list[str]:
        """All known immutable revision names (commit shas), for prefix resolution."""
        return sorted(self._revisions)

    def _resolve_revision(self, revision: str | None) -> str:
        if revision is None:
            return self._head
        if revision in self._revisions:
            return revision
        if revision in self._refs:
            return self._refs[revision]
        raise HubDownloadError(f"unknown revision {revision!r} for {self.repo_id}")

    @property
    def repo_private(self) -> bool:
        return self.private

    def list_repo_files(self) -> list[str]:
        return sorted(self.files)

    def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes:
        tree = (
            self.files
            if revision is None
            else self._revision_tree(self._resolve_revision(revision))
        )
        if path_in_repo not in tree:
            raise HubDownloadError(
                f"path {path_in_repo!r} not found in {self.repo_id} "
                f"(revision={revision or self._head})"
            )
        return tree[path_in_repo]

    def upload_files(self, mapping: dict[str | Path, Path], commit_message: str) -> None:
        del commit_message  # recorded only for future assertion needs
        for path_in_repo, local_path in mapping.items():
            self.files[str(path_in_repo)] = Path(local_path).read_bytes()

    def _revision_tree(self, revision: str) -> dict[str, bytes]:
        tree = self._revisions.get(revision)
        if tree is None:
            raise HubDownloadError(f"unknown revision {revision!r} for {self.repo_id}")
        return copy.copy(tree)
