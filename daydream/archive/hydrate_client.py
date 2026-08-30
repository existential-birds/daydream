"""In-memory Hub double for offline hydration tests (issue #982).

Implements the :class:`~daydream.archive.hydrate.HubClient` protocol against a
plain ``dict[str, bytes]`` file tree. Kept in-package so integration tests and
future tooling share one fake; production modules must never import it.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

from daydream.archive.hydrate import HubDownloadError, HydrationError, RepoInfo


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
        # Every successful download_file() call, in order (test observation seam).
        self.downloaded_log: list[str] = []
        self._head = head_sha
        # Revision name -> immutable snapshot of the file tree at that revision.
        self._revisions: dict[str, dict[str, bytes]] = {head_sha: dict(self.files)}
        # Symbolic refs (branches) -> sha they currently point at.
        self._refs: dict[str, str] = {}
        # One entry per upload commit, in order: {"contains": [paths], "sha": sha}.
        self.commit_order: list[dict[str, Any]] = []
        # Every path ever successfully uploaded (test observation seam).
        self.uploaded_paths: list[str] = []
        # When True, upload_files raises (simulated Hub outage).
        self.fail_uploads = False

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

    def list_repo_files(self, revision: str | None = None) -> list[str]:
        tree = (
            self.files
            if revision is None
            else self._revision_tree(self._resolve_revision(revision))
        )
        return sorted(tree)

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
        self.downloaded_log.append(path_in_repo)
        return tree[path_in_repo]

    def mutate_bundle(self, revision: str, session_id: str, content: bytes) -> None:
        """Re-commit one bundle file at ``revision`` with new content (test setup seam).

        Used to stage an identity collision: same session identity, different
        bytes. The pinned revision's tree is updated in place, so a client
        downloading that revision observes the mutation while older pinned
        revisions remain untouched.
        """
        path = f"bundles/{session_id}/manifest.json"
        self.files[path] = content
        if revision in self._revisions:
            self._revisions[revision][path] = content

    def upload_files(self, mapping: dict[str | Path, Path], commit_message: str) -> None:
        if self.fail_uploads:
            raise HydrationError(f"upload failed for {self.repo_id}: simulated Hub outage")
        paths: list[str] = []
        for path_in_repo, local_path in mapping.items():
            self.files[str(path_in_repo)] = Path(local_path).read_bytes()
            paths.append(str(path_in_repo))
        self.uploaded_paths.extend(paths)
        # Each upload is a commit: pin the new tree under a deterministic 40-hex sha.
        payload = commit_message + "\n" + "\n".join(sorted(paths))
        sha = hashlib.sha256(payload.encode()).hexdigest()[:40]
        self._revisions[sha] = dict(self.files)
        self._head = sha
        self.commit_order.append({"contains": sorted(paths), "sha": sha})

    def _revision_tree(self, revision: str) -> dict[str, bytes]:
        tree = self._revisions.get(revision)
        if tree is None:
            raise HubDownloadError(f"unknown revision {revision!r} for {self.repo_id}")
        return copy.copy(tree)
