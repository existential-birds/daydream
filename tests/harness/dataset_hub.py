"""Offline HF boundary with immutable revision trees and real commit conflicts."""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping

from daydream.dataset_hub_client import HubConflict, HubError


class FakeDatasetHub:
    """Store bytes at content-derived commits; inject only external HF failures."""

    def __init__(self, *, private: bool = True) -> None:
        self.private = private
        self.fail_before = False
        self.lose_response = False
        self.before_commit: Callable[[], None] | None = None
        self.commits: list[str] = []
        self.trees: dict[str, dict[str, bytes]] = {}
        self.revision = self._save({})

    def _save(self, tree: Mapping[str, bytes]) -> str:
        digest = hashlib.sha256()
        for path, content in sorted(tree.items()):
            encoded = path.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        revision = digest.hexdigest()[:40]
        self.trees[revision] = dict(tree)
        return revision

    def private_revision(self, repo_id: str) -> str:
        del repo_id
        if not self.private:
            raise HubError("public_destination")
        if self.fail_before:
            raise HubError("network_failed")
        return self.revision

    def read_file(self, repo_id: str, path: str, revision: str) -> bytes | None:
        del repo_id
        if self.fail_before:
            raise HubError("network_failed")
        if revision not in self.trees:
            raise HubError("unknown_revision")
        return self.trees[revision].get(path)

    def commit(self, repo_id: str, files: Mapping[str, bytes], parent_revision: str) -> str:
        del repo_id
        if self.fail_before:
            raise HubError("network_failed")
        callback, self.before_commit = self.before_commit, None
        if callback is not None:
            callback()
        if not self.private:
            raise HubError("public_destination")
        if parent_revision != self.revision:
            raise HubConflict("concurrent_update")
        self.revision = self._save({**self.trees[self.revision], **files})
        self.commits.append(self.revision)
        if self.lose_response:
            self.lose_response = False
            raise HubError("network_failed")
        return self.revision
