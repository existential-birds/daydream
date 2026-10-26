
import importlib.metadata
from typing import Any

import pytest

import daydream
from daydream import git_ops
from daydream.archive import provenance
from daydream.git_ops import GitError


def test_commit_and_dirty_resolve_known_git_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_IMAGE_DIGEST", "sha256:abc123")
    expected_sha = "a" * 40
    monkeypatch.setattr(git_ops, "head_sha", lambda _repo: expected_sha)
    monkeypatch.setattr(git_ops, "status_porcelain", lambda _repo: " M tests/test_provenance.py\n")

    p = provenance.capture_executable_provenance()
    assert p.commit == expected_sha
    assert p.dirty is True
    assert p.container_digest == "sha256:abc123"
    assert p.version == daydream.__version__
    assert p.install_source in {"editable", "git", "package", "unknown"}

def test_commit_and_dirty_unknown_on_git_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """One capture retains explicit unknowns for every independently unavailable source."""
    def _raise(_repo: Any) -> None:
        raise GitError("synthetic git failure")
    def _missing_distribution(_name: Any) -> None:
        raise importlib.metadata.PackageNotFoundError("synthetic dist lookup failure")

    monkeypatch.delenv("DAYDREAM_IMAGE_DIGEST", raising=False)
    monkeypatch.setattr(git_ops, "head_sha", _raise)
    monkeypatch.setattr(git_ops, "status_porcelain", _raise)
    monkeypatch.setattr(importlib.metadata, "distribution", _missing_distribution)
    p = provenance.capture_executable_provenance()
    assert p.commit == "unknown"
    assert p.dirty == "unknown"
    assert p.to_dict() == {
        "version": daydream.__version__,
        "install_source": "unknown",
        "commit": "unknown",
        "dirty": "unknown",
        "container_digest": "unknown",
    }
