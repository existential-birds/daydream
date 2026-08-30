
from typing import Any

import pytest

from daydream import git_ops
from daydream.archive import provenance


def test_version_is_package_version() -> None:
    import daydream
    p = provenance.capture_executable_provenance()
    assert p.version == daydream.__version__


def test_container_digest_from_optin_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_IMAGE_DIGEST", "sha256:abc123")
    assert provenance.capture_executable_provenance().container_digest == "sha256:abc123"


def test_container_digest_unknown_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYDREAM_IMAGE_DIGEST", raising=False)
    assert provenance.capture_executable_provenance().container_digest == "unknown"


def test_commit_and_dirty_resolve_known_git_state(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_sha = "a" * 40
    monkeypatch.setattr(git_ops, "head_sha", lambda _repo: expected_sha)
    monkeypatch.setattr(
        git_ops,
        "status_porcelain",
        lambda _repo: " M tests/test_provenance.py\n",
    )

    p = provenance.capture_executable_provenance()
    assert p.commit == expected_sha
    assert p.dirty is True


def test_install_source_is_known_or_unknown() -> None:
    p = provenance.capture_executable_provenance()
    assert p.install_source in {"editable", "git", "package", "unknown"}


def test_commit_and_dirty_unknown_on_git_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.git_ops import GitError

    def _raise(_repo: Any) -> None:
        raise GitError("synthetic git failure")

    monkeypatch.setattr("daydream.git_ops.head_sha", _raise)
    monkeypatch.setattr("daydream.git_ops.status_porcelain", _raise)
    p = provenance.capture_executable_provenance()
    # A deterministic git failure must resolve to the explicit "unknown"
    # sentinel, never raise and never fabricate a value.
    assert p.commit == "unknown"
    assert p.dirty == "unknown"


def test_to_dict_never_omits_unknown() -> None:
    p = provenance.capture_executable_provenance()
    d = p.to_dict()
    assert set(d) == {"version", "install_source", "commit", "dirty", "container_digest"}


def test_install_source_unknown_on_distribution_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministically exercise the distribution-failure branch of
    _resolve_install_source: when importlib.metadata.distribution("daydream")
    is unavailable (e.g. a non-installed/import-time-only environment), the
    field must resolve to the explicit "unknown" sentinel, never raise.

    Closes the remaining smoke-test gap in finding #5 (daydream covered the
    git-error branch; this covers the distribution-error branch).
    """
    import importlib.metadata

    def _raise(_name: Any) -> None:
        raise importlib.metadata.PackageNotFoundError("synthetic dist lookup failure")

    monkeypatch.setattr(importlib.metadata, "distribution", _raise)
    p = provenance.capture_executable_provenance()
    assert p.install_source == "unknown"
