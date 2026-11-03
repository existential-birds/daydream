"""Supplementary grammar and captured-environment checks for local identity."""
from pathlib import Path

import pytest

from daydream.git_ops import resolve_github_repository
from tests.harness.git_helpers import git


def test_repository_identity_grammar_admits_only_unambiguous_github_identities(git_repo: Path) -> None:
    """Invalid overrides never fall back to a valid origin; only exact github.com remotes resolve."""
    for override, expected in [
        ("acme/widgets", ("acme", "widgets")), ("github.com/acme/widgets", ("acme", "widgets")),
        ("https://github.com/acme/widgets.git", ("acme", "widgets")), ("acme/widgets.git", ("acme", "widgets.git")),
    ]:
        assert resolve_github_repository(git_repo, base_environment={"GH_REPO": override}) == expected
    git(git_repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    for identity in [
        " acme/widgets", "acme/widgets\x00", "acme/widgets/extra", "acme/..", "acme/widget%73",
        "https://github.com:443/acme/widgets", "https://github.com/acme/widgets?token=secret",
        "https://github.com/acme/widgets#fragment", "https://foreign.invalid/acme/widgets",
    ]:
        assert resolve_github_repository(git_repo, base_environment={"GH_REPO": identity}) is None, identity
    for origin in [
        "https://github.com/acme/widgets", "https://github.com/acme/widgets.git",
        "https://user:password@github.com/acme/widgets.git", "git@github.com:acme/widgets.git",
        "ssh://git@github.com/acme/widgets.git",
    ]:
        git(git_repo, "remote", "set-url", "origin", origin)
        assert resolve_github_repository(git_repo, base_environment={}) == ("acme", "widgets"), origin
    for origin in [
        "ssh://git@github.com:22/acme/widgets", "https://github.com/acme/widgets/extra",
        "https://github.com.evil.invalid/acme/widgets", "file:///acme/widgets",
        "https://github.com/acme/widgets\nhttps://github.com/acme/other",
    ]:
        git(git_repo, "remote", "set-url", "origin", origin)
        assert resolve_github_repository(git_repo, base_environment={}) is None, origin


def test_explicit_identity_and_captured_environment_win(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    git(git_repo, "remote", "add", "origin", "https://github.com/origin/repository.git")
    monkeypatch.setenv("GH_REPO", "ambient/repository")
    monkeypatch.setenv("GH_HOST", "foreign.invalid")
    captured = {"GH_REPO": "captured/repository", "GH_HOST": "github.com"}
    assert resolve_github_repository(git_repo, "explicit/repository", base_environment=captured) == (
        "explicit", "repository",
    )
    assert resolve_github_repository(git_repo, base_environment=captured) == ("captured", "repository")
    assert resolve_github_repository(git_repo, base_environment={}) == ("origin", "repository")
    assert captured == {"GH_REPO": "captured/repository", "GH_HOST": "github.com"}
    assert resolve_github_repository(git_repo, "invalid/extra/path", base_environment=captured) is None
    assert resolve_github_repository(
        git_repo, "explicit/repository", base_environment={"GH_HOST": "foreign.invalid"},
    ) is None
