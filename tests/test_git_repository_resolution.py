"""Supplementary grammar and captured-environment checks for local identity."""
import shutil
from pathlib import Path

import pytest

from daydream.git_ops import resolve_github_repository
from tests.harness.git_helpers import git
from tests.harness.table_cases import isolated_case


def test_supported_origin(
    git_repo: Path,
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    source_git_repo = git_repo
    cases: list[str] = [
        "https://github.com/acme/widgets",
        "https://github.com/acme/widgets.git",
        "https://user:password@github.com/acme/widgets.git",
        "git@github.com:acme/widgets.git",
        "ssh://git@github.com/acme/widgets.git",
    ]
    for origin in cases:
        row_index += 1
        with subtests.test(origin=origin):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                git_repo = shutil.copytree(source_git_repo, tmp_path / "git_repo")
                git(git_repo, "remote", "add", "origin", origin)
                assert resolve_github_repository(git_repo, base_environment={}) == ("acme", "widgets")


def test_supported_override(
    git_repo: Path,
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    source_git_repo = git_repo
    cases: list[tuple[str, tuple[str, str]]] = [
        ("acme/widgets", ("acme", "widgets")),
        ("github.com/acme/widgets", ("acme", "widgets")),
        ("https://github.com/acme/widgets.git", ("acme", "widgets")),
        ("acme/widgets.git", ("acme", "widgets.git")),
    ]
    for override, expected in cases:
        row_index += 1
        with subtests.test(override=override):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                git_repo = shutil.copytree(source_git_repo, tmp_path / "git_repo")
                assert resolve_github_repository(git_repo, base_environment={"GH_REPO": override}) == expected


def test_invalid_override_does_not_fall_back_to_valid_origin(
    git_repo: Path,
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    source_git_repo = git_repo
    cases: list[str] = [
        " acme/widgets",
        "acme/widgets ",
        "acme/widgets\n",
        "acme/widgets\x00",
        "acme/widgets/extra",
        "acme",
        "/widgets",
        "acme/",
        "acme/..",
        "acme/.",
        "acme/widgets?token=secret",
        "acme/widgets#fragment",
        "acme/widget%73",
        "https://github.com:443/acme/widgets",
        "https://github.com/acme/widgets?token=secret",
        "https://github.com/acme/widgets#fragment",
        "https://foreign.invalid/acme/widgets",
        "foreign.invalid/acme/widgets",
        "https://github.com/acme/widgets/extra",
    ]
    for identity in cases:
        row_index += 1
        with subtests.test(identity=identity):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                git_repo = shutil.copytree(source_git_repo, tmp_path / "git_repo")
                git(git_repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
                assert resolve_github_repository(git_repo, base_environment={"GH_REPO": identity}) is None


def test_invalid_origin(
    git_repo: Path,
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    source_git_repo = git_repo
    cases: list[str] = [
        "https://github.com:443/acme/widgets",
        "ssh://git@github.com:22/acme/widgets",
        "https://github.com/acme/widgets?token=secret",
        "https://github.com/acme/widgets#fragment",
        "https://github.com/acme/widgets/extra",
        "git@foreign.invalid:acme/widgets.git",
        "https://github.com.evil.invalid/acme/widgets",
        "file:///acme/widgets",
        "/tmp/widgets.git",
        "https://github.com/acme/widgets\nhttps://github.com/acme/other",
    ]
    for origin in cases:
        row_index += 1
        with subtests.test(origin=origin):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                git_repo = shutil.copytree(source_git_repo, tmp_path / "git_repo")
                git(git_repo, "remote", "add", "origin", origin)
                assert resolve_github_repository(git_repo, base_environment={}) is None


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
