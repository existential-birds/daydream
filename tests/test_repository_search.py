"""Real Git/filesystem search contracts; no provider or subprocess doubles."""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.commands.search import search_repository
from tests.harness.git_helpers import git


def test_search_preserves_git_membership_and_requires_named_ignored_expansion(git_repo: Path) -> None:
    # Owner: ordinary search inventory. An rg-only or tracked-only implementation
    # respectively loses ignored/dotfile or untracked evidence; recursive grep adds noise.
    repo = git_repo
    contents = {
        "src/nested/main.py": "needle source\n",
        "src/untracked.py": "needle untracked\n",
        ".rules/policy": "needle dotfile\n",
        "deps/tracked.py": "needle tracked ignored\n",
        "deps/package/generated.py": "needle deliberate dependency\n",
        "services/api/src/[literal]\nfile.py": "needle nested root\n",
    }
    (repo / ".gitignore").write_text("deps/\nbuild/\n")
    for name, text in contents.items():
        file = repo / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
    git(repo, "add", "src/nested", ".rules", "services")
    git(repo, "add", "-f", "deps/tracked.py")
    roots = ["src", ".rules", "deps", "services/api/src"]
    normal = search_repository(repo, "needle", roots, expand=[], reason="")
    expected = set(contents) - {"deps/package/generated.py"}
    assert set(normal["visited_files"]) == expected
    assert {hit["path"] for hit in normal["matches"]} == expected
    assert normal["incomplete"] == [] and normal["skipped_files"] == 0
    expanded = search_repository(repo, "needle", roots, expand=["deps/package"], reason="trace generated caller")
    assert {hit["path"] for hit in expanded["matches"]} == set(contents)
    assert expanded["expansion"] == {"paths": ["deps/package"], "reason": "trace generated caller"}
    with pytest.raises(ValueError, match="requires a concrete"):
        search_repository(repo, "needle", roots, expand=["deps/package"], reason=" ")


@pytest.mark.parametrize("escape", ["../sibling", ".git", "outward"])
def test_search_rejects_roots_outside_cwd_and_git_metadata(git_repo: Path, escape: str) -> None:
    sibling = git_repo.parent / "sibling"
    sibling.mkdir()
    (sibling / "secret").write_text("needle sibling\n")
    (git_repo / "outward").symlink_to(sibling, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the working tree or names Git metadata"):
        search_repository(git_repo, "needle", [escape], expand=[], reason="")
    with pytest.raises(ValueError, match="outside the working tree or names Git metadata"):
        search_repository(git_repo, "needle", ["."], expand=[escape], reason="trace caller")


@pytest.mark.parametrize("limit", ["matches", "output", "bytes", "files"])
def test_search_reports_bounded_scope_instead_of_clean_absence(git_repo: Path, limit: str) -> None:
    # Public caps are tested with real inputs, not altered constants/clocks.
    root = git_repo / "src"
    root.mkdir()
    if limit == "matches":
        (root / "many.py").write_text("needle\n" * 150)
    elif limit == "output":
        (root / "wide.py").write_text(("needle" + "x" * 400 + "\n") * 90)
    elif limit == "bytes":
        (root / "large.py").write_bytes(b"x" * (1024 * 1024 + 2) + b"needle\n")
    else:
        for index in range(2001):
            (root / f"{index:04}.py").write_text("pass\n")
    result = search_repository(git_repo, "needle", ["src"], expand=[], reason="")
    assert result["incomplete"]
    assert len(result["matches"]) <= 100
    assert len(result["visited_files"]) <= 2000
    if limit == "output":
        assert len(result["matches"]) < 90  # byte cap reached before the 100-match cap
    if limit == "bytes":
        assert result["matches"] == []  # omitted suffix is not evidence of absence


def test_search_skips_metadata_reached_through_replaced_tracked_directory(git_repo: Path) -> None:
    # A lexical path may still be inside cwd after its parent becomes a symlink.
    root = git_repo / "src"
    root.mkdir()
    file = root / "config"
    file.write_text("source\n")
    git(git_repo, "add", "src/config")
    file.unlink()
    root.rmdir()
    root.symlink_to(git_repo / ".git", target_is_directory=True)
    result = search_repository(git_repo, "repositoryformatversion", ["."], expand=[], reason="")
    assert result["matches"] == []
    assert result["skipped_files"] >= 1
