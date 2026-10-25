"""Real Git/filesystem search contracts; no provider or subprocess doubles."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from daydream.commands.search import search_repository
from tests.harness.git_helpers import git
from tests.harness.table_cases import isolated_case


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


def test_search_cli_excerpts_retain_literal_hits_with_bounded_omissions(
    git_repo: Path,
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    source_git_repo = git_repo
    cases: list[tuple[bytes, str, bytes, tuple[bool, bool]]] = [
        (b"x" * 500, "needle", b"y" * 500, (True, True)),
        (b"", "needle", b"y" * 500, (False, True)),
        (b"x" * 500, "needle", b"", (True, False)),
        ("é".encode() * 250, "針🔍", "🙂".encode() * 250, (True, True)),
        (b"\xff" * 500, "needle", b"y" * 500, (True, True)),
        (b"x" * 500, "n" * 234, b"y" * 500, (True, True)),
        (b"", "n" * 237, b"y" * 500, (False, True)),
        (b"x" * 500, "n" * 237, b"", (True, False)),
        (b"", "針" * 80, b"", (False, False)),
        (b"x" * 500, "針" * 100, b"y" * 500, (True, True)),
    ]
    for prefix, needle, suffix, omissions in cases:
        row_index += 1
        with subtests.test(needle=needle):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                git_repo = shutil.copytree(source_git_repo, tmp_path / "git_repo")
                (git_repo / ".gitignore").write_text("deps/\n")
                generated = git_repo / "deps/pkg/generated.js"
                generated.parent.mkdir(parents=True)
                generated.write_bytes(prefix + needle.encode() + suffix + b"\n")
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "daydream",
                        "search",
                        needle,
                        ".",
                        "--expand",
                        "deps/pkg",
                        "--reason",
                        "inspect the generated caller",
                    ],
                    cwd=git_repo,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert completed.returncode == 0, completed.stderr
                result = json.loads(completed.stdout)
                assert len(result["matches"]) == 1
                hit = result["matches"][0]
                assert (hit["path"], hit["line"]) == ("deps/pkg/generated.js", 1)
                text = hit["text"]
                if len(needle.encode()) <= 240 - 3 * sum(omissions):
                    assert needle in text
                else:
                    assert needle.startswith(text[3:-3]) and text[3:-3]
                assert len(text.encode("utf-8")) <= 240
                assert (text.startswith("..."), text.endswith("...")) == omissions
                if omissions == (True, True) and len(needle.encode()) < 234:
                    before, after = text.split(needle)
                    assert len(before) > 3 and len(after) > 3
                if not any(omissions):
                    assert text == needle


def test_search_rejects_roots_outside_cwd_and_git_metadata(
    git_repo: Path,
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    source_git_repo = git_repo
    cases: list[str] = ["../sibling", ".git", "outward"]
    for escape in cases:
        row_index += 1
        with subtests.test(escape=escape):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                git_repo = shutil.copytree(source_git_repo, tmp_path / "git_repo")
                sibling = git_repo.parent / "sibling"
                sibling.mkdir()
                (sibling / "secret").write_text("needle sibling\n")
                (git_repo / "outward").symlink_to(sibling, target_is_directory=True)
                for args in ([escape], [".", "--expand", escape, "--reason", "trace caller"]):
                    completed = subprocess.run(
                        [sys.executable, "-m", "daydream", "search", "needle", *args],
                        cwd=git_repo,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    assert completed.returncode == 2
                    assert "outside the working tree or names Git metadata" in json.loads(completed.stdout)["error"]


def test_search_cli_rejects_incomplete_expansion_requests(
    git_repo: Path,
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    source_git_repo = git_repo
    cases: list[tuple[list[str], str]] = [
        (["--expand", "deps/pkg", "--reason", "trace caller"], "the following arguments are required: roots"),
        (["deps/pkg", "--expand", "deps/pkg"], "--expand requires a concrete --reason"),
        (["deps/pkg", "--expand", "deps/pkg", "--reason", " "], "--expand requires a concrete --reason"),
    ]
    for args, error in cases:
        row_index += 1
        with subtests.test(error=error):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                git_repo = shutil.copytree(source_git_repo, tmp_path / "git_repo")
                (git_repo / ".gitignore").write_text("deps/\n")
                package = git_repo / "deps/pkg"
                package.mkdir(parents=True)
                (package / "generated.py").write_text("needle ignored dependency\n")
                completed = subprocess.run(
                    [sys.executable, "-m", "daydream", "search", "needle", *args],
                    cwd=git_repo,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert completed.returncode == 2
                assert error in completed.stderr + completed.stdout


def test_search_reports_bounded_scope_instead_of_clean_absence(
    git_repo: Path,
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    source_git_repo = git_repo
    cases: list[str] = ["matches", "output", "bytes", "files"]
    for limit in cases:
        row_index += 1
        with subtests.test(limit=limit):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                git_repo = shutil.copytree(source_git_repo, tmp_path / "git_repo")
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
  # omitted suffix is not evidence of absence


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
