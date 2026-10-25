"""Real Git/filesystem search contracts; no provider or subprocess doubles."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from daydream.commands.search import handle_search_command, search_repository
from tests.harness.git_helpers import git


def test_search_preserves_git_membership_and_requires_named_ignored_expansion(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # One real Git population joins admitted paths and their literal bounded CLI match evidence.
    repo = git_repo
    contents = {
        "src/nested/main.py": b"x" * 500 + b"needle" + b"y" * 500 + b"\n",
        "src/untracked.py": b"needle" + b"y" * 500 + b"\n",
        ".rules/policy": b"x" * 500 + b"needle\n",
        "deps/tracked.py": b"needle tracked ignored\n",
        "deps/package/generated.py": b"\xff" * 500 + b"needle" + b"y" * 500 + b"\n",
        "services/api/src/[literal]\nfile.py": b"needle nested root\n",
    }
    (repo / ".gitignore").write_text("deps/\nbuild/\n")
    for name, content in contents.items():
        file = repo / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
    git(repo, "add", "src/nested", ".rules", "services")
    git(repo, "add", "-f", "deps/tracked.py")
    roots = ["src", ".rules", "deps", "services/api/src"]
    normal = search_repository(repo, "needle", roots, expand=[], reason="")
    expected = set(contents) - {"deps/package/generated.py"}
    assert set(normal["visited_files"]) == expected
    assert {hit["path"] for hit in normal["matches"]} == expected
    assert normal["incomplete"] == [] and normal["skipped_files"] == 0
    monkeypatch.chdir(repo)
    assert handle_search_command(["needle", *roots, "--expand", "deps/package",
        "--reason", "trace generated caller",
    ]) == 0
    expanded = json.loads(capsys.readouterr().out)
    assert {hit["path"] for hit in expanded["matches"]} == set(contents)
    assert expanded["expansion"] == {"paths": ["deps/package"], "reason": "trace generated caller"}
    assert {hit["path"]: (hit["line"], hit["text"].startswith("..."), hit["text"].endswith("..."))
            for hit in expanded["matches"]} == {
        "src/nested/main.py": (1, True, True), "src/untracked.py": (1, False, True),
        ".rules/policy": (1, True, False), "deps/package/generated.py": (1, True, True),
        "deps/tracked.py": (1, False, False), "services/api/src/[literal]\nfile.py": (1, False, False),
    }
    for hit in expanded["matches"]:
        text = hit["text"]
        assert "needle" in text and len(text.encode("utf-8")) <= 240
        if text.startswith("...") and text.endswith("..."):
            before, after = text.split("needle")
            assert len(before) > 3 and len(after) > 3
    with pytest.raises(ValueError, match="requires a concrete"):
        search_repository(repo, "needle", roots, expand=["deps/package"], reason=" ")



@pytest.mark.parametrize(("prefix", "needle", "suffix", "omissions"), [
    pytest.param("é".encode() * 250, "針🔍", "🙂".encode() * 250, (True, True), id="utf8"),
    pytest.param(b"x" * 500, "n" * 234, b"y" * 500, (True, True), id="full-budget-hit"),
    pytest.param(b"", "n" * 237, b"y" * 500, (False, True), id="full-budget-start"),
    pytest.param(b"x" * 500, "n" * 237, b"", (True, False), id="full-budget-end"),
    pytest.param(b"", "針" * 80, b"", (False, False), id="short-utf8-line"),
    pytest.param(b"x" * 500, "針" * 100, b"y" * 500, (True, True), id="oversized-hit"),
])
def test_search_cli_excerpts_retain_literal_hits_with_bounded_omissions(
    git_repo: Path, prefix: bytes, needle: str, suffix: bytes, omissions: tuple[bool, bool],
) -> None:
    # Owner: CLI match evidence. Existing cap tests match at byte zero and miss
    # an excerpt that hides the actual hit in a minified expanded dependency.
    (git_repo / ".gitignore").write_text("deps/\n")
    generated = git_repo / "deps/pkg/generated.js"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(prefix + needle.encode() + suffix + b"\n")
    completed = subprocess.run(
        [sys.executable, "-m", "daydream", "search", needle, ".", "--expand", "deps/pkg",
         "--reason", "inspect the generated caller"],
        cwd=git_repo, capture_output=True, text=True, timeout=30,
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


@pytest.mark.parametrize("escape", ["../sibling", ".git", "outward"])
def test_search_rejects_roots_outside_cwd_and_git_metadata(git_repo: Path, escape: str) -> None:
    # Public CLI must retain the same guard for positional roots and expansion paths.
    sibling = git_repo.parent / "sibling"
    sibling.mkdir()
    (sibling / "secret").write_text("needle sibling\n")
    (git_repo / "outward").symlink_to(sibling, target_is_directory=True)
    for args in ([escape], [".", "--expand", escape, "--reason", "trace caller"]):
        completed = subprocess.run(
            [sys.executable, "-m", "daydream", "search", "needle", *args],
            cwd=git_repo, capture_output=True, text=True, timeout=30,
        )
        assert completed.returncode == 2
        assert "outside the working tree or names Git metadata" in json.loads(completed.stdout)["error"]


@pytest.mark.parametrize(("args", "error"), [
    (["--expand", "deps/pkg", "--reason", "trace caller"], "the following arguments are required: roots"),
    (["deps/pkg", "--expand", "deps/pkg"], "--expand requires a concrete --reason"),
    (["deps/pkg", "--expand", "deps/pkg", "--reason", " "], "--expand requires a concrete --reason"),
])
def test_search_cli_rejects_incomplete_expansion_requests(git_repo: Path, args: list[str], error: str) -> None:
    # Parser/dispatch owner: the library-only reason check cannot catch missing
    # positional roots being silently defaulted or CLI errors being swallowed.
    (git_repo / ".gitignore").write_text("deps/\n")
    package = git_repo / "deps/pkg"
    package.mkdir(parents=True)
    (package / "generated.py").write_text("needle ignored dependency\n")
    completed = subprocess.run(
        [sys.executable, "-m", "daydream", "search", "needle", *args],
        cwd=git_repo, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 2
    assert error in completed.stderr + completed.stdout


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
