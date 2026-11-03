"""Real Git/filesystem search contracts; no provider or subprocess doubles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.commands.search import handle_search_command, search_repository
from daydream.repository_paths import is_test_path
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


def test_search_cli_bounds_excerpts_and_rejects_escaping_or_incomplete_requests(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # Owner: CLI match evidence and argument guards. Cap tests match at byte zero and miss an
    # excerpt that hides the hit in a minified expanded dependency; the library-only reason check
    # cannot catch missing positional roots being defaulted or escaping roots/expansions.
    monkeypatch.chdir(git_repo)
    (git_repo / ".gitignore").write_text("deps/\n")
    generated = git_repo / "deps/pkg/generated.js"
    generated.parent.mkdir(parents=True)
    for prefix, needle, suffix, omissions in [
        ("é".encode() * 250, "針🔍", "🙂".encode() * 250, (True, True)),
        (b"x" * 500, "n" * 234, b"y" * 500, (True, True)),
        (b"", "n" * 237, b"y" * 500, (False, True)),
        (b"x" * 500, "n" * 237, b"", (True, False)),
        (b"", "針" * 80, b"", (False, False)),
        (b"x" * 500, "針" * 100, b"y" * 500, (True, True)),
    ]:
        generated.write_bytes(prefix + needle.encode() + suffix + b"\n")
        assert handle_search_command([needle, ".", "--expand", "deps/pkg", "--reason", "inspect the caller"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert len(result["matches"]) == 1
        hit = result["matches"][0]
        case = (len(prefix), needle[:3], len(needle), len(suffix))
        assert (hit["path"], hit["line"]) == ("deps/pkg/generated.js", 1), case
        text = hit["text"]
        if len(needle.encode()) <= 240 - 3 * sum(omissions):
            assert needle in text, case
        else:
            assert needle.startswith(text[3:-3]) and text[3:-3], case
        assert len(text.encode("utf-8")) <= 240, case
        assert (text.startswith("..."), text.endswith("...")) == omissions, case
        if omissions == (True, True) and len(needle.encode()) < 234:
            before, after = text.split(needle)
            assert len(before) > 3 and len(after) > 3, case
        if not any(omissions):
            assert text == needle, case

    sibling = git_repo.parent / "sibling"
    sibling.mkdir()
    (sibling / "secret").write_text("needle sibling\n")
    (git_repo / "outward").symlink_to(sibling, target_is_directory=True)
    for escape in ("../sibling", ".git", "outward"):
        for args in ([escape], [".", "--expand", escape, "--reason", "trace caller"]):
            assert handle_search_command(["needle", *args]) == 2, args
            error = json.loads(capsys.readouterr().out)["error"]
            assert "outside the working tree or names Git metadata" in error, args
    for args, error in [
        (["--expand", "deps/pkg", "--reason", "trace caller"], "the following arguments are required: roots"),
        (["deps/pkg", "--expand", "deps/pkg"], "--expand requires a concrete --reason"),
        (["deps/pkg", "--expand", "deps/pkg", "--reason", " "], "--expand requires a concrete --reason"),
    ]:
        code: object
        try:
            code = handle_search_command(["needle", *args])
        except SystemExit as exit_:
            code = exit_.code
        assert code == 2, args
        captured = capsys.readouterr()
        assert error in captured.err + captured.out, args


def test_test_path_classification_uses_names_and_directories() -> None:
    """Test files are recognized by parent directories (case-folded) or case-sensitive name conventions."""
    assert is_test_path("app/spec/user.py") is True  # a parent directory named spec
    assert is_test_path("app/specs.py") is False  # a file stem is not a directory convention
    assert is_test_path("src/WidgetTest.java") is True
    assert is_test_path("src/widgettest.java") is False
    assert is_test_path("SRC/Tests/widget.java") is True
