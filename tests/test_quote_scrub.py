import os
from pathlib import Path
from typing import Any

import pytest

from daydream.git_ops import GitError
from daydream.quote_scrub import (
    scrub_smart_quotes_changed_files,
)
from tests.harness.git_helpers import git, init_repo


def _baseline_repo(tmp_path: Path, files: dict[str, str | bytes], *, symlinks: dict[str, str] | None = None,) -> Path:
    """Real git repo with *files* (and optional *symlinks*) committed as baseline."""
    repo = tmp_path / "repo"
    init_repo(repo)
    for name, content in files.items():
        path = repo / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    for name, target in (symlinks or {}).items():
        (repo / name).symlink_to(target)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    return repo


def test_scrub_driver_never_rewrites_baseline_smart_quotes_in_literals(tmp_path: Path) -> None:
    """A real mixed fix batch attributes both hunks and preserves every unowned byte."""
    name = "naïve ve.go"
    padding = "// stable padding\n" * 20
    baseline = "package main\nconst s = 'it’s'\n" + padding + "// end\n"
    repo = _baseline_repo(tmp_path, {name: baseline, "keep.py": "x = 1\n",
        "models_generated.go": "// generated ”\n", "blob.bin": b"\x00\x01"})
    git(repo, "config", "core.quotepath", "true")
    git(repo, "config", "diff.noprefix", "true")
    source = repo / name
    source.write_text("package main\nconst s = 'it’s'\n// “added” ‘single’\n" + padding
                      + "+ b/not-a-header\n// “tail”\n// end\n", encoding="utf-8")
    (repo / "new.go").write_text("// “new” ‘single’\n", encoding="utf-8")
    (repo / "keep.py").write_text("x = 1\nprint('ascii')\n", encoding="utf-8")
    (repo / "outside.go").write_text("// “operator”\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\x00\x02")
    scrubbed = scrub_smart_quotes_changed_files(
        repo, [name, "new.go", "keep.py", "models_generated.go", "blob.bin", "gone.go"], pre_fix_ref="HEAD",
    )
    assert scrubbed == [name, "new.go"]
    assert source.read_text(encoding="utf-8") == (
        "package main\nconst s = 'it’s'\n// \"added\" 'single'\n" + padding
        + "+ b/not-a-header\n// \"tail\"\n// end\n"
    )
    assert (repo / "new.go").read_text() == "// \"new\" 'single'\n"
    assert (repo / "keep.py").read_text() == "x = 1\nprint('ascii')\n"
    assert (repo / "models_generated.go").read_text() == "// generated ”\n"
    assert (repo / "blob.bin").read_bytes() == b"\x00\x02"
    assert (repo / "outside.go").read_text() == "// “operator”\n"
    assert not (repo / "gone.go").exists()


def test_scrub_driver_atomic_write_leaves_original_intact_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "main.go"
    src.write_text("// not \u201d\n", encoding="utf-8")
    def _failing_replace(tmp: Any, dst: Any) -> None:
        raise OSError("No space left on device")
    monkeypatch.setattr(os, "replace", _failing_replace)
    assert scrub_smart_quotes_changed_files(tmp_path, ["main.go"]) == []
    assert src.read_text(encoding="utf-8") == "// not \u201d\n"
    # No sibling temp file is left behind by the failed atomic write.
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]

def test_scrub_driver_raises_git_error_on_unattributable_diffs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose GitError so the caller can warn and continue without unattributed edits."""
    repo = _baseline_repo(tmp_path / "failed", {"main.go": "x\n"})
    (repo / "main.go").write_text("// \u201d\n", encoding="utf-8")

    def _fail_diff(repo: Any, ref: Any, paths: Any) -> None:
        raise GitError("git diff failed")

    with monkeypatch.context() as failing:
        failing.setattr("daydream.quote_scrub.diff_worktree_against", _fail_diff)
        with pytest.raises(GitError):
            scrub_smart_quotes_changed_files(repo, ["main.go"], pre_fix_ref="HEAD")
    # Undecodable attribution is translated into GitError for the caller's fail-open guard.
    repo = _baseline_repo(tmp_path / "latin", {"latin.go": b"// caf\xe9\n"})
    (repo / "latin.go").write_bytes(b"// caf\xe9\n// \xe2\x80\x9cadded\xe2\x80\x9d\n")
    with pytest.raises(GitError):
        scrub_smart_quotes_changed_files(repo, ["latin.go"], pre_fix_ref="HEAD")
    # Unattributable external diff output is rejected without normalizing baseline quotes.
    repo = _baseline_repo(tmp_path / "external", {"main.go": "// baseline \u201d quote\n"})
    src = repo / "main.go"
    src.write_text("// baseline \u201d quote\n// added \u201cquote\u201d\n", encoding="utf-8")

    def _garbage_diff(repo: Any, ref: Any, paths: Any) -> str:
        return "raw external-driver output without unified-diff structure\n"

    monkeypatch.setattr("daydream.quote_scrub.diff_worktree_against", _garbage_diff)
    with pytest.raises(GitError):
        scrub_smart_quotes_changed_files(repo, ["main.go"], pre_fix_ref="HEAD")
    assert src.read_text(encoding="utf-8") == "// baseline \u201d quote\n// added \u201cquote\u201d\n"


def test_scrub_driver_skips_binary_diffs_and_preserves_symlinks(tmp_path: Path) -> None:
    """Binary diff output is valid: skip undecodable content and still scrub
    the untracked text sibling; a rewrite through a symlink keeps the link."""
    repo = _baseline_repo(tmp_path, {"blob.bin": b"\x00\x01", "real.go": "// baseline\n"},
                          symlinks={"link.go": "real.go"})
    blob = repo / "blob.bin"
    blob.write_bytes(b"\x00\x02")
    new_file = repo / "new.go"
    new_file.write_text("// \u201cnew\u201d\n", encoding="utf-8")
    target = repo / "real.go"
    target.write_text("// baseline\n// added \u201cquote\u201d\n", encoding="utf-8")
    assert scrub_smart_quotes_changed_files(repo, ["blob.bin", "new.go"], pre_fix_ref="HEAD") == ["new.go"]
    assert new_file.read_text(encoding="utf-8") == '// "new"\n'
    assert scrub_smart_quotes_changed_files(repo, ["link.go"], pre_fix_ref="HEAD") == ["link.go"]
    assert (repo / "link.go").is_symlink()
    assert target.read_text(encoding="utf-8") == '// baseline\n// added "quote"\n'
