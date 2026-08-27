from pathlib import Path
from typing import Any

import pytest

from daydream.git_ops import GitError
from daydream.quote_scrub import (
    _added_line_numbers,
    normalize_smart_quotes,
    scrub_smart_quotes_changed_files,
)
from tests.harness.git_helpers import git, init_repo


def test_normalize_smart_quotes_maps_all_four_code_points_to_ascii() -> None:
    assert normalize_smart_quotes("\u201cleft\u201d \u2018single\u2019") == "\"left\" 'single'"
    assert normalize_smart_quotes("not \u201d") == 'not "'


def test_normalize_smart_quotes_identity_on_ascii() -> None:
    # Legitimate ASCII code/strings are byte-identical.
    assert normalize_smart_quotes('fmt.Println("x")') == 'fmt.Println("x")'
    # The empty-string literal in a Go comment is already ASCII — untouched.
    assert normalize_smart_quotes("// not ''") == "// not ''"


def test_scrub_driver_rewrites_and_reports_changed_files(tmp_path: Path) -> None:
    (tmp_path / "main.go").write_text("package main\n\n// not \u201d\n")
    (tmp_path / "keep.py").write_text("x = 1\n")
    scrubbed = scrub_smart_quotes_changed_files(tmp_path, ["main.go", "keep.py"])
    assert scrubbed == ["main.go"]
    assert (tmp_path / "main.go").read_text() == 'package main\n\n// not "\n'
    assert (tmp_path / "keep.py").read_text() == "x = 1\n"


def test_scrub_driver_skips_generated_binary_and_out_of_scope(tmp_path: Path) -> None:
    gen = tmp_path / "models_generated.go"
    gen.write_text("// generated \u201d\n")
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\x00\xff\xfe smart \x80")  # invalid UTF-8
    outside = tmp_path / "outside.txt"
    outside.write_text("\u201doutside\u201d")
    assert scrub_smart_quotes_changed_files(tmp_path, ["models_generated.go", "blob.bin"]) == []
    assert gen.read_text() == "// generated \u201d\n"
    assert binary.read_bytes() == b"\x00\xff\xfe smart \x80"
    assert outside.read_text() == "\u201doutside\u201d"  # not in changed set → untouched


def test_scrub_driver_skips_missing_file(tmp_path: Path) -> None:
    assert scrub_smart_quotes_changed_files(tmp_path, ["gone.go"]) == []


def test_scrub_driver_normalizes_only_added_lines(tmp_path: Path) -> None:
    """Real git repo: pre-existing baseline smart quotes stay untouched; only
    lines the fix pass added are normalized (issue #687 finding 1)."""
    repo = tmp_path / "repo"
    init_repo(repo)
    src = repo / "main.go"
    src.write_text("package main\n\n// baseline \u201d quote\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    # The fix pass adds a smart quote on a new line only.
    src.write_text(
        "package main\n\n// baseline \u201d quote\n// added \u201cquote\u201d\n",
        encoding="utf-8",
    )
    scrubbed = scrub_smart_quotes_changed_files(repo, ["main.go"], pre_fix_ref="HEAD")
    assert scrubbed == ["main.go"]
    assert src.read_text(encoding="utf-8") == ('package main\n\n// baseline \u201d quote\n// added "quote"\n')


def test_scrub_driver_never_rewrites_baseline_smart_quotes_in_literals(tmp_path: Path) -> None:
    """A baseline single-quoted literal containing U+2019 (which whole-file
    normalization would turn into a syntax error) survives byte-identical; an
    added line is still scrubbed."""
    repo = tmp_path / "repo"
    init_repo(repo)
    src = repo / "main.go"
    src.write_text("package main\n\nconst s = 'it\u2019s'\n\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    src.write_text(
        "package main\n\nconst s = 'it\u2019s'\n\n// \u201cadded\u201d\n",
        encoding="utf-8",
    )
    scrubbed = scrub_smart_quotes_changed_files(repo, ["main.go"], pre_fix_ref="HEAD")
    assert scrubbed == ["main.go"]
    assert src.read_text(encoding="utf-8") == ("package main\n\nconst s = 'it\u2019s'\n\n// \"added\"\n")


def test_scrub_driver_guards_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write failure (read-only fs, ENOSPC) skips the file instead of
    propagating an OSError past the caller's GitError-only guard (findings 2/3)."""
    src = tmp_path / "main.go"
    src.write_text("// not \u201d\n", encoding="utf-8")

    def _fail_write(path: Any, data: Any) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr("daydream.quote_scrub._atomic_write_bytes", _fail_write)
    assert scrub_smart_quotes_changed_files(tmp_path, ["main.go"]) == []
    assert src.read_text(encoding="utf-8") == "// not \u201d\n"


def test_scrub_driver_atomic_write_leaves_original_intact_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-write failure must not truncate the source file: the original
    bytes survive and no temp files are left behind (finding 3)."""
    import os

    src = tmp_path / "main.go"
    src.write_text("// not \u201d\n", encoding="utf-8")

    def _failing_replace(tmp: Any, dst: Any) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(os, "replace", _failing_replace)
    assert scrub_smart_quotes_changed_files(tmp_path, ["main.go"]) == []
    assert src.read_text(encoding="utf-8") == "// not \u201d\n"
    # No sibling temp file is left behind by the failed atomic write.
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_scrub_driver_raises_git_error_on_attribution_diff_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the attribution diff cannot be computed the driver raises the
    documented GitError (finding 3): the orchestrator's fail-open guard turns
    that into a warning and continues — never an abort."""
    repo = tmp_path / "repo"
    init_repo(repo)
    src = repo / "main.go"
    src.write_text("x\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    src.write_text("// \u201d\n", encoding="utf-8")

    def _fail_diff(repo: Any, ref: Any, paths: Any) -> None:
        raise GitError("git diff failed")

    monkeypatch.setattr("daydream.quote_scrub.diff_worktree_against", _fail_diff)
    with pytest.raises(GitError):
        scrub_smart_quotes_changed_files(repo, ["main.go"], pre_fix_ref="HEAD")


def test_scrub_driver_with_non_utf8_diff_content_raises_git_error(tmp_path: Path) -> None:
    """A changed file whose content is not valid UTF-8 (latin-1) makes the
    attribution diff undecodable; the driver must raise the documented GitError
    (degrading to the caller's warn-and-continue) instead of crashing with a
    raw UnicodeDecodeError (finding 1, F1)."""
    repo = tmp_path / "repo"
    init_repo(repo)
    src = repo / "latin.go"
    src.write_bytes(b"// caf\xe9\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    src.write_bytes(b"// caf\xe9\n// \xe2\x80\x9cadded\xe2\x80\x9d\n")
    with pytest.raises(GitError):
        scrub_smart_quotes_changed_files(repo, ["latin.go"], pre_fix_ref="HEAD")


def test_scrub_driver_with_quotepath_non_ascii_path_preserves_attribution(tmp_path: Path) -> None:
    """git's default core.quotepath quotes non-ASCII paths in diff output
    (``+++ \"b/caf\\303\\251.go\"``); the added-line parser must unquote them so
    the file stays line-targeted instead of falling through to whole-file
    normalization, which would rewrite baseline quotes (finding 1, F2)."""
    repo = tmp_path / "repo"
    init_repo(repo)
    src = repo / "caf\u00e9.go"
    src.write_text("package main\n\n// baseline \u201d quote\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    src.write_text(
        "package main\n\n// baseline \u201d quote\n// added \u201cquote\u201d\n",
        encoding="utf-8",
    )
    scrubbed = scrub_smart_quotes_changed_files(repo, ["caf\u00e9.go"], pre_fix_ref="HEAD")
    assert scrubbed == ["caf\u00e9.go"]
    assert src.read_text(encoding="utf-8") == (
        'package main\n\n// baseline \u201d quote\n// added "quote"\n'
    )


def test_scrub_driver_normalizes_untracked_new_file_in_full(tmp_path: Path) -> None:
    """An untracked new file is absent from the attribution diff, so every line
    is agent-authored and normalized in full (finding 2) — the whole-file
    branch for ``added is None`` on a per-path lookup."""
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / "base.go").write_text("x\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    new_file = repo / "new.go"
    new_file.write_text("// \u201cnew file\u201d\n", encoding="utf-8")
    scrubbed = scrub_smart_quotes_changed_files(repo, ["new.go"], pre_fix_ref="HEAD")
    assert scrubbed == ["new.go"]
    assert new_file.read_text(encoding="utf-8") == '// "new file"\n'


def test_added_line_numbers_parses_unified_diff() -> None:
    diff = (
        "diff --git a/main.go b/main.go\n"
        "index 422c2b7..63da0a2 100644\n"
        "--- a/main.go\n"
        "+++ b/main.go\n"
        "@@ -1,2 +1,3 @@\n"
        " package main\n"
        " \n"
        "+// \u201cquote\u201d\n"
        "@@ -10,0 +12,2 @@\n"
        "+\n"
        "+b\n"
        "--- a/other.py\n"
        "+++ b/other.py\n"
        "@@ -5 +5 @@\n"
        "-old\n"
        "+new\n"
    )
    assert _added_line_numbers(diff) == {"main.go": {3, 12, 13}, "other.py": {5}}


def test_added_line_numbers_added_line_looking_like_header_is_content(tmp_path: Path) -> None:
    """An added line whose content starts with ``++ b/`` renders as ``+++ b/...``
    and must be parsed as an added line, not a file header re-keying the current
    file (findings 4/6): later added smart quotes stay attributed to the real
    path."""
    diff = (
        "--- a/main.go\n"
        "+++ b/main.go\n"
        "@@ -1,2 +1,4 @@\n"
        " package main\n"
        " \n"
        "++ b/not-a-header\n"
        "+// \u201cquote\u201d\n"
        "--- a/other.go\n"
        "+++ b/other.go\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert _added_line_numbers(diff) == {"main.go": {3, 4}, "other.go": {1}}


def test_added_line_numbers_noprefix_and_space_paths() -> None:
    """diff.noprefix=true drops the a//b/ prefixes (``+++ main.go``), and git
    appends a trailing tab after space-containing paths (``+++ b/has space.py\t``);
    both must still key the real repo-relative path (finding 5, finding 2 tab
    separator)."""
    diff = (
        "diff --git main.go main.go\n"
        "--- main.go\n"
        "+++ main.go\n"
        "@@ -1 +1,2 @@\n"
        " package main\n"
        "+// \u201cquote\u201d\n"
        "diff --git a/has space.py b/has space.py\n"
        "--- a/has space.py\t\n"
        "+++ b/has space.py\t\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
    )
    assert _added_line_numbers(diff) == {"main.go": {2}, "has space.py": {1}}


def test_added_line_numbers_quoted_path_with_space() -> None:
    """A quotepath-quoted path that also contains a space gets the trailing tab
    separator after the closing quote (``+++ \"b/na\\303\\257ve ve.py\"\t``); the
    unquoted key must match the real path (finding 2)."""
    diff = (
        "--- \"a/na\\303\\257ve ve.py\"\n"
        "+++ \"b/na\\303\\257ve ve.py\"\t\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
    )
    assert _added_line_numbers(diff) == {"na\u00efve ve.py": {1}}


def test_added_line_numbers_delegates_to_shared_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    import daydream.hunk_index as hunk_index

    calls = {"n": 0}
    real = hunk_index.parse_hunks

    def spy(diff_text: Any) -> Any:
        calls["n"] += 1
        return real(diff_text)

    monkeypatch.setattr(hunk_index, "parse_hunks", spy)
    diff = (
        "diff --git a/main.go b/main.go\n--- a/main.go\n+++ b/main.go\n"
        "@@ -1,2 +1,4 @@\n package main\n \n+// added quote\n"
    )
    added = _added_line_numbers(diff)
    assert added == {"main.go": {3}}  # the added line is new-file line 3
    assert calls["n"] == 1



def test_scrub_driver_raises_git_error_on_external_driver_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-unified attribution diff (external diff driver output) cannot be
    attributed; the driver raises the documented GitError so the caller's
    fail-open guard skips instead of silently whole-file normalizing baseline
    smart quotes in tracked files (finding 5)."""
    repo = tmp_path / "repo"
    init_repo(repo)
    src = repo / "main.go"
    src.write_text("// baseline \u201d quote\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    src.write_text("// baseline \u201d quote\n// added \u201cquote\u201d\n", encoding="utf-8")

    def _garbage_diff(repo: Any, ref: Any, paths: Any) -> str:
        return "raw external-driver output without unified-diff structure\n"

    monkeypatch.setattr("daydream.quote_scrub.diff_worktree_against", _garbage_diff)
    with pytest.raises(GitError):
        scrub_smart_quotes_changed_files(repo, ["main.go"], pre_fix_ref="HEAD")
    # Fail-open: nothing was rewritten.
    assert src.read_text(encoding="utf-8") == "// baseline \u201d quote\n// added \u201cquote\u201d\n"


def test_scrub_driver_binary_only_diff_does_not_raise(tmp_path: Path) -> None:
    """A binary-only diff (``Binary files ... differ``) is legitimate git
    output: the binary file is skipped as undecodable and an untracked sibling
    still normalizes whole-file, without tripping the external-driver check
    (finding 5)."""
    repo = tmp_path / "repo"
    init_repo(repo)
    blob = repo / "blob.bin"
    blob.write_bytes(b"\x00\x01")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    blob.write_bytes(b"\x00\x02")
    new_file = repo / "new.go"
    new_file.write_text("// \u201cnew\u201d\n", encoding="utf-8")
    scrubbed = scrub_smart_quotes_changed_files(
        repo, ["blob.bin", "new.go"], pre_fix_ref="HEAD",
    )
    assert scrubbed == ["new.go"]
    assert new_file.read_text(encoding="utf-8") == '// "new"\n'


def test_scrub_driver_preserves_symlink_on_rewrite(tmp_path: Path) -> None:
    """A tracked symlink must survive the scrub: the normalized bytes land in
    the link target, and the directory entry stays a symlink (finding 7)."""
    repo = tmp_path / "repo"
    init_repo(repo)
    target = repo / "real.go"
    target.write_text("// baseline\n", encoding="utf-8")
    link = repo / "link.go"
    link.symlink_to(target.name)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    target.write_text("// baseline\n// added \u201cquote\u201d\n", encoding="utf-8")
    scrubbed = scrub_smart_quotes_changed_files(repo, ["link.go"], pre_fix_ref="HEAD")
    assert scrubbed == ["link.go"]
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == '// baseline\n// added "quote"\n'
