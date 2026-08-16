from daydream.quote_scrub import (
    _added_line_numbers,
    normalize_smart_quotes,
    scrub_smart_quotes_changed_files,
)
from tests.harness.git_helpers import git, init_repo


def test_normalize_smart_quotes_maps_all_four_code_points_to_ascii():
    assert normalize_smart_quotes("\u201cleft\u201d \u2018single\u2019") == "\"left\" 'single'"
    assert normalize_smart_quotes("not \u201d") == 'not "'


def test_normalize_smart_quotes_identity_on_ascii():
    # Legitimate ASCII code/strings are byte-identical.
    assert normalize_smart_quotes('fmt.Println("x")') == 'fmt.Println("x")'
    # The empty-string literal in a Go comment is already ASCII — untouched.
    assert normalize_smart_quotes("// not ''") == "// not ''"


def test_scrub_driver_rewrites_and_reports_changed_files(tmp_path):
    (tmp_path / "main.go").write_text("package main\n\n// not \u201d\n")
    (tmp_path / "keep.py").write_text("x = 1\n")
    scrubbed = scrub_smart_quotes_changed_files(tmp_path, ["main.go", "keep.py"])
    assert scrubbed == ["main.go"]
    assert (tmp_path / "main.go").read_text() == 'package main\n\n// not "\n'
    assert (tmp_path / "keep.py").read_text() == "x = 1\n"


def test_scrub_driver_skips_generated_binary_and_out_of_scope(tmp_path):
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


def test_scrub_driver_skips_missing_file(tmp_path):
    assert scrub_smart_quotes_changed_files(tmp_path, ["gone.go"]) == []


def test_scrub_driver_normalizes_only_added_lines(tmp_path):
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


def test_scrub_driver_never_rewrites_baseline_smart_quotes_in_literals(tmp_path):
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


def test_scrub_driver_guards_write_failure(tmp_path, monkeypatch):
    """A write failure (read-only fs, ENOSPC) skips the file instead of
    propagating an OSError past the caller's GitError-only guard (findings 2/3)."""
    src = tmp_path / "main.go"
    src.write_text("// not \u201d\n", encoding="utf-8")

    def _fail_write(path, data):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("daydream.quote_scrub._atomic_write_bytes", _fail_write)
    assert scrub_smart_quotes_changed_files(tmp_path, ["main.go"]) == []
    assert src.read_text(encoding="utf-8") == "// not \u201d\n"


def test_scrub_driver_atomic_write_leaves_original_intact_on_failure(tmp_path, monkeypatch):
    """A mid-write failure must not truncate the source file: the original
    bytes survive and no temp files are left behind (finding 3)."""
    import os

    src = tmp_path / "main.go"
    src.write_text("// not \u201d\n", encoding="utf-8")

    def _failing_replace(tmp, dst):
        raise OSError("No space left on device")

    monkeypatch.setattr(os, "replace", _failing_replace)
    assert scrub_smart_quotes_changed_files(tmp_path, ["main.go"]) == []
    assert src.read_text(encoding="utf-8") == "// not \u201d\n"
    # No sibling temp file is left behind by the failed atomic write.
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_added_line_numbers_parses_unified_diff():
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
