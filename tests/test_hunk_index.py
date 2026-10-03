"""Shared hunk parsing preserves posting ranges, coverage totals, and added-line numbering."""
from __future__ import annotations

from pathlib import Path

import pytest

from daydream.hunk_index import (
    added_line_numbers,
    files_in_index,
    head_side_ranges,
    head_side_ranges_by_file,
    load_hunk_index,
    parse_hunks,
    write_hunk_index,
)


@pytest.mark.parametrize("quote_paths", ["true", "false"])
def test_native_git_paths_bind_framing_impact_and_inline_placement(tmp_path: Path, quote_paths: str) -> None:
    """Every consumer uses the actual Git names, including C-escaped controls."""
    import shutil

    from daydream.deep.diff import iter_diff_blocks
    from daydream.exploration_runner import count_changed_files
    from daydream.pr_review import ParsedIssue, PRInfo, classify
    from daydream.tree_sitter_index import detect_affected_files
    from tests.harness.git_helpers import commit, git, init_repo, write_and_stage

    repo = tmp_path / "repo"
    init_repo(repo)
    modified = ["plain.py", "space name.py", "café.py", "tab\tname.py", "line\nname.py",
                'quote"name.py', "back\\slash.py", "nested b/path.py"]
    mode_paths = ["mode file.py", 'quote" b/mode.py']
    binary_paths = ["binary file.dat", 'quote" b/binary.dat']
    for path in [*modified, *mode_paths, "rename source.py", "copy source.py"]:
        write_and_stage(repo, path, "VALUE = 1\nRESULT = VALUE\n")
    write_and_stage(repo, "deleted café.py", "DELETED_UNIQUE_VALUE = 7153\n")
    for path in binary_paths:
        write_and_stage(repo, path, b"\x00old binary")
    base = commit(repo, "base")
    for path in modified:
        (repo / path).write_text("VALUE = 1\nRESULT = VALUE + 1\n")
    for path in mode_paths:
        (repo / path).chmod(0o755)
    (repo / "deleted café.py").unlink()
    git(repo, "mv", "rename source.py", "renamed café.py")
    shutil.copy2(repo / "copy source.py", repo / "copied café.py")
    for path in binary_paths:
        (repo / path).write_bytes(b"\x00new binary")
    git(repo, "add", ".")
    head = commit(repo, "feature")
    options = ("--find-renames", "--find-copies-harder")
    diff = git(repo, "-c", f"core.quotepath={quote_paths}", "diff", *options, base, head)
    names = git(repo, "diff", *options, "--name-only", "-z", base, head).split("\0")[:-1]
    blocks = list(iter_diff_blocks(diff))
    assert [path for path, _ in blocks] == names
    deleted_block = dict(blocks)["deleted café.py"]
    assert "deleted file mode " in deleted_block and "rename to " not in deleted_block
    assert "".join(block for _, block in blocks) == diff
    assert count_changed_files(diff) == len(names)
    assert set(parse_hunks(diff)) == set(modified)
    affected = detect_affected_files(diff, repo)
    assert {item.path for item in affected if item.role == "modified"} == set(names)
    pr = PRInfo(number=1, head_sha=head, base_sha=base, base_ref="main", head_ref="feature",
                owner="o", repo="r", url="https://github.com/o/r/pull/1")
    findings = [ParsedIssue(path=path, line=2, title="Native changed line", body="The changed value.")
                for path in modified]
    placed = classify(repo, pr, findings, snapshot_diff=diff)
    assert [item.path for item in placed.inline] == modified
    assert all(item.line == 2 for item in placed.inline)
    assert not placed.file_level and not placed.body_only


@pytest.mark.parametrize("escape", [r"\400", r"\777"])
def test_quoted_git_path_rejects_unrepresentable_byte(escape: str) -> None:
    diff = f'diff --git a/x b/x\n--- a/x\n+++ "b/{escape}.py"\n@@ -1 +1 @@\n-old\n+new\n'
    with pytest.raises(ValueError, match="byte must be in range"):
        parse_hunks(diff)


def test_native_filename_containing_delete_marker_keeps_dependency_analysis(tmp_path: Path) -> None:
    from daydream.tree_sitter_index import detect_affected_files
    from tests.harness.git_helpers import commit, git, init_repo, write_and_stage

    repo = tmp_path / "repo"
    init_repo(repo)
    path = "deleted file mode.py"
    write_and_stage(repo, path, "import dependency\nVALUE = 1\n")
    write_and_stage(repo, "dependency.py", "VALUE = 1\n")
    base = commit(repo, "base")
    (repo / path).write_text("import dependency\nVALUE = 2\n")
    git(repo, "add", ".")
    head = commit(repo, "feature")
    affected = detect_affected_files(git(repo, "diff", base, head), repo)
    assert (path, "modified") in {(item.path, item.role) for item in affected}
    assert ("dependency.py", "imports") in {(item.path, item.role) for item in affected}


def test_native_deleted_filename_containing_hunk_marker_skips_local_imports(tmp_path: Path) -> None:
    from daydream.tree_sitter_index import detect_affected_files
    from tests.harness.git_helpers import commit, git, init_repo, write_and_stage

    repo = tmp_path / "repo"
    init_repo(repo)
    path = "@@ file.py"
    source = "import dependency\nVALUE = 1\n"
    write_and_stage(repo, path, source)
    write_and_stage(repo, "dependency.py", "VALUE = 1\n")
    base = commit(repo, "base")
    (repo / path).unlink()
    git(repo, "add", ".")
    head = commit(repo, "feature")
    diff = git(repo, "diff", base, head)
    # The diff's deletion marker controls analysis even when a local file exists.
    (repo / path).write_text(source)
    affected = detect_affected_files(diff, repo)
    assert {(item.path, item.role) for item in affected} == {(path, "modified")}


def test_parse_hunks_matches_pr_head_side_ranges() -> None:
    diff = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1,3 +10,5 @@\n old\n+new1\n+new2\n@@ -20 +30,2 @@\n+new3\n"
    )
    parsed = parse_hunks(diff)
    # Head-side ranges are inclusive: (new_start,new_start+count-1)
    assert head_side_ranges(parsed) == [(10, 14), (30, 31)]
    # coverage.hunk_change_line_count contract: total + lines, headers excluded
    assert sum(fi["added_total"] + fi["removed_total"] for fi in parsed.values()) == 3
    # quote_scrub._added_line_numbers contract: new-side numbers of '+' lines
    assert added_line_numbers(parsed) == {"x.py": {11, 12, 30}}

def test_write_hunk_index_round_trips(tmp_path: Path) -> None:
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n x\n+y\n"
    write_hunk_index(tmp_path, diff)
    idx = load_hunk_index(tmp_path)
    assert files_in_index(idx) == ["a.py"]
    assert idx["a.py"]["added_total"] == 1 and idx["a.py"]["removed_total"] == 0
    assert idx["a.py"]["hunks"][0]["new_end"] == 2
    assert "added_lines" not in idx["a.py"]

def test_head_side_ranges_by_file_groups_the_flat_view_per_path() -> None:
    """#1113: the per-file view answers "is this line in a changed hunk of THIS
    file", which the flattened ``head_side_ranges`` cannot."""
    diff = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1,3 +10,5 @@\n old\n+new1\n+new2\n@@ -20 +30,2 @@\n+new3\n"
        "diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n"
        "@@ -1 +1,2 @@\n z\n+w\n"
    )
    parsed = parse_hunks(diff)
    by_file = head_side_ranges_by_file(parsed)
    assert by_file == {"x.py": [(10, 14), (30, 31)], "y.py": [(1, 2)]}
    # Same ranges as the flat view, only grouped — no range invented or lost.
    flat = [r for ranges in by_file.values() for r in ranges]
    assert sorted(flat) == sorted(head_side_ranges(parsed))

def test_head_side_ranges_by_file_reads_the_persisted_index(tmp_path: Path) -> None:
    """#1113: persistence drops only ``added_lines``, so the per-file accessor
    works identically on a loaded ``hunk-index.json``."""
    diff = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n x\n+y\n")
    write_hunk_index(tmp_path, diff)
    loaded = load_hunk_index(tmp_path)
    assert head_side_ranges_by_file(loaded) == head_side_ranges_by_file(parse_hunks(diff))
    assert head_side_ranges_by_file(loaded) == {"a.py": [(1, 2)]}

def test_head_side_ranges_by_file_keeps_pure_deletion_files_as_empty() -> None:
    """#1113: a file whose only hunk was a pure deletion has no head-side range
    but is still a changed file, so it maps to ``[]`` rather than vanishing."""
    diff = ("diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ b/gone.py\n" "@@ -1,2 +0,0 @@\n-a\n-b\n")
    parsed = parse_hunks(diff)
    assert head_side_ranges_by_file(parsed) == {"gone.py": []}
