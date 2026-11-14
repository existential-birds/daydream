"""Review Merge And Verifier."""
from __future__ import annotations

from daydream.deep.diff import _diff_changed_files, bound_deep_diff


def test_diff_changed_files_handles_modify_add_delete_binary() -> None:
    """Non-rename diff shapes emit exactly one path each."""
    mixed = (
        "diff --git a/keep.py b/keep.py\n"
        "--- a/keep.py\n"
        "+++ b/keep.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
        "diff --git a/old.py b/old.py\n"
        "deleted file mode 100644\n"
        "--- a/old.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-x = 1\n"
        "diff --git a/logo.png b/logo.png\n"
        "index 1234..5678 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    mixed += (
        'diff --git "a/removed\\tfile.py" "b/removed\\tfile.py"\n'
        'deleted file mode 100644\n--- "a/removed\\tfile.py"\n+++ /dev/null\n'
        '@@ -1 +0,0 @@\n-x = 1\n'
        'diff --git "a/logo\\"quoted\\".png" "b/logo\\"quoted\\".png"\n'
        'index 1234..5678 100644\nBinary files differ\n'
    )
    assert _diff_changed_files(mixed) == ["keep.py", "new.py", "old.py", "logo.png",
                                        "removed\tfile.py", 'logo"quoted".png']
    # Bounding keeps whole byte-identical blocks: an oversized leading block survives, later ones drop by name.
    big = "diff --git a/0big.py b/0big.py\n--- a/0big.py\n+++ b/0big.py\n@@ -0,0 +1 @@\n+" + "x" * 200 + "\n"
    bounded, info = bound_deep_diff(big + mixed, budget=100)
    assert info.marker is not None and bounded == info.marker + big
    assert "dropped: keep.py, new.py, old.py, logo.png" in info.marker
    assert (info.retained_blocks, info.oversize_paths) == (1, ["0big.py"])


