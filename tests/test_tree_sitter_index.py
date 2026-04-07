"""Test scaffolding for daydream.tree_sitter_index.detect_affected_files().

Wave 0 placeholder: the module does not exist yet. The entire test module is
skipped via importorskip so pytest collection stays green. Wave 1 will create
daydream/tree_sitter_index.py, at which point importorskip becomes a no-op and
the xfail markers below must be removed as the real implementations come online.
"""

from pathlib import Path

import pytest

pytest.importorskip("daydream.tree_sitter_index")

# The import above means the rest of this file only runs once the module exists.
# When that happens, Wave 1 should delete the xfail markers one by one as each
# case is implemented.
from daydream.tree_sitter_index import detect_affected_files  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "diffs"


@pytest.mark.xfail(reason="Wave 1: detect_affected_files() not implemented yet", strict=True)
def test_python_impact_surface():
    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    results = detect_affected_files(diff_text, Path("/tmp/repo"), depth=1)
    assert len(results) >= 2


@pytest.mark.xfail(reason="Wave 1: detect_affected_files() not implemented yet", strict=True)
def test_typescript_impact_surface():
    diff_text = (FIXTURES / "typescript_multifile.diff").read_text()
    results = detect_affected_files(diff_text, Path("/tmp/repo"), depth=1)
    assert len(results) >= 2


@pytest.mark.xfail(reason="Wave 1: detect_affected_files() not implemented yet", strict=True)
def test_go_impact_surface():
    diff_text = (FIXTURES / "go_multifile.diff").read_text()
    results = detect_affected_files(diff_text, Path("/tmp/repo"), depth=1)
    assert len(results) >= 2


@pytest.mark.xfail(reason="Wave 1: detect_affected_files() not implemented yet", strict=True)
def test_rust_impact_surface():
    diff_text = (FIXTURES / "rust_multifile.diff").read_text()
    results = detect_affected_files(diff_text, Path("/tmp/repo"), depth=1)
    assert len(results) >= 2


@pytest.mark.xfail(reason="Wave 1: detect_affected_files() not implemented yet", strict=True)
def test_default_depth_is_one():
    import inspect

    sig = inspect.signature(detect_affected_files)
    assert sig.parameters["depth"].default == 1


@pytest.mark.xfail(reason="Wave 1: detect_affected_files() not implemented yet", strict=True)
def test_unsupported_language_gets_modified_role():
    diff_text = (
        "diff --git a/lib/foo.rb b/lib/foo.rb\n"
        "index 1111111..2222222 100644\n"
        "--- a/lib/foo.rb\n"
        "+++ b/lib/foo.rb\n"
        "@@ -1,1 +1,2 @@\n"
        " class Foo\n"
        "+  def bar; end\n"
    )
    results = detect_affected_files(diff_text, Path("/tmp/repo"), depth=1)
    assert len(results) == 1
    assert results[0].role == "modified"


@pytest.mark.xfail(reason="Wave 1: detect_affected_files() not implemented yet", strict=True)
def test_deleted_file_does_not_raise_filenotfound():
    diff_text = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "index 1111111..0000000\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-print('bye')\n"
    )
    results = detect_affected_files(diff_text, Path("/tmp/repo"), depth=1)
    assert len(results) == 1
    assert results[0].role == "modified"
