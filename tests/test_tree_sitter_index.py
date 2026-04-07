"""Unit tests for daydream.tree_sitter_index.detect_affected_files()."""

import inspect
from pathlib import Path

from daydream.tree_sitter_index import detect_affected_files

FIXTURES = Path(__file__).parent / "fixtures" / "diffs"


def _materialize(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create files under tmp_path with the given relative paths and contents."""
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


def test_python_impact_surface(tmp_path: Path):
    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    repo = _materialize(
        tmp_path,
        {
            "daydream_demo/__init__.py": "",
            "daydream_demo/api.py": '"""API module."""\nfrom .models import User\n\ndef get_user():\n    return User()\n',
            "daydream_demo/models.py": '"""Models module."""\n\nclass User:\n    pass\n',
        },
    )
    results = detect_affected_files(diff_text, repo, depth=1)
    paths_by_role = {(r.path, r.role) for r in results}
    assert ("daydream_demo/api.py", "modified") in paths_by_role
    assert ("daydream_demo/models.py", "modified") in paths_by_role
    # The api.py -> models.py edge must be visible somewhere.
    assert any(r.role == "imports" and r.path.endswith("models.py") for r in results) or any(
        r.role == "imported_by" for r in results
    )
    assert len(results) >= 2


def test_typescript_impact_surface(tmp_path: Path):
    diff_text = (FIXTURES / "typescript_multifile.diff").read_text()
    repo = _materialize(
        tmp_path,
        {
            "src/api.ts": '// API module\nimport { User } from "./models";\n\nexport function getUser(): User {\n  return new User();\n}\n',
            "src/models.ts": "// Models module\n\nexport class User {}\n",
        },
    )
    results = detect_affected_files(diff_text, repo, depth=1)
    assert any(r.path == "src/api.ts" and r.role == "modified" for r in results)
    assert any(r.path == "src/models.ts" and r.role == "modified" for r in results)
    assert any(r.role == "imports" and r.path.endswith("models.ts") for r in results) or any(
        r.role == "imported_by" for r in results
    )
    assert len(results) >= 2


def test_go_impact_surface(tmp_path: Path):
    diff_text = (FIXTURES / "go_multifile.diff").read_text()
    repo = _materialize(
        tmp_path,
        {
            "api.go": 'package main\n\nimport "example.com/m/models"\n\nfunc GetUser() *models.User {\n\treturn &models.User{}\n}\n',
            "models/user.go": "// user model\npackage models\ntype User struct{}\n",
        },
    )
    results = detect_affected_files(diff_text, repo, depth=1)
    assert any(r.path == "api.go" and r.role == "modified" for r in results)
    assert any(r.path == "models/user.go" and r.role == "modified" for r in results)
    assert len(results) >= 2


def test_rust_impact_surface(tmp_path: Path):
    diff_text = (FIXTURES / "rust_multifile.diff").read_text()
    repo = _materialize(
        tmp_path,
        {
            "src/api.rs": "// api module\nuse crate::models::User;\n\npub fn get_user() -> User {\n    User\n}\n",
            "src/models.rs": "// models module\n\npub struct User;\n",
        },
    )
    results = detect_affected_files(diff_text, repo, depth=1)
    assert any(r.path == "src/api.rs" and r.role == "modified" for r in results)
    assert any(r.path == "src/models.rs" and r.role == "modified" for r in results)
    assert any(r.role == "imports" and r.path.endswith("models.rs") for r in results) or any(
        r.role == "imported_by" for r in results
    )
    assert len(results) >= 2


def test_default_depth_is_one():
    sig = inspect.signature(detect_affected_files)
    assert sig.parameters["depth"].default == 1


def test_unsupported_language_gets_modified_role(tmp_path: Path):
    diff_text = (
        "diff --git a/lib/foo.rb b/lib/foo.rb\n"
        "index 1111111..2222222 100644\n"
        "--- a/lib/foo.rb\n"
        "+++ b/lib/foo.rb\n"
        "@@ -1,1 +1,2 @@\n"
        " class Foo\n"
        "+  def bar; end\n"
    )
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "foo.rb").write_text("class Foo\n  def bar; end\nend\n")
    results = detect_affected_files(diff_text, tmp_path, depth=1)
    assert len(results) == 1
    assert results[0].path == "lib/foo.rb"
    assert results[0].role == "modified"


def test_deleted_file_does_not_raise_filenotfound(tmp_path: Path):
    diff_text = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "index 1111111..0000000\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-print('bye')\n"
    )
    results = detect_affected_files(diff_text, tmp_path, depth=1)
    assert len(results) == 1
    assert results[0].path == "gone.py"
    assert results[0].role == "modified"
