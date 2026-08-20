"""Unit tests for daydream.tree_sitter_index.detect_affected_files()."""

import inspect
from pathlib import Path
from typing import Any

import pytest
from conftest import _commit, _git, _make_repo_with_main

from daydream import git_ops
from daydream.tree_sitter_index import (
    _MAX_IMPORTERS,
    _is_generic_or_invalid_stem,
    detect_affected_files,
)

FIXTURES = Path(__file__).parent / "fixtures" / "diffs"


def _modified_diff(path: str) -> str:
    """Minimal unified diff marking *path* as modified."""
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1,2 @@\n"
        " x = 1\n"
        "+y = 2\n"
    )


def _importers(results) -> set[str]:
    return {r.path for r in results if r.role == "imported_by"}


def _materialize(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create files under tmp_path with the given relative paths and contents."""
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


def test_detect_affected_files_rows_are_static_provenance(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "widget.py").write_text("x = 1\n")
    (tmp_path / "app.py").write_text("import pkg.widget\n")
    results = detect_affected_files(_modified_diff("app.py"), tmp_path, depth=1)
    assert results, "static resolution must produce rows"
    assert all(f.provenance == "static" for f in results)


def test_python_impact_surface(tmp_path: Path):
    diff_text = (FIXTURES / "python_multifile.diff").read_text()
    repo = _materialize(
        tmp_path,
        {
            "daydream_demo/__init__.py": "",
            "daydream_demo/api.py": (
                '"""API module."""\nfrom .models import User\n\ndef get_user():\n    return User()\n'
            ),
            "daydream_demo/models.py": '"""Models module."""\n\nclass User:\n    pass\n',
        },
    )
    results = detect_affected_files(diff_text, repo, depth=1)
    paths_by_role = {(r.path, r.role) for r in results}
    assert ("daydream_demo/api.py", "modified") in paths_by_role
    assert ("daydream_demo/models.py", "modified") in paths_by_role
    # The api.py -> models.py forward edge must be exact.
    assert ("daydream_demo/models.py", "imports") in paths_by_role
    assert len(results) >= 2


_SHARED_FIXTURES: dict[str, str] = {
    "package/__init__.py": "",
    "package/models.py": '"""Models module."""\n\nclass User:\n    pass\n',
    "package/feature/__init__.py": "",
}


@pytest.mark.parametrize(
    "api_rel, files, expected_import_path",
    [
        # `from ..models` in package/feature/api.py resolves to the parent package.
        (
            "package/feature/api.py",
            _SHARED_FIXTURES
            | {
                "package/feature/api.py": (
                    '"""API module."""\nfrom ..models import User\n\ndef get_user():\n    return User()\n'
                ),
            },
            "package/models.py",
        ),
        # `from ...models` in package/feature/nested/api.py resolves to the grandparent package.
        (
            "package/feature/nested/api.py",
            _SHARED_FIXTURES
            | {
                "package/feature/nested/__init__.py": "",
                "package/feature/nested/api.py": (
                    '"""API module."""\nfrom ...models import User\n\ndef get_user():\n    return User()\n'
                ),
            },
            "package/models.py",
        ),
        # `from . import something` resolves to the current package's __init__.py
        # AND the sibling module package/feature/something.py, because the
        # capture now spans the whole statement, not just the relative import
        # node (the dot).
        (
            "package/feature/api.py",
            _SHARED_FIXTURES
            | {
                "package/feature/api.py": (
                    '"""API module."""\nfrom . import something\n\nthing = something.thing\n'
                ),
                "package/feature/something.py": "",
            },
            {"package/feature/__init__.py", "package/feature/something.py"},
        ),
        # `from ....something import name` with 4+ dots ascends to the great-grandparent package.
        (
            "package/feature/nested/deep/api.py",
            _SHARED_FIXTURES
            | {
                "package/feature/nested/__init__.py": "",
                "package/feature/nested/deep/__init__.py": "",
                "package/something.py": '"""Ancestor sibling module."""\n\nthing = 1\n',
                "package/feature/nested/deep/api.py": (
                    '"""API module."""\nfrom ....something import thing\n\nthing\n'
                ),
            },
            "package/something.py",
        ),
    ],
)
def test_python_multilevel_relative_imports(
    tmp_path: Path, api_rel: str, files: dict[str, str], expected_import_path: str | set[str]
):
    repo = _materialize(tmp_path, files)
    results = detect_affected_files(_modified_diff(api_rel), repo, depth=1)
    imports_pairs = {(r.path, r.role) for r in results if r.role == "imports"}
    if isinstance(expected_import_path, str):
        assert imports_pairs == {(expected_import_path, "imports")}
    else:
        assert imports_pairs == {(p, "imports") for p in expected_import_path}


@pytest.mark.parametrize(
    "api_rel, files, expected_import_paths",
    [
        # R2 regression pin: `from ..models` in package/feature/api.py resolves
        # to the parent package's models module. Already works on main after
        # #390; pinned so it cannot regress.
        (
            "package/feature/api.py",
            _SHARED_FIXTURES
            | {
                "package/feature/api.py": (
                    '"""API module."""\nfrom ..models import User\n\ndef get_user():\n    return User()\n'
                ),
            },
            {"package/models.py"},
        ),
        # R3 primary fix: `from .. import services` in package/feature/api.py
        # resolves to BOTH the parent package __init__.py AND the imported
        # sibling package's __init__.py — and must NOT include the importer's
        # own package/feature/__init__.py (C3).
        (
            "package/feature/api.py",
            _SHARED_FIXTURES
            | {
                "package/services/__init__.py": "",
                "package/feature/api.py": (
                    '"""API module."""\nfrom .. import services\n'
                ),
            },
            {"package/__init__.py", "package/services/__init__.py"},
        ),
    ],
)
def test_python_parent_relative_imports(
    tmp_path: Path, api_rel: str, files: dict[str, str], expected_import_paths: set[str]
):
    repo = _materialize(tmp_path, files)
    results = detect_affected_files(_modified_diff(api_rel), repo, depth=1)
    imports_paths = {r.path for r in results if r.role == "imports"}
    assert imports_paths == expected_import_paths


def test_typescript_impact_surface(tmp_path: Path):
    diff_text = (FIXTURES / "typescript_multifile.diff").read_text()
    repo = _materialize(
        tmp_path,
        {
            "src/api.ts": (
                '// API module\nimport { User } from "./models";\n\n'
                "export function getUser(): User {\n  return new User();\n}\n"
            ),
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
            "api.go": (
                'package main\n\nimport "example.com/m/models"\n\n'
                "func GetUser() *models.User {\n\treturn &models.User{}\n}\n"
            ),
            "models/user.go": "// user model\npackage models\ntype User struct{}\n",
        },
    )
    results = detect_affected_files(diff_text, repo, depth=1)
    assert any(r.path == "api.go" and r.role == "modified" for r in results)
    assert any(r.path == "models/user.go" and r.role == "modified" for r in results)
    assert len(results) >= 2


def test_go_imports_reuse_one_package_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _materialize(
        tmp_path,
        {
            "cmd/alpha.go": 'package main\n\nimport "example.com/models"\n',
            "cmd/beta.go": 'package main\n\nimport "example.com/helpers"\n',
            "models/user.go": "package models\n",
            "models/order.go": "package models\n",
            "helpers/format.go": "package helpers\n",
        },
    )
    real_rglob = Path.rglob
    calls = 0

    def one_go_traversal(self: Path, pattern: str) -> Any:
        nonlocal calls
        if self == repo and pattern == "*.go":
            calls += 1
            if calls > 1:
                raise AssertionError("repo_root.rglob('*.go') called more than once")
            return real_rglob(self, pattern)
        raise AssertionError(f"unexpected rglob: {pattern!r}")

    monkeypatch.setattr(Path, "rglob", one_go_traversal)

    results = detect_affected_files(
        _modified_diff("cmd/alpha.go") + _modified_diff("cmd/beta.go"), repo, depth=1
    )
    imports_paths = {r.path for r in results if r.role == "imports"}
    assert imports_paths == {"models/user.go", "models/order.go", "helpers/format.go"}
    assert {r.path for r in results if r.role == "modified"} == {"cmd/alpha.go", "cmd/beta.go"}


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


# --- Reverse-edge (importers) behavior: real git repo -----------------------


def test_reverse_edge_finds_code_importer(tmp_path: Path):
    repo = _make_repo_with_main(tmp_path)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "widget.py").write_text("x = 1\ny = 2\n")
    (repo / "caller.py").write_text("from pkg.widget import W\n\nW()\n")
    _git(repo, "add", "pkg/widget.py", "caller.py")
    _commit(repo, "add widget + caller")

    results = detect_affected_files(_modified_diff("pkg/widget.py"), repo, depth=1)
    assert "caller.py" in _importers(results)


def test_reverse_edge_skips_generic_stem(tmp_path: Path):
    # "app" is a generic stem: a bare grep would match unrelated prose/code.
    repo = _make_repo_with_main(tmp_path)
    (repo / "app.py").write_text("x = 1\ny = 2\n")
    (repo / "unrelated.py").write_text("# the app starts here\nrun_app()\n")
    _git(repo, "add", "app.py", "unrelated.py")
    _commit(repo, "add app + unrelated")

    results = detect_affected_files(_modified_diff("app.py"), repo, depth=1)
    assert _importers(results) == set()


def test_reverse_edge_excludes_non_code_files(tmp_path: Path):
    # A markdown/doc file cannot import a code module; it must never be an importer.
    repo = _make_repo_with_main(tmp_path)
    (repo / "widget.py").write_text("x = 1\ny = 2\n")
    (repo / "notes.md").write_text("The widget is documented here.\n")
    (repo / "caller.py").write_text("import widget\n")
    _git(repo, "add", "widget.py", "notes.md", "caller.py")
    _commit(repo, "add code + docs")

    importers = _importers(detect_affected_files(_modified_diff("widget.py"), repo, depth=1))
    assert "caller.py" in importers
    assert "notes.md" not in importers


def test_reverse_edge_capped_at_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _make_repo_with_main(tmp_path)
    for stem in ("widget", "gadget"):
        (repo / f"{stem}.py").write_text("x = 1\ny = 2\n")
    overshoot = _MAX_IMPORTERS + 5
    for stem in ("widget", "gadget"):
        for i in range(overshoot):
            (repo / f"{stem}_importer_{i}.py").write_text(f"import {stem}\n")
    _git(repo, "add", "widget.py", "gadget.py",
         *[f"{stem}_importer_{i}.py" for stem in ("widget", "gadget") for i in range(overshoot)])
    _commit(repo, "many importers")

    real_grep_fixed = git_ops.grep_fixed_matches
    calls = 0

    def one_batch(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("grep_fixed_matches called more than once")
        return real_grep_fixed(*args, **kwargs)

    def no_legacy_grep(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("legacy git_ops.grep must not be called")

    monkeypatch.setattr(git_ops, "grep_fixed_matches", one_batch)
    monkeypatch.setattr(git_ops, "grep", no_legacy_grep)

    results = detect_affected_files(
        _modified_diff("widget.py") + _modified_diff("gadget.py"), repo, depth=1
    )
    importers = _importers(results)
    widget = {p for p in importers if p.startswith("widget_importer_")}
    gadget = {p for p in importers if p.startswith("gadget_importer_")}
    assert len(widget) == _MAX_IMPORTERS
    assert len(gadget) == _MAX_IMPORTERS
    assert len(importers) == _MAX_IMPORTERS * 2
    assert {r.path for r in results if r.role == "modified"} == {"widget.py", "gadget.py"}


@pytest.mark.parametrize(
    "stem",
    ["", "__init__", "mod", "index", "main", "app", "utils", "config", "tests", "conftest"],
)
def test_is_generic_or_invalid_stem_skips_empty_and_generic(stem: str) -> None:
    assert _is_generic_or_invalid_stem(stem) is True


@pytest.mark.parametrize("stem", ["a\x00b", "a\rb", "a\nb"])
def test_is_generic_or_invalid_stem_skips_nul_cr_lf(stem: str) -> None:
    assert _is_generic_or_invalid_stem(stem) is True


@pytest.mark.parametrize("stem", ["widget", "gadget", "indexer", "app_store"])
def test_is_generic_or_invalid_stem_accepts_specific_stems(stem: str) -> None:
    assert _is_generic_or_invalid_stem(stem) is False
