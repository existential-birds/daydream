# tests/rlm/test_environment.py
"""Tests for REPL environment data structures."""

from daydream.rlm.environment import (
    FileInfo,
    Service,
    RepoContext,
)


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_file_info_creation(self):
        """FileInfo should store parsed file metadata."""
        info = FileInfo(
            language="python",
            functions=["main", "helper"],
            classes=["MyClass"],
            imports=["os", "sys"],
            exports=[],
        )
        assert info.language == "python"
        assert info.functions == ["main", "helper"]
        assert info.classes == ["MyClass"]
        assert info.imports == ["os", "sys"]
        assert info.exports == []


class TestService:
    """Tests for Service dataclass."""

    def test_service_creation(self):
        """Service should store service boundary metadata."""
        svc = Service(
            name="auth",
            root="services/auth",
            files=["services/auth/main.py", "services/auth/utils.py"],
            dependencies=["db", "cache"],
        )
        assert svc.name == "auth"
        assert svc.root == "services/auth"
        assert len(svc.files) == 2
        assert svc.dependencies == ["db", "cache"]


class TestRepoContext:
    """Tests for RepoContext dataclass."""

    def test_repo_context_creation(self):
        """RepoContext should hold all codebase data."""
        ctx = RepoContext(
            files={"main.py": "print('hello')"},
            structure={"main.py": FileInfo("python", ["main"], [], [], [])},
            services={},
            file_sizes={"main.py": 10},
            total_tokens=10,
            file_count=1,
            largest_files=[("main.py", 10)],
            languages=["python"],
            changed_files=None,
        )
        assert ctx.file_count == 1
        assert ctx.total_tokens == 10
        assert "main.py" in ctx.files

    def test_repo_context_with_changed_files(self):
        """RepoContext should support PR mode with changed_files."""
        ctx = RepoContext(
            files={},
            structure={},
            services={},
            file_sizes={},
            total_tokens=0,
            file_count=0,
            largest_files=[],
            languages=[],
            changed_files=["src/api.py", "tests/test_api.py"],
        )
        assert ctx.changed_files == ["src/api.py", "tests/test_api.py"]
