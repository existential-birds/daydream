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


class TestBuildReplNamespace:
    """Tests for build_repl_namespace function."""

    def test_namespace_contains_repo(self):
        """Namespace should contain repo context."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={"a.py": "x=1"},
            structure={},
            services={},
            file_sizes={"a.py": 5},
            total_tokens=5,
            file_count=1,
            largest_files=[("a.py", 5)],
            languages=["python"],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "response")
        assert "repo" in ns
        assert ns["repo"].files == {"a.py": "x=1"}

    def test_namespace_contains_llm_query(self):
        """Namespace should contain llm_query function."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        calls = []

        def mock_llm(prompt: str, model: str = "haiku") -> str:
            calls.append((prompt, model))
            return "mocked"

        ns = build_repl_namespace(ctx, llm_query_fn=mock_llm)
        result = ns["llm_query"]("test prompt")
        assert result == "mocked"
        assert calls == [("test prompt", "haiku")]

    def test_llm_query_handles_context_kwarg(self):
        """llm_query should handle hallucinated 'context' parameter gracefully."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        calls = []

        def mock_llm(prompt: str, model: str = "haiku") -> str:
            calls.append((prompt, model))
            return "analyzed"

        ns = build_repl_namespace(ctx, llm_query_fn=mock_llm)
        # Model often hallucinates a 'context' parameter - should merge into prompt
        result = ns["llm_query"]("Analyze this code", context="def foo(): pass")
        assert result == "analyzed"
        assert len(calls) == 1
        assert calls[0][0] == "Analyze this code\n\ndef foo(): pass"
        assert calls[0][1] == "haiku"

    def test_namespace_contains_llm_query_parallel(self):
        """Namespace should contain llm_query_parallel function."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )

        def mock_llm(prompt: str, model: str = "haiku") -> str:
            return f"response to: {prompt[:10]}"

        def mock_parallel(prompts: list[str], model: str = "haiku") -> list[str]:
            return [f"parallel: {p[:5]}" for p in prompts]

        ns = build_repl_namespace(
            ctx,
            llm_query_fn=mock_llm,
            llm_query_parallel_fn=mock_parallel,
        )
        results = ns["llm_query_parallel"](["a", "b"])
        assert results == ["parallel: a", "parallel: b"]

    def test_namespace_contains_files_containing(self):
        """Namespace should contain files_containing search function."""
        from daydream.rlm.environment import build_repl_namespace
        import re

        ctx = RepoContext(
            files={
                "a.py": "def foo(): pass",
                "b.py": "def bar(): pass",
                "c.py": "x = 1",
            },
            structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=3, largest_files=[], languages=["python"],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "")
        matches = ns["files_containing"](r"def \w+")
        assert set(matches) == {"a.py", "b.py"}

    def test_namespace_contains_files_importing(self):
        """Namespace should contain files_importing search function."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={
                "a.py": "import os\nimport sys",
                "b.py": "from os import path",
                "c.py": "import json",
            },
            structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=3, largest_files=[], languages=["python"],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "")
        matches = ns["files_importing"]("os")
        assert set(matches) == {"a.py", "b.py"}

    def test_namespace_contains_get_file_slice(self):
        """Namespace should contain get_file_slice function."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={"a.py": "line1\nline2\nline3\nline4\nline5"},
            structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=1, largest_files=[], languages=["python"],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "")
        # get_file_slice uses 1-based line numbers
        result = ns["get_file_slice"]("a.py", 2, 4)
        assert result == "line2\nline3\nline4"

    def test_namespace_contains_final_functions(self):
        """Namespace should contain FINAL and FINAL_VAR functions."""
        from daydream.rlm.environment import build_repl_namespace, FinalAnswer

        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "")

        # Test FINAL raises FinalAnswer with the answer
        try:
            ns["FINAL"]("my answer")
            assert False, "Should have raised FinalAnswer"
        except FinalAnswer as e:
            assert e.answer == "my answer"

        # Test FINAL_VAR raises FinalAnswer with variable value
        ns["my_var"] = "variable content"
        try:
            ns["FINAL_VAR"]("my_var")
            assert False, "Should have raised FinalAnswer"
        except FinalAnswer as e:
            assert e.answer == "variable content"
