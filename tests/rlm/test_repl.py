# tests/rlm/test_repl.py
"""Tests for REPL process manager."""

import pytest

from daydream.rlm.repl import (
    REPLProcess,
    ExecuteResult,
)
from daydream.rlm.environment import RepoContext


class TestExecuteResult:
    """Tests for ExecuteResult dataclass."""

    def test_execute_result_success(self):
        """ExecuteResult should capture successful output."""
        result = ExecuteResult(
            output="hello\n",
            error=None,
            final_answer=None,
        )
        assert result.output == "hello\n"
        assert result.is_error is False
        assert result.is_final is False

    def test_execute_result_error(self):
        """ExecuteResult should capture errors."""
        result = ExecuteResult(
            output="",
            error="NameError: name 'x' is not defined",
            final_answer=None,
        )
        assert result.is_error is True
        assert "NameError" in result.error

    def test_execute_result_final(self):
        """ExecuteResult should capture final answer."""
        result = ExecuteResult(
            output="",
            error=None,
            final_answer="# Code Review Report\n...",
        )
        assert result.is_final is True
        assert result.final_answer.startswith("# Code Review")


class TestREPLProcess:
    """Tests for REPLProcess class."""

    def test_repl_process_init(self):
        """REPLProcess should initialize with context."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        repl = REPLProcess(ctx, llm_callback=lambda p, m: "")
        assert repl.context == ctx
        assert repl.is_running is False

    def test_repl_process_not_running_initially(self):
        """REPLProcess should not be running initially."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        repl = REPLProcess(ctx, llm_callback=lambda p, m: "")
        assert repl.is_running is False


class TestREPLProcessExecution:
    """Tests for REPLProcess code execution."""

    def test_execute_simple_code(self):
        """Should execute simple Python code."""
        ctx = RepoContext(
            files={"a.py": "x=1"},
            structure={}, services={}, file_sizes={"a.py": 3},
            total_tokens=3, file_count=1, largest_files=[("a.py", 3)],
            languages=["python"], changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            result = repl.execute("print('hello world')")
            assert result.output.strip() == "hello world"
            assert result.is_error is False

    def test_execute_accesses_repo(self):
        """Should access repo context in execution."""
        ctx = RepoContext(
            files={"main.py": "print('hi')"},
            structure={}, services={}, file_sizes={"main.py": 10},
            total_tokens=10, file_count=1, largest_files=[("main.py", 10)],
            languages=["python"], changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            result = repl.execute("print(len(repo.files))")
            assert "1" in result.output

    def test_execute_catches_errors(self):
        """Should capture execution errors."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            result = repl.execute("undefined_variable")
            assert result.is_error is True
            assert "NameError" in result.error

    def test_execute_final_stops_execution(self):
        """FINAL() should produce final answer."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            result = repl.execute('FINAL("my report")')
            assert result.is_final is True
            assert result.final_answer == "my report"

    def test_execute_llm_query(self):
        """llm_query should route to callback."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        calls = []

        def mock_llm(prompt: str, model: str) -> str:
            calls.append((prompt, model))
            return "LLM response"

        with REPLProcess(ctx, llm_callback=mock_llm) as repl:
            result = repl.execute('result = llm_query("analyze this")\nprint(result)')
            assert "LLM response" in result.output
            assert len(calls) == 1
            assert calls[0][0] == "analyze this"
            assert calls[0][1] == "haiku"

    def test_execute_truncates_long_output(self):
        """Should truncate output exceeding limit."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            # Generate output longer than truncation limit
            result = repl.execute("print('x' * 100000)")
            assert "[truncated" in result.output
            assert len(result.output) < 60000  # Should be truncated
