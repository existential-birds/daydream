# tests/rlm/test_runner_integration.py
"""Integration tests for RLM runner orchestration loop.

These tests mock at the LLM boundary, allowing real REPL execution
to verify the full orchestration loop works correctly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from daydream.rlm.runner import (
    RLMConfig,
    RLMRunner,
    load_codebase,
)
from daydream.rlm.environment import RepoContext


class TestRLMRunnerSystemPrompt:
    """Tests for system prompt generation."""

    def test_build_system_prompt_includes_codebase_metadata(self, tmp_path):
        """System prompt should include codebase metadata."""
        # Create test files
        (tmp_path / "main.py").write_text("def main():\n    print('hello')\n")
        (tmp_path / "utils.py").write_text("def helper():\n    pass\n")

        cfg = RLMConfig(workspace_path=str(tmp_path), languages=["python"])
        runner = RLMRunner(cfg)
        runner._context = load_codebase(tmp_path, ["python"])

        prompt = runner._build_system_prompt()

        # Should include key metadata
        assert "file_count" in prompt.lower() or "2 files" in prompt.lower() or "files: 2" in prompt.lower()
        assert "python" in prompt.lower()
        assert "repo" in prompt
        assert "llm_query" in prompt
        assert "FINAL" in prompt

    def test_build_system_prompt_includes_available_functions(self, tmp_path):
        """System prompt should document available REPL functions."""
        (tmp_path / "app.py").write_text("x = 1")

        cfg = RLMConfig(workspace_path=str(tmp_path), languages=["python"])
        runner = RLMRunner(cfg)
        runner._context = load_codebase(tmp_path, ["python"])

        prompt = runner._build_system_prompt()

        # Should document all available functions
        assert "files_containing" in prompt
        assert "files_importing" in prompt
        assert "get_file_slice" in prompt
        assert "FINAL(" in prompt
        assert "FINAL_VAR" in prompt


class TestRLMRunnerLLMCallback:
    """Tests for LLM callback handling."""

    def test_handle_llm_query_routes_to_callback(self, tmp_path):
        """llm_query in REPL should route through runner."""
        (tmp_path / "test.py").write_text("x = 1")

        cfg = RLMConfig(workspace_path=str(tmp_path), languages=["python"])
        runner = RLMRunner(cfg)
        runner._context = load_codebase(tmp_path, ["python"])

        calls = []

        def mock_llm(prompt: str, model: str) -> str:
            calls.append((prompt, model))
            return f"Response to: {prompt[:20]}"

        runner._llm_callback = mock_llm

        result = runner._handle_llm_query("Analyze this code", "haiku")

        assert len(calls) == 1
        assert calls[0][0] == "Analyze this code"
        assert calls[0][1] == "haiku"
        assert "Response to:" in result


class TestRLMRunnerOrchestrationLoop:
    """Tests for the main orchestration loop."""

    @pytest.mark.asyncio
    async def test_run_executes_code_from_llm(self, tmp_path):
        """Runner should execute code returned by LLM."""
        # Create test codebase
        (tmp_path / "main.py").write_text("def greet(): return 'hello'")

        cfg = RLMConfig(
            workspace_path=str(tmp_path),
            languages=["python"],
            use_container=False,
        )
        runner = RLMRunner(cfg)

        # Track iterations
        iterations = []

        # Mock LLM that returns code, then FINAL (raw Python, no markdown)
        async def mock_llm_call(prompt: str) -> str:
            iterations.append(prompt)
            if len(iterations) == 1:
                # First call: return code to execute
                return '''files = list(repo.files.keys())
print(f"Found {len(files)} files")'''
            else:
                # Second call: return FINAL
                return 'FINAL("Review complete: 1 file found")'

        runner._call_llm = mock_llm_call

        result = await runner.run()

        assert "Review complete" in result or "1 file" in result.lower()
        assert len(iterations) >= 1

    @pytest.mark.asyncio
    async def test_run_handles_final_answer(self, tmp_path):
        """Runner should stop when FINAL is called."""
        (tmp_path / "app.py").write_text("x = 1")

        cfg = RLMConfig(
            workspace_path=str(tmp_path),
            languages=["python"],
            use_container=False,
        )
        runner = RLMRunner(cfg)

        # Mock LLM that immediately returns FINAL (raw Python, no markdown)
        async def mock_llm_call(prompt: str) -> str:
            return 'FINAL("# Code Review Report\\n\\nNo issues found.")'

        runner._call_llm = mock_llm_call

        result = await runner.run()

        assert "Code Review Report" in result
        assert "No issues found" in result

    @pytest.mark.asyncio
    async def test_run_handles_execution_errors(self, tmp_path):
        """Runner should continue after REPL errors."""
        (tmp_path / "src.py").write_text("code = 'test'")

        cfg = RLMConfig(
            workspace_path=str(tmp_path),
            languages=["python"],
            use_container=False,
        )
        runner = RLMRunner(cfg)

        call_count = [0]

        async def mock_llm_call(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                # Return code with error (raw Python, no markdown)
                return 'undefined_variable  # This will raise NameError'
            else:
                # After error, return FINAL (raw Python, no markdown)
                return 'FINAL("Review complete despite error")'

        runner._call_llm = mock_llm_call

        result = await runner.run()

        # Should have continued after error and produced final result
        assert "Review complete" in result
        assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_run_respects_max_iterations(self, tmp_path):
        """Runner should stop after max iterations."""
        (tmp_path / "file.py").write_text("x = 1")

        cfg = RLMConfig(
            workspace_path=str(tmp_path),
            languages=["python"],
            use_container=False,
            max_iterations=3,
        )
        runner = RLMRunner(cfg)

        call_count = [0]

        async def mock_llm_call(prompt: str) -> str:
            call_count[0] += 1
            # Never call FINAL - just keep executing code (raw Python, no markdown)
            return 'print("iteration")'

        runner._call_llm = mock_llm_call

        result = await runner.run()

        # Should have stopped after max iterations
        assert call_count[0] <= 3
        # Should return some result even without FINAL
        assert result is not None

    @pytest.mark.asyncio
    async def test_run_passes_execution_context_to_llm(self, tmp_path):
        """Runner should include execution results in follow-up prompts."""
        (tmp_path / "test.py").write_text("value = 42")

        cfg = RLMConfig(
            workspace_path=str(tmp_path),
            languages=["python"],
            use_container=False,
        )
        runner = RLMRunner(cfg)

        prompts_received = []

        async def mock_llm_call(prompt: str) -> str:
            prompts_received.append(prompt)
            if len(prompts_received) == 1:
                # Raw Python, no markdown
                return 'print("Hello from REPL")'
            else:
                return 'FINAL("Done")'

        runner._call_llm = mock_llm_call

        await runner.run()

        # Second prompt should include output from first execution
        assert len(prompts_received) >= 2
        assert "Hello from REPL" in prompts_received[1]


class TestRLMRunnerSubLLMCalls:
    """Tests for sub-LLM (llm_query) handling."""

    @pytest.mark.asyncio
    async def test_llm_query_in_code_routes_correctly(self, tmp_path):
        """llm_query calls in REPL should route to sub-LLM handler."""
        (tmp_path / "code.py").write_text("def foo(): pass")

        cfg = RLMConfig(
            workspace_path=str(tmp_path),
            languages=["python"],
            use_container=False,
            sub_model="haiku",
        )
        runner = RLMRunner(cfg)

        sub_llm_calls = []

        def mock_sub_llm(prompt: str, model: str) -> str:
            sub_llm_calls.append((prompt, model))
            return "Analyzed code looks good"

        runner._llm_callback = mock_sub_llm

        call_count = [0]

        async def mock_llm_call(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                # Raw Python, no markdown
                return '''analysis = llm_query("Analyze the foo function")
print(analysis)'''
            else:
                return 'FINAL("Review done with sub-LLM analysis")'

        runner._call_llm = mock_llm_call

        result = await runner.run()

        # Sub-LLM should have been called
        assert len(sub_llm_calls) == 1
        assert "foo function" in sub_llm_calls[0][0]
        assert sub_llm_calls[0][1] == "haiku"


class TestRLMRunnerCodeExtraction:
    """Tests for Python-only code extraction from LLM responses."""

    def test_extract_code_returns_stripped_response(self, tmp_path):
        """Response is returned as-is (stripped) in Python-only mode."""
        (tmp_path / "x.py").write_text("x=1")

        cfg = RLMConfig(workspace_path=str(tmp_path), languages=["python"])
        runner = RLMRunner(cfg)

        response = '''  files = list(repo.files.keys())
print(f"Found {len(files)} files")  '''

        code = runner._extract_code(response)

        assert "files = list(repo.files.keys())" in code
        assert "print" in code
        # In Python-only mode, response IS the code (just stripped)
        assert code == response.strip()

    def test_extract_code_handles_final(self, tmp_path):
        """FINAL calls are passed through as-is."""
        (tmp_path / "x.py").write_text("x=1")

        cfg = RLMConfig(workspace_path=str(tmp_path), languages=["python"])
        runner = RLMRunner(cfg)

        response = "FINAL('Review complete')"
        code = runner._extract_code(response)

        assert code == "FINAL('Review complete')"
