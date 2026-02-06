# tests/rlm/test_runner.py
"""Tests for RLM runner orchestration."""

import pytest

from daydream.rlm.runner import (
    MAX_CONSECUTIVE_ERRORS,
    RLMConfig,
    RLMRunner,
    load_codebase,
)


class TestRLMConfig:
    """Tests for RLMConfig dataclass."""

    def test_config_defaults(self):
        """RLMConfig should have sensible defaults."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        assert cfg.workspace_path == "/repo"
        assert cfg.languages == ["python"]
        assert cfg.model == "opus"
        assert cfg.sub_model == "haiku"
        assert cfg.use_container is True

    def test_config_pr_mode(self):
        """RLMConfig should support PR mode."""
        cfg = RLMConfig(
            workspace_path="/repo",
            languages=["python"],
            pr_number=123,
        )
        assert cfg.pr_number == 123


class TestLoadCodebase:
    """Tests for load_codebase function."""

    def test_load_codebase_python(self, tmp_path):
        """Should load Python files from directory."""
        # Create test files
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "readme.md").write_text("# Readme")

        ctx = load_codebase(tmp_path, languages=["python"])

        assert ctx.file_count == 2
        assert "main.py" in ctx.files or str(tmp_path / "main.py") in ctx.files
        assert ctx.languages == ["python"]

    def test_load_codebase_excludes_hidden(self, tmp_path):
        """Should exclude hidden directories."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("gitconfig")
        (tmp_path / "main.py").write_text("x=1")

        ctx = load_codebase(tmp_path, languages=["python"])

        assert ctx.file_count == 1
        assert not any(".git" in f for f in ctx.files.keys())

    def test_load_codebase_excludes_node_modules(self, tmp_path):
        """Should exclude node_modules."""
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("module")
        (tmp_path / "app.ts").write_text("const x = 1")

        ctx = load_codebase(tmp_path, languages=["typescript"])

        assert ctx.file_count == 1


class TestCodeExtractionPythonOnly:
    """Tests for Python-only code extraction."""

    def test_extract_code_returns_response_stripped(self):
        """_extract_code should return the response stripped of whitespace."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        response = '  print("hello")  \n  '
        code = runner._extract_code(response)
        assert code == 'print("hello")'

    def test_extract_code_empty_response(self):
        """Empty/whitespace response should return empty string."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        assert runner._extract_code("") == ""
        assert runner._extract_code("   ") == ""
        assert runner._extract_code("\n\t\n") == ""

    def test_extract_code_preserves_multiline(self):
        """Multiline code should be preserved (just stripped)."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        response = '''
print("line 1")
print("line 2")
'''
        code = runner._extract_code(response)
        assert 'print("line 1")' in code
        assert 'print("line 2")' in code

    def test_extract_code_with_final(self):
        """FINAL calls should be passed through as-is."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        response = 'FINAL("done")'
        code = runner._extract_code(response)
        assert code == 'FINAL("done")'


class TestRLMRunner:
    """Tests for RLMRunner class."""

    def test_runner_init(self):
        """RLMRunner should initialize with config."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)
        assert runner.config == cfg

    async def test_consecutive_error_limit_exits_loop(self, tmp_path):
        """Runner should exit after MAX_CONSECUTIVE_ERRORS consecutive empty responses."""
        # Create a minimal test file
        (tmp_path / "main.py").write_text("x = 1")

        cfg = RLMConfig(
            workspace_path=str(tmp_path),
            languages=["python"],
            use_container=False,  # Don't use container for testing
            max_iterations=20,  # Higher than MAX_CONSECUTIVE_ERRORS
        )
        runner = RLMRunner(cfg)

        # Track how many times LLM was called
        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            # Always return empty response (stripped to empty string)
            return "   "

        runner._call_llm = mock_llm

        # Track emitted events
        events = []

        def track_events(iteration: int, event_type: str, data: dict) -> None:
            events.append((iteration, event_type, data))

        runner._on_event = track_events

        # Run the review
        result = await runner.run()

        # Should have exited after MAX_CONSECUTIVE_ERRORS
        assert call_count == MAX_CONSECUTIVE_ERRORS
        assert "Code Review (Failed)" in result
        assert f"{MAX_CONSECUTIVE_ERRORS} consecutive iterations" in result
        assert "empty responses" in result

        # Verify consecutive_error_limit event was emitted
        error_limit_events = [
            e for e in events if e[1] == "consecutive_error_limit"
        ]
        assert len(error_limit_events) == 1
        assert error_limit_events[0][2]["consecutive_errors"] == MAX_CONSECUTIVE_ERRORS

    async def test_consecutive_error_counter_resets_on_valid_code(self, tmp_path):
        """Consecutive error counter should reset when non-empty code is found."""
        (tmp_path / "main.py").write_text("x = 1")

        cfg = RLMConfig(
            workspace_path=str(tmp_path),
            languages=["python"],
            use_container=False,
            max_iterations=10,
        )
        runner = RLMRunner(cfg)

        call_count = 0

        async def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            # First few calls: empty, then valid code with FINAL
            if call_count <= 3:
                # Empty responses for first 3 calls
                return "   "
            elif call_count == 4:
                # Valid Python code that calls FINAL (no markdown fences needed)
                return 'FINAL("Review complete")'
            else:
                # Should not reach here if FINAL works
                return "print('more')"

        runner._call_llm = mock_llm

        events = []

        def track_events(iteration: int, event_type: str, data: dict) -> None:
            events.append((iteration, event_type, data))

        runner._on_event = track_events

        result = await runner.run()

        # Should have completed with FINAL, not hit error limit
        assert "Review complete" in result
        assert "Code Review (Failed)" not in result

        # Verify no consecutive_error_limit event was emitted
        error_limit_events = [
            e for e in events if e[1] == "consecutive_error_limit"
        ]
        assert len(error_limit_events) == 0
