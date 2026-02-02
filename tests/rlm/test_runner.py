# tests/rlm/test_runner.py
"""Tests for RLM runner orchestration."""

import pytest

from daydream.rlm.runner import (
    CODE_BLOCK_PATTERN,
    FINAL_CALL_PATTERN,
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


class TestFinalCallPattern:
    """Tests for FINAL_CALL_PATTERN regex extraction."""

    def test_extracts_final_from_prose(self):
        """FINAL('answer') in prose should be extracted correctly."""
        prose = """Based on my analysis, I've completed the review.
        The codebase looks good overall with minor issues.
        FINAL("This is my review summary.")
        Let me know if you need more details."""

        match = FINAL_CALL_PATTERN.search(prose)
        assert match is not None
        assert match.group(0) == 'FINAL("This is my review summary.")'

    def test_extracts_final_with_single_quotes(self):
        """FINAL with single quotes should be extracted."""
        prose = "Here is the result: FINAL('The answer is 42') done."

        match = FINAL_CALL_PATTERN.search(prose)
        assert match is not None
        assert match.group(0) == "FINAL('The answer is 42')"

    def test_extracts_final_var(self):
        """FINAL_VAR('var_name') should be extracted."""
        prose = "I've stored the result. FINAL_VAR('review_report') - that's it."

        match = FINAL_CALL_PATTERN.search(prose)
        assert match is not None
        assert match.group(0) == "FINAL_VAR('review_report')"

    def test_extracts_final_with_whitespace(self):
        """FINAL with extra whitespace should be extracted."""
        prose = "Result: FINAL( 'answer with spaces' ) end."

        match = FINAL_CALL_PATTERN.search(prose)
        assert match is not None
        assert match.group(0) == "FINAL( 'answer with spaces' )"

    def test_no_match_for_prose_without_valid_final(self):
        """Prose mentioning FINAL but without valid pattern should not match."""
        # This prose mentions FINAL but doesn't have a valid FINAL() call
        prose = """The code looks good. I noticed the FINAL function is used
        for completion. However, I need to do more analysis before calling it.
        The FINAL step will be to summarize findings."""

        match = FINAL_CALL_PATTERN.search(prose)
        assert match is None

    def test_no_match_for_incomplete_final_call(self):
        """Incomplete FINAL call should not match."""
        prose = "FINAL(some_variable) is invalid because it needs quotes."

        match = FINAL_CALL_PATTERN.search(prose)
        assert match is None


class TestCodeBlockPattern:
    """Tests for CODE_BLOCK_PATTERN regex."""

    def test_extracts_python_tagged_block(self):
        """```python blocks should be extracted."""
        response = '```python\nx = 1\n```'
        match = CODE_BLOCK_PATTERN.search(response)
        assert match is not None
        assert "x = 1" in match.group(1)

    def test_extracts_py_tagged_block(self):
        """```py blocks should be extracted."""
        response = '```py\nx = 1\n```'
        match = CODE_BLOCK_PATTERN.search(response)
        assert match is not None
        assert "x = 1" in match.group(1)

    def test_ignores_bare_code_block(self):
        """Bare ``` blocks should NOT be extracted."""
        response = '```\n┌───┐\n```'
        match = CODE_BLOCK_PATTERN.search(response)
        assert match is None

    def test_ignores_yaml_block(self):
        """```yaml blocks should NOT be extracted."""
        response = '```yaml\nkey: value\n```'
        match = CODE_BLOCK_PATTERN.search(response)
        assert match is None

    def test_ignores_json_block(self):
        """```json blocks should NOT be extracted."""
        response = '```json\n{"key": "value"}\n```'
        match = CODE_BLOCK_PATTERN.search(response)
        assert match is None


class TestRLMRunner:
    """Tests for RLMRunner class."""

    def test_runner_init(self):
        """RLMRunner should initialize with config."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)
        assert runner.config == cfg

    def test_no_code_recovery_prompt_contains_context(self, tmp_path):
        """Recovery prompt should include file count and languages for context."""
        # Create test files
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "app.py").write_text("def app(): pass")

        cfg = RLMConfig(workspace_path=str(tmp_path), languages=["python"])
        runner = RLMRunner(cfg)

        # Load the context manually (normally done in run())
        runner._context = load_codebase(tmp_path, languages=["python"])

        # Build the recovery prompt
        prompt = runner._build_no_code_recovery_prompt(iteration=5)

        # Verify it contains task context
        assert "Iteration 5" in prompt
        assert "3 files" in prompt  # file_count
        assert "python" in prompt  # language
        assert "FINAL()" in prompt or "FINAL(report)" in prompt
        assert "FINAL_VAR" in prompt
        assert "repo.files" in prompt
        assert "llm_query" in prompt
        assert "files_containing" in prompt

    def test_no_code_recovery_prompt_multiple_languages(self, tmp_path):
        """Recovery prompt should list all languages."""
        # Create test files
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "app.ts").write_text("const x = 1")

        cfg = RLMConfig(
            workspace_path=str(tmp_path),
            languages=["python", "typescript"],
        )
        runner = RLMRunner(cfg)
        runner._context = load_codebase(
            tmp_path,
            languages=["python", "typescript"],
        )

        prompt = runner._build_no_code_recovery_prompt(iteration=1)

        # Should contain both languages
        assert "python" in prompt
        assert "typescript" in prompt

    async def test_consecutive_error_limit_exits_loop(self, tmp_path):
        """Runner should exit after MAX_CONSECUTIVE_ERRORS consecutive no-code responses."""
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
            # Always return a response without valid code blocks
            return "I'll analyze the code but I forgot to include code blocks."

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
        assert "without valid Python code" in result

        # Verify consecutive_error_limit event was emitted
        error_limit_events = [
            e for e in events if e[1] == "consecutive_error_limit"
        ]
        assert len(error_limit_events) == 1
        assert error_limit_events[0][2]["consecutive_errors"] == MAX_CONSECUTIVE_ERRORS

    async def test_consecutive_error_counter_resets_on_valid_code(self, tmp_path):
        """Consecutive error counter should reset when valid code is found."""
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
            # First few calls: no code, then valid code, then no code again
            if call_count <= 3:
                # No code for first 3 calls
                return "Analyzing the codebase..."
            elif call_count == 4:
                # Valid code that calls FINAL
                return '```python\nFINAL("Review complete")\n```'
            else:
                # Should not reach here if FINAL works
                return "More analysis..."

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
