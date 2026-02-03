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


class TestCodeExtraction:
    """Tests for smart code extraction from LLM responses."""

    def test_prefers_final_block(self):
        """Should prefer code blocks containing FINAL/FINAL_VAR calls."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        # Response with documentation example followed by FINAL call
        response = '''# Analysis Report

Here's an example of the issue:
```python
@dataclass
class AgentState:
    value: int
```

Now completing the review:
```python
report = "Review complete"
FINAL_VAR("report")
```
'''
        code = runner._extract_code(response)
        assert "FINAL_VAR" in code
        assert "@dataclass" not in code

    def test_skips_bare_decorator_examples(self):
        """Should skip code that looks like documentation examples."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        # Response with a documentation example (bare decorator without imports)
        response = '''The codebase uses dataclasses:
```python
@dataclass
class Config:
    name: str
```

Let me analyze further:
```python
print(f"Files: {len(repo.files)}")
```
'''
        code = runner._extract_code(response)
        # Should get the analysis code, not the example
        assert "repo.files" in code
        assert "@dataclass" not in code

    def test_uses_last_block_as_fallback(self):
        """Should use last code block when no FINAL and no obvious examples."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        response = '''First I'll check the structure:
```python
print(repo.file_count)
```

Now deeper analysis:
```python
for f in repo.files:
    print(f)
```
'''
        code = runner._extract_code(response)
        # Should get the last block
        assert "for f in repo.files" in code

    def test_handles_markdown_report_with_embedded_examples(self):
        """Should handle real-world case of report with embedded code examples."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        # This simulates the actual failure from test_run.log
        response = '''Based on my analysis, here is the report:

# Code Review Report

## Architecture

The code uses dataclasses:
```python
@dataclass
class AgentState:
    debug_log: TextIO | None = None
    quiet_mode: bool = False
```

## Issues Found
- Issue 1: Missing validation

```python
report = """# Final Report
Found 3 issues total.
"""
FINAL_VAR("report")
```
'''
        code = runner._extract_code(response)
        # Should find the FINAL_VAR block, not the dataclass example
        assert "FINAL_VAR" in code
        assert "AgentState" not in code


class TestIsLikelyExample:
    """Tests for _is_likely_example helper."""

    def test_bare_decorator_is_example(self):
        """Bare decorator without imports should be detected as example."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        code = '''@dataclass
class Config:
    name: str'''
        assert runner._is_likely_example(code) is True

    def test_short_snippet_without_repl_is_example(self):
        """Short code without REPL functions should be detected as example."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        code = "x = 1 + 2"
        assert runner._is_likely_example(code) is True

    def test_code_with_repo_is_not_example(self):
        """Code using repo.* is not an example."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        code = "print(repo.files)"
        assert runner._is_likely_example(code) is False

    def test_code_with_final_is_not_example(self):
        """Code with FINAL is not an example."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        code = 'FINAL("done")'
        assert runner._is_likely_example(code) is False

    def test_decorator_with_repl_usage_is_not_example(self):
        """Decorator with REPL functions in body is not an example."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)

        code = '''@something
def analyze():
    print(repo.files)'''
        assert runner._is_likely_example(code) is False


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
