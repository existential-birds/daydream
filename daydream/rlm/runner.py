# daydream/rlm/runner.py
"""RLM runner orchestration.

This module provides the main orchestration for RLM code reviews,
coordinating the REPL, container, and LLM interactions.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from daydream.rlm.environment import FileInfo, RepoContext
from daydream.rlm.repl import ExecuteResult, REPLProcess

# File extensions by language
LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "go": [".go"],
}

# Directories to exclude from codebase loading
EXCLUDED_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".next",
    ".nuxt",
}

# Default maximum iterations for the orchestration loop
DEFAULT_MAX_ITERATIONS = 50


@dataclass
class RLMConfig:
    """Configuration for RLM code review.

    Attributes:
        workspace_path: Path to the repository to review.
        languages: List of languages to include in review.
        model: Model to use for root LLM (orchestration).
        sub_model: Model to use for sub-LLM calls (analysis).
        use_container: Whether to use devcontainer sandboxing.
        pr_number: PR number for focused review (optional).
        timeout: Maximum time for review in seconds.
        max_iterations: Maximum number of LLM iterations.
    """

    workspace_path: str
    languages: list[str]
    model: str = "opus"
    sub_model: str = "haiku"
    use_container: bool = True
    pr_number: int | None = None
    timeout: float = 600.0
    max_iterations: int = DEFAULT_MAX_ITERATIONS


def _estimate_tokens(content: str) -> int:
    """Estimate token count for content.

    Uses simple heuristic of ~4 characters per token.

    Args:
        content: Text content to estimate.

    Returns:
        Estimated token count.
    """
    return len(content) // 4


def _should_exclude_dir(name: str) -> bool:
    """Check if directory should be excluded."""
    return name in EXCLUDED_DIRS or name.startswith(".")


def load_codebase(
    workspace_path: Path,
    languages: list[str],
    changed_files: list[str] | None = None,
) -> RepoContext:
    """Load codebase files into RepoContext.

    Args:
        workspace_path: Path to repository root.
        languages: List of languages to include.
        changed_files: Optional list of changed files for PR mode.

    Returns:
        RepoContext with loaded files and metadata.
    """
    # Build set of extensions to include
    extensions: set[str] = set()
    for lang in languages:
        extensions.update(LANGUAGE_EXTENSIONS.get(lang, []))

    files: dict[str, str] = {}
    file_sizes: dict[str, int] = {}
    structure: dict[str, FileInfo] = {}

    for root, dirs, filenames in os.walk(workspace_path):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not _should_exclude_dir(d)]

        for filename in filenames:
            # Check extension
            ext = os.path.splitext(filename)[1]
            if ext not in extensions:
                continue

            filepath = Path(root) / filename
            rel_path = str(filepath.relative_to(workspace_path))

            try:
                content = filepath.read_text(encoding="utf-8")
                files[rel_path] = content
                tokens = _estimate_tokens(content)
                file_sizes[rel_path] = tokens

                # Basic structure extraction (placeholder for tree-sitter)
                structure[rel_path] = FileInfo(
                    language=_detect_language(ext),
                    functions=[],  # TODO: tree-sitter parsing
                    classes=[],
                    imports=[],
                    exports=[],
                )
            except (UnicodeDecodeError, PermissionError):
                # Skip binary or unreadable files
                continue

    # Calculate totals
    total_tokens = sum(file_sizes.values())
    file_count = len(files)

    # Get largest files
    sorted_files = sorted(file_sizes.items(), key=lambda x: x[1], reverse=True)
    largest_files = sorted_files[:10]

    return RepoContext(
        files=files,
        structure=structure,
        services={},  # TODO: service detection
        file_sizes=file_sizes,
        total_tokens=total_tokens,
        file_count=file_count,
        largest_files=largest_files,
        languages=languages,
        changed_files=changed_files,
    )


def _detect_language(ext: str) -> str:
    """Detect language from file extension."""
    ext_to_lang = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".go": "go",
    }
    return ext_to_lang.get(ext, "unknown")


# Pattern to extract Python code from markdown fenced blocks
CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL,
)


class RLMRunner:
    """Orchestrates RLM code review execution.

    Manages the iterative loop of:
    1. Generating system prompt with codebase metadata
    2. Sending prompts to root LLM
    3. Executing returned code in REPL
    4. Handling llm_query callbacks
    5. Looping until FINAL is called or timeout
    """

    def __init__(self, config: RLMConfig):
        """Initialize RLM runner.

        Args:
            config: RLM configuration.
        """
        self.config = config
        self._context: RepoContext | None = None
        self._repl: REPLProcess | None = None
        self._call_llm: Callable[[str], Awaitable[str]] | None = None
        self._llm_callback: Callable[[str, str], str] | None = None

    def _build_system_prompt(self) -> str:
        """Build the system prompt describing the REPL environment.

        Returns:
            System prompt string for the LLM.
        """
        if self._context is None:
            raise RuntimeError("Context not loaded")

        ctx = self._context

        # Build file list preview (show first 20)
        file_list = list(ctx.files.keys())[:20]
        file_preview = "\n".join(f"  - {f}" for f in file_list)
        if len(ctx.files) > 20:
            file_preview += f"\n  ... and {len(ctx.files) - 20} more files"

        # Build largest files section
        largest_section = "\n".join(
            f"  - {path}: ~{tokens:,} tokens"
            for path, tokens in ctx.largest_files[:5]
        )

        return f"""You are a code review agent with access to a Python REPL environment.

## Repository Context

- **Files**: {ctx.file_count:,} files loaded
- **Languages**: {', '.join(ctx.languages)}
- **Total tokens**: ~{ctx.total_tokens:,} estimated
{f'- **Changed files (PR mode)**: {len(ctx.changed_files)} files' if ctx.changed_files else ''}

### File Preview
{file_preview}

### Largest Files
{largest_section}

## Available Objects and Functions

You have access to the following in the REPL namespace:

### `repo` - Repository Context Object
- `repo.files` - Dict[str, str]: Mapping of file paths to their contents
- `repo.structure` - Dict[str, FileInfo]: Parsed metadata for each file
- `repo.file_sizes` - Dict[str, int]: Token counts per file
- `repo.total_tokens` - int: Total token count
- `repo.file_count` - int: Number of files
- `repo.largest_files` - List[Tuple[str, int]]: Top files by size
- `repo.languages` - List[str]: Detected languages
- `repo.changed_files` - Optional[List[str]]: Changed files in PR mode

### `llm_query(prompt: str, model: str = "haiku") -> str`
Make a fresh-context sub-LLM call. Use this to analyze code snippets, summarize findings,
or perform detailed analysis. The sub-LLM has no memory of previous calls.

### `llm_query_parallel(prompts: List[str], model: str = "haiku") -> List[str]`
Execute multiple independent queries in parallel for efficiency.

### `files_containing(pattern: str) -> List[str]`
Grep-like regex search. Returns list of file paths matching the pattern.

### `files_importing(module: str) -> List[str]`
Find files that import a given module (e.g., `files_importing("os")`).

### `get_file_slice(path: str, start_line: int, end_line: int) -> str`
Get specific line range from a file (1-based, inclusive).

### `FINAL(answer: str) -> None`
Signal completion and return the final review report. Call this when done.

### `FINAL_VAR(var_name: str) -> None`
Signal completion, returning the value of a REPL variable as the final answer.

## Your Task

Review this codebase and produce a comprehensive code review report. Your report should:

1. Identify potential bugs, security issues, and code quality problems
2. Prioritize findings by severity (Critical, High, Medium, Low)
3. Provide specific file paths and line numbers for each issue
4. Suggest fixes where appropriate

## Instructions

1. Generate Python code to analyze the codebase
2. I will execute your code and return the output
3. Based on the output, generate more code or produce findings
4. When ready, call `FINAL()` with your review report

Start by exploring the codebase structure, then dive into specific areas of concern.

Respond with Python code in a fenced code block:
```python
# Your code here
```
"""

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response.

        Args:
            response: LLM response text.

        Returns:
            Extracted Python code, or empty string if no code found.
        """
        match = CODE_BLOCK_PATTERN.search(response)
        if match:
            return match.group(1).strip()
        return ""

    def _handle_llm_query(self, prompt: str, model: str) -> str:
        """Handle llm_query callback from REPL.

        Args:
            prompt: The query prompt.
            model: The model to use (e.g., "haiku").

        Returns:
            LLM response text.
        """
        if self._llm_callback is None:
            return "[Error: No LLM callback configured]"
        return self._llm_callback(prompt, model)

    def _build_continuation_prompt(
        self,
        iteration: int,
        execution_result: ExecuteResult,
    ) -> str:
        """Build prompt for continuation after code execution.

        Args:
            iteration: Current iteration number.
            execution_result: Result from REPL execution.

        Returns:
            Continuation prompt for LLM.
        """
        parts = [f"## Execution Result (iteration {iteration})"]

        if execution_result.output:
            parts.append(f"\n### Output\n```\n{execution_result.output}\n```")

        if execution_result.is_error:
            parts.append(f"\n### Error\n```\n{execution_result.error}\n```")
            parts.append(
                "\nThe code raised an error. Fix the issue and continue the review."
            )

        parts.append(
            "\n\nContinue your analysis. When ready, call `FINAL()` with your report."
        )
        parts.append("\n\n```python\n# Your next code here\n```")

        return "\n".join(parts)

    async def run(self) -> str:
        """Execute the RLM code review.

        Returns:
            Final review report.
        """
        # Load codebase
        workspace = Path(self.config.workspace_path)
        self._context = load_codebase(
            workspace,
            self.config.languages,
        )

        # If no LLM callback is set, use a default that returns a placeholder
        if self._call_llm is None:
            # Default implementation - can be overridden for testing or real usage
            return await self._run_with_default_llm()

        # Set up the llm_query callback for the REPL
        if self._llm_callback is None:
            # Use a simple echo for testing
            self._llm_callback = lambda prompt, model: f"[Sub-LLM response to: {prompt[:50]}...]"

        # Create REPL with llm_query callback
        self._repl = REPLProcess(
            context=self._context,
            llm_callback=self._llm_callback,
        )

        try:
            self._repl.start()

            # Build initial system prompt
            system_prompt = self._build_system_prompt()
            current_prompt = system_prompt

            final_result: str | None = None
            iteration = 0

            while iteration < self.config.max_iterations:
                iteration += 1

                # Get code from LLM
                response = await self._call_llm(current_prompt)
                code = self._extract_code(response)

                if not code:
                    # No code found - try to extract any runnable content
                    # or ask for clarification
                    if "FINAL(" in response:
                        # Try to execute the raw response
                        code = response.strip()
                    else:
                        # No executable code, continue to next iteration
                        current_prompt = (
                            "I couldn't find Python code in your response. "
                            "Please provide code in a fenced code block:\n"
                            "```python\n# Your code here\n```"
                        )
                        continue

                # Execute code in REPL
                result = self._repl.execute(code)

                # Check for final answer
                if result.is_final:
                    final_result = result.final_answer
                    break

                # Build continuation prompt with execution result
                current_prompt = self._build_continuation_prompt(iteration, result)

            # If we hit max iterations without FINAL, generate a summary
            if final_result is None:
                final_result = (
                    f"# Code Review (Incomplete)\n\n"
                    f"Review stopped after {iteration} iterations without completion.\n"
                    f"Please review the execution log for partial findings."
                )

            return final_result

        finally:
            if self._repl is not None:
                self._repl.stop()
                self._repl = None

    async def _run_with_default_llm(self) -> str:
        """Run with a default placeholder LLM for testing.

        Returns:
            Placeholder review result.
        """
        if self._context is None:
            return "# Code Review\n\nError: No codebase loaded."

        # Return a simple summary when no LLM is configured
        return (
            f"# Code Review\n\n"
            f"**Repository Summary**\n\n"
            f"- Files: {self._context.file_count:,}\n"
            f"- Languages: {', '.join(self._context.languages)}\n"
            f"- Estimated tokens: {self._context.total_tokens:,}\n\n"
            f"*Note: Full RLM review requires LLM configuration.*"
        )
