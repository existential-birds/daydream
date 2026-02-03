# daydream/rlm/runner.py
"""RLM runner orchestration.

This module provides the main orchestration for RLM code reviews,
coordinating the REPL, container, and LLM interactions.
"""

import asyncio
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from daydream.prompts.review_system_prompt import get_review_prompt
from daydream.rlm.container import DevContainer
from daydream.rlm.environment import FileInfo, RepoContext
from daydream.rlm.history import ConversationHistory
from daydream.rlm.repl import ExecuteResult, REPLProcess

logger = logging.getLogger(__name__)

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

# Maximum consecutive iterations without valid code before aborting
MAX_CONSECUTIVE_ERRORS = 5

# Minimum number of sub-LLM calls required for large repositories
MIN_SUBLM_CALLS_FOR_LARGE_REPO = 2


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


@dataclass
class RLMMetrics:
    """Metrics tracking for RLM runner.

    Tracks tool usage and execution statistics to help understand
    agent behavior and identify inefficiencies.

    Attributes:
        iterations: Total number of iterations completed.
        llm_query_calls: Number of single llm_query() calls.
        llm_query_parallel_calls: Number of llm_query_parallel() calls.
        files_containing_calls: Number of files_containing() calls.
        files_importing_calls: Number of files_importing() calls.
        repo_files_accesses: Number of times repo.files was accessed.
        unique_files_accessed: Set of unique file paths accessed.
        code_extraction_failures: Number of times code extraction failed.
        final_call_attempts: Number of FINAL/FINAL_VAR call attempts.
    """

    iterations: int = 0
    llm_query_calls: int = 0
    llm_query_parallel_calls: int = 0
    files_containing_calls: int = 0
    files_importing_calls: int = 0
    repo_files_accesses: int = 0
    unique_files_accessed: set[str] = field(default_factory=set)
    code_extraction_failures: int = 0
    final_call_attempts: int = 0


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


def get_changed_files(workspace: Path) -> list[str] | None:
    """Get list of changed files compared to the default branch.

    Attempts to detect files changed in the current branch compared to
    origin/main or origin/master. Useful for PR-focused reviews.

    Args:
        workspace: Path to the git repository root.

    Returns:
        List of changed file paths relative to workspace, or None if:
        - Not a git repository
        - No changes found
        - Any error occurs
    """
    try:
        # Try to get the default branch name from symbolic ref
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            # Parse "refs/remotes/origin/main" -> "main"
            default_branch = result.stdout.strip().split("/")[-1]
        else:
            # Fall back to trying main, then master
            default_branch = None
            for branch in ["main", "master"]:
                check_result = subprocess.run(
                    ["git", "rev-parse", "--verify", f"origin/{branch}"],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if check_result.returncode == 0:
                    default_branch = branch
                    break

            if default_branch is None:
                return None

        # Get changed files using three-dot diff (merge-base comparison)
        diff_result = subprocess.run(
            ["git", "diff", "--name-only", f"origin/{default_branch}...HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if diff_result.returncode != 0:
            return None

        # Parse output into list of file paths
        changed = [line.strip() for line in diff_result.stdout.strip().split("\n") if line.strip()]

        if not changed:
            return None

        return changed

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        # Any error means we fall back to full repo review
        return None


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
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL,
)

# Pattern to extract FINAL() or FINAL_VAR() calls from prose
FINAL_CALL_PATTERN = re.compile(
    r'FINAL\s*\(\s*["\'].*?["\']\s*\)|FINAL_VAR\s*\(\s*["\'][\w]+["\']\s*\)',
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

    # Patterns for validating review quality
    # Severity markers: [CRITICAL], [HIGH], [MEDIUM], [LOW] or Critical:, High:, etc.
    SEVERITY_PATTERN = re.compile(
        r"\[(CRITICAL|HIGH|MEDIUM|LOW)\]|"
        r"(Critical|High|Medium|Low)\s*:|"
        r"Severity\s*:\s*(Critical|High|Medium|Low)",
        re.IGNORECASE,
    )

    # File:line references like file.py:123 or path/to/file.ts:45
    FILE_LINE_PATTERN = re.compile(
        r"[\w./\\-]+\.(py|ts|tsx|js|jsx|go|rs|java|rb|php|c|cpp|h|hpp):\d+",
    )

    # Keywords indicating architecture summary (not a proper review)
    ARCHITECTURE_KEYWORDS = [
        "architecture report",
        "codebase overview",
        "project structure",
        "directory structure",
        "folder structure",
        "system design",
        "high-level overview",
    ]

    # Keywords indicating actual code review issues
    ISSUE_KEYWORDS = [
        "bug",
        "issue",
        "vulnerability",
        "security",
        "error",
        "warning",
        "fix",
        "problem",
        "defect",
        "flaw",
        "risk",
    ]

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
        # Logging callback for progress events (iteration, event_type, data)
        self._on_event: Callable[[int, str, dict], None] | None = None
        self._history: ConversationHistory | None = None
        # Metrics tracking
        self._metrics = RLMMetrics()

    def _emit_event(self, iteration: int, event_type: str, data: dict) -> None:
        """Emit a progress event for logging/UI.

        Args:
            iteration: Current iteration number.
            event_type: Type of event (e.g., "code_extracted", "repl_result").
            data: Event-specific data dictionary.
        """
        if self._on_event is not None:
            self._on_event(iteration, event_type, data)

    def _validate_review_quality(self, answer: str) -> tuple[bool, list[str]]:
        """Validate that the final answer is a proper code review, not just an architecture summary.

        Checks for:
        - Severity markers like [CRITICAL], [HIGH], [MEDIUM], [LOW], or similar patterns
        - File:line references (pattern like `file.py:123` or `path/to/file.ts:45`)
        - Not just an architecture/structure summary without issue-related content

        Args:
            answer: The final answer string to validate.

        Returns:
            Tuple of (is_valid, list_of_warnings). is_valid is True if the review
            appears to be a proper code review, False otherwise. list_of_warnings
            contains specific issues found with the review quality.
        """
        warnings: list[str] = []
        answer_lower = answer.lower()

        # Check for severity markers
        has_severity = bool(self.SEVERITY_PATTERN.search(answer))
        if not has_severity:
            warnings.append(
                "No severity markers found (e.g., [CRITICAL], [HIGH], [MEDIUM], [LOW], or 'Critical:', 'High:', etc.)"
            )

        # Check for file:line references
        has_file_refs = bool(self.FILE_LINE_PATTERN.search(answer))
        if not has_file_refs:
            warnings.append(
                "No file:line references found (e.g., 'file.py:123'). Code reviews should include specific locations."
            )

        # Check if it looks like an architecture summary without issue keywords
        has_architecture_keywords = any(kw in answer_lower for kw in self.ARCHITECTURE_KEYWORDS)
        has_issue_keywords = any(kw in answer_lower for kw in self.ISSUE_KEYWORDS)

        if has_architecture_keywords and not has_issue_keywords:
            warnings.append(
                "Review appears to be an architecture summary without actual code issues. "
                "Expected to find bug reports, security issues, or code quality problems."
            )

        # Determine if the review is valid
        # Valid if: has severity markers OR has file references OR has issue keywords
        is_valid = has_severity or has_file_refs or has_issue_keywords

        return is_valid, warnings

    def _generate_initial_probe(self) -> str:
        """Generate the initial probe code to run before the first iteration.

        This probe executes automatically to provide context to the model about
        the repository structure and guide it toward the right approach.

        Returns:
            Python code to execute as the initial probe.
        """
        return """print("=== RLM Code Review Session ===")
if repo.changed_files:
    mode = "PR Review (" + str(len(repo.changed_files)) + " changed files)"
else:
    mode = "Full Repository Review"
print(f"Mode: {mode}")
print(f"Total files: {len(repo.files)}")
print(f"Total tokens: {repo.total_tokens:,}")
print(f"Languages: {', '.join(repo.languages)}")
if repo.changed_files:
    print("\\nChanged files to review:")
    for f in repo.changed_files[:10]:
        print(f"  - {f}")
    if len(repo.changed_files) > 10:
        print(f"  ... and {len(repo.changed_files) - 10} more")
print("\\n📋 Next steps:")
print("1. Use files_containing(pattern) to find relevant files")
print("2. Use llm_query_parallel(prompts) to batch-analyze files")
print("3. Build findings in a variable, then call FINAL_VAR()")
"""

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
        largest_section = "\n".join(f"  - {path}: ~{tokens:,} tokens" for path, tokens in ctx.largest_files[:5])

        return f"""You are a code review agent with access to a Python REPL environment.

## Repository Context

- **Files**: {ctx.file_count:,} files loaded
- **Languages**: {", ".join(ctx.languages)}
- **Total tokens**: ~{ctx.total_tokens:,} estimated
{f"- **Changed files (PR mode)**: {len(ctx.changed_files)} files" if ctx.changed_files else ""}

### File Preview
{file_preview}

### Largest Files
{largest_section}

## Available Objects and Functions

You have access to the following in the REPL namespace:

### `repo` - Repository Context Object
- `repo.files` - Dict[str, str]: Mapping of file paths to their contents
  **IMPORTANT**: Values are strings directly, NOT objects. Use `repo.files["path"]` NOT `repo.files["path"].content`
  Example: `content = repo.files["main.py"][:1000]  # First 1000 chars`
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

**WARNING**: Do NOT use `FINAL()` with multi-line strings containing triple-quotes,
backticks, or code examples - Python will fail to parse the string literal.

### `FINAL_VAR(var_name: str) -> None`
Signal completion, returning the value of a REPL variable as the final answer.

**RECOMMENDED** for complex reports: Build your report in a variable first:
```
report = '''Your markdown report...'''
FINAL_VAR("report")  # Safe - avoids string parsing issues
```

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

        Uses smart extraction to avoid picking up documentation examples:
        1. Prefer code blocks containing FINAL/FINAL_VAR calls
        2. Skip blocks that look like embedded examples (bare decorators, etc.)
        3. Fall back to the last code block (most likely to be executable)
        4. Sanitize FINAL() calls with embedded backticks

        Args:
            response: LLM response text.

        Returns:
            Extracted Python code, or empty string if no code found.
        """
        matches = list(CODE_BLOCK_PATTERN.finditer(response))
        if not matches:
            return ""

        # Prefer code blocks containing FINAL/FINAL_VAR calls
        for match in matches:
            code = match.group(1).strip()
            if "FINAL(" in code or "FINAL_VAR(" in code:
                # Sanitize FINAL() calls that might have embedded backticks
                return self._sanitize_final_call(code)

        # Filter out blocks that look like documentation examples
        executable_blocks = []
        for match in matches:
            code = match.group(1).strip()
            if not self._is_likely_example(code):
                executable_blocks.append(code)

        # Return the last executable block (most likely to be the intended code)
        if executable_blocks:
            return self._sanitize_final_call(executable_blocks[-1])

        # Fall back to last block if all were filtered
        return self._sanitize_final_call(matches[-1].group(1).strip())

    def _is_likely_example(self, code: str) -> bool:
        """Detect if code looks like a documentation example, not executable code.

        Args:
            code: Python code string to check.

        Returns:
            True if the code looks like an embedded example, not executable code.
        """
        lines = code.strip().split("\n")
        first_line = lines[0].strip() if lines else ""

        # Bare decorator without imports (likely a class definition example)
        if first_line.startswith("@") and "import" not in code:
            # But allow if it's using our REPL functions
            if not any(fn in code for fn in ["repo.", "llm_query", "FINAL", "files_"]):
                return True

        # Very short snippets without REPL functions are likely examples
        if len(code) < 100:
            has_repl_usage = any(
                fn in code for fn in ["print(", "repo.", "llm_query", "FINAL", "files_containing", "files_importing"]
            )
            if not has_repl_usage:
                return True

        return False

    def _sanitize_final_call(self, code: str) -> str:
        """Transform FINAL() calls with embedded backticks to use FINAL_VAR.

        When the agent uses FINAL() with triple-quoted strings containing backticks
        (e.g., markdown code blocks), Python may fail to parse the string literal.
        This method detects such patterns and transforms them to use FINAL_VAR instead.

        Example transformation:
            FINAL('''# Report
            ```code```
            ''')

        Becomes:
            _final_report = '''# Report
            ```code```
            '''
            FINAL_VAR("_final_report")

        Args:
            code: Python code string to sanitize.

        Returns:
            Sanitized code with problematic FINAL() calls transformed.
        """
        # Pattern to match FINAL() with triple-quoted strings
        # Matches: FINAL(""" or FINAL(''' with optional whitespace
        final_triple_pattern = re.compile(
            r'FINAL\s*\(\s*("""|\'\'\')(.*?)\1\s*\)',
            re.DOTALL,
        )

        match = final_triple_pattern.search(code)
        if not match:
            return code

        # Check if the string content contains backticks
        quote_style = match.group(1)
        string_content = match.group(2)

        if "`" not in string_content:
            # No backticks, no transformation needed
            return code

        # Build the replacement: assign to variable and use FINAL_VAR
        full_match = match.group(0)
        variable_assignment = f"_final_report = {quote_style}{string_content}{quote_style}"
        final_var_call = 'FINAL_VAR("_final_report")'

        # Replace the FINAL() call with variable assignment + FINAL_VAR()
        replacement = f"{variable_assignment}\n{final_var_call}"
        sanitized_code = code.replace(full_match, replacement)

        return sanitized_code

    def _handle_llm_query(self, prompt: str, model: str) -> str:
        """Handle llm_query callback from REPL.

        Args:
            prompt: The query prompt.
            model: The model to use (e.g., "haiku").

        Returns:
            LLM response text.
        """
        self._metrics.llm_query_calls += 1
        if self._llm_callback is None:
            return "[Error: No LLM callback configured]"
        return self._llm_callback(prompt, model)

    def _handle_llm_query_parallel(self, prompts: list[str], model: str) -> list[str]:
        """Handle llm_query_parallel callback from REPL.

        Args:
            prompts: List of query prompts.
            model: The model to use (e.g., "haiku").

        Returns:
            List of LLM response texts.
        """
        self._metrics.llm_query_parallel_calls += 1
        # Fall back to sequential execution
        return [self._handle_llm_query(p, model) for p in prompts]

    def _check_sublm_usage_before_final(self) -> str | None:
        """Check if sub-LLM usage meets minimum requirements before accepting FINAL.

        For large repositories (>100k tokens), we require at least MIN_SUBLM_CALLS_FOR_LARGE_REPO
        sub-LLM calls to ensure thorough analysis rather than superficial reviews.

        Returns:
            Warning message if requirements not met, None otherwise.
        """
        if self._context is None:
            return None

        # Check if this is a large repository
        if self._context.total_tokens <= 100_000:
            return None

        # Count total sub-LLM calls
        total_sublm_calls = self._metrics.llm_query_calls + self._metrics.llm_query_parallel_calls

        # Check against minimum requirement
        if total_sublm_calls < MIN_SUBLM_CALLS_FOR_LARGE_REPO:
            return (
                f"⚠️ Sub-LLM Usage Warning: Large repository ({self._context.total_tokens:,} tokens) "
                f"but only {total_sublm_calls} sub-LLM call(s) made so far. "
                f"For thorough analysis, please use llm_query() or llm_query_parallel() "
                f"at least {MIN_SUBLM_CALLS_FOR_LARGE_REPO} times before calling FINAL(). "
                f"This ensures batch processing and comprehensive review rather than superficial analysis."
            )

        return None

    def _handle_files_containing(self, pattern: str) -> list[str]:
        """Handle files_containing callback from REPL.

        Args:
            pattern: Regex pattern to search for.

        Returns:
            List of file paths matching the pattern.
        """
        self._metrics.files_containing_calls += 1
        if self._context is None:
            return []
        compiled = re.compile(pattern)
        results = [path for path, content in self._context.files.items() if compiled.search(content)]
        self._metrics.unique_files_accessed.update(results)
        return results

    def _handle_files_importing(self, module: str) -> list[str]:
        """Handle files_importing callback from REPL.

        Args:
            module: Module name to search for.

        Returns:
            List of file paths that import the module.
        """
        self._metrics.files_importing_calls += 1
        if self._context is None:
            return []
        pattern = rf"(?:^|\n)\s*(?:import\s+{re.escape(module)}|from\s+{re.escape(module)}\s+import)"
        compiled = re.compile(pattern)
        results = [path for path, content in self._context.files.items() if compiled.search(content)]
        self._metrics.unique_files_accessed.update(results)
        return results

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
        parts = []

        # Include conversation history
        if self._history is not None:
            history_section = self._history.format_for_prompt()
            if history_section:
                parts.append(history_section)
                parts.append("")

        parts.append(f"## Current Execution Result (iteration {iteration})")

        if execution_result.output:
            parts.append(f"\n### Output\n```\n{execution_result.output}\n```")

        if execution_result.is_error:
            parts.append(f"\n### Error\n```\n{execution_result.error}\n```")

            error_msg = execution_result.error or ""
            if "unterminated" in error_msg.lower() and "string" in error_msg.lower():
                parts.append(
                    "\n**String Literal Error**: Use `FINAL_VAR()` for complex reports:\n"
                    "```python\n"
                    "report = '''Your report...'''\n"
                    'FINAL_VAR("report")\n'
                    "```"
                )
            else:
                parts.append("\nThe code raised an error. Fix the issue and continue the review.")

        parts.append(
            "\n\n**RESPOND WITH PYTHON CODE ONLY** — no prose or explanations outside code blocks. "
            "Continue your analysis. When ready, call `FINAL()` with your report."
        )
        parts.append("\n\n```python\n# Your next code here\n```")

        return "\n".join(parts)

    def _build_no_code_recovery_prompt(self, iteration: int) -> str:
        """Build recovery prompt when no code is found in LLM response.

        Provides context about the task to help the LLM recover.

        Args:
            iteration: Current iteration number.

        Returns:
            Recovery prompt with task context.
        """
        if self._context is None:
            raise RuntimeError("Context not loaded")

        prompt = f"""## Iteration {iteration} - Code Required

I couldn't find executable Python code in your last response.

**REMINDER**: You are reviewing a codebase with {self._context.file_count} files ({", ".join(self._context.languages)}).
Your goal is to analyze the code and call `FINAL()` with your review report when done.

**Available functions in the REPL**:
- `repo.files` - dict of file paths to contents
- `repo.largest_files` - list of (path, tokens) tuples
- `llm_query(prompt, model="haiku")` - sub-LLM queries
- `files_containing(pattern)` - regex search across files
- `FINAL(report)` - complete the review (simple strings only)
- `FINAL_VAR(varname)` - complete using a variable (recommended for reports)

**Pro Tip**: For markdown reports with code examples:
```python
report = '''# Code Review Report
## Findings...'''
FINAL_VAR("report")
```

Please provide Python code in a fenced code block:
```python
# Continue your code review analysis
# When done, call FINAL("Your review report here")
```"""

        # Show top-level paths to help orientation
        if self._context:
            top_level = sorted(set(p.split("/")[0] for p in self._context.files.keys()))[:10]
            if top_level:
                prompt += f"\n\n**Available top-level paths**: {', '.join(top_level)}"

        return prompt

    async def run(self) -> str:
        """Execute the RLM code review.

        Returns:
            Final review report.
        """
        # Load codebase
        workspace = Path(self.config.workspace_path)
        changed_files = get_changed_files(workspace)
        self._context = load_codebase(
            workspace,
            self.config.languages,
            changed_files=changed_files,
        )

        # Initialize conversation history
        self._history = ConversationHistory(
            recent_count=3,
        )

        # Container lifecycle
        container: DevContainer | None = None
        if self.config.use_container:
            container = DevContainer(workspace)

        try:
            if container:
                await container.start()

            # If no LLM callback is set, use a default that returns a placeholder
            if self._call_llm is None:
                # Default implementation - can be overridden for testing or real usage
                return await self._run_with_default_llm()

            # Set up the llm_query callback for the REPL
            if self._llm_callback is None:
                # Use a simple echo for testing
                self._llm_callback = lambda prompt, model: f"[Sub-LLM response to: {prompt[:50]}...]"

            # Create REPL with instrumented callbacks for metrics tracking
            self._repl = REPLProcess(
                context=self._context,
                llm_callback=lambda p, m: self._handle_llm_query(p, m),
                llm_parallel_callback=lambda ps, m: self._handle_llm_query_parallel(ps, m),
            )

            self._repl.start()

            # Execute initial probe to set context before the first iteration
            initial_probe = self._generate_initial_probe()
            self._emit_event(
                0,
                "initial_probe",
                {
                    "code_length": len(initial_probe),
                },
            )

            # Run the probe in a worker thread (to avoid potential deadlocks)
            initial_result = await asyncio.to_thread(self._repl.execute, initial_probe)

            # Add probe to history as iteration 0
            self._history.add_exchange(
                iteration=0,
                code=initial_probe,
                output=initial_result.output or "",
                error=initial_result.error,
            )

            self._emit_event(
                0,
                "initial_probe_executed",
                {
                    "output_length": len(initial_result.output) if initial_result.output else 0,
                    "output_preview": initial_result.output if initial_result.output else "",
                    "is_error": initial_result.is_error,
                },
            )

            # Build initial system prompt using the comprehensive review prompt
            system_prompt = get_review_prompt(
                file_count=self._context.file_count,
                total_tokens=self._context.total_tokens,
                languages=self._context.languages,
                largest_files=self._context.largest_files,
                changed_files=self._context.changed_files,
            )

            # Include probe output in first prompt
            if initial_result.output:
                current_prompt = (
                    f"{system_prompt}\n\n"
                    f"## Initial Repository Scan\n\n"
                    f"```\n{initial_result.output}\n```\n\n"
                    f"Based on the above scan, generate Python code to continue the review."
                )
            else:
                current_prompt = system_prompt

            final_result: str | None = None
            iteration = 0
            consecutive_no_code = 0

            while iteration < self.config.max_iterations:
                iteration += 1
                self._metrics.iterations = iteration

                self._emit_event(
                    iteration,
                    "iteration_start",
                    {
                        "max_iterations": self.config.max_iterations,
                    },
                )

                # Get code from LLM
                response = await self._call_llm(current_prompt)
                code = self._extract_code(response)

                self._emit_event(
                    iteration,
                    "code_extracted",
                    {
                        "code_length": len(code) if code else 0,
                        "code_preview": code if code else "",
                        "has_code": bool(code),
                    },
                )

                if not code:
                    self._metrics.code_extraction_failures += 1
                    # No code found - try to extract any runnable content
                    # or ask for clarification
                    if "FINAL(" in response:
                        final_match = FINAL_CALL_PATTERN.search(response)
                        if final_match:
                            code = final_match.group(0)
                            self._emit_event(
                                iteration,
                                "fallback_final_detection",
                                {
                                    "response_length": len(response),
                                    "extracted_final": code,
                                },
                            )
                        else:
                            # Could not extract FINAL cleanly, treat as no code
                            consecutive_no_code += 1

                            if consecutive_no_code >= MAX_CONSECUTIVE_ERRORS:
                                self._emit_event(
                                    iteration,
                                    "consecutive_error_limit",
                                    {
                                        "consecutive_errors": consecutive_no_code,
                                    },
                                )
                                final_result = (
                                    "# Code Review (Failed)\n\n"
                                    f"Review failed after {consecutive_no_code} consecutive iterations "
                                    "without valid Python code.\n"
                                    "The model could not provide executable code in the expected format."
                                )
                                break

                            self._emit_event(
                                iteration,
                                "no_code_found",
                                {
                                    "response_preview": response[:200],
                                    "reason": "FINAL detected but could not extract cleanly",
                                    "consecutive_count": consecutive_no_code,
                                },
                            )
                            current_prompt = self._build_no_code_recovery_prompt(iteration)
                            continue
                    else:
                        # No executable code found
                        consecutive_no_code += 1

                        if consecutive_no_code >= MAX_CONSECUTIVE_ERRORS:
                            self._emit_event(
                                iteration,
                                "consecutive_error_limit",
                                {
                                    "consecutive_errors": consecutive_no_code,
                                },
                            )
                            final_result = (
                                "# Code Review (Failed)\n\n"
                                f"Review failed after {consecutive_no_code} consecutive iterations "
                                "without valid Python code.\n"
                                "The model could not provide executable code in the expected format."
                            )
                            break

                        self._emit_event(
                            iteration,
                            "no_code_found",
                            {
                                "response_preview": response[:200] if response else "",
                                "consecutive_count": consecutive_no_code,
                            },
                        )
                        current_prompt = self._build_no_code_recovery_prompt(iteration)
                        continue

                # Reset consecutive error counter on successful code extraction
                consecutive_no_code = 0

                # Execute code in REPL
                result = await asyncio.to_thread(self._repl.execute, code)

                # Add exchange to history
                self._history.add_exchange(
                    iteration=iteration,
                    code=code,
                    output=result.output or "",
                    error=result.error,
                )

                self._emit_event(
                    iteration,
                    "repl_executed",
                    {
                        "output_length": len(result.output) if result.output else 0,
                        "output_preview": (result.output if result.output else ""),
                        "is_error": result.is_error,
                        "error": result.error if result.is_error else None,
                        "is_final": result.is_final,
                    },
                )

                # Check for final answer
                if result.is_final:
                    self._metrics.final_call_attempts += 1

                    # Check if sub-LLM usage meets requirements before accepting FINAL
                    sublm_warning = self._check_sublm_usage_before_final()
                    if sublm_warning:
                        # Log the warning and continue instead of accepting FINAL
                        logger.warning(sublm_warning)
                        self._emit_event(
                            iteration,
                            "sublm_usage_warning",
                            {
                                "warning": sublm_warning,
                                "llm_query_count": self._metrics.llm_query_calls,
                                "llm_query_parallel_count": self._metrics.llm_query_parallel_calls,
                            },
                        )

                        # Add warning to conversation so the model knows to use sub-LLM queries
                        current_prompt = self._build_continuation_prompt(iteration, result)
                        current_prompt = f"{sublm_warning}\n\n{current_prompt}"
                        continue

                    final_result = result.final_answer
                    self._emit_event(
                        iteration,
                        "final_answer",
                        {
                            "answer_length": len(final_result) if final_result else 0,
                        },
                    )
                    break

                # Build continuation prompt with execution result
                current_prompt = self._build_continuation_prompt(iteration, result)

            # If we hit max iterations without FINAL, generate a summary
            if final_result is None:
                self._emit_event(
                    iteration,
                    "max_iterations_reached",
                    {
                        "iterations_completed": iteration,
                        "max_iterations": self.config.max_iterations,
                    },
                )
                final_result = (
                    f"# Code Review (Incomplete)\n\n"
                    f"Review stopped after {iteration} iterations without completion.\n"
                    f"Please review the execution log for partial findings."
                )

            # Validate review quality and emit warning if issues found
            is_valid, warnings = self._validate_review_quality(final_result)
            if not is_valid:
                self._emit_event(iteration, "review_quality_warning", {"warnings": warnings})

            # Emit metrics
            self._emit_event(
                iteration,
                "metrics",
                {
                    "iterations": self._metrics.iterations,
                    "llm_query_calls": self._metrics.llm_query_calls,
                    "llm_query_parallel_calls": self._metrics.llm_query_parallel_calls,
                    "files_containing_calls": self._metrics.files_containing_calls,
                    "files_importing_calls": self._metrics.files_importing_calls,
                    "repo_files_accesses": self._metrics.repo_files_accesses,
                    "unique_files_accessed": len(self._metrics.unique_files_accessed),
                    "code_extraction_failures": self._metrics.code_extraction_failures,
                    "final_call_attempts": self._metrics.final_call_attempts,
                },
            )

            # Log metrics summary
            logger.info(
                "RLM Metrics: iterations=%d, llm_query=%d, llm_query_parallel=%d, "
                "files_containing=%d, files_importing=%d, unique_files=%d, "
                "extraction_failures=%d, final_attempts=%d",
                self._metrics.iterations,
                self._metrics.llm_query_calls,
                self._metrics.llm_query_parallel_calls,
                self._metrics.files_containing_calls,
                self._metrics.files_importing_calls,
                len(self._metrics.unique_files_accessed),
                self._metrics.code_extraction_failures,
                self._metrics.final_call_attempts,
            )

            return final_result

        finally:
            if self._repl is not None:
                self._repl.stop()
                self._repl = None
            if container:
                await container.stop()

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
