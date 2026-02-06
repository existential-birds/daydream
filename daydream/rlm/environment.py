# daydream/rlm/environment.py
"""REPL environment data structures.

This module defines the data structures available to model-generated code
in the sandboxed REPL environment.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class FileInfo:
    """Parsed metadata about a source file.

    Attributes:
        language: Programming language ("python", "typescript", "go").
        functions: List of function/method names defined in the file.
        classes: List of class/struct/interface names defined in the file.
        imports: List of import statements or imported modules.
        exports: List of exported symbols (primarily for TS/Go).
    """

    language: str
    functions: list[str]
    classes: list[str]
    imports: list[str]
    exports: list[str]


@dataclass
class Service:
    """Metadata about a detected service boundary.

    Attributes:
        name: Service identifier (e.g., "auth", "billing").
        root: Root directory path (e.g., "services/auth").
        files: List of all file paths belonging to this service.
        dependencies: List of other service names this service imports from.
    """

    name: str
    root: str
    files: list[str]
    dependencies: list[str]


@dataclass
class RepoContext:
    """Complete codebase context available to the REPL.

    This is the `repo` object exposed in the REPL namespace.

    Attributes:
        files: Mapping of file paths to their contents.
        structure: Mapping of file paths to parsed FileInfo.
        services: Mapping of service names to Service metadata.
        file_sizes: Mapping of file paths to token counts.
        total_tokens: Total token count across all files.
        file_count: Number of files in the repository.
        largest_files: Top files by token count as (path, tokens) tuples.
        languages: List of detected programming languages.
        changed_files: Files changed in PR (None for full repo review).
    """

    files: dict[str, str]
    structure: dict[str, FileInfo]
    services: dict[str, Service]
    file_sizes: dict[str, int]
    total_tokens: int
    file_count: int
    largest_files: list[tuple[str, int]]
    languages: list[str]
    changed_files: list[str] | None


class FinalAnswer(Exception):
    """Raised when FINAL() or FINAL_VAR() is called to signal completion.

    Attributes:
        answer: The final answer string to return to the user.
    """

    def __init__(self, answer: str):
        self.answer = answer
        super().__init__(answer)


def build_repl_namespace(
    ctx: RepoContext,
    llm_query_fn: Callable[[str, str], str],
    llm_query_parallel_fn: Callable[[list[str], str], list[str]] | None = None,
) -> dict[str, Any]:
    """Build the namespace dict for the REPL environment.

    Creates a dictionary containing all objects and functions available
    to model-generated code in the REPL.

    Args:
        ctx: The RepoContext with codebase data.
        llm_query_fn: Function to call for llm_query(prompt, model).
        llm_query_parallel_fn: Optional function for parallel queries.
            If None, falls back to sequential calls.

    Returns:
        Dictionary to use as REPL globals/locals.
    """
    namespace: dict[str, Any] = {}

    # Expose repo context
    namespace["repo"] = ctx

    # Wrap llm_query with default model parameter
    def llm_query(prompt: str, model: str = "haiku", **kwargs) -> str:
        """Fresh-context sub-LLM call. Returns response text."""
        # Handle common misuse: model often hallucinates 'context' parameter
        if "context" in kwargs:
            prompt = f"{prompt}\n\n{kwargs['context']}"
        return llm_query_fn(prompt, model)

    namespace["llm_query"] = llm_query

    # Parallel query function
    if llm_query_parallel_fn is not None:
        def llm_query_parallel(prompts: list[str], model: str = "haiku") -> list[str]:
            """Batch multiple independent queries for efficiency."""
            return llm_query_parallel_fn(prompts, model)
    else:
        def llm_query_parallel(prompts: list[str], model: str = "haiku") -> list[str]:
            """Fallback: execute queries sequentially."""
            return [llm_query_fn(p, model) for p in prompts]

    namespace["llm_query_parallel"] = llm_query_parallel

    # Search function: files_containing
    def files_containing(pattern: str) -> list[str]:
        """Grep-like regex search, returns matching file paths."""
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            print(f"[Warning] Invalid regex pattern: {e}")
            return []
        return [path for path, content in ctx.files.items() if compiled.search(content)]

    namespace["files_containing"] = files_containing

    # Search function: files_importing
    def files_importing(module: str) -> list[str]:
        """Find files that import a given module."""
        # Match: import X, from X import, import X as Y
        pattern = rf"(?:^|\n)\s*(?:import\s+{re.escape(module)}|from\s+{re.escape(module)}\s+import)"
        compiled = re.compile(pattern)
        return [path for path, content in ctx.files.items() if compiled.search(content)]

    namespace["files_importing"] = files_importing

    # File existence check
    def file_exists(path: str) -> bool:
        """Check if a file exists in the repository."""
        return path in ctx.files

    namespace["file_exists"] = file_exists

    # Glob-like file matching
    def list_files_matching(pattern: str) -> list[str]:
        """List files matching a glob-like pattern (e.g., 'src/*.py', '**/*.ts')."""
        import fnmatch
        return [p for p in ctx.files.keys() if fnmatch.fnmatch(p, pattern)]

    namespace["list_files_matching"] = list_files_matching

    # File slice function
    def get_file_slice(path: str, start_line: int, end_line: int) -> str:
        """Get specific line range from a file (1-based, inclusive)."""
        content = ctx.files.get(path, "")
        if not content:
            return ""
        lines = content.split("\n")
        # Clamp to valid range (1-based input)
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        selected = lines[start_idx:end_idx]
        return "\n".join(selected)

    namespace["get_file_slice"] = get_file_slice

    # FINAL function - signals completion with direct answer
    def FINAL(answer: str) -> None:
        """Signal task completion with direct answer."""
        # Convert to string in case model passes dict/list/etc
        raise FinalAnswer(str(answer))

    namespace["FINAL"] = FINAL

    # FINAL_VAR function - signals completion using a variable
    def FINAL_VAR(var_name: str) -> None:
        """Signal task completion, returning a REPL variable as output.

        Note: Uses the shared namespace dict which is mutated by exec().
        Variables assigned in code blocks will be available here.
        """
        if var_name not in namespace:
            # Provide diagnostic info to help debug missing variables
            available_vars = [
                k for k in namespace.keys()
                if not k.startswith("_") and not callable(namespace.get(k))
            ]
            # Filter out built-in functions we injected
            builtin_funcs = {
                "repo", "llm_query", "llm_query_parallel", "files_containing",
                "files_importing", "file_exists", "list_files_matching",
                "get_file_slice", "FINAL", "FINAL_VAR",
            }
            user_vars = [v for v in available_vars if v not in builtin_funcs]
            hint = f"Available user variables: {user_vars}" if user_vars else "No user variables defined"
            raise FinalAnswer(
                f"[FINAL_VAR Error] Variable '{var_name}' not found in namespace. {hint}. "
                f"Ensure the variable is assigned before calling FINAL_VAR('{var_name}')."
            )
        raise FinalAnswer(str(namespace[var_name]))

    namespace["FINAL_VAR"] = FINAL_VAR

    return namespace
