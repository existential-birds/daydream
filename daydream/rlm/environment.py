# daydream/rlm/environment.py
"""REPL environment data structures.

This module defines the data structures available to model-generated code
in the sandboxed REPL environment.
"""

from dataclasses import dataclass


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
