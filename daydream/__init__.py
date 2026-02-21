"""Daydream - Automated code review and fix loop using Claude.

Provide an automated workflow for code review, issue parsing, fix application,
and test verification using Claude as the AI backend. The package orchestrates
a continuous loop that reviews code, identifies issues, applies fixes, and
validates changes through testing.

Exports:
    __version__: str - The current version of the daydream package.

Submodules:
    agent: Claude SDK client and helper functions for AI interactions.
    cli: Command-line interface with entry point and signal handling.
    config: Configuration constants and settings.
    phases: Review, parse, fix, and test phase implementations.
    runner: Main orchestration logic for the review-fix loop.
    ui: User interface utilities for terminal output.
"""

__version__ = "0.7.0"
