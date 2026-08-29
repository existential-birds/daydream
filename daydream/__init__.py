"""Daydream - Automated code review and fix loop.

Provide an automated workflow for code review, issue parsing, fix application,
and test verification using a configured AI backend. The package orchestrates
a continuous loop that reviews code, identifies issues, applies fixes, and
validates changes through testing.

Exports:
    __version__: str - The current version of the daydream package.

Submodules:
    agent: Backend client and helper functions for AI interactions.
    cli: Command-line interface with entry point and signal handling.
    config: Configuration constants and settings.
    phases: Review, parse, fix, and test phase implementations.
    runner: Main orchestration logic for the review-fix loop.
    ui: User interface utilities for terminal output.
"""

__version__ = "0.28.0"
