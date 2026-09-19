"""Lazy console warning shared by the archive modules."""

from __future__ import annotations


def warn(message: str) -> None:
    """Print a one-line warning through the daydream console (never raises)."""
    from daydream.ui import create_console, print_warning

    print_warning(create_console(), message)
