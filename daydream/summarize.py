"""Render an evidence document or run directory as the PR-comment run-info block.

Directory input includes the root and sibling evidence documents. Print the rollup,
phase table and version footer without GitHub writes or diff filtering."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from daydream.eval.analyzer import collect_trajectory_paths
from daydream.pr_comment_renderer import FALLBACK_NOTE, render_run_info_block
from daydream.trajectory import DOCUMENT_SUFFIX, RUN_DOCUMENT_NAME, SIBLINGS_DIRNAME
from daydream.ui import NEON_THEME, print_error


def summarize(path: Path) -> int:
    """Print run-info markdown; return nonzero for unresolved or unparseable input."""
    # Errors go to stderr so stdout stays clean for the markdown summary.
    err_console = Console(stderr=True, theme=NEON_THEME)

    if not path.exists():
        print_error(err_console, "Invalid Path", f"path does not exist: {path}")
        return 1

    if path.is_file():
        if not path.name.endswith(DOCUMENT_SUFFIX):
            print_error(
                err_console,
                "Invalid Path",
                f"file must be a {DOCUMENT_SUFFIX} evidence document: {path}",
            )
            return 1
        paths = [path]
    elif path.is_dir():
        paths = collect_trajectory_paths(path)
        if not paths:
            print_error(
                err_console,
                "Invalid Path",
                f"no {RUN_DOCUMENT_NAME} or {SIBLINGS_DIRNAME}/ found under {path}",
            )
            return 1
    else:
        print_error(
            err_console,
            "Invalid Path",
            f"path is not a file or directory: {path}",
        )
        return 1

    output = render_run_info_block(paths)
    print(output)

    # Renderer fell back because every input failed to parse — surface a
    # non-zero exit so callers know they didn't get a real summary.
    if FALLBACK_NOTE in output:
        return 2
    return 0


__all__ = ["summarize"]
