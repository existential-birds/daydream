"""Shared full-diff reads and changed-file parsing for deep and Improve flows."""

from __future__ import annotations

from pathlib import Path

from daydream.agent import console
from daydream.deep.prompts import _DIFF_BLOCK_SPLIT, _diff_block_path
from daydream.deep.state import DeepState
from daydream.flows.engine import FlowContext
from daydream.ui import print_warning


def _diff_changed_files(diff: str) -> list[str]:
    """Extract changed files from a unified diff.

    Parses one file per ``diff --git`` block and contributes a single path
    for each. Prefers the post-state path (``+++ b/<path>``) so renames
    produce only the destination. Falls back to the pre-state path for
    deletions (``+++ /dev/null``) and to the ``diff --git`` header for
    binary / mode-only diffs that lack ``---``/``+++`` lines.

    The per-block path resolution is delegated to ``_diff_block_path`` so the
    unified-diff parsing contract is shared with ``_diff_blocks_for_files``
    rather than duplicated here.

    Returns:
        Unique, insertion-ordered list of changed file paths (excluding
        ``/dev/null`` sentinels).
    """
    files: list[str] = []
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        path = _diff_block_path(block)
        if path and path not in files:
            files.append(path)
    return files


def _read_full_diff(ctx: FlowContext) -> str:
    """Read the full on-disk PR diff (``ctx.data["diff_path"]``).

    Issue #644 — the gather-time bound shrinks the in-memory
    ``ctx.data["diff"]`` to ``INLINE_DIFF_BUDGET_BYTES`` via whole-block
    retention, so consumers that need the COMPLETE diff (the exploration
    pre-scan, the intent/wonder TTT phases, and the uncovered sweep) must
    re-read ``diff_path``, which is always written full at gather. This is
    that single re-read site.

    Raises ``OSError`` on a failed read so each caller keeps its own
    degradation shape (warn-and-fallback for exploration / TTT,
    propagate-to-the-fail-open-wrapper for the sweep). When the ctx carries no
    ``diff_path`` (defensive legacy fallback only, never the default), the
    bounded in-memory ``ctx.data["diff"]`` is returned instead of crashing.
    """
    deep_state = DeepState(ctx.data)
    diff_path = deep_state.diff_path_or_none
    if diff_path is None:
        return str(deep_state.diff)
    return Path(diff_path).read_text(encoding="utf-8")


def _ttt_diff_text(ctx: FlowContext) -> str | None:
    """The diff text handed to the TTT phases (intent/wonder).

    Issue #644 — the gather-time bound shrinks an over-budget diff to whole
    retained blocks so ``ctx.data["diff"]`` fits the inline prompt budget, but
    the inline prompts promise "the complete diff is inlined below (do NOT
    re-Read ``diff.patch``)". Inlining the bounded text would make that claim
    over a truncated diff with no path to the dropped blocks, so when the diff
    WAS truncated the FULL on-disk diff (``ctx.data["diff_path"]``, always
    written full at gather) is handed over instead: it never fits the inline
    budget, so worktree-cwd backends keep the ``diff.patch`` pointer and
    disposable-clone read-only backends keep their documented truncated-inline
    fallback. The bounded text is therefore only ever inlined when it genuinely
    IS the complete diff. A full-diff read failure degrades to the bounded text
    with a warning (defensive only — ``diff.patch`` was written this run).
    """
    deep_state = DeepState(ctx.data)
    if not deep_state.diff_truncated:
        return str(deep_state.diff)
    try:
        return _read_full_diff(ctx)
    except OSError as exc:
        truncation = deep_state.diff_truncation
        detail = (
            f" ({truncation.marker.strip()})"
            if truncation is not None and truncation.marker is not None
            else ""
        )
        print_warning(
            console,
            f"Could not read the full diff for the intent/wonder phases "
            f"({exc}); falling back to the bounded in-memory diff{detail}",
        )
    return str(deep_state.diff)
