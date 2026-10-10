"""Shared full-diff reads and changed-file parsing for deep and Improve flows."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from daydream.agent import console
from daydream.deep.state import DeepState
from daydream.flows.engine import FlowContext
from daydream.hunk_index import _unquote_git_path
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES, fits_inline_diff_budget
from daydream.ui import print_warning

_DIFF_BLOCK_SPLIT = re.compile(r"^(?=diff --git )", re.MULTILINE)
_DIFF_PLUS_HEADER = re.compile(r"^\+\+\+ (.+)$", re.MULTILINE)
_DIFF_MINUS_HEADER = re.compile(r"^--- (.+)$", re.MULTILINE)
_DIFF_GIT_HEADER = re.compile(r'^diff --git ("(?:[^"\\]|\\.)*"|a/.+?) ("(?:[^"\\]|\\.)*"|b/.+)$', re.MULTILINE)
_DIFF_RENAME_HEADER = re.compile(r"^rename to (.+)$", re.MULTILINE)
# Markers carry dropped paths without matching any diff header, so all block
# consumers skip the marker while file selection can reject partial input.
_DIFF_TRUNCATION_MARKER = re.compile(
    r"^# daydream: deep diff truncated: \d+ -> \d+ bytes "
    r"\(\d+/\d+ blocks retained(?:; dropped: (?P<dropped>[^)]+))?\)\n"
)


def _diff_block_path(block: str) -> str | None:
    """Resolve the changed path, preferring the post-state destination on renames.

    Deletions use the pre-state path; binary and mode-only changes use the git
    header. Skip /dev/null sentinels and non-diff fragments.
    """

    if not block.startswith("diff --git "):
        return None
    for pattern, prefix in ((_DIFF_PLUS_HEADER, "b/"), (_DIFF_MINUS_HEADER, "a/")):
        match = pattern.search(block)
        if match:
            path = _unquote_git_path(match.group(1).rstrip("\t"))
            if path != "/dev/null":
                return path.removeprefix(prefix)
    renamed = _DIFF_RENAME_HEADER.search(block)
    if renamed:
        return _unquote_git_path(renamed.group(1))
    # Unquoted Git paths may contain spaces or " b/"; symmetric a/ and b/ halves resolve unchanged identities.
    header = block.partition("\n")[0].removeprefix("diff --git ")
    midpoint = len(header) // 2
    before, after = header[:midpoint], header[midpoint + 1:]
    if (header[midpoint:midpoint + 1] == " " and before.startswith("a/")
            and after.startswith("b/") and before[2:] == after[2:]):
        return after[2:]
    git = _DIFF_GIT_HEADER.match(block)
    return _unquote_git_path(git.group(2)).removeprefix("b/") if git else None


def iter_diff_blocks(diff: str) -> Iterator[tuple[str, str]]:
    """Yield resolved paths and untouched file blocks in source order."""
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        path = _diff_block_path(block)
        if path is not None:
            yield path, block


def _diff_blocks_for_files(diff: str, files: list[str]) -> str | None:
    """Select complete file blocks within the inline byte budget.

    Return None when no blocks match, the result exceeds the budget, or a wanted
    file appears in the truncation marker's dropped list. A wanted file absent
    from the original diff is harmless; a dropped block would hide changed hunks.
    """
    wanted = set(files)
    if not wanted:
        return None

    marker = _DIFF_TRUNCATION_MARKER.match(diff)
    if marker is not None and marker.group("dropped") is not None:
        dropped = {p.strip() for p in marker.group("dropped").split(",") if p.strip()}
        if wanted & dropped:
            return None

    result = "".join(
        block if block.endswith("\n") else block + "\n"
        for path, block in iter_diff_blocks(diff)
        if path in wanted
    )
    return result if result and fits_inline_diff_budget(result) else None


@dataclass
class DeepDiffBoundInfo:
    """Whole-block truncation statistics; retained_bytes excludes the marker.

    Only a leading oversized block survives the soft cap and enters oversize_paths.
    Later oversized blocks enter dropped_paths, also written into the marker.
    """

    truncated: bool
    original_bytes: int = 0
    retained_bytes: int = 0
    total_blocks: int = 0
    retained_blocks: int = 0
    oversize_paths: list[str] = field(default_factory=list)
    dropped_paths: list[str] = field(default_factory=list)
    marker: str | None = None


def bound_deep_diff(diff: str, budget: int = INLINE_DIFF_BUDGET_BYTES) -> tuple[str, DeepDiffBoundInfo]:
    """Bound a diff by retaining whole, byte-identical file blocks.

    Fitting input is unchanged. A leading oversized block survives the soft cap;
    other blocks fit only while their addition stays within budget. Unresolved
    fragments are skipped. The marker names dropped paths so prompt extraction
    can reject partially retained stack scopes.
    """
    original_bytes = len(diff.encode("utf-8"))
    if original_bytes <= budget:
        return diff, DeepDiffBoundInfo(truncated=False, original_bytes=original_bytes, marker=None)

    retained: list[str] = []
    retained_bytes = 0
    total_blocks = 0
    oversize_paths: list[str] = []
    dropped_paths: list[str] = []
    for block_path, block in iter_diff_blocks(diff):
        total_blocks += 1
        block_bytes = len(block.encode("utf-8"))
        fits = retained_bytes + block_bytes <= budget
        if fits or not retained:
            retained.append(block)
            retained_bytes += block_bytes
            if not fits:
                oversize_paths.append(block_path)
        else:
            dropped_paths.append(block_path)

    dropped_clause = f"; dropped: {', '.join(dropped_paths)}" if dropped_paths else ""
    marker = (
        f"# daydream: deep diff truncated: {original_bytes} -> {retained_bytes} bytes "
        f"({len(retained)}/{total_blocks} blocks retained{dropped_clause})\n"
    )
    bounded = marker + "".join(retained)
    return bounded, DeepDiffBoundInfo(
        truncated=True,
        original_bytes=original_bytes,
        retained_bytes=retained_bytes,
        total_blocks=total_blocks,
        retained_blocks=len(retained),
        oversize_paths=oversize_paths,
        dropped_paths=dropped_paths,
        marker=marker,
    )


def _full_diff_pointer(diff_path: Path) -> str:
    """Shared paragraph pointing agents at the on-disk full PR diff."""
    return (
        f"The full PR diff (base..HEAD) is at {diff_path}. Read it directly; "
        "do NOT run `git diff` without a base ref -- on a clean branch that "
        "returns empty and hides committed changes."
    )


def _hunk_index_authority(diff_path: Path) -> str:
    """Name the hunk index adjacent to the routed full-diff artifact."""
    return (
        f"Changed line ranges are authoritative in `{diff_path.parent / 'hunk-index.json'}` "
        "— do not re-derive them with `git diff` (the index is written once at "
        "gather and is the single persisted source of changed-file/line ranges)."
    )


def _diff_changed_files(diff: str) -> list[str]:
    """Return unique changed paths in diff order."""
    return list(dict.fromkeys(path for path, _ in iter_diff_blocks(diff) if path))


def _read_full_diff(ctx: FlowContext) -> str:
    """Read the full persisted diff; fall back to memory when no path exists.

    The in-memory diff may be bounded, so callers needing complete evidence must
    use this read. OSError propagates for callers to choose their fallback.
    """
    deep_state = DeepState(ctx.data)
    diff_path = deep_state.diff_path_or_none
    if diff_path is None:
        return str(deep_state.diff)
    return Path(diff_path).read_text(encoding="utf-8")


def _ttt_diff_text(ctx: FlowContext) -> str | None:
    """Give intent/wonder the complete diff so inline claims remain truthful.

    A truncated in-memory diff is replaced by the full artifact, which exceeds
    the inline budget and uses the backend's pointer or truncated-clone transport.
    Warn and use the bounded value only if the artifact cannot be read.
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
