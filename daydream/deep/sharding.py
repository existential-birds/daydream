"""Deterministic deep-review stack sharding (issue #731).

A shard is a :class:`~daydream.deep.detection.StackAssignment` carrying a
synthetic ``stack_name`` (``python#0``, ``python#1``) so it rides the existing
``stack_name``-keyed pipeline (artifact paths, sorted parse/merge ordering, the
capacity limiter) unchanged. ``shard_stacks`` is pure and deterministic: the
same inputs yield the same shard names and assignments.

Only *non-structural* stacks are shardable; the ``structure`` meta-stack is
passed through unchanged (same object). The structural stack carries the union
of all changed files and must never be split or counted against the fan-out cap.
"""

from __future__ import annotations

from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import StackAssignment
from daydream.deep.prompts import _DIFF_BLOCK_SPLIT, _diff_block_path


def _per_file_change_bytes(diff: str, file: str) -> int:
    """Return the changed-byte size of ``file`` from a full unified ``diff``.

    Splits ``diff`` into ``diff --git`` blocks and returns ``len(block.encode(
    "utf-8"))`` for the block whose post-state path matches ``file``. A file
    with no resolvable block sizes as 1 byte (still assigned, never dropped).
    Mirrors the shared ``_diff_block_path`` / ``_DIFF_BLOCK_SPLIT`` parse used
    by ``prompts._diff_blocks_for_files``.
    """
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        if _diff_block_path(block) == file:
            return len(block.encode("utf-8"))
    return 1


def _changed_bytes(diff: str, files: list[str]) -> int:
    """Total changed-byte size of ``files`` across ``diff``."""
    return sum(_per_file_change_bytes(diff, f) for f in files)


def _split_by_file_count(stack: StackAssignment, max_files: int) -> list[StackAssignment]:
    """Split ``stack`` into consecutive ``max_files``-sized shards in sorted order."""
    files = sorted(stack.files)
    shards: list[StackAssignment] = []
    for start in range(0, len(files), max_files):
        chunk = files[start : start + max_files]
        shards.append(
            StackAssignment(
                stack_name=f"{stack.stack_name}#{len(shards)}",
                skill_invocation=stack.skill_invocation,
                files=chunk,
                is_docs_only=stack.is_docs_only,
                frontier_files=[],
            )
        )
    return shards


def _split_by_bytes(
    stack: StackAssignment, diff: str, max_bytes: int, max_files: int
) -> list[StackAssignment]:
    """Greedily pack ``stack``'s sorted files into shards up to ``max_bytes``.

    Fill shards in sorted-file order up to ``max_bytes``; a single oversized file
    forms its own shard (never split a file). Files-per-shard never exceeds
    ``max_files`` either.
    """
    files = sorted(stack.files)
    shards: list[StackAssignment] = []
    current: list[str] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if current:
            shards.append(
                StackAssignment(
                    stack_name=f"{stack.stack_name}#{len(shards)}",
                    skill_invocation=stack.skill_invocation,
                    files=current,
                    is_docs_only=stack.is_docs_only,
                    frontier_files=[],
                )
            )
        current = []
        current_bytes = 0

    for file in files:
        size = _per_file_change_bytes(diff, file)
        fits = (
            len(current) < max_files
            and (current_bytes == 0 or current_bytes + size <= max_bytes)
        )
        if current and not fits:
            flush()
        current.append(file)
        current_bytes += size
    flush()
    return shards



def shard_stacks(
    stacks: list[StackAssignment],
    diff: str,
    *,
    max_files: int,
    max_bytes: int,
    fanout_cap: int,
    frontier_max: int,
    graph: dict[str, set[str]] | None = None,
) -> list[StackAssignment]:
    """Split oversized per-language stacks into bounded shards.

    Pure and deterministic. Non-structural stacks whose file count exceeds
    ``max_files`` are split into consecutive ``<name>#<i>`` shards in sorted-file
    order; the union of all shard file sets equals the source stack's set with no
    duplicates. Stacks at/under every bound are returned unsplit with their
    original ``stack_name``. The structural meta-stack is passed through
    unchanged.

    The byte bound (``max_bytes``), total fan-out cap (``fanout_cap``),
    dependency-aware co-location and bounded frontier (``graph`` /
    ``frontier_max``) are filled in by subsequent tasks; this task implements
    the file-count split and deterministic naming.
    """
    out: list[StackAssignment] = []
    for stack in stacks:
        if stack.stack_name == STRUCTURE_STACK_NAME:
            out.append(stack)
        elif len(stack.files) <= max_files and _changed_bytes(diff, stack.files) <= max_bytes:
            out.append(stack)
        elif len(stack.files) > max_files:
            out.extend(_split_by_file_count(stack, max_files))
        else:
            out.extend(_split_by_bytes(stack, diff, max_bytes, max_files))
    return out
