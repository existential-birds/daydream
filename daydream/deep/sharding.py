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
        if stack.stack_name == STRUCTURE_STACK_NAME or len(stack.files) <= max_files:
            out.append(stack)
        else:
            out.extend(_split_by_file_count(stack, max_files))
    return out
