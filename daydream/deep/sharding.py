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
from daydream.deep.dependency import co_locate_groups
from daydream.deep.detection import StackAssignment
from daydream.deep.prompts import _DIFF_BLOCK_SPLIT, _diff_block_path


def _file_change_bytes(diff: str) -> dict[str, int]:
    """Map every changed file to its changed-byte size (single diff parse).

    Splits ``diff`` into ``diff --git`` blocks exactly once and records
    ``len(block.encode("utf-8"))`` under the block's post-state path (first
    matching block wins). A file absent from the map sizes as 1 byte (still
    assigned, never dropped). Mirrors the shared ``_diff_block_path`` /
    ``_DIFF_BLOCK_SPLIT`` parse used by ``prompts._diff_blocks_for_files``.
    """
    sizes: dict[str, int] = {}
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        path = _diff_block_path(block)
        if path is not None:
            sizes.setdefault(path, len(block.encode("utf-8")))
    return sizes


def _changed_bytes(sizes: dict[str, int], files: list[str]) -> int:
    """Total changed-byte size of ``files`` from a pre-parsed sizes map."""
    return sum(sizes.get(f, 1) for f in files)


def _pack_shards(
    stack: StackAssignment,
    sizes: dict[str, int],
    max_files: int,
    max_bytes: int,
    blocks: list[list[str]],
) -> list[StackAssignment]:
    """Pack ``blocks`` (whole components or singleton files) into bounded shards.

    Files-per-shard never exceeds ``max_files`` nor the byte budget; a block
    larger than the bound splits deterministically into consecutive shards (a
    single oversized file forms its own shard -- never split a file).
    """
    shards: list[StackAssignment] = []
    current: list[str] = []
    current_bytes = 0

    def emit(cur: list[str]) -> None:
        if cur:
            shards.append(
                StackAssignment(
                    stack_name=f"{stack.stack_name}#{len(shards)}",
                    skill_invocation=stack.skill_invocation,
                    files=list(cur),
                    is_docs_only=stack.is_docs_only,
                    frontier_files=[],
                )
            )

    for block in blocks:
        size = sum(sizes.get(f, 1) for f in block)
        fits = len(current) + len(block) <= max_files and current_bytes + size <= max_bytes
        if current and not fits:
            emit(current)
            current, current_bytes = [], 0
            fits = len(block) <= max_files and size <= max_bytes
        if fits:
            current.extend(block)
            current_bytes += size
            continue
        # Block too large for a (possibly fresh) shard: split per-file.
        for f in block:
            fsize = sizes.get(f, 1)
            if current and (
                len(current) >= max_files or (current_bytes > 0 and current_bytes + fsize > max_bytes)
            ):
                emit(current)
                current, current_bytes = [], 0
            current.append(f)
            current_bytes += fsize
    emit(current)
    return shards


def _assign_frontiers(
    shards: list[StackAssignment], edges: dict[str, set[str]], frontier_max: int
) -> None:
    """Populate each shard's bounded cross-shard frontier in place.

    ``frontier_files`` of a shard = the (sorted) set of files in *other* shards
    of the same language sharing an undirected import edge with this shard's
    files, capped at ``frontier_max``. Frontier files are never added to primary
    ``files`` (union of primary sets stays the changed set).
    """
    adjacency: dict[str, set[str]] = {}
    for src, deps in edges.items():
        adjacency.setdefault(src, set()).update(deps)
        for dep in deps:
            adjacency.setdefault(dep, set()).add(src)

    for shard in shards:
        shard_set = set(shard.files)
        frontier: set[str] = set()
        for f in shard.files:
            for nbr in adjacency.get(f, set()):
                if nbr not in shard_set:
                    frontier.add(nbr)
        shard.frontier_files = sorted(frontier)[:frontier_max]


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

    Pure and deterministic. Non-structural stacks whose file count or total
    changed-byte size exceeds the bounds are split into consecutive
    ``<name>#<i>`` shards; the union of all shard file sets equals the source
    stack's set with no duplicates. When ``graph`` is provided and non-empty,
    files are co-located by undirected import connected component (whole
    components stay together when the shard has room) and cross-shard shared
    files surface as a bounded frontier. Stacks at/under every bound are
    returned unsplit with their original ``stack_name``; the structural
    meta-stack is passed through unchanged. Total tasks never exceed
    ``fanout_cap`` whenever the unsplit stacks alone fit under it: when the
    total exceeds the cap, the largest sharded stacks are returned unsplit
    (each un-split removes ``len(shards) - 1`` tasks) until it fits. A stack
    that packs into a single shard -- one oversized file is never split -- is
    unsplit-equivalent and keeps its original name. When the unsplit
    non-structural stacks alone outnumber ``fanout_cap``, the total necessarily
    exceeds it (files are never dropped or merged).
    """
    sizes = _file_change_bytes(diff)
    structural: list[StackAssignment] = []
    unsharded: list[StackAssignment] = []
    sharded: list[tuple[StackAssignment, list[StackAssignment]]] = []
    edges: dict[str, set[str]] = graph or {}

    for stack in stacks:
        if stack.stack_name == STRUCTURE_STACK_NAME:
            structural.append(stack)
            continue
        if len(stack.files) <= max_files and _changed_bytes(sizes, stack.files) <= max_bytes:
            unsharded.append(stack)
            continue
        # Co-locate when a non-empty graph is available, else sorted singletons.
        if edges:
            blocks = co_locate_groups(stack.files, edges)
        else:
            blocks = [[f] for f in sorted(stack.files)]
        shards = _pack_shards(stack, sizes, max_files, max_bytes, blocks)
        if len(shards) == 1:
            # One oversized file forms its own shard ("never split a file");
            # that shard is unsplit-equivalent, so keep the stack unsplit under
            # its original name -- it can never reduce the fan-out excess.
            unsharded.append(stack)
            continue
        _assign_frontiers(shards, edges, frontier_max)
        sharded.append((stack, shards))

    total = len(unsharded) + sum(len(shards) for _, shards in sharded)
    if total > fanout_cap and sharded:
        # ``sharded`` holds only multi-shard packs (single-shard packs stay in
        # ``unsharded`` above), so every un-split removes >= 1 task and the
        # total provably fits the cap unless the unsplit stacks alone outnumber
        # it -- in which case no shard exists to un-split.
        excess = total - fanout_cap
        for stack, shards in sorted(sharded, key=lambda t: (-len(t[1]), t[0].stack_name)):
            if excess <= 0:
                break
            sharded.remove((stack, shards))
            unsharded.append(stack)
            excess -= len(shards) - 1

    out: list[StackAssignment] = []
    out.extend(structural)
    out.extend(unsharded)
    for _, shards in sharded:
        out.extend(shards)
    return out
