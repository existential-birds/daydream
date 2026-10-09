"""Deterministic language-stack shards named <stack>#<index>.

The structure meta-stack retains its original object and all changed files;
it is neither split nor counted against the fan-out cap.
"""

from __future__ import annotations

from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.dependency import co_locate_groups
from daydream.deep.detection import StackAssignment
from daydream.deep.diff import iter_diff_blocks


def file_change_bytes(diff: str) -> dict[str, int]:
    """Map changed paths to UTF-8 block sizes; the first block wins.

    Files absent from the map still receive a one-byte shard weight.
    """
    sizes: dict[str, int] = {}
    for path, block in iter_diff_blocks(diff):
        sizes.setdefault(path, len(block.encode("utf-8")))
    return sizes


def pack_file_batches(
    stack: StackAssignment,
    sizes: dict[str, int],
    max_files: int,
    max_bytes: int,
    blocks: list[list[str]],
) -> list[StackAssignment]:
    """Pack components within file/byte bounds, splitting oversized components into files.

    Never split a file: one oversized file may exceed the byte bound.
    """
    shards: list[StackAssignment] = []
    current: list[str] = []
    current_bytes = 0

    def emit(cur: list[str]) -> None:
        if cur:
            shards.append(
                StackAssignment(
                    stack_name=f"{stack.stack_name}#{len(shards)}",
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
    """Assign sorted undirected import neighbors outside each shard, capped at frontier_max.

    Frontier context never enters the primary file set.
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
    """Split oversized language stacks deterministically without dropping or duplicating files.

    Keep fitting import components together; otherwise pack sorted singletons.
    Frontiers provide bounded cross-shard context. Stacks within both bounds, or
    packing into one shard, keep their original names. Structure passes through.

    Reduce the largest groups just enough by merging adjacent lowest-weight pairs.
    Cap-induced groups may exceed soft file/byte targets. If distinct stacks alone exceed
    fanout_cap, retain them all: the cap cannot discard or merge stacks.
    """
    sizes = file_change_bytes(diff)
    structural: list[StackAssignment] = []
    unsharded: list[StackAssignment] = []
    sharded: list[tuple[StackAssignment, list[StackAssignment]]] = []
    edges: dict[str, set[str]] = graph or {}

    for stack in stacks:
        if stack.stack_name == STRUCTURE_STACK_NAME:
            structural.append(stack)
            continue
        if len(stack.files) <= max_files and sum(sizes.get(f, 1) for f in stack.files) <= max_bytes:
            unsharded.append(stack)
            continue
        # Co-locate when a non-empty graph is available, else sorted singletons.
        if edges:
            blocks = co_locate_groups(stack.files, edges)
        else:
            blocks = [[f] for f in sorted(stack.files)]
        shards = pack_file_batches(stack, sizes, max_files, max_bytes, blocks)
        if len(shards) == 1:
            # Single-shard packs keep their original identity and cannot reduce fan-out.
            unsharded.append(stack)
            continue
        sharded.append((stack, shards))

    total = len(unsharded) + sum(len(shards) for _, shards in sharded)
    if total > fanout_cap and sharded:
        excess = total - fanout_cap
        for stack, shards in sorted(sharded, key=lambda t: (-len(t[1]), t[0].stack_name)):
            if excess <= 0:
                break
            chosen = max(1, len(shards) - excess)
            excess -= len(shards) - chosen
            weights = [sum(sizes.get(path, 1) for path in shard.files) for shard in shards]
            while len(shards) > chosen:
                left = min(range(len(shards) - 1), key=lambda i: (weights[i] + weights[i + 1], i))
                shards[left].files.extend(shards[left + 1].files)
                weights[left] += weights.pop(left + 1)
                del shards[left + 1]

    for stack, shards in sharded:
        for index, shard in enumerate(shards):
            shard.stack_name = stack.stack_name if len(shards) == 1 else f'{stack.stack_name}#{index}'
        _assign_frontiers(shards, edges, frontier_max)

    out: list[StackAssignment] = []
    out.extend(structural)
    out.extend(unsharded)
    for _, shards in sharded:
        out.extend(shards)
    return out
