"""Bounded partition cover and stack-homogeneous grouping for improve audits.

Pure logic: the audit fan-out axis is derived here so every agent's search
surface is bounded by a file count, and so prompts can carry directory roots
instead of inlined file lists.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.detection import StackAssignment
from daydream.improve.services import Service

PARTITION_MAX_FILES: int = 400

_GENERIC_STACK = "generic"
_RESIDUE_NAME = "residue"
_TOP_LEVEL = ""


@dataclass(frozen=True)
class Partition:
    """One bounded slice of the repository's tracked files.

    Attributes:
        name: Unique partition name — a service name, or the directory root.
        root: Repo-relative POSIX directory; ``"."`` only for the residue partition.
        source: ``"service:<Service.source>"``, ``"directory"``, or ``"residue"``.
        service: Owning service name, kept through oversized-service splits.
        files: Sorted member files. Host-side only — never serialized into a prompt.
    """

    name: str
    root: str
    source: str
    service: str | None
    files: tuple[str, ...]


@dataclass(frozen=True)
class PartitionGroup:
    """A stack-homogeneous bundle of partitions handed to one audit agent."""

    name: str
    stack: str | None
    partitions: tuple[Partition, ...]

    @property
    def file_count(self) -> int:
        """Return the total number of files across member partitions."""
        return sum(len(partition.files) for partition in self.partitions)

    @property
    def roots(self) -> tuple[str, ...]:
        """Return the sorted member roots."""
        return tuple(sorted(partition.root for partition in self.partitions))


def build_partitions(
    files: Sequence[str],
    services: Sequence[Service],
    *,
    max_files: int = PARTITION_MAX_FILES,
) -> list[Partition]:
    """Cover every file with disjoint partitions bounded by ``max_files``.

    Files inside a service root belong to that service (deepest root wins); a
    service larger than the bound splits by subdirectory but keeps its service
    stamp. Remaining files partition by directory, and repo-root loose files
    form a single ``residue`` partition. A flat directory that cannot split
    stays oversized rather than being dropped.
    """
    tracked = sorted({path for path in files if path})
    ordered_services = sorted(services, key=lambda service: (-len(service.root.parts), service.root.as_posix()))

    service_files: dict[str, list[str]] = defaultdict(list)
    uncovered: list[str] = []
    for path in tracked:
        owner = _owning_service(path, ordered_services)
        if owner is None:
            uncovered.append(path)
        else:
            service_files[owner.root.as_posix()].append(path)

    partitions: list[Partition] = []
    for service in sorted(ordered_services, key=lambda item: item.root.as_posix()):
        root = service.root.as_posix()
        owned = service_files.get(root)
        if not owned:
            continue
        for chunk_root, chunk_files in _split_directory(root, owned, max_files):
            relative = chunk_root[len(root) + 1 :] if chunk_root != root else ""
            partitions.append(
                Partition(
                    name=f"{service.name}/{relative}" if relative else service.name,
                    root=chunk_root,
                    source=f"service:{service.source}",
                    service=service.name,
                    files=chunk_files,
                )
            )

    directory_files: dict[str, list[str]] = defaultdict(list)
    residue: list[str] = []
    for path in uncovered:
        top, separator, _ = path.partition("/")
        if separator:
            directory_files[top].append(path)
        else:
            residue.append(path)

    for top in sorted(directory_files):
        for chunk_root, chunk_files in _split_directory(top, directory_files[top], max_files):
            partitions.append(
                Partition(name=chunk_root, root=chunk_root, source="directory", service=None, files=chunk_files)
            )

    if residue:
        partitions.append(
            Partition(
                name=_RESIDUE_NAME,
                root=".",
                source=_RESIDUE_NAME,
                service=None,
                files=tuple(sorted(residue)),
            )
        )

    return sorted(partitions, key=lambda partition: (partition.source == _RESIDUE_NAME, partition.root))


def stack_by_path(stacks: Sequence[StackAssignment]) -> dict[str, str]:
    """Map each routed file to its stack, ignoring the ``structure`` meta-stack."""
    mapping: dict[str, str] = {}
    for assignment in stacks:
        if assignment.stack_name == STRUCTURE_STACK_NAME:
            continue
        for path in assignment.files:
            mapping.setdefault(path, assignment.stack_name)
    return mapping


def group_partitions(
    partitions: Sequence[Partition],
    stack_of: Mapping[str, str],
    *,
    max_files: int = PARTITION_MAX_FILES,
    max_groups: int | None = None,
) -> tuple[list[PartitionGroup], list[Partition]]:
    """Pack partitions into stack-homogeneous groups bounded by ``max_files``.

    Returns:
        The kept groups (renamed ``group-01``..) and, when ``max_groups`` binds,
        every partition belonging to a dropped group — the not-audited list.
    """
    buckets: dict[str, list[Partition]] = defaultdict(list)
    for partition in partitions:
        for stack in sorted({stack_of.get(path, _GENERIC_STACK) for path in partition.files}):
            buckets[stack].append(partition)

    bins: list[tuple[str, list[Partition]]] = []
    for stack in sorted(buckets):
        for members in _pack(buckets[stack], max_files):
            bins.append((stack, sorted(members, key=lambda partition: partition.name)))

    bins.sort(key=lambda entry: (entry[0], entry[1][0].name))

    skipped: list[Partition] = []
    if max_groups is not None and len(bins) > max_groups:
        ranked = sorted(bins, key=lambda entry: (-sum(len(p.files) for p in entry[1]), entry[1][0].name))
        kept = {id(entry) for entry in ranked[:max_groups]}
        skipped = sorted(
            (partition for entry in bins if id(entry) not in kept for partition in entry[1]),
            key=lambda partition: partition.root,
        )
        bins = [entry for entry in bins if id(entry) in kept]

    groups = [
        PartitionGroup(name=f"group-{index:02d}", stack=stack, partitions=tuple(members))
        for index, (stack, members) in enumerate(bins, start=1)
    ]
    return groups, skipped


def _owning_service(path: str, ordered_services: Sequence[Service]) -> Service | None:
    for service in ordered_services:
        root = service.root.as_posix()
        if root in {"", "."} or path.startswith(f"{root}/"):
            return service
    return None


def _split_directory(root: str, files: Sequence[str], max_files: int) -> list[tuple[str, tuple[str, ...]]]:
    """Split one directory subtree into chunks of at most ``max_files`` files."""
    ordered = tuple(sorted(files))
    if len(ordered) <= max_files:
        return [(root, ordered)]

    prefix = f"{root}/"
    direct: list[str] = []
    children: dict[str, list[str]] = defaultdict(list)
    for path in ordered:
        remainder = path[len(prefix) :]
        head, separator, _ = remainder.partition("/")
        if separator:
            children[head].append(path)
        else:
            direct.append(path)

    if not children:
        return [(root, ordered)]

    chunks: list[tuple[str, tuple[str, ...]]] = []
    if direct:
        chunks.append((root, tuple(direct)))
    for child in sorted(children):
        chunks.extend(_split_directory(f"{prefix}{child}", children[child], max_files))
    return chunks


def _parent(partition: Partition) -> str:
    """Return the partition root's parent directory, or a repo-top sentinel."""
    root, separator, _ = partition.root.rpartition("/")
    return root if separator else _TOP_LEVEL


def _pack(partitions: Sequence[Partition], max_files: int) -> list[list[Partition]]:
    """Bin-pack sibling-first, so one subtree's slices land in one agent's group.

    Packing purely by size scatters a split subtree across groups — on a real
    1892-file monorepo it put one React app's eleven sibling partitions in four
    different groups, making four agents rebuild the same context. Sibling
    clusters are placed whole wherever one bin has room; a cluster larger than
    the bound spills into as few adjacent bins as possible. Within a cluster
    the order stays first-fit-decreasing, and an oversized partition still gets
    its own bin.
    """
    clusters = defaultdict(list)
    for partition in partitions:
        clusters[_parent(partition)].append(partition)

    bins: list[list[Partition]] = []
    sizes: list[int] = []

    def place(partition: Partition, preferred: int | None) -> int:
        size = len(partition.files)
        if preferred is not None and sizes[preferred] + size <= max_files:
            bins[preferred].append(partition)
            sizes[preferred] += size
            return preferred
        for index, current in enumerate(sizes):
            if current + size <= max_files:
                bins[index].append(partition)
                sizes[index] = current + size
                return index
        bins.append([partition])
        sizes.append(size)
        return len(bins) - 1

    ordered_clusters = sorted(
        clusters.items(),
        key=lambda entry: (-sum(len(p.files) for p in entry[1]), entry[0]),
    )
    for _, members in ordered_clusters:
        members = sorted(members, key=lambda item: (-len(item.files), item.name))
        total = sum(len(partition.files) for partition in members)
        whole = next(
            (index for index, size in enumerate(sizes) if size + total <= max_files),
            None,
        )
        if whole is not None:
            bins[whole].extend(members)
            sizes[whole] += total
            continue
        home: int | None = None
        for partition in members:
            home = place(partition, home)
    return bins
