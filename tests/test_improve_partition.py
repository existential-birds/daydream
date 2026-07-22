"""Unit tests for the improve-flow partition cover and grouping."""

from __future__ import annotations

from pathlib import Path

from daydream.improve.partition import (
    PARTITION_MAX_FILES,
    Partition,
    PartitionGroup,
    build_partitions,
    group_partitions,
    stack_by_path,
)
from daydream.improve.services import Service


def test_service_files_cover_into_service_partitions() -> None:
    files = ["apps/billing/api.py", "apps/billing/pyproject.toml", "apps/catalog/api.py"]
    services = [
        Service(name="billing", root=Path("apps/billing"), source="heuristic:conventional-root"),
        Service(name="catalog", root=Path("apps/catalog"), source="heuristic:conventional-root"),
    ]
    partitions = build_partitions(files, services)
    assert [(p.name, p.root, p.service, p.files) for p in partitions] == [
        ("billing", "apps/billing", "billing", ("apps/billing/api.py", "apps/billing/pyproject.toml")),
        ("catalog", "apps/catalog", "catalog", ("apps/catalog/api.py",)),
    ]


def test_uncovered_tree_becomes_directory_partitions_and_root_files_residue() -> None:
    files = ["frontend/src/a.tsx", "frontend/src/b.tsx", "README.md"]
    partitions = build_partitions(files, [])
    assert [(p.name, p.root, p.source) for p in partitions] == [
        ("frontend", "frontend", "directory"),
        ("residue", ".", "residue"),
    ]
    assert partitions[1].files == ("README.md",)


def test_oversized_directory_splits_by_subdirectory() -> None:
    files = [f"web/{sub}/f{i}.ts" for sub in ("a", "b", "c") for i in range(4)]
    partitions = build_partitions(files, [], max_files=6)
    # web has 12 files > 6 -> split at web/*; a+b = 8 > 6 so siblings cannot merge;
    # every child <= 6 stands alone.
    assert [(p.name, p.root, len(p.files)) for p in partitions] == [
        ("web/a", "web/a", 4),
        ("web/b", "web/b", 4),
        ("web/c", "web/c", 4),
    ]


def test_directory_at_the_bound_is_one_partition() -> None:
    # Exactly at the bound: the subtree is never split, so its children stay a
    # single partition. Coalescing *split* siblings is not a partition-layer
    # merge (a merged root would overlap the siblings that did not merge) — it
    # happens when groups are packed, see test_sibling_partitions_stay_in_one_group.
    files = [f"web/{sub}/f{i}.ts" for sub in ("a", "b", "c") for i in range(2)]
    partitions = build_partitions(files, [], max_files=6)
    assert [(p.name, p.root, len(p.files)) for p in partitions] == [("web", "web", 6)]


def test_directory_one_file_over_the_bound_splits_into_children() -> None:
    files = [f"web/{sub}/f{i}.ts" for sub in ("a", "b", "c") for i in range(2)]
    files.append("web/a/extra.ts")
    partitions = build_partitions(files, [], max_files=6)
    assert [(p.name, len(p.files)) for p in partitions] == [
        ("web/a", 3),
        ("web/b", 2),
        ("web/c", 2),
    ]


def test_oversized_service_splits_but_keeps_service_stamp() -> None:
    files = [f"apps/big/{sub}/f{i}.py" for sub in ("x", "y") for i in range(4)]
    services = [Service(name="big", root=Path("apps/big"), source="config")]
    partitions = build_partitions(files, services, max_files=6)
    assert all(p.service == "big" for p in partitions)
    assert [p.name for p in partitions] == ["big/x", "big/y"]


def test_unsplittable_flat_directory_stays_oversized() -> None:
    files = [f"flat/f{i}.py" for i in range(10)]
    partitions = build_partitions(files, [], max_files=6)
    assert [(p.root, len(p.files)) for p in partitions] == [("flat", 10)]


def test_partition_cover_is_total_and_disjoint_at_scale() -> None:
    files = sorted(f"pkg{i:03d}/mod{j}/f{k}.go" for i in range(50) for j in range(4) for k in range(5))
    partitions = build_partitions(files, [], max_files=40)
    covered = [f for p in partitions for f in p.files]
    assert sorted(covered) == files and len(covered) == len(files)
    assert all(len(p.files) <= 40 for p in partitions)


def test_stack_by_path_excludes_the_structure_meta_stack() -> None:
    from daydream.deep.detection import detect_stacks

    stacks = detect_stacks(["a.py", "b.go"])
    mapping = stack_by_path(stacks)
    assert mapping == {"a.py": "python", "b.go": "go"}


def test_grouping_is_stack_homogeneous_and_bounded() -> None:
    parts = build_partitions(
        ["apps/s1/a.py", "apps/s2/b.py", "web/c.tsx", "README.md"],
        [
            Service(name="s1", root=Path("apps/s1"), source="config"),
            Service(name="s2", root=Path("apps/s2"), source="config"),
        ],
    )
    stack_of = {"apps/s1/a.py": "python", "apps/s2/b.py": "python", "web/c.tsx": "react"}
    groups, skipped = group_partitions(parts, stack_of, max_files=10, max_groups=None)
    assert skipped == []
    assert [(g.name, g.stack, tuple(p.name for p in g.partitions)) for g in groups] == [
        ("group-01", "generic", ("residue",)),
        ("group-02", "python", ("s1", "s2")),
        ("group-03", "react", ("web",)),
    ]


def test_group_ceiling_keeps_largest_groups_and_reports_the_rest() -> None:
    parts = build_partitions([f"apps/s{i}/f{j}.py" for i in range(3) for j in range(i + 1)], [], max_files=2)
    stack_of = {f: "python" for p in parts for f in p.files}
    groups, skipped = group_partitions(parts, stack_of, max_files=2, max_groups=1)
    assert len(groups) == 1
    assert groups[0].file_count == max(2, groups[0].file_count)  # largest kept
    assert {p.name for p in skipped} and all(isinstance(p, Partition) for p in skipped)


def test_empty_and_single_file_repos() -> None:
    assert build_partitions([], []) == []
    partitions = build_partitions(["main.py"], [])
    assert [(p.root, p.source) for p in partitions] == [(".", "residue")]


def test_sibling_partitions_stay_in_one_group() -> None:
    # web/ splits into four children; a big unrelated tree competes for space.
    files = [f"web/{sub}/f{i}.ts" for sub in ("a", "b", "c", "d") for i in range(2)]
    files += [f"lib/f{i}.ts" for i in range(6)]
    partitions = build_partitions(files, [], max_files=6)
    stack_of = dict.fromkeys(files, "ts")
    groups, skipped = group_partitions(partitions, stack_of, max_files=8, max_groups=None)
    assert skipped == []
    by_group = {group.name: set(group.roots) for group in groups}
    # Packing by size alone would fill the first bin with lib + two web children
    # and strand the rest; the siblings must travel together instead.
    assert {"web/a", "web/b", "web/c", "web/d"} in by_group.values()
    assert {"lib"} in by_group.values()


def test_oversized_sibling_cluster_spills_into_adjacent_groups() -> None:
    files = [f"web/{sub}/f{i}.ts" for sub in ("a", "b", "c") for i in range(4)]
    partitions = build_partitions(files, [], max_files=4)
    stack_of = dict.fromkeys(files, "ts")
    groups, _ = group_partitions(partitions, stack_of, max_files=8, max_groups=None)
    # 12 files against an 8-file group bound: two groups, neither over the bound,
    # and every partition still placed.
    assert [group.file_count for group in groups] == [8, 4]
    assert sorted(root for group in groups for root in group.roots) == [
        "web/a",
        "web/b",
        "web/c",
    ]


def test_split_service_slices_stay_in_one_group() -> None:
    files = [f"apps/big/{sub}/f{i}.py" for sub in ("x", "y") for i in range(3)]
    files += [f"apps/other/f{i}.py" for i in range(5)]
    services = [
        Service(name="big", root=Path("apps/big"), source="config"),
        Service(name="other", root=Path("apps/other"), source="config"),
    ]
    partitions = build_partitions(files, services, max_files=4)
    stack_of = dict.fromkeys(files, "python")
    groups, _ = group_partitions(partitions, stack_of, max_files=8, max_groups=None)
    by_group = {group.name: {p.name for p in group.partitions} for group in groups}
    # An oversized service's slices keep their identity *and* their locality.
    assert {"big/x", "big/y"} in by_group.values()


def test_group_exposes_file_count_and_roots() -> None:
    parts = build_partitions(["apps/s1/a.py", "web/c.tsx"], [Service(name="s1", root=Path("apps/s1"), source="config")])
    groups, _ = group_partitions(parts, {"apps/s1/a.py": "python", "web/c.tsx": "python"})
    assert len(groups) == 1
    assert groups[0].file_count == 2
    assert groups[0].roots == ("apps/s1", "web")


def test_partition_max_files_default_is_the_module_bound() -> None:
    files = [f"web/{i:04d}/f.ts" for i in range(PARTITION_MAX_FILES + 10)]
    partitions = build_partitions(files, [])
    assert all(len(p.files) <= PARTITION_MAX_FILES for p in partitions)
    assert isinstance(partitions[0], Partition)
    assert isinstance(group_partitions(partitions, {})[0][0], PartitionGroup)
