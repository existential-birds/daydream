"""Unit tests for the improve-flow partition cover and grouping."""

from __future__ import annotations

from pathlib import Path

from daydream.config_file import DaydreamFileConfig
from daydream.improve.partition import (
    build_partitions,
    group_partitions,
)
from daydream.services import Service, enumerate_services
from tests.test_improve_flow import _nested_service_repo


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
    assert [(p.name, len(p.files)) for p in partitions] == [("web/a", 3), ("web/b", 2), ("web/c", 2),]


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







def test_sibling_partitions_stay_in_one_group() -> None:
    # web/ splits into four children; a big unrelated tree competes for space.
    files = [f"web/{sub}/f{i}.ts" for sub in ("a", "b", "c", "d") for i in range(2)]
    files += [f"lib/f{i}.ts" for i in range(6)]
    partitions = build_partitions(files, [], max_files=6)
    stack_of = dict.fromkeys(files, "ts")
    groups, omissions = group_partitions(partitions, stack_of, max_files=8, max_groups=None)
    assert omissions == []
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
    assert sorted(root for group in groups for root in group.roots) == ["web/a", "web/b", "web/c",]

def test_split_service_slices_stay_in_one_group() -> None:
    files = [f"apps/big/{sub}/f{i}.py" for sub in ("x", "y") for i in range(3)]
    files += [f"apps/other/f{i}.py" for i in range(5)]
    services = [Service(name="big", root=Path("apps/big"), source="config"),
        Service(name="other", root=Path("apps/other"), source="config"),
    ]
    partitions = build_partitions(files, services, max_files=4)
    stack_of = dict.fromkeys(files, "python")
    groups, _ = group_partitions(partitions, stack_of, max_files=8, max_groups=None)
    by_group = {group.name: {p.name for p in group.partitions} for group in groups}
    # An oversized service's slices keep their identity *and* their locality.
    assert {"big/x", "big/y"} in by_group.values()



def test_service_ownership_freezes_nested_and_repo_root_partitioning(tmp_path: Path) -> None:
    repo = _nested_service_repo(tmp_path)
    services = enumerate_services(
        repo, DaydreamFileConfig(improve_service_roots=["services/*", "services/api/inner", "."])
    )
    assert [(s.name, s.root.as_posix()) for s in services] == [
        ("repo", "."), ("api", "services/api"), ("inner", "services/api/inner"),
    ]

    partitions = build_partitions([
            "services/api/handler.py", "services/api/inner/main.py", "services/api", "services/api/inner",
            "scripts/tool.py", "README.md",
        ], services,
    )

    assert [(p.name, p.root, p.service, p.files) for p in partitions] == [
        ("repo", ".", "repo", ("README.md", "scripts/tool.py", "services/api")),
        ("api", "services/api", "api", ("services/api/handler.py", "services/api/inner")),
        ("inner", "services/api/inner", "inner", ("services/api/inner/main.py",)),
    ]

def test_path_equal_to_a_service_root_falls_through_without_a_repo_root_service(tmp_path: Path) -> None:
    repo = _nested_service_repo(tmp_path)
    services = enumerate_services(repo, DaydreamFileConfig(improve_service_roots=["services/*", "services/api/inner"]))
    assert [(p.name, p.root, p.service, p.source) for p in build_partitions(["services/api"], services)] == [
        ("services", "services", None, "directory")
    ]
