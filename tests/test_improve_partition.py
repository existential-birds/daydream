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


def test_mixed_service_primary_files_are_disjoint_with_generic_fallback() -> None:
    files = [
        *[f"bound/f{i}.py" for i in range(4)], *[f"flat/f{i}.py" for i in range(9)],
        *[f"over/a/f{i}.py" for i in range(3)], *[f"over/b/f{i}.py" for i in range(2)],
        *[f"siblings/{sub}/f{i}.ts" for sub in ("a", "b", "c", "d") for i in range(2)],
        *[f"lib/f{i}.ts" for i in range(6)],
        *[f"spill/{sub}/f{i}.go" for sub in ("a", "b", "c") for i in range(4)],
        *[f"apps/big/{sub}/f{i}.rb" for sub in ("x", "y") for i in range(3)],
        *[f"apps/other/f{i}.rb" for i in range(5)],
        "apps/api/handler.py", "apps/api/pyproject.toml", "apps/api/client.tsx",
        "apps/api/README.md", "apps/api/Dockerfile", "apps/api/events.proto", "README.md",
    ]
    services = [Service("api", Path("apps/api"), "config"), Service("big", Path("apps/big"), "config"),
        Service("other", Path("apps/other"), "config")]
    partitions = build_partitions(files, services, max_files=4)
    assert [(p.name, p.root, p.source, p.service, len(p.files)) for p in partitions] == [
        ("api", "apps/api", "service:config", "api", 6),
        ("big/x", "apps/big/x", "service:config", "big", 3),
        ("big/y", "apps/big/y", "service:config", "big", 3),
        ("other", "apps/other", "service:config", "other", 5),
        ("bound", "bound", "directory", None, 4), ("flat", "flat", "directory", None, 9),
        ("lib", "lib", "directory", None, 6),
        ("over/a", "over/a", "directory", None, 3), ("over/b", "over/b", "directory", None, 2),
        ("siblings/a", "siblings/a", "directory", None, 2), ("siblings/b", "siblings/b", "directory", None, 2),
        ("siblings/c", "siblings/c", "directory", None, 2), ("siblings/d", "siblings/d", "directory", None, 2),
        ("spill/a", "spill/a", "directory", None, 4), ("spill/b", "spill/b", "directory", None, 4),
        ("spill/c", "spill/c", "directory", None, 4), ("residue", ".", "residue", None, 1),
    ]
    covered = [path for partition in partitions for path in partition.files]
    assert sorted(covered) == sorted(files) and len(covered) == len(files)
    stack_of = {path: {".py": "python", ".ts": "ts", ".tsx": "react", ".go": "go", ".rb": "ruby"}[Path(path).suffix]
        for path in files if Path(path).suffix in {".py", ".ts", ".tsx", ".go", ".rb"}}
    stack_of["apps/api/pyproject.toml"] = "python"
    groups, omissions = group_partitions(partitions, stack_of, max_files=8)
    assert omissions == []
    assert [(g.stack, g.file_count, tuple(p.name for p in g.partitions)) for g in groups] == [
        ("generic", 4, ("api", "residue")),
        ("go", 8, ("spill/a", "spill/b")), ("go", 4, ("spill/c",)),
        ("python", 4, ("api", "over/b")), ("python", 7, ("bound", "over/a")), ("python", 9, ("flat",)),
        ("react", 1, ("api",)), ("ruby", 6, ("big/x", "big/y")), ("ruby", 5, ("other",)),
        ("ts", 6, ("lib",)), ("ts", 8, ("siblings/a", "siblings/b", "siblings/c", "siblings/d")),
    ]
    slices = {(g.stack, p.name): p for g in groups for p in g.partitions}
    assert slices[("generic", "api")].files == (
        "apps/api/Dockerfile", "apps/api/README.md", "apps/api/events.proto",
    )
    assert slices[("python", "api")].files == ("apps/api/handler.py", "apps/api/pyproject.toml")
    assert slices[("react", "api")].files == ("apps/api/client.tsx",)
    assert {(p.name, p.root, p.source, p.service) for key, p in slices.items() if key[1] == "api"} == {
        ("api", "apps/api", "service:config", "api"),
    }
    assert sorted(path for group in groups for path in group.primary_files) == sorted(files)


def test_group_ceiling_ranks_owned_slices_and_preserves_omitted_files() -> None:
    # A mixed partition's full size used to make every stack equally large.
    # The ceiling must retain the larger primary surface and name exact omitted
    # slices, even though all slices retain the same partition identity.
    files = ["svc/a.py", "svc/b.py", "svc/c.py", "svc/a.ts", "svc/b.ts", "svc/README.md",
        "other/a.py", "other/b.py"]
    partitions = build_partitions(files, [], max_files=10)
    stack_of = {"svc/a.py": "python", "svc/b.py": "python", "svc/c.py": "python",
        "svc/a.ts": "react", "svc/b.ts": "react", "other/a.py": "python", "other/b.py": "python"}
    groups, omissions = group_partitions(partitions, stack_of, max_files=3, max_groups=1)

    assert [(group.name, group.stack, group.file_count, group.primary_files) for group in groups] == [
        ("group-01", "python", 3, ("svc/a.py", "svc/b.py", "svc/c.py")),
    ]
    assert [(item.partition.name, item.partition.root, item.stack, item.partition.files) for item in omissions] == [
        ("other", "other", "python", ("other/a.py", "other/b.py")),
        ("svc", "svc", "generic", ("svc/README.md",)),
        ("svc", "svc", "react", ("svc/a.ts", "svc/b.ts")),
    ]
    covered = [path for group in groups for path in group.primary_files]
    covered.extend(path for item in omissions for path in item.partition.files)
    assert sorted(covered) == sorted(files)
    assert group_partitions(list(reversed(partitions)), dict(reversed(list(stack_of.items()))),
        max_files=3, max_groups=1) == (groups, omissions)



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
