"""Repository scope, coverage accounting, and grounded finding ownership for Improve."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.config import (
    EffortTier,
)
from daydream.config_file import DaydreamFileConfig
from daydream.deep.detection import GENERIC_STACK, StackAssignment
from daydream.deep.diff import iter_diff_blocks
from daydream.improve.command_contract import (
    path_is_confined,
)
from daydream.improve.partition import (
    PARTITION_MAX_FILES,
    Partition,
    PartitionGroup,
    PartitionStackOmission,
    build_partitions,
    group_partitions,
    stack_by_path,
)
from daydream.improve.prompts import (
    CHANGE_SHAPES,
    MAINTENANCE_SIGNALS,
)
from daydream.pr_review import compute_fingerprint
from daydream.repository_paths import strip_dot_slash
from daydream.services import (
    RepoRootPolicy,
    Service,
    ServiceMatch,
    owning_services,
)

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext




_EVIDENCE_LOCATION = re.compile(r"^`?(.+?):(\d+)(?::(\d+))?(?:`|\b)")
_MAINTENANCE_SIGNALS = set(MAINTENANCE_SIGNALS)
_CHANGE_SHAPES = set(CHANGE_SHAPES)
_REUSE_TARGET = re.compile(r"^(?:repo:[^#\s]+#[^\s]+|stdlib:[^\s]+|dep:[^:\s]+:[^\s]+)$")


def _partition_repository(
    ctx: FlowContext,
    tracked: list[str],
    services: list[Service],
    stacks: list[StackAssignment],
    *,
    branch_focus: bool,
) -> tuple[list[Partition], list[PartitionGroup], list[PartitionStackOmission]]:
    """Partition the audited surface; quick and branch runs use one whole-surface group per category."""
    stack_of = stack_by_path(stacks)
    file_config = ctx.config.file_config or DaydreamFileConfig()
    max_files = file_config.improve_partition_max_files or PARTITION_MAX_FILES
    tier: EffortTier = ctx.data["effort_tier"]
    max_groups = (
        file_config.improve_max_partition_groups or tier.max_partition_groups
    )

    if branch_focus or ctx.config.improve_effort == "quick":
        whole = Partition(
            name="branch" if branch_focus else "repository",
            root=".",
            source="branch" if branch_focus else "quick",
            service=None,
            files=tuple(tracked),
        )
        return [whole], [_whole_surface_group(whole, stack_of)], []

    partitions = build_partitions(tracked, services, max_files=max_files)
    groups, omissions = group_partitions(
        partitions,
        stack_of,
        max_files=max_files,
        max_groups=max_groups,
    )
    return partitions, groups, omissions


def _whole_surface_group(
    partition: Partition, stack_of: dict[str, str]
) -> PartitionGroup:
    counts = Counter(stack_of.get(path, GENERIC_STACK) for path in partition.files)
    dominant = (
        min(sorted(counts), key=lambda stack: (-counts[stack], stack))
        if counts
        else GENERIC_STACK
    )
    return PartitionGroup(
        name="group-01", stack=dominant, partitions=(partition,)
    )


def _partition_dict(partition: Partition) -> dict[str, Any]:
    return {
        "name": partition.name,
        "root": partition.root,
        "file_count": len(partition.files),
        "service": partition.service,
    }


def _group_dict(group: PartitionGroup) -> dict[str, Any]:
    return {
        "name": group.name,
        "stack": group.stack,
        "file_count": group.file_count,
        "partitions": [
            _partition_dict(partition) for partition in group.partitions
        ],
    }


def _coverage_ledger(
    partitions: list[Partition],
    groups: list[PartitionGroup],
    omissions: list[PartitionStackOmission],
    *,
    categories: tuple[str, ...] = (),
    failures: dict[str, str] | None = None,
    submodule_references: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Account for owned files per category; context availability grants no coverage.

    Before dispatch, retained files are planned, not audited. After dispatch a
    file is fully covered only when every requested category succeeded.
    """
    failures = failures or {}
    group_entries: list[dict[str, Any]] = []
    for group in groups:
        failed = [category for category in categories if f"{category}:{group.name}" in failures]
        originals = [p for p in partitions if any(_same_partition(p, member) for member in group.partitions)]
        available = {path for p in originals for path in p.files}
        group_entries.append({
            "name": group.name,
            "stack": group.stack,
            "file_count": group.file_count,
            "primary_files": list(group.primary_files),
            "context_file_count": len(available - set(group.primary_files) - set(submodule_references or {})),
            "partitions": [p.name for p in group.partitions],
            "status": (
                "planned" if not categories else
                "failed" if len(failed) == len(categories) else
                "partial" if failed else "audited"
            ),
            "categories": {
                category: "failed" if category in failed else "audited"
                for category in categories
            },
        })
        if submodule_references is not None:
            reference_count = sum(path in submodule_references for path in group.primary_files)
            group_entries[-1].update({
                "primary_file_count": group.file_count - reference_count,
                "submodule_reference_count": reference_count,
            })

    not_audited: list[dict[str, Any]] = []
    partially_audited: list[dict[str, Any]] = []
    for partition in partitions:
        retained = [(group, member) for group in groups for member in group.partitions
                    if _same_partition(partition, member)]
        dropped = [o for o in omissions if _same_partition(partition, o.partition)]
        omitted = {path for o in dropped for path in o.partition.files}
        omitted_stacks = {o.stack for o in dropped}
        audited_stacks: set[str] = set()
        reached: set[str] = set()
        failed_here = False
        missing_categories: dict[str, list[str]] = {}
        for category in categories:
            missing = set(omitted)
            for group, member in retained:
                if f"{category}:{group.name}" in failures:
                    missing.update(member.files)
                    failed_here = True
                    if group.stack is not None:
                        omitted_stacks.add(group.stack)
                else:
                    reached.update(member.files)
                    if group.stack is not None:
                        audited_stacks.add(group.stack)
            missing_categories[category] = sorted(missing)
        missing = omitted | {path for paths in missing_categories.values() for path in paths}
        if not missing:
            continue
        entry: dict[str, Any] = {
            "partition": partition.name,
            "root": partition.root,
            "service": partition.service,
            "file_count": len(partition.files),
            "audited_file_count": len(set(partition.files) - missing) if categories else 0,
            "reached_file_count": len(reached),
            "omitted_file_count": len(missing),
            "omitted_primary_files": sorted(missing),
            "missing_by_category": missing_categories,
            "reason": "group-ceiling-and-failed" if dropped and failed_here else
                      "group-failed" if failed_here else "group-ceiling",
            "omitted_stacks": sorted(omitted_stacks),
        }
        # At recon time, retained ownership is planned, never claimed audited.
        if reached:
            partially_audited.append({**entry, "audited_stacks": sorted(audited_stacks)})
        else:
            not_audited.append(entry)

    ledger: dict[str, Any] = {
        "artifact_type": "daydream.improve-coverage",
        "stage": "audited" if categories else "planned",
        "categories_run": list(categories),
        "partitions": [{**_partition_dict(p), "source": p.source} for p in partitions],
        "groups": group_entries,
        "not_audited": not_audited,
        "partially_audited": partially_audited,
        "failed_assignments": dict(sorted(failures.items())),
    }
    if submodule_references is not None:
        ledger["unavailable_submodule_contents"] = [
            {"path": path, "commit": commit, "file_count": None, "reason": "not-captured"}
            for path, commit in sorted(submodule_references.items())
        ]
    return ledger


def _same_partition(left: Partition, right: Partition) -> bool:
    """Match original partitions to owned slices without comparing their files."""
    return (left.name, left.root, left.source, left.service) == (right.name, right.root, right.source, right.service)


def _owning_services(
    path: str, services: list[Service]
) -> tuple[Service, ...]:
    """The improve orchestrator's service-ownership rule, stated once."""
    return owning_services(
        path,
        services,
        match=ServiceMatch.ALL,
        repo_root=RepoRootPolicy.ORDINARY,
        match_root_equal=True,
    )


def _services_for_files(
    services: list[Service], files: tuple[str, ...]
) -> list[Service]:
    if not files:
        return services
    owners = {
        service
        for path in files
        for service in _owning_services(path, services)
    }
    return [service for service in services if service in owners]


def _restrict_diff_to_services(
    diff: str, services: list[Service]
) -> tuple[str, list[str]]:
    """Restrict both diff text and filenames so scoped prompts cannot expose other services."""
    selected: list[str] = []
    files: list[str] = []
    for path, block in iter_diff_blocks(diff):
        if _owning_services(path, services):
            selected.append(block)
            if path not in files:
                files.append(path)
    return "".join(selected), files


def _stacks_for_services(
    stacks: list[StackAssignment],
    services: list[Service],
) -> list[StackAssignment]:
    scoped: list[StackAssignment] = []
    for stack in stacks:
        files = [
            path
            for path in stack.files
            if _owning_services(path, services)
        ]
        if files:
            scoped.append(
                StackAssignment(
                    stack_name=stack.stack_name,
                    files=files,
                    is_docs_only=stack.is_docs_only,
                    has_explicit_rule=stack.has_explicit_rule,
                )
            )
    return scoped


def _evidence_paths(
    finding: dict[str, Any], *, repo: Path
) -> list[str] | None:
    evidence = finding.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return None
    paths: list[str] = []
    for entry in evidence:
        if not isinstance(entry, str):
            return None
        match = _EVIDENCE_LOCATION.match(entry.strip())
        if match is None:
            return None
        path = match.group(1).strip("`")
        if not path_is_confined(repo, path):
            return None
        resolved = (repo / path).resolve()
        if not resolved.is_file():
            return None
        try:
            line_count = len(resolved.read_text(errors="replace").splitlines())
        except OSError:
            return None
        start_line = int(match.group(2))
        end_line = int(match.group(3) or start_line)
        if start_line < 1 or end_line < start_line or end_line > line_count:
            return None
        # A leading ``./`` is a legal spelling since the grammar relaxed, but
        # partition/service attribution matches against git-derived roots
        # (never ``./``-prefixed). Normalize so newly-legal ``./x`` evidence is
        # not silently dropped from attribution.
        paths.append(strip_dot_slash(path))
    return paths


def _owning_partition(
    partitions: list[Partition], evidence_paths: list[str]
) -> str | None:
    if not evidence_paths:
        return None
    path = evidence_paths[0]
    for partition in sorted(
        partitions, key=lambda item: -len(item.root)
    ):
        if partition.root == "." or path.startswith(f"{partition.root}/"):
            return partition.name
    return None


def _stamp_finding(
    finding: dict[str, Any],
    category: str,
    services: list[Service],
    partitions: list[Partition],
    *,
    repo: Path,
) -> dict[str, Any] | None:
    evidence_paths = _evidence_paths(finding, repo=repo)
    if evidence_paths is None:
        return None
    stamped = dict(finding)
    raw_signals = stamped.get("maintenance_signals")
    stamped["maintenance_signals"] = (
        list(dict.fromkeys(signal for signal in raw_signals if signal in _MAINTENANCE_SIGNALS))
        if isinstance(raw_signals, list)
        else []
    )
    if stamped.get("change_shape") not in _CHANGE_SHAPES:
        stamped["change_shape"] = "unknown"
    reuse_target = stamped.get("reuse_target")
    if not isinstance(reuse_target, str) or _REUSE_TARGET.fullmatch(reuse_target) is None:
        stamped["reuse_target"] = None
    stamped["category"] = category
    stamped["partition"] = _owning_partition(partitions, evidence_paths)
    stamped["services"] = [service.name for service in _services_for_files(services, tuple(evidence_paths))]
    stamped["fingerprint"] = compute_fingerprint(
        str(stamped.get("path", "")),
        str(stamped.get("title", "")),
        str(stamped.get("body", "")),
    )
    return stamped


def _correct_primary_evidence_location(
    finding: dict[str, Any],
    *,
    path: str,
    line: int | None,
) -> None:
    """Align the primary evidence citation with a vetted path correction."""
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return
    evidence = list(evidence)
    finding["evidence"] = evidence
    for index, entry in enumerate(evidence):
        if not isinstance(entry, str):
            continue
        match = _EVIDENCE_LOCATION.match(entry.strip())
        if match is None:
            continue
        evidence_line = line if line is not None else int(match.group(2))
        location = f"{path}:{evidence_line}"
        if entry.lstrip().startswith("`"):
            location = f"`{location}`"
        evidence[index] = location + entry.strip()[match.end() :]
        return
