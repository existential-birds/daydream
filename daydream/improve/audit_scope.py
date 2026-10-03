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
from daydream.improve.schemas import (
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
    failed_groups: set[str] | None = None,
) -> dict[str, Any]:
    """Record partition coverage, counting failed group stacks as uncovered."""
    failed_groups = failed_groups or set()
    omitted_by_partition: dict[Partition, list[str]] = {}
    for omission in omissions:
        omitted_by_partition.setdefault(omission.partition, []).append(omission.stack)

    not_audited: list[dict[str, Any]] = []
    partially_audited: list[dict[str, Any]] = []
    for partition, omitted_stacks in omitted_by_partition.items():
        retained_groups = [
            group for group in groups if partition in group.partitions
        ]
        audited_stacks = sorted(
            {
                group.stack
                for group in retained_groups
                if group.name not in failed_groups and group.stack is not None
            }
        )
        failed_stacks = sorted(
            {
                group.stack
                for group in retained_groups
                if group.name in failed_groups and group.stack is not None
            }
        )
        entry = {
            "partition": partition.name,
            "root": partition.root,
            "file_count": len(partition.files),
            "reason": (
                "group-failed"
                if retained_groups and not audited_stacks
                else "group-ceiling"
            ),
            "omitted_stacks": sorted(set(omitted_stacks) | set(failed_stacks)),
        }
        if audited_stacks:
            partially_audited.append({**entry, "audited_stacks": audited_stacks})
        else:
            not_audited.append(entry)

    return {
        "artifact_type": "daydream.improve-coverage",
        "partitions": [
            {**_partition_dict(partition), "source": partition.source}
            for partition in partitions
        ],
        "groups": [
            {
                "name": group.name,
                "stack": group.stack,
                "file_count": group.file_count,
                "partitions": [
                    partition.name for partition in group.partitions
                ],
            }
            for group in groups
        ],
        "not_audited": not_audited,
        "partially_audited": partially_audited,
    }


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
