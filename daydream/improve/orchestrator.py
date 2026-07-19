"""Registered flow steps for repository-wide improve advising."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream import git_ops
from daydream.agent import console, run_agent
from daydream.config_file import DaydreamFileConfig
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.orchestrator import get_installed_skills
from daydream.exploration_runner import repo_scan
from daydream.extensions.api import FlowStep, Stop
from daydream.improve.artifacts import (
    recon_path,
    report_path,
    services_path,
)
from daydream.improve.services import Service, enumerate_services, filter_scope
from daydream.trajectory import DaydreamPhase, phase_scope
from daydream.ui import print_error, print_success

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


_RECON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "languages",
        "commands",
        "conventions",
        "intent_docs",
    ],
    "properties": {
        "languages": {"type": "array", "items": {"type": "string"}},
        "commands": {
            "type": "object",
            "additionalProperties": False,
            "required": ["build", "test", "lint"],
            "properties": {
                key: {"type": "array", "items": {"type": "string"}}
                for key in ("build", "test", "lint")
            },
        },
        "conventions": {"type": "array", "items": {"type": "string"}},
        "intent_docs": {"type": "array", "items": {"type": "string"}},
    },
}


def _service_dict(service: Service) -> dict[str, str]:
    return {
        "name": service.name,
        "root": service.root.as_posix(),
        "source": service.source,
    }


def _tracked_files(repo: Path) -> list[str]:
    proc = git_ops._run_git(  # noqa: SLF001 - git_ops owns the subprocess boundary
        repo,
        ["ls-files", "-z"],
        capture_bytes=True,
    )
    if proc.returncode != 0:
        return []
    stdout = proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    return [
        path.decode("utf-8", errors="surrogateescape")
        for path in stdout.split(b"\0")
        if path
    ]


def _build_recon_prompt(
    repo: Path, services: list[Service], exploration_summary: str
) -> str:
    service_lines = "\n".join(
        f"- {service.name}: {service.root.as_posix()}" for service in services
    )
    return f"""IMPROVE_RECON

Read the repository at {repo} without modifying it. Return structured
reconnaissance facts only:

- languages and frameworks in active use;
- build, test, and lint commands supported by repository files;
- conventions that implementation plans must preserve;
- intent documents such as README, roadmap, ADR, and architecture files.

Services:
{service_lines or "- repository root"}

Existing repository scan:
{exploration_summary or "No additional conventions detected."}
"""


async def _step_recon(ctx: FlowContext) -> Stop | None:
    """Enumerate services, inspect repository conventions, and detect stacks."""
    target = ctx.work.repo
    directory: Path = ctx.data["improve_dir"]
    services = enumerate_services(
        target,
        ctx.config.file_config or DaydreamFileConfig(),
    )
    if ctx.config.improve_scope:
        try:
            services = filter_scope(services, ctx.config.improve_scope)
        except ValueError as exc:
            print_error(console, "Invalid Improve Scope", str(exc))
            return Stop(1)

    services_path(directory).write_text(
        json.dumps(
            {"services": [_service_dict(service) for service in services]},
            indent=2,
        )
        + "\n"
    )

    backend = ctx.backend_for("recon")
    async with phase_scope(DaydreamPhase.RECON):
        exploration = await repo_scan(backend, target)
        recon, _, _ = await run_agent(
            backend,
            target,
            _build_recon_prompt(target, services, exploration.to_prompt_section()),
            phase=DaydreamPhase.RECON,
            output_schema=_RECON_SCHEMA,
            read_only=True,
        )

    recon_data = recon if isinstance(recon, dict) else {}
    recon_path(directory).write_text(json.dumps(recon_data, indent=2) + "\n")

    installed = get_installed_skills()
    availability = (
        installed if installed is not None else ctx.registry.stack_keys()
    )
    stacks = detect_stacks(
        _tracked_files(target),
        skill_availability=availability,
        registry=ctx.registry,
    )
    ctx.data["services"] = services
    ctx.data["recon"] = recon_data
    ctx.data["stacks"] = stacks
    return None


def _render_report(
    services: list[Service], stacks: list[StackAssignment]
) -> str:
    service_lines = (
        "\n".join(
            f"- **{service.name}** — `{service.root.as_posix()}`"
            for service in services
        )
        or "- No service roots detected."
    )
    stack_lines = (
        "\n".join(f"- **{stack.stack_name}**" for stack in stacks)
        or "- No stacks detected."
    )
    return (
        "# Improve Report\n\n"
        "## Services\n\n"
        f"{service_lines}\n\n"
        "## Stacks\n\n"
        f"{stack_lines}\n\n"
        "## What ran\n\n"
        "- Read-only repository reconnaissance\n"
    )


async def _step_report(ctx: FlowContext) -> None:
    """Render the initial recon-only improve report."""
    report_path(ctx.data["improve_dir"]).write_text(
        _render_report(ctx.data["services"], ctx.data["stacks"])
    )
    print_success(
        console,
        "Improve reconnaissance complete: "
        f"{len(ctx.data['services'])} services, {len(ctx.data['stacks'])} stacks.",
    )


STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="recon", run=_step_recon),
    FlowStep(
        name="improve-report",
        run=_step_report,
        config_phase="recon",
    ),
)
