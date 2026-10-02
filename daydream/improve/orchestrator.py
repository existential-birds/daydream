"""Registered Improve flow: recon, audit, vet, selection, planning, publication, report."""

from __future__ import annotations

from typing import TYPE_CHECKING

from daydream.extensions.api import FlowStep

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


from daydream.improve.audit import (
    _step_audit,
    _step_vet,
)
from daydream.improve.issue_publication import (
    _step_publish_issues,
)
from daydream.improve.planning import (
    _step_select,
    _step_write_plans,
)
from daydream.improve.recon import (
    _step_recon,
)
from daydream.improve.reporting import (
    _step_report,
)


def _is_audit_run(ctx: FlowContext) -> bool:
    return ctx.config.improve_plan_description is None


STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="recon", run=_step_recon),
    FlowStep(name="audit", run=_step_audit, enabled=_is_audit_run),
    FlowStep(name="vet", run=_step_vet, enabled=_is_audit_run),
    FlowStep(name="select-plans", run=_step_select, enabled=_is_audit_run),
    FlowStep(
        name="write-plans",
        run=_step_write_plans,
        config_phase="plan_write",
    ),
    FlowStep(
        name="publish-improve-issues",
        run=_step_publish_issues,
        config_phase="recon",
    ),
    FlowStep(
        name="improve-report",
        run=_step_report,
        config_phase="recon",
    ),
)
