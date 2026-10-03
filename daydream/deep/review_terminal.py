"""Finalize one review result inside the recorder, before artifact publication."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema

from daydream.agent import console
from daydream.deep.artifacts import DeepArtifact
from daydream.flows.engine import FlowContext
from daydream.json_utils import atomic_write_json
from daydream.phases.schemas import MERGED_ITEMS_SCHEMA
from daydream.pr_review import PRInfo, resolve_review_renderers
from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info
from daydream.review_budget import review_warnings
from daydream.review_result import ReasonCode
from daydream.trajectory import get_current_recorder
from daydream.ui import print_warning


def finalize_review(ctx: FlowContext, pipeline_state: str, *, no_diff: bool = False) -> int:
    """Freeze validated coverage and findings once, without dispatching a model."""
    from daydream.runner import _emit_findings_from_items

    state = ctx.deep_data()
    coverage = state["review_coverage"]
    items: list[dict[str, Any]] = []
    projection_valid = no_diff
    recovery_failed = False
    path = ctx.data.get("items_file", DeepArtifact.MERGED_ITEMS.at(state["dd"]))
    if not no_diff and isinstance(path, Path) and not path.is_file() and "record_pool" in ctx.data:
        from daydream.phases.findings import _write_single_stack_merged_items
        try:
            _write_single_stack_merged_items(
                ctx.work.repo, state["dd"], state["record_pool"],
                failed_stacks=state["review_coverage"].unfinished_scopes or None, artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
        except ValueError as exc:
            recovery_failed = True
            print_warning(console, f"Cannot recover terminal findings: {type(exc).__name__}")
    if not no_diff and isinstance(path, Path) and path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Host canonical items add UIDs/placement provenance and optional held items.
            # Validate authored fields before projecting; never normalize malformed records away.
            schema = copy.deepcopy(MERGED_ITEMS_SCHEMA)
            schema["additionalProperties"] = True
            item_schema = schema["properties"]["items"]["items"]
            item_schema["additionalProperties"] = True
            item_schema["required"].remove("related_files")
            item_schema["properties"]["confidence"] = {"enum": ["HIGH", "MEDIUM", "LOW"]}
            jsonschema.validate(payload, schema)
            items = payload["items"]
            projection_valid = True
            if "findings" in coverage.required_phases:
                coverage.record_phase("findings", "complete")
        except (OSError, ValueError, jsonschema.ValidationError) as exc:
            coverage.require_phase("findings")
            coverage.record_phase("findings", "failed", reasons=[ReasonCode.MALFORMED_ARTIFACT])
            print_warning(console, f"Cannot validate terminal findings: {type(exc).__name__}")
    elif not no_diff:
        coverage.require_phase("findings")
        coverage.record_phase("findings", "failed", reasons=[
            ReasonCode.MALFORMED_ARTIFACT if recovery_failed else ReasonCode.MISSING_ARTIFACT,
        ])
    if pipeline_state == "completed" and projection_valid and "pipeline" in coverage.required_phases:
        coverage.record_phase("pipeline", "complete")
    result = coverage.finalize(pipeline_state, projection_valid=projection_valid)
    atomic_write_json(DeepArtifact.REVIEW_COVERAGE.at(state["dd"]), coverage.to_dict())
    if ctx.config.findings_out is None:
        return 0
    pr = ctx.data.get("analyzed_pr")
    if not isinstance(pr, PRInfo):
        return 1
    info = render_live_run_info(LiveRunInfoSource(get_current_recorder(), ctx.artifacts))
    if info.diagnostic is not None:
        print_warning(console, info.diagnostic)
    return _emit_findings_from_items(
        ctx.work.repo, ctx.config, items, run_info=info.markdown,
        review_warnings=review_warnings(state["dd"]), renderers=resolve_review_renderers(ctx.registry),
        diagrams=(state.get("diagrams") or {}).get("payload"), auth=ctx.github_execution.auth,
        captured_pr=pr, terminal_result=result, snapshot_diff=ctx.data["snapshot_diff"],
    )
