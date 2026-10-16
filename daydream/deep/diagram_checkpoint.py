"""Snapshot-bound recovery of completed diagram generation on durable resume."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from typing import Any

from daydream.agent import console
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.diagram_prompts import build_flowchart_prompt, build_sequence_diagram_prompt
from daydream.deep.reuse_key import exploration_digest, phase_identity_for
from daydream.deep.state import DeepState
from daydream.flows.engine import FlowContext
from daydream.json_utils import atomic_write_json
from daydream.ui import print_warning


def diagram_checkpoint_key(ctx: FlowContext, *, settings: Any, eligibility: dict[str, Any]) -> str | None:
    """Bind recovery to pinned source, author inputs, model and extension contracts."""
    if not ctx.work.head_sha:
        return None
    if (ctx.registry.prompt("diagram_sequence") is not build_sequence_diagram_prompt
            or ctx.registry.prompt("diagram_flowchart") is not build_flowchart_prompt):
        return None  # Custom closures/globals have no declared durable identity.
    try:
        prompts = {name: inspect.getsource(ctx.registry.prompt(name))
                   for name in ("diagram_sequence", "diagram_flowchart")}
    except (OSError, TypeError):
        return None  # A dynamic override has no durable contract to reuse.
    state = DeepState(ctx.data)
    inputs = {
        "version": 1,
        "head": ctx.work.head_sha,
        "base": ctx.work.base_sha,
        "diff": ctx.data.get("snapshot_diff", state.diff),
        "settings": asdict(settings),
        "eligibility": eligibility,
        "identity": asdict(phase_identity_for(ctx, "diagram")),
        "prompts": prompts,
        "exploration": exploration_digest(state.exploration_dir_or_none),
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def restore_diagram_checkpoint(ctx: FlowContext, key: str | None) -> dict[str, Any] | None:
    """Accept only a complete matching publication; corrupt/stale data is a miss."""
    if key is None or ctx.config.start_at not in ("per-stack", "merge"):
        return None
    dd = DeepState(ctx.data).dd
    try:
        marker = json.loads(DeepArtifact.DIAGRAM_COMPLETE.at(dd).read_text(encoding="utf-8"))
        raw = DeepArtifact.DIAGRAM.at(dd).read_bytes()
        if marker != {"key": key, "sha256": hashlib.sha256(raw).hexdigest()}:
            return None
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
            return None
        return payload
    except (OSError, ValueError):
        return None


def complete_diagram_checkpoint(ctx: FlowContext, key: str | None) -> None:
    """Best-effort recovery metadata must not discard completed diagrams."""
    if key is not None:
        dd = DeepState(ctx.data).dd
        try:
            raw = DeepArtifact.DIAGRAM.at(dd).read_bytes()
            atomic_write_json(DeepArtifact.DIAGRAM_COMPLETE.at(dd), {
                "key": key, "sha256": hashlib.sha256(raw).hexdigest(),
            })
        except OSError as exc:
            print_warning(console, f"Diagram recovery checkpoint unavailable ({type(exc).__name__}); "
                          "completed diagrams remain available.")
