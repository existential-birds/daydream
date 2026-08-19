"""Shared trajectory/manifest test builders.

One canonical copy of the recorder, trajectory-read, invocation-observe,
manifest and unified-diff builders that were copy-pasted across the trajectory,
archive and training test modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.archive.manifest import Manifest
from daydream.backends import ResultEvent, TextEvent
from daydream.trajectory import (
    DaydreamRunFlow,
    Invocation,
    TrajectoryRecorder,
)


def make_recorder(
    tmp_path: Path,
    *,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL,
    agent_model_name: str = "opus",
    on_write: Any = None,
) -> TrajectoryRecorder:
    """Construct a TrajectoryRecorder rooted in tmp_path."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=run_flow,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
        session_id="test",
        on_write=on_write,
    )


def read_trajectory(path: Path) -> dict[str, Any]:
    """Load the produced trajectory JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def step_token_sum(traj: dict[str, Any], key: str) -> int:
    """Sum ``metrics[key]`` across agent steps that carry it.

    Reconciliation invariant: the step-level rollup's per-dimension sum must
    equal the recorder's final_metrics total (``Σ steps == final``). Steps
    whose ``metrics`` block is absent, or which lack the requested key, are
    skipped so a phantom all-zero residual step can never contribute a zero
    line item to the sum.
    """
    return sum(
        s["metrics"][key]
        for s in traj["steps"]
        if s.get("metrics") and s["metrics"].get(key)
    )


def observe_text_and_result(inv: Invocation, text: str = "output") -> None:
    """Observe a TextEvent + ResultEvent to produce a minimal agent step."""
    inv.observe(TextEvent(text=text))
    inv.observe(ResultEvent(structured_output=None, continuation=None))


def make_manifest(session_id: str = "sess-0001", **overrides: Any) -> Manifest:
    """Build a minimal indexed manifest.

    ``pr_number``/``pr_repo`` are plain ``Manifest`` fields (see
    ``daydream/archive/manifest.py``), so PR-attached rows are produced by
    passing them through ``overrides``.
    """
    defaults: dict[str, Any] = {
        "session_id": session_id,
        "archived_at": "2026-04-29T00:00:00+00:00",
        "status": "complete",
        "run_flow": "normal",
        "skill": "python",
        "model": "opus",
        "backend": "claude",
        "archive_path": "/tmp/archive/runs/sess-0001",
    }
    defaults.update(overrides)
    return Manifest(**defaults)


def diff_adding(line: str, *, file: str = "app.py") -> str:
    """One-hunk unified diff that adds ``line`` to ``file``."""
    return (
        f"diff --git a/{file} b/{file}\n"
        "index 1111111..2222222 100644\n"
        f"--- a/{file}\n"
        f"+++ b/{file}\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        f"+{line}\n"
    )
