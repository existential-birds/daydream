"""Public deep-flow composition contract pinned before stage extraction."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from daydream.deep.orchestrator import DIAGRAM_STEPS, STEPS
from daydream.extensions import LoopGroup, Registry, build_registry
from daydream.flows.engine import FlowContext
from daydream.runner import RunConfig
from daydream.workspace import WorkContext

_DEEP_FIELDS = (
    ("exploration", "exploration", None),
    ("intent", "intent", None),
    ("per-stack-reviews", "per_stack_review", "per_stack_review"),
    ("per-stack-parse", "parse", "parse"),
    ("uncovered-sweep", "parse", "parse"),
    ("arbiter", "arbiter", None),
    ("cross-stack-merge", "merge", "merge"),
    ("single-stack-merge", "single-stack-merge", None),
    ("load-items", "load-items", None),
    ("supervise", "supervise", "supervise"),
    ("diagram", "diagram", "diagram"),
    ("findings-out", "findings-out", None),
    ("post-review", "post-review", None),
    ("fix-gate", "fix-gate", None),
    ("verify", "verify", None),
    ("fix", "fix", None),
    ("fix-verify", "fix-verify", None),
    ("test", "test", None),
    ("commit", "fix", "fix"),
    ("remote-ci", "remote-ci", None),
)


def test_deep_and_diagram_flow_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: Callable[..., RunConfig],
    make_work: Callable[..., WorkContext],
) -> None:
    """Registry order, phase routing, identity, and loop placement stay stable."""
    monkeypatch.delenv("DAYDREAM_EXT_DIR", raising=False)
    registry = build_registry()

    assert tuple((step.name, step.phase_key, step.config_phase) for step in STEPS) == _DEEP_FIELDS

    deep_entries = registry.flow("deep")
    assert tuple(
        entry if isinstance(entry, str) else entry.name for entry in deep_entries
    ) == (
        "exploration",
        "intent",
        "per-stack-reviews",
        "per-stack-parse",
        "uncovered-sweep",
        "arbiter",
        "cross-stack-merge",
        "single-stack-merge",
        "load-items",
        "supervise",
        "diagram",
        "findings-out",
        "post-review",
        "fix-gate",
        "verify",
        "fix-verify-loop",
        "test",
        "commit",
        "remote-ci",
    )
    loop = next(entry for entry in deep_entries if isinstance(entry, LoopGroup))
    assert loop.name == "fix-verify-loop"
    assert loop.steps == ("fix", "fix-verify")
    assert loop.max_iterations(
        FlowContext(
            config=make_config(tmp_path),
            work=make_work(tmp_path),
            registry=Registry(),
        )
    ) == 3

    assert tuple(
        (step.name, step.phase_key, step.config_phase) for step in DIAGRAM_STEPS
    ) == (("post-diagram", "post-diagram", None),)
    assert "post-diagram" not in tuple(step.name for step in STEPS)
    assert registry.flow("diagram") == ["exploration", "diagram", "post-diagram"]
    assert registry.phase("exploration") is STEPS[0]
    assert registry.phase("diagram") is next(step for step in STEPS if step.name == "diagram")
    assert registry.phase("post-diagram") is DIAGRAM_STEPS[0]
