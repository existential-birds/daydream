"""Tests for daydream.runner.RunConfig."""

from __future__ import annotations

from daydream.exploration import ExplorationContext
from daydream.runner import RunConfig


def test_run_config_exploration_depth():
    assert RunConfig().exploration_depth == 1
    cfg = RunConfig(exploration_depth=2)
    assert cfg.exploration_depth == 2


def test_run_config_exploration_context_defaults_to_none():
    cfg = RunConfig()
    assert cfg.exploration_context is None
    explicit = ExplorationContext()
    cfg2 = RunConfig(exploration_context=explicit)
    assert cfg2.exploration_context is explicit
