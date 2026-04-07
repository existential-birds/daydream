"""Tests for the per-subagent ExplorationLivePanel (D-15)."""

from __future__ import annotations

from daydream.ui import ExplorationLivePanel, create_console


def test_exploration_live_panel_rows():
    console = create_console()
    panel = ExplorationLivePanel(console, tier="parallel")

    assert set(panel.row_keys) == {"pattern-scanner", "dependency-tracer", "test-mapper"}
    for key in panel.row_keys:
        assert panel.states[key] == "pending"

    for key in panel.row_keys:
        panel.mark_done(key)
        assert panel.states[key] == "done"


def test_exploration_live_panel_single_tier():
    console = create_console()
    panel = ExplorationLivePanel(console, tier="single")
    assert panel.row_keys == ("dependency-tracer",)
    panel.mark_start("dependency-tracer")
    assert panel.states["dependency-tracer"] == "running"
    panel.mark_done("dependency-tracer")
    assert panel.states["dependency-tracer"] == "done"


def test_exploration_live_panel_marks_pending_as_failed_on_exit():
    console = create_console()
    panel = ExplorationLivePanel(console, tier="parallel")
    # Enter and exit without calling mark_done -- everything should flip to failed.
    with panel:
        panel.mark_done("pattern-scanner")
    assert panel.states["pattern-scanner"] == "done"
    assert panel.states["dependency-tracer"] == "failed"
    assert panel.states["test-mapper"] == "failed"
