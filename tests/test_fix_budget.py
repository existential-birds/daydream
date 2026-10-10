"""Per-file-group budget precedence and config override parsing.

Real-path enforcement lives in deep_orchestrator/test_precision_budgets_and_tiers.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.config_file import load_file_config
from daydream.file_group_budget import FileGroupBudget
from tests.harness.fake_clock import FakeClock

# -- FileGroupBudget -------------------------------------------------------


def test_item_limit_takes_precedence_over_wall() -> None:
    # check() order is items -> wall: when both ceilings are breached at once,
    # the reason is deterministic.
    budget = FileGroupBudget(max_wall_seconds=0.0, max_serial_items=1)
    budget.record_item()
    assert budget.check() == "group_serial_item_limit"

def test_group_budget_deadline_and_remaining_track_the_injected_clock(monkeypatch: pytest.MonkeyPatch,) -> None:

    fake = FakeClock(monotonic_value=1_000.0).install(monkeypatch)
    budget = FileGroupBudget(max_wall_seconds=600.0, max_serial_items=6)

    assert budget.deadline == 1_600.0
    assert budget.remaining() == 600.0
    fake.advance(599.0)
    assert budget.check() is None
    assert budget.remaining() == 1.0
    fake.advance(1.0)
    assert budget.check() == "group_wall_budget_exceeded"
    assert budget.remaining() == 0.0  # clamped, never negative

# -- config-file overrides -------------------------------------------------

def test_group_budget_junk_values_degrade_to_none(tmp_path: Path) -> None:
    # bool subclasses int/float but is never a meaningful budget; strings/lists too.
    (tmp_path / ".daydream.toml").write_text("group_max_wall_s = true\n" 'group_max_serial_items = "lots"\n')
    cfg = load_file_config(tmp_path)
    assert cfg.group_max_wall_s is None
    assert cfg.group_max_serial_items is None

@pytest.mark.parametrize("raw", ["nan", "inf", "-1", "-0.5"])
def test_group_max_wall_s_rejects_negative_and_non_finite(tmp_path: Path, raw: str) -> None:
    (tmp_path / ".daydream.toml").write_text(f"group_max_wall_s = {raw}\n")
    assert load_file_config(tmp_path).group_max_wall_s is None  # default applies
