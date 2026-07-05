"""Unit tests for the per-file-group fix budget (issue #201).

Covers the two pieces the real-path enforcement tests (in
``test_deep_orchestrator.py``) build on:

1. ``FileGroupBudget`` — the aggregate guard's ``check``/``record_item``
   semantics (which ceiling fires, in what order).
2. The config-file parser — ``group_max_*`` overrides parse from
   ``[tool.daydream]`` and junk values degrade to ``None`` (default applies).

The token axis was removed: it can only bound multi-call fallback groups (the
same population ``wall`` + ``serial`` already govern), where output tokens are
collinear with wall-time and call-count, so it added no independent signal.
"""

from __future__ import annotations

from pathlib import Path

from daydream.config import (
    DEFAULT_GROUP_MAX_SERIAL_ITEMS,
    DEFAULT_GROUP_MAX_WALL_S,
)
from daydream.config_file import load_file_config
from daydream.file_group_budget import FileGroupBudget

# -- FileGroupBudget -------------------------------------------------------


def _budget(*, wall: float = 1e9, items: int = 1_000) -> FileGroupBudget:
    """Construct a budget with generous ceilings; callers tighten one at a time."""
    return FileGroupBudget(max_wall_seconds=wall, max_serial_items=items)


def test_fresh_budget_is_under_all_ceilings() -> None:
    assert _budget().check() is None


def test_serial_item_limit_fires_after_n_items() -> None:
    budget = _budget(items=3)
    for _ in range(3):
        assert budget.check() is None  # room remains
        budget.record_item()
    assert budget.items_processed == 3
    assert budget.check() == "group_serial_item_limit"


def test_wall_limit_fires_when_elapsed_reached() -> None:
    # monotonic elapsed is always >= 0, so a zero ceiling trips deterministically
    # without sleeping (and without the item ceiling masking it: items=1000).
    budget = _budget(wall=0.0, items=1000)
    assert budget.check() == "group_wall_budget_exceeded"


def test_item_limit_takes_precedence_over_wall() -> None:
    # check() order is items -> wall: when both ceilings are breached at once,
    # the reason is deterministic.
    budget = FileGroupBudget(max_wall_seconds=0.0, max_serial_items=1)
    budget.record_item()
    assert budget.check() == "group_serial_item_limit"


# -- config-file overrides -------------------------------------------------


def test_group_budget_overrides_parse_from_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.daydream]\n"
        "group_max_wall_s = 300\n"  # int round-trips to float
        "group_max_serial_items = 4\n"
    )
    cfg = load_file_config(tmp_path)
    assert cfg.group_max_wall_s == 300.0
    assert cfg.group_max_serial_items == 4


def test_group_budget_absent_is_none_so_defaults_apply(tmp_path: Path) -> None:
    cfg = load_file_config(tmp_path)
    assert cfg.group_max_wall_s is None
    assert cfg.group_max_serial_items is None
    # The constants remain the fallback the orchestrator resolves to.
    assert DEFAULT_GROUP_MAX_WALL_S == 600.0
    assert DEFAULT_GROUP_MAX_SERIAL_ITEMS == 6


def test_group_budget_junk_values_degrade_to_none(tmp_path: Path) -> None:
    # bool subclasses int/float but is never a meaningful budget; strings/lists too.
    (tmp_path / ".daydream.toml").write_text(
        "group_max_wall_s = true\n"
        'group_max_serial_items = "lots"\n'
    )
    cfg = load_file_config(tmp_path)
    assert cfg.group_max_wall_s is None
    assert cfg.group_max_serial_items is None


def test_group_budget_dotfile_float_overrides_pyproject_int(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.daydream]\ngroup_max_wall_s = 600\n")
    (tmp_path / ".daydream.toml").write_text("group_max_wall_s = 90.5\n")
    cfg = load_file_config(tmp_path)
    assert cfg.group_max_wall_s == 90.5
