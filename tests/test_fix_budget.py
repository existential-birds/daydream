"""Unit tests for the per-file-group fix budget (issue #201).

Covers the three pieces the real-path enforcement test (in
``test_deep_orchestrator.py``) builds on:

1. ``FileGroupBudget`` — the aggregate guard's ``check``/``add_tokens``/
   ``record_item`` semantics (which ceiling fires, in what order).
2. ``run_agent``'s ``on_metrics`` callback — receives ``prompt_tokens +
   completion_tokens`` per ``MetricsEvent`` and never alters turn behaviour.
3. The config-file parser — ``group_max_*`` overrides parse from
   ``[tool.daydream]`` and junk values degrade to ``None`` (default applies).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.agent import run_agent
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    MetricsEvent,
    ResultEvent,
    TextEvent,
)
from daydream.config import (
    DEFAULT_GROUP_MAX_CUMULATIVE_TOKENS,
    DEFAULT_GROUP_MAX_SERIAL_ITEMS,
    DEFAULT_GROUP_MAX_WALL_S,
)
from daydream.config_file import load_file_config
from daydream.phases import FileGroupBudget
from daydream.trajectory import DaydreamPhase

# -- FileGroupBudget -------------------------------------------------------


def _budget(*, wall: float = 1e9, items: int = 1_000, tokens: int = 1_000_000_000) -> FileGroupBudget:
    """Construct a budget with generous ceilings; callers tighten one at a time."""
    return FileGroupBudget(max_wall_seconds=wall, max_serial_items=items, max_cumulative_tokens=tokens)


def test_fresh_budget_is_under_all_ceilings() -> None:
    assert _budget().check() is None


def test_serial_item_limit_fires_after_n_items() -> None:
    budget = _budget(items=3)
    for _ in range(3):
        assert budget.check() is None  # room remains
        budget.record_item()
    assert budget.items_processed == 3
    assert budget.check() == "group_serial_item_limit"


def test_token_limit_fires_when_cumulative_tokens_reached() -> None:
    budget = _budget(tokens=1000)
    budget.add_tokens(600)
    assert budget.check() is None
    budget.add_tokens(400)  # now 1000 >= 1000
    assert budget.check() == "group_token_budget_exceeded"


def test_wall_limit_fires_when_elapsed_reached() -> None:
    # monotonic elapsed is always >= 0, so a zero ceiling trips deterministically
    # without sleeping (and without the item ceiling masking it: items=1000).
    budget = _budget(wall=0.0, items=1000)
    assert budget.check() == "group_wall_budget_exceeded"


def test_item_limit_takes_precedence_over_token_and_wall() -> None:
    # check() order is items -> wall -> tokens: when several ceilings are breached
    # at once, the reason is deterministic.
    budget = FileGroupBudget(max_wall_seconds=0.0, max_serial_items=1, max_cumulative_tokens=1)
    budget.record_item()
    budget.add_tokens(5)
    assert budget.check() == "group_serial_item_limit"


def test_add_tokens_does_not_bump_item_count() -> None:
    budget = _budget()
    budget.add_tokens(10_000)
    assert budget.items_processed == 0  # only record_item() advances the serial count


# -- run_agent on_metrics callback ----------------------------------------


@dataclass
class _MetricsBackend:
    """Minimal Backend replaying a canned event list (mirrors test_multi_turn_tokens)."""

    model = "mock-model"
    fanout_concurrency = 4
    events: list[AgentEvent]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        events = self.events

        async def _gen() -> AsyncIterator[AgentEvent]:
            for event in events:
                yield event

        return _gen()

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


def _metrics(prompt_tokens: int, completion_tokens: int, msg_id: str) -> MetricsEvent:
    return MetricsEvent(
        message_id=msg_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=None,
        cost_usd=None,
    )


async def test_on_metrics_receives_prompt_plus_completion_per_event(tmp_path: Path) -> None:
    """The callback fires once per MetricsEvent with prompt+completion tokens."""
    backend = _MetricsBackend([
        TextEvent(text="working"),
        _metrics(100, 20, "m1"),
        _metrics(150, 30, "m2"),
        ResultEvent(structured_output=None, continuation=None),
    ])
    received: list[int] = []
    _, _, budget_reason = await run_agent(
        backend, tmp_path, "prompt", phase=DaydreamPhase.FIX, on_metrics=received.append
    )
    assert received == [120, 180]  # (100+20), (150+30)
    assert budget_reason is None  # on_metrics never aborts the turn (Approach B)


async def test_on_metrics_none_is_a_noop(tmp_path: Path) -> None:
    """Omitting on_metrics leaves run_agent behaviour unchanged (no crash)."""
    backend = _MetricsBackend([
        TextEvent(text="working"),
        _metrics(10, 5, "m1"),
        ResultEvent(structured_output=None, continuation=None),
    ])
    output, _, budget_reason = await run_agent(backend, tmp_path, "prompt", phase=DaydreamPhase.FIX)
    assert budget_reason is None
    assert isinstance(output, str)


async def test_on_metrics_feeds_a_file_group_budget(tmp_path: Path) -> None:
    """End-to-end wiring: run_agent -> budget.add_tokens accumulates real tokens."""
    backend = _MetricsBackend([
        _metrics(120_000, 40_000, "m1"),
        ResultEvent(structured_output=None, continuation=None),
    ])
    budget = _budget(tokens=150_000)
    await run_agent(backend, tmp_path, "prompt", phase=DaydreamPhase.FIX, on_metrics=budget.add_tokens)
    # 160_000 >= 150_000 -> the group's token ceiling is now tripped.
    assert budget.check() == "group_token_budget_exceeded"


# -- config-file overrides -------------------------------------------------


def test_group_budget_overrides_parse_from_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.daydream]\n"
        "group_max_wall_s = 300\n"  # int round-trips to float
        "group_max_serial_items = 4\n"
        "group_max_cumulative_tokens = 120000\n"
    )
    cfg = load_file_config(tmp_path)
    assert cfg.group_max_wall_s == 300.0
    assert cfg.group_max_serial_items == 4
    assert cfg.group_max_cumulative_tokens == 120_000


def test_group_budget_absent_is_none_so_defaults_apply(tmp_path: Path) -> None:
    cfg = load_file_config(tmp_path)
    assert cfg.group_max_wall_s is None
    assert cfg.group_max_serial_items is None
    assert cfg.group_max_cumulative_tokens is None
    # The constants remain the fallback the orchestrator resolves to.
    assert DEFAULT_GROUP_MAX_WALL_S == 600.0
    assert DEFAULT_GROUP_MAX_SERIAL_ITEMS == 6
    assert DEFAULT_GROUP_MAX_CUMULATIVE_TOKENS == 200_000


def test_group_budget_junk_values_degrade_to_none(tmp_path: Path) -> None:
    # bool subclasses int/float but is never a meaningful budget; strings/lists too.
    (tmp_path / ".daydream.toml").write_text(
        "group_max_wall_s = true\n"
        'group_max_serial_items = "lots"\n'
        "group_max_cumulative_tokens = false\n"
    )
    cfg = load_file_config(tmp_path)
    assert cfg.group_max_wall_s is None
    assert cfg.group_max_serial_items is None
    assert cfg.group_max_cumulative_tokens is None


def test_group_budget_dotfile_float_overrides_pyproject_int(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.daydream]\ngroup_max_wall_s = 600\n")
    (tmp_path / ".daydream.toml").write_text("group_max_wall_s = 90.5\n")
    cfg = load_file_config(tmp_path)
    assert cfg.group_max_wall_s == 90.5
