"""Review rendering: stable Markdown, token/cost rollups, and fork aggregation."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

import daydream.pr_comment_renderer as renderer
from daydream.atif import Trajectory
from daydream.pr_comment_renderer import (
    FALLBACK_NOTE,
    _format_duration,
    render_run_info,
    render_run_info_block,
)
from daydream.pricing import ModelPrice, resolve_prices
from daydream.trajectory import DaydreamPhase
from tests.harness.trajectory import make_recorder, observe_claude_shape

# Committed fixtures cover Claude (cost_usd present) and Codex (cost_usd null,
# synthesized via pricing). Tests needing a specific shape build inline via _write_trajectory.
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "trajectories"
_SINGLE_PHASE = _FIXTURE_DIR / "single_phase_claude.json"
_MULTI_PHASE_CODEX = _FIXTURE_DIR / "multi_phase_codex.json"
_SIBLING_FIX = _FIXTURE_DIR / "sibling_fix.json"
_MULTI_PHASE_CLAUDE = _FIXTURE_DIR / "multi_phase_claude.json"
_MIXED_MODELS = _FIXTURE_DIR / "mixed_models.json"
_CODEX_WITH_CACHED = _FIXTURE_DIR / "codex_with_cached.json"
_UNKNOWN_MODEL = _FIXTURE_DIR / "unknown_model.json"
_DEEP_PARENT = _FIXTURE_DIR / "deep_mode_parent.json"
_DEEP_FORK_A = _FIXTURE_DIR / "deep_mode_fork_a.json"
_DEEP_FORK_B = _FIXTURE_DIR / "deep_mode_fork_b.json"


def _agent_step(
    *, step_id: int, phase: str, model: str | None, prompt: int = 1000, completion: int = 100, cached: int = 500,
    cost_usd: float | None = None, tool_calls: int = 0,
) -> dict[str, Any]:
    """Build a minimal valid agent step dict for fixture construction."""
    step: dict[str, Any] = {
        "step_id": step_id, "timestamp": f"2026-05-02T00:00:{step_id:02d}.000000Z", "source": "agent", "message": "ok",
        "extra": {"daydream_phase": phase, "daydream_run_flow": "ttt"},
        "metrics": {
            "prompt_tokens": prompt, "completion_tokens": completion, "cached_tokens": cached, "cost_usd": cost_usd,
        },
    }
    if model is not None:
        step["model_name"] = model
    if tool_calls > 0:
        tcs = [{"tool_call_id": f"s{step_id}-t{i}", "function_name": "X", "arguments": {}} for i in range(tool_calls)]
        step["tool_calls"] = tcs
        step["observation"] = {"results": [{"source_call_id": tc["tool_call_id"], "content": "ok"} for tc in tcs]}
    return step


def _user_step(phase: str = "review") -> dict[str, Any]:
    """Build the leading user step (step_id 1) the Trajectory validator expects."""
    return {"step_id": 1, "timestamp": "2026-05-02T00:00:00.000000Z", "source": "user", "message": "go",
        "extra": {"daydream_phase": phase, "daydream_run_flow": "ttt"},
    }


def _write_trajectory(tmp_path: Path, *, name: str = "t.json", session_id: str = "fixture", model: str = "gpt-5.5",
    steps: list[dict[str, Any]],
) -> Path:
    """Write validator-compatible Agent and sequential user/agent steps."""
    full = {"schema_version": "ATIF-v1.6", "session_id": session_id,
        "agent": {"name": "daydream", "version": "0.14.0", "model_name": model}, "steps": steps,
    }
    p = tmp_path / name
    p.write_text(json.dumps(full), encoding="utf-8")
    return p


def _single_phase_trajectory(
    tmp_path: Path, *, name: str = "t.json", model: str | None = "gpt-5.5", phase: str = "review", prompt: int = 1000,
    completion: int = 100, cached: int = 500, cost_usd: float | None = None,
) -> Path:
    """Write one user and one agent step; model=None uses the gpt-5.5 root fallback."""
    return _write_trajectory(tmp_path, name=name, model=model or "gpt-5.5",
        steps=[_user_step(),
            _agent_step(step_id=2, phase=phase, model=model, prompt=prompt, completion=completion, cached=cached,
                cost_usd=cost_usd,
            ),
        ],
    )


def _sonnet5_transition_step() -> dict[str, Any]:
    """Archived claude-sonnet-5 review step at its introductory rate."""
    step = _agent_step(
        step_id=2, phase="review", model="claude-sonnet-5", prompt=2_000_000, completion=1_000_000, cached=1_000_000,
    )
    step["timestamp"] = "2026-08-31T12:00:00.000000Z"
    return step

def test_archived_claude_sonnet_5_usage_keeps_its_introductory_rate(tmp_path: Path) -> None:
    """Rendering after the transition uses the archived step's usage date."""
    step = _sonnet5_transition_step()
    trajectory = _write_trajectory(tmp_path, steps=[_user_step(), step], model="claude-sonnet-5")
    assert "- **Cost:** $12.20" in render_run_info_block([trajectory])


def test_m6b_user_override_synthesizes_cost_for_unknown_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices_file = tmp_path / "prices.toml"
    prices_file.write_text(
        '[prices."custom-codex-op"]\ninput = 2.0\ncached_input = 0.5\noutput = 8.0\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices_file))

    p = _single_phase_trajectory(
        tmp_path, model="custom-codex-op", prompt=1_000_000, completion=0, cached=0, cost_usd=None,
    )
    out = render_run_info_block([p])
    # 1M uncached input * $2.00/1M = $2.00 synthesized (no cost_usd from backend).
    assert "- **Cost:** $2.00" in out
    assert "not in the price table" not in out





def test_m10_number_formatting_rules(tmp_path: Path) -> None:
    """Thousand separators on >=1,000; sub-cent cost renders <$0.01; cache-hit omitted when input == 0."""
    p = _single_phase_trajectory(tmp_path, name="subcent.json", prompt=500, completion=10, cached=0, cost_usd=0.001,)
    out = render_run_info_block([p])
    assert "- **Cost:** <$0.01" in out
    assert "500 in" in out
    assert "0% hit" not in out
    assert "cached" not in out.split("**Tokens:**", 1)[1].split("\n", 1)[0]

    p2 = _single_phase_trajectory(
        tmp_path, name="big.json", prompt=33_600, completion=6_900, cached=22_600, cost_usd=0.42,
    )
    out2 = render_run_info_block([p2])
    assert "33,600 in" in out2
    assert "22,600 cached" in out2
    assert "6,900 out" in out2

    p3 = _single_phase_trajectory(tmp_path, name="zero_input.json", prompt=0, completion=10, cached=0, cost_usd=0.0,)
    out3 = render_run_info_block([p3])
    assert "0 in → 10 out" in out3



def test_value_and_path_renderers_are_byte_identical() -> None:
    paths = [_DEEP_PARENT, _DEEP_FORK_A, _DEEP_FORK_B]
    trajectories = [Trajectory.model_validate_json(path.read_bytes()) for path in paths]

    assert render_run_info(trajectories) == render_run_info_block(paths)

def test_value_renderer_does_not_read_ambient_user_prices(monkeypatch: pytest.MonkeyPatch,) -> None:

    def fail() -> dict[str, ModelPrice]:
        raise AssertionError("value renderer read ambient pricing")

    monkeypatch.setattr(renderer, "load_user_prices", fail)
    trajectory = Trajectory.model_validate_json(_MULTI_PHASE_CODEX.read_bytes())

    assert "- **Cost:**" in render_run_info([trajectory])

def test_value_renderer_respects_an_explicit_empty_price_table() -> None:
    trajectory = Trajectory.model_validate_json(_MULTI_PHASE_CODEX.read_bytes())

    rendered = render_run_info([trajectory], prices={})

    assert "- **Cost:** —" in rendered
    assert "not in the price table" in rendered

def test_value_renderer_preserves_resolved_price_policy(tmp_path: Path) -> None:
    trajectory = Trajectory.model_validate_json(_write_trajectory(
            tmp_path, model="claude-sonnet-5", steps=[_user_step(), _sonnet5_transition_step()],
        ).read_bytes()
    )
    resolved = resolve_prices()

    assert "- **Cost:** $12.20" in render_run_info([trajectory], prices=resolved)


def test_path_renderer_skips_price_lookup_when_no_document_is_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    attempts = 0

    def fail() -> dict[str, ModelPrice]:
        nonlocal attempts
        attempts += 1
        return {}

    monkeypatch.setattr(renderer, "load_user_prices", fail)

    assert FALLBACK_NOTE in render_run_info_block([])
    assert FALLBACK_NOTE in render_run_info_block([malformed])
    assert attempts == 0


def test_aggregates_across_multiple_trajectory_files() -> None:
    """Parent and sibling fixes combine to 13K input, 6K cached, and 2.5K output.

    Including review, the run totals 23K input, 11K cached, and 3.5K output."""
    out = render_run_info_block([_MULTI_PHASE_CODEX, _SIBLING_FIX])
    assert "23,000 in" in out
    assert "11,000 cached" in out
    assert "3,500 out" in out
    assert "- **Steps / tool calls:** 3 / 4" in out
    fix_row = next(line for line in out.splitlines() if line.startswith("| Fix |"))
    assert "13,000" in fix_row
    assert "2,500" in fix_row


def test_e2e_single_phase_claude_renders_full_block() -> None:
    out = render_run_info_block([_SINGLE_PHASE])
    assert "- **Mode:**" not in out
    assert "**Mode:**" not in out
    assert out.startswith("- **Model:**")
    assert "- **Model:** claude-sonnet-4-5" in out
    assert "- **Cost:** $0.13" in out
    assert "- **Tokens:** 12,400 in (8,200 cached, 66% hit) → 1,800 out" in out
    assert "- **Steps / tool calls:** 1 / 2" in out
    assert "<details><summary>Per-phase breakdown</summary>" in out
    assert "| Phase | Model | Tools | Input (cached) | Output | Cost |" in out
    assert out.count("| Review |") == 1
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    assert "claude-sonnet-4-5" in review_row
    assert "12,400" in review_row
    assert "1,800" in review_row
    assert "$0.13" in review_row
    assert "| Fix |" not in out
    assert "not in the price table" not in out
    assert out.rstrip().endswith("</sub>")

def test_e2e_multi_phase_renders_per_phase_table_rows() -> None:
    out = render_run_info_block([_MULTI_PHASE_CLAUDE])
    assert "</details>" in out
    assert "- **Model:** claude-sonnet-4-5" in out
    assert "- **Cost:** $0.18" in out
    assert "- **Tokens:** 13,800 in (8,300 cached, 60% hit) → 3,100 out" in out
    assert "- **Steps / tool calls:** 4 / 8" in out
    rows = [line for line in out.splitlines() if line.startswith("| ") and "|" in line[2:]]
    phase_rows = [r for r in rows if not r.startswith("| Phase |") and not r.startswith("|---")]
    assert len(phase_rows) == 4
    labels_in_order = [r.split("|")[1].strip() for r in phase_rows]
    assert labels_in_order == ["Review", "Parse Feedback", "Fix", "Test & Heal"]
    fix_row = next(r for r in phase_rows if r.startswith("| Fix |"))
    assert "7,000" in fix_row
    assert "2,000" in fix_row
    assert "$0.10" in fix_row

def test_e2e_mixed_models_renders_mixed_label() -> None:
    """Mixed Claude/Codex output retains both models and synthesizes missing Codex cost."""
    out = render_run_info_block([_MIXED_MODELS])
    assert "- **Model:** mixed — see breakdown" in out
    # Cost: 0.08 (Claude SDK) + synth(gpt-5-codex, 5K uncached + 5K cached + 2K out)
    # = 0.08 + (5000*1.25 + 5000*1.25 + 2000*10)/1M = 0.08 + 0.0325 = 0.1125 -> $0.11.
    assert "- **Cost:** $0.11" in out
    # Per-phase: Review row uses claude-sonnet-4-5, Fix row uses gpt-5-codex.
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    fix_row = next(line for line in out.splitlines() if line.startswith("| Fix |"))
    assert "claude-sonnet-4-5" in review_row
    assert "gpt-5-codex" in fix_row
    assert "$0.08" in review_row
    assert "$0.03" in fix_row  # synthesized
    assert "not in the price table" not in out

def test_e2e_codex_cached_tokens_show_in_rollup() -> None:
    """A 70% cache hit on gpt-5.5 yields $0.067, displayed as $0.07."""
    out = render_run_info_block([_CODEX_WITH_CACHED])
    assert "- **Model:** gpt-5.5" in out
    assert "- **Cost:** $0.07" in out
    assert "- **Tokens:** 20,000 in (14,000 cached, 70% hit) → 1,000 out" in out
    # Per-phase row mirrors the rollup since there's only one phase.
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    assert "gpt-5.5" in review_row
    assert "20,000" in review_row
    assert "$0.07" in review_row

def test_e2e_unknown_model_emits_named_footnote() -> None:
    out = render_run_info_block([_UNKNOWN_MODEL])
    assert "- **Cost:** —" in out
    rows = [line for line in out.splitlines() if line.startswith("| ") and "|" in line[2:]]
    phase_rows = [r for r in rows if not r.startswith("| Phase |") and not r.startswith("|---")]
    for row in phase_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        # Columns: Phase, Model, Tools, Input (cached), Output, Cost, Latency
        assert cells[5] == "—", f"Cost cell should be '—' for unknown model: {row!r}"
    assert "<sub>Cost unavailable: model `gpt-6.0-experimental` is not in the price table.</sub>" in out

def test_e2e_deep_mode_aggregates_fork_files() -> None:
    """Parent review/parse and two sibling fixes form one rollup with summed Fix totals."""
    out = render_run_info_block([_DEEP_PARENT, _DEEP_FORK_A, _DEEP_FORK_B])
    assert "- **Model:** claude-sonnet-4-5" in out
    # Input: 8000 (parent review) + 1000 (parent parse) + 3000 (fork A) + 2000 (fork B) = 14,000
    # Cached: 4000 + 800 + 1500 + 1000 = 7,300
    # Output: 1500 + 200 + 800 + 500 = 3,000
    assert "14,000 in" in out
    assert "7,300 cached" in out
    assert "3,000 out" in out
    # Cost: 0.10 + 0.02 + 0.04 + 0.03 = 0.19
    assert "- **Cost:** $0.19" in out
    # Steps: 4 agent steps total. Tools: 2 (parent review) + 0 (parse) + 2 + 1 = 5.
    assert "- **Steps / tool calls:** 4 / 5" in out
    # Fix row aggregates across both forks: 5,000 input (50% cached) / 1,300 out.
    fix_row = next(line for line in out.splitlines() if line.startswith("| Fix |"))
    assert "5,000" in fix_row
    assert "1,300" in fix_row
    assert "$0.07" in fix_row  # 0.04 + 0.03



def test_metrics_clamped_when_cached_exceeds_prompt(tmp_path: Path) -> None:
    """Cached tokens are a subset of prompt tokens, including in malformed input."""
    p = _single_phase_trajectory(tmp_path, prompt=10, completion=5, cached=20, cost_usd=0.0)
    out = render_run_info_block([p])
    assert "10 in (10 cached, 100% hit) → 5 out" in out
    assert "20 cached" not in out
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    cells = [c.strip() for c in review_row.split("|")]
    # Columns: ['', 'Review', model, tools, input(cached), output, cost, latency, '']
    assert cells[4] == "10 (100%)"

def test_metrics_clamp_negative_token_counts(tmp_path: Path) -> None:
    """Clamp corrupt negative counts to zero and omit the zero-cache hit ratio."""
    p = _single_phase_trajectory(tmp_path, prompt=-5, completion=-2, cached=-3, cost_usd=0.0)
    out = render_run_info_block([p])
    # Allow hyphens in model names and table borders; reject negative numeric cells.
    assert re.search(r"(?:^|\s|\|\s*)-\d", out) is None
    assert "0 in → 0 out" in out
    assert "cached" not in out.split("**Tokens:**", 1)[1].split("\n", 1)[0]
    assert "% hit" not in out


def test_step_model_falls_back_to_root_agent_model(tmp_path: Path) -> None:
    """Omitted step models inherit the root model for attribution and cost synthesis."""
    p = _single_phase_trajectory(tmp_path,
        # The agent step omits model_name (model=None) and falls back to the
        # root agent model, which is in MODEL_PRICES so cost synthesis lands.
        model=None, prompt=10_000, completion=200, cached=0, cost_usd=None,
    )
    out = render_run_info_block([p])
    assert "- **Model:** gpt-5.5" in out
    assert "- **Model:** unknown" not in out
    assert "- **Cost:** —" not in out
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    cells = [c.strip() for c in review_row.split("|")]
    # Columns: ['', 'Review', model, tools, input(cached), output, cost, latency, '']
    assert cells[2] == "gpt-5.5"
    assert "not in the price table" not in out

def test_osprey_backend_alias_is_not_rendered_as_model(tmp_path: Path) -> None:
    """The backend name is a fallback, not an actual model identity."""
    p = _single_phase_trajectory(tmp_path, model="osprey", phase="exploration", prompt=0, completion=0, cached=0)

    out = render_run_info_block([p])

    assert "- **Model:** unknown" in out
    exploration_row = next(line for line in out.splitlines() if line.startswith("| Exploration |"))
    assert [cell.strip() for cell in exploration_row.split("|")][2] == "unknown"
    assert "| osprey |" not in out


@pytest.mark.parametrize(("seconds", "expected"),
    [(None, "—"), (0.0, "<1s"), (0.5, "<1s"), (0.999, "<1s"), (1.0, "1s"), (30.0, "30s"), (59.9, "59s"),
        (60.0, "1m"), (61.0, "1m 1s"), (150.0, "2m 30s"), (3599.0, "59m 59s"), (3600.0, "1h"), (3660.0, "1h 1m"),
        (7200.0, "2h"), (7380.0, "2h 3m"),
    ], ids=["none", "zero", "half", "sub-second", "1s", "30s", "59s", "1m", "1m1s", "2m30s", "59m59s",
        "1h", "1h1m", "2h", "2h3m",
    ],
)
def test_format_duration(seconds: float | None, expected: str) -> None:
    """_format_duration covers None, sub-second, seconds, minutes, and hours."""
    assert _format_duration(seconds) == expected


def test_duration_in_rollup() -> None:
    out = render_run_info_block([_SINGLE_PHASE])
    assert "- **Duration:** 1s" in out

def test_latency_column_in_phase_table() -> None:
    out = render_run_info_block([_MULTI_PHASE_CLAUDE])
    assert "| Latency |" in out
    # Each phase spans exactly 1 second in the fixture.
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    assert review_row.rstrip().endswith("| 1s |")
    fix_row = next(line for line in out.splitlines() if line.startswith("| Fix |"))
    assert fix_row.rstrip().endswith("| 1s |")
    # Total duration across all 4 phases: 00:00:00 to 00:00:07 = 7s.
    assert "- **Duration:** 7s" in out

def test_duration_degrades_gracefully(tmp_path: Path) -> None:
    p = _write_trajectory(tmp_path,
        steps=[{"step_id": 1, "timestamp": None, "source": "user", "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            }, {"step_id": 2, "timestamp": None, "source": "agent", "message": "ok", "model_name": "gpt-5.5",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
                "metrics": {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 0, "cost_usd": 0.01},
            },
        ],
    )
    out = render_run_info_block([p])
    assert "- **Duration:** —" in out
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    assert review_row.rstrip().endswith("| — |")

def test_deep_mode_latency_aggregates_across_forks() -> None:
    out = render_run_info_block([_DEEP_PARENT, _DEEP_FORK_A, _DEEP_FORK_B])
    # Fix phase: fork A starts at 01:00, fork B ends at 02:01 -> 61s.
    fix_row = next(line for line in out.splitlines() if line.startswith("| Fix |"))
    assert fix_row.rstrip().endswith("| 1m 1s |")
    # Total run: 00:00:00 to 02:00:01 -> 121s.
    assert "- **Duration:** 2m 1s" in out




def _write_reconciled_trajectory(tmp_path: Path) -> Path:
    """Use the real recorder to reconcile per-turn usage to the final session total.

    Include the leading user step required by the renderer/validator."""

    async def _build() -> Path:
        # Use a priced model so cost reconciliation runs instead of rendering '—'.
        recorder = make_recorder(tmp_path, agent_model_name="claude-sonnet-5")
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe_user_step(prompt="go")
                observe_claude_shape(inv)
        return recorder.path

    return asyncio.run(_build())

def test_reconciled_phase_output_consistent_with_session_total(tmp_path: Path) -> None:
    """Render reconciled per-phase output totals, not the smaller message-usage sum."""
    path = _write_reconciled_trajectory(tmp_path)
    rendered = render_run_info_block([path])
    # Reconciliation must use the session total; the per-message sum is only ~50.
    assert "66,737 out" in rendered
    assert "50 out" not in rendered   # no collapsed value leaks
    assert "$0.50" in rendered
    assert "Cost unavailable" not in rendered
