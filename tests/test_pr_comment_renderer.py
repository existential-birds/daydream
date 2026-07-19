"""Tests for daydream.pr_comment_renderer.

One test per Must-Have requirement (M1–M10), plus single-phase baseline,
purity/idempotence, and the deep-mode multi-trajectory aggregation case
called out in S2.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from daydream.pr_comment_renderer import _PHASE_LABELS, _format_duration, render_run_info_block
from daydream.trajectory import DaydreamPhase

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
_CORRUPTED = _FIXTURE_DIR / "corrupted.json"


def _agent_step(
    *,
    step_id: int,
    phase: str,
    model: str | None,
    prompt: int = 1000,
    completion: int = 100,
    cached: int = 500,
    cost_usd: float | None = None,
    tool_calls: int = 0,
) -> dict[str, Any]:
    """Build a minimal valid agent step dict for fixture construction."""
    step: dict[str, Any] = {
        "step_id": step_id,
        "timestamp": f"2026-05-02T00:00:{step_id:02d}.000000Z",
        "source": "agent",
        "message": "ok",
        "extra": {"daydream_phase": phase, "daydream_run_flow": "ttt"},
        "metrics": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
            "cost_usd": cost_usd,
        },
    }
    if model is not None:
        step["model_name"] = model
    if tool_calls > 0:
        tcs = [
            {"tool_call_id": f"s{step_id}-t{i}", "function_name": "X", "arguments": {}}
            for i in range(tool_calls)
        ]
        step["tool_calls"] = tcs
        step["observation"] = {
            "results": [{"source_call_id": tc["tool_call_id"], "content": "ok"} for tc in tcs]
        }
    return step


def _user_step(phase: str = "review") -> dict[str, Any]:
    """Build the leading user step (step_id 1) the Trajectory validator expects."""
    return {
        "step_id": 1,
        "timestamp": "2026-05-02T00:00:00.000000Z",
        "source": "user",
        "message": "go",
        "extra": {"daydream_phase": phase, "daydream_run_flow": "ttt"},
    }


def _write_trajectory(
    tmp_path: Path,
    *,
    name: str = "t.json",
    session_id: str = "fixture",
    model: str = "gpt-5.5",
    steps: list[dict[str, Any]],
) -> Path:
    """Write a trajectory dict to tmp and return the path.

    Wraps the steps with a minimal Agent block + the single user step the
    Trajectory validator expects (step_id 1 sequential).
    """
    full = {
        "schema_version": "ATIF-v1.6",
        "session_id": session_id,
        "agent": {"name": "daydream", "version": "0.14.0", "model_name": model},
        "steps": steps,
    }
    p = tmp_path / name
    p.write_text(json.dumps(full), encoding="utf-8")
    return p


def test_archived_claude_sonnet_5_usage_keeps_its_introductory_rate(tmp_path: Path) -> None:
    """Rendering after the transition uses the archived step's usage date."""
    step = _agent_step(
        step_id=2,
        phase="review",
        model="claude-sonnet-5",
        prompt=2_000_000,
        completion=1_000_000,
        cached=1_000_000,
    )
    step["timestamp"] = "2026-08-31T12:00:00.000000Z"
    trajectory = _write_trajectory(tmp_path, steps=[_user_step(), step], model="claude-sonnet-5")
    assert "- **Cost:** $12.20" in render_run_info_block([trajectory])


# M1 — visible rollup
def test_m1_visible_rollup_includes_required_fields() -> None:
    """Rollup must list Model, Cost, Tokens, Steps/tool calls (Mode line dropped)."""
    out = render_run_info_block([_SINGLE_PHASE])
    assert "- **Mode:**" not in out
    assert "- **Model:** claude-sonnet-4-5" in out
    assert "- **Cost:** $0.13" in out
    assert "- **Tokens:** 12,400 in" in out
    assert "- **Steps / tool calls:** 1 / 2" in out


# M2 — collapsed per-phase table
def test_m2_per_phase_breakdown_is_collapsed_table() -> None:
    """Per-phase breakdown is a markdown table inside <details>."""
    out = render_run_info_block([_MULTI_PHASE_CODEX])
    assert "<details><summary>Per-phase breakdown</summary>" in out
    assert "</details>" in out
    assert "| Phase | Model | Tools | Input (cached) | Output | Cost |" in out
    assert "| Review |" in out
    assert "| Fix |" in out


# M4 — uniform field set
def test_m4_uniform_layout() -> None:
    """Section headers/columns are stable across separate calls."""
    a = render_run_info_block([_SINGLE_PHASE])
    for label in ("- **Model:**", "- **Cost:**", "- **Tokens:**", "- **Steps / tool calls:**"):
        assert label in a
    header = "| Phase | Model | Tools | Input (cached) | Output | Cost |"
    assert header in a
    assert "- **Mode:**" not in a


# M5 — cost source per backend
def test_m5_cost_source_per_backend(tmp_path: Path) -> None:
    """Claude steps use Metrics.cost_usd verbatim; Codex steps synthesize from pricing.

    A trajectory with one Claude step (cost_usd=0.50) and one Codex step
    on gpt-5.5 (cost_usd=None, 1M uncached input tokens at $5.00/1M) should
    sum to $5.50: 0.50 SDK + 5.00 synthesized.
    """
    p = _write_trajectory(
        tmp_path,
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="claude-sonnet-4-5",
                prompt=1000,
                completion=0,
                cached=0,
                cost_usd=0.50,
            ),
            _agent_step(
                step_id=3,
                phase="fix",
                model="gpt-5.5",
                prompt=1_000_000,
                completion=0,
                cached=0,
                cost_usd=None,
            ),
        ],
    )
    out = render_run_info_block([p])
    # 0.50 + 5.00 = 5.50
    assert "- **Cost:** $5.50" in out


# M6 — unknown-model fallback
def test_m6_unknown_model_renders_dash_and_footnote(tmp_path: Path) -> None:
    """Unknown OpenAI model: rollup '—', per-phase '—', footnote names the model."""
    p = _write_trajectory(
        tmp_path,
        model="mystery-model",
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="mystery-model",
                cost_usd=None,
            ),
        ],
    )
    out = render_run_info_block([p])
    assert "- **Cost:** —" in out
    assert "| —" in out
    assert "mystery-model" in out
    assert "not in the price table" in out


# M6b — user price override synthesizes cost for an otherwise-unknown model (#156)
def test_m6b_user_override_synthesizes_cost_for_unknown_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user-supplied price (via DAYDREAM_PRICES_FILE) for an unknown Codex
    model produces a SYNTHESIZED cost — not a dash and not the unavailable
    footnote.

    Extends M6: without the override the model renders '—' (cost unavailable).
    With the override the renderer's resolve_prices(load_user_prices()) merges
    it in and compute_cost synthesizes a dollar figure. This is the #156/#61
    acceptance criterion #1 (Codex cost synthesis reads the overridable table).
    """
    prices_file = tmp_path / "prices.toml"
    prices_file.write_text(
        '[prices."custom-codex-op"]\ninput = 2.0\ncached_input = 0.5\noutput = 8.0\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices_file))

    p = _write_trajectory(
        tmp_path,
        model="custom-codex-op",
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="custom-codex-op",
                prompt=1_000_000,
                completion=0,
                cached=0,
                cost_usd=None,
            ),
        ],
    )
    out = render_run_info_block([p])
    # 1M uncached input * $2.00/1M = $2.00 synthesized (no cost_usd from backend).
    assert "- **Cost:** $2.00" in out
    # The unavailable footnote must NOT appear — the model IS priced via override.
    assert "not in the price table" not in out


# M7 — mixed-model label
def test_m7_mixed_models_render_breakdown_pointer(tmp_path: Path) -> None:
    """Two phases on different models -> rollup Model = 'mixed — see breakdown'."""
    p = _write_trajectory(
        tmp_path,
        steps=[
            _user_step(),
            _agent_step(step_id=2, phase="review", model="gpt-5.5", cost_usd=0.10),
            _agent_step(step_id=3, phase="fix", model="claude-sonnet-4-5", cost_usd=0.20),
        ],
    )
    out = render_run_info_block([p])
    assert "- **Model:** mixed — see breakdown" in out


# M8 — cached-tokens awareness in the rollup
def test_m8_cache_hit_ratio_rendered_when_input_nonzero(tmp_path: Path) -> None:
    """Cache-hit ratio is computed from (cached / input) and shown next to cached count."""
    p = _write_trajectory(
        tmp_path,
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="gpt-5.5",
                prompt=10_000,
                completion=200,
                cached=6_700,
                cost_usd=0.01,
            ),
        ],
    )
    out = render_run_info_block([p])
    # 6700 / 10000 = 67%
    assert "10,000 in (6,700 cached, 67% hit)" in out


# M9 — missing-trajectory fallback
def test_m9_missing_trajectory_renders_fallback(tmp_path: Path) -> None:
    """No paths, missing path, or malformed JSON -> 'run details unavailable' + footer."""
    out = render_run_info_block([])
    assert "- **Mode:**" not in out
    assert "*run details unavailable*" in out
    assert "<sub>Generated by daydream v" in out

    out2 = render_run_info_block([tmp_path / "nope.json"])
    assert "*run details unavailable*" in out2
    assert "<sub>Generated by daydream v" in out2

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    out3 = render_run_info_block([bad])
    assert "*run details unavailable*" in out3
    # Per-phase table must NOT render in fallback mode.
    assert "Per-phase breakdown" not in out3
    assert "<sub>Generated by daydream v" in out3


# M10 — number formatting
def test_m10_number_formatting_rules(tmp_path: Path) -> None:
    """Thousand separators on >=1,000; sub-cent cost renders <$0.01; cache-hit
    omitted when input == 0."""
    # Sub-cent cost
    p = _write_trajectory(
        tmp_path,
        name="subcent.json",
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="gpt-5.5",
                prompt=500,
                completion=10,
                cached=0,
                cost_usd=0.001,
            ),
        ],
    )
    out = render_run_info_block([p])
    assert "- **Cost:** <$0.01" in out
    # 500 in -> no thousand separator
    assert "500 in" in out
    # Input nonzero but cached=0 -> omit hit ratio.
    assert "0% hit" not in out
    assert "cached" not in out.split("**Tokens:**", 1)[1].split("\n", 1)[0]

    # Thousand separators in tokens line.
    p2 = _write_trajectory(
        tmp_path,
        name="big.json",
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="gpt-5.5",
                prompt=33_600,
                completion=6_900,
                cached=22_600,
                cost_usd=0.42,
            ),
        ],
    )
    out2 = render_run_info_block([p2])
    assert "33,600 in" in out2
    assert "22,600 cached" in out2
    assert "6,900 out" in out2

    # Zero-input edge: no hit ratio.
    p3 = _write_trajectory(
        tmp_path,
        name="zero_input.json",
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="gpt-5.5",
                prompt=0,
                completion=10,
                cached=0,
                cost_usd=0.0,
            ),
        ],
    )
    out3 = render_run_info_block([p3])
    # Tokens line uses the no-cache form when input is 0.
    assert "0 in → 10 out" in out3


# Baseline single-phase render
def test_single_phase_run_renders_baseline() -> None:
    """The baseline single-phase Claude trajectory renders all sections cleanly."""
    out = render_run_info_block([_SINGLE_PHASE])
    assert out.startswith("- **Model:**")
    assert "<details><summary>Per-phase breakdown</summary>" in out
    assert out.rstrip().endswith("</sub>")
    assert out.count("| Review |") == 1
    assert "| Fix |" not in out


# Purity / idempotence
def test_renderer_is_pure_idempotent() -> None:
    """Calling the renderer twice with identical inputs returns identical output.

    Pure function: no globals, no time.now(), no env. Two consecutive
    renders must be byte-equal. Catches accidental UUID generation,
    cache-busting timestamps, or random ordering of phase rows.
    """
    a = render_run_info_block([_MULTI_PHASE_CODEX])
    b = render_run_info_block([_MULTI_PHASE_CODEX])
    assert a == b


# S2 — deep-mode multi-trajectory aggregation
def test_aggregates_across_multiple_trajectory_files() -> None:
    """Sibling fork trajectories sum into the parent's per-phase rollup.

    multi_phase_codex.json: review (10K input, 5K cached, 1K out) +
                            fix    (8K input, 4K cached, 2K out).
    sibling_fix.json:       fix    (5K input, 2K cached, 0.5K out).

    Combined fix totals: 13K input, 6K cached, 2.5K output.
    Combined run totals: 23K input, 11K cached, 3.5K output.
    """
    out = render_run_info_block([_MULTI_PHASE_CODEX, _SIBLING_FIX])
    # Rollup totals.
    assert "23,000 in" in out
    assert "11,000 cached" in out
    assert "3,500 out" in out
    # Steps: 2 review-agent steps from multi_phase + ... wait, multi_phase
    # has 1 agent step in review and 1 in fix. sibling has 1 in fix.
    # So total agent steps = 3, tools = 1 + 2 + 1 = 4.
    assert "- **Steps / tool calls:** 3 / 4" in out
    # Fix row aggregates across both files: 13,000 input (46% cached), 2,500 output.
    fix_row = next(line for line in out.splitlines() if line.startswith("| Fix |"))
    assert "13,000" in fix_row
    assert "2,500" in fix_row


# Mode line is gone — assert it never renders.
def test_mode_line_never_renders() -> None:
    """The Mode rollup line was removed; no caller-controlled label appears."""
    out = render_run_info_block([_SINGLE_PHASE])
    assert "- **Mode:**" not in out
    assert "**Mode:**" not in out


# S2 — End-to-end fixture-driven scenarios: load a committed trajectory and
# assert the whole rendered block against the shape a reviewer sees in a PR comment.
def test_e2e_single_phase_claude_renders_full_block() -> None:
    """Single Claude review: rollup + 1-row table + footer, no footnote.

    Covers M1, M2, M3, baseline single-phase (S2). Uses the Task 4 baseline
    fixture verbatim — verifies it satisfies the e2e expectations.
    """
    out = render_run_info_block([_SINGLE_PHASE])
    # Rollup block (Mode line dropped).
    assert "- **Mode:**" not in out
    assert "- **Model:** claude-sonnet-4-5" in out
    assert "- **Cost:** $0.13" in out
    assert "- **Tokens:** 12,400 in (8,200 cached, 66% hit) → 1,800 out" in out
    assert "- **Steps / tool calls:** 1 / 2" in out
    # Per-phase table — one Review row, no Fix.
    assert "<details><summary>Per-phase breakdown</summary>" in out
    assert "| Phase | Model | Tools | Input (cached) | Output | Cost |" in out
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    assert "claude-sonnet-4-5" in review_row
    assert "12,400" in review_row
    assert "1,800" in review_row
    assert "$0.13" in review_row
    assert "| Fix |" not in out
    # No unknown-model footnote.
    assert "not in the price table" not in out
    # Block ends with the version footer (renderer-owned).
    assert out.rstrip().endswith("</sub>")


def test_e2e_multi_phase_renders_per_phase_table_rows() -> None:
    """Review/parse/fix/test all on Claude: four rows in execution order.

    Covers M2 (table content), M4 (uniform layout), and the multi-phase
    Claude scenario from S2. Asserts row order matches the order phases
    were first seen in the trajectory (Review → Parse → Fix → Test).
    """
    out = render_run_info_block([_MULTI_PHASE_CLAUDE])
    # Rollup totals (sum across all four Claude phases).
    assert "- **Model:** claude-sonnet-4-5" in out
    assert "- **Cost:** $0.18" in out
    assert "- **Tokens:** 13,800 in (8,300 cached, 60% hit) → 3,100 out" in out
    assert "- **Steps / tool calls:** 4 / 8" in out
    # Each phase produces exactly one row.
    rows = [line for line in out.splitlines() if line.startswith("| ") and "|" in line[2:]]
    phase_rows = [r for r in rows if not r.startswith("| Phase |") and not r.startswith("|---")]
    assert len(phase_rows) == 4
    # Execution order: Review → Parse Feedback → Fix → Test & Heal.
    labels_in_order = [r.split("|")[1].strip() for r in phase_rows]
    assert labels_in_order == ["Review", "Parse Feedback", "Fix", "Test & Heal"]
    # Spot-check token counts on Fix row (largest).
    fix_row = next(r for r in phase_rows if r.startswith("| Fix |"))
    assert "7,000" in fix_row
    assert "2,000" in fix_row
    assert "$0.10" in fix_row


def test_e2e_mixed_models_renders_mixed_label() -> None:
    """Claude review + Codex fix: rollup says 'mixed', table shows both models.

    Covers M7 (mixed-model rollup label) and M5 (per-backend cost source).
    Codex fix row has cost_usd=None and gets cost synthesized from the
    price table (gpt-5-codex: $1.25/1M in cached + uncached, $10/1M out).
    """
    out = render_run_info_block([_MIXED_MODELS])
    # Rollup model label per M7.
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
    # No 'unknown' footnote — both models price-table-known.
    assert "not in the price table" not in out


def test_e2e_codex_cached_tokens_show_in_rollup() -> None:
    """Codex run with cached_input_tokens > 0: rollup shows hit ratio + synth cost.

    Covers M8 (cache-hit ratio rendered) and exercises the codex pricing
    path end-to-end through the renderer (sanity-check Task 2 fix).
    20K input / 14K cached / 1K out on gpt-5.5 → 70% cache hit, synth cost
    $0.07 (uncached: 6K * $5/1M = $0.030; cached: 14K * $0.50/1M = $0.007;
    output: 1K * $30/1M = $0.030; total $0.067 → rounds to $0.07).
    """
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
    """Unknown model: rollup cost '—', per-phase '—', footnote names model.

    Covers M6. Both Review and Fix ran on the unknown model so both rows
    must show '—' in the cost cell, and the footnote names the model
    exactly once with backticks.
    """
    out = render_run_info_block([_UNKNOWN_MODEL])
    assert "- **Cost:** —" in out
    # Per-phase rows both end in '| — |'.
    rows = [line for line in out.splitlines() if line.startswith("| ") and "|" in line[2:]]
    phase_rows = [r for r in rows if not r.startswith("| Phase |") and not r.startswith("|---")]
    for row in phase_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        # Columns: Phase, Model, Tools, Input (cached), Output, Cost, Latency
        assert cells[5] == "—", f"Cost cell should be '—' for unknown model: {row!r}"
    # Footnote names the unknown model with backticks.
    assert "<sub>Cost unavailable: model `gpt-6.0-experimental` is not in the price table.</sub>" in out


def test_e2e_deep_mode_aggregates_fork_files() -> None:
    """Deep mode: parent + 2 sibling forks aggregate into one rollup.

    Covers C1 (multi-trajectory aggregation) end-to-end. The parent
    trajectory has Review + Parse Feedback steps; forks A and B each have
    one Fix step. Combined Fix row totals must sum across both forks.
    """
    out = render_run_info_block([_DEEP_PARENT, _DEEP_FORK_A, _DEEP_FORK_B])
    # Single model across all files.
    assert "- **Model:** claude-sonnet-4-5" in out
    # Tokens summed across all three files.
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


def test_deep_mode_phase_rows_preserve_traversal_order() -> None:
    """Phase rows in deep mode follow first-encounter order across files.

    Regression for CodeRabbit #6 on PR #66: each fork trajectory restarts
    step_id at 1, so sorting phases by ``first_seen_step_id`` would rank a
    fork's first phase (Fix, step_id=2) ahead of the parent's later phase
    (Parse Feedback, step_id=4). The fix relies on dict insertion order
    instead — parent phases land before fork phases.
    """
    out = render_run_info_block([_DEEP_PARENT, _DEEP_FORK_A, _DEEP_FORK_B])
    phase_rows = [
        line
        for line in out.splitlines()
        if line.startswith("| ") and "---" not in line and "Phase" not in line
    ]
    labels = [row.split("|")[1].strip() for row in phase_rows]
    assert labels == ["Review", "Parse Feedback", "Fix"], labels


def test_e2e_corrupted_trajectory_falls_back() -> None:
    """Truncated/invalid JSON: renderer never raises, posts fallback block.

    Covers M9 + K8 end-to-end. Fallback block has the
    'run details unavailable' note plus the version footer — and
    crucially NOT the per-phase table (no half-rendered output).
    """
    out = render_run_info_block([_CORRUPTED])
    assert "- **Mode:**" not in out
    assert "*run details unavailable*" in out
    assert "Per-phase breakdown" not in out
    assert "| Phase |" not in out
    assert "<sub>Generated by daydream v" in out


# Regression lock: cached turn renders true total input (fix landed in the
# Claude backend fold; this locks the renderer's total-form display contract).
def test_cached_turn_renders_total_input(tmp_path: Path) -> None:
    """A cached Claude turn renders its total input, not the uncached remainder.

    The backend folds cache read/creation into prompt_tokens (the ATIF total),
    so a step with prompt=20000, cached=15000 must render the true total input
    with an honest read/total hit ratio.
    """
    p = _write_trajectory(
        tmp_path,
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="claude-sonnet-4-5",
                prompt=20000,
                completion=800,
                cached=15000,
                cost_usd=0.1,
            ),
        ],
    )
    out = render_run_info_block([p])
    assert "20,000 in (15,000 cached, 75% hit) → 800 out" in out


# Regression: corrupt-metrics bleed (CodeRabbit #3 on PR #66)
def test_metrics_clamped_when_cached_exceeds_prompt(tmp_path: Path) -> None:
    """malformed input where cached > prompt is defensively clamped.

    Per ATIF v1.6, cached_tokens is a SUBSET of prompt_tokens. A trajectory
    that reports prompt=10, cached=20 (corrupt or racy upstream) must not
    bleed the raw 20 into the rollup or per-phase row.
    """
    p = _write_trajectory(
        tmp_path,
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="gpt-5.5",
                prompt=10,
                completion=5,
                cached=20,
                cost_usd=0.0,
            ),
        ],
    )
    out = render_run_info_block([p])
    # Rollup: cached cell shows clamped 10, hit-ratio 100%, never raw 20.
    assert "10 in (10 cached, 100% hit) → 5 out" in out
    assert "20 cached" not in out
    # Per-phase row: Input (cached) cell reads "10 (100%)" — clamped.
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    cells = [c.strip() for c in review_row.split("|")]
    # Columns: ['', 'Review', model, tools, input(cached), output, cost, latency, '']
    assert cells[4] == "10 (100%)"


def test_metrics_clamp_negative_token_counts(tmp_path: Path) -> None:
    """Negative token counts on a step must not surface as negative numbers.

    Token counts are by definition non-negative; a negative value implies a
    corrupt trajectory. The renderer clamps to 0 at aggregation and the
    rollup falls back to the no-cache form (cached==0 omits hit ratio).
    """
    p = _write_trajectory(
        tmp_path,
        steps=[
            _user_step(),
            _agent_step(
                step_id=2,
                phase="review",
                model="gpt-5.5",
                prompt=-5,
                completion=-2,
                cached=-3,
                cost_usd=0.0,
            ),
        ],
    )
    out = render_run_info_block([p])
    # No negative numbers anywhere in the rendered markdown. (Hyphens
    # inside model names like ``gpt-5.5`` and the table separator row
    # ``|---|...`` are fine; we only ban ``-`` adjacent to whitespace or
    # a markdown cell boundary, which is what a leaked negative token
    # count would look like.)
    assert re.search(r"(?:^|\s|\|\s*)-\d", out) is None
    # Tokens line: 0 in → 0 out, no hit-ratio segment (cached==0 rule).
    assert "0 in → 0 out" in out
    assert "cached" not in out.split("**Tokens:**", 1)[1].split("\n", 1)[0]
    assert "% hit" not in out


# Regression: ATIF v1.6 root-agent model fallback (CodeRabbit #2 on PR #66)
def test_step_model_falls_back_to_root_agent_model(tmp_path: Path) -> None:
    """ATIF v1.6: step.model_name=None implies Trajectory.agent.model_name.

    Per the Step.model_name field docstring ("Omission implies the model
    defined in the root-level agent config"), an agent step that omits
    model_name must be attributed to the trajectory's root agent model —
    both for the per-phase model cell AND for cost synthesis. Today
    daydream's recorder always stamps step.model_name explicitly, so this
    is a defensive/spec-conformant guarantee on the renderer side.
    """
    p = _write_trajectory(
        tmp_path,
        # Use a model that's in MODEL_PRICES so cost synthesis lands.
        model="gpt-5.5",
        steps=[
            _user_step(),
            # All agent steps omit model_name -> renderer must fall back to
            # agent.model_name from the root config.
            _agent_step(
                step_id=2,
                phase="review",
                model=None,
                prompt=10_000,
                completion=200,
                cached=0,
                cost_usd=None,
            ),
        ],
    )
    out = render_run_info_block([p])
    # Rollup model line uses the root agent's model, not "unknown".
    assert "- **Model:** gpt-5.5" in out
    assert "- **Model:** unknown" not in out
    # Rollup cost is not the unknown sentinel — synthesized via the price
    # table using the fallback model id.
    assert "- **Cost:** —" not in out
    # Per-phase Model cell also shows the fallback model.
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    cells = [c.strip() for c in review_row.split("|")]
    # Columns: ['', 'Review', model, tools, input(cached), output, cost, latency, '']
    assert cells[2] == "gpt-5.5"
    # No 'unknown model' footnote, since the fallback resolved to a priced model.
    assert "not in the price table" not in out


# Duration formatting
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "—"),
        (0.0, "<1s"),
        (0.5, "<1s"),
        (0.999, "<1s"),
        (1.0, "1s"),
        (30.0, "30s"),
        (59.9, "59s"),
        (60.0, "1m"),
        (61.0, "1m 1s"),
        (150.0, "2m 30s"),
        (3599.0, "59m 59s"),
        (3600.0, "1h"),
        (3660.0, "1h 1m"),
        (7200.0, "2h"),
        (7380.0, "2h 3m"),
    ],
    ids=[
        "none", "zero", "half", "sub-second",
        "1s", "30s", "59s",
        "1m", "1m1s", "2m30s", "59m59s",
        "1h", "1h1m", "2h", "2h3m",
    ],
)
def test_format_duration(seconds: float | None, expected: str) -> None:
    """_format_duration covers None, sub-second, seconds, minutes, and hours."""
    assert _format_duration(seconds) == expected


# Duration in rollup and phase table
def test_duration_in_rollup() -> None:
    """Rollup must include a Duration line derived from step timestamps."""
    out = render_run_info_block([_SINGLE_PHASE])
    assert "- **Duration:** 1s" in out


def test_latency_column_in_phase_table() -> None:
    """Per-phase breakdown must include a Latency column with per-phase durations."""
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
    """Duration renders '—' when step timestamps are absent."""
    p = _write_trajectory(
        tmp_path,
        steps=[
            {
                "step_id": 1,
                "timestamp": None,
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
            {
                "step_id": 2,
                "timestamp": None,
                "source": "agent",
                "message": "ok",
                "model_name": "gpt-5.5",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
                "metrics": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "cached_tokens": 0,
                    "cost_usd": 0.01,
                },
            },
        ],
    )
    out = render_run_info_block([p])
    assert "- **Duration:** —" in out
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    assert review_row.rstrip().endswith("| — |")


def test_deep_mode_latency_aggregates_across_forks() -> None:
    """Fix phase latency spans across fork trajectory files."""
    out = render_run_info_block([_DEEP_PARENT, _DEEP_FORK_A, _DEEP_FORK_B])
    # Fix phase: fork A starts at 01:00, fork B ends at 02:01 -> 61s.
    fix_row = next(line for line in out.splitlines() if line.startswith("| Fix |"))
    assert fix_row.rstrip().endswith("| 1m 1s |")
    # Total run: 00:00:00 to 02:00:01 -> 121s.
    assert "- **Duration:** 2m 1s" in out


# PHASE_LABELS completeness
def test_phase_labels_covers_all_daydream_phases() -> None:
    """Every DaydreamPhase member must have a display label in _PHASE_LABELS."""
    for phase in DaydreamPhase:
        assert phase.value in _PHASE_LABELS, (
            f"DaydreamPhase.{phase.name} (value={phase.value!r}) is missing from _PHASE_LABELS"
        )
