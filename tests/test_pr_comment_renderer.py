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

from daydream.pr_comment_renderer import _PHASE_LABELS, render_run_info_block
from daydream.trajectory import DaydreamPhase

# Fixture trajectory files committed under tests/fixtures/trajectories/.
# Those small-but-realistic files cover Claude (cost_usd present) and
# Codex (cost_usd null, synthesized via pricing.compute_cost). Tests that
# need a *specific* shape (mixed-model, unknown-model, missing data) build
# their trajectory inline via _write_trajectory below.
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "trajectories"
_SINGLE_PHASE = _FIXTURE_DIR / "single_phase_claude.json"
_MULTI_PHASE_CODEX = _FIXTURE_DIR / "multi_phase_codex.json"
_SIBLING_FIX = _FIXTURE_DIR / "sibling_fix.json"
# S2 e2e fixtures — see tests/fixtures/trajectories/ for the JSON.
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


# ---------------------------------------------------------------------------
# M1 — visible rollup
# ---------------------------------------------------------------------------
def test_m1_visible_rollup_includes_required_fields() -> None:
    """Rollup must list Mode, Model, Cost, Tokens, Steps/tool calls."""
    out = render_run_info_block([_SINGLE_PHASE], "trust-the-technology")
    assert "- **Mode:** trust-the-technology" in out
    assert "- **Model:** claude-sonnet-4-5" in out
    assert "- **Cost:** $0.13" in out
    assert "- **Tokens:** 12,400 in" in out
    assert "- **Steps / tool calls:** 1 / 2" in out


# ---------------------------------------------------------------------------
# M2 — collapsed per-phase table
# ---------------------------------------------------------------------------
def test_m2_per_phase_breakdown_is_collapsed_table() -> None:
    """Per-phase breakdown is a markdown table inside <details>."""
    out = render_run_info_block([_MULTI_PHASE_CODEX], "deep review")
    assert "<details><summary>Per-phase breakdown</summary>" in out
    assert "</details>" in out
    # Header row with all required columns.
    assert "| Phase | Model | Tools | Input (cached) | Output | Cost |" in out
    # Both phases present.
    assert "| Review |" in out
    assert "| Fix |" in out


# ---------------------------------------------------------------------------
# M4 — uniform field set across modes
# ---------------------------------------------------------------------------
def test_m4_uniform_layout_across_mode_labels() -> None:
    """Different mode labels produce the same set of section headers/columns."""
    a = render_run_info_block([_SINGLE_PHASE], "comment")
    b = render_run_info_block([_SINGLE_PHASE], "feedback")
    c = render_run_info_block([_SINGLE_PHASE], "deep review")
    for label in ("- **Model:**", "- **Cost:**", "- **Tokens:**", "- **Steps / tool calls:**"):
        assert label in a
        assert label in b
        assert label in c
    # Same column header in all three.
    header = "| Phase | Model | Tools | Input (cached) | Output | Cost |"
    assert header in a
    assert header in b
    assert header in c


# ---------------------------------------------------------------------------
# M5 — cost source per backend
# ---------------------------------------------------------------------------
def test_m5_cost_source_per_backend(tmp_path: Path) -> None:
    """Claude steps use Metrics.cost_usd verbatim; Codex steps synthesize from pricing.

    A trajectory with one Claude step (cost_usd=0.50) and one Codex step
    on gpt-5.5 (cost_usd=None, 1M uncached input tokens at $5.00/1M) should
    sum to $5.50: 0.50 SDK + 5.00 synthesized.
    """
    p = _write_trajectory(
        tmp_path,
        steps=[
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
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
    out = render_run_info_block([p], "ttt")
    # 0.50 + 5.00 = 5.50
    assert "- **Cost:** $5.50" in out


# ---------------------------------------------------------------------------
# M6 — unknown-model fallback
# ---------------------------------------------------------------------------
def test_m6_unknown_model_renders_dash_and_footnote(tmp_path: Path) -> None:
    """Unknown OpenAI model: rollup '—', per-phase '—', footnote names the model."""
    p = _write_trajectory(
        tmp_path,
        model="mystery-model",
        steps=[
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
            _agent_step(
                step_id=2,
                phase="review",
                model="mystery-model",
                cost_usd=None,
            ),
        ],
    )
    out = render_run_info_block([p], "ttt")
    assert "- **Cost:** —" in out
    assert "| —" in out  # per-phase cost cell ends with —
    assert "mystery-model" in out
    assert "not in the price table" in out


# ---------------------------------------------------------------------------
# M7 — mixed-model label
# ---------------------------------------------------------------------------
def test_m7_mixed_models_render_breakdown_pointer(tmp_path: Path) -> None:
    """Two phases on different models -> rollup Model = 'mixed — see breakdown'."""
    p = _write_trajectory(
        tmp_path,
        steps=[
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
            _agent_step(step_id=2, phase="review", model="gpt-5.5", cost_usd=0.10),
            _agent_step(step_id=3, phase="fix", model="claude-sonnet-4-5", cost_usd=0.20),
        ],
    )
    out = render_run_info_block([p], "ttt")
    assert "- **Model:** mixed — see breakdown" in out


# ---------------------------------------------------------------------------
# M8 — cached-tokens awareness in the rollup
# ---------------------------------------------------------------------------
def test_m8_cache_hit_ratio_rendered_when_input_nonzero(tmp_path: Path) -> None:
    """Cache-hit ratio is computed from (cached / input) and shown next to cached count."""
    p = _write_trajectory(
        tmp_path,
        steps=[
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
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
    out = render_run_info_block([p], "ttt")
    # 6700 / 10000 = 67%
    assert "10,000 in (6,700 cached, 67% hit)" in out


# ---------------------------------------------------------------------------
# M9 — missing-trajectory fallback
# ---------------------------------------------------------------------------
def test_m9_missing_trajectory_renders_fallback(tmp_path: Path) -> None:
    """No paths, missing path, or malformed JSON -> Mode line + 'run details unavailable'."""
    # Empty list.
    out = render_run_info_block([], "ttt")
    assert "- **Mode:** ttt" in out
    assert "*run details unavailable*" in out

    # Path that doesn't exist.
    out2 = render_run_info_block([tmp_path / "nope.json"], "ttt")
    assert "*run details unavailable*" in out2

    # Malformed JSON.
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    out3 = render_run_info_block([bad], "ttt")
    assert "*run details unavailable*" in out3
    # Per-phase table must NOT render in fallback mode.
    assert "Per-phase breakdown" not in out3


# ---------------------------------------------------------------------------
# M10 — number formatting
# ---------------------------------------------------------------------------
def test_m10_number_formatting_rules(tmp_path: Path) -> None:
    """Thousand separators on >=1,000; sub-cent cost renders <$0.01; cache-hit
    omitted when input == 0."""
    # Sub-cent cost.
    p = _write_trajectory(
        tmp_path,
        name="subcent.json",
        steps=[
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
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
    out = render_run_info_block([p], "ttt")
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
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
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
    out2 = render_run_info_block([p2], "ttt")
    assert "33,600 in" in out2
    assert "22,600 cached" in out2
    assert "6,900 out" in out2

    # Zero-input edge: no hit ratio.
    p3 = _write_trajectory(
        tmp_path,
        name="zero_input.json",
        steps=[
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
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
    out3 = render_run_info_block([p3], "ttt")
    # Tokens line uses the no-cache form when input is 0.
    assert "0 in → 10 out" in out3


# ---------------------------------------------------------------------------
# Baseline single-phase render
# ---------------------------------------------------------------------------
def test_single_phase_run_renders_baseline() -> None:
    """The baseline single-phase Claude trajectory renders all sections cleanly."""
    out = render_run_info_block([_SINGLE_PHASE], "ttt")
    # Rollup and per-phase table both present.
    assert out.startswith("- **Mode:**")
    assert "<details><summary>Per-phase breakdown</summary>" in out
    assert out.rstrip().endswith("</details>")
    # Single 'Review' phase row only.
    assert out.count("| Review |") == 1
    assert "| Fix |" not in out


# ---------------------------------------------------------------------------
# Purity / idempotence
# ---------------------------------------------------------------------------
def test_renderer_is_pure_idempotent() -> None:
    """Calling the renderer twice with identical inputs returns identical output.

    Pure function: no globals, no time.now(), no env. Two consecutive
    renders must be byte-equal. Catches accidental UUID generation,
    cache-busting timestamps, or random ordering of phase rows.
    """
    a = render_run_info_block([_MULTI_PHASE_CODEX], "deep review")
    b = render_run_info_block([_MULTI_PHASE_CODEX], "deep review")
    assert a == b


# ---------------------------------------------------------------------------
# S2 — deep-mode multi-trajectory aggregation
# ---------------------------------------------------------------------------
def test_aggregates_across_multiple_trajectory_files() -> None:
    """Sibling fork trajectories sum into the parent's per-phase rollup.

    multi_phase_codex.json: review (10K input, 5K cached, 1K out) +
                            fix    (8K input, 4K cached, 2K out).
    sibling_fix.json:       fix    (5K input, 2K cached, 0.5K out).

    Combined fix totals: 13K input, 6K cached, 2.5K output.
    Combined run totals: 23K input, 11K cached, 3.5K output.
    """
    out = render_run_info_block([_MULTI_PHASE_CODEX, _SIBLING_FIX], "deep review")
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


# ---------------------------------------------------------------------------
# Defensive: a fork-only trajectory still uses Mode label & footer.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode_label",
    ["comment", "feedback 123", "deep review", "trust-the-technology"],
)
def test_mode_label_round_trips_verbatim(mode_label: str) -> None:
    """Whatever string the caller hands in for Mode lands verbatim on the line."""
    out = render_run_info_block([_SINGLE_PHASE], mode_label)
    assert f"- **Mode:** {mode_label}" in out


# ---------------------------------------------------------------------------
# S2 — End-to-end fixture-driven scenarios. Each test loads a committed
# trajectory file (see tests/fixtures/trajectories/) and asserts the whole
# rendered block, top to bottom, against the realistic shape a reviewer
# would actually see in a GitHub PR comment.
# ---------------------------------------------------------------------------
def test_e2e_single_phase_claude_renders_full_block() -> None:
    """Single Claude review: rollup + 1-row table + footer, no footnote.

    Covers M1, M2, M3, baseline single-phase (S2). Uses the Task 4 baseline
    fixture verbatim — verifies it satisfies the e2e expectations.
    """
    out = render_run_info_block([_SINGLE_PHASE], "trust-the-technology")
    # Rollup block.
    assert "- **Mode:** trust-the-technology" in out
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
    # Block ends with the per-phase table close (footer is owned by pr_review).
    assert out.rstrip().endswith("</details>")


def test_e2e_multi_phase_renders_per_phase_table_rows() -> None:
    """Review/parse/fix/test all on Claude: four rows in execution order.

    Covers M2 (table content), M4 (uniform layout), and the multi-phase
    Claude scenario from S2. Asserts row order matches the order phases
    were first seen in the trajectory (Review → Parse → Fix → Test).
    """
    out = render_run_info_block([_MULTI_PHASE_CLAUDE], "deep review")
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
    out = render_run_info_block([_MIXED_MODELS], "comment")
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
    out = render_run_info_block([_CODEX_WITH_CACHED], "feedback 99")
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
    out = render_run_info_block([_UNKNOWN_MODEL], "ttt")
    assert "- **Cost:** —" in out
    # Per-phase rows both end in '| — |'.
    rows = [line for line in out.splitlines() if line.startswith("| ") and "|" in line[2:]]
    phase_rows = [r for r in rows if not r.startswith("| Phase |") and not r.startswith("|---")]
    for row in phase_rows:
        assert row.rstrip().endswith("| — |")
    # Footnote names the unknown model with backticks.
    assert "<sub>Cost unavailable: model `gpt-6.0-experimental` is not in the price table.</sub>" in out


def test_e2e_deep_mode_aggregates_fork_files() -> None:
    """Deep mode: parent + 2 sibling forks aggregate into one rollup.

    Covers C1 (multi-trajectory aggregation) end-to-end. The parent
    trajectory has Review + Parse Feedback steps; forks A and B each have
    one Fix step. Combined Fix row totals must sum across both forks.
    """
    out = render_run_info_block(
        [_DEEP_PARENT, _DEEP_FORK_A, _DEEP_FORK_B],
        "deep review",
    )
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
    out = render_run_info_block(
        [_DEEP_PARENT, _DEEP_FORK_A, _DEEP_FORK_B],
        "deep review",
    )
    phase_rows = [
        line
        for line in out.splitlines()
        if line.startswith("| ") and "---" not in line and "Phase" not in line
    ]
    labels = [row.split("|")[1].strip() for row in phase_rows]
    assert labels == ["Review", "Parse Feedback", "Fix"], labels


def test_e2e_corrupted_trajectory_falls_back_to_single_mode_line() -> None:
    """Truncated/invalid JSON: renderer never raises, posts fallback block.

    Covers M9 + K8 end-to-end. Fallback block has the Mode line and the
    'run details unavailable' note — and crucially NOT the per-phase
    table (no half-rendered output).
    """
    out = render_run_info_block([_CORRUPTED], "ttt")
    assert "- **Mode:** ttt" in out
    assert "*run details unavailable*" in out
    assert "Per-phase breakdown" not in out
    assert "| Phase |" not in out


# ---------------------------------------------------------------------------
# Regression: corrupt-metrics bleed (CodeRabbit #3 on PR #66)
# ---------------------------------------------------------------------------
def test_metrics_clamped_when_cached_exceeds_prompt(tmp_path: Path) -> None:
    """cached_tokens > prompt_tokens must be clamped to prompt at aggregation.

    Per ATIF v1.6, cached_tokens is a SUBSET of prompt_tokens. A trajectory
    that reports prompt=10, cached=20 (corrupt or racy upstream) must not
    bleed the raw 20 into the rollup or per-phase row.
    """
    p = _write_trajectory(
        tmp_path,
        steps=[
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
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
    out = render_run_info_block([p], "ttt")
    # Rollup: cached cell shows clamped 10, hit-ratio 100%, never raw 20.
    assert "10 in (10 cached, 100% hit) → 5 out" in out
    assert "20 cached" not in out
    # Per-phase row: Input (cached) cell reads "10 (100%)" — clamped.
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    cells = [c.strip() for c in review_row.split("|")]
    # Columns: ['', 'Review', model, tools, input(cached), output, cost, '']
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
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
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
    out = render_run_info_block([p], "ttt")
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


# ---------------------------------------------------------------------------
# Regression: ATIF v1.6 root-agent model fallback (CodeRabbit #2 on PR #66)
# ---------------------------------------------------------------------------
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
            {
                "step_id": 1,
                "timestamp": "2026-05-02T00:00:00.000000Z",
                "source": "user",
                "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            },
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
    out = render_run_info_block([p], "ttt")
    # Rollup model line uses the root agent's model, not "unknown".
    assert "- **Model:** gpt-5.5" in out
    assert "- **Model:** unknown" not in out
    # Rollup cost is not the unknown sentinel — synthesized via the price
    # table using the fallback model id.
    assert "- **Cost:** —" not in out
    # Per-phase Model cell also shows the fallback model.
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    cells = [c.strip() for c in review_row.split("|")]
    # Columns: ['', 'Review', model, tools, input(cached), output, cost, '']
    assert cells[2] == "gpt-5.5"
    # No 'unknown model' footnote, since the fallback resolved to a priced model.
    assert "not in the price table" not in out


# ---------------------------------------------------------------------------
# PHASE_LABELS completeness
# ---------------------------------------------------------------------------
def test_phase_labels_covers_all_daydream_phases() -> None:
    """Every DaydreamPhase member must have a display label in _PHASE_LABELS."""
    for phase in DaydreamPhase:
        assert phase.value in _PHASE_LABELS, (
            f"DaydreamPhase.{phase.name} (value={phase.value!r}) is missing from _PHASE_LABELS"
        )
