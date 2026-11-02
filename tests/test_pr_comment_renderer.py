"""Review rendering: stable Markdown, token/cost rollups, and fork aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import daydream.pr_comment_renderer as renderer
from daydream.atif import Trajectory
from daydream.pr_comment_renderer import (
    render_run_info,
    render_run_info_block,
)
from daydream.pricing import ModelPrice, resolve_prices

# Committed fixtures cover Claude (cost_usd present) and Codex (cost_usd null,
# synthesized via pricing). Tests needing a specific shape build inline via _write_trajectory.
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "trajectories"
_MULTI_PHASE_CODEX = _FIXTURE_DIR / "multi_phase_codex.json"
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


def test_e2e_deep_mode_aggregates_fork_files(tmp_path: Path) -> None:
    """Forks and archived introductory pricing retain exact rows in both rendering APIs."""
    archived = _write_trajectory(
        tmp_path, model="claude-sonnet-5", steps=[_user_step(), _sonnet5_transition_step()],
    )
    paths = [_DEEP_PARENT, _DEEP_FORK_A, _DEEP_FORK_B, archived]
    trajectories = [Trajectory.model_validate_json(path.read_bytes()) for path in paths]
    out = render_run_info_block(paths)
    assert render_run_info(trajectories, prices=resolve_prices()) == out
    assert "- **Model:** mixed — see breakdown" in out
    assert "- **Cost:** $12.39" in out
    assert "- **Tokens:** 2,014,000 in (1,007,300 cached, 50% hit) → 1,003,000 out" in out
    assert "- **Steps / tool calls:** 5 / 5" in out
    assert "| Review | mixed | 2 | 2,008,000 (50%) | 1,001,500 | $12.30 |" in out
    assert "| Parse Feedback | claude-sonnet-4-5 | 0 | 1,000 (80%) | 200 | $0.02 |" in out
    assert "| Fix | claude-sonnet-4-5 | 3 | 5,000 (50%) | 1,300 | $0.07 |" in out



def test_duration_degrades_gracefully(tmp_path: Path) -> None:
    p = _write_trajectory(tmp_path,
        steps=[{"step_id": 1, "timestamp": None, "source": "user", "message": "go",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
            }, {"step_id": 2, "timestamp": None, "source": "agent", "message": "ok", "model_name": "gpt-5.5",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "ttt"},
                "metrics": {"prompt_tokens": 500, "completion_tokens": 10, "cached_tokens": 0, "cost_usd": 0.001},
            },
        ],
    )
    out = render_run_info_block([p])
    assert "- **Duration:** —" in out
    assert "- **Cost:** <$0.01" in out
    assert "- **Tokens:** 500 in → 10 out" in out
    assert "0% hit" not in out
    assert "cached" not in out.split("**Tokens:**", 1)[1].split("\n", 1)[0]
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    assert review_row.rstrip().endswith("| — |")
