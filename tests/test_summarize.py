"""CLI summarization, renderer parity, and user price overrides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.summarize import summarize

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "trajectories"
_DEEP_PARENT = _FIXTURE_DIR / "deep_mode_parent.json"
_DEEP_FORK_A = _FIXTURE_DIR / "deep_mode_fork_a.json"
_DEEP_FORK_B = _FIXTURE_DIR / "deep_mode_fork_b.json"


def test_missing_empty_unparseable_or_non_json_input_returns_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert summarize(tmp_path / "no-such-thing.json") != 0
    assert "does not exist" in capsys.readouterr().err
    empty = tmp_path / "empty-run"
    empty.mkdir()
    assert summarize(empty) != 0
    assert "no trajectory" in capsys.readouterr().err.lower()
    run_dir = tmp_path / "broken-run"
    run_dir.mkdir()
    (run_dir / "trajectory.json").write_text("{not json", encoding="utf-8")
    # The renderer fallback signals "no real summary produced" via non-zero.
    assert summarize(run_dir) != 0
    assert "*run details unavailable*" in capsys.readouterr().out
    bad = tmp_path / "trajectory.txt"
    bad.write_text(json.dumps({"schema_version": "ATIF-v1.6"}), encoding="utf-8")
    assert summarize(bad) != 0
    assert ".json" in capsys.readouterr().err


# --- User-overridable pricing through the summarize entrypoint ----------------

# Deliberately absent from built-in MODEL_PRICES, so the cost cell can only
# render a dollar value via a user-supplied price override.
_CUSTOM_MODEL = "my-custom-model"

# Chosen so cost lands on a clean string: 100_000*10/1e6 + 50_000*20/1e6 = $2.00.
_PROMPT_TOKENS = 100_000
_CACHED_TOKENS = 0
_COMPLETION_TOKENS = 50_000
_INPUT_PRICE = 10.0
_OUTPUT_PRICE = 20.0
_EXPECTED_COST_CELL = "$2.00"


def _write_custom_model_trajectory(tmp_path: Path) -> Path:
    """Write an unknown model's tokens without cost. A dollar cell can appear
    only if the user price override is loaded."""
    traj = {"schema_version": "ATIF-v1.6", "session_id": "fixture-custom-model-summarize",
        "agent": {"name": "daydream", "version": "0.14.0", "model_name": _CUSTOM_MODEL},
        "steps": [{"step_id": 1, "timestamp": "2026-05-02T00:00:00.000000Z", "source": "user", "message": "review",
                "extra": {"daydream_phase": "review", "daydream_run_flow": "normal"},
            }, {"step_id": 2, "timestamp": "2026-05-02T00:00:01.000000Z", "source": "agent",
                "model_name": _CUSTOM_MODEL, "message": "Reviewed.",
                "tool_calls": [{"tool_call_id": "r1", "function_name": "Read", "arguments": {}}],
                "observation": {"results": [{"source_call_id": "r1", "content": "ok"}]},
                "metrics": {"prompt_tokens": _PROMPT_TOKENS, "completion_tokens": _COMPLETION_TOKENS,
                    "cached_tokens": _CACHED_TOKENS, "cost_usd": None,
                }, "extra": {"daydream_phase": "review", "daydream_run_flow": "normal"},
            },
        ], "final_metrics": {"total_prompt_tokens": _PROMPT_TOKENS, "total_completion_tokens": _COMPLETION_TOKENS,
            "total_cached_tokens": _CACHED_TOKENS, "total_steps": 2,
        },
    }
    traj_path = tmp_path / "trajectory.json"
    traj_path.write_text(json.dumps(traj), encoding="utf-8")
    return traj_path


def _write_prices_toml(tmp_path: Path) -> Path:
    """Write a prices.toml covering the custom model with clean values."""
    prices_path = tmp_path / "prices.toml"
    prices_path.write_text(
        f'[prices."{_CUSTOM_MODEL}"]\n'
        f"input = {_INPUT_PRICE}\n"
        f"output = {_OUTPUT_PRICE}\n",
        encoding="utf-8",
    )
    return prices_path

def test_summarize_uses_user_price_override_for_unknown_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    traj = _write_custom_model_trajectory(tmp_path)
    prices_file = _write_prices_toml(tmp_path)
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices_file))

    rc = summarize(traj)
    assert rc == 0
    out = capsys.readouterr().out

    # Rollup cost line and the per-phase Fix/Review row both carry the synthesized $2.00.
    assert f"- **Cost:** {_EXPECTED_COST_CELL}" in out
    review_row = next(line for line in out.splitlines() if line.startswith("| Review |"))
    assert _EXPECTED_COST_CELL in review_row
    assert "| — |" not in review_row
    # No footnote — the model WAS priced via the override.
    assert "not in the price table" not in out

# --- Per-phase token cells reflect whole-invocation usage --------------------
