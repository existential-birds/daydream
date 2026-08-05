"""Tests for daydream.eval.analyzer trajectory loading and grounding.

Focused on session-id resolution semantics inside ``load_trajectories``:
ambiguous prefixes must raise instead of silently picking one, exact
matches must take precedence over prefix matches, and unique prefixes
must still resolve.

Also covers ``analyze_grounding``'s rate arithmetic, including the
undefined (zero-findings) case, which must not report a perfect score.
"""

import json
from pathlib import Path

import pytest

from daydream.backends import MetricsEvent, ResultEvent, TextEvent
from daydream.eval.analyzer import analyze_costs, analyze_grounding, load_trajectories
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder


def _write_run(daydream_dir: Path, session_id: str, marker: str) -> Path:
    """Create a minimal ``runs/<session_id>/trajectory.json`` fixture.

    ``load_trajectories`` only reads the file with ``json.loads`` and
    stuffs a ``_source_file`` key onto the returned dict, so any valid
    JSON object is enough for the resolution path we're exercising.
    """
    run_dir = daydream_dir / "runs" / session_id
    run_dir.mkdir(parents=True)
    traj = run_dir / "trajectory.json"
    traj.write_text(json.dumps({"session_id": session_id, "marker": marker}))
    return run_dir


def test_ambiguous_prefix_raises(tmp_path: Path):
    daydream_dir = tmp_path / ".daydream"
    _write_run(daydream_dir, "abcd1234-0000-0000-0000-000000000001", "first")
    _write_run(daydream_dir, "abcd1234-0000-0000-0000-000000000002", "second")

    with pytest.raises(ValueError, match="matches multiple runs"):
        load_trajectories(daydream_dir, session_id="abcd1234")


def test_unique_prefix_resolves(tmp_path: Path):
    daydream_dir = tmp_path / ".daydream"
    _write_run(daydream_dir, "abcd1234-0000-0000-0000-000000000001", "first")
    _write_run(daydream_dir, "ffff0000-0000-0000-0000-000000000002", "second")

    result = load_trajectories(daydream_dir, session_id="abcd1234")

    assert result["main"] is not None
    assert result["main"]["marker"] == "first"
    assert result["forked"] == []


def test_exact_match_takes_precedence(tmp_path: Path):
    """An exact dir name must win even if a longer dir would also prefix-match."""
    daydream_dir = tmp_path / ".daydream"
    # Exact id and a sibling whose name starts with the same string.
    _write_run(daydream_dir, "abcd1234", "exact")
    _write_run(daydream_dir, "abcd1234-extra", "prefix-only")

    result = load_trajectories(daydream_dir, session_id="abcd1234")

    assert result["main"] is not None
    assert result["main"]["marker"] == "exact"


def test_analyze_costs_preserves_fractional_aggregate_precision():
    trajectories = {
        "main": {
            "_source_file": "trajectory.json",
            "final_metrics": {"total_cost_usd": 0.00006},
        },
        "forked": [],
    }

    result = analyze_costs(trajectories)

    assert result["total_cost_usd"] == 0.00006
    assert sum(agent["cost_usd"] for agent in result["by_agent"]) == result["total_cost_usd"]


def test_analyze_costs_includes_cached_tokens_when_prompt_dominates():
    trajectories = {
        "main": {
            "_source_file": "trajectory.json",
            "final_metrics": {
                "total_prompt_tokens": 140,
                "total_cached_tokens": 14,
            },
        },
        "forked": [],
    }

    result = analyze_costs(trajectories)

    assert result["total_input_tokens"] == 140
    assert result["cache_hit_rate"] == 0.1


def test_analyze_costs_aggregates_legacy_fork_metrics():
    trajectories = {
        "main": {
            "_source_file": "trajectory.json",
            "final_metrics": {"total_cost_usd": 1.0},
        },
        "forked": [
            {
                "_source_file": "fork.json",
                "final_metrics": {"total_cost_usd": 0.5},
            }
        ],
    }

    result = analyze_costs(trajectories)

    assert result["total_cost_usd"] == 1.5


async def test_analyze_costs_assigns_nested_forks_their_own_metrics(tmp_path: Path):
    session = "nested-forks"
    daydream_dir = tmp_path / ".daydream"
    recorder = TrajectoryRecorder(
        path=daydream_dir / "runs" / session / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id=session,
    )

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="main"))
            inv.observe(MetricsEvent("main", 10, 1, 0, 0.1))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("outer") as outer:
            async with outer.invocation(phase=DaydreamPhase.DEEP) as inv:
                inv.observe(TextEvent(text="outer"))
                inv.observe(MetricsEvent("outer", 20, 2, 1, 0.2))
                inv.observe(ResultEvent(structured_output=None, continuation=None))
            async with outer.fork("inner") as inner:
                async with inner.invocation(phase=DaydreamPhase.DEEP) as inv:
                    inv.observe(TextEvent(text="inner"))
                    inv.observe(MetricsEvent("inner", 30, 3, 1, 0.3))
                    inv.observe(ResultEvent(structured_output=None, continuation=None))

    trajectories = load_trajectories(daydream_dir, session)

    result = analyze_costs(trajectories)

    by_agent = {agent["agent"]: agent for agent in result["by_agent"]}
    assert by_agent["main"]["cost_usd"] == pytest.approx(0.1)
    assert by_agent["outer"]["cost_usd"] == pytest.approx(0.2)
    assert by_agent["inner"]["cost_usd"] == pytest.approx(0.3)
    assert by_agent["main"]["steps"] == 1
    assert by_agent["outer"]["steps"] == 1
    assert by_agent["inner"]["steps"] == 1
    assert sum(agent["cost_usd"] for agent in result["by_agent"]) == pytest.approx(
        result["total_cost_usd"]
    )
    assert sum(agent["steps"] for agent in result["by_agent"]) == 3


# --- analyze_grounding ---


def _read_traj(source_file: str, *read_paths: str) -> dict:
    """Forked-trajectory fixture whose agent Read each of ``read_paths``.

    Shaped for ``_extract_tool_calls``: every step needs a ``step_id`` and its
    ``tool_calls`` need ``function_name``/``arguments``. ``_files_read`` keeps
    only the ``file_path`` of ``Read`` calls, and ``_agent_label`` derives the
    ``deep-<stack>`` key from ``_source_file``.
    """
    return {
        "_source_file": source_file,
        "steps": [
            {
                "step_id": f"s{i}",
                "tool_calls": [
                    {"function_name": "Read", "arguments": {"file_path": path}}
                ],
            }
            for i, path in enumerate(read_paths)
        ],
    }


def test_grounding_rate_is_undefined_with_zero_findings():
    """A review that produced NO findings has an undefined grounding rate.

    Reporting 1.0 here would hand a review that found nothing a perfect
    grounding score, which flows into the manifest and becomes a top RL
    reward. ``None`` is the only honest value for 0/0.
    """
    trajectories = {"main": None, "forked": [_read_traj("deep-python.json", "/repo/api.py")]}

    result = analyze_grounding(trajectories, [])

    assert result["total_findings"] == 0
    assert result["grounded_count"] == 0
    assert result["ungrounded_count"] == 0
    assert result["grounding_rate"] is None
