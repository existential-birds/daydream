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
from daydream.eval.analyzer import (
    _files_read,
    analyze_costs,
    analyze_coverage,
    analyze_grounding,
    load_trajectories,
)
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


# --- cross-backend read extraction (issue #307) ---


CODEX_READ_COMMAND = (
    "sed -n '1,240p' .daydream/exploration/summary.md && "
    "sed -n '1,280p' .daydream/diff.patch; "
    "rg -n -C 3 'cache_write_tokens|total_cache_write_tokens' "
    "core/osprey-cli docs README.md 2>/dev/null && "
    "cat some/file.rb; nl -ba pkg/services/cleanup.go"
)


def test_files_read_extracts_codex_shell_paths():
    calls = [{"function_name": "shell", "arguments": {"command": CODEX_READ_COMMAND}}]

    paths = _files_read(calls)

    assert ".daydream/exploration/summary.md" in paths
    assert ".daydream/diff.patch" in paths
    assert "core/osprey-cli" in paths
    assert "docs" in paths
    assert "README.md" in paths
    assert "some/file.rb" in paths
    assert "pkg/services/cleanup.go" in paths
    assert "1,240p" not in paths
    assert "1,280p" not in paths
    assert "cache_write_tokens|total_cache_write_tokens" not in paths
    assert "2>/dev/null" not in paths


def test_files_read_extracts_pi_read_and_bash_paths():
    calls = [
        {"function_name": "read", "arguments": {"path": "/repo/pkg/services/cleanup.go"}},
        {"function_name": "bash", "arguments": {"command": "cat README.md && nl -ba core/osprey-cli"}},
    ]

    paths = _files_read(calls)

    assert "/repo/pkg/services/cleanup.go" in paths
    assert "README.md" in paths
    assert "core/osprey-cli" in paths


def _shell_reads(command: str) -> set[str]:
    """Extract read paths from a single codex ``shell`` call."""
    return _files_read([{"function_name": "shell", "arguments": {"command": command}}])


def test_files_read_skips_separated_redirect_target():
    paths = _shell_reads("cat source.txt > target.py")

    assert "source.txt" in paths
    assert "target.py" not in paths


def test_files_read_skips_redirect_and_pattern_for_rg():
    paths = _shell_reads("rg -n 'pat' a.py b.py 2>/dev/null")

    assert "a.py" in paths
    assert "b.py" in paths
    assert "/dev/null" not in paths
    assert "pat" not in paths


def test_files_read_preserves_quoted_paths_with_spaces():
    assert "my file.py" in _shell_reads("cat 'my file.py'")
    assert "my file.py" in _shell_reads('cat "my file.py"')


def test_files_read_skips_rg_option_values():
    paths = _shell_reads("rg -C 3 --glob '*.py' 'needle' src/app.py")

    assert paths == {"src/app.py"}
    assert "3" not in paths
    assert "*.py" not in paths
    assert "needle" not in paths


def test_files_read_claude_read_and_grep_unchanged():
    calls = [
        {"function_name": "Read", "arguments": {"file_path": "/repo/api.py"}},
        {"function_name": "Grep", "arguments": {"path": "src/"}},
    ]

    paths = _files_read(calls)

    assert paths == {"/repo/api.py", "src/"}


def test_analyze_coverage_counts_codex_and_pi_reads(tmp_path: Path):
    daydream_dir = tmp_path / ".daydream"
    daydream_dir.mkdir()
    (daydream_dir / "diff.patch").write_text(
        "diff --git a/pkg/services/cleanup.go b/pkg/services/cleanup.go\n"
        "diff --git a/core/osprey-cli b/core/osprey-cli\n"
    )
    trajectories = {
        "main": {
            "_source_file": "trajectory.json",
            "steps": [
                {
                    "step_id": "m0",
                    "extra": {"daydream_phase": "deep"},
                    "tool_calls": [
                        {"function_name": "shell", "arguments": {"command": "sed -n '1,240p' .daydream/diff.patch"}}
                    ],
                }
            ],
        },
        "forked": [
            {
                "_source_file": "deep-python.json",
                "steps": [
                    {
                        "step_id": "s0",
                        "tool_calls": [
                            {"function_name": "read", "arguments": {"path": "/repo/pkg/services/cleanup.go"}},
                            {"function_name": "bash", "arguments": {"command": "cat core/osprey-cli"}},
                        ],
                    }
                ],
            }
        ],
    }

    result = analyze_coverage(trajectories, daydream_dir)

    assert result["coverage_ratio"] == 1.0
    assert result["files_read_by_reviewers"] == 2
    assert result["uncovered_files"] == []


def test_analyze_grounding_counts_codex_and_pi_reads():
    trajectories = {
        "main": None,
        "forked": [
            {
                "_source_file": "deep-python.json",
                "steps": [
                    {
                        "step_id": "s0",
                        "tool_calls": [
                            {"function_name": "shell", "arguments": {"command": "cat /repo/api.py"}},
                            {"function_name": "read", "arguments": {"path": "/repo/api.py"}},
                        ],
                    }
                ],
            }
        ],
    }
    findings = [
        {
            "id": "py-1",
            "_stack": "python",
            "file": "api.py",
            "rationale": "Read api.py; flag the missing validation.",
            "confidence": "HIGH",
        }
    ]

    result = analyze_grounding(trajectories, findings)

    entry = result["grounded"][0]
    assert entry["file_was_read"] is True
    assert result["grounded_count"] == 1
    assert result["ungrounded_count"] == 0
    assert result["grounding_rate"] == 1.0
