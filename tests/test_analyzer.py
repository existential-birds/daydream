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

from daydream.eval.analyzer import analyze_grounding, load_trajectories


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


def test_grounding_rate_is_the_real_fraction_for_mixed_findings():
    """Regression guard: with findings present the rate stays the true ratio.

    One finding cites a file the matching stack agent actually read; the other
    cites a file it never opened. Half of two findings are grounded -> 0.5.
    """
    trajectories = {
        "main": None,
        "forked": [_read_traj("deep-python.json", "/repo/api.py")],
    }
    findings = [
        {
            "id": 1,
            "_stack": "python",
            "file": "api.py",
            "confidence": "HIGH",
            "rationale": "The handler is missing a guard.",
        },
        {
            "id": 2,
            "_stack": "python",
            "file": "utils.py",
            "confidence": "MEDIUM",
            "rationale": "The helper is missing a guard.",
        },
    ]

    result = analyze_grounding(trajectories, findings)

    assert result["total_findings"] == 2
    assert result["grounded_count"] == 1
    assert result["ungrounded_count"] == 1
    assert result["grounding_rate"] == 0.5
    assert [f["id"] for f in result["grounded"]] == [1]
    assert [f["id"] for f in result["ungrounded"]] == [2]
