"""Tests for daydream.eval.analyzer trajectory loading.

Focused on session-id resolution semantics inside ``load_trajectories``:
ambiguous prefixes must raise instead of silently picking one, exact
matches must take precedence over prefix matches, and unique prefixes
must still resolve.
"""

import json
from pathlib import Path

import pytest

from daydream.eval.analyzer import load_trajectories


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
