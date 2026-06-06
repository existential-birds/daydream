"""Tests for the ``daydream label`` human-override subcommand.

Drives ``cli._handle_label_command`` directly (the dispatch in ``main`` is a
thin one-liner). Assertions pin observable state — the denormalized ``runs``
cache value, the human-sourced observation in history, and the prior label
echoed to stdout — not mere dispatch.
"""

from typing import Any

from daydream import cli
from daydream.archive.index import (
    append_label_observation,
    label_observation_history,
    query_runs,
    upsert_run,
)
from daydream.archive.manifest import Manifest


def _make_manifest(session_id: str = "sess-0001", **overrides: Any) -> Manifest:
    defaults: dict[str, Any] = {
        "session_id": session_id,
        "archived_at": "2026-04-29T00:00:00+00:00",
        "status": "complete",
        "run_flow": "normal",
        "skill": "python",
        "model": "opus",
        "backend": "claude",
        "archive_path": "/tmp/archive/runs/sess-0001",
    }
    defaults.update(overrides)
    return Manifest(**defaults)


def test_label_command_sets_human_label_and_shows_prior(tmp_path, archive_dir, capsys):
    upsert_run(archive_dir, _make_manifest(session_id="sess-0001"))
    append_label_observation(
        archive_dir,
        "sess-0001",
        labels=["rejected"],
        pr_state="closed",
        labeler_version="auto-v1",
        evidence_sha="sha1",
        source="auto",
    )
    rc = cli._handle_label_command(["sess-0001", "--outcome", "accepted"])
    assert rc == 0
    row = query_runs(archive_dir, "session_id = ?", ("sess-0001",))[0]
    assert row["outcome_labels"] == '["accepted"]'
    hist = label_observation_history(archive_dir, "sess-0001")
    assert hist[-1]["source"] == "human"
    assert "rejected" in capsys.readouterr().out  # shows what it overrode (Should-Have)


def test_label_command_accepts_unknown(tmp_path, archive_dir):
    upsert_run(archive_dir, _make_manifest(session_id="sess-0002"))
    assert cli._handle_label_command(["sess-0002", "--outcome", "unknown"]) == 0


def test_label_command_unknown_session_returns_1(tmp_path, archive_dir):
    assert cli._handle_label_command(["no-such", "--outcome", "accepted"]) == 1
