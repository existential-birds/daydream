"""Tests for the ``daydream label`` human-override subcommand.

Drives ``cli._handle_label_command`` directly (the dispatch in ``main`` is a
thin one-liner). Assertions pin observable state — the denormalized ``runs``
cache value, the human-sourced observation in history, and the prior label
echoed to stdout — not mere dispatch.
"""
from pathlib import Path
from typing import Any

import pytest

from daydream import cli
from daydream.archive.index import (
    append_label_observation,
    label_observation_history,
    query_runs,
    upsert_run,
)
from tests.harness.trajectory import make_manifest


def test_label_command_sets_human_label_and_shows_prior(
    tmp_path: Path,
    archive_dir: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upsert_run(archive_dir, make_manifest(session_id="sess-0001"))
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


def test_label_command_accepts_unknown(tmp_path: Path, archive_dir: Any) -> None:
    upsert_run(archive_dir, make_manifest(session_id="sess-0002"))
    assert cli._handle_label_command(["sess-0002", "--outcome", "unknown"]) == 0


def test_label_command_unknown_session_returns_1(tmp_path: Path, archive_dir: Any) -> None:
    assert cli._handle_label_command(["no-such", "--outcome", "accepted"]) == 1
