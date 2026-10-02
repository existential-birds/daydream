"""Exercise corpus label against cache/history state and the prior-label stdout echo."""
from typing import Any

import pytest

from daydream.archive.index import (
    append_label_observation,
    label_observation_history,
    query_runs,
    upsert_run,
)
from daydream.commands import corpus as cli_corpus
from tests.harness.trajectory import make_manifest


def test_label_command_sets_human_label_and_shows_prior(archive_dir: Any, capsys: pytest.CaptureFixture[str],) -> None:
    upsert_run(archive_dir, make_manifest(session_id="sess-0001"))
    append_label_observation(
        archive_dir, "sess-0001", labels=["rejected"], pr_state="closed", labeler_version="auto-v1",
        evidence_sha="sha1", source="auto",
    )
    rc = cli_corpus._handle_label_command(["sess-0001", "--outcome", "accepted"])
    assert rc == 0
    row = query_runs(archive_dir, "session_id = ?", ("sess-0001",))[0]
    assert row["outcome_labels"] == '["accepted"]'
    hist = label_observation_history(archive_dir, "sess-0001")
    assert hist[-1]["source"] == "human"
    assert "rejected" in capsys.readouterr().out  # shows what it overrode (Should-Have)

def test_label_command_accepts_unknown(archive_dir: Any) -> None:
    upsert_run(archive_dir, make_manifest(session_id="sess-0002"))
    assert cli_corpus._handle_label_command(["sess-0002", "--outcome", "unknown"]) == 0
    row = query_runs(archive_dir, "session_id = ?", ("sess-0002",))[0]
    assert row["outcome_labels"] == '["unknown"]'
    history = label_observation_history(archive_dir, "sess-0002")
    assert history[-1]["labels"] == '["unknown"]'
    assert history[-1]["source"] == "human"

def test_label_command_unknown_session_returns_1() -> None:
    assert cli_corpus._handle_label_command(["no-such", "--outcome", "accepted"]) == 1
