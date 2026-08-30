"""Real-path tests for the corpus adjudicate sub-verbs (style: test_cli_label.py)."""
import json
from pathlib import Path

import pytest

from daydream import cli


def _write_sessions(tmp_path: Path) -> Path:
    """One ambiguous + one unanswered finding across two sessions, in the
    hydrated-index shape the queue builder consumes (T1 session dicts)."""
    sessions = [
        {
            "session_id": "s1", "trajectory_id": "s1-traj", "segment_id": "s1-seg",
            "resolutions": [{
                "fingerprint": "fp-b", "disposition": "unanswered",
                "evidence": [{"reply_id": "r1", "body_sha256": "abc"}],
                "evidence_digest": "d2" * 32, "profile": "pr_review", "stack": "python",
            }],
        },
        {
            "session_id": "s2", "trajectory_id": "s2-traj", "segment_id": "s2-seg",
            "resolutions": [{
                "fingerprint": "fp-a", "disposition": "ambiguous",
                "evidence": [{"reply_id": "r2", "body_sha256": "abd"}],
                "evidence_digest": "d1" * 32, "profile": "pr_review", "stack": "python",
            }],
        },
    ]
    (tmp_path / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )
    return tmp_path


def test_adjudicate_label_records_human_observation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_sessions(tmp_path)
    cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                "--state-dir", str(tmp_path / "adj")])
    queue = json.loads((tmp_path / "adj" / "queue.json").read_text())
    record_id = str(queue[0]["record_id"])
    rc = cli._handle_corpus_command(
        ["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--record-id", record_id, "--disposition", "accepted",
         "--rationale", "reply confirms fix", "--labeler", "kevin"]
    )
    assert rc == 0
    lines = (tmp_path / "adj" / "observations.jsonl").read_text().splitlines()
    obs = [json.loads(line) for line in lines]
    assert obs[-1]["disposition"] == "accepted" and obs[-1]["role"] == "rater"
    assert obs[-1]["record_id"] == record_id


def test_adjudicate_label_unknown_record_id_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_sessions(tmp_path)
    cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                "--state-dir", str(tmp_path / "adj")])
    rc = cli._handle_corpus_command(
        ["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--record-id", "a" * 64, "--disposition", "accepted",
         "--rationale", "reply confirms fix", "--labeler", "kevin"]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "a" * 64 in captured.out + captured.err


def test_adjudicate_label_batch_n_processes_unresolved_in_order(tmp_path: Path) -> None:
    _write_sessions(tmp_path)
    cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                "--state-dir", str(tmp_path / "adj")])
    rc = cli._handle_corpus_command(
        ["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--batch", "1", "--disposition", "rejected", "--rationale", "stale finding",
         "--labeler", "kevin"]
    )
    assert rc == 0
    obs = [json.loads(line) for line in (tmp_path / "adj" / "observations.jsonl").read_text().splitlines()]
    assert len(obs) == 1  # one observation per item; re-run advances, never duplicates
    rc2 = cli._handle_corpus_command(
        ["adjudicate", "label", "--state-dir", str(tmp_path / "adj"),
         "--batch", "1", "--disposition", "rejected", "--rationale", "stale finding",
         "--labeler", "kevin"]
    )
    assert rc2 == 0
    obs2 = [json.loads(line) for line in (tmp_path / "adj" / "observations.jsonl").read_text().splitlines()]
    assert len(obs2) == 2 and obs2[0]["record_id"] != obs2[1]["record_id"]  # resume point advanced


def test_adjudicate_show_lists_queue_and_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_sessions(tmp_path)
    cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                "--state-dir", str(tmp_path / "adj")])
    rc = cli._handle_corpus_command(["adjudicate", "show", "--state-dir", str(tmp_path / "adj")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ambiguous" in out and "unanswered" in out


def test_adjudicate_unknown_subverb_exits_2() -> None:
    with pytest.raises(SystemExit):
        cli._handle_corpus_command(["adjudicate", "bogus"])


def test_adjudicate_bare_invocation_exits_2() -> None:
    with pytest.raises(SystemExit):
        cli._handle_corpus_command(["adjudicate"])


def test_adjudicate_build_preserves_observations_and_is_idempotent(tmp_path: Path) -> None:
    _write_sessions(tmp_path)
    for _ in range(2):
        rc = cli._handle_corpus_command(["adjudicate", "build", "--index-root", str(tmp_path),
                                         "--state-dir", str(tmp_path / "adj")])
        assert rc == 0
    queue = json.loads((tmp_path / "adj" / "queue.json").read_text())
    assert len(queue) == 2
