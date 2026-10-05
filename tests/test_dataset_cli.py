"""Capture, redaction and optional-store failures through the production CLI."""
import hashlib
import json
from pathlib import Path

import pytest

from daydream import cli
from daydream.dataset import LocalRecordStore, serialize_record
from tests.harness.dataset import read_records
from tests.harness.stub_backend import install_stub_backend, silence


@pytest.mark.parametrize("destination", ["private", "symlink", "file", "runtime-owned"])
def test_cli_capture_and_refusal_preserve_completed_review(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], destination: str,
) -> None:
    silence(monkeypatch, prompts=False)
    backend = install_stub_backend(monkeypatch, feature_branch_repo, pin_skill_availability=False)
    sensitive = "Review credential exposure: sk-offlineplaceholder0123456789"
    backend.parse_by_stack = {"python": {
        "severity": "high", "confidence": "HIGH", "file": "main.py", "line": 1, "description": sensitive}}
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda _repo, **_kwargs: None)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda _repo, _branch, **_kwargs: None)
    store_path = tmp_path / "records"
    unowned = tmp_path / "unowned"
    if destination == "runtime-owned":
        store_path = feature_branch_repo / ".daydream" / "dataset"
    elif destination == "symlink":
        unowned.mkdir()
        (unowned / "prior.txt").write_text("preserve me")
        store_path.symlink_to(unowned, target_is_directory=True)
    elif destination == "file":
        store_path.write_text("preserve me")
    with pytest.raises(SystemExit) as finished:
        cli.main([
            "--review", "--shallow", "--stack", "python", "--non-interactive", "--capture-data",
            "--no-archive", "--no-eval", "--no-tracing", "--dataset-store", str(store_path), str(feature_branch_repo)])
    assert finished.value.code == 0
    review = (feature_branch_repo / ".review-output.md").read_text()
    assert sensitive in review
    trajectories = list((feature_branch_repo / ".daydream" / "runs").glob("*/trajectory.json"))
    assert len(trajectories) == 1
    trajectory = json.loads(trajectories[0].read_text())
    assert trajectory["session_id"] == trajectories[0].parent.name and trajectory["steps"]
    assert not trajectory["extra"].get("partial", False)
    if destination == "private":
        records = read_records(LocalRecordStore(store_path))
        assert len(records.runs) == 1
        assert records.runs[0].outcome == "success" and records.runs[0].trace_id is None
        scoring = serialize_record(records.runs[0])["scoring"]["value"]
        assert "sk-offlineplaceholder" not in scoring["review_text"]
        assert "[REDACTED_API_KEY]" in scoring["review_text"]
        assert scoring["length"] == len(review) != len(scoring["review_text"])
        assert scoring["source_review_sha256"] == hashlib.sha256(review.encode()).hexdigest()
        assert scoring["review_text_redaction"]["applied"]
        assert scoring["persisted_breakdown"]["composite"] is None
    else:
        assert "Data Collection" in capsys.readouterr().out
        if destination == "symlink":
            assert [path.name for path in unowned.iterdir()] == ["prior.txt"]
            assert (unowned / "prior.txt").read_text() == "preserve me"
        elif destination == "file":
            assert store_path.read_text() == "preserve me"
        else:
            assert not store_path.exists()
