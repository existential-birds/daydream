"""Operator capture choices through the production CLI with an offline backend."""

import hashlib
import json
from pathlib import Path

import pytest

from daydream import cli
from daydream.dataset import LocalRecordStore
from tests.harness.stub_backend import install_stub_backend, silence


def test_cli_capture_is_independent_of_archiving(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    silence(monkeypatch, prompts=False)
    install_stub_backend(monkeypatch, feature_branch_repo, pin_skill_availability=False)
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda _repo, **_kwargs: None)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda _repo, _branch, **_kwargs: None)
    store_path = tmp_path / "records"
    arguments = [
        "--review", "--shallow", "--stack", "python", "--non-interactive",
        "--no-archive", "--no-eval", "--no-tracing", "--dataset-store", str(store_path),
        str(feature_branch_repo),
    ]
    with pytest.raises(SystemExit) as enabled:
        cli.main(["--capture-data", *arguments])
    assert enabled.value.code == 0
    store = LocalRecordStore(store_path)
    captured = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00Z"))
    assert len(captured.runs) == 1
    assert captured.runs[0].outcome == "success"
    assert captured.runs[0].trace_id is None


def test_cli_capture_redacts_review_text_and_preserves_exact_scoring_length(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    silence(monkeypatch, prompts=False)
    backend = install_stub_backend(monkeypatch, feature_branch_repo, pin_skill_availability=False)
    sensitive_text = "Review credential exposure: sk-offlineplaceholder0123456789"
    backend.parse_by_stack = {"python": {
        "severity": "high", "confidence": "HIGH", "file": "main.py", "line": 1,
        "description": sensitive_text,
    }}
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda _repo, **_kwargs: None)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda _repo, _branch, **_kwargs: None)
    store_path = tmp_path / "records"
    with pytest.raises(SystemExit) as finished:
        cli.main([
            "--review", "--shallow", "--stack", "python", "--non-interactive", "--capture-data",
            "--no-archive", "--no-eval", "--no-tracing", "--dataset-store", str(store_path),
            str(feature_branch_repo),
        ])
    assert finished.value.code == 0
    original_review = (feature_branch_repo / ".review-output.md").read_text()
    assert sensitive_text in original_review
    store = LocalRecordStore(store_path)
    records = store.read_snapshot(store.select_snapshot(observed_before="2100-01-01T00:00:00Z"))
    assert len(records.runs) == 1
    scoring = records.runs[0].scoring.value
    assert "sk-offlineplaceholder" not in scoring["review_text"]
    assert "[REDACTED_API_KEY]" in scoring["review_text"]
    assert scoring["length"] == len(original_review)
    assert scoring["length"] != len(scoring["review_text"])
    assert scoring["source_review_sha256"] == hashlib.sha256(original_review.encode()).hexdigest()
    assert scoring["review_text_redaction"]["applied"]
    assert scoring["persisted_breakdown"]["composite"] is None


@pytest.mark.parametrize("refusal", ["symlink", "file", "runtime-owned"])
def test_cli_dataset_refusal_preserves_completed_review_outputs(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], refusal: str,
) -> None:
    silence(monkeypatch, prompts=False)
    install_stub_backend(monkeypatch, feature_branch_repo, pin_skill_availability=False)
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda _repo, **_kwargs: None)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda _repo, _branch, **_kwargs: None)
    store_path = tmp_path / "records"
    if refusal == "runtime-owned":
        store_path = feature_branch_repo / ".daydream" / "dataset"
    elif refusal == "symlink":
        unowned = tmp_path / "unowned"
        unowned.mkdir()
        (unowned / "prior.txt").write_text("preserve me")
        store_path.symlink_to(unowned, target_is_directory=True)
    else:
        store_path.write_text("preserve me")
    with pytest.raises(SystemExit) as finished:
        cli.main([
            "--review", "--shallow", "--stack", "python", "--non-interactive", "--capture-data",
            "--no-archive", "--no-eval", "--no-tracing", "--dataset-store", str(store_path),
            str(feature_branch_repo),
        ])
    assert finished.value.code == 0
    assert "Sample issue" in (feature_branch_repo / ".review-output.md").read_text()
    trajectories = list((feature_branch_repo / ".daydream" / "runs").glob("*/trajectory.json"))
    assert len(trajectories) == 1
    trajectory = json.loads(trajectories[0].read_text())
    assert trajectory["session_id"] == trajectories[0].parent.name
    assert trajectory["steps"]
    assert not trajectory["extra"].get("partial", False)
    assert "Data Collection" in capsys.readouterr().out
    if refusal == "symlink":
        assert [path.name for path in unowned.iterdir()] == ["prior.txt"]
        assert (unowned / "prior.txt").read_text() == "preserve me"
    elif refusal == "file":
        assert store_path.read_text() == "preserve me"
    else:
        assert not store_path.exists()
