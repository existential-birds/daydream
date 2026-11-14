"""Archive capture through real runner/worktree paths with only backend calls mocked.
Verify default/disabled evaluation and separate recommended versus reviewed patches in
deep and shallow flows.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
from collections.abc import AsyncIterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.atif import validate as atif_validate
from daydream.backends import (
    AgentEvent,
    ResultEvent,
    TextEvent,
)
from daydream.config_file import DaydreamFileConfig
from daydream.dataset import LocalRecordStore
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.training.labeler_signals import local_commit_applied_signal
from tests.deep_orchestrator.support import _only_archived_run
from tests.harness.dataset import read_records
from tests.harness.git_helpers import bare_remote, git
from tests.harness.remote_ci import NoCIRemote
from tests.harness.stub_backend import (
    StubBackend,
    force_interactive,
    install_stub_backend,
    silence,
)
from tests.harness.trajectory import diff_adding
from tests.test_deep_orchestrator import _merge_item, _noop_commit, _ok


def _deep_run_config(target: Path, **overrides: Any) -> RunConfig:
    config: dict[str, Any] = {
        "target": str(target), "test_command": f"{sys.executable} -c pass",
        "assume": "yes", "output_mode": "loop", "cleanup": False,
    }
    config.update(overrides)
    return RunConfig(**config)


class _ArchiveCaptureBackend(StubBackend):
    async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if prompt.startswith("The daydream changes are already staged"):
            run_id = prompt.split("Daydream-Run: ", 1)[1].splitlines()[0]
            version = prompt.split("Daydream-Version: ", 1)[1].splitlines()[0]
            # The index is pre-staged by _do_commit (deterministic staging,
            # issue #543) — commit it as-is, never `git add --all`.
            git(cwd, "commit", "-m",
                (f"fix: apply daydream recommendation\n\nDaydream-Run: {run_id}\nDaydream-Version: {version}"),
            )
            git(cwd, "push", "-u", "archive", git(cwd, "branch", "--show-current"))
            yield TextEvent(text="Committed and pushed the recommendation.")
            yield ResultEvent(structured_output=None, continuation=None)
            return
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


def _install_deep_capture_backend(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, *, real_internal_phases: bool = False,
) -> Any:
    """Install the shared deep-run backend and optional focused phase seams."""
    silence(monkeypatch)
    force_interactive(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    if real_internal_phases:
        stub = _ArchiveCaptureBackend(multi_stack_target)
        # One instance per resolved phase backend, as the real factory builds them; the copies share the script.
        monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: copy.copy(stub))
    stub.merge_items = [_merge_item(1, "api.py", "high")]
    if not real_internal_phases:
        monkeypatch.setattr("daydream.deep.fix_steps.phase_test_and_heal", lambda *a, **k: _ok(**k),)
        monkeypatch.setattr("daydream.deep.fix_steps.phase_commit_push", _noop_commit)
    return stub


async def _run_real_phases_deep(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, no_ci_remote: NoCIRemote, *,
    remote_name: str = "origin.git", pr_repo: str | None = None,
    fix_edit_line: str = "# daydream recommended change\n",
    untracked_fix: str | None = None, dataset_store: Path | None = None, dump_path: Path | None = None,
) -> tuple[Path, int]:
    """Run real internal phases against a bare remote; return ``(remote, exit_code)``.

    ``untracked_fix`` also seeds pre-existing notes.txt to test untracked-file exclusions.
    """
    remote = bare_remote(archive_dir.parent / remote_name)
    no_ci_remote.connect(multi_stack_target, remote)
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch, real_internal_phases=True)
    stub.fix_edit_line = fix_edit_line
    stub.review_claims = [{"title": "Sample issue", "file": "api.py", "line": 1}]
    if dataset_store is not None:
        assert stub.merge_items is not None
        stub.merge_items[0]["source_uids"] = ["python:1"]
    if untracked_fix is not None:
        stub.fix_new_generated = untracked_fix
        assert stub.merge_items is not None
        stub.merge_items[0]["related_files"] = [untracked_fix]
        (multi_stack_target / "notes.txt").write_text("pre-existing\n")
    exit_code = await run(_deep_run_config(
            multi_stack_target, backend="pi",
            pr_number=no_ci_remote.pr_number, pr_repo=pr_repo or no_ci_remote.base_repository,
            dataset_capture=dataset_store is not None, dataset_store_path=dataset_store,
            dump_artifacts=None if dump_path is None else str(dump_path),
            file_config=DaydreamFileConfig(verify_all=True),
        )
    )
    # Review context stays on the reviewer's own instances: no turn that may edit the checkout runs on one.
    assert {call["review_bound"] for call in stub.calls if call["review_instructions"]} == {True}
    assert {call["review_bound"] for call in stub.calls if not call["read_only"]} == {False}
    return remote, exit_code


async def test_default_deep_run_populates_eval_captures_patch_and_current_merge_phase_state(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, no_ci_remote: NoCIRemote,
    tmp_path: Path,
) -> None:
    """Archive and push tracked/generated fixes while leaving pre-existing notes untracked."""
    head_before = git_ops.head_sha(multi_stack_target)
    base_before = git_ops.resolve_diff_merge_base(multi_stack_target, "main", head_before)
    original_diff = git_ops.diff(multi_stack_target, "main")
    store = LocalRecordStore(archive_dir.parent / "records")
    dump_dir = tmp_path / "uploaded-artifacts"
    remote, exit_code = await _run_real_phases_deep(
        multi_stack_target, monkeypatch, archive_dir, no_ci_remote, dataset_store=store.root, dump_path=dump_dir)
    assert exit_code == 0
    head_after = git_ops.head_sha(multi_stack_target)
    assert head_after != head_before
    assert git(remote, "rev-parse", "refs/heads/feature") == head_after
    commit_message = git(multi_stack_target, "log", "-1", "--format=%B")
    assert "Daydream-Run:" in commit_message
    assert "Daydream-Version:" in commit_message

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    siblings = [json.loads(path.read_bytes()) for path in (run_dir / "trajectories").glob("*.json")]
    assert len(siblings) >= 1
    for document in (trajectory, *siblings):
        assert document["session_id"] == manifest["session_id"]
        assert atif_validate(document, validate_images=False)
        assert [step["step_id"] for step in document["steps"]] == list(range(1, len(document["steps"]) + 1))
        assert all(
            step["extra"]["daydream_run_flow"] == "deep" for step in document["steps"] if step["source"] == "agent"
        )
    assert not any("Run the project's test suite" in step.get("message", "") for step in trajectory["steps"])
    verdict = json.loads((multi_stack_target / ".daydream/deep/test-verdict.json").read_text())
    assert verdict["passed"] and all(attempt["kind"] == "host" for attempt in verdict["attempts"])

    metrics = manifest["metrics"]
    assert "grounding_rate" not in metrics
    assert metrics["total_findings"] is not None
    assert "coverage_ratio" not in metrics
    assert metrics["cost_per_finding_usd"] is not None
    assert metrics["erosion"] == metrics["verbosity"] == 0.0
    with closing(sqlite3.connect(archive_dir / "index.db")) as connection:
        row = connection.execute("SELECT erosion, verbosity, total_findings, cost_per_finding_usd FROM runs").fetchone()
    assert row == (0.0, 0.0, metrics["total_findings"], metrics["cost_per_finding_usd"])
    assert (run_dir / "evaluation.json").is_file()
    assert manifest["phase_states"]["merge"] == {"ran": True, "status": "succeeded"}
    assert manifest["pipeline_status"] == "succeeded"
    merge_events = [event for event in trajectory["extra"]["phase_events"] if event["phase"] == "merge"]
    assert len(merge_events) == 2
    assert {event["session_id"] for event in merge_events} == {manifest["session_id"]}

    recommended = run_dir / "recommended.patch"
    diff = run_dir / "diff.patch"
    assert recommended.is_file()
    assert diff.is_file()
    recommended_text = recommended.read_text()
    diff_text = diff.read_text()
    assert recommended_text != diff_text
    assert "# daydream recommended change" in recommended_text
    assert "# daydream recommended change" not in diff_text
    for filename in ("manifest.json", "trajectory.json", "diff.patch", "evaluation.json"):
        assert (dump_dir / filename).is_file()
        assert (dump_dir / filename).read_bytes() == (run_dir / filename).read_bytes()
    captured = read_records(store).runs[0]
    task = captured["original_task"]["value"]
    assert (task["analyzed_revision"]["head_sha"], task["analyzed_revision"]["merge_base_sha"]) == (
        head_before, base_before)
    assert task["diff"] == original_diff and captured["final_state"]["value"]["head_sha"] == head_after
    assert captured["recommended_patch"]["value"]["patch"] == recommended_text
    item = captured["findings"]["value"]["items"][0]
    assert item["item_uid"] and item["source_uids"] == [item["uid"]]
    assert item["description"] == "Sample issue" and "severity" not in item
    verification = captured["verification"]["value"]
    verdicts = verification["recommendation-verdicts.json"]
    assert verdicts["selection"]["decisions"][0]["item_uid"] == item["item_uid"]
    assert verdicts["selection"]["mode"] == "verify_all"
    assert verdicts["verdicts"][0]["issue_id"] == item["id"]
    assert verification["fix-outcomes.json"]["outcomes"][item["item_uid"]]["verdict"] == "resolved"
    assert captured["scoring"]["value"]["persisted_breakdown"]["correctness_per_finding"] == [1.0]


def test_local_commit_applied_signal_uses_recommended_patch(tmp_path: Path,
) -> None:
    file_contents = "existing\nreviewed = 2\n"
    expected_verdict = "rejected"
    (tmp_path / "diff.patch").write_text(diff_adding("reviewed = 2"))
    (tmp_path / "recommended.patch").write_text(diff_adding("recommended = 1"))
    row = {"repo_slug": "org/repo", "head_sha": "abc", "branch": "feature",
           "recommended_patch": (tmp_path / "recommended.patch").read_text()}
    sig = local_commit_applied_signal(
        row, repo_clone=tmp_path, commits_since_fetcher=lambda repo, branch, since: ["c1"],
        file_at_fetcher=lambda repo, path, sha: file_contents,
    )
    assert sig.verdict == expected_verdict
