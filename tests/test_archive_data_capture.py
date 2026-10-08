"""Archive capture through real runner/worktree paths with only backend calls mocked.
Verify default/disabled evaluation and separate recommended versus reviewed patches in
deep and shallow flows.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import AsyncIterator
from contextlib import closing
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from daydream import git_ops
from daydream.archive import scan
from daydream.backends import (
    AgentEvent,
    DiagnosticEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend
from daydream.config_file import DaydreamFileConfig
from daydream.dataset import LocalRecordStore
from daydream.phases import TestAndHealResult, TestAttemptEvidence
from daydream.phases.review import ReviewOutputError
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.training.labeler_signals import fix_applied_signal, local_commit_applied_signal
from tests.deep_orchestrator.support import _only_archived_run
from tests.harness.backend import ScriptedBackend
from tests.harness.codex_replay import make_mock_process
from tests.harness.dataset import read_records
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import bare_remote, git
from tests.harness.remote_ci import NoCIRemote
from tests.harness.stub_backend import (
    StubBackend,
    completed_stage_reads,
    force_interactive,
    install_stub_backend,
    review_stage_result,
    review_stage_state,
    silence,
)
from tests.harness.trajectory import diff_adding
from tests.test_archive import _manifest_write_snapshot, _strict_archive
from tests.test_deep_orchestrator import _merge_item, _noop_commit, _ok, _pin_findings_pr


def _deep_run_config(target: Path, **overrides: Any) -> RunConfig:
    config: dict[str, Any] = {"target": str(target), "assume": "yes", "output_mode": "loop", "cleanup": False,}
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

def _deep_python_trajectory(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "trajectories").glob("deep-python*.json"))
    assert len(candidates) == 1
    assert re.fullmatch(r"deep-python(?:--[0-9a-f]{64})?\.json", candidates[0].name)
    return candidates[0]

def _install_deep_capture_backend(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, *, real_internal_phases: bool = False,
) -> Any:
    """Install the shared deep-run backend and optional focused phase seams."""
    silence(monkeypatch)
    force_interactive(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    if real_internal_phases:
        stub = _ArchiveCaptureBackend(multi_stack_target)
        monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub,)
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
    if dataset_store is not None:
        assert stub.merge_items is not None
        stub.merge_items[0]["source_uids"] = ["python:1"]
    if untracked_fix is not None:
        stub.fix_new_generated = untracked_fix
        assert stub.merge_items is not None
        stub.merge_items[0]["related_files"] = [untracked_fix]
        (multi_stack_target / "notes.txt").write_text("pre-existing\n")
    exit_code = await run(_deep_run_config(
            multi_stack_target, pr_number=no_ci_remote.pr_number, pr_repo=pr_repo or no_ci_remote.base_repository,
            dataset_capture=dataset_store is not None, dataset_store_path=dataset_store,
            dump_artifacts=None if dump_path is None else str(dump_path),
        )
    )
    return remote, exit_code

async def _ok_with_heal_edit(target: Path, **kwargs: Any) -> Any:
    before = kwargs["capture_tree_key"]()
    (target / "heal_edit.py").write_text("def healed():\n    pass\n")
    after = kwargs["capture_tree_key"]()
    return TestAndHealResult(passed=True, retries=0, proceed=True, ignored=False,
        attempts=(TestAttemptEvidence(session_id=kwargs["session_id"], kind="agent", command=None,
            passed=True, input_tree_key=before, output_tree_key=after,
        ),),
    )

async def test_default_deep_run_populates_eval_captures_patch_and_current_merge_phase_state(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, no_ci_remote: NoCIRemote,
    tmp_path: Path,
) -> None:
    """Editing tracked api.py gives real test/heal and commit phases a nonempty recommended diff."""
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
    test_steps = [step for step in trajectory["steps"] if step.get("extra", {}).get("daydream_phase") == "test"]
    assert any("Run the project's test suite" in step["message"] for step in test_steps)
    assert any("2 passed, 0 failed" in step["message"] for step in test_steps)

    metrics = manifest["metrics"]
    assert "grounding_rate" not in metrics
    assert metrics["total_findings"] is not None
    assert "coverage_ratio" not in metrics
    assert metrics["cost_per_finding_usd"] is not None
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
    assert item["item_uid"] and item["source_uids"]
    verification = captured["verification"]["value"]
    verdicts = verification["recommendation-verdicts.json"]
    assert verdicts["selection"]["decisions"][0]["item_uid"] == item["item_uid"]
    assert verdicts["verdicts"][0]["issue_id"] == item["id"]
    assert verification["fix-outcomes.json"]["outcomes"][item["item_uid"]]["verdict"] == "resolved"
    assert captured["scoring"]["value"]["persisted_breakdown"]["correctness_per_finding"] == [1.0]

async def test_mixed_case_pr_identity_reaches_remote_ci_and_archives_success(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, no_ci_remote: NoCIRemote,
    fake_gh: FakeGh,
) -> None:
    """Operator/P04 casing normalizes, while the exact PR URL stays bound."""
    response_base = "Base-User/Project"
    response_head = "Fork-User/Project"
    response_url = f"https://github.com/{response_base}/pull/{no_ci_remote.pr_number}"
    original_serve_no_ci = no_ci_remote._serve_no_ci  # noqa: SLF001

    def serve_pr(*, branch: str, head_sha: str) -> None:
        fake_gh.set_response("repo-view", value=response_base)
        fake_gh.serve_pr_view({"number": no_ci_remote.pr_number, "title": "Fixture PR", "body": "", "state": "OPEN",
                "headRefName": branch, "baseRefName": "main", "headRefOid": head_sha, "url": response_url,
                "headRepository": {"nameWithOwner": response_head}, "headRepositoryOwner": {"login": "Fork-User"},
            }
        )

    def serve_no_ci(*, branch: str, head_sha: str) -> None:
        # Preserve the harness's normalized lowercase endpoint catalog, then
        # replace only the REST response identity returned at that endpoint.
        original_serve_no_ci(branch=branch, head_sha=head_sha)
        fake_gh.set_response("GET", f"repos/{no_ci_remote.base_repository}/pulls/{no_ci_remote.pr_number}",
            {"number": no_ci_remote.pr_number, "html_url": response_url, "state": "open",
                "base": {"ref": "main", "repo": {"full_name": response_base}},
                "head": {"ref": branch, "sha": head_sha, "repo": {"full_name": response_head},},
                "merge_commit_sha": None,
            },
        )

    monkeypatch.setattr(no_ci_remote, "_serve_pr", serve_pr)
    monkeypatch.setattr(no_ci_remote, "_serve_no_ci", serve_no_ci)
    _remote, exit_code = await _run_real_phases_deep(multi_stack_target, monkeypatch, archive_dir, no_ci_remote,
        remote_name="mixed-case-origin.git", pr_repo="bAsE-uSeR/pRoJeCt",
        fix_edit_line="# daydream mixed-case identity\n",
    )

    assert exit_code == 0
    lower_base = no_ci_remote.base_repository
    pull_endpoint = f"repos/{lower_base}/pulls/{no_ci_remote.pr_number}"
    assert fake_gh.calls("GET", pull_endpoint)
    verdict = json.loads((multi_stack_target / ".daydream/deep/remote-ci-verdict.json").read_text())
    push = json.loads((multi_stack_target / ".daydream/deep/push-verdict.json").read_text())
    assert verdict["status"] == "no_ci"
    polling = verdict["polling"]
    assert polling["elapsed_seconds"] >= polling["discovery_seconds"]
    assert polling["stable_polls"] >= polling["required_stable_polls"]
    assert push["pushed_repository"] == no_ci_remote.head_repository
    assert verdict["target"]["base_repository"] == lower_base
    assert verdict["target"]["head_repository"] == no_ci_remote.head_repository
    assert verdict["binding"]["base_repository"] == lower_base
    assert verdict["binding"]["head_repository"] == no_ci_remote.head_repository
    assert verdict["target"]["pr_url"] == response_url
    assert verdict["binding"]["pr_url"] == response_url

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["pr"] == {"number": no_ci_remote.pr_number, "repo": "bAsE-uSeR/pRoJeCt",}
    assert manifest["phase_states"]["push"]["status"] == "succeeded"
    assert manifest["phase_states"]["remote_ci"]["status"] == "succeeded"
    assert manifest["pipeline_status"] == "succeeded"
    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    remote_ends = [event
        for event in trajectory["extra"]["phase_events"]
        if event["phase"] == "remote-ci" and event["event"] == "phase_end"
    ]
    assert len(remote_ends) == 1
    assert remote_ends[0]["status"] == "succeeded"
    assert "reason_code" not in remote_ends[0]
    assert remote_ends[0]["metadata"]["stop_reason"] == "no_ci"

async def test_deep_archive_excludes_preexisting_untracked_files_from_patch_and_push(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, no_ci_remote: NoCIRemote,
) -> None:
    # One run, two surfaces: the archived patch and the pushed ref must both
    # carry the fix-created file and neither may pick up pre-existing notes.txt.
    remote, exit_code = await _run_real_phases_deep(
        multi_stack_target, monkeypatch, archive_dir, no_ci_remote, untracked_fix="migrations/0002_add_x.sql",
    )
    assert exit_code == 0
    recommended = (_only_archived_run(archive_dir) / "recommended.patch").read_text()
    assert "migrations/0002_add_x.sql" in recommended  # fix-created file present
    assert "notes.txt" not in recommended              # pre-existing file excluded

    # Inspect the ref actually pushed by the stub.
    branch = git(multi_stack_target, "branch", "--show-current")
    committed = git(remote, "ls-tree", "-r", "--name-only", branch).splitlines()
    assert "migrations/0002_add_x.sql" in committed
    assert "notes.txt" not in committed
    assert "notes.txt" in git(multi_stack_target, "status", "--porcelain")

async def test_deep_rejects_unauthorized_heal_file_before_archiving_recommendation(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path,
) -> None:
    remote = bare_remote(archive_dir.parent / "origin.git")
    git(multi_stack_target, "remote", "add", "origin", str(remote))
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch)  # real_internal_phases=False
    stub.fix_edit_line = "# daydream recommended change\n"
    monkeypatch.setattr(
        "daydream.deep.fix_steps.phase_test_and_heal", lambda *a, **k: _ok_with_heal_edit(multi_stack_target, **k),
    )

    exit_code = await run(_deep_run_config(multi_stack_target))
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    assert "heal_edit.py" not in (run_dir / "recommended.patch").read_text()
    assert not (multi_stack_target / "heal_edit.py").exists()
    sidecar = json.loads((multi_stack_target / ".daydream" / "deep" / "recommended-capture.json").read_text())
    assert sidecar["session_id"] == run_dir.name
    assert sidecar["capture_point"] == "post_test"
    assert sidecar["tree_key"] == sidecar["evidence_key"]["tree_key"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["recommended_patch_capture"] == "post_test"


async def test_failed_findings_export_retains_requested_diagnostics(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, tmp_path: Path, fake_gh: FakeGh,
) -> None:
    """A required provider failure after snapshot capture exports truthful failure and diagnostics."""
    class MissingSupervisorBackend(StubBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if "supervisor adjudication" in prompt.lower():
                self.calls.append({"prompt": prompt, "model": self.model})
                yield ResultEvent(structured_output=None, continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    silence(monkeypatch)
    backend = MissingSupervisorBackend(multi_stack_target)
    backend.merge_items = [_merge_item(1, "api.py", "high")]
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    _pin_findings_pr(monkeypatch, multi_stack_target)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    diagnostics = tmp_path / "diagnostics"
    trajectory = diagnostics / "trajectory.json"
    bundle = diagnostics / "bundle"
    findings = tmp_path / "findings.json"
    with pytest.raises(ReviewOutputError):
        await run(RunConfig(
            target=str(multi_stack_target), output_mode="review", non_interactive=True, cleanup=False, pr_number=7,
            findings_out=str(findings), trajectory_path=trajectory, dump_artifacts=str(bundle),
            file_config=DaydreamFileConfig(supervisor="llm"),
        ))

    result = json.loads(findings.read_text())["terminal_result"]
    assert result["pipeline_state"] == "failed"
    assert result["analysis_state"] == "incomplete"
    assert "missing_output" in result["reason_codes"]
    exported = json.loads(trajectory.read_text())
    bundled = json.loads((bundle / "trajectory.json").read_text())
    assert exported["trajectory_id"] == bundled["trajectory_id"]
    assert exported["steps"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["phase_states"]["merge"] == {"ran": True, "status": "succeeded"}
    assert (bundle / "diff.patch").is_file()
    assert (bundle / "evaluation.json").is_file()
    assert fake_gh.calls("POST") == []
    assert (bundle / "manifest.json").read_bytes() == (_only_archived_run(archive_dir) / "manifest.json").read_bytes()

def _commit_scanned_file(target: Path, name: str, body: str) -> None:
    """Put source bytes through the real egress path: commit -> diff.patch -> archive scan."""
    (target / name).write_text(body, encoding="utf-8")
    git(target, "add", name)
    git(target, "commit", "-m", f"add {name}")

async def _assert_target_is_reusable(target: Path) -> None:
    """Diagnostic publication must leave the target ready for another review."""
    exit_code = await run(_deep_run_config(target, output_mode="review"))
    assert exit_code == 0

@pytest.mark.parametrize(("filename", "content", "canary", "expected_rule"),
    [pytest.param("creds.py",
            'GITHUB_TOKEN = "ghp_canaryfake123"\n',
            "ghp_canaryfake123", "api_key", id="token-canary",
        ),
        pytest.param("deploy_key.pem",
            "-----BEGIN PRIVATE KEY-----\n"
            "MIIFAKEKEYMATERIALFORTESTSONLY\n"
            "MIIFAKEKEYMATERIALFORTESTSONLY\n"
            "-----END PRIVATE KEY-----\n",
            "MIIFAKEKEYMATERIALFORTESTSONLY", "pem_key", id="multiline-pem",
        ),
    ],
)
async def test_dump_artifacts_copies_credentials_in_diff(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, tmp_path: Path,
    capfd: pytest.CaptureFixture[str], filename: str, content: str, canary: str, expected_rule: str,
) -> None:
    """Diagnostic dumps preserve credential bytes and completed review evidence."""
    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target)
    _commit_scanned_file(multi_stack_target, filename, content)
    dump_dir = tmp_path / "uploaded-artifacts"
    exit_code = await run(_deep_run_config(
        multi_stack_target, output_mode="review", dump_artifacts=str(dump_dir),
    ))
    assert exit_code == 0
    run_dir = _only_archived_run(archive_dir)
    with closing(sqlite3.connect(f"{(archive_dir / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert (multi_stack_target / ".review-output.md").is_file()
    assert canary in (run_dir / "diff.patch").read_text()
    assert (dump_dir / "diff.patch").read_bytes() == (run_dir / "diff.patch").read_bytes()
    assert expected_rule in {finding.category for finding in scan.scan_run_dir(run_dir).findings}
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    assert manifest["session_id"] == json.loads((run_dir / "manifest.json").read_text())["session_id"]
    assert json.loads((dump_dir / "trajectory.json").read_text())["session_id"] == manifest["session_id"]

    out = "".join(capfd.readouterr())
    assert canary not in out
    await _assert_target_is_reusable(multi_stack_target)

@pytest.mark.parametrize("failure", ["evaluation", "filesystem", "index", "upload"])
async def test_collection_failure_preserves_deep_review_exports(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str],
    fake_gh: FakeGh, archive_dir: Path, failure: str,
) -> None:
    """Optional collection failure cannot roll back completed deep-review outputs."""
    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target)
    fake_gh.serve_open_pr(multi_stack_target)

    secret = "SECRET_COLLECTION_FAILURE /private/runtime"
    reached: list[str] = []
    def fail_collection(*_args: Any, **_kwargs: Any) -> Any:
        reached.append(failure)
        raise OSError(secret)

    if failure in ("evaluation", "index"):
        monkeypatch.setattr(
            "daydream.eval.analyzer.analyze_session"
            if failure == "evaluation" else "daydream.archive.finalize.upsert_run",
            fail_collection,
        )
    elif failure == "filesystem":
        unavailable = tmp_path / "SECRET_COLLECTION_FAILURE"
        unavailable.touch()
        monkeypatch.setenv("DAYDREAM_ARCHIVE_DIR", str(unavailable))
    else:
        monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", fail_collection)
    dump = tmp_path / "uploaded-artifacts"
    dump.mkdir()
    (dump / "prior.txt").write_text("operator baseline")
    trajectory = tmp_path / "trajectory.json"
    findings = tmp_path / "findings.json"
    review = multi_stack_target / ".review-output.md"
    review.write_text("operator baseline")
    exit_code = await run(_deep_run_config(
        multi_stack_target, output_mode="review", non_interactive=True, dump_artifacts=str(dump),
        pr_number=7, findings_out=str(findings), trajectory_path=trajectory,
        trajectory_hub_repo="test/new-runs" if failure == "upload" else None,
        dataset_store_path=tmp_path / "records",
    ))
    assert exit_code == 0
    assert reached == ([] if failure == "filesystem" else [failure])
    assert review.read_text() != "operator baseline"
    assert json.loads(findings.read_text())["findings"]
    document = json.loads(trajectory.read_text())
    assert document["steps"]
    public = multi_stack_target / ".daydream" / "runs" / document["session_id"]
    assert (public / "trajectory.json").read_bytes() == trajectory.read_bytes()
    if failure in ("index", "upload"):
        assert (dump / "trajectory.json").read_bytes() == trajectory.read_bytes()
        assert (dump / "findings.json").read_bytes() == findings.read_bytes()
    else:
        assert list(dump.iterdir()) == [dump / "prior.txt"]
        assert (dump / "prior.txt").read_text() == "operator baseline"
    if failure == "upload":
        archived = archive_dir / "runs" / document["session_id"]
        assert (archived / "trajectory.json").read_bytes() == trajectory.read_bytes()
    out = "".join(capfd.readouterr())
    assert "Data Collection" in out
    assert "SECRET_COLLECTION_FAILURE" not in out
    assert "/private/runtime" not in out

async def test_no_eval_leaves_manifest_eval_fields_null(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path,
) -> None:
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch)
    stub.fix_edit_line = "# daydream recommended change\n"
    exit_code = await run(_deep_run_config(multi_stack_target, run_eval=False,))
    assert exit_code == 0
    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = manifest["metrics"]
    assert "grounding_rate" not in metrics
    assert metrics["total_findings"] is None
    assert "coverage_ratio" not in metrics
    assert metrics["cost_per_finding_usd"] is None
    assert not (run_dir / "evaluation.json").exists()

def _fix_editing_backend(repo: Path) -> ScriptedBackend:
    """Use shallow dispatch with a real tracked main.py edit so recommended-patch capture is nonempty."""
    def responder(cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> list[AgentEvent]:

        pl = prompt.lower()
        # Dispatch native review prompts by their judgment-prose markers.
        if "review" in pl and ("inclusion obligation" in pl
            or "full change spans" in pl
            or "language-agnostic review practices" in pl
            or "assigned to this stack" in pl
            or "repository-wide interactions" in pl
        ):
            state = review_stage_state(prompt)
            assert state is not None
            return [*completed_stage_reads(Path(cwd), state), TextEvent(text="Review complete."),
                ResultEvent(structured_output=review_stage_result(prompt, [{
                                "id": 1, "description": "Add a guard", "file": "main.py", "line": 1,
                                "severity": "medium", "confidence": "HIGH", "rationale": "guard missing",
                                "evidence": "main.py:1",
                            }
                        ]), continuation=None,
                ),
            ]
        if "fix this issue" in pl or pl.startswith("fix these"):
            main_py = Path(cwd) / "main.py"
            main_py.write_text(main_py.read_text() + "# daydream recommended change\n")
            return [TextEvent(text="Fixed."), ResultEvent(structured_output=None, continuation=None)]
        if "post-fix fix-verifier agent" in pl:
            ids = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", prompt)]
            return [TextEvent(text=""),
                ResultEvent(structured_output={"verdicts": [
                            {"issue_id": issue_id, "verdict": "resolved", "reason": "complete",}
                            for issue_id in ids
                        ]
                    }, continuation=None,
                ),
            ]
        if "test suite" in pl or "run the project" in pl:
            return [
                TextEvent(text="All 1 tests passed. 0 failed."), ResultEvent(structured_output=None, continuation=None),
            ]
        return [TextEvent(text="OK"), ResultEvent(structured_output=None, continuation=None)]

    return ScriptedBackend(responder=responder, model="mock-model")

async def test_shallow_run_captures_recommended_patch(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, no_ci_remote: NoCIRemote,
) -> None:
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "n")
    # The shallow run commits and pushes for real.
    remote = bare_remote(archive_dir.parent / "origin.git")
    no_ci_remote.connect(feature_branch_repo, remote)
    backend = _fix_editing_backend(feature_branch_repo)
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    exit_code = await run(RunConfig(
            target=str(feature_branch_repo), stack="python", quiet=True, cleanup=False, shallow=True, assume="yes",
            pr_number=no_ci_remote.pr_number, pr_repo=no_ci_remote.base_repository,
        )
    )
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    recommended = run_dir / "recommended.patch"
    diff = run_dir / "diff.patch"
    assert recommended.is_file()
    assert diff.is_file()
    recommended_text = recommended.read_text()
    diff_text = diff.read_text()
    assert recommended_text != diff_text
    assert "+# daydream recommended change" in recommended_text
    assert "+# daydream recommended change" not in diff_text

@pytest.mark.parametrize(("has_recommendation", "post_window", "expected_verdict", "expected_hunks"), [
    (True, "existing\nrecommended = 1\n", "applied", 1),
    (False, "existing\nreviewed = 2\n", "not_applied", 0),
])
def test_fix_applied_signal_uses_captured_recommendation_only(
    tmp_path: Path, has_recommendation: bool, post_window: str, expected_verdict: str, expected_hunks: int,
) -> None:
    (tmp_path / "diff.patch").write_text(diff_adding("reviewed = 2"))
    recommended = tmp_path / "recommended.patch"
    if has_recommendation:
        recommended.write_text(diff_adding("recommended = 1"))
    row = {"repo_slug": "org/repo", "head_sha": "abc", "base_branch": "main",
           "recommended_patch": recommended.read_text() if has_recommendation else ""}
    signal = fix_applied_signal(
        row, changed_files=["app.py"], repo_clone=tmp_path, diff_fetcher=lambda repo, base, head: ["app.py"],
        commits_in_window_fetcher=lambda repo, base, head: ["c1"], file_at_fetcher=lambda repo, path, sha: post_window,
    )
    assert signal.verdict == expected_verdict
    assert signal.hunks_total == signal.hunks_applied == expected_hunks

@pytest.mark.parametrize(("file_contents", "expected_verdict"),
    [
        pytest.param("existing\nrecommended = 1\n", "applied", id="recommended-line-present"),
        pytest.param("existing\nreviewed = 2\n", "rejected", id="recommended-line-absent"),
    ],
)
def test_local_commit_applied_signal_uses_recommended_patch(tmp_path: Path, file_contents: str, expected_verdict: str,
) -> None:
    (tmp_path / "diff.patch").write_text(diff_adding("reviewed = 2"))
    (tmp_path / "recommended.patch").write_text(diff_adding("recommended = 1"))
    row = {"repo_slug": "org/repo", "head_sha": "abc", "branch": "feature",
           "recommended_patch": (tmp_path / "recommended.patch").read_text()}
    sig = local_commit_applied_signal(
        row, repo_clone=tmp_path, commits_since_fetcher=lambda repo, branch, since: ["c1"],
        file_at_fetcher=lambda repo, path, sha: file_contents,
    )
    assert sig.verdict == expected_verdict

async def test_deep_run_archives_location_and_shipped_duplication_axes(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path,
) -> None:
    """Score the near-duplicate pair's original lines 1 and 88 after demotion.

    The harness adds one distinct structural finding to the real deep run.
    """
    stub = _install_deep_capture_backend(multi_stack_target, monkeypatch)
    stub.fix_edit_line = "# daydream recommended change\n"
    anchored = _merge_item(1, "api.py", "high", desc="The loader does not validate its config path")
    mis_anchored = {
        **_merge_item(2, "api.py", "medium", desc="The loader fails to validate the config path"), "line": 88,
        "evidence": "api.py:88",
    }
    stub.merge_items = [anchored, mis_anchored]
    exit_code = await run(_deep_run_config(multi_stack_target))
    assert exit_code == 0

    run_dir = _only_archived_run(archive_dir)
    evaluation = json.loads((run_dir / "evaluation.json").read_text())

    location = evaluation["location"]
    assert location["hunk_source"] == "hunk-index.json"
    assert location["shipped_items"] == 3
    assert location["scored_items"] == 3
    assert location["tiers"] == {
        "in_hunk": 2,           # the anchored finding + the structural item
        "within_tolerance": 0,
        "beyond_tolerance": 1,  # the mis-anchored twin
        "file_absent": 0,
    }
    assert location["in_hunk_rate"] == 0.6667
    assert location["distrusted_items"] == 1
    assert location["relocated_items"] == 0   # demoted, not relocated (no snap)
    beyond = [row for row in location["items"] if row["tier"] == "beyond_tolerance"]
    assert len(beyond) == 1
    assert beyond[0]["cited_line"] == 88
    assert beyond[0]["location_distrust"] is True

    duplication = evaluation["findings"]["shipped_duplication"]
    assert duplication["shipped_items"] == 3
    assert duplication["comparable_pairs"] == 3
    assert duplication["near_duplicate_pairs"] == 1   # the escape
    assert duplication["same_file_pairs"] == 3        # every item cites api.py
    assert duplication["same_file_near_duplicate_pairs"] == 1
    assert duplication["max_similarity"] >= 0.5
    escape = duplication["pairs"][0]
    assert (escape["a_id"], escape["b_id"]) == ("1", "2")
    assert escape["same_file"] is True
    assert "grounding" not in evaluation

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "grounding_rate" not in manifest["metrics"]
    assert manifest["metrics"]["total_findings"] == 3

class _CodexEvidenceBackend(StubBackend):
    """Emit one isolated standard-event evidence stream on the Python child."""

    def __init__(self, target: Path, *, evidence: bool) -> None:
        super().__init__(target)
        self.evidence = evidence

    async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        if "you are reviewing the python stack" in prompt.lower():
            if self.evidence:
                for index in range(15):
                    call_id = f"shell-{index}"
                    yield ToolStartEvent(id=call_id, name="shell",
                        input={"command": f"printf 'shell {index}\\n'"},
                    )
                    if index == 1:
                        continue
                    yield ToolResultEvent(id=call_id, output="failed" if index == 0 else "ok", is_error=index == 0,)
                for index in range(1):
                    call_id = f"patch-{index}"
                    yield ToolStartEvent(id=call_id, name="patch", input={"patch": f"*** patch {index} ***"},)
                    yield ToolResultEvent(id=call_id, output="applied", is_error=False)
                yield ToolResultEvent(id="unmatched-result", output="orphan", is_error=True)
                yield DiagnosticEvent(
                    code="codex_transport_coverage", message="current public stream has incomplete tool coverage",
                    metadata={"occurrences": 1},
                )
                yield DiagnosticEvent(
                    code="codex_parser_coverage", message="bounded parser gap evidence", metadata={"unknown_items": 1},
                )
            else:
                yield ToolStartEvent(id="clean-read", name="read", input={"path": "api.py"},)
                yield ToolResultEvent(id="clean-read", output="file content", is_error=False,)
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event

def _install_codex_evidence_backend(target: Path, monkeypatch: pytest.MonkeyPatch, *, evidence: bool,
) -> _CodexEvidenceBackend:
    """Install the evidence backend while keeping unrelated phases tool-free."""
    _install_deep_capture_backend(target, monkeypatch)
    monkeypatch.setattr("daydream.agent.console", Console(file=StringIO(), force_terminal=False))
    backend = _CodexEvidenceBackend(target, evidence=evidence)
    backend.merge_items = [_merge_item(1, "api.py", "high")]
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend,)
    return backend

async def test_codex_evidence_integrity_archives_semantic_counts_and_review_flags(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path,
) -> None:
    # Keep telemetry within one bounded stage while retaining failures, missing
    # results, unmatched results, writes and parser/transport diagnostics.
    _install_codex_evidence_backend(multi_stack_target, monkeypatch, evidence=True,)

    assert await run(_deep_run_config(multi_stack_target)) == 0

    run_dir = _only_archived_run(archive_dir)
    child = json.loads(_deep_python_trajectory(run_dir).read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    child_calls = [call for step in child["steps"] for call in step.get("tool_calls") or []]
    assert len(child_calls) == 17
    assert child_calls[0]["arguments"]["command"] == "printf 'shell 0\\n'"
    assert sum(evaluation["tools"]["by_agent"]["deep-python"].values()) == 17
    assert evaluation["tools"]["total_calls"] == 24
    assert evaluation["tools"]["by_type"] == {"shell": 15, "patch": 1, "Read": 8}
    assert evaluation["tools"]["write_ratio"] == 0.0417

    agent_steps = [step for step in child["steps"] if step["source"] == "agent"]
    result_extras = [result.get("extra", {})
        for step in agent_steps
        for result in (step.get("observation") or {}).get("results", [])
    ]
    assert any(extra.get("is_error") is True for extra in result_extras)
    assert any(extra.get("status") == "interrupted" for extra in result_extras)
    assert any("unmatched-result" in step.get("extra", {}).get("unmatched_tool_results", []) for step in agent_steps)
    diagnostics = [diagnostic
        for step in agent_steps
        for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])
    ]
    assert [diagnostic["code"] for diagnostic in diagnostics] == ["codex_transport_coverage", "codex_parser_coverage",]
    training = next(row for row in evaluation["training_signals"]["trajectories"] if row["trajectory"] == "deep-python")
    assert training["training_quality"] == "review"
    assert training["noise_flags"][:5] == [
        "failed_tool_result", "incomplete_tool_call", "unmatched_tool_result", "incomplete_telemetry",
        "parser_coverage_gap",
    ]

async def test_codex_evidence_integrity_clean_archive_stays_clean(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path,
) -> None:
    _install_codex_evidence_backend(multi_stack_target, monkeypatch, evidence=False,)
    assert await run(_deep_run_config(multi_stack_target)) == 0
    run_dir = _only_archived_run(archive_dir)
    child = json.loads(_deep_python_trajectory(run_dir).read_text(encoding="utf-8"))
    assert all(not step.get("extra", {}).get("backend_diagnostics") for step in child["steps"])
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    training = next(row for row in evaluation["training_signals"]["trajectories"] if row["trajectory"] == "deep-python")
    assert evaluation["tools"]["total_calls"] == 9
    assert evaluation["tools"]["by_type"] == {"read": 1, "Read": 8}
    assert training["noise_flags"] == []
    assert training["training_quality"] == "clean"

async def test_malformed_codex_tool_name_survives_real_log_mode_runner_archive(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path,
) -> None:
    class MalformedToolBackend(StubBackend):
        async def execute(self, cwd: Any, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            if "you are reviewing the python stack" in prompt.lower():
                item = {"id": "malformed-mcp", "type": "mcp_tool_call",
                    "tool": {"unexpected": "name-shape"}, "arguments": {"path": "api.py"},
                }
                process = make_mock_process([json.dumps({"type": "item.started", "item": item}),
                        json.dumps({"type": "item.completed", "item": {**item, "result": {"content": []}},}),
                        json.dumps({"type": "turn.completed", "usage": {}}),
                    ]
                )
                with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=process,):
                    async for event in CodexBackend(model="fixture-model").execute(cwd, prompt):
                        if isinstance(event, (ToolStartEvent, ToolResultEvent, DiagnosticEvent)):
                            yield event
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    _install_deep_capture_backend(multi_stack_target, monkeypatch)
    backend = MalformedToolBackend(multi_stack_target)
    backend.merge_items = [_merge_item(1, "api.py", "high")]
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    assert await run(_deep_run_config(multi_stack_target, log_mode=True)) == 0

    child = json.loads(_deep_python_trajectory(_only_archived_run(archive_dir)).read_text())
    calls = [call for step in child["steps"] for call in (step.get("tool_calls") or [])]
    assert [(call["tool_call_id"], call["function_name"]) for call in calls
            if call["tool_call_id"] == "malformed-mcp"] == [("malformed-mcp", "unknown")]
    diagnostics = [diagnostic for step in child["steps"]
        for diagnostic in step.get("extra", {}).get("backend_diagnostics", [])
    ]
    assert diagnostics[0]["metadata"]["warnings"]["reasons"] == {"tool_not_string": 1}

class _JoinedArtifactEvidenceBackend(StubBackend):
    """Exercise sanctioned artifact reads at the external backend boundary."""

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.private_intent_paths: list[Path] = []
        self.private_intent_payloads: list[bytes] = []
        self.python_sanctioned_entries: list[tuple[tuple[str, Path], ...]] = []
        self.python_prompts: list[str] = []
        self.python_entry_visibility: list[tuple[bool, bool]] = []

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        lowered = prompt.lower()
        python_request = "you are reviewing the python stack" in lowered
        generic_request = "you are reviewing the generic-fallback stack" in lowered
        artifact_reference: str | None = None

        if python_request:
            header = "Sanctioned phase inputs (read only these exact files):"
            prompt_lines = prompt.splitlines()
            assert prompt_lines.count(header) == 1
            start = prompt_lines.index(header) + 1
            rendered_entries: list[tuple[str, Path]] = []
            for line in prompt_lines[start:]:
                assert line.startswith("- ")
                label, raw_path = line.removeprefix("- ").split(": ", 1)
                rendered_entries.append((label, Path(raw_path)))

            intent_entries = [path for label, path in rendered_entries if label == "intent"]
            assert len(intent_entries) == 1
            intent_path = intent_entries[0]
            assert str(intent_path).endswith("/deep/intent.md")
            assert intent_path.is_absolute()
            assert intent_path.is_file()
            assert not intent_path.is_symlink()
            assert not intent_path.resolve().is_relative_to(cwd.resolve())

            cwd_visibility = ((cwd / ".daydream").exists(), (cwd / ".review-output.md").exists(),)
            assert cwd_visibility == (False, False)
            self.python_entry_visibility.append(cwd_visibility)
            self.python_prompts.append(prompt)
            self.python_sanctioned_entries.append(tuple(rendered_entries))
            self.private_intent_paths.append(intent_path)
            self.private_intent_payloads.append(intent_path.read_bytes())
            artifact_reference = str(intent_path)
        elif generic_request:
            artifact_reference = ".daydream/deep/intent.md"

        async for event in super().execute(cwd, prompt, *args, **kwargs):
            if isinstance(event, ResultEvent) and artifact_reference is not None:
                call_id = ("joined-private-intent" if python_request else "joined-relative-intent")
                yield ToolStartEvent(id=call_id, name="Read", input={"file_path": artifact_reference},)
                yield ToolResultEvent(id=call_id, output="artifact evidence", is_error=False,)

                payload = event.structured_output
                assert isinstance(payload, dict)
                candidates = payload.get("candidates")
                assert isinstance(candidates, list)
                assert len(candidates) == 1
                finding = candidates[0]["finding"]
                assert isinstance(finding, dict)
                candidate = {**candidates[0], "finding": {**finding, "rationale": f"Evidence: {artifact_reference}"}}
                yield replace(event, structured_output={**payload, "candidates": [candidate]},)
                continue
            yield event

async def test_real_deep_archive_preserves_sanctioned_artifacts_and_findings(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, artifact_runtime_root: Path,
) -> None:
    silence(monkeypatch)
    force_interactive(monkeypatch)
    backend = _JoinedArtifactEvidenceBackend(multi_stack_target)
    backend.per_stack_emit_reads = True
    backend.parse_by_stack = {"python": {"severity": "medium", "confidence": "MEDIUM", "file": "api.py", "line": 1,
            "description": "Python private-artifact rationale control",
        },
        "react": {"severity": "medium", "confidence": "MEDIUM", "file": "App.tsx", "line": 1,
            "description": "React source-only rationale control",
        },
        "generic": {"severity": "medium", "confidence": "MEDIUM", "file": "README.md", "line": 1,
            "description": "Generic relative-artifact rationale control",
        },
    }
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend,)

    exit_code = await run(RunConfig(
            target=str(multi_stack_target), output_mode="review", assume="no", non_interactive=True, cleanup=False,
            archive=True, run_eval=True, shallow_fanout_threshold=0,
        )
    )
    assert exit_code == 0
    assert len(backend.private_intent_paths) == 1
    private_intent = backend.private_intent_paths[0]
    assert backend.private_intent_payloads == [private_intent.read_bytes()]
    assert backend.private_intent_payloads[0]
    assert backend.python_entry_visibility == [(False, False)]
    assert len(backend.python_prompts) == 1
    assert len(backend.python_sanctioned_entries) == 1

    sanctioned_entries = backend.python_sanctioned_entries[0]
    assert len({label for label, _path in sanctioned_entries}) == len(sanctioned_entries)
    assert all(path.is_absolute() and path.is_file() for _label, path in sanctioned_entries)
    sanctioned_paths = {path for _label, path in sanctioned_entries}
    assert private_intent in sanctioned_paths
    private_relative = private_intent.relative_to(artifact_runtime_root)
    assert private_relative.parts[1] == "runs"
    assert private_relative.parts[3:] == ("live", ".daydream", "deep", "intent.md",)
    assert not private_intent.resolve().is_relative_to(multi_stack_target.resolve())
    live_dir = private_intent.parents[2]
    parent_artifact_dir = private_intent.parents[1]
    assert artifact_runtime_root not in sanctioned_paths
    assert live_dir not in sanctioned_paths
    assert parent_artifact_dir not in sanctioned_paths
    assert private_intent.parent not in sanctioned_paths

    run_dir = _only_archived_run(archive_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert manifest["session_id"] == private_relative.parts[2]
    assert evaluation["daydream_dir"] == str(multi_stack_target / ".daydream")

    python_trajectory = json.loads(_deep_python_trajectory(run_dir).read_text(encoding="utf-8"))
    python_completed_ids = {result["source_call_id"]
        for step in python_trajectory["steps"]
        for result in (step.get("observation") or {}).get("results", [])
        if result.get("extra", {}).get("is_error") is False
    }
    python_reads = {call["arguments"].get("file_path") or call["arguments"].get("path")
        for step in python_trajectory["steps"]
        for call in step.get("tool_calls") or []
        if call["tool_call_id"] in python_completed_ids
        and call["function_name"].casefold() == "read"
    }
    assert "api.py" in python_reads
    assert str(private_intent) in python_reads

    generic_candidates = sorted((run_dir / "trajectories").glob("deep-generic*.json"))
    assert len(generic_candidates) == 1
    assert re.fullmatch(
        r"deep-generic(?:--[0-9a-f]{64})?\.json",
        generic_candidates[0].name,
    )
    generic_trajectory = json.loads(generic_candidates[0].read_text(encoding="utf-8"))
    generic_completed_ids = {result["source_call_id"]
        for step in generic_trajectory["steps"]
        for result in (step.get("observation") or {}).get("results", [])
        if result.get("extra", {}).get("is_error") is False
    }
    generic_reads = {call["arguments"].get("file_path") or call["arguments"].get("path")
        for step in generic_trajectory["steps"]
        for call in step.get("tool_calls") or []
        if call["tool_call_id"] in generic_completed_ids
        and call["function_name"].casefold() == "read"
    }
    assert "README.md" in generic_reads
    assert ".daydream/deep/intent.md" in generic_reads

    controlled_records: list[dict[str, Any]] = []
    expected_records = {"python": ("api.py", "Python private-artifact rationale control", str(private_intent),),
        "react": ("App.tsx", "React source-only rationale control", "stub",),
        "generic": ("README.md", "Generic relative-artifact rationale control", ".daydream/deep/intent.md",),
    }
    for stack, (file, description, rationale) in expected_records.items():
        records_payload = json.loads((run_dir / "deep" / f"stack-{stack}-records.json").read_text(encoding="utf-8"))
        records = (records_payload["issues"] if isinstance(records_payload, dict) else records_payload)
        matches = [record
            for record in records
            if record.get("file") == file
            and record.get("description") == description
        ]
        assert len(matches) == 1
        assert matches[0]["line"] == 1
        assert matches[0]["evidence"] == f"{file}:1"
        assert rationale in matches[0]["rationale"]
        controlled_records.append(matches[0])
    assert len({record["description"] for record in controlled_records}) == 3
    assert "coverage" not in evaluation
    assert "grounding" not in evaluation
    assert "grounding_rate" not in manifest["metrics"]
    assert "coverage_ratio" not in manifest["metrics"]

def _retry_stop_event(*, reason: str, attempts: int, backoff_s: float, backend_s: float, retry_recovery_spent_s: float,
    circuit_state: str,
) -> dict[str, Any]:
    return {"phase": "fix", "event": "agent_budget_stop", "timestamp": "2026-01-01T00:00:00Z",
        "metadata": {"limit_expired": "retry_ladder", "elapsed_s": backend_s + backoff_s, "backend_s": backend_s,
            "backoff_s": backoff_s, "attempts": attempts, "retry_stop_reason": reason, "circuit_state": circuit_state,
            "retry_recovery_spent_s": retry_recovery_spent_s, "partial_edit_handling": "discarded",
        },
    }

def _finalize_minimal_run(*, archive_dir: Path, tmp_path: Path, session_id: str, phase_events: list[dict[str, Any]],
) -> Path:
    """Run frozen phase events through the production manifest reducer and strict finalizer."""
    target = tmp_path / f"target-{session_id}"
    target.mkdir()
    _strict_archive(
        target=target, session_id=session_id, config=RunConfig(target=str(target), archive=True, run_eval=False),
        write_snapshot=_manifest_write_snapshot(session_id=session_id,
            final_metrics={"total_prompt_tokens": 10, "total_completion_tokens": 5, "total_cached_tokens": 0,
                "total_cost_usd": 0.01,
            }, extra={"phase_events": phase_events},
        ),
    )
    return archive_dir / "runs" / session_id

@pytest.fixture
def archive_run_with_retry_stops(archive_dir: Path, tmp_path: Path) -> Path:
    return _finalize_minimal_run(
        archive_dir=archive_dir, tmp_path=tmp_path, session_id="retry-stops-0000-0000-0000-000000000001",
        phase_events=[_retry_stop_event(
                reason="retry_recovery_allowance_exhausted", attempts=2, backoff_s=1.5, backend_s=3.0,
                retry_recovery_spent_s=5.0, circuit_state="open",
            ),
            _retry_stop_event(
                reason="circuit_open", attempts=1, backoff_s=0.25, backend_s=1.0, retry_recovery_spent_s=5.25,
                circuit_state="open",
            ),
        ],
    )

@pytest.fixture
def legacy_archive_run(archive_dir: Path, tmp_path: Path) -> Path:
    return _finalize_minimal_run(
        archive_dir=archive_dir, tmp_path=tmp_path, session_id="legacy-run-0000-0000-0000-000000000002",
        phase_events=[],
    )

def test_the_manifest_carries_a_retry_and_circuit_summary(archive_run_with_retry_stops: Path) -> None:
    manifest = json.loads((archive_run_with_retry_stops / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest["retry_summary"]
    assert summary["stops"] == {"retry_recovery_allowance_exhausted": 1, "circuit_open": 1}
    assert summary["attempts"] == 3 and summary["backoff_s"] == 1.75
    assert summary["circuit_states"] == ["open"]
    assert "deadline" not in json.dumps(summary)      # durations and counts only

def test_a_run_without_retry_events_has_no_retry_summary(legacy_archive_run: Path) -> None:
    manifest = json.loads((legacy_archive_run / "manifest.json").read_text(encoding="utf-8"))
    assert "retry_summary" not in manifest


async def test_direct_upload_refuses_credentials_with_environment_destination(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target)
    _commit_scanned_file(multi_stack_target, "credentials.py", 'token = "ghp_finalizationcanary"\n')
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", "env/repo")
    from daydream.dataset_hub import DatasetUploader
    from tests.harness.dataset_hub import FakeDatasetHub

    backend = FakeDatasetHub()
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: backend)
    assert await run(_deep_run_config(
        multi_stack_target, output_mode="review", dataset_store_path=archive_dir.parent / "raw-records",
    )) == 0
    run_dir = _only_archived_run(archive_dir)
    assert b"ghp_finalizationcanary" in (run_dir / "diff.patch").read_bytes()
    assert backend.commits == []
    status = DatasetUploader(LocalRecordStore(archive_dir.parent / "raw-records"), "env/repo", backend=backend).status()
    assert status.failed == 1
    out = "".join(capfd.readouterr())
    assert "upload failure" in out
    assert "ghp_finalizationcanary" not in out
