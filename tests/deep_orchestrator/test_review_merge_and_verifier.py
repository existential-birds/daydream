"""Review Merge And Verifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import json_utils, pr_review, review_reporting, runner as _runner
from daydream.backends import ResultEvent
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.config_file import DaydreamFileConfig
from daydream.deep.artifacts import (
    DeepArtifact,
)
from daydream.deep.diff import _diff_changed_files, bound_deep_diff
from daydream.findings import load_findings_artifact
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.deep_orchestrator.precision_overlap_support import PrecisionOverlapBackend
from tests.deep_orchestrator.support import (
    _count_review_prompts,
    _install_post_recorder,
)
from tests.harness.fake_gh import FakeGh
from tests.harness.review_result import merge_result, saved_coverage
from tests.test_deep_orchestrator import (
    _force_interactive,
    _install_stub_backend,
    _pin_findings_pr,
    _prime_merge_resume,
    _record,
    _run_deep,
    _silence,
)


@pytest.mark.parametrize('presentation_fault', [
    pytest.param('context', id='unavailable-report-fresh'),
    pytest.param('corrupt-items', id='corrupt-items'),
    pytest.param('render', id='summary-render'),
])
async def test_synthesis_receives_retained_structural_records_and_owns_publication(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    presentation_fault: str, fake_gh: FakeGh,
) -> None:
    """Complete input delivery and exact published output prevent structural resurrection.

    Existing host-fold tests use near-identical titles; the observed same-severity
    paraphrase and intentionally omitted concern bypass that lexical safeguard.
    """
    language = _record(description="Default Codex configurations fail initialization after empty API keys "
                       "are serialized as redaction markers", severity="high", confidence="HIGH")
    twin = _record(description="Preserve empty API keys across configuration serialization", severity="high",
                   confidence="HIGH", evidence="api.py:1 Harbor emits **** and revalidation rejects the marker; "
                   "App.tsx:1 is the dependent caller")
    distinct = _record(id=2, description="Shared dependency violates whole-file layering", line=0,
                       evidence="The API dependency crosses the documented module boundary")
    omitted = _record(id=3, description="Separate concern omitted after synthesis")
    combined = {**language, "lens": "cross-stack", "source_uids": ["python:1", "structure:1"],
                "related_files": ["App.tsx"], "evidence": twin["evidence"]}
    whole_file = {**distinct, "lens": "structural", "source_uids": ["structure:2"]}
    expected = [combined, whole_file]
    supplied: dict[str, dict[str, Any]] = {}

    def respond(prompt: str) -> list[Any] | None:
        if "cross-stack merge agent" not in prompt.lower():
            return None
        import re
        for path in re.findall(r"(?:  - )?(\S+-records\.json)", prompt):
            for record in json.loads(Path(path).read_text())["issues"]:
                supplied[record["uid"]] = record
        return [ResultEvent(structured_output=merge_result(expected), continuation=None)]

    backend = EmptyReviewBackend(multi_stack_target, forbid_merge=False, forbid_supervise=False,
        review_by_stack={"python": [language], "structure": [twin, distinct, omitted]}, responder=respond)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    published: list[dict[str, Any]] = []
    summaries: list[str | None] = []
    post = pr_review.post_review_to_pr_from_report

    async def publish(_repo: Path, items_file: Path, **_kwargs: Any) -> Any:
        summaries.append(_kwargs.get('review_summary'))
        assert 'Stale optional publication assertion' not in _kwargs['run_info']
        if presentation_fault == 'corrupt-items':
            return await post(_repo, items_file, **_kwargs)
        published.extend(json.loads(items_file.read_text())["items"])

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", publish)
    config = empty_review_config(multi_stack_target, tmp_path / "synthesis-trajectory.json",
                                output_mode="comment", pr_number=7)
    failed: list[Path] = []
    stage, read = json_utils._stage_bytes, Path.read_text
    ready_to_post = False
    frozen_review_calls: list[int] = []
    def fail_context_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
        nonlocal ready_to_post
        if path.name == 'review-report-context.json':
            ready_to_post = True
            frozen_review_calls.append(_count_review_prompts(backend.calls))
            if presentation_fault == 'context':
                failed.append(path)
                stale = json.loads(content)
                stale['author_reported'] = {'source': 'pull_request_body', 'head_sha': None,
                                            'body': 'Stale optional publication assertion', 'truncated': False}
                path.write_text(json.dumps(stale))
                raise OSError('optional report context staging failed')
        return stage(path, content, **kwargs)
    monkeypatch.setattr(json_utils, '_stage_bytes', fail_context_stage)
    if presentation_fault == 'corrupt-items':
        def fail_items_read(path: Path, *args: Any, **kwargs: Any) -> str:
            if ready_to_post and path.name == 'merged-items.json':
                failed.append(path)
                path.write_text('{corrupt canonical items after freeze')
            return read(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'read_text', fail_items_read)
    elif presentation_fault == 'render':
        render = review_reporting.render_github_review_context
        def fail_summary_render(*args: Any, **kwargs: Any) -> str:
            if args[0] is None:
                return render(*args, **kwargs)
            failed.append(Path('publication-summary'))
            raise TypeError('optional summary render failed')
        monkeypatch.setattr(review_reporting, 'render_github_review_context', fail_summary_render)
    assert await _runner.run(config) == 0
    assert frozen_review_calls and frozen_review_calls[0] > 0
    assert _count_review_prompts(backend.calls) == frozen_review_calls[0]
    if presentation_fault == 'context':
        assert failed, 'context staging fault was not injected'
    if presentation_fault == 'render':
        assert failed and summaries == ['⚠️ **Review completeness unavailable.**']
    else:
        assert summaries == [''], 'complete review needs no coverage warning or rich context on GitHub'
    assert set(supplied) == {"python:1", "structure:1", "structure:2", "structure:3"}
    assert supplied["python:1"]["severity"] == supplied["structure:1"]["severity"] == "high"
    deep = multi_stack_target / ".daydream/deep"
    coverage = saved_coverage(deep)
    assert coverage.revision.head_sha == pr.head_sha and coverage.revision.merge_base_sha == pr.base_sha
    assert all(scope['status'] == 'complete' for scope in coverage.scopes.values())
    report = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert report == DeepArtifact.MERGED_REPORT.at(deep).read_text()
    assert all(item["description"] in report for item in expected)
    assert "## Structural Review" in report
    assert omitted["description"] not in report and twin["description"] not in report
    assert not (deep / "folded-structural.json").exists()
    if presentation_fault == 'corrupt-items':
        assert failed and published == [] and fake_gh.calls('POST', '/repos/o/r/pulls/7/reviews') == []
        assert all(scope['status'] == 'complete' for scope in saved_coverage(deep).scopes.values())
        return
    items = json.loads(read(DeepArtifact.MERGED_ITEMS.at(deep)))["items"]
    assert [item["description"] for item in items] == [item["description"] for item in expected]
    assert [item["source_uids"] for item in items] == [["python:1", "structure:1"], ["structure:2"]]
    assert items[0]["related_files"] == ["App.tsx"]
    assert items[0]["severity"] == "high"
    assert items == published


async def test_precision_overlap_keeps_independent_failure_policies_and_incomplete_publication(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An unbound arbiter verdict fails only its pass after joining a completed suppression sibling.

    Serial missing-verdict tests cannot detect sibling cancellation, a lost
    suppression result, or optimistic completion after concurrent failure.
    """
    backend = PrecisionOverlapBackend(multi_stack_target, blocked=True, fault=("arbiter", "foreign-id"))
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    config = empty_review_config(multi_stack_target, tmp_path / "failure-trajectory.json",
                                precision_mode=True, review_cache_enabled=True, latency_profile="forensic")
    results: list[int] = []

    async def review() -> None:
        results.append(await _runner.run(config))

    with anyio.fail_after(30):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(review)
            for event in backend.started.values():
                await event.wait()
            backend.release["arbiter"].set()
            await backend.closed["arbiter"].wait()
            assert not backend.closed["suppression"].is_set()
            backend.release["suppression"].set()
    assert results == [0]
    assert "suppression" in backend.completed
    assert all(event.is_set() for event in backend.closed.values())
    deep = multi_stack_target / ".daydream/deep"
    records = {row["uid"]: row for path in deep.glob("stack-*-records.json")
               for row in json.loads(path.read_text())["issues"]}
    coverage = saved_coverage(deep)
    assert set(records) == {"generic:1", "python:1", "react:2", "structure:1", "structure:2"}
    assert records["python:1"]["description"] == "Confirmed Python defect"
    assert records["python:1"]["severity"] == "high"
    assert records["react:2"]["description"].startswith("Confirmed:")
    assert coverage.phases["arbiter"]["status"] in {"failed", "incomplete"}
    assert coverage.phases["suppression"]["status"] == "complete"
    assert not DeepArtifact.ADJUDICATION_COMPLETE.at(deep).exists()
    manifests = list((deep.parent / "review-cache/entries").glob("*/manifest.json"))
    assert all(json.loads(path.read_text())["unit"] != "arbiter" for path in manifests)


async def test_cold_empty_merge_preserves_failed_stack_diagnostics(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = EmptyReviewBackend(multi_stack_target, incomplete_stack="python")
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)

    policy = DaydreamFileConfig(supervisor="llm", tool_supervisor="rules", tool_bash_deny=["blocked-review-tool"])
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    output = tmp_path / "findings.json"
    config = empty_review_config(multi_stack_target, tmp_path / "trajectory.json", file_config=policy,
                                 findings_out=str(output), pr_number=pr.number)
    assert await _runner.run(config) == 0

    assert not any("cross-stack merge agent" in call["prompt"].lower() for call in backend.calls)
    assert not any("supervisor adjudication" in call["prompt"].lower() for call in backend.calls)
    deep = multi_stack_target / ".daydream" / "deep"
    assert json.loads(DeepArtifact.MERGED_ITEMS.at(deep).read_text()) == {"items": [], "held": []}
    failures = saved_coverage(deep).unfinished_scopes
    assert "python" in failures
    assert "python" in capsys.readouterr().out
    assert "Review incomplete" in (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "python" in DeepArtifact.MERGED_REPORT.at(deep).read_text()
    loaded = load_findings_artifact(output, expected_repo="o/r", expected_pr_number=pr.number,
                                    expected_head_sha=pr.head_sha)
    assert loaded.findings == []
    assert any("python" in warning for warning in loaded.review_warnings)
    public = json.loads(output.read_text())
    assert public["schema_version"] == 2
    result = public["terminal_result"]
    assert result["analysis_state"] == "incomplete"
    assert result["pipeline_state"] == "completed"
    assert set(result["completed_stacks"]) == {"generic", "react", "structure"}
    failed_scope = next(row for row in result["stack_outcomes"] if row["scope_id"] == "python")
    assert failed_scope["reason_codes"] == ["policy_veto"]
    assert "tool_vetoed" in failures["python"] and "budget exhausted" not in failures["python"]


def test_diff_changed_files_handles_modify_add_delete_binary() -> None:
    """Non-rename diff shapes emit exactly one path each."""
    mixed = (
        "diff --git a/keep.py b/keep.py\n"
        "--- a/keep.py\n"
        "+++ b/keep.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
        "diff --git a/old.py b/old.py\n"
        "deleted file mode 100644\n"
        "--- a/old.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-x = 1\n"
        "diff --git a/logo.png b/logo.png\n"
        "index 1234..5678 100644\n"
        "Binary files a/logo.png and b/logo.png differ\n"
    )
    mixed += (
        'diff --git "a/removed\\tfile.py" "b/removed\\tfile.py"\n'
        'deleted file mode 100644\n--- "a/removed\\tfile.py"\n+++ /dev/null\n'
        '@@ -1 +0,0 @@\n-x = 1\n'
        'diff --git "a/logo\\"quoted\\".png" "b/logo\\"quoted\\".png"\n'
        'index 1234..5678 100644\nBinary files differ\n'
    )
    assert _diff_changed_files(mixed) == ["keep.py", "new.py", "old.py", "logo.png",
                                        "removed\tfile.py", 'logo"quoted".png']
    # Bounding keeps whole byte-identical blocks: an oversized leading block survives, later ones drop by name.
    big = "diff --git a/0big.py b/0big.py\n--- a/0big.py\n+++ b/0big.py\n@@ -0,0 +1 @@\n+" + "x" * 200 + "\n"
    bounded, info = bound_deep_diff(big + mixed, budget=100)
    assert info.marker is not None and bounded == info.marker + big
    assert "dropped: keep.py, new.py, old.py, logo.png" in info.marker
    assert (info.retained_blocks, info.oversize_paths) == (1, ["0big.py"])


async def test_resume_fix_skips_pr_post(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    post_calls: list[dict[str, Any]] = []
    _install_post_recorder(monkeypatch, post_calls)

    # Prime the fix-resume artifacts: the verifier and fix gate both read the
    # canonical merged-items.json, so prime it alongside the markdown report.
    deep = _prime_merge_resume(multi_stack_target)
    (multi_stack_target / REVIEW_OUTPUT_FILE).write_text(
        "# Review\n\n## Issues\n\n1. [api.py:1] primed issue\n   rationale\n"
    )
    (deep / "merged-items.json").write_text(json.dumps({"items": [{
                        "id": 1, "lens": "per-stack", "file": "api.py", "line": 1, "severity": "medium",
                        "description": "primed issue", "confidence": "MEDIUM", "rationale": "rationale",
                    }
                ]
            }
        )
    )

    exit_code = await _run_deep(multi_stack_target, start_at="fix")
    assert exit_code == 0
    assert post_calls == [], (
        f"post_review_to_pr_from_report should be skipped on --start-at fix, got {len(post_calls)} call(s)"
    )
